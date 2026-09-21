 #!/usr/bin/env python3
"""One-time merge of level-1 (full_set=1) results into the level-0 results tree.

Situation
---------
Two results roots mirror the same ORCA inputs tree, so a job's relative path
under the inputs root is the join key. Job folders sit at variable depth, so
the full job list is the universe and depth is never inferred from the tree.

Roles: --dst_root (alias --l0_root) is the tree apply writes into;
--src_root (alias --l1_root) is read-only. The L0/L1 names in states and
columns describe what a folder CONTAINS (validation level), not which root it
is in. For OMol4M (Sep 2026) the 2026-09-16 plan over lustre/omol showed:
vast (dst) validates at level 0 in ~460k jobs, lustre (src) validates at
level 1 in ~181k and holds ~520k crashed level-1 attempts, most under stale
.processing.lock files. Final tree is VAST: dst = vast, src = lustre.
Dominant actions: OVERLAY (lustre L1 keys onto vast L0), REPLACE (lustre L1
where vast has nothing), SALVAGE (crashed lustre folders' root files).

What level 1 adds on top of level 0 (see qtaim_gen/source/data/multiwfn.py):
    charge.json      vdd, mbis, chelpg
    bond.json        ibsi_bond
    fuzzy_full.json  elf_fuzzy, mbis_fuzzy_density, mbis_fuzzy_spin (open-shell)
    timings.json     the same routine names
    out_files.zip    <routine>.out for the same routines
qtaim.json, other.json and orca.json are identical between the two levels.

Actions (decided per job from validation_checks on both sides)
    OVERLAY      L0 valid on the L0 side, L1 valid on the L1 side: add the
                 L1-only keys/zip members to the L0 files. Existing keys are
                 never overwritten, so the provenance-corrected qtaim.json and
                 all L0 numbers stay exactly as they are.
    REPLACE      L0 side missing or invalid, L1 side valid at level 1: copy the
                 L1 generator/ folder over.
    REPLACE_L0   L0 side missing or invalid, L1 side valid only at level 0.
    PATCH_QTAIM  dst fails only the qtaim provenance / bond-CP checks while
                 src's qtaim passes them (and the charge canary agrees): copy
                 src qtaim.json and its qtaim.out/CPprop.txt zip members into
                 dst, keeping every other dst file.
    REPLACE_LOOSE  dst missing/invalid and src is a level-1 record that fails
                 only qtaim provenance: copy it, verify with the loose flags.
                 Opt-in via --actions; the result still needs the qtaim regen.
    OVERLAY also fires when src is level 1 but fails only qtaim provenance: the
    L1-only keys do not depend on qtaim, and dst keeps its own qtaim.json.
    ALREADY_L1   L0 side already validates at level 1: nothing to do.
    KEEP_L0      L0 side valid, L1 side not valid at level 1: nothing to do.
    CONFLICT_*   both valid but the two sides disagree (atom count, spin, or
                 the hirshfeld charge canary): never merged automatically.
    Charge-sum rule: a side whose hirshfeld charges sum to the formal charge
    plus more than --charge_sum_tol (default 0.5 e) is INVALID with flag
    charge_sum_bad. This is the wrong-electron-count wavefunction defect; it
    passes every structural check, so validation_checks alone cannot see it.
    SKIP_L1_RUNNING  L1 folder holds a .processing.lock younger than
                 --lock_max_age_hours (default 6 h, above the 5 h walltime) or
                 was written within --quiet_minutes. Older locks are stale
                 (flag stale_lock) and do not block. Re-plan only those rows
                 later with --rows_from PLAN --rows_actions SKIP_L1_RUNNING.
    SALVAGE      neither side valid but the src folder still holds root files
                 worth keeping: carry them over (add-only, no validation) so
                 the rerun starts from dst. Still a rerun candidate.
    NEEDS_RERUN  neither side valid and nothing to salvage.
    NOTHING      no L1 folder and L0 valid.

Root-file policy (every write action, add-only, dst copy always kept):
    carried : *.json, *.out (not orca.out), *.tar.zst, *.gbw.zstd0, *.tgz,
              *.inp, *.xyz, gbw_analysis.log (as gbw_analysis.l1.log)
    left    : .wfn/.wfx/.gbw wavefunctions, density_mat.npz, molden, .mfwn,
              .txt intermediates, .processing.lock
    zip     : OVERLAY adds every out_files.zip member dst lacks, except
              qtaim.out/CPprop.txt (dst keeps its own qtaim provenance) and
              any .out that never reached Multiwfn's final menu (walltime
              kill; listed under zip_skipped_incomplete in the detail).
    Files present on both sides with different sizes are listed under
    root_overlap_kept_dst in the apply detail / marker.

Depth rectification: with --scan_l1_root the L1 tree is walked (pruned at job
folders) and every discovered job folder is matched to the universe by exact
relative path, else by unique suffix match on its last two path components.
Remapped, ambiguous and unmatched folders are flagged in the plan.

Safety
    * plan never writes anything under either root.
    * apply only acts on rows of a plan file, re-checks a size/mtime
      fingerprint of both sides first, backs up what it changes to
      <job>/.pre_l1_merge/, writes JSON atomically, verifies the merged folder
      with validation_checks at the target level and rolls back on failure.
    * the L1 root is never modified.

Usage
-----
    python scripts/merge_l1_into_l0.py plan \\
        --dst_root /p/vast1/vargas58/OMol4M \\
        --src_root /p/lustre5/vargas58/OMol4M \\
        --inputs_root /p/lustre5/bennion1/Omol2025-4M-DiversitySet \\
        --job_list /p/lustre5/vargas58/generator_working/job_lists/omol.txt \\
        --scan_l1_root --scan_subdirs omol --workers 32 --out l1_merge_plan.tsv

--scan_l1_root discovers what level 1 actually wrote. --l1_remaining_list is
the runner's remaining-jobs file; jobs absent from it are added as candidates,
and a scanned folder whose job is still listed as remaining is flagged
l1_still_in_remaining (in progress, crashed, or the list is stale).

Scanned folders that match no job_list line (other verticals sharing the
results root) are counted per top-level dir and written to <out>.unmatched.txt;
they are classified only with --include_unmatched. --prefix restricts planning
to a subtree and --limit N takes a seeded random sample, so a quick sample is
representative rather than the alphabetically-first N.

    python scripts/merge_l1_into_l0.py summarize --plan l1_merge_plan.tsv

    python scripts/merge_l1_into_l0.py apply --plan l1_merge_plan.tsv \\
        --actions OVERLAY,REPLACE --workers 16
"""
import argparse
import contextlib
import csv
import io
import json
import logging
import os
import random
import shutil
import sys
import time
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

from qtaim_gen.source.utils.atomic_write import atomic_json_write
from qtaim_gen.source.utils.io import multiwfn_out_complete
from qtaim_gen.source.utils.validation import (
    get_charge_spin_n_atoms_from_folder,
    validation_checks,
)

RESULT_JSONS = [
    "timings.json",
    "charge.json",
    "bond.json",
    "fuzzy_full.json",
    "qtaim.json",
    "other.json",
    "orca.json",
]
L1_ROUTINES = [
    "vdd",
    "mbis",
    "chelpg",
    "ibsi_bond",
    "elf_fuzzy",
    "mbis_fuzzy_density",
    "mbis_fuzzy_spin",
]
L1_ONLY_KEYS = {
    "charge.json": ["vdd", "mbis", "chelpg"],
    "bond.json": ["ibsi_bond"],
    "fuzzy_full.json": ["elf_fuzzy", "mbis_fuzzy_density", "mbis_fuzzy_spin"],
    "timings.json": L1_ROUTINES,
}
L1_ONLY_ZIP_MEMBERS = {r + ".out" for r in L1_ROUTINES}
TIMINGS_PATCHED_KEY = "_timings_patched"
QTAIM_ZIP_MEMBERS = ("qtaim.out", "CPprop.txt")
# Root-level files worth carrying to the destination (add-only). Wavefunctions
# (.wfn/.wfx/.gbw), density matrices and Multiwfn intermediates stay behind.
SALVAGE_EXTS = (".json", ".out", ".tar.zst", ".gbw.zstd0", ".tgz", ".inp", ".xyz")
SALVAGE_EXCLUDE = {"orca.out", "output.out"}
# A second input file in the job root is not inert: validation reads
# charge/spin/n_atoms from whichever *.inp os.listdir yields first, so a decoy
# swaps the molecule or crashes the parser (orca.property.inp has no "* xyz"
# block, which raises UnboundLocalError in get_spin_charge_from_orca_inp).
INPUT_EXTS = (".inp", ".in")
INPUT_DECOYS = {"orca.property.inp", "convert.in"}
SALVAGE_MARKER = "l1_salvage.json"
JOB_MARKERS = ("generator", "gbw_analysis.log", "orca.inp", "timings.json")
MARKER = "l1_merge.json"
BACKUP_DIR = ".pre_l1_merge"
CANARY_KEY = "hirshfeld"

PLAN_COLUMNS = [
    "rel",
    "l0_dir",
    "l1_dir",
    "l0_state",
    "l1_state",
    "l0_flags",
    "l1_flags",
    "l0_reason",
    "l1_reason",
    "n_atoms_l0",
    "n_atoms_l1",
    "spin_l0",
    "spin_l1",
    "canary_max_abs_diff",
    "l1_keys_to_add",
    "action",
    "reason",
    "l0_fp",
    "l1_fp",
]
APPLY_COLUMNS = ["rel", "action", "result", "detail"]

STRICT_DEFAULT = {
    "check_orca": True,
    "check_bcp_count": True,
    "require_qtaim_provenance": True,
}
LOOSE = {"check_orca": False, "check_bcp_count": False, "require_qtaim_provenance": False}


# --------------------------------------------------------------------------
# job lists and paths
# --------------------------------------------------------------------------
def read_list(path):
    out = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            out.append(line.rstrip("/"))
    return out


def rel_of(path, root):
    root = root.rstrip("/")
    if path == root or path.startswith(root + "/"):
        return path[len(root):].lstrip("/")
    return None


def scan_job_dirs(root, workers=16):
    """Breadth-first walk pruned at job folders (any JOB_MARKERS entry)."""
    root = root.rstrip("/")
    found = []
    errors = []

    def visit(d):
        try:
            with os.scandir(d) as it:
                entries = list(it)
        except OSError as e:
            return d, None, str(e)
        names = {e.name for e in entries}
        if any(m in names for m in JOB_MARKERS):
            return d, [], None
        subs = [
            e.path
            for e in entries
            if e.is_dir(follow_symlinks=False) and not e.name.startswith(".")
        ]
        return d, subs, None

    frontier = [root]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        while frontier:
            nxt = []
            for d, subs, err in ex.map(visit, frontier):
                if err:
                    errors.append((d, err))
                elif subs == [] and d != root:
                    found.append(d)
                elif subs:
                    nxt.extend(subs)
            frontier = nxt
    return found, errors


class Universe:
    """Canonical relative paths from the full job list plus a last-2-component
    index used to remap folders found at a different depth."""

    def __init__(self, rels):
        self.rels = set(rels)
        self.by_tail = defaultdict(list)
        for r in rels:
            parts = r.split("/")
            self.by_tail["/".join(parts[-2:])].append(r)

    def canonicalize(self, rel):
        """Return (canonical_rel, kind) with kind in exact|remap|ambiguous|unmatched."""
        if rel in self.rels:
            return rel, "exact"
        parts = rel.split("/")
        cands = self.by_tail.get("/".join(parts[-2:]), [])
        cands = [
            c for c in cands if c.endswith("/" + rel) or rel.endswith("/" + c) or c == rel
        ]
        if len(cands) == 1:
            return cands[0], "remap"
        if len(cands) > 1:
            return None, "ambiguous"
        return None, "unmatched"


# --------------------------------------------------------------------------
# per-side classification
# --------------------------------------------------------------------------
class _ListHandler(logging.Handler):
    def __init__(self, buf):
        super().__init__(level=logging.WARNING)
        self.buf = buf

    def emit(self, record):
        self.buf.append(record.getMessage())


def _validate(folder, full_set, flags):
    buf = []
    log = logging.getLogger(f"merge_l1.validate.{os.getpid()}")
    log.handlers.clear()
    log.propagate = False
    log.setLevel(logging.WARNING)
    log.addHandler(_ListHandler(buf))
    with contextlib.redirect_stdout(io.StringIO()):
        try:
            ok = bool(
                validation_checks(
                    folder,
                    full_set=full_set,
                    verbose=False,
                    move_results=True,
                    logger=log,
                    check_orca=flags["check_orca"],
                    check_bcp_count=flags["check_bcp_count"],
                    require_qtaim_provenance=flags["require_qtaim_provenance"],
                )
            )
        except Exception as e:
            ok = False
            buf.append(f"exception {type(e).__name__}: {e}")
    reason = buf[0].replace("\t", " ").replace("\n", " ")[:200] if buf else ""
    return ok, reason


def fingerprint(job_dir):
    if not job_dir or not os.path.isdir(job_dir):
        return "missing"
    parts = []
    gen = os.path.join(job_dir, "generator")
    for name in RESULT_JSONS + ["out_files.zip", MARKER]:
        p = os.path.join(gen, name)
        try:
            st = os.stat(p)
            parts.append(f"{name}:{st.st_size}:{int(st.st_mtime)}")
        except OSError:
            pass
    if os.path.exists(os.path.join(job_dir, ".processing.lock")):
        parts.append("lock")
    return ";".join(parts) or "empty"


def hirshfeld_sum_error(job_dir, formal_charge):
    """|sum(hirshfeld charges) - formal charge|, or None when unreadable.
    A wavefunction with the wrong electron count yields charges that still pass
    every structural check but sum to the formal charge plus an integer."""
    c = _load_gen_json(job_dir, "charge.json")
    try:
        q = c[CANARY_KEY]["charge"]
        vals = list(q.values()) if isinstance(q, dict) else list(q)
        return abs(sum(float(v) for v in vals) - float(formal_charge))
    except (KeyError, TypeError, ValueError):
        return None


def classify_side(job_dir, strict, quiet_minutes, lock_max_age_hours=6.0, charge_sum_tol=0.5):
    s = {
        "dir": job_dir or "",
        "state": "MISSING",
        "reason": "",
        "flags": [],
        "n_atoms": "",
        "spin_tf": "",
        "fp": "missing",
    }
    if not job_dir or not os.path.isdir(job_dir):
        return s
    s["fp"] = fingerprint(job_dir)
    try:
        names = set(os.listdir(job_dir))
    except OSError as e:
        s["state"] = "UNREADABLE"
        s["reason"] = str(e)
        return s
    if ".processing.lock" in names:
        # a lock older than the longest possible walltime is a leftover from a
        # killed worker, not a running job
        try:
            age_h = (time.time() - os.stat(os.path.join(job_dir, ".processing.lock")).st_mtime) / 3600.0
        except OSError:
            age_h = 0.0
        s["flags"].append("locked" if age_h < lock_max_age_hours else "stale_lock")
    if any(n in names for n in ("timings.json", "memory.json", "settings.ini")) or any(
        n.endswith((".mfwn", ".wfn", ".wfx", ".gbw")) for n in names
    ):
        s["flags"].append("root_intermediates")
    if any(n.endswith((".tar.zst", ".gbw.zstd0", ".tgz")) for n in names):
        s["flags"].append("compressed_sources")
    gen = os.path.join(job_dir, "generator")
    if not os.path.isdir(gen):
        s["state"] = "NO_GENERATOR"
        s["reason"] = "no generator/ dir"
        return s
    gen_names = set(os.listdir(gen))
    if MARKER in gen_names:
        s["flags"].append("merged_before")
    if any(n.endswith(".corrupt") for n in gen_names):
        s["flags"].append("corrupt_quarantine")
    missing = [n for n in RESULT_JSONS if n not in gen_names and n != "orca.json"]
    if missing:
        s["flags"].append("missing:" + "+".join(m.split(".")[0] for m in missing))
    newest = 0.0
    for base, nm in [(gen, n) for n in gen_names] + [(job_dir, n) for n in names]:
        try:
            newest = max(newest, os.stat(os.path.join(base, nm)).st_mtime)
        except OSError:
            pass
    if quiet_minutes > 0 and time.time() - newest < quiet_minutes * 60:
        s["flags"].append("recent")

    try:
        with contextlib.redirect_stdout(io.StringIO()):
            dft = get_charge_spin_n_atoms_from_folder(job_dir)
    except Exception:
        dft = None
    if not dft:
        s["state"] = "INVALID"
        s["reason"] = "no parsable .inp"
        return s
    s["n_atoms"] = len(dft["mol"])
    s["spin_tf"] = int(dft.get("spin", 1) != 1)

    err = hirshfeld_sum_error(job_dir, dft.get("charge", 0))
    if err is not None and err > charge_sum_tol:
        s["flags"].append(f"charge_sum_bad:{err:.2f}")
        s["state"] = "INVALID"
        s["reason"] = f"hirshfeld charges sum to formal {err:+.2f} e off: bad wavefunction (electron count)"
        return s

    ok1, r1 = _validate(job_dir, 1, strict)
    if ok1:
        lacking = missing_l1_keys(job_dir, s["n_atoms"], bool(s["spin_tf"]))
        if not lacking:
            s["state"] = "L1_VALID"
            return s
        s["flags"].append("l1_keys_missing:" + "+".join(lacking))
        r1 = "L1 keys missing: " + ",".join(lacking)
    ok0, r0 = _validate(job_dir, 0, strict)
    if ok0:
        s["state"] = "L0_VALID"
        s["reason"] = "L1 fail: " + r1
    else:
        s["state"] = "INVALID"
        s["reason"] = "L0 fail: " + r0
    if strict != LOOSE:
        okl1, _ = _validate(job_dir, 1, LOOSE)
        if okl1:
            s["flags"].append("l1_loose_only")
        elif not ok0:
            okl0, _ = _validate(job_dir, 0, LOOSE)
            if okl0:
                s["flags"].append("l0_loose_only")
    return s


def _load_gen_json(job_dir, name):
    p = os.path.join(job_dir, "generator", name)
    try:
        with open(p, "r") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def missing_l1_keys(job_dir, n_atoms, spin_tf):
    """L1-only keys a level-1 folder must hold. validation_checks does not pass
    full_set to validate_charge_dict, so a folder missing mbis/vdd/chelpg still
    passes it; this closes that gap for merge decisions."""
    expected = {
        "charge.json": ["vdd", "mbis", "chelpg"],
        "bond.json": [] if (n_atoms is not None and n_atoms <= 2) else ["ibsi_bond"],
        "fuzzy_full.json": ["elf_fuzzy", "mbis_fuzzy_density"] + (["mbis_fuzzy_spin"] if spin_tf else []),
    }
    missing = []
    for name, keys in expected.items():
        d = _load_gen_json(job_dir, name) or {}
        missing += [f"{name.split('.')[0]}:{k}" for k in keys if k not in d]
    return missing


def canary_diff(l0_dir, l1_dir):
    a = _load_gen_json(l0_dir, "charge.json")
    b = _load_gen_json(l1_dir, "charge.json")
    try:
        qa = a[CANARY_KEY]["charge"]
        qb = b[CANARY_KEY]["charge"]
    except (KeyError, TypeError):
        return None, "canary_missing"
    if isinstance(qa, dict) and isinstance(qb, dict):
        if set(qa) != set(qb):
            return None, "canary_atom_labels_differ"
        pairs = [(qa[k], qb[k]) for k in qa]
    elif isinstance(qa, list) and isinstance(qb, list):
        if len(qa) != len(qb):
            return None, "canary_len_mismatch"
        pairs = list(zip(qa, qb))
    else:
        return None, "canary_type_mismatch"
    if not pairs:
        return None, "canary_empty"
    return max(abs(float(x) - float(y)) for x, y in pairs), ""


def keys_to_add(l0_dir, l1_dir):
    """Which L1-only keys the L1 side holds that the L0 side lacks."""
    out = []
    for name, keys in L1_ONLY_KEYS.items():
        a = _load_gen_json(l0_dir, name) or {}
        b = _load_gen_json(l1_dir, name) or {}
        for k in keys:
            if k in b and k not in a:
                out.append(f"{name.split('.')[0]}:{k}")
    return out


# --------------------------------------------------------------------------
# plan
# --------------------------------------------------------------------------
_PLAN_CTX = {}


def _plan_init(strict, quiet_minutes, canary_tol, lock_max_age_hours, charge_sum_tol):
    _PLAN_CTX["strict"] = strict
    _PLAN_CTX["quiet_minutes"] = quiet_minutes
    _PLAN_CTX["canary_tol"] = canary_tol
    _PLAN_CTX["lock_max_age_hours"] = lock_max_age_hours
    _PLAN_CTX["charge_sum_tol"] = charge_sum_tol


def plan_one(task):
    rel, l0_dir, l1_dir, extra_flags = task
    strict = _PLAN_CTX["strict"]
    qm = _PLAN_CTX["quiet_minutes"]
    tol = _PLAN_CTX["canary_tol"]
    la = _PLAN_CTX["lock_max_age_hours"]
    cst = _PLAN_CTX["charge_sum_tol"]
    try:
        s0 = classify_side(l0_dir, strict, 0, la, cst)
        s1 = classify_side(l1_dir, strict, qm, la, cst)
    except Exception as e:
        return {
            "rel": rel, "l0_dir": l0_dir, "l1_dir": l1_dir or "",
            "l0_state": "ERROR", "l1_state": "ERROR", "action": "ERROR",
            "reason": f"{type(e).__name__}: {e}"[:200],
        }
    row = {
        "rel": rel,
        "l0_dir": s0["dir"],
        "l1_dir": s1["dir"],
        "l0_state": s0["state"],
        "l1_state": s1["state"],
        "l0_flags": "|".join(s0["flags"]),
        "l1_flags": "|".join(extra_flags + s1["flags"]),
        "l0_reason": s0["reason"],
        "l1_reason": s1["reason"],
        "n_atoms_l0": s0["n_atoms"],
        "n_atoms_l1": s1["n_atoms"],
        "spin_l0": s0["spin_tf"],
        "spin_l1": s1["spin_tf"],
        "canary_max_abs_diff": "",
        "l1_keys_to_add": "",
        "action": "",
        "reason": "",
        "l0_fp": s0["fp"],
        "l1_fp": s1["fp"],
    }
    st0, st1 = s0["state"], s1["state"]
    l0_ok = st0 in ("L0_VALID", "L1_VALID")
    l1_ok = st1 in ("L0_VALID", "L1_VALID")

    if "locked" in s1["flags"] or "recent" in s1["flags"]:
        row["action"] = "SKIP_L1_RUNNING"
        row["reason"] = "L1 folder locked or modified within quiet window"
        return row
    if "locked" in s0["flags"]:
        row["action"] = "SKIP_L0_LOCKED"
        row["reason"] = "L0 folder holds .processing.lock"
        return row

    if st1 in ("MISSING", "NO_GENERATOR", "UNREADABLE"):
        if l0_ok:
            row["action"] = "NOTHING" if st1 == "MISSING" else "KEEP_L0"
            row["reason"] = f"L1 {st1.lower()}"
            return row
        salvage = [n for n in salvageable_files(l1_dir) if not n.endswith(".inp")] if st1 == "NO_GENERATOR" else []
        if salvage:
            row["action"] = "SALVAGE"
            row["reason"] = f"needs rerun; {len(salvage)} root files to carry: " + ",".join(salvage)[:150]
        else:
            row["action"] = "NEEDS_RERUN"
            row["reason"] = f"L0 {st0.lower()}, L1 {st1.lower()}"
        return row

    qtaim_words = ("qtaim.out", "CPprop", "bond critical", "bond-CP", "bond CPs")

    def _qtaim_only_failure(s):
        return "l1_loose_only" in s["flags"] and any(w in s["reason"] for w in qtaim_words)

    # Source is a level-1 record whose only defect is QTAIM provenance. Its
    # L1-only keys do not depend on qtaim, so they can still be overlaid onto a
    # destination that keeps its own (fixed) qtaim.json.
    src_l1_loose = (
        st1 == "INVALID"
        and _qtaim_only_failure(s1)
        and not missing_l1_keys(l1_dir, s1["n_atoms"], bool(s1["spin_tf"]))
    )
    src_has_l1 = st1 == "L1_VALID" or src_l1_loose

    # Destination holds a fuller record that fails only the QTAIM provenance /
    # bond-CP checks, while the source's qtaim passed them: patch qtaim.json
    # (and qtaim.out / CPprop.txt in the zip) from the source instead of
    # discarding the destination's extra keys with a wholesale replace. Only
    # when the source is NOT itself a full level-1 record, else REPLACE wins.
    if (
        st0 == "INVALID"
        and st1 == "L0_VALID"
        and _qtaim_only_failure(s0)
        and s0["n_atoms"] == s1["n_atoms"]
        and s0["spin_tf"] == s1["spin_tf"]
    ):
        diff, why = canary_diff(l0_dir, l1_dir)
        row["canary_max_abs_diff"] = "" if diff is None else f"{diff:.3e}"
        if diff is not None and diff <= tol:
            row["action"] = "PATCH_QTAIM"
            row["reason"] = "dst fails only qtaim provenance; src qtaim passes: " + s0["reason"][:120]
            return row

    if l0_ok and (l1_ok or src_has_l1):
        if s0["n_atoms"] != s1["n_atoms"]:
            row["action"] = "CONFLICT_NATOMS"
            row["reason"] = f"n_atoms {s0['n_atoms']} vs {s1['n_atoms']}"
            return row
        if s0["spin_tf"] != s1["spin_tf"]:
            row["action"] = "CONFLICT_SPIN"
            row["reason"] = f"spin_tf {s0['spin_tf']} vs {s1['spin_tf']}"
            return row
        diff, why = canary_diff(l0_dir, l1_dir)
        row["canary_max_abs_diff"] = "" if diff is None else f"{diff:.3e}"
        if diff is None:
            row["action"] = "CONFLICT_CANARY"
            row["reason"] = why
            return row
        if diff > tol:
            row["action"] = "CONFLICT_CANARY"
            row["reason"] = f"{CANARY_KEY} max|diff| {diff:.3e} > {tol:g}"
            return row

    if src_has_l1:
        note = " (src qtaim fails provenance, not taken)" if src_l1_loose else ""
        if st0 == "L1_VALID":
            row["action"] = "ALREADY_L1"
            row["reason"] = "L0 side already validates at level 1"
        elif st0 == "L0_VALID":
            adds = keys_to_add(l0_dir, l1_dir)
            row["l1_keys_to_add"] = ",".join(adds)
            if adds:
                row["action"] = "OVERLAY"
                row["reason"] = f"add {len(adds)} L1 keys" + note
            else:
                row["action"] = "KEEP_L0"
                row["reason"] = "L1 valid but no missing keys on L0 side"
        elif src_l1_loose and "merged_before" in s0["flags"] and _qtaim_only_failure(s0):
            row["action"] = "ALREADY_L1_LOOSE"
            row["reason"] = "dst already holds this L1 record; both fail only qtaim provenance"
        elif src_l1_loose:
            row["action"] = "REPLACE_LOOSE"
            row["reason"] = f"L0 {st0.lower()}; src is L1 but fails qtaim provenance"
        else:
            row["action"] = "REPLACE"
            row["reason"] = f"L0 {st0.lower()}: {s0['reason']}"[:200]
        return row

    if st1 == "L0_VALID":
        if l0_ok:
            row["action"] = "KEEP_L0"
            row["reason"] = "L1 side only valid at level 0: " + s1["reason"]
        else:
            row["action"] = "REPLACE_L0"
            row["reason"] = f"L0 {st0.lower()}; L1 valid at level 0 only"
        return row

    if l0_ok:
        row["action"] = "KEEP_L0"
        row["reason"] = "L1 invalid: " + s1["reason"]
        return row
    # Neither side valid. If the source folder still holds files worth keeping
    # (compressed inputs, Multiwfn .out, partial JSONs), carry them over so the
    # rerun can start from lustre; the job stays a rerun candidate either way.
    salvage = [n for n in salvageable_files(l1_dir) if not n.endswith(".inp")]
    if salvage:
        row["action"] = "SALVAGE"
        row["reason"] = f"needs rerun; {len(salvage)} root files to carry: " + ",".join(salvage)[:150]
    else:
        row["action"] = "NEEDS_RERUN"
        row["reason"] = f"L0: {s0['reason']} ; L1: {s1['reason']}"[:200]
    return row


def build_targets(args):
    if args.rows_from:
        acts = set(a.strip() for a in args.rows_actions.split(",") if a.strip())
        tasks = []
        with open(args.rows_from, "r", newline="") as f:
            for r in csv.DictReader(f, delimiter="\t"):
                if acts and r["action"] not in acts:
                    continue
                tasks.append((r["rel"], r["l0_dir"], r["l1_dir"] or None, ["replan"]))
        print(f"[plan] re-planning {len(tasks)} rows from {args.rows_from} with actions {sorted(acts) or 'ALL'}")
        if args.limit and args.limit < len(tasks):
            tasks = random.Random(args.seed).sample(tasks, args.limit)
            tasks.sort()
        return tasks
    if not args.inputs_root or not args.job_list:
        raise SystemExit("plan needs --inputs_root and --job_list (or --rows_from)")
    inputs_root = args.inputs_root.rstrip("/")
    l0_root = args.l0_root.rstrip("/")
    l1_root = args.l1_root.rstrip("/")

    universe_rels = []
    not_under_root = []
    for p in read_list(args.job_list):
        r = rel_of(p, inputs_root)
        if r is None:
            not_under_root.append(p)
        else:
            universe_rels.append(r)
    uni = Universe(universe_rels)
    print(f"[plan] universe: {len(uni.rels)} jobs from {args.job_list}")
    if not_under_root:
        print(f"[plan] WARNING {len(not_under_root)} job_list lines not under inputs_root (first: {not_under_root[0]})")

    targets = {}

    def add(rel, l1_dir, flags):
        t = targets.setdefault(rel, {"l1_dirs": [], "flags": []})
        if l1_dir and l1_dir not in t["l1_dirs"]:
            t["l1_dirs"].append(l1_dir)
        t["flags"].extend(f for f in flags if f not in t["flags"])

    remaining = set()
    if args.l1_remaining_list:
        n_bad = 0
        for p in read_list(args.l1_remaining_list):
            r = rel_of(p, inputs_root)
            if r is None:
                n_bad += 1
                continue
            canon, _ = uni.canonicalize(r)
            remaining.add(canon or r)
        expected_done = uni.rels - remaining
        for r in sorted(expected_done):
            add(r, os.path.join(l1_root, r), ["l1_expected_done"])
        print(
            f"[plan] l1_remaining_list: {len(remaining)} still pending, "
            f"{len(expected_done)} expected done -> candidates ({n_bad} lines not under inputs_root)"
        )

    if args.scan_l1_root:
        t0 = time.time()
        scan_roots = [l1_root]
        if args.scan_subdirs:
            scan_roots = [os.path.join(l1_root, s.strip("/")) for s in args.scan_subdirs.split(",") if s.strip()]
        found, errs = [], []
        for sr in scan_roots:
            if not os.path.isdir(sr):
                print(f"[plan] WARNING scan root {sr} does not exist")
                continue
            f_, e_ = scan_job_dirs(sr, workers=args.scan_workers)
            found += f_
            errs += e_
        print(f"[plan] scanned {scan_roots}: {len(found)} job folders, {len(errs)} unreadable dirs, {time.time() - t0:.0f}s")
        kinds = Counter()
        unmatched = []
        for d in found:
            r = rel_of(d, l1_root)
            canon, kind = uni.canonicalize(r)
            kinds[kind] += 1
            if canon is None:
                unmatched.append((r, kind))
                if args.include_unmatched:
                    add(r, d, [f"scan_{kind}"])
            else:
                flags = ["scan_found"] + ([f"scan_{kind}"] if kind != "exact" else [])
                if canon in remaining:
                    flags.append("l1_still_in_remaining")
                add(canon, d, flags)
        print(f"[plan] scan match kinds: {dict(kinds)}")
        if unmatched:
            side = args.out + ".unmatched.txt"
            with open(side, "w") as f:
                for r, kind in unmatched:
                    f.write(f"{kind}\t{os.path.join(l1_root, r)}\n")
            census = Counter(r.split("/")[0] for r, _ in unmatched)
            print(f"[plan] {len(unmatched)} scanned folders not in job_list -> {side}"
                  + ("" if args.include_unmatched else " (not classified; pass --include_unmatched to classify)"))
            print("| top-level dir | unmatched folders |\n|---|---|")
            for k, c in census.most_common(20):
                print(f"| {k} | {c} |")

    if args.probe_all:
        for r in uni.rels:
            if r not in targets:
                add(r, os.path.join(l1_root, r), ["probe"])

    if args.include_l0_only:
        for r in uni.rels:
            if r not in targets:
                add(r, None, ["l0_only"])

    tasks = []
    for rel in sorted(targets):
        t = targets[rel]
        l1_dirs = [d for d in t["l1_dirs"] if d]
        if len(l1_dirs) > 1:
            existing = [d for d in l1_dirs if os.path.isdir(d)]
            l1_dirs = existing or l1_dirs[:1]
        l1_dirs = l1_dirs or [None]
        flags = list(t["flags"])
        if len(l1_dirs) > 1:
            flags.append(f"dup_l1_dirs:{len(l1_dirs)}")
        l0_dir = os.path.join(l0_root, rel)
        for d in l1_dirs:
            tasks.append((rel, l0_dir, d, flags))
    if args.prefix:
        tasks = [t for t in tasks if t[0].startswith(args.prefix)]
        print(f"[plan] prefix {args.prefix!r}: {len(tasks)} tasks")
    if args.limit and args.limit < len(tasks):
        tasks = random.Random(args.seed).sample(tasks, args.limit)
        tasks.sort()
        print(f"[plan] random sample of {len(tasks)} tasks (seed {args.seed})")
    return tasks


def cmd_plan(args):
    strict = {
        "check_orca": not args.no_check_orca,
        "check_bcp_count": not args.no_bcp_count,
        "require_qtaim_provenance": not args.no_provenance,
    }
    tasks = build_targets(args)
    done = set()
    mode = "w"
    if args.resume and os.path.exists(args.out):
        with open(args.out, "r", newline="") as f:
            for r in csv.DictReader(f, delimiter="\t"):
                done.add((r["rel"], r["l1_dir"]))
        mode = "a"
        tasks = [t for t in tasks if (t[0], t[2] or "") not in done]
        print(f"[plan] resume: {len(done)} rows already in {args.out}, {len(tasks)} remaining")
    print(f"[plan] {len(tasks)} tasks, strict={strict}, workers={args.workers}")

    n = 0
    with open(args.out, mode, newline="") as f:
        w = csv.DictWriter(f, fieldnames=PLAN_COLUMNS, delimiter="\t", extrasaction="ignore")
        if mode == "w":
            w.writeheader()
        with ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=_plan_init,
            initargs=(strict, args.quiet_minutes, args.canary_tol, args.lock_max_age_hours, args.charge_sum_tol),
        ) as ex:
            for row in ex.map(plan_one, tasks, chunksize=args.chunksize):
                w.writerow({c: row.get(c, "") for c in PLAN_COLUMNS})
                n += 1
                if n % 5000 == 0:
                    f.flush()
                    print(f"[plan] {n}/{len(tasks)}", flush=True)
    print(f"[plan] wrote {n} rows to {args.out}")
    summarize(args.out)
    return 0


# --------------------------------------------------------------------------
# summary
# --------------------------------------------------------------------------
def _pct(v, edges=(0.5, 0.9, 0.99, 1.0)):
    if not v:
        return "n/a"
    v = sorted(v)
    parts = []
    for q in edges:
        i = min(len(v) - 1, max(0, int(round(q * (len(v) - 1)))))
        parts.append(f"p{int(q * 100)}={v[i]:.2e}")
    return " ".join(parts)


def summarize(plan_path):
    rows = []
    with open(plan_path, "r", newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    if not rows:
        print("[summary] empty plan")
        return
    print(f"\n[summary] {len(rows)} rows in {plan_path}")

    def table(title, counter, key_names):
        print(f"\n{title}")
        print("| " + " | ".join(key_names) + " | count |")
        print("|" + "---|" * (len(key_names) + 1))
        for k, c in counter.most_common():
            ks = k if isinstance(k, tuple) else (k,)
            print("| " + " | ".join(str(x) for x in ks) + f" | {c} |")

    table("Actions", Counter(r["action"] for r in rows), ["action"])
    table(
        "State pairs (L0 side, L1 side)",
        Counter((r["l0_state"], r["l1_state"]) for r in rows),
        ["l0_state", "l1_state"],
    )
    fl = Counter()
    for r in rows:
        for side in ("l0_flags", "l1_flags"):
            for x in filter(None, r[side].split("|")):
                fl[(side, x.split(":")[0])] += 1
    table("Flags", fl, ["side", "flag"])

    diffs = [float(r["canary_max_abs_diff"]) for r in rows if r["canary_max_abs_diff"]]
    print(f"\nCanary ({CANARY_KEY} charge max|L0-L1|) over {len(diffs)} pairs: {_pct(diffs)}")

    adds = Counter()
    for r in rows:
        for k in filter(None, r["l1_keys_to_add"].split(",")):
            adds[k] += 1
    table("L1 keys that would be added (OVERLAY rows)", adds, ["key"])

    reasons = Counter()
    for r in rows:
        if r["action"] in ("KEEP_L0", "NEEDS_RERUN", "REPLACE", "CONFLICT_CANARY", "ERROR"):
            reasons[(r["action"], r["reason"][:90])] += 1
    print("\nTop reasons (non-merge outcomes)")
    print("| action | reason | count |")
    print("|---|---|---|")
    for (a, why), c in reasons.most_common(25):
        print(f"| {a} | {why} | {c} |")

    print("\nExamples per action (first 3 rel paths)")
    seen = defaultdict(list)
    for r in rows:
        if len(seen[r["action"]]) < 3:
            seen[r["action"]].append(r["rel"])
    for a in sorted(seen):
        for rel in seen[a]:
            print(f"  {a:18s} {rel}")


def cmd_summarize(args):
    summarize(args.plan)
    return 0


# --------------------------------------------------------------------------
# apply
# --------------------------------------------------------------------------
_APPLY_CTX = {}


def _apply_init(strict, backup, force, lock_max_age_hours):
    _APPLY_CTX["strict"] = strict
    _APPLY_CTX["backup"] = backup
    _APPLY_CTX["force"] = force
    _APPLY_CTX["lock_max_age_hours"] = lock_max_age_hours


def _lock_live(job_dir, max_age_hours):
    p = os.path.join(job_dir or "", ".processing.lock")
    try:
        age_h = (time.time() - os.stat(p).st_mtime) / 3600.0
    except OSError:
        return False
    return age_h < max_age_hours


def _backup(job_dir, rel_paths):
    bdir = os.path.join(job_dir, BACKUP_DIR)
    for rp in rel_paths:
        src = os.path.join(job_dir, rp)
        if not os.path.exists(src):
            continue
        dst = os.path.join(bdir, rp)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.isdir(src):
            if os.path.exists(dst):
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)


def _restore(job_dir, rel_paths):
    bdir = os.path.join(job_dir, BACKUP_DIR)
    for rp in rel_paths:
        src = os.path.join(bdir, rp)
        dst = os.path.join(job_dir, rp)
        if os.path.isdir(src):
            if os.path.exists(dst):
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
        elif os.path.exists(src):
            shutil.copy2(src, dst)
        elif os.path.isdir(dst):
            shutil.rmtree(dst)
        elif os.path.exists(dst):
            os.remove(dst)


def salvageable_files(job_dir):
    """Root-level files of job_dir that the migration carries over."""
    try:
        names = os.listdir(job_dir)
    except OSError:
        return []
    out = []
    for n in names:
        if n in SALVAGE_EXCLUDE or n.startswith("."):
            continue
        if not n.endswith(SALVAGE_EXTS):
            continue
        if os.path.isfile(os.path.join(job_dir, n)):
            out.append(n)
    return sorted(out)


def _copy_root_files(src_dir, dst_dir):
    """Copy salvageable root files src -> dst, never overwriting. Returns
    (copied, overlap) where overlap lists files present on both sides with a
    different size (dst kept)."""
    copied, overlap = [], []
    os.makedirs(dst_dir, exist_ok=True)
    try:
        dst_names = os.listdir(dst_dir)
    except OSError:
        dst_names = []
    have_input = any(
        n.endswith(INPUT_EXTS) and n not in INPUT_DECOYS for n in dst_names
    )
    for n in salvageable_files(src_dir):
        if n.endswith(INPUT_EXTS):
            # never carry a decoy, and carry a real input only when the
            # destination has none at all
            if n in INPUT_DECOYS or have_input:
                continue
            have_input = True
        s = os.path.join(src_dir, n)
        d = os.path.join(dst_dir, n)
        if os.path.exists(d):
            if os.path.getsize(d) != os.path.getsize(s):
                overlap.append(n)
            continue
        tmp = d + ".l1.tmp"
        shutil.copy2(s, tmp)
        os.replace(tmp, d)
        copied.append(n)
    log1 = os.path.join(src_dir, "gbw_analysis.log")
    if os.path.isfile(log1) and not os.path.exists(os.path.join(dst_dir, "gbw_analysis.l1.log")):
        shutil.copy2(log1, os.path.join(dst_dir, "gbw_analysis.l1.log"))
        copied.append("gbw_analysis.l1.log")
    return copied, overlap


def _zip_union_add_only(dst_zip, src_zip, only_members=None, exclude=()):
    """Add members of src_zip that dst_zip lacks. Existing members untouched.
    only_members=None means every member not in exclude. A Multiwfn .out that
    never reached its final menu (walltime kill) is not added; it is returned
    in the second list so the caller can record it.
    Returns (added, skipped_incomplete)."""

    def wanted(name):
        if name in exclude:
            return False
        return only_members is None or name in only_members

    if not os.path.isfile(src_zip):
        return [], []
    have = set()
    if os.path.isfile(dst_zip):
        with zipfile.ZipFile(dst_zip, "r") as d:
            have = set(d.namelist())
    added, skipped = [], []
    with zipfile.ZipFile(src_zip, "r") as s:
        want = []
        for info in s.infolist():
            if not wanted(info.filename) or info.filename in have:
                continue
            data = s.read(info.filename)
            if info.filename.endswith(".out") and not multiwfn_out_complete(data):
                skipped.append(info.filename)
                continue
            want.append((info, data))
        if not want:
            return [], skipped
        tmp = dst_zip + ".l1.tmp"
        os.makedirs(os.path.dirname(dst_zip), exist_ok=True)
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as o:
            if have:
                with zipfile.ZipFile(dst_zip, "r") as d:
                    for info in d.infolist():
                        o.writestr(info, d.read(info.filename))
            for info, data in want:
                o.writestr(info, data)
                added.append(info.filename)
    os.replace(tmp, dst_zip)
    return added, skipped


def _overlay(l0_dir, l1_dir):
    gen0 = os.path.join(l0_dir, "generator")
    gen1 = os.path.join(l1_dir, "generator")
    added = {}
    for name, keys in L1_ONLY_KEYS.items():
        a = _load_gen_json(l0_dir, name)
        b = _load_gen_json(l1_dir, name)
        if a is None or b is None:
            raise RuntimeError(f"cannot read {name} on one side")
        new = dict(a)
        got = []
        for k in keys:
            if k in b and k not in new:
                new[k] = b[k]
                got.append(k)
        if name == "timings.json":
            pa = a.get(TIMINGS_PATCHED_KEY) or {}
            pb = b.get(TIMINGS_PATCHED_KEY) or {}
            merged_patch = {**{k: v for k, v in pb.items() if k in got}, **pa}
            if merged_patch:
                new[TIMINGS_PATCHED_KEY] = merged_patch
        if got:
            atomic_json_write(os.path.join(gen0, name), new)
            added[name] = got
    # every .out the source zip holds that the destination lacks, except the
    # qtaim pair: dst keeps its own qtaim.json, so it must keep its own
    # qtaim.out/CPprop.txt (or none) for the bond-CP cross-check to stay honest
    try:
        zadded, zskipped = _zip_union_add_only(
            os.path.join(gen0, "out_files.zip"),
            os.path.join(gen1, "out_files.zip"),
            exclude=QTAIM_ZIP_MEMBERS,
        )
    except zipfile.BadZipFile as e:
        # a corrupt source archive costs only the .out provenance; the JSON
        # keys already merged are still good, so record it and carry on
        zadded, zskipped = [], [f"BadZipFile: {e}"]
    return added, zadded, zskipped


def _replace(l0_dir, l1_dir):
    gen0 = os.path.join(l0_dir, "generator")
    gen1 = os.path.join(l1_dir, "generator")
    tmp = gen0 + ".l1.tmp"
    if os.path.exists(tmp):
        shutil.rmtree(tmp)
    shutil.copytree(gen1, tmp)
    if os.path.isdir(gen0):
        shutil.rmtree(gen0)
    os.rename(tmp, gen0)


def _patch_qtaim(l0_dir, l1_dir):
    """Replace dst qtaim.json plus qtaim.out/CPprop.txt in out_files.zip with src's."""
    gen0 = os.path.join(l0_dir, "generator")
    gen1 = os.path.join(l1_dir, "generator")
    src_q = os.path.join(gen1, "qtaim.json")
    if not os.path.isfile(src_q):
        raise RuntimeError("source has no qtaim.json")
    tmp_q = os.path.join(gen0, "qtaim.json.l1.tmp")
    shutil.copy2(src_q, tmp_q)
    os.replace(tmp_q, os.path.join(gen0, "qtaim.json"))
    src_zip = os.path.join(gen1, "out_files.zip")
    dst_zip = os.path.join(gen0, "out_files.zip")
    if not os.path.isfile(src_zip):
        raise RuntimeError("source has no out_files.zip")
    replaced = []
    tmp = dst_zip + ".l1.tmp"
    with zipfile.ZipFile(src_zip, "r") as s:
        src_names = set(s.namelist())
        if "qtaim.out" not in src_names:
            raise RuntimeError("source zip has no qtaim.out")
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as o:
            if os.path.isfile(dst_zip):
                with zipfile.ZipFile(dst_zip, "r") as d:
                    for info in d.infolist():
                        if info.filename in QTAIM_ZIP_MEMBERS:
                            continue
                        o.writestr(info, d.read(info.filename))
            for name in QTAIM_ZIP_MEMBERS:
                if name in src_names:
                    o.writestr(s.getinfo(name), s.read(name))
                    replaced.append(name)
    os.replace(tmp, dst_zip)
    return replaced


def apply_one(row):
    strict = _APPLY_CTX["strict"]
    backup = _APPLY_CTX["backup"]
    force = _APPLY_CTX["force"]
    rel, action = row["rel"], row["action"]
    l0_dir, l1_dir = row["l0_dir"], row["l1_dir"]
    out = {"rel": rel, "action": action, "result": "", "detail": ""}
    if action not in ("OVERLAY", "REPLACE", "REPLACE_L0", "REPLACE_LOOSE", "PATCH_QTAIM", "SALVAGE"):
        out["result"] = "SKIP_ACTION"
        return out
    try:
        if not force:
            if fingerprint(l1_dir) != row["l1_fp"]:
                out["result"] = "SKIP_CHANGED"
                out["detail"] = "L1 side changed since plan"
                return out
            if fingerprint(l0_dir) != row["l0_fp"]:
                out["result"] = "SKIP_CHANGED"
                out["detail"] = "L0 side changed since plan"
                return out
        la = _APPLY_CTX["lock_max_age_hours"]
        if _lock_live(l1_dir, la) or _lock_live(l0_dir, la):
            out["result"] = "SKIP_LOCKED"
            return out
        marker_path = os.path.join(l0_dir, "generator", MARKER)
        salvage_marker = os.path.join(l0_dir, SALVAGE_MARKER)
        if not force and (os.path.exists(marker_path) or (action == "SALVAGE" and os.path.exists(salvage_marker))):
            out["result"] = "SKIP_MARKER"
            out["detail"] = "already merged (marker present)"
            return out
        os.makedirs(l0_dir, exist_ok=True)

        if action == "SALVAGE":
            # no generator/ to validate; just carry root files, add-only
            copied, overlap = _copy_root_files(l1_dir, l0_dir)
            detail = {"root_copied": copied, "root_overlap_kept_dst": overlap}
            atomic_json_write(
                salvage_marker,
                {"action": action, "source": l1_dir, "time": time.strftime("%Y-%m-%dT%H:%M:%S"), "detail": detail},
            )
            out["result"] = "OK"
            out["detail"] = json.dumps(detail, separators=(",", ":"))
            return out

        target_level = 1 if action in ("OVERLAY", "REPLACE", "PATCH_QTAIM", "REPLACE_LOOSE") else 0
        verify_flags = LOOSE if action == "REPLACE_LOOSE" else strict
        touched = ["generator"]
        root_copied = []
        if backup:
            _backup(l0_dir, touched)
        try:
            if action == "OVERLAY":
                added, zadded, zskipped = _overlay(l0_dir, l1_dir)
                detail = {"json": added, "zip": zadded, "zip_skipped_incomplete": zskipped}
            elif action in ("REPLACE", "REPLACE_L0", "REPLACE_LOOSE"):
                _replace(l0_dir, l1_dir)
                detail = {"replaced": "generator"}
            else:
                detail = {"qtaim_json": "from src", "zip": _patch_qtaim(l0_dir, l1_dir)}
            root_copied, overlap = _copy_root_files(l1_dir, l0_dir)
            detail["root_copied"] = root_copied
            detail["root_overlap_kept_dst"] = overlap
            ok, why = _validate(l0_dir, target_level, verify_flags)
            if not ok:
                raise RuntimeError(f"post-merge validation failed at level {target_level}: {why}")
            atomic_json_write(
                marker_path,
                {
                    "action": action,
                    "source": l1_dir,
                    "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "target_level": target_level,
                    "detail": detail,
                    "canary_max_abs_diff": row.get("canary_max_abs_diff", ""),
                },
            )
            out["result"] = "OK"
            out["detail"] = json.dumps(detail, separators=(",", ":"))
        except Exception as e:
            for n in root_copied:
                try:
                    os.remove(os.path.join(l0_dir, n))
                except OSError:
                    pass
            if backup:
                _restore(l0_dir, touched)
                out["detail"] = f"rolled back: {type(e).__name__}: {e}"[:300]
            else:
                out["detail"] = f"NOT rolled back (no backup): {type(e).__name__}: {e}"[:300]
            out["result"] = "FAILED"
    except Exception as e:
        out["result"] = "ERROR"
        out["detail"] = f"{type(e).__name__}: {e}"[:300]
    return out


def cmd_apply(args):
    strict = {
        "check_orca": not args.no_check_orca,
        "check_bcp_count": not args.no_bcp_count,
        "require_qtaim_provenance": not args.no_provenance,
    }
    actions = set(a.strip() for a in args.actions.split(",") if a.strip())
    with open(args.plan, "r", newline="") as f:
        rows = [r for r in csv.DictReader(f, delimiter="\t") if r["action"] in actions]
    if args.limit:
        rows = rows[: args.limit]
    out_path = args.out or (args.plan + ".applied.tsv")
    print(f"[apply] {len(rows)} rows with actions {sorted(actions)} -> {out_path}")
    if not args.yes:
        print("[apply] refusing to write without --yes")
        return 2
    res = Counter()
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=APPLY_COLUMNS, delimiter="\t")
        w.writeheader()
        with ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=_apply_init,
            initargs=(strict, not args.no_backup, args.force, args.lock_max_age_hours),
        ) as ex:
            for i, r in enumerate(ex.map(apply_one, rows, chunksize=args.chunksize), 1):
                w.writerow(r)
                res[r["result"]] += 1
                if i % 1000 == 0:
                    f.flush()
                    print(f"[apply] {i}/{len(rows)} {dict(res)}", flush=True)
    print("\n| result | count |\n|---|---|")
    for k, c in res.most_common():
        print(f"| {k} | {c} |")
    return 0 if res.get("FAILED", 0) == 0 and res.get("ERROR", 0) == 0 else 1


# --------------------------------------------------------------------------
def _add_strict_flags(p):
    p.add_argument("--no_check_orca", action="store_true", help="do not require orca.json")
    p.add_argument("--no_bcp_count", action="store_true", help="skip the bond-CP count cross-check")
    p.add_argument("--no_provenance", action="store_true", help="do not require qtaim.out provenance")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--chunksize", type=int, default=64)
    p.add_argument("--limit", type=int, default=0, help="process only the first N tasks/rows")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("plan", help="classify both sides and write a plan TSV (read-only)")
    p.add_argument("--l0_root", "--dst_root", dest="l0_root", required=True, help="destination tree (written by apply)")
    p.add_argument("--l1_root", "--src_root", dest="l1_root", required=True, help="source tree (read-only)")
    p.add_argument("--inputs_root", default=None)
    p.add_argument("--job_list", default=None, help="full job list (universe of canonical paths)")
    p.add_argument("--rows_from", default=None, help="re-plan only the rows of this earlier plan TSV (skips list/scan)")
    p.add_argument("--rows_actions", default="", help="with --rows_from: comma list of actions to re-plan (default all)")
    p.add_argument("--charge_sum_tol", type=float, default=0.5, help="|sum(hirshfeld) - formal charge| above this marks the record as a bad wavefunction")
    p.add_argument("--lock_max_age_hours", type=float, default=6.0, help=".processing.lock older than this is stale, not running")
    p.add_argument(
        "--l1_remaining_list",
        default=None,
        help="jobs the level-1 run has NOT done yet; candidates = job_list minus this",
    )
    p.add_argument("--scan_l1_root", action="store_true", help="walk l1_root for job folders (depth-agnostic)")
    p.add_argument("--scan_workers", type=int, default=32)
    p.add_argument("--scan_subdirs", default=None, help="comma list of subdirs of l1_root to walk instead of the whole root (e.g. omol)")
    p.add_argument("--include_unmatched", action="store_true", help="also classify scanned folders absent from job_list")
    p.add_argument("--prefix", default=None, help="only plan jobs whose relative path starts with this (e.g. omol/metal_organics)")
    p.add_argument("--seed", type=int, default=0, help="seed for the --limit random sample")
    p.add_argument("--probe_all", action="store_true", help="also probe l1_root/<rel> for every universe job")
    p.add_argument("--include_l0_only", action="store_true", help="also classify universe jobs with no L1 candidate")
    p.add_argument("--quiet_minutes", type=float, default=30.0, help="L1 folders modified within this window are treated as running")
    p.add_argument("--canary_tol", type=float, default=1e-3, help="max allowed |hirshfeld charge| difference between sides")
    p.add_argument("--out", default="l1_merge_plan.tsv")
    p.add_argument("--resume", action="store_true")
    _add_strict_flags(p)
    p.set_defaults(func=cmd_plan)

    p = sub.add_parser("summarize", help="print the summary tables for an existing plan")
    p.add_argument("--plan", required=True)
    p.set_defaults(func=cmd_summarize)

    p = sub.add_parser("apply", help="execute plan rows (writes under l0_root only)")
    p.add_argument("--plan", required=True)
    p.add_argument("--actions", default="OVERLAY,REPLACE,REPLACE_L0,PATCH_QTAIM,SALVAGE", help="comma list of plan actions to execute (REPLACE_LOOSE is opt-in)")
    p.add_argument("--out", default=None)
    p.add_argument("--no_backup", action="store_true")
    p.add_argument("--force", action="store_true", help="ignore fingerprint mismatch and existing merge marker")
    p.add_argument("--yes", action="store_true", help="actually write")
    p.add_argument("--lock_max_age_hours", type=float, default=6.0, help=".processing.lock older than this is stale, not running")
    _add_strict_flags(p)
    p.set_defaults(func=cmd_apply)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
