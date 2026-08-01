"""audit-qtaim-connectivity: find qtaim.json records with missing bond CPs.

Motivation: a truncated Multiwfn `CPprop.txt` silently loses critical points
from the *tail* of the CP list. `validate_qtaim_dict` cannot detect this because
it only checks that the nuclear-CP count matches the atom count, and Multiwfn
numbers nuclear CPs first - so the surviving prefix always passes.

The signature it does leave is a **broken molecular graph**: with
`bonding_scheme="qtaim"` the BCP set is the graph edge set, so lost BCPs show up
as atoms with no bond critical point at all, or as covalently-close atom pairs
with no BCP between them.

Distinguishing a real defect from legitimate topology needs a reference, since
several verticals (5A_elytes, droplet, noble_gas) are genuinely multi-fragment
and can contain bare ions with no bonds. This audit uses covalent-radius
bonding from the structure LMDB (`get_bonds_from_coords`) as that reference and
reports two signals per record:

- `n_isolated_bonded`: atoms with zero BCPs that the covalent reference says
  are bonded to something. Near-unambiguous: a covalently bound atom always has
  at least one BCP.
- `n_missing_cov_bonds`: covalent reference bonds with no corresponding BCP.
  Broader, and expected to be nonzero for ionic/metal contacts, so use it as a
  ranking signal rather than a pass/fail.

Runs entirely on existing LMDBs or job folders - no wavefunctions, no
recomputation.

Two modes:

  --mode audit    (default) scan for records with missing bond CPs
  --mode explain  attribute a known set of defects to a cause, with the
                  forensic evidence that separates a killed run from ordinary
                  parse behaviour

Explain mode needs CPprop.txt, which only runs after the archiving fix retain.
It reports, per job: whether the shortfall is real truncation or one of three
legitimate merge drops (a CP with no "Connected atoms:" line, two CPs colliding
on one atom pair, an attractor with no nuclear-CP match); whether CPprop.txt
stops mid-write (an OOM/SIGKILL signature) or ends cleanly; Multiwfn's reported
count against the blocks actually present; and the Atoms/Basis/GTF sizes, which
with --compare_csv turn successfully repaired jobs into a control group for the
memory question.

Examples:
    audit-qtaim-connectivity --lmdb_root data/OMol4M_lmdbs \
        --out_csv qtaim_connectivity_audit.csv

    audit-qtaim-connectivity --mode explain \
        --from_csv qtaim_rerun_test_verified.csv \
        --compare_csv qtaim_rerun_test_verified.csv \
        --out_csv residual_causes.csv
"""

import argparse
import collections
import csv
import json
import os
import pickle
import re
import sys
import tempfile
import warnings
import zipfile
from typing import List, Optional

warnings.filterwarnings("ignore")


def bcp_pairs(record: dict) -> List[tuple]:
    """0-based atom pairs from a qtaim.json/lmdb record."""
    pairs = []
    for k in record:
        if k == "_meta" or "_" not in k:
            continue
        try:
            i, j = (int(x) for x in k.split("_"))
        except ValueError:
            continue
        pairs.append((min(i, j), max(i, j)))
    return pairs


def n_components(n_atoms: int, pairs) -> int:
    """Connected components of the BCP graph."""
    adj = {i: set() for i in range(n_atoms)}
    for i, j in pairs:
        if i in adj and j in adj:
            adj[i].add(j)
            adj[j].add(i)
    seen, comps = set(), 0
    for a in range(n_atoms):
        if a in seen:
            continue
        comps += 1
        stack = [a]
        while stack:
            u = stack.pop()
            if u in seen:
                continue
            seen.add(u)
            stack.extend(adj[u] - seen)
    return comps


def audit_record(structure_rec: dict, qtaim_rec: dict, covalent_factor: float) -> dict:
    from qtaim_gen.source.utils.io import get_bonds_from_coords

    mol = structure_rec["molecule"]
    species = [str(el) for el in mol.species]
    coords = mol.cart_coords
    n_atoms = len(species)

    pairs = set(bcp_pairs(qtaim_rec))
    n_ncp = sum(1 for k in qtaim_rec if k != "_meta" and "_" not in k)

    cov = {(min(i, j), max(i, j)) for i, j in
           get_bonds_from_coords(species, coords, covalent_factor)}
    cov_degree = {a: 0 for a in range(n_atoms)}
    for i, j in cov:
        cov_degree[i] += 1
        cov_degree[j] += 1

    bcp_degree = {a: 0 for a in range(n_atoms)}
    for i, j in pairs:
        if i in bcp_degree:
            bcp_degree[i] += 1
        if j in bcp_degree:
            bcp_degree[j] += 1

    isolated_bonded = [
        a for a in range(n_atoms) if bcp_degree[a] == 0 and cov_degree[a] > 0
    ]
    missing_cov = sorted(cov - pairs)

    return {
        "n_atoms": n_atoms,
        "n_ncp": n_ncp,
        "n_bcp": len(pairs),
        "n_cov_bonds": len(cov),
        "n_components": n_components(n_atoms, pairs),
        "n_isolated_bonded": len(isolated_bonded),
        "isolated_bonded": " ".join(str(a) for a in isolated_bonded[:12]),
        "n_missing_cov_bonds": len(missing_cov),
        "missing_cov_bonds": " ".join(f"{i}_{j}" for i, j in missing_cov[:12]),
        "ncp_matches_atoms": int(n_ncp == n_atoms),
    }


# Multiwfn's own wavefunction summary; GTFs drive CP-search allocations
HEADER_RE = re.compile(
    r"Atoms:\s*(\d+),\s*Basis functions:\s*(\d+),\s*GTFs:\s*(\d+)"
)
BLOCK_RE = re.compile(r"^ -{4,}\s+CP\s+\d+", re.M)
# things Multiwfn or the shell print when a run dies badly
ERROR_SIGNATURES = (
    "Error", "error", "insufficient memory", "Insufficient", "allocat",
    "forrtl", "SIGSEGV", "Killed", "killed", "cannot", "Warning: Unable",
)
COMPLETION_MARKER = "have been outputted to CPprop.txt"




def read_from_zip_or_disk(folder: str, name: str) -> Optional[str]:
    for rel in (name, os.path.join("generator", name)):
        path = os.path.join(folder, rel)
        if os.path.isfile(path) and os.path.getsize(path) > 0:
            with open(path, "r", errors="replace") as f:
                return f.read()
    zip_path = os.path.join(folder, "generator", "out_files.zip")
    if os.path.isfile(zip_path):
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                if name in zf.namelist():
                    return zf.read(name).decode("utf-8", errors="replace")
        except (zipfile.BadZipFile, OSError, KeyError):
            return None
    return None


def zip_inventory(folder: str) -> str:
    zip_path = os.path.join(folder, "generator", "out_files.zip")
    if not os.path.isfile(zip_path):
        return "NO_ZIP"
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            names = zf.namelist()
        return f"{len(names)} entries" + ("" if "CPprop.txt" in names else " NO_CPPROP")
    except (zipfile.BadZipFile, OSError):
        return "BAD_ZIP"


def _fmt_time(value):
    try:
        return f"{float(value):.0f}s"
    except (TypeError, ValueError):
        return "?"


def diagnose(folder: str, cause: str) -> dict:
    from qtaim_gen.source.core.parse_qtaim import get_qtaim_descs, only_atom_cps

    out = {"folder": folder, "cause": cause, "zip": zip_inventory(folder)}

    qout = read_from_zip_or_disk(folder, "qtaim.out")
    # The "Atoms / Basis functions / GTFs" line is printed when Multiwfn *loads*
    # the wavefunction, which happens in the convert step -- qtaim.out usually
    # does not carry it, so search the other step logs too.
    for src in ("qtaim.out", "convert.out", "orca.out"):
        text = qout if src == "qtaim.out" else read_from_zip_or_disk(folder, src)
        if not text:
            continue
        m = HEADER_RE.search(text)
        if m:
            out["n_atoms_mwfn"] = int(m.group(1))
            out["n_basis"] = int(m.group(2))
            out["n_gtf"] = int(m.group(3))
            out["header_from"] = src
            break
    if qout:
        out["qtaim_out_bytes"] = len(qout)
        out["export_marker"] = COMPLETION_MARKER in qout
        tail = qout.rstrip().splitlines()[-1:] or [""]
        out["qtaim_out_last_line"] = tail[0].strip()[:90]
        hits = sorted({sig for sig in ERROR_SIGNATURES if sig in qout})
        out["error_signatures"] = " ".join(hits)[:90]
    else:
        out["qtaim_out_bytes"] = 0

    if qout:
        found = re.findall(r"Number of \(3,-1\) CPs:\s*(\d+)", qout)
        out["reported_bcp"] = int(found[-1]) if found else None
        out["cpprop_is_loose"] = os.path.isfile(os.path.join(folder, "CPprop.txt")) or (
            os.path.isfile(os.path.join(folder, "generator", "CPprop.txt"))
        )

    cpprop = read_from_zip_or_disk(folder, "CPprop.txt")
    if cpprop:
        out["cpprop_bytes"] = len(cpprop)
        out["n_cp_blocks"] = len(BLOCK_RE.findall(cpprop))
        out["n_bcp_blocks"] = len(re.findall(r"Type \(3,-1\)", cpprop))
        out["n_nuclear_blocks"] = len(re.findall(r"Type \(3,-3\)", cpprop))
        out["n_connected_lines"] = len(re.findall(r"Connected atoms:", cpprop))
        # A cleanly finished file ends with a complete final line. Stopping
        # mid-line means the process died while writing -- the SIGKILL/OOM
        # signature we are looking for.
        out["ends_with_newline"] = cpprop.endswith("\n")
        last = cpprop.rstrip().splitlines()[-1:] or [""]
        out["cpprop_last_line"] = last[0].strip()[:90]
        # a complete CP block ends on a property line, not a header
        out["ends_mid_block"] = bool(
            BLOCK_RE.search(cpprop[-400:] if len(cpprop) > 400 else cpprop)
        )
    else:
        out["cpprop_bytes"] = 0

    if cause.startswith("duplicate_pair") and cpprop:
        try:
            tmp = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), ".cpprop_tmp"
            )
            with open(tmp, "w") as f:
                f.write(cpprop)
            _atoms, bonds = only_atom_cps(get_qtaim_descs(tmp))
            os.remove(tmp)
            by_pair = collections.defaultdict(list)
            for v in bonds.values():
                if v.get("connected_bond_paths"):
                    by_pair[tuple(sorted(v["connected_bond_paths"]))].append(
                        round(float(v.get("density_all", 0.0)), 5)
                    )
            collisions = {p: r for p, r in by_pair.items() if len(r) > 1}
            out["collision_detail"] = "; ".join(
                f"{a}-{b}: rho={sorted(rhos, reverse=True)}"
                for (a, b), rhos in list(collisions.items())[:4]
            )[:180]
        except Exception as e:
            out["collision_detail"] = f"{type(e).__name__}: {e}"[:60]

    for base in (folder, os.path.join(folder, "generator")):
        tpath = os.path.join(base, "timings.json")
        if os.path.isfile(tpath):
            try:
                with open(tpath) as f:
                    out["qtaim_time_s"] = json.load(f).get("qtaim")
            except (ValueError, OSError):
                pass
            break
    return out


def find_cpprop(folder: str) -> Optional[str]:
    """Path to a readable CPprop.txt, extracting from the zip if needed.

    Returns a path the caller should treat as read-only; when extracted from the
    archive it lands in a temp dir the caller need not clean up eagerly.
    """
    for rel in ("CPprop.txt", os.path.join("generator", "CPprop.txt")):
        path = os.path.join(folder, rel)
        if os.path.isfile(path) and os.path.getsize(path) > 0:
            return path
    zip_path = os.path.join(folder, "generator", "out_files.zip")
    if os.path.isfile(zip_path):
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                if "CPprop.txt" in zf.namelist():
                    tmp = tempfile.mkdtemp(prefix="cpprop_")
                    zf.extract("CPprop.txt", tmp)
                    return os.path.join(tmp, "CPprop.txt")
        except (zipfile.BadZipFile, OSError, KeyError):
            pass
    return None


def explain_shortfall(folder: str) -> dict:
    """Attribute a folder's bond-CP shortfall to the causes above."""
    from qtaim_gen.source.core.parse_qtaim import get_qtaim_descs, only_atom_cps
    from qtaim_gen.source.utils.validation import qtaim_run_status

    out = {"folder": folder, "cause": "", "detail": ""}

    qpath = None
    for base in (folder, os.path.join(folder, "generator")):
        cand = os.path.join(base, "qtaim.json")
        if os.path.isfile(cand):
            qpath = cand
            break
    if qpath is None:
        out["cause"] = "no_qtaim_json"
        return out
    with open(qpath) as f:
        stored = json.load(f)
    n_stored = sum(1 for k in stored if k != "_meta" and "_" in k)
    out["n_bcp_stored"] = n_stored

    status = qtaim_run_status(folder)
    out["reported_bcp"] = status["reported_bcp"]
    out["export_done"] = status["export_done"]

    cpprop = find_cpprop(folder)
    if cpprop is None:
        out["cause"] = "no_cpprop"
        out["detail"] = "CPprop.txt not retained; cannot attribute"
        return out

    descs = get_qtaim_descs(cpprop)
    _atoms, bonds = only_atom_cps(descs)
    out["n_bcp_blocks"] = len(bonds)

    with_paths = {k: v for k, v in bonds.items() if v.get("connected_bond_paths")}
    out["n_no_bond_path"] = len(bonds) - len(with_paths)

    pairs = [tuple(sorted(v["connected_bond_paths"])) for v in with_paths.values()]
    counts = collections.Counter(pairs)
    dups = {p: c for p, c in counts.items() if c > 1}
    out["n_duplicate_pairs"] = sum(c - 1 for c in dups.values())
    out["duplicate_pairs"] = " ".join(f"{a}-{b}" for a, b in list(dups)[:6])
    out["n_unique_pairs"] = len(counts)

    reported = status["reported_bcp"]
    if reported is not None and len(bonds) < reported:
        out["cause"] = "truncated"
        out["detail"] = (
            f"CPprop.txt holds {len(bonds)} (3,-1) blocks vs {reported} reported"
        )
        return out

    # CPprop.txt is complete; the shortfall came from the merge
    if out["n_duplicate_pairs"] and out["n_no_bond_path"]:
        out["cause"] = "duplicate_pair+no_bond_path"
    elif out["n_duplicate_pairs"]:
        out["cause"] = "duplicate_pair"
    elif out["n_no_bond_path"]:
        out["cause"] = "no_bond_path"
    elif n_stored < len(bonds):
        out["cause"] = "unmatched_attractor"
    else:
        out["cause"] = "explained_none"
    out["detail"] = (
        f"{len(bonds)} blocks -> {out['n_unique_pairs']} unique pairs -> "
        f"{n_stored} stored"
    )
    return out


JOB_INPUT_NAMES = ("orca.inp", "input.in", "orca.in", "input.inp")


def find_job_folders(root: str, max_depth: int = 8, require_qtaim: bool = True):
    """Yield job folders under root, i.e. those holding a qtaim.json.

    The OMol4M hierarchy is jagged, so this walks rather than globbing a fixed
    depth. Checks the folder root and generator/ (post-cleanup layout).

    With require_qtaim=False, a folder holding an ORCA input but no qtaim.json
    is yielded too. Keying discovery on qtaim.json alone makes a job whose
    QTAIM step never ran indistinguishable from a job that was never submitted:
    it simply does not appear, so the audit reports the surviving folders as
    clean. Those absences are the population that has to be rerun for the
    dataset to be uniform, so they need to be enumerable.
    """
    root = os.path.abspath(root)
    base_depth = root.rstrip(os.sep).count(os.sep)
    for dirpath, dirnames, filenames in os.walk(root):
        if dirpath.count(os.sep) - base_depth >= max_depth:
            dirnames[:] = []
        if "qtaim.json" in filenames:
            yield dirpath
            dirnames[:] = [d for d in dirnames if d != "generator"]
        elif "generator" in dirnames and os.path.isfile(
            os.path.join(dirpath, "generator", "qtaim.json")
        ):
            yield dirpath
            dirnames[:] = [d for d in dirnames if d != "generator"]
        elif not require_qtaim and (
            any(name in filenames for name in JOB_INPUT_NAMES)
            or (
                "generator" in dirnames
                and any(
                    os.path.isfile(os.path.join(dirpath, "generator", name))
                    for name in JOB_INPUT_NAMES
                )
            )
        ):
            yield dirpath
            dirnames[:] = [d for d in dirnames if d != "generator"]


def audit_folder(folder: str, covalent_factor: float) -> dict:
    """Audit one job folder on disk (no LMDBs needed).

    On a production folder this also gets the stronger signal: Multiwfn's own
    reported (3,-1) count from qtaim.out, which survives inside
    generator/out_files.zip after cleanup.
    """
    import json as _json

    from qtaim_gen.source.core.parse_qtaim import dft_inp_to_dict
    from qtaim_gen.source.utils.validation import qtaim_run_status

    qtaim_rec = None
    for base in (folder, os.path.join(folder, "generator")):
        qpath = os.path.join(base, "qtaim.json")
        if os.path.isfile(qpath) and os.path.getsize(qpath) > 0:
            with open(qpath) as f:
                qtaim_rec = _json.load(f)
            break

    inp = None
    for cand in JOB_INPUT_NAMES:
        for base in (folder, os.path.join(folder, "generator")):
            p = os.path.join(base, cand)
            if os.path.isfile(p):
                inp = p
                break
        if inp:
            break
    if inp is None:
        raise FileNotFoundError("no ORCA input found for geometry")

    dft = dft_inp_to_dict(inp)
    species = [dft[k]["element"] for k in sorted(dft)]
    coords = [dft[k]["pos"] for k in sorted(dft)]

    class _Mol:
        pass

    mol = _Mol()
    mol.species = species
    mol.cart_coords = coords
    if qtaim_rec is None:
        # No QTAIM record at all. Everything downstream keys off n_atoms, so
        # report the geometry and leave the CP columns blank rather than zero:
        # a zero here would read as "searched, found nothing".
        row = {
            "n_atoms": len(species),
            "n_ncp": None,
            "n_bcp": None,
            "n_cov_bonds": None,
            "n_components": None,
            "n_isolated_bonded": None,
            "isolated_bonded": "",
            "n_missing_cov_bonds": None,
            "missing_cov_bonds": "",
            "ncp_matches_atoms": 0,
        }
    else:
        row = audit_record({"molecule": mol}, qtaim_rec, covalent_factor)
    row["have_qtaim_json"] = qtaim_rec is not None

    status = qtaim_run_status(folder)
    reported = status["reported_bcp"]
    row["have_qtaim_out"] = status["have_qtaim_out"]
    row["reported_bcp"] = reported
    # A matching count does not prove completeness: the run can die during the
    # search (no count at all) or during the CPprop.txt export (count present,
    # partial file). Carry both markers so those are not read as clean.
    row["search_done"] = status["search_done"]
    row["export_done"] = status["export_done"]
    row["bcp_shortfall"] = (
        reported - row["n_bcp"]
        if reported is not None and row["n_bcp"] is not None
        else None
    )

    # Leading hypothesis for lost CPs is the QTAIM step being killed mid-write
    # (walltime/OOM), which would leave a partial CPprop.txt that still parses.
    # Carrying the step's wall time lets that be tested directly: affected jobs
    # should cluster at long qtaim times, or pile up against a ceiling.
    row["qtaim_time_s"] = None
    row["total_time_s"] = None
    for base in (folder, os.path.join(folder, "generator")):
        tpath = os.path.join(base, "timings.json")
        if os.path.isfile(tpath) and os.path.getsize(tpath) > 0:
            try:
                with open(tpath) as f:
                    timings = _json.load(f)
            except (ValueError, OSError):
                continue
            if isinstance(timings, dict):
                row["qtaim_time_s"] = timings.get("qtaim")
                numeric = [
                    v for v in timings.values() if isinstance(v, (int, float)) and v > 0
                ]
                row["total_time_s"] = round(sum(numeric), 2) if numeric else None
            break
    return row


def _run_folder_mode(args) -> int:
    import concurrent.futures
    import csv as _csv

    roots = args.folder_root.split(",") if "," in args.folder_root else [args.folder_root]
    if args.verticals:
        roots = [os.path.join(args.folder_root, v) for v in args.verticals]

    folders = []
    for r in roots:
        if not os.path.isdir(r):
            print(f"skip missing root: {r}", file=sys.stderr)
            continue
        found = list(
            find_job_folders(
                r,
                args.max_depth,
                require_qtaim=not getattr(args, "include_missing_qtaim", False),
            )
        )
        print(f"{os.path.basename(r.rstrip(os.sep)):<24} job folders: {len(found)}", flush=True)
        folders.extend(found)
        if args.limit and len(folders) >= args.limit:
            folders = folders[: args.limit]
            break
    if not folders:
        print("no job folders with qtaim.json found", file=sys.stderr)
        return 1

    def one(folder):
        try:
            row = audit_folder(folder, args.covalent_factor)
        except Exception as e:
            row = {"n_isolated_bonded": -1, "error": f"{type(e).__name__}: {e}"[:120]}
        row["vertical"] = os.path.relpath(folder, args.folder_root).split(os.sep)[0]
        row["key"] = os.path.basename(folder)
        row["folder"] = folder
        return row

    rows = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        for i, row in enumerate(ex.map(one, folders)):
            rows.append(row)
            if (i + 1) % 2000 == 0:
                print(f"  {i + 1}/{len(folders)} audited", flush=True)

    fields = [
        "vertical", "key", "folder", "n_atoms", "n_ncp", "n_bcp", "reported_bcp",
        "bcp_shortfall", "have_qtaim_json", "have_qtaim_out",
        "search_done", "export_done", "qtaim_time_s", "total_time_s",
        "n_cov_bonds", "n_components", "n_isolated_bonded",
        "isolated_bonded", "n_missing_cov_bonds", "missing_cov_bonds",
        "ncp_matches_atoms", "error",
    ]
    with open(args.out_csv, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    # rows with no qtaim.json carry a geometry but no CP columns; keep them out
    # of the per-CP statistics rather than letting None compare against 0
    missing_json = [r for r in rows if r.get("have_qtaim_json") is False]
    no_prov_rows = [r for r in rows if r.get("have_qtaim_out") is False]
    ok = [r for r in rows if r.get("n_atoms") and r.get("n_bcp") is not None]
    errs = [r for r in rows if r.get("error")]
    empty = [r for r in ok if r.get("n_bcp") == 0]
    short = [r for r in ok if (r.get("bcp_shortfall") or 0) > 0]
    iso = [r for r in ok if r.get("n_isolated_bonded", 0) > 0]
    severe = [
        r for r in iso
        if r["n_atoms"] and r["n_isolated_bonded"] / r["n_atoms"] > 0.10
    ]
    n = len(ok) or 1
    with_prov = [r for r in ok if r.get("reported_bcp") is not None]
    print(f"\n{len(rows)} folders audited -> {args.out_csv}  ({len(errs)} errors)")
    if getattr(args, "include_missing_qtaim", False):
        print(
            f"  no qtaim.json at all:          {len(missing_json):>7} "
            "<- QTAIM never ran or its output was lost; needs a full rerun"
        )
    print(
        f"  no qtaim.out (no provenance):  {len(no_prov_rows):>7} "
        "<- completeness is UNVERIFIABLE, not verified"
    )
    print(
        f"  qtaim.out provenance available: {len(with_prov):>7} "
        f"({100*len(with_prov)/n:.1f}%)"
    )
    if not with_prov:
        print(
            "  !! WARNING: no folder had qtaim.out (checked root, generator/, and\n"
            "     generator/out_files.zip), so the BCP-shortfall check below is\n"
            "     VACUOUS -- a 0% result means 'not checked', not 'clean'. Only the\n"
            "     connectivity signals are meaningful for this run."
        )
    elif len(with_prov) < 0.5 * n:
        print(
            f"  !! WARNING: only {100*len(with_prov)/n:.1f}% of folders had qtaim.out, so the "
            "shortfall rate\n     below is computed over that subset only."
        )
    m = len(with_prov) or 1
    incomplete = [
        r for r in ok
        if r.get("search_done") is False or r.get("export_done") is False
    ]
    print(f"  empty BCP set:                 {len(empty):>7} ({100*len(empty)/n:.3f}%)")
    print(
        f"  qtaim.out shows an unfinished run: {len(incomplete):>4} "
        f"({100*len(incomplete)/m:.3f}% of those with provenance)  <- proof of "
        f"an incomplete write"
    )
    print(
        f"  fewer BCPs than Multiwfn said: {len(short):>7} "
        f"({100*len(short)/m:.3f}% of those with provenance)  <- strongest signal"
    )
    print(f"  severe (>10% atoms isolated):  {len(severe):>7} ({100*len(severe)/n:.3f}%)")
    print(f"  any isolated bonded atom:      {len(iso):>7} ({100*len(iso)/n:.3f}%)")
    unamb = (
        {id(r) for r in empty}
        | {id(r) for r in short}
        | {id(r) for r in severe}
        | {id(r) for r in incomplete}
    )
    print(f"  UNAMBIGUOUS DEFECTS:           {len(unamb):>7} ({100*len(unamb)/n:.3f}%)")

    import statistics as _st

    def _aff(r):
        return (r.get("bcp_shortfall") or 0) > 0

    timed = [r for r in with_prov if isinstance(r.get("qtaim_time_s"), (int, float))]
    if timed:
        bad = [r["qtaim_time_s"] for r in timed if _aff(r)]
        good = [r["qtaim_time_s"] for r in timed if not _aff(r)]
        if bad and good:
            print(
                f"\n  qtaim step wall time (s), median: "
                f"affected={_st.median(bad):.1f}  unaffected={_st.median(good):.1f}"
            )
            print(
                f"    max affected={max(bad):.1f}  max unaffected={max(good):.1f}"
            )
            if max(good) >= max(bad):
                print(
                    "    note: the longest job is UNAFFECTED, so there is no runtime\n"
                    "    ceiling -- this argues against a walltime kill."
                )

    # Runtime and size are confounded: larger systems take longer AND have more
    # CPs to lose. Bin by size, then compare times within a bin, to see which
    # actually drives the loss.
    sized = [r for r in with_prov if isinstance(r.get("n_atoms"), int)]
    if sized:
        print("\n  shortfall rate by system size (controls for the size/time confound):")
        print(f"    {'atoms':<12}{'n':>7}{'affected':>10}{'rate':>9}"
              f"{'med t aff':>11}{'med t un':>10}{'med BCPs':>10}")
        for lo, hi in ((0, 50), (50, 100), (100, 200), (200, 400), (400, 10**6)):
            sub = [r for r in sized if lo <= r["n_atoms"] < hi]
            if not sub:
                continue
            a = [r for r in sub if _aff(r)]
            u = [r for r in sub if not _aff(r)]
            ta = [r["qtaim_time_s"] for r in a
                  if isinstance(r.get("qtaim_time_s"), (int, float))]
            tu = [r["qtaim_time_s"] for r in u
                  if isinstance(r.get("qtaim_time_s"), (int, float))]
            bcps = [r["reported_bcp"] for r in sub
                    if isinstance(r.get("reported_bcp"), int)]
            label = f"{lo}-{hi}" if hi < 10**6 else f"{lo}+"
            print(
                f"    {label:<12}{len(sub):>7}{len(a):>10}{100*len(a)/len(sub):>8.2f}%"
                f"{(_st.median(ta) if ta else float('nan')):>11.1f}"
                f"{(_st.median(tu) if tu else float('nan')):>10.1f}"
                f"{(_st.median(bcps) if bcps else float('nan')):>10.1f}"
            )
        print(
            "    if the rate rises with size but times match within a bin, size (memory\n"
            "    or write volume) is the driver, not elapsed time."
        )
    if errs:
        import collections as _c
        print("\n  error kinds:", dict(_c.Counter(
            r["error"].split(":")[0] for r in errs).most_common(5)))
    return 0


def _run_explain_mode(args) -> int:
    """Attribute a set of defects to a cause, with the forensic evidence.

    Folds together what used to be two sequential tools: attribution needed
    CPprop.txt and the forensics needed attribution's output, so running them
    as one pass removes an intermediate CSV and a second walk of the same
    folders.
    """
    import statistics as _st

    folders, source = [], args.from_csv
    if source:
        with open(source) as f:
            reader = list(csv.DictReader(f))
        if reader and "verdict" in reader[0]:
            folders = [r["folder"] for r in reader if r["verdict"] in args.verdicts]
        elif reader and "cause" in reader[0]:
            folders = [r["folder"] for r in reader]
        else:
            folders = [r["folder"] for r in reader if r.get("folder")]
    if args.folders:
        folders += list(args.folders)
    if not folders:
        print("no folders to explain", file=sys.stderr)
        return 2

    print(f"explaining {len(folders)} folders\n")
    rows = []
    for folder in folders:
        row = explain_shortfall(folder)
        row.update(
            {k: v for k, v in diagnose(folder, row["cause"]).items() if k != "folder"}
        )
        rows.append(row)

    control = []
    if args.compare_csv:
        with open(args.compare_csv) as f:
            fixed = [
                r["folder"] for r in csv.DictReader(f) if r.get("verdict") == "fixed"
            ]
        control = [diagnose(f, "fixed_control") for f in fixed]

    fields = [
        "cause", "detail", "n_bcp_stored", "reported_bcp", "n_bcp_blocks",
        "n_unique_pairs", "n_no_bond_path", "n_duplicate_pairs", "duplicate_pairs",
        "n_connected_lines", "n_nuclear_blocks", "n_atoms_mwfn", "n_basis", "n_gtf",
        "header_from", "qtaim_time_s", "cpprop_bytes", "cpprop_is_loose",
        "ends_with_newline", "ends_mid_block", "cpprop_last_line", "export_done",
        "export_marker", "qtaim_out_last_line", "error_signatures", "zip",
        "collision_detail", "folder",
    ]
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows + control)

    causes = collections.Counter(r["cause"] for r in rows)
    for cause, n in causes.most_common():
        print(f"  {cause:<28}{n:>5}")
    repairable = causes["truncated"]
    deterministic = sum(
        n for c, n in causes.items()
        if c in ("duplicate_pair", "no_bond_path", "duplicate_pair+no_bond_path",
                 "unmatched_attractor")
    )
    print(f"\n  repairable by rerunning (truncated): {repairable}")
    print(f"  deterministic merge behaviour:        {deterministic}")
    if causes["no_cpprop"]:
        print(f"  unattributable (no CPprop.txt):       {causes['no_cpprop']}")

    with_file = [r for r in rows if r.get("cpprop_bytes")]
    mid = [
        r for r in with_file
        if r.get("ends_mid_block") or r.get("ends_with_newline") is False
    ]
    if with_file:
        print(
            f"\n  CPprop.txt write integrity: {len(mid)} of {len(with_file)} stop "
            "mid-line or mid-block."
        )
        if not mid:
            print(
                "  Every file ends cleanly, so none was killed while writing -- that\n"
                "  rules out OOM truncation for this set, and a count shortfall against\n"
                "  a complete file is a reporting/parse mismatch, not lost output."
            )
    no_file = [r for r in rows if not r.get("cpprop_bytes")]
    if no_file:
        print(
            f"\n  {len(no_file)} job(s) produced no CPprop.txt at all; with no export\n"
            "  marker either, these are the only genuine killed-run candidates here."
        )

    trunc = [r for r in rows if r["cause"] == "truncated" and r.get("n_gtf")]
    ctrl = [r for r in control if r.get("n_gtf")]
    if trunc and ctrl:
        print("\n  memory hypothesis: truncated vs jobs that completed")
        for label, key in (("GTFs", "n_gtf"), ("basis fns", "n_basis"),
                           ("atoms", "n_atoms_mwfn")):
            t = [r[key] for r in trunc if r.get(key)]
            c = [r[key] for r in ctrl if r.get(key)]
            if t and c:
                print(
                    f"    {label:<10} truncated median={_st.median(t):>9.0f} "
                    f"max={max(t):>9.0f} | completed median={_st.median(c):>9.0f} "
                    f"max={max(c):>9.0f}"
                )
        over = [r for r in ctrl if r["n_gtf"] >= max(x["n_gtf"] for x in trunc)]
        print(
            f"    {len(over)} completed job(s) are at least as large by GTFs as the "
            "largest truncated one"
        )
        print(
            "    -> a hard memory ceiling requires zero such jobs; any of them points\n"
            "       at run-time concurrency instead."
            if over else
            "    -> every truncated job is larger than every completed one, which is\n"
            "       what a memory ceiling looks like."
        )
    elif trunc:
        print("\n  memory hypothesis: pass --compare_csv for a control group")

    print(f"\n  -> {args.out_csv}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    import lmdb

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lmdb_root",
        default=None,
        help="audit from per-vertical LMDBs ({root}/{vertical}/{qtaim,structure}.lmdb)",
    )
    parser.add_argument(
        "--folder_root",
        default=None,
        help=(
            "audit raw job-folder trees instead (what production HPC layouts "
            "look like). Also cross-checks Multiwfn's reported BCP count from "
            "qtaim.out, which the LMDB mode cannot see."
        ),
    )
    parser.add_argument(
        "--workers", type=int, default=8, help="folder-mode parallelism"
    )
    parser.add_argument("--max_depth", type=int, default=8)
    parser.add_argument(
        "--verticals",
        nargs="+",
        default=None,
        help="default: every subdirectory holding both qtaim.lmdb and structure.lmdb",
    )
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--covalent_factor", type=float, default=1.3)
    parser.add_argument(
        "--mode",
        choices=("audit", "explain"),
        default="audit",
        help="audit: scan for defects. explain: attribute known defects to a "
        "cause, with forensics",
    )
    parser.add_argument(
        "--from_csv",
        default=None,
        help="explain mode: a verify-qtaim-rerun or prior explain CSV",
    )
    parser.add_argument(
        "--folders", nargs="+", default=None, help="explain mode: folders directly"
    )
    parser.add_argument(
        "--verdicts",
        nargs="+",
        default=["unchanged_still_broken", "changed_still_broken"],
        help="explain mode: which verify verdicts to explain",
    )
    parser.add_argument(
        "--compare_csv",
        default=None,
        help="explain mode: 'fixed' jobs from a verify CSV become the size "
        "control group for the memory comparison",
    )
    parser.add_argument(
        "--include_missing_qtaim",
        action="store_true",
        help=(
            "folder mode: also audit job folders that hold no qtaim.json at all "
            "(discovered by their ORCA input instead). Those are invisible to "
            "the default walk, so a vertical whose QTAIM step never ran looks "
            "clean rather than absent. Reported with have_qtaim_json=False."
        ),
    )
    parser.add_argument("--limit", type=int, default=None, help="records per vertical")
    args = parser.parse_args(argv)

    if args.mode == "explain":
        return _run_explain_mode(args)

    if not args.lmdb_root and not args.folder_root:
        print("give --lmdb_root or --folder_root", file=sys.stderr)
        return 2

    if args.folder_root:
        return _run_folder_mode(args)

    verticals = args.verticals
    if verticals is None:
        verticals = sorted(
            d
            for d in os.listdir(args.lmdb_root)
            if os.path.isfile(os.path.join(args.lmdb_root, d, "qtaim.lmdb"))
            and os.path.isfile(os.path.join(args.lmdb_root, d, "structure.lmdb"))
        )
    if not verticals:
        print("no verticals with both qtaim.lmdb and structure.lmdb", file=sys.stderr)
        return 2

    rows = []
    for vert in verticals:
        qpath = os.path.join(args.lmdb_root, vert, "qtaim.lmdb")
        spath = os.path.join(args.lmdb_root, vert, "structure.lmdb")
        qenv = lmdb.open(qpath, readonly=True, lock=False, subdir=False)
        senv = lmdb.open(spath, readonly=True, lock=False, subdir=False)
        n_seen = n_flag = 0
        try:
            with qenv.begin() as qt, senv.begin() as st:
                for k, v in qt.cursor():
                    key = k.decode()
                    if key in ("length", "__len__"):
                        continue
                    raw_s = st.get(k)
                    if raw_s is None:
                        continue
                    try:
                        row = audit_record(
                            pickle.loads(raw_s), pickle.loads(v), args.covalent_factor
                        )
                    except Exception as e:
                        row = {
                            "n_atoms": None,
                            "error": f"{type(e).__name__}: {e}"[:120],
                            "n_isolated_bonded": -1,
                        }
                    row["vertical"] = vert
                    row["key"] = key
                    rows.append(row)
                    n_seen += 1
                    if row.get("n_isolated_bonded", 0) > 0:
                        n_flag += 1
                    if args.limit and n_seen >= args.limit:
                        break
        finally:
            qenv.close()
            senv.close()
        pct = 100 * n_flag / n_seen if n_seen else 0.0
        print(f"{vert:<22} records={n_seen:>7} flagged={n_flag:>5} ({pct:.2f}%)", flush=True)

    if not rows:
        print("no records audited", file=sys.stderr)
        return 1

    fields = [
        "vertical", "key", "n_atoms", "n_ncp", "n_bcp", "n_cov_bonds",
        "n_components", "n_isolated_bonded", "isolated_bonded",
        "n_missing_cov_bonds", "missing_cov_bonds", "ncp_matches_atoms", "error",
    ]
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    flagged = [r for r in rows if r.get("n_isolated_bonded", 0) > 0]
    passed_validation = [r for r in flagged if r.get("ncp_matches_atoms") == 1]
    print(f"\n{len(rows)} records -> {args.out_csv}")
    print(
        f"flagged (an atom is covalently bonded but has no BCP): {len(flagged)} "
        f"({100 * len(flagged) / len(rows):.2f}%)"
    )
    print(
        f"  of those, passing the current nuclear-CP validator: "
        f"{len(passed_validation)} - i.e. invisible to validate_qtaim_dict"
    )
    worst = sorted(flagged, key=lambda r: -r.get("n_isolated_bonded", 0))[:10]
    if worst:
        print("\nworst records:")
        for r in worst:
            print(
                f"  {r['vertical']}/{r['key'][:44]:<44} "
                f"atoms={r['n_atoms']} bcp={r['n_bcp']} cov={r['n_cov_bonds']} "
                f"isolated={r['n_isolated_bonded']} [{r['isolated_bonded']}]"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
