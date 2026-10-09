#!/usr/bin/env python3
"""Rebuild qtaim.json from the archived CPprop.txt where the stored record drifted from it.

Before 4c79864, move_results_to_folder merged qtaim.json key by key. Its keys are critical
points, so every rerun or reparse kept the CPs the previous record had and the new one did
not: bond CPs that the pre-fix find_cp_map stored under the wrong atom pair (a self pair like
'9_9', or a wrong neighbour) survived next to the corrected key. The archived CPprop.txt of
the run still holds the right answer, so no Multiwfn is needed: re-parse it exactly as
parse_multiwfn does and replace the stored record.

Per folder, under the runners' .processing.lock (no lock with --dry_run):
  1. the run must be verifiable: qtaim.out (folder, generator/ or out_files.zip) shows the
     CP search and the CPprop.txt export finished, and CPprop.txt is available
  2. re-parse CPprop.txt with the geometry input (results folder first, then the input folder)
  3. replace only when the re-parse is sound and comes from the same run as the stored record:
     nuclear CPs == atoms, bond CPs <= Multiwfn's reported count, and every CP the two share
     has the same properties (a stored record from a later, unarchived run is left alone)

Statuses: same (nothing to do), replaced, would_replace (--dry_run), missing (no folder),
no_qtaim_json, no_provenance (no qtaim.out, or the run did not finish), no_cpprop (pre-archiving
run), no_inp, parse_failed, ncp_mismatch, count_mismatch, different_run, locked, failed.
--list_remaining writes the entries (as given) that need a real QTAIM rerun or a look:
everything except same, replaced and would_replace.

Examples:
    requalify-qtaim --folder_list rgd_uks_stale.txt \\
        --root_omol_inputs /global/scratch/users/santiagovargas/OMol4M_raw/ \\
        --root_omol_results /global/scratch/users/santiagovargas/OMol4M/ \\
        --workers 16 --dry_run --report rgd_uks_requalify_plan.json
"""

import argparse
import json
import logging
import math
import os
import random
import sys
import tempfile
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm

STATUS_SAME = "same"
STATUS_REPLACED = "replaced"
STATUS_WOULD_REPLACE = "would_replace"
STATUS_MISSING = "missing"
STATUS_NO_QTAIM_JSON = "no_qtaim_json"
STATUS_NO_PROVENANCE = "no_provenance"
STATUS_NO_CPPROP = "no_cpprop"
STATUS_NO_INP = "no_inp"
STATUS_PARSE_FAILED = "parse_failed"
STATUS_NCP_MISMATCH = "ncp_mismatch"
STATUS_COUNT_MISMATCH = "count_mismatch"
STATUS_DIFFERENT_RUN = "different_run"
STATUS_LOCKED = "locked"
STATUS_FAILED = "failed"
STATUSES = (STATUS_SAME, STATUS_REPLACED, STATUS_WOULD_REPLACE, STATUS_MISSING, STATUS_NO_QTAIM_JSON,
            STATUS_NO_PROVENANCE, STATUS_NO_CPPROP, STATUS_NO_INP, STATUS_PARSE_FAILED,
            STATUS_NCP_MISMATCH, STATUS_COUNT_MISMATCH, STATUS_DIFFERENT_RUN, STATUS_LOCKED,
            STATUS_FAILED)
DONE = (STATUS_SAME, STATUS_REPLACED, STATUS_WOULD_REPLACE)


def _paths(entry: str, root_inputs: Optional[str], root_results: Optional[str]) -> Tuple[str, Optional[str]]:
    """(results folder, input folder or None) for an entry given as either path."""
    if root_inputs and root_results:
        if entry.startswith(root_inputs):
            return os.path.join(root_results, entry[len(root_inputs):].lstrip(os.sep)), entry
        if entry.startswith(root_results):
            return entry, os.path.join(root_inputs, entry[len(root_results):].lstrip(os.sep))
    return entry, None


def _stored_path(folder: str) -> Optional[str]:
    for rel in (os.path.join("generator", "qtaim.json"), "qtaim.json"):
        p = os.path.join(folder, rel)
        if os.path.isfile(p):
            return p
    return None


def _cpprop(folder: str, tmpdir: str) -> Optional[str]:
    for rel in ("CPprop.txt", os.path.join("generator", "CPprop.txt")):
        p = os.path.join(folder, rel)
        if os.path.isfile(p) and os.path.getsize(p) > 0:
            return p
    zip_path = os.path.join(folder, "generator", "out_files.zip")
    try:
        with zipfile.ZipFile(zip_path) as zf:
            if "CPprop.txt" in zf.namelist():
                zf.extract("CPprop.txt", tmpdir)
                return os.path.join(tmpdir, "CPprop.txt")
    except (OSError, zipfile.BadZipFile):
        pass
    return None


def _geometry(bases: List[Optional[str]]) -> Optional[Tuple[str, int]]:
    from qtaim_gen.source.core.parse_qtaim import dft_inp_to_dict
    from qtaim_gen.source.utils.validation import geometry_input_candidates

    for base in bases:
        if not (base and os.path.isdir(base)):
            continue
        for name in geometry_input_candidates(base):
            path = os.path.join(base, name)
            try:
                return path, len(dft_inp_to_dict(path, parse_charge_spin=True)["mol"])
            except Exception:
                continue
    return None


def _close(a, b) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_close(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return all(_close(a[k], b[k]) for k in a.keys() & b.keys())
    return a == b


def _plan(folder: str, inputs: Optional[str]) -> Tuple[str, Optional[dict], str]:
    """(status, re-parsed record to write or None, stored path)."""
    from qtaim_gen.source.core.parse_multiwfn import parse_qtaim
    from qtaim_gen.source.utils.validation import (
        QTAIM_META_KEY, qtaim_run_status, qtaim_topology_meta, read_qtaim_out,
    )

    stored_path = _stored_path(folder)
    if stored_path is None:
        return STATUS_NO_QTAIM_JSON, None, ""
    run = qtaim_run_status(folder)
    if not (run["have_qtaim_out"] and run["search_done"] and run["export_done"]):
        return STATUS_NO_PROVENANCE, None, stored_path
    geom = _geometry([folder, inputs])
    with tempfile.TemporaryDirectory(prefix="requalify_") as tmp:
        cpprop = _cpprop(folder, tmp)
        if cpprop is None:
            return STATUS_NO_CPPROP, None, stored_path
        if geom is None:
            return STATUS_NO_INP, None, stored_path
        inp, n_atoms = geom
        try:
            parsed = json.loads(json.dumps(parse_qtaim(cprop_file=cpprop, inp_loc=inp,
                                                       orca_tf=inp.endswith(".inp"))))
            with open(cpprop, "rb") as f:
                meta = qtaim_topology_meta(f.read(), read_qtaim_out(folder))
        except Exception:
            return STATUS_PARSE_FAILED, None, stored_path
    if not parsed:
        return STATUS_PARSE_FAILED, None, stored_path
    if sum(1 for k in parsed if "_" not in k) != n_atoms:
        return STATUS_NCP_MISMATCH, None, stored_path
    if run["reported_bcp"] is not None and sum(1 for k in parsed if "_" in k) > run["reported_bcp"]:
        return STATUS_COUNT_MISMATCH, None, stored_path
    with open(stored_path) as f:
        stored = json.load(f)
    if not all(_close(parsed[k], stored[k]) for k in parsed.keys() & stored.keys()):
        return STATUS_DIFFERENT_RUN, None, stored_path
    if parsed.keys() == stored.keys() - {QTAIM_META_KEY}:
        return STATUS_SAME, None, stored_path
    parsed[QTAIM_META_KEY] = meta
    return STATUS_REPLACED, parsed, stored_path


def process_folder(entry: str, root_inputs: Optional[str], root_results: Optional[str],
                   dry_run: bool) -> Dict[str, object]:
    from qtaim_gen.source.core.workflow import acquire_lock, release_lock
    from qtaim_gen.source.utils.atomic_write import atomic_json_write

    folder, inputs = _paths(entry, root_inputs, root_results)
    result: Dict[str, object] = {"entry": entry, "folder": folder, "status": "", "only_stored": [],
                                 "only_reparsed": [], "error": ""}
    if not os.path.isdir(folder):
        result["status"] = STATUS_MISSING
        return result
    if not dry_run and not acquire_lock(folder):
        result["status"] = STATUS_LOCKED
        return result
    try:
        status, parsed, stored_path = _plan(folder, inputs)
        if parsed is not None:
            with open(stored_path) as f:
                stored_keys = set(json.load(f)) - {"_meta"}
            result["only_stored"] = sorted(stored_keys - parsed.keys())
            result["only_reparsed"] = sorted(parsed.keys() - stored_keys - {"_meta"})
            if dry_run:
                status = STATUS_WOULD_REPLACE
            else:
                atomic_json_write(stored_path, parsed)
        result["status"] = status
    except Exception as e:
        result["status"] = STATUS_FAILED
        result["error"] = f"{type(e).__name__}: {e}"
    finally:
        if not dry_run:
            release_lock(folder)
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--folder_list", required=True,
                    help="One job folder per line (input-tree paths with the two roots, or results paths).")
    ap.add_argument("--root_omol_inputs", default=None)
    ap.add_argument("--root_omol_results", default=None)
    ap.add_argument("--workers", type=int, default=0, help="Parallel workers (default: cpu_count, 1 disables).")
    ap.add_argument("--ordered", action="store_true", help="List order instead of a random order.")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None, help="Process at most N folders.")
    ap.add_argument("--dry_run", action="store_true", help="Compare only, write nothing, take no lock.")
    ap.add_argument("--report", default=None, help="Write a JSON report of per-folder results.")
    ap.add_argument("--list_remaining", default=None,
                    help="Write the entries that need a QTAIM rerun or a look (every status except "
                         + ", ".join(DONE) + ").")
    args = ap.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    with open(args.folder_list) as f:
        entries = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    if not args.ordered:
        random.Random(args.seed).shuffle(entries)
    if args.limit is not None:
        entries = entries[: args.limit]
    n = len(entries)
    if n == 0:
        print("No folders listed.", file=sys.stderr)
        return 1
    workers = min(args.workers if args.workers > 0 else (os.cpu_count() or 1), n)
    print(f"Requalifying {n} folders with {workers} worker(s); dry_run={args.dry_run}", file=sys.stderr)

    kwargs = dict(root_inputs=args.root_omol_inputs, root_results=args.root_omol_results, dry_run=args.dry_run)
    results: List[Dict[str, object]] = []
    t0 = time.time()
    with tqdm(total=n, desc="Requalifying", unit="folder", file=sys.stderr, mininterval=2.0) as bar:
        if workers <= 1:
            for e in entries:
                results.append(process_folder(e, **kwargs))
                bar.update(1)
        else:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futs = [pool.submit(process_folder, e, **kwargs) for e in entries]
                for fut in as_completed(futs):
                    results.append(fut.result())
                    bar.update(1)

    agg: Dict[str, object] = {"folders_total": n}
    for s in STATUSES:
        agg[s] = sum(1 for r in results if r["status"] == s)
    agg["elapsed_sec"] = round(time.time() - t0, 2)
    print("\nRequalify summary:", file=sys.stderr)
    for k, v in agg.items():
        print(f"  {k:24s} {v}", file=sys.stderr)
    for r in [r for r in results if r["status"] == STATUS_FAILED][:10]:
        print(f"  FAILED {r['folder']}: {r['error']}", file=sys.stderr)

    if args.report:
        os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
        with open(args.report, "w") as f:
            json.dump({"aggregate": agg, "per_folder": results}, f, indent=2, default=str)
        print(f"\nReport written: {args.report}", file=sys.stderr)
    if args.list_remaining:
        os.makedirs(os.path.dirname(args.list_remaining) or ".", exist_ok=True)
        remaining = [r["entry"] for r in results if r["status"] not in DONE]
        with open(args.list_remaining, "w") as f:
            f.write("\n".join(remaining) + ("\n" if remaining else ""))
        print(f"{len(remaining)} folders need a QTAIM rerun or a look: {args.list_remaining}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
