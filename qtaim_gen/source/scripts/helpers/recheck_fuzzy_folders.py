#!/usr/bin/env python3
"""Apply the fuzzy recheck (#28) to folders that need no Multiwfn, in place.

For the sweep's ``reparse_only`` class the fix is a rewrite of fuzzy_full.json /
bond.json from archived Multiwfn output, or a rebuild of hirsh_fuzzy_density from
the stored Hirshfeld charges. The runner can do that too, but only after copying
and decompressing each folder's wavefunction sources; this does it directly.

Per folder, under the runners' .processing.lock:
  1. plan the recheck (read-only); a folder whose plan reruns any step is left
     untouched and reported ``needs_multiwfn`` (it belongs on the runner list)
  2. apply the reparses (recheck_fuzzy never invalidates anything when there is
     nothing to rerun, and removes no wavefunction)
  3. re-validate (validation_checks with --recheck_fuzzy and, if given, --check_orca)

Statuses: clean (nothing to fix), fixed (applied, folder now validates),
still_invalid (applied, but validation fails for another reason), needs_multiwfn,
would_fix (--dry_run), no_mult (multiplicity unreadable), locked, failed.
--list_remaining writes the entries (as given in --folder_list) that still need
the runner: needs_multiwfn, still_invalid, no_mult, locked, failed.

Examples:
    recheck-fuzzy --folder_list nakb_fuzzy_reparse.txt \\
        --root_omol_inputs /lus/eagle/projects/OMol25/ \\
        --root_omol_results /lus/eagle/projects/generator/OMol25_postprocessing/ \\
        --check_orca --workers 64 --report nakb_recheck.json --list_remaining nakb_recheck_left.txt
"""

import argparse
import json
import logging
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional

from tqdm import tqdm

STATUS_CLEAN = "clean"
STATUS_FIXED = "fixed"
STATUS_STILL_INVALID = "still_invalid"
STATUS_NEEDS_MULTIWFN = "needs_multiwfn"
STATUS_WOULD_FIX = "would_fix"
STATUS_NO_MULT = "no_mult"
STATUS_LOCKED = "locked"
STATUS_FAILED = "failed"
STATUSES = (STATUS_CLEAN, STATUS_FIXED, STATUS_STILL_INVALID, STATUS_NEEDS_MULTIWFN,
            STATUS_WOULD_FIX, STATUS_NO_MULT, STATUS_LOCKED, STATUS_FAILED)
REMAINING = (STATUS_NEEDS_MULTIWFN, STATUS_STILL_INVALID, STATUS_NO_MULT, STATUS_LOCKED, STATUS_FAILED)


def _results_folder(entry: str, root_inputs: Optional[str], root_results: Optional[str]) -> str:
    if root_inputs and root_results and entry.startswith(root_inputs):
        return os.path.join(root_results, entry[len(root_inputs):].lstrip(os.sep))
    return entry


def _multiplicity(folder: str, entry: str) -> Optional[int]:
    from qtaim_gen.source.utils.validation import get_charge_spin_n_atoms_from_folder

    for base in dict.fromkeys((folder, entry)):
        if os.path.isdir(base):
            d = get_charge_spin_n_atoms_from_folder(base)
            if d and d.get("spin") is not None:
                return int(d["spin"])
    return None


def process_folder(
    entry: str,
    root_inputs: Optional[str],
    root_results: Optional[str],
    full_set: int,
    move_results: bool,
    check_orca: bool,
    dry_run: bool,
) -> Dict[str, object]:
    from qtaim_gen.source.core.workflow import acquire_lock, release_lock
    from qtaim_gen.source.utils.fuzzy_recheck import recheck_fuzzy
    from qtaim_gen.source.utils.validation import validation_checks

    folder = _results_folder(entry, root_inputs, root_results)
    result: Dict[str, object] = {"entry": entry, "folder": folder, "status": "", "reparse": [],
                                 "rerun": [], "derived": [], "error": ""}
    try:
        mult = _multiplicity(folder, entry)
        if mult is None:
            result["status"] = STATUS_NO_MULT
            return result
        plan = recheck_fuzzy(folder, mult, dry_run=True)
        result.update(reparse=plan["reparse"], rerun=plan["rerun"], derived=plan["derived"])
        if plan["rerun"]:
            result["status"] = STATUS_NEEDS_MULTIWFN
            return result
        if not plan["reparse"]:
            result["status"] = STATUS_CLEAN
            return result
        if dry_run:
            result["status"] = STATUS_WOULD_FIX
            return result
        if not acquire_lock(folder):
            result["status"] = STATUS_LOCKED
            return result
        try:
            # re-plan under the lock: another job may have fixed or changed the folder
            plan = recheck_fuzzy(folder, mult, dry_run=True)
            if plan["rerun"]:
                result.update(rerun=plan["rerun"], status=STATUS_NEEDS_MULTIWFN)
                return result
            if not plan["reparse"]:
                result["status"] = STATUS_CLEAN
                return result
            recheck_fuzzy(folder, mult, logger=logging.getLogger("recheck_fuzzy"))
            valid = validation_checks(
                folder, full_set=full_set, verbose=False, move_results=move_results,
                check_orca=check_orca, recheck_fuzzy=True,
            )
            result["status"] = STATUS_FIXED if valid else STATUS_STILL_INVALID
        finally:
            release_lock(folder)
    except Exception as e:
        result["status"] = STATUS_FAILED
        result["error"] = f"{type(e).__name__}: {e}"
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--folder_list", required=True,
                    help="One job folder per line (input-tree paths with the two roots, or results paths).")
    ap.add_argument("--root_omol_inputs", default=None)
    ap.add_argument("--root_omol_results", default=None)
    ap.add_argument("--full_set", type=int, default=0, help="Level for the final validation (default 0).")
    ap.add_argument("--no_move_results", dest="move_results", action="store_false",
                    help="Flat layout: json files at the job-folder root (default: generator/).")
    ap.add_argument("--check_orca", action="store_true",
                    help="Also require a current orca.json in the final validation.")
    ap.add_argument("--workers", type=int, default=0, help="Parallel workers (default: cpu_count, 1 disables).")
    ap.add_argument("--ordered", action="store_true", help="List order instead of a random order.")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None, help="Process at most N folders.")
    ap.add_argument("--dry_run", action="store_true", help="Plan only, write nothing, take no lock.")
    ap.add_argument("--report", default=None, help="Write a JSON report of per-folder results.")
    ap.add_argument("--list_remaining", default=None,
                    help="Write the entries that still need the runner (" + ", ".join(REMAINING) + ").")
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
    print(f"Rechecking {n} folders with {workers} worker(s); full_set={args.full_set}; "
          f"check_orca={args.check_orca}; dry_run={args.dry_run}", file=sys.stderr)

    kwargs = dict(root_inputs=args.root_omol_inputs, root_results=args.root_omol_results,
                  full_set=args.full_set, move_results=args.move_results,
                  check_orca=args.check_orca, dry_run=args.dry_run)
    results: List[Dict[str, object]] = []
    t0 = time.time()
    with tqdm(total=n, desc="Rechecking", unit="folder", file=sys.stderr, mininterval=2.0) as bar:
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
    agg["derived_from_charges"] = sum(1 for r in results if r["derived"])
    agg["elapsed_sec"] = round(time.time() - t0, 2)
    print("\nRecheck summary:", file=sys.stderr)
    for k, v in agg.items():
        print(f"  {k:24s} {v}", file=sys.stderr)
    failed = [r for r in results if r["status"] == STATUS_FAILED]
    for r in failed[:10]:
        print(f"  FAILED {r['folder']}: {r['error']}", file=sys.stderr)

    if args.report:
        os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
        with open(args.report, "w") as f:
            json.dump({"aggregate": agg, "per_folder": results}, f, indent=2, default=str)
        print(f"\nReport written: {args.report}", file=sys.stderr)
    if args.list_remaining:
        os.makedirs(os.path.dirname(args.list_remaining) or ".", exist_ok=True)
        remaining = [r["entry"] for r in results if r["status"] in REMAINING]
        with open(args.list_remaining, "w") as f:
            f.write("\n".join(remaining) + ("\n" if remaining else ""))
        print(f"{len(remaining)} folders still need the runner: {args.list_remaining}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
