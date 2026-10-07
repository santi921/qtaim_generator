#!/usr/bin/env python3
"""Relabel qtaim.json records whose nuclear CPs were filed under the wrong atom, without a rerun.

Before the exact-index pass in find_cp_map, nuclear CPs were matched to atoms by distance, so a
close neighbour of the same element (in practice H-H pairs 0.4-1.0 A apart) could claim the other
atom's CP. The stored values are right; only the keys are wrong: nuclear key k holds atom j's CP,
and every bond key built through that map names j where it means k. Relabeling both reproduced a
fresh rerun to <= 1.7e-10 relative on every shared key (LRC pilot, 2026-10-07).

Per folder, under the runners' .processing.lock (never broken as stale; no lock with --dry_run):
  1. every qtaim.json copy present (generator/ and a leftover root copy) is checked against the
     geometry input (results folder first, then the input folder)
  2. a nuclear CP k is moved when its pos_ang is > 0.1 A from atom k; its target j is the nearest
     atom, which must be within 0.05 A, have the same element, and match the CP's own Multiwfn atom
     label (number == j + 1) when the record carries one
  3. write only when the moves form a clean exchange (a permutation of the moved keys); then
     nuclear key k -> pi(k) and bond key i_j -> sorted(pi^-1(i), pi^-1(j)) (the old map sent real
     atom r to stored index pi(r))
Anything else (a CP sitting on an atom whose own CP did not move, a different element, no
geometry) is left for a QTAIM rerun. A relabeled record is clean, so a second pass reports clean.

Statuses: relabeled, would_relabel (--dry_run), clean (nothing to do), not_clean, no_inp,
no_qtaim_json, missing (no folder), locked, failed. Each result also counts self pairs (bond keys
i_i, stale CPs from an earlier run that a relabel cannot remove).
--list_remaining writes the entries (as given) that need a QTAIM rerun or a look: not_clean, no_inp,
no_qtaim_json, missing, locked, failed, and any folder left with self pairs.

Example:
    relabel-qtaim-cps --folder_list scan/tm_react_cp_mislabeled.txt --workers 32 --dry_run \\
        --report qtaim/tm_react_relabel_plan.json --list_remaining qtaim/tm_react_relabel_rerun.txt
"""

import argparse
import json
import logging
import math
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm

STATUS_RELABELED = "relabeled"
STATUS_WOULD_RELABEL = "would_relabel"
STATUS_CLEAN = "clean"
STATUS_NOT_CLEAN = "not_clean"
STATUS_NO_INP = "no_inp"
STATUS_NO_QTAIM_JSON = "no_qtaim_json"
STATUS_MISSING = "missing"
STATUS_LOCKED = "locked"
STATUS_FAILED = "failed"
STATUSES = (STATUS_RELABELED, STATUS_WOULD_RELABEL, STATUS_CLEAN, STATUS_NOT_CLEAN, STATUS_NO_INP,
            STATUS_NO_QTAIM_JSON, STATUS_MISSING, STATUS_LOCKED, STATUS_FAILED)
DONE = (STATUS_RELABELED, STATUS_WOULD_RELABEL, STATUS_CLEAN)

MOVED_A = 0.1
ON_ATOM_A = 0.05
QTAIM_COPIES = (os.path.join("generator", "qtaim.json"), "qtaim.json")


def _paths(entry: str, root_inputs: Optional[str], root_results: Optional[str]) -> Tuple[str, Optional[str]]:
    """(results folder, input folder or None) for an entry given as either path."""
    if root_inputs and root_results:
        if entry.startswith(root_inputs):
            return os.path.join(root_results, entry[len(root_inputs):].lstrip(os.sep)), entry
        if entry.startswith(root_results):
            return entry, os.path.join(root_inputs, entry[len(root_results):].lstrip(os.sep))
    return entry, None


def _geometry(bases: List[Optional[str]]) -> Optional[Dict[int, Tuple[str, List[float]]]]:
    from qtaim_gen.source.utils.validation import get_charge_spin_n_atoms_from_folder

    for base in bases:
        if base and os.path.isdir(base):
            parsed = get_charge_spin_n_atoms_from_folder(base)
            if parsed:
                return {int(i): (a["element"], a["pos"]) for i, a in parsed["mol"].items()}
    return None


def permutation(record: dict, atoms: Dict[int, Tuple[str, List[float]]]) -> Optional[Dict[int, int]]:
    """{stored atom index: atom the CP sits on} for the moved nuclear CPs; {} if none moved;
    None if the moves are not a clean exchange."""
    pi = {}
    for key, cp in record.items():
        if "_" in key or not key.isdigit() or not isinstance(cp, dict) or not cp.get("pos_ang"):
            continue
        if not all(isinstance(x, (int, float)) and math.isfinite(x) for x in cp["pos_ang"]):
            return None
        k = int(key)
        if k not in atoms or math.dist(cp["pos_ang"], atoms[k][1]) <= MOVED_A:
            continue
        j = min(atoms, key=lambda n: math.dist(cp["pos_ang"], atoms[n][1]))
        if math.dist(cp["pos_ang"], atoms[j][1]) > ON_ATOM_A or atoms[j][0] != atoms[k][0]:
            return None
        if str(cp.get("number", "")).isdigit() and int(cp["number"]) != j + 1:
            return None
        pi[k] = j
    if set(pi.values()) != set(pi):
        return None
    return pi


def relabel(record: dict, pi: Dict[int, int]) -> dict:
    # Nuclear key k holds the CP of atom pi(k). A bond key was built through the same wrong map
    # (merge_qtaim_inds: real atom r -> stored index pi(r)), so stored index s means atom pi^-1(s).
    # A swap is its own inverse; a 3-cycle is not.
    inv = {v: k for k, v in pi.items()}
    out = {}
    for key, cp in record.items():
        if "_" not in key:
            new = str(pi.get(int(key), int(key))) if key.isdigit() else key
        else:
            parts = key.split("_")
            if len(parts) == 2 and all(p.isdigit() for p in parts):
                i, j = sorted(inv.get(int(p), int(p)) for p in parts)
                new = f"{i}_{j}"
            else:
                new = key
        out[new] = cp
    return out


def self_pairs(record: dict) -> int:
    return sum(1 for k in record if "_" in k and len(set(k.split("_"))) == 1)


def _plan(folder: str, inputs: Optional[str]) -> Tuple[str, Dict[str, dict], Dict[str, dict], int]:
    """(status, {path: relabeled record} to write, {relpath: permutation}, self pairs left)."""
    records = {}
    for rel in QTAIM_COPIES:
        path = os.path.join(folder, rel)
        if os.path.isfile(path):
            with open(path) as f:
                records[path] = json.load(f)
    if not records:
        return STATUS_NO_QTAIM_JSON, {}, {}, 0
    atoms = _geometry([folder, inputs])
    if not atoms:
        return STATUS_NO_INP, {}, {}, 0
    fixed, perms = {}, {}
    for path, record in records.items():
        pi = permutation(record, atoms)
        if pi is None:
            return STATUS_NOT_CLEAN, {}, {}, max(self_pairs(r) for r in records.values())
        if pi:
            fixed[path] = relabel(record, pi)
            perms[os.path.relpath(path, folder)] = {str(k): v for k, v in sorted(pi.items())}
    left = max(self_pairs(fixed.get(p, r)) for p, r in records.items())
    return (STATUS_RELABELED if fixed else STATUS_CLEAN), fixed, perms, left


def process_folder(entry: str, root_inputs: Optional[str], root_results: Optional[str],
                   dry_run: bool) -> Dict[str, object]:
    from qtaim_gen.source.core.workflow import acquire_lock, release_lock
    from qtaim_gen.source.utils.atomic_write import atomic_json_write

    folder, inputs = _paths(entry, root_inputs, root_results)
    result: Dict[str, object] = {"entry": entry, "folder": folder, "status": "", "permutation": {},
                                 "self_pairs": 0, "error": ""}
    if not os.path.isdir(folder):
        result["status"] = STATUS_MISSING
        return result
    locked = False
    try:
        if not dry_run:
            # a runner's lock is never broken: a stalled heavy job can hold it for days
            if not acquire_lock(folder, max_age_s=float("inf")):
                result["status"] = STATUS_LOCKED
                return result
            locked = True
        status, fixed, perms, left = _plan(folder, inputs)
        result["permutation"], result["self_pairs"] = perms, left
        if fixed:
            if dry_run:
                status = STATUS_WOULD_RELABEL
            else:
                for path, record in fixed.items():
                    atomic_json_write(path, record)
        result["status"] = status
    except Exception as e:
        result["status"] = STATUS_FAILED
        result["error"] = f"{type(e).__name__}: {e}"
    finally:
        if locked:
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
    ap.add_argument("--dry_run", action="store_true", help="Classify only, write nothing, take no lock.")
    ap.add_argument("--report", default=None, help="Write a JSON report of per-folder results.")
    ap.add_argument("--list_remaining", default=None,
                    help="Write the entries that need a QTAIM rerun or a look (every status except "
                         + ", ".join(DONE) + ", plus folders left with self pairs).")
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
    print(f"Relabeling qtaim.json CPs in {n} folders with {workers} worker(s); dry_run={args.dry_run}",
          file=sys.stderr)

    kwargs = dict(root_inputs=args.root_omol_inputs, root_results=args.root_omol_results, dry_run=args.dry_run)
    results: List[Dict[str, object]] = []
    t0 = time.time()
    with tqdm(total=n, desc="Relabeling", unit="folder", file=sys.stderr, mininterval=2.0) as bar:
        if workers <= 1:
            for e in entries:
                results.append(process_folder(e, **kwargs))
                bar.update(1)
        else:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futs = {pool.submit(process_folder, e, **kwargs): e for e in entries}
                for fut in as_completed(futs):
                    try:
                        results.append(fut.result())
                    except Exception as err:
                        # a dead worker must not cost the report of everything already relabeled
                        results.append({"entry": futs[fut], "folder": futs[fut], "status": STATUS_FAILED,
                                        "permutation": {}, "self_pairs": 0,
                                        "error": f"{type(err).__name__}: {err}"})
                    bar.update(1)

    agg: Dict[str, object] = {"folders_total": n}
    for s in STATUSES:
        agg[s] = sum(1 for r in results if r["status"] == s)
    agg["with_self_pairs"] = sum(1 for r in results if r["self_pairs"])
    agg["elapsed_sec"] = round(time.time() - t0, 2)
    print("\nRelabel summary:", file=sys.stderr)
    for k, v in agg.items():
        print(f"  {k:24s} {v}", file=sys.stderr)
    for r in [r for r in results if r["status"] == STATUS_FAILED][:10]:
        print(f"  FAILED {r['folder']}: {r['error']}", file=sys.stderr)

    if args.report:
        os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
        with open(args.report, "w") as f:
            json.dump({"aggregate": agg, "dry_run": args.dry_run,
                       "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "per_folder": results},
                      f, indent=2, default=str)
        print(f"\nReport written: {args.report}", file=sys.stderr)
    if args.list_remaining:
        os.makedirs(os.path.dirname(args.list_remaining) or ".", exist_ok=True)
        remaining = [r["entry"] for r in results if r["status"] not in DONE or r["self_pairs"]]
        with open(args.list_remaining, "w") as f:
            f.write("\n".join(remaining) + ("\n" if remaining else ""))
        print(f"{len(remaining)} folders need a QTAIM rerun or a look: {args.list_remaining}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
