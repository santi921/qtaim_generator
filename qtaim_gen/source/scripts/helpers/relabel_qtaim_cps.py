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

With --drop_stale_bond_keys, after any relabel, a stale bond key s (one that does not name the atoms
in its CP's connected_bond_paths, e.g. a self pair 9_9) is dropped when the key c those paths name is
present and the CP at c names c too. A pre-4c79864 key-by-key merge left the older run's CP under the
wrong key next to the later run's CP under the right one, so c holds the value to keep (every stale key
on LLNL and LRC, 2026-10-07, had such a partner). Stale keys without one are kept and listed. Dropped
duplicates no longer pad the bond-CP count, so a folder whose record now falls short of the count
qtaim.out reported (the restart gate's rule under --check_bcp_count) is flagged bcp_short and listed.
Rewritten files keep their modification time: the record is still from the same run, and the runner
keeps a root qtaim.json only when its CPprop.txt is newer than generator/qtaim.json.

Statuses: relabeled, would_relabel (--dry_run), clean (no nuclear CP to move; stale bond keys may
still be dropped, see dropped_bond_keys), not_clean, no_inp, no_qtaim_json, missing (no folder),
locked, failed. Each result also counts the stale bond keys dropped (would be dropped with --dry_run;
the larger count of the two copies) and those left: keys that do not name the atoms in their CP's
connected_bond_paths (self pairs like 9_9, wrong-pair keys kept by a pre-4c79864 key-by-key merge).
--list_remaining writes the entries (as given) that need requalify-qtaim, a QTAIM rerun or a look:
not_clean, no_inp, no_qtaim_json, missing, locked, failed, any folder left with stale bond keys,
and any flagged bcp_short.

Example:
    relabel-qtaim-cps --folder_list scan/tm_react_cp_mislabeled.txt --workers 32 --dry_run \\
        --report qtaim/tm_react_relabel_plan.json --list_remaining qtaim/tm_react_relabel_rerun.txt
    relabel-qtaim-cps --folder_list qtaim/tm_react_relabel_rerun.txt --drop_stale_bond_keys --workers 32 \\
        --report qtaim/tm_react_bond_keys.json --list_remaining qtaim/tm_react_bond_keys_rerun.txt
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

from qtaim_gen.source.utils.validation import CP_MOVED_A as MOVED_A, CP_ON_ATOM_A as ON_ATOM_A

BCP_TOLERANCE = 2  # the runners' --bcp_tolerance default

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
        if k not in atoms:
            return None  # a record from a different geometry
        if math.dist(cp["pos_ang"], atoms[k][1]) <= MOVED_A:
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


def _own_bond_key(cp: dict) -> Optional[str]:
    """The key a bond CP's connected_bond_paths (Multiwfn's 1-based atom numbers) name, or None."""
    paths = cp.get("connected_bond_paths")
    if (isinstance(paths, list) and len(paths) == 2 and all(isinstance(p, int) for p in paths)
            and paths[0] != paths[1]):
        i, j = sorted(p - 1 for p in paths)
        return f"{i}_{j}"
    return None


def drop_stale_bond_keys(record: dict) -> Tuple[dict, int]:
    """(record without its duplicate stale bond keys, number dropped). A stale key s goes when the key c
    its CP's connected_bond_paths name is present and the CP at c names c too: the older run's CP under
    the wrong key goes, the later run's CP under the right key stays."""
    drop = set()
    for key, cp in record.items():
        parts = key.split("_")
        if len(parts) != 2 or not all(x.isdigit() for x in parts) or not isinstance(cp, dict):
            continue
        own = _own_bond_key(cp)
        if own is None or own == key:
            continue
        partner = record.get(own)
        if isinstance(partner, dict) and _own_bond_key(partner) == own:
            drop.add(key)
    if not drop:
        return record, 0
    return {k: v for k, v in record.items() if k not in drop}, len(drop)


def stale_bond_keys(record: dict) -> int:
    """Bond keys that do not name the atoms their CP connects (connected_bond_paths, Multiwfn's 1-based
    atom numbers): self pairs like 9_9 and wrong-pair keys a pre-4c79864 key-by-key merge kept next to
    a correct reparse. A relabel cannot remove them; drop_stale_bond_keys removes those with a partner."""
    n = 0
    for key, cp in record.items():
        if "_" not in key or not isinstance(cp, dict):
            continue
        parts = key.split("_")
        if len(parts) == 2 and len(set(parts)) == 1:
            n += 1
            continue
        paths = cp.get("connected_bond_paths")
        if (isinstance(paths, list) and len(paths) == 2 and all(isinstance(p, int) for p in paths)
                and parts != [str(x) for x in sorted(p - 1 for p in paths)]):
            n += 1
    return n


def _bcp_short(folder: str, records: List[dict]) -> bool:
    """Whether a record has fewer bond CPs than qtaim.out reported, beyond the tolerance: the restart
    gate's rule under --check_bcp_count (_qtaim_output_complete), raw count first, then the storable one."""
    from qtaim_gen.source.utils.validation import qtaim_run_status, storable_bcp_count

    reported = qtaim_run_status(folder)["reported_bcp"]
    if reported is None:
        return False
    n_bcp = min(sum(1 for k in r if k != "_meta" and "_" in k) for r in records)
    if reported - n_bcp <= BCP_TOLERANCE:
        return False
    storable = storable_bcp_count(folder)
    return (storable if storable is not None else reported) - n_bcp > BCP_TOLERANCE


def _plan(folder: str, inputs: Optional[str], drop_bonds: bool = False
          ) -> Tuple[str, Dict[str, dict], Dict[str, dict], int, int, bool]:
    """(status, {path: record} to write, {relpath: permutation}, stale bond keys left, bond keys dropped,
    whether the dropped record falls short of qtaim.out's bond-CP count)."""
    records = {}
    for rel in QTAIM_COPIES:
        path = os.path.join(folder, rel)
        if os.path.isfile(path):
            with open(path) as f:
                records[path] = json.load(f)
    if not records:
        return STATUS_NO_QTAIM_JSON, {}, {}, 0, 0, False
    atoms = _geometry([folder, inputs])
    if not atoms:
        return STATUS_NO_INP, {}, {}, 0, 0, False
    fixed, perms, dropped = {}, {}, 0
    for path, record in records.items():
        pi = permutation(record, atoms)
        if pi is None:
            return STATUS_NOT_CLEAN, {}, {}, max(stale_bond_keys(r) for r in records.values()), 0, False
        new = relabel(record, pi) if pi else record
        n = 0
        if drop_bonds:
            new, n = drop_stale_bond_keys(new)
            dropped = max(dropped, n)
        if pi or n:
            fixed[path] = new
        if pi:
            perms[os.path.relpath(path, folder)] = {str(k): v for k, v in sorted(pi.items())}
    final = [fixed.get(p, r) for p, r in records.items()]
    left = max(stale_bond_keys(r) for r in final)
    short = bool(dropped) and _bcp_short(folder, final)
    return (STATUS_RELABELED if perms else STATUS_CLEAN), fixed, perms, left, dropped, short


def process_folder(entry: str, root_inputs: Optional[str], root_results: Optional[str],
                   dry_run: bool, drop_bonds: bool = False) -> Dict[str, object]:
    from qtaim_gen.source.core.workflow import acquire_lock, release_lock
    from qtaim_gen.source.utils.atomic_write import atomic_json_write

    folder, inputs = _paths(entry, root_inputs, root_results)
    result: Dict[str, object] = {"entry": entry, "folder": folder, "status": "", "permutation": {},
                                 "stale_bond_keys": 0, "dropped_bond_keys": 0, "bcp_short": False,
                                 "error": ""}
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
        status, fixed, perms, left, dropped, short = _plan(folder, inputs, drop_bonds)
        if dry_run and status == STATUS_RELABELED:
            status = STATUS_WOULD_RELABEL
        elif fixed and not dry_run:
            for path, record in fixed.items():
                st = os.stat(path)
                atomic_json_write(path, record)
                # same run: a newer mtime would make a fresh root qtaim.json look like a leftover
                os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns))
        result["permutation"], result["stale_bond_keys"] = perms, left
        result["dropped_bond_keys"], result["bcp_short"] = dropped, short
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
    ap.add_argument("--drop_stale_bond_keys", action="store_true",
                    help="Also drop a stale bond key when the key its CP's connected_bond_paths name is "
                         "present and the CP there names that same key (the older run's duplicate); the "
                         "rest stay and are listed.")
    ap.add_argument("--report", default=None, help="Write a JSON report of per-folder results.")
    ap.add_argument("--list_remaining", default=None,
                    help="Write the entries that need a QTAIM rerun or a look (every status except "
                         + ", ".join(DONE) + ", plus folders left with stale bond keys or bcp_short).")
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

    kwargs = dict(root_inputs=args.root_omol_inputs, root_results=args.root_omol_results, dry_run=args.dry_run,
                  drop_bonds=args.drop_stale_bond_keys)
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
                                        "permutation": {}, "stale_bond_keys": 0, "dropped_bond_keys": 0,
                                        "bcp_short": False, "error": f"{type(err).__name__}: {err}"})
                    bar.update(1)

    agg: Dict[str, object] = {"folders_total": n}
    for s in STATUSES:
        agg[s] = sum(1 for r in results if r["status"] == s)
    agg["with_stale_bond_keys"] = sum(1 for r in results if r["stale_bond_keys"])
    agg["with_dropped_bond_keys"] = sum(1 for r in results if r["dropped_bond_keys"])
    agg["dropped_bond_keys"] = sum(r["dropped_bond_keys"] for r in results)
    agg["bcp_short"] = sum(1 for r in results if r["bcp_short"])
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
        remaining = [r["entry"] for r in results
                     if r["status"] not in DONE or r["stale_bond_keys"] or r["bcp_short"]]
        with open(args.list_remaining, "w") as f:
            f.write("\n".join(remaining) + ("\n" if remaining else ""))
        print(f"{len(remaining)} folders need a QTAIM rerun or a look: {args.list_remaining}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
