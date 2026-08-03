#!/usr/bin/env python3
"""Backfill ORCA-derived bond orders + charges into existing bond.json / charge.json.

Re-applies ``merge_orca_into_bond_json`` and ``merge_orca_into_charge_json`` to
every job folder under a root (or every folder listed in --folder_list). Both
merges read the per-job ``orca.json`` (parsed ORCA .out output) and copy
ORCA-derived bond orders / charges into the matching descriptor JSON under
``mayer_orca`` / ``loewdin_orca`` / ``mulliken_orca`` / ``hirshfeld_orca`` /
``mbis_orca`` keys.

This patches the silent-skip in ``parse_orca.merge_orca_into_*_json`` for jobs
processed before the 2026-02-22 merge feature landed (or before bond.json
existed at parse time). The merges are idempotent: rerunning is safe.

After backfill, rebuild bond.lmdb / charge.lmdb with ``json-to-lmdb`` to
propagate the merged JSONs into the LMDBs consumed by the converter.

Usage:
    backfill-orca-into-json --root_dir data/OMol4M/droplet/ [--move_files]
    backfill-orca-into-json --folder_list droplet_folders.txt --workers 16
    backfill-orca-into-json --folder_list droplet_folders.txt --dry_run
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from glob import glob
from typing import Dict, List, Optional, Tuple

from qtaim_gen.source.core.parse_orca import (
    merge_orca_into_bond_json,
    merge_orca_into_charge_json,
)


# Source-key in orca.json -> destination-key in target json. Mirrors
# merge_orca_into_bond_json / merge_orca_into_charge_json. A folder needs
# backfill iff its orca.json carries a source key that the target json lacks.
BOND_MERGE_MAP = {
    "mayer_bond_orders": "mayer_orca",
    "loewdin_bond_orders": "loewdin_orca",
}
CHARGE_MERGE_MAP = {
    "mulliken_charges": "mulliken_orca",
    "loewdin_charges": "loewdin_orca",
    "hirshfeld_charges": "hirshfeld_orca",
    "mayer_charges": "mayer_orca",
    "mbis_charges": "mbis_orca",
}


def resolve_paths(folder: str, move_files: bool) -> Dict[str, str]:
    """Resolve the on-disk paths for orca.json / charge.json / bond.json.

    Mirrors _run_orca_parse: prefer generator/ subdir when move_files=True and
    the file is there, fall back to folder root otherwise.
    """
    paths = {}
    for name in ("orca.json", "charge.json", "bond.json"):
        gen_path = os.path.join(folder, "generator", name)
        root_path = os.path.join(folder, name)
        if move_files and os.path.isfile(gen_path):
            paths[name] = gen_path
        elif os.path.isfile(root_path):
            paths[name] = root_path
        elif os.path.isfile(gen_path):
            paths[name] = gen_path
        else:
            paths[name] = ""
    return paths


def needs_backfill(
    json_path: str,
    orca_dict: Dict,
    merge_map: Dict[str, str],
) -> bool:
    """Return True iff orca_dict carries a source key whose merged dest is
    absent from json_path. This is the real "this merge would do something"
    signal: it ignores expected keys whose source data isn't in orca.json
    (e.g. rmechdb has no hirshfeld_charges, so hirshfeld_orca will never
    appear in charge.json regardless of how many times we run the merge).
    """
    if not json_path or not os.path.isfile(json_path):
        return False
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return False
    if not isinstance(data, dict):
        return False
    for src_key, dest_key in merge_map.items():
        if src_key in orca_dict and dest_key not in data:
            return True
    return False


def backfill_folder(folder: str, move_files: bool, dry_run: bool) -> Dict[str, str]:
    """Apply both merges to a single folder. Returns status counters."""
    result = {
        "folder": folder,
        "orca_present": False,
        "charge_present": False,
        "bond_present": False,
        "charge_was_missing_keys": False,
        "bond_was_missing_keys": False,
        "charge_merged": False,
        "bond_merged": False,
        "error": "",
    }

    paths = resolve_paths(folder, move_files)
    orca_path = paths["orca.json"]
    charge_path = paths["charge.json"]
    bond_path = paths["bond.json"]

    result["orca_present"] = bool(orca_path)
    result["charge_present"] = bool(charge_path)
    result["bond_present"] = bool(bond_path)

    if not orca_path:
        return result

    try:
        with open(orca_path, "r") as f:
            orca_dict = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        result["error"] = f"orca.json read: {type(e).__name__}: {e}"
        return result

    result["charge_was_missing_keys"] = needs_backfill(
        charge_path, orca_dict, CHARGE_MERGE_MAP
    )
    result["bond_was_missing_keys"] = needs_backfill(
        bond_path, orca_dict, BOND_MERGE_MAP
    )

    if dry_run:
        return result

    if charge_path and result["charge_was_missing_keys"]:
        try:
            merge_orca_into_charge_json(orca_dict, charge_path)
            result["charge_merged"] = True
        except Exception as e:
            result["error"] = f"charge merge: {type(e).__name__}: {e}"

    if bond_path and result["bond_was_missing_keys"]:
        try:
            merge_orca_into_bond_json(orca_dict, bond_path)
            result["bond_merged"] = True
        except Exception as e:
            err = f"bond merge: {type(e).__name__}: {e}"
            result["error"] = (result["error"] + "; " + err) if result["error"] else err

    return result


def discover_folders(
    root_dir: Optional[str],
    folder_list: Optional[str],
    move_files: bool,
) -> List[str]:
    if folder_list:
        with open(folder_list, "r") as f:
            raw = [line.strip() for line in f]
        candidates = [line for line in raw if line and not line.startswith("#")]
        return [p for p in candidates if os.path.isdir(p)]
    if root_dir:
        # Top-level subdirs that contain either generator/ or any of the json files.
        subs = sorted(
            d for d in glob(os.path.join(root_dir, "*"))
            if os.path.isdir(d)
        )
        out = []
        for d in subs:
            if move_files and os.path.isdir(os.path.join(d, "generator")):
                out.append(d)
                continue
            for name in ("orca.json", "bond.json", "charge.json"):
                if os.path.isfile(os.path.join(d, name)):
                    out.append(d)
                    break
        return out
    raise ValueError("Provide --root_dir or --folder_list")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--root_dir",
        help="Root containing one job folder per subdir (depth-1).",
    )
    src.add_argument(
        "--folder_list",
        help="Text file with one absolute job folder path per line "
             "(blanks and '#' lines skipped).",
    )

    ap.add_argument(
        "--move_files",
        action="store_true",
        default=True,
        help="JSON files live under <folder>/generator/ (default: True; "
             "OMol4M layout). Falls back to folder root if missing.",
    )
    ap.add_argument(
        "--no_move_files",
        dest="move_files",
        action="store_false",
        help="Force flat folder layout (json files at job-folder root).",
    )

    ap.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Parallel workers (default: cpu_count, 1 disables).",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most N folders (debug).",
    )
    ap.add_argument(
        "--dry_run",
        action="store_true",
        help="Report counts of folders needing backfill, do not write.",
    )
    ap.add_argument(
        "--report",
        default=None,
        help="Optional path to write a JSON report of per-folder results.",
    )
    ap.add_argument(
        "--list_remaining",
        default=None,
        help="Optional path to write the list of folders that still need "
             "backfill (post-run; empty if all succeeded).",
    )

    args = ap.parse_args()

    folders = discover_folders(args.root_dir, args.folder_list, args.move_files)
    if args.limit is not None:
        folders = folders[: args.limit]

    n = len(folders)
    if n == 0:
        print("No job folders found.", file=sys.stderr)
        return 1

    workers = args.workers if args.workers > 0 else (os.cpu_count() or 1)
    workers = min(workers, n)

    print(
        f"Scanning {n} folders with {workers} worker(s); "
        f"move_files={args.move_files}; dry_run={args.dry_run}",
        file=sys.stderr,
    )

    results: List[Dict[str, str]] = []
    t0 = time.time()
    progress_every = max(1, n // 20)

    if workers <= 1:
        for i, folder in enumerate(folders):
            results.append(backfill_folder(folder, args.move_files, args.dry_run))
            if (i + 1) % progress_every == 0 or (i + 1) == n:
                rate = (i + 1) / max(time.time() - t0, 1e-6)
                print(
                    f"  {i+1}/{n} ({rate:.1f}/s)",
                    file=sys.stderr,
                )
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futs = {
                pool.submit(backfill_folder, f, args.move_files, args.dry_run): f
                for f in folders
            }
            done = 0
            for fut in as_completed(futs):
                results.append(fut.result())
                done += 1
                if done % progress_every == 0 or done == n:
                    rate = done / max(time.time() - t0, 1e-6)
                    print(f"  {done}/{n} ({rate:.1f}/s)", file=sys.stderr)

    elapsed = time.time() - t0

    # Aggregate
    agg = {
        "folders_total": n,
        "orca_present": sum(1 for r in results if r["orca_present"]),
        "charge_present": sum(1 for r in results if r["charge_present"]),
        "bond_present": sum(1 for r in results if r["bond_present"]),
        "charge_needed_backfill": sum(1 for r in results if r["charge_was_missing_keys"]),
        "bond_needed_backfill": sum(1 for r in results if r["bond_was_missing_keys"]),
        "charge_merged": sum(1 for r in results if r["charge_merged"]),
        "bond_merged": sum(1 for r in results if r["bond_merged"]),
        "errors": sum(1 for r in results if r["error"]),
        "elapsed_sec": round(elapsed, 2),
    }

    print("", file=sys.stderr)
    print("Backfill summary:", file=sys.stderr)
    for k, v in agg.items():
        print(f"  {k:<24} {v}", file=sys.stderr)

    if agg["errors"] > 0:
        print("\nFirst 10 errors:", file=sys.stderr)
        n_shown = 0
        for r in results:
            if not r["error"]:
                continue
            print(f"  {r['folder']}: {r['error']}", file=sys.stderr)
            n_shown += 1
            if n_shown >= 10:
                break

    if args.report:
        os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
        with open(args.report, "w") as f:
            json.dump(
                {"aggregate": agg, "per_folder": results},
                f,
                indent=2,
                default=str,
            )
        print(f"\nReport written: {args.report}", file=sys.stderr)

    if args.list_remaining:
        os.makedirs(os.path.dirname(args.list_remaining) or ".", exist_ok=True)
        remaining = [
            r["folder"] for r in results
            if r["charge_was_missing_keys"] and not r["charge_merged"]
            or r["bond_was_missing_keys"] and not r["bond_merged"]
        ]
        with open(args.list_remaining, "w") as f:
            for p in remaining:
                f.write(p + "\n")
        print(
            f"List of {len(remaining)} folders still needing backfill: "
            f"{args.list_remaining}",
            file=sys.stderr,
        )

    return 0 if agg["errors"] == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
