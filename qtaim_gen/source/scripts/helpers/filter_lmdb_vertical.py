#!/usr/bin/env python3
"""Filter an LMDB vertical by removing records with invalid data.

For each data type in the input directory the validator classifies every
record. Keys whose status falls in --exclude-statuses are collected.  The
union of bad keys across all types is then excluded from every output LMDB so
all output files share the same consistent key set.

Optionally, keys present in structure.lmdb but absent from any descriptor
LMDB (silent cross-drops) are also excluded via --exclude-cross-drop.

If >50 %% of records would be excluded the script aborts unless --force is
given.  Use --dry-run to preview what would be removed without writing.

Outputs (written to --output dir):
  filter_report.json    machine-readable: every dropped key, per-type status
  filter_report.md      human-readable summary table + dropped key list

Example:
    lmdb-filter-vertical \\
        --input  data/OMol4M_lmdbs/electrolytes_reactivity/ \\
        --output data/OMol4M_lmdbs/electrolytes_reactivity_filtered/
"""

import argparse
import json
import os
import pickle
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional, Set, Tuple

import lmdb

from qtaim_gen.source.utils.lmdb_audit import (
    DEFAULT_DATA_TYPES,
    DEFAULT_EXCLUDE_STATUSES,
    collect_bad_keys,
)


def _infer_data_types(input_dir: str) -> List[str]:
    return [dt for dt in DEFAULT_DATA_TYPES if os.path.exists(os.path.join(input_dir, f"{dt}.lmdb"))]


def _read_all_keys(lmdb_path: str) -> Set[str]:
    keys: Set[str] = set()
    try:
        env = lmdb.open(lmdb_path, readonly=True, lock=False, subdir=False, readahead=False, meminit=False)
        with env.begin() as txn:
            cursor = txn.cursor()
            for k, _ in cursor:
                ks = k.decode("ascii", errors="replace")
                if ks != "length":
                    keys.add(ks)
        env.close()
    except Exception:
        pass
    return keys


def collect_cross_drop_keys(input_dir: str, data_types: List[str]) -> Set[str]:
    """Return keys in structure.lmdb that are missing from any descriptor LMDB."""
    struct_path = os.path.join(input_dir, "structure.lmdb")
    if not os.path.exists(struct_path):
        return set()
    structure_keys = _read_all_keys(struct_path)
    if not structure_keys:
        return set()
    dropped: Set[str] = set()
    for dt in data_types:
        if dt == "structure":
            continue
        path = os.path.join(input_dir, f"{dt}.lmdb")
        if not os.path.exists(path):
            continue
        dropped |= structure_keys - _read_all_keys(path)
    return dropped


def filter_one_lmdb(
    input_path: str,
    output_path: str,
    keys_to_exclude: Set[str],
) -> Tuple[int, int]:
    """Copy input_path to output_path, skipping keys_to_exclude.

    Returns (kept, excluded) counts.
    """
    env_in = lmdb.open(input_path, readonly=True, lock=False, subdir=False, readahead=False, meminit=False)
    # Use 2x the source map_size: the source may be nearly full and a fresh
    # LMDB needs extra overhead pages. LMDB sparse-allocates so disk usage
    # only grows to match actual data written.
    map_size = env_in.info()["map_size"] * 2
    env_out = lmdb.open(output_path, map_size=map_size, subdir=False, meminit=False)

    kept = 0
    excluded = 0

    with env_in.begin() as txn_in, env_out.begin(write=True) as txn_out:
        cursor = txn_in.cursor()
        for k, v in cursor:
            key_str = k.decode("ascii", errors="replace")
            if key_str == "length":
                continue
            if key_str in keys_to_exclude:
                excluded += 1
            else:
                txn_out.put(k, v)
                kept += 1

    with env_out.begin(write=True) as txn_out:
        txn_out.put(b"length", pickle.dumps(kept))

    env_in.close()
    env_out.close()
    return kept, excluded


def _collect_bad_for_type(args_tuple):
    path, dt, exclude_statuses = args_tuple
    return dt, collect_bad_keys(path, dt, exclude_statuses)


def _build_dropped_index(
    bad_by_type: Dict[str, Dict[str, str]],
    cross_drop_keys: Set[str],
) -> Dict[str, Dict[str, str]]:
    """Build {key: {data_type: status}} for every key that will be dropped."""
    dropped: Dict[str, Dict[str, str]] = defaultdict(dict)
    for dt, key_status in bad_by_type.items():
        for key, status in key_status.items():
            dropped[key][dt] = status
    for key in cross_drop_keys:
        dropped[key].setdefault("cross_drop", "missing_from_descriptor_lmdb")
    return dict(dropped)


def write_filter_report(
    output_dir: str,
    input_dir: str,
    data_types: List[str],
    total_input: int,
    total_kept: int,
    dropped_index: Dict[str, Dict[str, str]],
    dry_run: bool = False,
) -> None:
    """Write filter_report.json and filter_report.md to output_dir."""
    os.makedirs(output_dir, exist_ok=True)

    # Tally by status across all types
    status_counts: Dict[str, int] = defaultdict(int)
    for key_reasons in dropped_index.values():
        for status in key_reasons.values():
            status_counts[status] += 1

    report_json = {
        "input_dir": input_dir,
        "output_dir": output_dir,
        "dry_run": dry_run,
        "data_types_scanned": data_types,
        "total_input": total_input,
        "total_kept": total_kept,
        "total_dropped": len(dropped_index),
        "drop_fraction": round(len(dropped_index) / total_input, 6) if total_input else 0,
        "status_counts": dict(status_counts),
        "dropped_keys": {
            key: reasons
            for key, reasons in sorted(dropped_index.items())
        },
    }

    json_path = os.path.join(output_dir, "filter_report.json")
    with open(json_path, "w") as f:
        json.dump(report_json, f, indent=2)

    md_path = os.path.join(output_dir, "filter_report.md")
    with open(md_path, "w") as f:
        f.write("# LMDB filter report\n\n")
        f.write(f"Input:  `{input_dir}`  \n")
        f.write(f"Output: `{output_dir}`  \n")
        if dry_run:
            f.write("**Dry run - no LMDBs written.**  \n")
        f.write("\n")
        f.write("## Summary\n\n")
        f.write(f"| | count |\n|---|---|\n")
        f.write(f"| total input records | {total_input:,} |\n")
        f.write(f"| records kept | {total_kept:,} |\n")
        f.write(f"| records dropped | {len(dropped_index):,} |\n")
        f.write(f"| drop fraction | {report_json['drop_fraction']:.4%} |\n")
        f.write("\n")

        if status_counts:
            f.write("## Drop reasons\n\n")
            f.write("| status | keys affected |\n|---|---|\n")
            for status, count in sorted(status_counts.items()):
                f.write(f"| {status} | {count:,} |\n")
            f.write("\n")

        if dropped_index:
            f.write("## Dropped keys\n\n")
            # Collect all data types that appear as reasons
            reason_types = sorted({dt for reasons in dropped_index.values() for dt in reasons})
            header = ["key"] + reason_types
            f.write("| " + " | ".join(header) + " |\n")
            f.write("|" + "|".join(["---"] * len(header)) + "|\n")
            for key in sorted(dropped_index):
                reasons = dropped_index[key]
                row = [key] + [reasons.get(dt, "") for dt in reason_types]
                f.write("| " + " | ".join(row) + " |\n")
        else:
            f.write("No records dropped.\n")


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Filter an LMDB vertical by removing invalid records.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--input", required=True, help="Directory containing {type}.lmdb files.")
    p.add_argument("--output", required=True, help="Directory for filtered output LMDBs and reports.")
    p.add_argument(
        "--exclude-statuses",
        nargs="+",
        default=sorted(DEFAULT_EXCLUDE_STATUSES),
        metavar="STATUS",
        help=(
            "Record statuses to treat as failures "
            f"(default: {sorted(DEFAULT_EXCLUDE_STATUSES)}). "
            "Valid choices: ok empty malformed missing_critical no_bonds unpickle_error."
        ),
    )
    p.add_argument(
        "--exclude-cross-drop",
        action="store_true",
        help="Also exclude keys present in structure.lmdb but absent from any descriptor LMDB.",
    )
    p.add_argument(
        "--data-types",
        nargs="+",
        default=None,
        metavar="TYPE",
        help="LMDB types to process (default: all found in --input).",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel workers for scanning (default: 1).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be excluded without writing filtered LMDBs.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Write output even if >50%% of records would be excluded.",
    )
    args = p.parse_args(argv)

    input_dir = os.path.abspath(args.input)
    output_dir = os.path.abspath(args.output)
    exclude_statuses: Set[str] = set(args.exclude_statuses)

    if not os.path.isdir(input_dir):
        print(f"ERROR: input dir does not exist: {input_dir}", file=sys.stderr)
        return 2

    data_types = args.data_types or _infer_data_types(input_dir)
    if not data_types:
        print(f"ERROR: no recognized .lmdb files found in {input_dir}", file=sys.stderr)
        return 2

    print(f"Input:   {input_dir}", file=sys.stderr)
    print(f"Output:  {output_dir}", file=sys.stderr)
    print(f"Types:   {data_types}", file=sys.stderr)
    print(f"Exclude: {sorted(exclude_statuses)}", file=sys.stderr)

    type_paths = [
        (os.path.join(input_dir, f"{dt}.lmdb"), dt, exclude_statuses)
        for dt in data_types
        if os.path.exists(os.path.join(input_dir, f"{dt}.lmdb"))
    ]

    # Scan for bad keys
    bad_by_type: Dict[str, Dict[str, str]] = {}
    workers = min(args.workers, len(type_paths)) if args.workers > 1 else 1

    if workers <= 1:
        for path, dt, excl in type_paths:
            print(f"  scanning {dt}...", file=sys.stderr)
            bad_by_type[dt] = collect_bad_keys(path, dt, excl)
            print(f"    {len(bad_by_type[dt])} bad keys", file=sys.stderr)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(_collect_bad_for_type, t): t[1] for t in type_paths}
            for fut in as_completed(futs):
                dt = futs[fut]
                result_dt, bad = fut.result()
                bad_by_type[result_dt] = bad
                print(f"  {result_dt}: {len(bad)} bad keys", file=sys.stderr)

    cross_drop_keys: Set[str] = set()
    if args.exclude_cross_drop:
        cross_drop_keys = collect_cross_drop_keys(input_dir, data_types)
        if cross_drop_keys:
            print(f"  cross-drop keys: {len(cross_drop_keys)}", file=sys.stderr)

    dropped_index = _build_dropped_index(bad_by_type, cross_drop_keys)
    all_bad: Set[str] = set(dropped_index.keys())

    # Count data records (excluding the internal 'length' key) from structure,
    # falling back to the largest present type if structure is absent/unreadable.
    def _count_data_records(lmdb_path: str) -> int:
        try:
            env = lmdb.open(lmdb_path, readonly=True, lock=False, subdir=False)
            with env.begin() as txn:
                n = txn.stat()["entries"]
                # subtract 1 if a 'length' sentinel key is present
                if txn.get(b"length") is not None:
                    n -= 1
            env.close()
            return n
        except Exception:
            return 0

    total_ref = 0
    struct_path = os.path.join(input_dir, "structure.lmdb")
    if os.path.exists(struct_path):
        total_ref = _count_data_records(struct_path)
    if total_ref == 0:
        for path, dt, _ in type_paths:
            total_ref = max(total_ref, _count_data_records(path))

    frac = len(all_bad) / total_ref if total_ref > 0 else 0.0
    print(f"\nBad keys (union): {len(all_bad):,} / {total_ref:,} ({frac:.1%})", file=sys.stderr)

    if frac > 0.5 and not args.force and not args.dry_run:
        print(
            f"ABORT: {frac:.1%} of records would be excluded. "
            "Re-run with --force to proceed anyway.",
            file=sys.stderr,
        )
        return 1

    total_kept = total_ref - len(all_bad)

    if args.dry_run:
        print("\n[dry-run] Writing report only (no filtered LMDBs).", file=sys.stderr)
        write_filter_report(output_dir, input_dir, data_types, total_ref, total_kept, dropped_index, dry_run=True)
        print(f"Report written to {output_dir}/filter_report.{{json,md}}", file=sys.stderr)
        _print_per_type_summary(bad_by_type, data_types, total_ref)
        return 0

    os.makedirs(output_dir, exist_ok=True)

    for dt in data_types:
        input_path = os.path.join(input_dir, f"{dt}.lmdb")
        if not os.path.exists(input_path):
            continue
        output_path = os.path.join(output_dir, f"{dt}.lmdb")
        print(f"  writing {dt}...", file=sys.stderr)
        kept, excl = filter_one_lmdb(input_path, output_path, all_bad)
        print(f"    kept {kept:,}  excluded {excl:,}", file=sys.stderr)

    write_filter_report(output_dir, input_dir, data_types, total_ref, total_kept, dropped_index)
    print(f"\nFiltered vertical written to {output_dir}", file=sys.stderr)
    print(f"Report: {output_dir}/filter_report.{{json,md}}", file=sys.stderr)
    return 0


def _print_per_type_summary(
    bad_by_type: Dict[str, Dict[str, str]], data_types: List[str], total_ref: int
) -> None:
    print("\nPer-type bad key counts:", file=sys.stderr)
    for dt in data_types:
        n = len(bad_by_type.get(dt, {}))
        pct = f"{100 * n / total_ref:.1f}%" if total_ref > 0 else "n/a"
        print(f"  {dt}: {n:,} ({pct})", file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main())
