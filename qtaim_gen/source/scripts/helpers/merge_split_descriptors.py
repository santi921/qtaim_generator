"""Merge per-vertical split descriptor LMDBs into combined train/val/test LMDBs.

Reads the output of `split_descriptor_lmdbs` (per-vertical, per-split,
per-descriptor LMDBs) and concatenates all verticals into a flat layout:

  <output_dir>/train/<descriptor>.lmdb
  <output_dir>/val/<descriptor>.lmdb
  <output_dir>/test/<descriptor>.lmdb

Keys in the merged LMDBs are prefixed with `{vertical}__` to avoid collisions
across verticals. This matches the convention used by pull_holdout_records.

The per-vertical split LMDBs under --splits_dir are read-only and are not
modified. Output is always written to a separate --output_dir.

Re-running overwrites each merged LMDB cleanly (existing files + locks are
removed before re-opening), so partial runs can be retried.

Typical invocation:

  python -m qtaim_gen.source.scripts.helpers.merge_split_descriptors \\
      --splits_dir /p/lustre5/vargas58/converters/splits \\
      --output_dir /p/lustre5/vargas58/converters/splits_merged \\
      --write_report /p/lustre5/vargas58/converters/splits_merged/merge_report.json
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Optional

import lmdb

DEFAULT_SPLITS_DIR = Path(
    "/p/lustre5/vargas58/converters/splits"
)
DEFAULT_OUTPUT_DIR = Path(
    "/p/lustre5/vargas58/converters/splits_merged"
)
DESCRIPTORS = (
    "structure", "charge", "bond", "qtaim", "fuzzy",
    "other", "orca", "timings",
)
SPLIT_NAMES = ("train", "val", "test")
NON_VERTICAL_DIRS = {
    "tm_bond_lists", "ln_bond_lists", "filter_csv_for_holdouts",
    "holdout_lmdbs", "holdouts", "splits", "splits_merged",
}
LMDB_MAP_SIZE = 1099511627776 * 2  # 2 TiB sparse map


def discover_verticals(splits_dir: Path) -> list[str]:
    """Verticals are subdirs that contain a split_assignment.json."""
    out: list[str] = []
    for child in sorted(splits_dir.iterdir()):
        if not child.is_dir() or child.name in NON_VERTICAL_DIRS:
            continue
        if (child / "split_assignment.json").exists():
            out.append(child.name)
    return out


def merge_one_pair(
    split: str,
    descriptor: str,
    verticals: list[str],
    splits_dir: Path,
    output_dir: Path,
) -> dict:
    """Merge one (split, descriptor) pair across all verticals."""
    out_path = output_dir / split / f"{descriptor}.lmdb"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("", "-lock"):
        p = Path(str(out_path) + ext)
        if p.exists():
            p.unlink()

    env_out = lmdb.open(
        str(out_path),
        map_size=LMDB_MAP_SIZE,
        subdir=False,
        meminit=False,
        map_async=True,
    )

    per_vertical: dict[str, int] = {}
    skipped: list[str] = []
    total = 0

    for vertical in verticals:
        src = splits_dir / vertical / split / f"{descriptor}.lmdb"
        if not src.exists():
            skipped.append(vertical)
            continue

        env_in = lmdb.open(
            str(src), subdir=False, readonly=True, lock=False,
            readahead=True, meminit=False,
        )
        n_written = 0
        prefix = f"{vertical}__".encode("ascii")
        with env_in.begin() as txn_in, env_out.begin(write=True) as txn_out:
            for key, value in txn_in.cursor():
                if key == b"length":
                    continue
                txn_out.put(prefix + key, value)
                n_written += 1
        env_in.close()

        per_vertical[vertical] = n_written
        total += n_written

    with env_out.begin(write=True) as txn_out:
        txn_out.put(b"length", pickle.dumps(total, protocol=-1))
    env_out.sync()
    env_out.close()

    return {
        "split": split,
        "descriptor": descriptor,
        "total": total,
        "per_vertical": per_vertical,
        "skipped_verticals": skipped,
        "output": str(out_path),
    }


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--splits_dir", type=Path, default=DEFAULT_SPLITS_DIR,
                   help=f"Per-vertical split LMDB root (output of "
                        f"split_descriptor_lmdbs). Default: {DEFAULT_SPLITS_DIR}")
    p.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                   help=f"Where to write merged LMDBs. Must differ from "
                        f"--splits_dir. Default: {DEFAULT_OUTPUT_DIR}")
    p.add_argument("--descriptors", nargs="+", default=list(DESCRIPTORS),
                   help=f"Descriptor families to merge. "
                        f"Default: {' '.join(DESCRIPTORS)}")
    p.add_argument("--splits", nargs="+", default=list(SPLIT_NAMES),
                   choices=list(SPLIT_NAMES),
                   help=f"Splits to merge. Default: {' '.join(SPLIT_NAMES)}")
    p.add_argument("--include_verticals", nargs="*", default=None,
                   help="Merge only these verticals (default: all under "
                        "--splits_dir with split_assignment.json).")
    p.add_argument("--write_report", type=Path, default=None,
                   help="Write a JSON merge report to this path.")
    args = p.parse_args(argv)

    if args.output_dir.resolve() == args.splits_dir.resolve():
        print("error: --output_dir must differ from --splits_dir "
              "(refuse to merge in place)", file=sys.stderr)
        return 2

    print(f"splits_dir: {args.splits_dir}")
    print(f"output_dir: {args.output_dir}")
    print(f"splits:     {args.splits}")
    print(f"descriptors:{args.descriptors}")

    verticals = discover_verticals(args.splits_dir)
    if args.include_verticals:
        keep = set(args.include_verticals)
        unknown = sorted(keep - set(verticals))
        if unknown:
            print(f"warning: --include_verticals contains {len(unknown)} "
                  f"verticals without split_assignment.json: {unknown}",
                  file=sys.stderr)
        verticals = [v for v in verticals if v in keep]

    print(f"\nmerging {len(verticals)} verticals\n")
    if not verticals:
        print("error: no verticals to merge", file=sys.stderr)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    for split in args.splits:
        for desc in args.descriptors:
            r = merge_one_pair(split, desc, verticals,
                               args.splits_dir, args.output_dir)
            results.append(r)
            skip_note = (f"  skipped={len(r['skipped_verticals'])}"
                         if r["skipped_verticals"] else "")
            print(f"[{split}/{desc:<10}] total={r['total']:>10,}{skip_note}")
            if r["skipped_verticals"]:
                print(f"    missing in: {r['skipped_verticals']}")

    print("\n=== summary ===")
    for split in args.splits:
        for desc in args.descriptors:
            r = next(x for x in results
                     if x["split"] == split and x["descriptor"] == desc)
            print(f"  {split}/{desc:<10} {r['total']:>10,}")

    if args.write_report:
        args.write_report.parent.mkdir(parents=True, exist_ok=True)
        with open(args.write_report, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nwrote report -> {args.write_report}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
