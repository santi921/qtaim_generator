"""Audit per-vertical descriptor LMDB splits against the source LMDBs.

Walks every vertical under --splits_dir, compares per-descriptor key
counts in the split LMDBs against the source LMDB at --lmdb_root, and
categorizes each vertical into one of:

  HEALTHY        every descriptor's (train+val+test+excluded) == source
  DRIFT          one or more descriptors have count mismatch
  NEVER_SPLIT    split_assignment.json missing under <splits_dir>/<vertical>
  SOURCE_BAD     source LMDB unopenable (e.g. numpy version mismatch)

Reports the rerun list (DRIFT + NEVER_SPLIT) at the end, suitable for
piping straight back into split_descriptor_lmdbs --include_verticals.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import lmdb

DEFAULT_LMDB_ROOT = Path(
    "/p/lustre5/vargas58/converters/converters_final"
)
DEFAULT_SPLITS_DIR = Path(
    "/p/lustre5/vargas58/converters/splits"
)
DESCRIPTORS = (
    "structure", "charge", "bond", "qtaim", "fuzzy",
    "other", "orca", "timings",
)
NON_VERTICAL_DIRS = {
    "tm_bond_lists", "ln_bond_lists", "filter_csv_for_holdouts",
    "holdout_lmdbs", "holdouts", "splits", "splits_merged",
}


def find_lmdb(root: Path, vertical: str, descriptor: str) -> Optional[Path]:
    direct = root / vertical / f"{descriptor}.lmdb"
    if direct.exists():
        return direct
    merged = root / vertical / "merged" / f"{descriptor}.lmdb"
    if merged.exists():
        return merged
    return None


def count_keys(path: Path) -> Optional[int]:
    """Return live key count, or None if the LMDB cannot be opened or read."""
    try:
        env = lmdb.open(
            str(path), subdir=False, readonly=True, lock=False,
            readahead=False, meminit=False,
        )
    except Exception:
        return None
    try:
        with env.begin() as txn:
            return sum(1 for k, _ in txn.cursor() if k != b"length")
    except Exception:
        return None
    finally:
        env.close()


def discover_verticals(root: Path) -> list[str]:
    return sorted(
        c.name for c in root.iterdir()
        if c.is_dir() and c.name not in NON_VERTICAL_DIRS
    )


def audit_vertical(
    vertical: str, lmdb_root: Path, splits_dir: Path,
) -> dict:
    """Return a structured audit dict for one vertical."""
    sa_path = splits_dir / vertical / "split_assignment.json"
    if not sa_path.exists():
        return {
            "vertical": vertical,
            "status": "NEVER_SPLIT",
            "reason": "no split_assignment.json",
            "descriptors": {},
        }

    try:
        sa = json.load(open(sa_path))
    except Exception as e:
        return {
            "vertical": vertical,
            "status": "DRIFT",
            "reason": f"split_assignment.json unparseable: {e}",
            "descriptors": {},
        }

    n_excluded = sa.get("n_excluded", 0)
    per_desc: dict[str, dict] = {}
    descriptor_status = "HEALTHY"
    source_ok = True

    for desc in DESCRIPTORS:
        src_path = find_lmdb(lmdb_root, vertical, desc)
        src_n = count_keys(src_path) if src_path else None
        if src_path and src_n is None:
            source_ok = False

        tr = count_keys(splits_dir / vertical / "train" / f"{desc}.lmdb")
        va = count_keys(splits_dir / vertical / "val" / f"{desc}.lmdb")
        te = count_keys(splits_dir / vertical / "test" / f"{desc}.lmdb")
        split_sum = (tr or 0) + (va or 0) + (te or 0)

        # Expected = source - excluded (excluded structures aren't copied
        # into any split LMDB).
        expected = (src_n - n_excluded) if src_n is not None else None
        match = (expected is not None and split_sum == expected)
        if not match and expected is not None:
            descriptor_status = "DRIFT"

        per_desc[desc] = {
            "source": src_n,
            "train": tr,
            "val": va,
            "test": te,
            "sum": split_sum,
            "expected": expected,
            "match": match,
        }

    if not source_ok and descriptor_status == "HEALTHY":
        # Sources unreadable but split_assignment.json claims a sane plan;
        # we can't tell if the splits are right.
        return {
            "vertical": vertical,
            "status": "SOURCE_BAD",
            "reason": "source LMDB(s) unreadable; cannot verify",
            "n_excluded": n_excluded,
            "descriptors": per_desc,
        }

    return {
        "vertical": vertical,
        "status": descriptor_status,
        "n_excluded": n_excluded,
        "n_total": sa.get("n_total"),
        "per_split_counts": sa.get("per_split_counts"),
        "descriptors": per_desc,
    }


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--lmdb_root", type=Path, default=DEFAULT_LMDB_ROOT,
                   help=f"Source per-vertical LMDB root. Default: {DEFAULT_LMDB_ROOT}")
    p.add_argument("--splits_dir", type=Path, default=DEFAULT_SPLITS_DIR,
                   help=f"Output of split_descriptor_lmdbs. Default: {DEFAULT_SPLITS_DIR}")
    p.add_argument("--include_verticals", nargs="*", default=None,
                   help="Audit only these verticals (default: all under --lmdb_root)")
    p.add_argument("--quiet", action="store_true",
                   help="Suppress per-descriptor breakdown for healthy verticals.")
    p.add_argument("--write_report", type=Path, default=None,
                   help="Write a JSON audit report to this path.")
    args = p.parse_args(argv)

    print(f"lmdb_root:  {args.lmdb_root}")
    print(f"splits_dir: {args.splits_dir}")

    verticals = discover_verticals(args.lmdb_root)
    if args.include_verticals:
        keep = set(args.include_verticals)
        verticals = [v for v in verticals if v in keep]
    print(f"\nauditing {len(verticals)} verticals\n")

    results: list[dict] = []
    for v in verticals:
        r = audit_vertical(v, args.lmdb_root, args.splits_dir)
        results.append(r)

        if r["status"] == "HEALTHY" and args.quiet:
            print(f"[{v:<26}] HEALTHY  n_total={r['n_total']:,} "
                  f"n_excluded={r['n_excluded']:,}")
            continue

        print(f"[{v:<26}] {r['status']}", end="")
        if "reason" in r:
            print(f"  ({r['reason']})", end="")
        if "n_excluded" in r:
            print(f"  excluded={r['n_excluded']}", end="")
        print()
        if r["status"] in ("DRIFT", "SOURCE_BAD"):
            print(f"  {'desc':<11} {'source':>10} {'train':>10} {'val':>9} "
                  f"{'test':>9} {'sum':>10} {'expected':>10} {'match':>6}")
            for desc, d in r["descriptors"].items():
                src = d["source"]
                exp = d["expected"]
                m = "OK" if d["match"] else "DRIFT"
                print(f"  {desc:<11} {str(src):>10} {str(d['train']):>10} "
                      f"{str(d['val']):>9} {str(d['test']):>9} "
                      f"{d['sum']:>10,} {str(exp):>10} {m:>6}")
        elif not args.quiet and r["status"] == "HEALTHY":
            for desc, d in r["descriptors"].items():
                print(f"  {desc:<11} src={d['source']!s:>8} sum={d['sum']:>8,} OK")

    # Categorize
    by_status: dict[str, list[str]] = {}
    for r in results:
        by_status.setdefault(r["status"], []).append(r["vertical"])

    print("\n=== summary ===")
    for status in ("HEALTHY", "DRIFT", "NEVER_SPLIT", "SOURCE_BAD"):
        names = by_status.get(status, [])
        print(f"{status:<14} ({len(names)}): {names}")

    rerun = sorted(set(by_status.get("DRIFT", []) + by_status.get("NEVER_SPLIT", [])))
    print(f"\n=== to rerun ({len(rerun)}) ===")
    if rerun:
        print(" ".join(rerun))
        print(f"\n# Suggested xargs invocation:")
        print(f"printf '%s\\n' " + " ".join(rerun) + " | xargs -n 1 -P 8 -I {} \\")
        print(f"    python -m qtaim_gen.source.scripts.helpers.split_descriptor_lmdbs \\")
        print(f"        --lmdb_root {args.lmdb_root} \\")
        print(f"        --output_dir {args.splits_dir} \\")
        print(f"        --include_verticals {{}}")
    else:
        print("(nothing to rerun)")

    if args.write_report:
        args.write_report.parent.mkdir(parents=True, exist_ok=True)
        with open(args.write_report, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nwrote report -> {args.write_report}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
