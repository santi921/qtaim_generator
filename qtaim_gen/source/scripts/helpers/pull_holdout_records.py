"""Pull holdout records out of per-vertical descriptor LMDBs into merged
cross-vertical holdout LMDBs.

Reads filter CSVs produced by `qtaim_gen.source.analysis.build_holdout_csvs`
(default location: `data/OMol4M_lmdbs/filter_csv_for_holdouts/INDEX.csv`),
walks each per-vertical descriptor LMDB
(`<lmdb_root>/<vertical>/<descriptor>.lmdb`), copies the matching records
into one LMDB per (holdout, descriptor), and emits a consolidated
`manifest_holdout.parquet` that the train/val/test split step uses to
exclude these keys.

Output layout:
    <output_dir>/
        H1/
            structure.lmdb
            charge.lmdb
            bond.lmdb
            qtaim.lmdb
            fuzzy.lmdb
            other.lmdb
            orca.lmdb
            timings.lmdb
        H3/ ...
        H6/ ...
        H7/ ...
        H8/ ...
        manifest_holdout.parquet
        pull_summary.csv

LMDB key in the merged output is `{vertical}__{rel_path}` so vertical
provenance is recoverable from the key alone, with no value rewriting.
A trailing `length` key holds the per-output count, matching the
convention in `qtaim_gen.source.utils.lmdbs`.

Holdout overlap is allowed: a structure that appears in both H1 and H7
will be written to both H1/*.lmdb and H7/*.lmdb (no dedup across holdouts).

Usage:
    python -m qtaim_gen.source.scripts.helpers.pull_holdout_records \\
        --holdout_dir data/OMol4M_lmdbs/filter_csv_for_holdouts \\
        --lmdb_root   data/OMol4M_lmdbs \\
        --output_dir  data/OMol4M_lmdbs/holdout_lmdbs

Verticals named in a holdout CSV but absent under --lmdb_root are
reported and skipped, not an error: this lets the script run on a dev
machine with a partial LMDB checkout.
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import lmdb
import pandas as pd

DEFAULT_HOLDOUT_DIR = Path(
    "/home/santiagovargas/dev/qtaim_generator/data/OMol4M_lmdbs/filter_csv_for_holdouts"
)
DEFAULT_LMDB_ROOT = Path(
    "/home/santiagovargas/dev/qtaim_generator/data/OMol4M_lmdbs"
)
DEFAULT_OUT_DIR = Path(
    "/home/santiagovargas/dev/qtaim_generator/data/OMol4M_lmdbs/holdout_lmdbs"
)

DEFAULT_DESCRIPTORS = (
    "structure", "charge", "bond", "qtaim", "fuzzy",
    "other", "orca", "timings",
)

# Subdirs that live under lmdb_root but are not per-vertical LMDBs
NON_VERTICAL_DIRS = {
    "tm_bond_lists",
    "ln_bond_lists",
    "filter_csv_for_holdouts",
    "holdout_lmdbs",
}

# Match build_holdout_csvs / utils.lmdbs convention: 2 TiB sparse map
LMDB_MAP_SIZE = 1099511627776 * 2


def discover_verticals(lmdb_root: Path) -> List[str]:
    """List subdirs of lmdb_root that look like per-vertical LMDB folders."""
    out = []
    for child in sorted(lmdb_root.iterdir()):
        if not child.is_dir():
            continue
        if child.name in NON_VERTICAL_DIRS:
            continue
        out.append(child.name)
    return out


def find_lmdb(lmdb_root: Path, vertical: str, descriptor: str) -> Optional[Path]:
    """Return path to <lmdb_root>/<vertical>[/merged]/<descriptor>.lmdb if present."""
    direct = lmdb_root / vertical / f"{descriptor}.lmdb"
    if direct.exists():
        return direct
    merged = lmdb_root / vertical / "merged" / f"{descriptor}.lmdb"
    if merged.exists():
        return merged
    return None


def manifest_rel_path_to_lmdb_key(vertical: str, rel_path: str) -> str:
    """Normalize manifest-style rel_path to the LMDB key.

    Manifest rel_path is `<vertical>/<path with / separators>` (or already
    the LMDB key for CSVs derived from bond extraction). LMDB keys follow
    the jagged-hierarchy convention `relpath(folder, root_dir).replace('/',
    '__')` where root_dir is the per-vertical source directory. Strip a
    leading `<vertical>/` if present, then replace `/` with `__`.
    """
    prefix = f"{vertical}/"
    if rel_path.startswith(prefix):
        rel_path = rel_path[len(prefix):]
    return rel_path.replace("/", "__")


def load_holdout_csvs(holdout_dir: Path) -> Dict[str, pd.DataFrame]:
    """Read INDEX.csv to discover holdout CSVs, return {holdout_id: df}.

    Each returned df has at minimum columns 'vertical' and 'rel_path'.
    """
    idx_path = holdout_dir / "INDEX.csv"
    if not idx_path.exists():
        raise FileNotFoundError(f"INDEX.csv not found in {holdout_dir}")
    idx = pd.read_csv(idx_path)
    holdouts: Dict[str, pd.DataFrame] = {}
    for _, row in idx.iterrows():
        hid = row["holdout_id"]
        fpath = holdout_dir / row["filename"]
        if not fpath.exists():
            print(f"  WARN: {hid} listed in INDEX but {fpath.name} missing, skipping",
                  file=sys.stderr)
            continue
        df = pd.read_csv(fpath, usecols=["vertical", "rel_path"])
        holdouts[hid] = df
    return holdouts


def pull_one(
    holdout_id: str,
    holdout_df: pd.DataFrame,
    lmdb_root: Path,
    descriptors: Iterable[str],
    output_dir: Path,
) -> Tuple[List[dict], List[dict]]:
    """Pull one holdout's records into <output_dir>/<holdout_id>/<descriptor>.lmdb.

    Returns (per_descriptor_summary_rows, manifest_rows). Manifest rows
    are unique per (vertical, rel_path) and include a 'descriptors_found'
    comma-separated column.
    """
    out_holdout_dir = output_dir / holdout_id
    out_holdout_dir.mkdir(parents=True, exist_ok=True)

    # Group requested keys by vertical so we open each source LMDB once.
    by_vertical: Dict[str, List[str]] = {}
    for v, rp in zip(holdout_df["vertical"], holdout_df["rel_path"]):
        by_vertical.setdefault(v, []).append(rp)

    requested_n = len(holdout_df)
    available_verticals = set(discover_verticals(lmdb_root))
    requested_verticals = set(by_vertical)
    missing_verticals = sorted(requested_verticals - available_verticals)
    if missing_verticals:
        n_missing_records = sum(len(by_vertical[v]) for v in missing_verticals)
        print(f"  {holdout_id}: {len(missing_verticals)} verticals not on disk, "
              f"{n_missing_records:,}/{requested_n:,} records will be skipped: "
              f"{missing_verticals[:5]}{'...' if len(missing_verticals) > 5 else ''}")

    # found_per_record[(vertical, rel_path)] = set of descriptors where record was pulled
    found_per_record: Dict[Tuple[str, str], Set[str]] = {}
    summary_rows: List[dict] = []

    for descriptor in descriptors:
        out_lmdb_path = out_holdout_dir / f"{descriptor}.lmdb"
        # Remove stale LMDB to avoid mixing keys across reruns
        for ext in ("", "-lock"):
            p = Path(str(out_lmdb_path) + ext)
            if p.exists():
                p.unlink()

        env_out = lmdb.open(
            str(out_lmdb_path),
            map_size=LMDB_MAP_SIZE,
            subdir=False,
            meminit=False,
            map_async=True,
        )

        n_found = 0
        n_missing_record = 0
        n_skipped_missing_lmdb = 0

        for vertical in sorted(requested_verticals):
            if vertical not in available_verticals:
                continue
            src = find_lmdb(lmdb_root, vertical, descriptor)
            if src is None:
                n_skipped_missing_lmdb += len(by_vertical[vertical])
                continue
            env_in = lmdb.open(
                str(src), subdir=False, readonly=True, lock=False,
                readahead=True, meminit=False,
            )
            with env_in.begin() as txn_in, env_out.begin(write=True) as txn_out:
                for rp in by_vertical[vertical]:
                    src_key = manifest_rel_path_to_lmdb_key(vertical, rp)
                    val = txn_in.get(src_key.encode("ascii"))
                    if val is None:
                        n_missing_record += 1
                        continue
                    out_key = f"{vertical}__{src_key}".encode("ascii")
                    txn_out.put(out_key, val)
                    n_found += 1
                    found_per_record.setdefault(
                        (vertical, rp), set()
                    ).add(descriptor)
            env_in.close()

        # length sentinel matches convention in utils.lmdbs
        with env_out.begin(write=True) as txn_out:
            txn_out.put(b"length", pickle.dumps(n_found, protocol=-1))
        env_out.sync()
        env_out.close()

        summary_rows.append({
            "holdout_id": holdout_id,
            "descriptor": descriptor,
            "requested": requested_n,
            "found": n_found,
            "missing_record": n_missing_record,
            "missing_due_to_absent_lmdb": n_skipped_missing_lmdb,
        })
        print(f"    {descriptor:<10} found={n_found:>7,}  "
              f"missing_record={n_missing_record:>5,}  "
              f"absent_lmdb={n_skipped_missing_lmdb:>6,}")

    manifest_rows: List[dict] = []
    for (v, rp), descs in found_per_record.items():
        manifest_rows.append({
            "holdout_id": holdout_id,
            "vertical": v,
            "rel_path": rp,
            "key": f"{v}__{rp}",
            "descriptors_found": ",".join(sorted(descs)),
            "n_descriptors_found": len(descs),
        })

    return summary_rows, manifest_rows


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--holdout_dir", type=Path, default=DEFAULT_HOLDOUT_DIR,
                   help=f"Directory holding INDEX.csv and h*_*.csv. "
                        f"Default: {DEFAULT_HOLDOUT_DIR}")
    p.add_argument("--lmdb_root", type=Path, default=DEFAULT_LMDB_ROOT,
                   help=f"Per-vertical LMDB root. Default: {DEFAULT_LMDB_ROOT}")
    p.add_argument("--output_dir", type=Path, default=DEFAULT_OUT_DIR,
                   help=f"Where to write H{{N}}/<descriptor>.lmdb files. "
                        f"Default: {DEFAULT_OUT_DIR}")
    p.add_argument("--descriptors", nargs="+", default=list(DEFAULT_DESCRIPTORS),
                   help=f"Descriptor families to pull. Default: "
                        f"{' '.join(DEFAULT_DESCRIPTORS)}")
    p.add_argument("--only", nargs="*", default=None,
                   help="Only pull these holdout IDs (e.g. --only H1 H6)")
    args = p.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"holdout_dir: {args.holdout_dir}")
    print(f"lmdb_root:   {args.lmdb_root}")
    print(f"output_dir:  {args.output_dir}")
    print(f"descriptors: {args.descriptors}")

    available = discover_verticals(args.lmdb_root)
    print(f"\nverticals on disk: {len(available)}")
    for v in available:
        print(f"  {v}")

    holdouts = load_holdout_csvs(args.holdout_dir)
    if args.only:
        keep = set(args.only)
        holdouts = {k: v for k, v in holdouts.items() if k in keep}

    all_summary: List[dict] = []
    all_manifest: List[dict] = []
    for hid, hdf in holdouts.items():
        print(f"\n=== {hid}: requested {len(hdf):,} records ===")
        summary_rows, manifest_rows = pull_one(
            hid, hdf, args.lmdb_root, args.descriptors, args.output_dir,
        )
        all_summary.extend(summary_rows)
        all_manifest.extend(manifest_rows)

    summary_df = pd.DataFrame(all_summary)
    summary_df.to_csv(args.output_dir / "pull_summary.csv", index=False)
    print(f"\nwrote pull_summary.csv ({len(summary_df)} rows)")

    if all_manifest:
        man_df = pd.DataFrame(all_manifest).sort_values(
            ["holdout_id", "vertical", "rel_path"]
        ).reset_index(drop=True)
        man_path = args.output_dir / "manifest_holdout.parquet"
        man_df.to_parquet(man_path, index=False)
        print(f"wrote manifest_holdout.parquet ({len(man_df):,} rows)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
