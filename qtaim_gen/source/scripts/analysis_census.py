"""CLI for Stream C: per-vertical census table T1.

Level-0 calcs only. Layout resolution:

- ``--root`` contains ``structure.lmdb`` -> single-vertical mode.
- Otherwise -> corpus mode: one row per immediate subdir ``<v>`` where
  ``--root/<v>/structure.lmdb`` exists. Subdirs in
  ``pull_holdout_records.NON_VERTICAL_DIRS`` are skipped, as are subdirs
  whose LMDBs live only under ``merged/`` (level-2 / orca refinement;
  not yet finalized at the time of writing).

Examples:

    analysis-census --root data/OMol4M_lmdbs/droplet --output census_droplet.parquet
    analysis-census --root data/OMol4M_lmdbs --output census_all.parquet

Sharded parallel usage (one job per vertical, then concat):

    analysis-census --root /p/lustre5/.../converters_final \\
        --include_verticals droplet --output census_shards/droplet.parquet
    # ... in parallel for every vertical ...
    analysis-census --merge_from census_shards/*.parquet --output census_all.parquet

Missing verticals named in ``--include_verticals`` are warned about and
skipped (matches ``pull_holdout_records`` behavior); they do not fail the
job.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from qtaim_gen.source.analysis.census import CENSUS_FIELDS, census, discover_verticals


def main():
    parser = argparse.ArgumentParser(
        description="Per-vertical census (Stream C / paper T1).",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Vertical root (containing structure.lmdb) or corpus root "
        "containing one subdirectory per vertical. Required unless --merge_from is set.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output parquet file path.",
    )
    parser.add_argument(
        "--include_verticals",
        nargs="*",
        default=None,
        metavar="V",
        help="Restrict to these vertical names (matches pull_holdout_records "
        "convention). Names not on disk are warned about and skipped.",
    )
    parser.add_argument(
        "--merge_from",
        nargs="*",
        type=Path,
        default=None,
        metavar="SHARD_PARQUET",
        help="Merge mode: read these shard parquet files (one or more rows "
        "each) and concatenate into a single sorted census parquet at "
        "--output. When set, --root is ignored.",
    )
    args = parser.parse_args()

    if args.merge_from:
        return _run_merge(args)
    return _run_compute(args)


def _run_merge(args) -> int:
    frames = []
    for shard in args.merge_from:
        if not shard.exists():
            print(f"warn: --merge_from shard {shard} does not exist; skipping",
                  file=sys.stderr)
            continue
        frames.append(pd.read_parquet(shard))
    if not frames:
        print("error: no shard parquets found via --merge_from", file=sys.stderr)
        return 2
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=["vertical"], keep="last")
    df = df.sort_values("vertical", kind="stable").reset_index(drop=True)
    df = df.reindex(columns=list(CENSUS_FIELDS))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output, index=False)
    print(f"merged {len(args.merge_from)} shards -> {len(df)} rows -> {args.output}")
    print(df.to_string(index=False))
    return 0


def _run_compute(args) -> int:
    if args.root is None:
        print("error: --root is required (unless --merge_from is set)", file=sys.stderr)
        return 2
    if not args.root.exists():
        print(f"error: --root {args.root} does not exist", file=sys.stderr)
        return 2

    verticals = discover_verticals(args.root)
    if args.include_verticals is not None:
        requested = set(args.include_verticals)
        unknown = sorted(requested - {name for name, _ in verticals})
        if unknown:
            print(f"warn: --include_verticals contains {len(unknown)} not under "
                  f"--root: {unknown}", file=sys.stderr)
        verticals = [(name, p) for name, p in verticals if name in requested]
    if not verticals:
        print(
            f"error: no verticals to process under {args.root}",
            file=sys.stderr,
        )
        return 2

    rows = [census(lmdb_dir, vertical_name=name) for name, lmdb_dir in verticals]
    df = pd.DataFrame(rows, columns=list(CENSUS_FIELDS))
    df = df.sort_values("vertical", kind="stable").reset_index(drop=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output, index=False)

    print(f"wrote {len(df)} rows to {args.output}")
    print(df.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
