"""CLI entry point: build-manifest.

Builds per-vertical Parquet manifests of every job folder under <root>.
See docs/plans/2026-04-28-manifest-spec.md for the full spec.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List

from qtaim_gen.source.utils.manifest import (
    build_vertical,
    list_verticals,
    merge_manifests,
)


def _resolve_verticals(
    root: str, verticals_arg: str, verticals_from: str | None
) -> List[str]:
    if verticals_from:
        with open(verticals_from) as f:
            return [line.strip() for line in f if line.strip()]
    if verticals_arg == "all":
        return list_verticals(root)
    return [v.strip() for v in verticals_arg.split(",") if v.strip()]


def main() -> int:
    p = argparse.ArgumentParser(description="Build per-vertical Parquet manifest.")
    p.add_argument("--root", help="Dataset root containing one folder per vertical.")
    p.add_argument("--out-dir", required=True, help="Output directory for parquet files.")
    p.add_argument(
        "--verticals",
        default="all",
        help='"all" or comma-separated list of vertical names.',
    )
    p.add_argument(
        "--verticals-from", default=None, help="File with one vertical name per line."
    )
    p.add_argument("--workers", type=int, default=1, help="Process pool size per vertical.")
    p.add_argument("--limit", type=int, default=None, help="Cap jobs per vertical (smoke test).")
    p.add_argument("--chunk-size", type=int, default=5000, help="Rows per Parquet write batch.")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing manifest_*.parquet.")
    p.add_argument("--progress", action="store_true", help="Show a tqdm progress bar per vertical.")
    p.add_argument(
        "--merge",
        action="store_true",
        help="Merge existing manifest_*.parquet -> manifest.parquet. No --root needed.",
    )
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.merge and not args.root:
        merge_manifests(args.out_dir)
        return 0

    if not args.root:
        p.error("--root is required unless --merge is used standalone")

    verticals = _resolve_verticals(args.root, args.verticals, args.verticals_from)
    if not verticals:
        print("No verticals to process.", file=sys.stderr)
        return 1

    print(f"Building manifest for {len(verticals)} verticals at {args.root}")
    totals = {"ok": 0, "missing_inp": 0, "corrupt_inp": 0, "parse_error": 0}
    for v in verticals:
        summary = build_vertical(
            root=args.root,
            vertical=v,
            out_dir=args.out_dir,
            workers=args.workers,
            limit=args.limit,
            chunk_size=args.chunk_size,
            overwrite=args.overwrite,
            progress=args.progress,
        )
        for k in totals:
            totals[k] += summary.get(k, 0)

    print("=" * 60)
    print(
        f"TOTAL ok={totals['ok']} corrupt={totals['corrupt_inp']} "
        f"missing={totals['missing_inp']} parse_error={totals['parse_error']}"
    )

    if args.merge:
        merge_manifests(args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
