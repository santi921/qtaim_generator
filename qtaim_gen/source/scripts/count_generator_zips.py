#!/usr/bin/env python3
"""
Count zip files in trans1x/*/generator folders using multiprocessing and tqdm.

Usage:
  ./scripts/count_generator_zips.py [ROOT] [--workers N]

Default ROOT: /lus/eagle/projects/generator/OMol25_postprocessing/trans1x

This script intentionally lists only the immediate children of ROOT and checks
for a `generator/` subdirectory in each child. That avoids blind recursive
walks and keeps the work units small and parallelizable.
"""
from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count
from typing import Iterable

from tqdm import tqdm


def iter_children_dirs(root: str) -> Iterable[str]:
    """Yield absolute paths of immediate subdirectories in `root`.

    Uses os.scandir for efficiency and to avoid recursion.
    """
    with os.scandir(root) as it:
        for entry in it:
            if entry.is_dir(follow_symlinks=False):
                yield entry.path


def count_zips_in_generator(dirpath: str) -> int:
    """Count regular files ending with .zip (case-insensitive) inside dirpath/generator.

    Returns 0 if the generator dir doesn't exist or an error occurs.
    """
    gen = os.path.join(dirpath, "generator")
    if not os.path.isdir(gen):
        return 0
    cnt = 0
    try:
        with os.scandir(gen) as it:
            for entry in it:
                if not entry.is_file(follow_symlinks=False):
                    continue
                if entry.name.lower().endswith(".zip"):
                    cnt += 1
    except Exception:
        # Ignore transient fs errors on a per-directory basis.
        return 0
    return cnt


def main() -> int:
    p = argparse.ArgumentParser(description="Count .zip files in trans1x/*/generator")
    p.add_argument(
        "root",
        nargs="?",
        default="/lus/eagle/projects/generator/OMol25_postprocessing/trans1x",
        help="Root directory containing many subfolders (default: trans1x)",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=max(4, cpu_count()),
        help="Number of worker processes (default: max(4, CPU count))",
    )
    args = p.parse_args()

    root = args.root
    if not os.path.isdir(root):
        print(f"Error: root directory '{root}' does not exist", flush=True)
        return 2

    children = list(iter_children_dirs(root))
    total_children = len(children)
    if total_children == 0:
        print("0")
        return 0
    print("Total subdirectories to process:", total_children, flush=True)

    total = 0
    # Use a ProcessPoolExecutor to parallelize counting across generator dirs.
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        # ex.map returns results in order; wrap with tqdm for progress.
        for cnt in tqdm(ex.map(count_zips_in_generator, children), total=total_children, desc="counting"):
            total += cnt

    print(total)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
