"""Fast parallel directory structure scanner for debugging layouts.

Provides a CLI to scan a root directory, find leaf folders, and print
summaries (counts per subset, largest folders, samples). Optimized for
I/O-bound workloads using threads and os.scandir.

Usage:
    python -m qtaim_gen.source.scripts.scan_structure --root /path/to/root -w 12 --sample 20
"""
from __future__ import annotations

import argparse
import os
import concurrent.futures
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, List, Tuple


@dataclass
class FolderInfo:
    subset: str
    path: str
    depth: int
    n_files: int
    n_subdirs: int


def iter_leaf_dirs(start_path: str) -> Iterable[str]:
    """Yield leaf directories (no subdirectories) under start_path.

    Iterative implementation using stack + os.scandir for speed.
    """
    stack = [start_path]
    while stack:
        path = stack.pop()
        try:
            with os.scandir(path) as it:
                subdirs = [entry.path for entry in it if entry.is_dir(follow_symlinks=False)]
        except (PermissionError, OSError):
            continue

        if not subdirs:
            yield path
        else:
            stack.extend(subdirs)


def inspect_folder(path: str) -> Tuple[int, int]:
    """Return (n_files, n_subdirs) for given folder (non-recursive)."""
    n_files = 0
    n_subdirs = 0
    try:
        with os.scandir(path) as it:
            for entry in it:
                if entry.is_dir(follow_symlinks=False):
                    n_subdirs += 1
                elif entry.is_file(follow_symlinks=False):
                    n_files += 1
    except (PermissionError, OSError):
        return 0, 0
    return n_files, n_subdirs


def scan_root(root: str, max_workers: int = 8, sample_limit: int = 100) -> List[FolderInfo]:
    """Scan the root folder and return FolderInfo entries.

    Strategy:
    - For each top-level category (child of root), list its subsets (dirs)
      and then find leaf directories under each subset.
    - Use ThreadPoolExecutor to parallelize inspection of leaf folders.
    """
    root = os.path.abspath(root)
    entries: List[FolderInfo] = []

    # Build list of subset roots (first-level categories under root)
    try:
        top_level = [os.path.join(root, d) for d in os.listdir(root)]
    except (PermissionError, FileNotFoundError):
        return []

    subset_roots = [p for p in top_level if os.path.isdir(p)]

    # For each subset root, find leaf directories and collect tasks
    tasks = []
    for subset_root in subset_roots:
        subset_name = os.path.basename(subset_root)
        # If this category has deeper nested layout (like 'omol'), find deepest leaves
        for leaf in iter_leaf_dirs(subset_root):
            # Post-process: if leaf basename is 'generator', use parent
            if os.path.basename(leaf) == "generator":
                leaf = os.path.dirname(leaf)
            rel = os.path.relpath(leaf, root)
            depth = rel.count(os.sep)
            tasks.append((subset_name, leaf, depth))

    # Inspect folders in parallel
    results: List[FolderInfo] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(inspect_folder, t[1]): t for t in tasks}
        for fut in concurrent.futures.as_completed(futures):
            subset_name, leaf, depth = futures[fut]
            try:
                n_files, n_subdirs = fut.result()
            except Exception:
                n_files, n_subdirs = 0, 0
            results.append(FolderInfo(subset=subset_name, path=leaf, depth=depth, n_files=n_files, n_subdirs=n_subdirs))

    # Optionally sort results (largest first by n_files)
    results.sort(key=lambda x: (x.n_files, -x.depth), reverse=True)
    return results[:sample_limit] if sample_limit and len(results) > sample_limit else results


def summarize(results: List[FolderInfo], show_top: int = 10) -> None:
    total = len(results)
    print(f"Total leaf folders found: {total}")
    by_subset = Counter(r.subset for r in results)
    print("Counts per subset (top 10):")
    for subset, cnt in by_subset.most_common(10):
        print(f"  {subset}: {cnt}")

    print(f"\nTop {show_top} folders by file count:")
    for fi in results[:show_top]:
        print(f"  files={fi.n_files:4d} subs={fi.n_subdirs:2d} depth={fi.depth:2d} - {fi.path}")


def main():
    p = argparse.ArgumentParser(description="Fast parallel directory structure scanner")
    p.add_argument("--root", "-r", required=True, help="Root directory to scan")
    p.add_argument("--workers", "-w", type=int, default=8, help="Number of worker threads")
    p.add_argument("--sample", "-s", type=int, default=200, help="Limit number of results to show")
    p.add_argument("--top", "-t", type=int, default=10, help="Top N to print by file count")
    args = p.parse_args()

    results = scan_root(args.root, max_workers=args.workers, sample_limit=args.sample)
    summarize(results, show_top=args.top)


if __name__ == "__main__":
    main()
