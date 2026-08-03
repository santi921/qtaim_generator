#!/usr/bin/env python3
"""Count orca.json files per vertical in root output directories."""
import os
import argparse
import concurrent.futures
from tqdm import tqdm


def find_leaf_folders(path):
    try:
        items = os.listdir(path)
    except (PermissionError, OSError):
        return []
    subdirs = [os.path.join(path, i) for i in items if os.path.isdir(os.path.join(path, i))]
    if not subdirs:
        return [path]
    leaves = []
    for subdir in subdirs:
        leaves.extend(find_leaf_folders(subdir))
    return leaves


def get_job_folders_for_vertical(root_dir, vertical):
    vert_path = os.path.join(root_dir, vertical)
    if not os.path.isdir(vert_path):
        return []

    if vertical == "omol":
        leaf_folders = find_leaf_folders(vert_path)
        processed = set()
        for f in leaf_folders:
            if os.path.basename(f) == "generator":
                processed.add(os.path.dirname(f))
            else:
                processed.add(f)
        return list(processed)
    else:
        folders = []
        for subset in os.listdir(vert_path):
            subset_path = os.path.join(vert_path, subset)
            if os.path.isdir(subset_path):
                folders.append(subset_path)
        return folders


def has_orca_json(folder):
    return (
        os.path.exists(os.path.join(folder, "generator", "orca.json"))
        or os.path.exists(os.path.join(folder, "orca.json"))
    )


def count_vertical(root_dir, vertical, workers):
    folders = get_job_folders_for_vertical(root_dir, vertical)
    if not folders:
        return 0, 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        results = list(tqdm(
            executor.map(has_orca_json, folders),
            total=len(folders),
            desc=vertical,
            leave=False,
        ))
    return sum(results), len(folders)


def main():
    parser = argparse.ArgumentParser(
        description="Count orca.json files per vertical in root output directories."
    )
    parser.add_argument(
        "--root_output_dirs",
        nargs="+",
        required=True,
        help="One or more root output directories to scan",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of worker threads (default: 8)",
    )
    parser.add_argument(
        "--verticals",
        nargs="+",
        default=None,
        help="Verticals to check (default: all subdirectories)",
    )
    args = parser.parse_args()

    for root_dir in args.root_output_dirs:
        print(f"checking verticals at root output dir: {root_dir}")
        if not os.path.isdir(root_dir):
            print(f"  [ERROR] directory not found: {root_dir}")
            continue

        verticals = args.verticals
        if verticals is None:
            verticals = sorted(
                d for d in os.listdir(root_dir)
                if os.path.isdir(os.path.join(root_dir, d))
            )

        for vertical in verticals:
            count, total = count_vertical(root_dir, vertical, args.workers)
            print(f"  {vertical}: {count} / {total}")


if __name__ == "__main__":
    main()
