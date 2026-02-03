#!/usr/bin/env python3

import os
import argparse
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm


STATIC_SUFFIXES = (
    "orca.engrad",
    ".densities",
    ".bibtex",
    ".core",
    "core.",
    ".mfwn",
    ".gbw",
    "other_esp.txt",
    "other_alie.txt",
    "other_geometry.txt",
    "other.txt",
    "adch.txt",
    "becke_fuzzy_density.txt",
    "becke.txt",
    "adch.txt",
    "cm5.txt",
    "fuzzy_bond.txt",
    "hirshfeld.txt",
    "hirsh_fuzzy_density.txt",
    ".molden.input",
    "settings.ini",
    "orca.gbw.zstd0",
    "orca.out"
)

def should_delete(filename: str) -> bool:
    if filename.endswith(STATIC_SUFFIXES):
        return True
    if ".tmp" in filename:  # matches .tmp, .tmp.0, .tmp.123, etc.
        return True
    if "core." in filename:
        return True
    return False


def iter_files(root):
    """Generator yielding files to delete as they are discovered"""
    stack = [root]

    while stack:
        path = stack.pop()
        try:
            with os.scandir(path) as it:
                for entry in it:
                    if entry.is_dir(follow_symlinks=False):
                        stack.append(entry.path)
                    elif entry.is_file(follow_symlinks=False):
                        if should_delete(entry.name):
                            yield entry.path
        except (PermissionError, FileNotFoundError):
            pass
    
def delete_file(path, dry_run=False):
    if dry_run:
        return 1
    try:
        os.unlink(path)
        return 1
    except (FileNotFoundError, PermissionError):
        return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", help="Root directory")
    parser.add_argument("-j", "--jobs", type=int, default=cpu_count())
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    print(f"Streaming scan from: {args.root}")
    print("Dry run enabled" if args.dry_run else "Deleting files")

    worker = partial(delete_file, dry_run=args.dry_run)
    deleted = 0

    with Pool(args.jobs) as p:
        for result in tqdm(
            p.imap_unordered(worker, iter_files(args.root), chunksize=64),
            desc="Processing",
            unit="files",
        ):
            deleted += result

    print(f"Total files processed: {deleted}")


if __name__ == "__main__":
    main()