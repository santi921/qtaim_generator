#!/usr/bin/env python3

import os
import re
import sys
import time
import argparse
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm


SCAN_RETRIES = 3
SCAN_RETRY_SLEEP = 0.5


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
)

def should_delete(filename: str) -> bool:
    if filename.endswith(STATIC_SUFFIXES):
        return True
    if ".tmp" in filename:  # matches .tmp, .tmp.0, .tmp.123, etc.
        return True
    if "core." in filename:
        return True
    if re.fullmatch(r"orca\.\d+", filename):
        return True
    return False


def _safe_scandir(path):
    """Open scandir with retries; return None on persistent failure."""
    last_err = None
    for attempt in range(SCAN_RETRIES):
        try:
            return os.scandir(path)
        except (FileNotFoundError, NotADirectoryError):
            return None
        except PermissionError as e:
            last_err = e
            return None
        except OSError as e:
            last_err = e
            time.sleep(SCAN_RETRY_SLEEP * (attempt + 1))
    print(f"[warn] scandir failed for {path}: {last_err}", file=sys.stderr)
    return None


def _entry_is_dir(entry):
    try:
        return entry.is_dir(follow_symlinks=False)
    except OSError:
        return False


def _entry_is_file(entry):
    try:
        return entry.is_file(follow_symlinks=False)
    except OSError:
        return False


def iter_files(root):
    """Generator yielding files to delete as they are discovered"""
    stack = [root]

    while stack:
        path = stack.pop()
        it = _safe_scandir(path)
        if it is None:
            continue
        try:
            with it:
                while True:
                    try:
                        entry = next(it)
                    except StopIteration:
                        break
                    except OSError as e:
                        print(f"[warn] iter failed in {path}: {e}", file=sys.stderr)
                        break
                    if _entry_is_dir(entry):
                        stack.append(entry.path)
                    elif _entry_is_file(entry):
                        if should_delete(entry.name):
                            yield entry.path
        except OSError as e:
            print(f"[warn] scandir context failed for {path}: {e}", file=sys.stderr)


def delete_file(path, dry_run=False):
    if dry_run:
        return 1
    try:
        os.unlink(path)
        return 1
    except FileNotFoundError:
        return 0
    except OSError as e:
        print(f"[warn] unlink failed for {path}: {e}", file=sys.stderr)
        return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", help="Root directory")
    parser.add_argument("-j", "--jobs", type=int, default=cpu_count())
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    print(f"Streaming scan from: {args.root}")
    print("Dry run enabled" if args.dry_run else "Deleting files")

    if args.dry_run:
        names = set()
        deleted = 0
        for path in tqdm(iter_files(args.root), desc="Scanning", unit="files"):
            names.add(os.path.basename(path))
            deleted += 1
        print(f"Total files that would be deleted: {deleted}")
        print(f"Unique filenames ({len(names)}):")
        for name in sorted(names):
            print(f"  {name}")
        return

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