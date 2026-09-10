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
    "cm5.txt",
    "fuzzy_bond.txt",
    "hirshfeld.txt",
    "hirsh_fuzzy_density.txt",
    # remaining level-0 Multiwfn stdin scripts (rewritten by create_jobs every pass)
    "hirsh_fuzzy_spin.txt",
    "becke_fuzzy_spin.txt",
    "qtaim.txt",
    "charge.txt",
    "bond.txt",
    "convert.txt",
    "convert.in",
    # level-1 scripts (full_set > 0)
    "vdd.txt",
    "mbis.txt",
    "chelpg.txt",
    "ibsi_bond.txt",
    "elf_fuzzy.txt",
    "mbis_fuzzy_density.txt",
    "mbis_fuzzy_spin.txt",
    # level-2 scripts (full_set > 1)
    "bader.txt",
    "laplacian_bond.txt",
    "laplacian_rho_fuzzy.txt",
    "grad_norm_rho_fuzzy.txt",
    # ORCA sidecars: unread by any parser, also archived inside orca.tar.zst
    "orca.property.txt",
    "orca_stderr",
    ".molden.input",
    "settings.ini",
    "orca.gbw.zstd0",
)

# Heavy ORCA artifacts in a results tree, mapped to the source-tree files that
# regenerate them on the next process_folder_alcf pass: the compressed inputs
# are copied over, orca.out is untarred from orca.tar.zst, and the wfx/wfn is
# converted from orca.gbw.zstd0. Only deleted with --purge-orca, and only when
# one of the listed source files exists non-empty in the mirrored source folder.
PURGE_SOURCES = {
    "orca.out": ("orca.out", "orca.tar.zst"),
    "orca.wfx": ("orca.wfx", "orca.gbw.zstd0", "orca.gbw"),
    "orca.wfn": ("orca.wfn", "orca.gbw.zstd0", "orca.gbw"),
    "orca.tar.zst": ("orca.tar.zst",),
    "density_mat.npz": ("density_mat.npz",),
}
PURGE_ORCA_NAMES = tuple(PURGE_SOURCES)

# Matches acquire_lock's stale threshold in core/workflow.py (_LOCK_MAX_AGE_S).
LOCK_MAX_AGE_S = 28800.0


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


def folder_is_live(path: str, lock_max_age: float) -> bool:
    """A .processing.lock younger than lock_max_age means a worker owns the folder."""
    try:
        age = time.time() - os.path.getmtime(os.path.join(path, ".processing.lock"))
    except OSError:
        return False
    return age < lock_max_age


def source_can_regenerate(source_folder: str, name: str) -> bool:
    """True if source_folder holds a non-empty file that regenerates `name`."""
    for src_name in PURGE_SOURCES[name]:
        try:
            if os.path.getsize(os.path.join(source_folder, src_name)) > 0:
                return True
        except OSError:
            continue
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


def iter_files(root, purge_orca=False, source_root=None, lock_max_age=LOCK_MAX_AGE_S, skips=None):
    """Generator yielding files to delete as they are discovered.

    Without purge_orca this yields plain paths, as before. With purge_orca,
    folders holding a live .processing.lock are skipped entirely (recorded in
    skips["locked"] when a dict is given) and PURGE_ORCA_NAMES are yielded as
    (path, source_folder, name) tuples; the source check happens in the
    worker (process_item) so its source-tree stats run in parallel and a hung
    stat cannot stall the scan.
    """
    stack = [root]

    while stack:
        path = stack.pop()
        if purge_orca and folder_is_live(path, lock_max_age):
            if skips is not None:
                skips["locked"].append(path)
            continue
        source_folder = None
        if purge_orca:
            source_folder = os.path.join(source_root, os.path.relpath(path, root))
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
                        elif purge_orca and entry.name in PURGE_SOURCES:
                            yield (entry.path, source_folder, entry.name)
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


def process_item(item, dry_run=False):
    """Pool worker. Returns (path, deleted_count, no_mirror_flag).

    A plain path is deleted unconditionally. A (path, source_folder, name)
    tuple from --purge-orca is deleted only if source_can_regenerate holds;
    otherwise it is reported back with no_mirror_flag set.
    """
    if isinstance(item, str):
        return item, delete_file(item, dry_run=dry_run), False
    path, source_folder, name = item
    if not source_can_regenerate(source_folder, name):
        return path, 0, True
    return path, delete_file(path, dry_run=dry_run), False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", help="Root directory")
    parser.add_argument("-j", "--jobs", type=int, default=cpu_count())
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--purge-orca",
        action="store_true",
        help="Also delete orca.out, orca.wfx/.wfn, orca.tar.zst and density_mat.npz "
        "from a results tree, but only where the mirrored folder under --source-root still "
        "holds the file or the compressed input that regenerates it (orca.tar.zst for "
        "orca.out, orca.gbw.zstd0 for the wfx/wfn). Folders with a live .processing.lock "
        "are skipped.",
    )
    parser.add_argument(
        "--source-root",
        help="Input tree that process_folder_alcf copies from; required with --purge-orca",
    )
    parser.add_argument(
        "--lock-max-age",
        type=float,
        default=LOCK_MAX_AGE_S,
        help="Seconds before a .processing.lock counts as stale (default: 8 h)",
    )
    args = parser.parse_args()

    if args.purge_orca:
        if not args.source_root:
            parser.error("--purge-orca requires --source-root")
        if not os.path.isdir(args.source_root):
            parser.error(f"--source-root is not a directory: {args.source_root}")
        if os.path.realpath(args.source_root) == os.path.realpath(args.root):
            parser.error("--source-root must differ from root")

    print(f"Streaming scan from: {args.root}")
    if args.purge_orca:
        print(f"Purging ORCA artifacts mirrored under: {args.source_root}")
    print("Dry run enabled" if args.dry_run else "Deleting files")

    skips = {"locked": [], "no_mirror": []}
    files = iter_files(
        args.root,
        purge_orca=args.purge_orca,
        source_root=args.source_root,
        lock_max_age=args.lock_max_age,
        skips=skips,
    )

    worker = partial(process_item, dry_run=args.dry_run)
    names = set()
    deleted = 0

    with Pool(args.jobs) as p:
        for path, count, no_mirror in tqdm(
            p.imap_unordered(worker, files, chunksize=64),
            desc="Scanning" if args.dry_run else "Processing",
            unit="files",
        ):
            if no_mirror:
                skips["no_mirror"].append(path)
                continue
            deleted += count
            if args.dry_run:
                names.add(os.path.basename(path))

    if args.dry_run:
        print(f"Total files that would be deleted: {deleted}")
        print(f"Unique filenames ({len(names)}):")
        for name in sorted(names):
            print(f"  {name}")
    else:
        print(f"Total files processed: {deleted}")

    if args.purge_orca:
        print(f"Folders skipped (live lock): {len(skips['locked'])}")
        print(f"Files skipped (no source to regenerate from): {len(skips['no_mirror'])}")
        if args.dry_run:
            for path in sorted(skips["locked"])[:20]:
                print(f"  locked    {path}")
            for path in sorted(skips["no_mirror"])[:20]:
                print(f"  no-source {path}")


if __name__ == "__main__":
    main()