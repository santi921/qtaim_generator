"""Find invalid or empty .json files under a root directory or job list.

Scans to a configurable depth using threaded partition-per-child scanning.
Reports files that are empty (0 bytes), fail to parse, or decode to an
empty container ({} or []).

Modes:
    # Scan an entire tree
    python -m qtaim_gen.source.scripts.helpers.find_bad_json --root /path/to/data --depth 4

    # Scan only job folders from a list file (paths point to results root)
    python -m qtaim_gen.source.scripts.helpers.find_bad_json \\
        --job-file jobs.txt --root-results /results/root

    # Scan job folders with input->results path remapping
    python -m qtaim_gen.source.scripts.helpers.find_bad_json \\
        --job-file jobs.txt --root-inputs /inputs/root --root-results /results/root

    # Preview what would be deleted (dry run)
    python -m qtaim_gen.source.scripts.helpers.find_bad_json --root /path/to/data --delete --dry-run

    # Actually delete bad JSON files
    python -m qtaim_gen.source.scripts.helpers.find_bad_json --root /path/to/data --delete
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import concurrent.futures
from typing import List, Tuple

from tqdm import tqdm


def _scan_subtree(
    subroot: str, max_depth: int, base_depth: int
) -> Tuple[List[str], List[str], List[str]]:
    """Walk *subroot* up to *max_depth* levels from the original root.

    Returns three lists of absolute paths:
        (empty_files, parse_errors, empty_content)
    """
    empty_files: List[str] = []
    parse_errors: List[str] = []
    empty_content: List[str] = []

    for dirpath, dirnames, filenames in os.walk(subroot, followlinks=False):
        # Enforce depth limit relative to the original root
        current_depth = dirpath.rstrip(os.sep).count(os.sep) - base_depth
        if current_depth >= max_depth:
            dirnames.clear()
            continue

        for fname in filenames:
            if not fname.endswith(".json"):
                continue

            fpath = os.path.join(dirpath, fname)
            try:
                size = os.path.getsize(fpath)
            except OSError:
                continue

            if size == 0:
                empty_files.append(fpath)
                continue

            try:
                with open(fpath, "r") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, UnicodeDecodeError, OSError):
                parse_errors.append(fpath)
                continue

            if data == {} or data == []:
                empty_content.append(fpath)

    return empty_files, parse_errors, empty_content


def _check_json_file(fpath: str) -> Tuple[str, str]:
    """Check a single .json file. Returns (fpath, category).

    category is one of: "ok", "empty", "invalid", "empty_content".
    """
    try:
        size = os.path.getsize(fpath)
    except OSError:
        return fpath, "ok"
    if size == 0:
        return fpath, "empty"
    try:
        with open(fpath, "r") as f:
            data = json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError, OSError):
        return fpath, "invalid"
    if data == {} or data == []:
        return fpath, "empty_content"
    return fpath, "ok"


def _scan_job_folder(job_dir: str) -> Tuple[List[str], List[str], List[str]]:
    """Check all .json files in a single job folder and its generator/ subdir."""
    empty: List[str] = []
    invalid: List[str] = []
    empty_content: List[str] = []

    dirs_to_check = [job_dir]
    gen_dir = os.path.join(job_dir, "generator")
    if os.path.isdir(gen_dir):
        dirs_to_check.append(gen_dir)

    for d in dirs_to_check:
        try:
            entries = os.listdir(d)
        except (FileNotFoundError, PermissionError):
            continue
        for fname in entries:
            if not fname.endswith(".json"):
                continue
            fpath = os.path.join(d, fname)
            if not os.path.isfile(fpath):
                continue
            _, cat = _check_json_file(fpath)
            if cat == "empty":
                empty.append(fpath)
            elif cat == "invalid":
                invalid.append(fpath)
            elif cat == "empty_content":
                empty_content.append(fpath)

    return empty, invalid, empty_content


def _remap_to_results(
    folder: str, root_inputs: str, root_results: str
) -> str:
    """Remap a folder path from input root to results root."""
    folder = folder.rstrip(os.sep)
    root_inputs = root_inputs.rstrip(os.sep)
    if folder.startswith(root_inputs):
        relative = folder[len(root_inputs) :].lstrip(os.sep)
        return os.path.join(root_results, relative)
    return folder


def _read_job_file(path: str) -> List[str]:
    """Read non-blank, stripped lines from a text file."""
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def find_bad_json(
    root: str, depth: int = 4, max_workers: int = 8
) -> Tuple[List[str], List[str], List[str]]:
    """Scan *root* for bad .json files, returning (empty, invalid, empty_content)."""
    root = os.path.abspath(root)
    base_depth = root.rstrip(os.sep).count(os.sep)

    try:
        children = [
            os.path.join(root, d)
            for d in os.listdir(root)
            if os.path.isdir(os.path.join(root, d))
        ]
    except (FileNotFoundError, PermissionError):
        return [], [], []

    if not children:
        children = [root]

    all_empty: List[str] = []
    all_invalid: List[str] = []
    all_empty_content: List[str] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(_scan_subtree, child, depth, base_depth): child
            for child in children
        }
        for fut in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc="Scanning",
        ):
            try:
                empty, invalid, empty_cont = fut.result()
                all_empty.extend(empty)
                all_invalid.extend(invalid)
                all_empty_content.extend(empty_cont)
            except Exception as e:
                subroot = futures[fut]
                print(f"Error scanning {subroot}: {e}")

    return all_empty, all_invalid, all_empty_content


def find_bad_json_from_jobs(
    job_file: str,
    root_results: str = None,
    root_inputs: str = None,
    max_workers: int = 8,
) -> Tuple[List[str], List[str], List[str]]:
    """Scan job folders listed in *job_file* for bad .json files.

    If *root_inputs* and *root_results* are both provided, paths in the job
    file are remapped from input root to results root before scanning.
    If only *root_results* is given, it is used as-is (paths already point
    to results).
    """
    folders = _read_job_file(job_file)
    if not folders:
        return [], [], []

    if root_inputs and root_results:
        folders = [_remap_to_results(f, root_inputs, root_results) for f in folders]
    elif root_results:
        # job file has relative paths or already results-rooted paths
        folders = [
            os.path.join(root_results, f) if not os.path.isabs(f) else f
            for f in folders
        ]

    print(f"Checking {len(folders)} job folders from {job_file}")

    all_empty: List[str] = []
    all_invalid: List[str] = []
    all_empty_content: List[str] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(_scan_job_folder, folder): folder for folder in folders
        }
        for fut in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc="Scanning jobs",
        ):
            try:
                empty, invalid, empty_cont = fut.result()
                all_empty.extend(empty)
                all_invalid.extend(invalid)
                all_empty_content.extend(empty_cont)
            except Exception as e:
                folder = futures[fut]
                print(f"Error scanning {folder}: {e}")

    return all_empty, all_invalid, all_empty_content


def _extract_first_json_object(text: str) -> str | None:
    """Extract the first balanced ``{...}`` from *text* via brace counting.

    Skips braces inside JSON string literals. Returns the substring if it
    parses as valid JSON, otherwise ``None``.
    """
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if escape:
            escape = False
            continue
        if ch == "\\":
            if in_string:
                escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : i + 1]
                try:
                    json.loads(candidate)
                    return candidate
                except (json.JSONDecodeError, ValueError):
                    return None
    return None


def _try_repair_json(fpath: str) -> bool:
    """Try to repair a corrupted JSON file by extracting the first valid object.

    Returns True if the file was successfully repaired.
    """
    try:
        with open(fpath, "r") as f:
            raw = f.read()
    except OSError:
        return False

    fixed = _extract_first_json_object(raw)
    if fixed is None:
        return False

    data = json.loads(fixed)
    if data == {} or data == []:
        return False

    with open(fpath, "w") as f:
        json.dump(data, f, indent=4)
    return True


def _fix_timings(bad_paths: List[str]) -> Tuple[int, int, int]:
    """For bad files ending in ``generator/timings.json``, try to fix them.

    Strategy (in order):
      1. Copy from sibling ``../timings.json`` if it exists and is valid.
      2. Extract the first balanced JSON object from the corrupted file itself.

    Returns (fixed_count, repaired_count, skipped_count).
    """
    suffix = os.path.join("generator", "timings.json")
    candidates = [p for p in bad_paths if p.endswith(suffix)]
    fixed = 0
    repaired = 0
    skipped = 0
    for bad in candidates:
        # .../job/generator/timings.json  ->  .../job/timings.json
        job_dir = os.path.dirname(os.path.dirname(bad))
        fallback = os.path.join(job_dir, "timings.json")

        # Strategy 1: copy from fallback sibling
        if os.path.isfile(fallback):
            try:
                with open(fallback, "r") as f:
                    data = json.load(f)
                if data != {} and data != []:
                    shutil.copy2(fallback, bad)
                    print(f"  FIXED (fallback): {bad}  <-  {fallback}")
                    fixed += 1
                    continue
            except (json.JSONDecodeError, UnicodeDecodeError, OSError):
                pass  # fallback is also bad, try strategy 2

        # Strategy 2: extract first valid JSON object from the file itself
        if _try_repair_json(bad):
            print(f"  REPAIRED (truncated): {bad}")
            repaired += 1
            continue

        print(f"  SKIP (unfixable): {bad}")
        skipped += 1
    return fixed, repaired, skipped


def main(argv=None) -> None:
    p = argparse.ArgumentParser(
        description="Find invalid or empty .json files under a root directory"
    )
    p.add_argument("--root", "-r", help="Root directory to scan (tree mode)")
    p.add_argument(
        "--depth", "-d", type=int, default=4, help="Max directory depth (default: 4)"
    )
    p.add_argument(
        "--workers", "-w", type=int, default=8, help="Number of threads (default: 8)"
    )
    p.add_argument(
        "--output", "-o", help="Write bad file paths to this file (one per line)"
    )
    p.add_argument(
        "--fix-timings",
        action="store_true",
        help="Overwrite bad generator/timings.json files from sibling timings.json",
    )
    p.add_argument(
        "--delete",
        action="store_true",
        help="Delete all bad JSON files found (use with --dry-run to preview first)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="With --delete, print what would be deleted without removing anything",
    )
    p.add_argument(
        "--job-file",
        help="File with one job-folder path per line (job-list mode)",
    )
    p.add_argument(
        "--root-results",
        help="Results root directory (used with --job-file for path remapping)",
    )
    p.add_argument(
        "--root-inputs",
        help="Inputs root directory (used with --job-file to remap input paths to results)",
    )
    args = p.parse_args(argv)

    if args.job_file:
        empty, invalid, empty_content = find_bad_json_from_jobs(
            args.job_file,
            root_results=args.root_results,
            root_inputs=args.root_inputs,
            max_workers=args.workers,
        )
    elif args.root:
        empty, invalid, empty_content = find_bad_json(
            args.root, depth=args.depth, max_workers=args.workers
        )
    else:
        p.error("Either --root or --job-file is required")

    all_bad = []

    if empty:
        print(f"\n--- Empty files (0 bytes): {len(empty)} ---")
        for p_ in sorted(empty):
            print(f"  {p_}")
        all_bad.extend(empty)

    if invalid:
        print(f"\n--- Invalid JSON (parse error): {len(invalid)} ---")
        for p_ in sorted(invalid):
            print(f"  {p_}")
        all_bad.extend(invalid)

    if empty_content:
        print(f"\n--- Empty content ({{}} or []): {len(empty_content)} ---")
        for p_ in sorted(empty_content):
            print(f"  {p_}")
        all_bad.extend(empty_content)

    total = len(all_bad)
    if total == 0:
        print("\nNo bad .json files found.")
    else:
        print(f"\nTotal bad .json files: {total}")

    if args.fix_timings:
        print("\n--- Fixing generator/timings.json files ---")
        fixed, repaired, skipped = _fix_timings(all_bad)
        print(f"Fixed (from fallback): {fixed}, Repaired (truncated): {repaired}, Skipped: {skipped}")

    if args.delete:
        if args.dry_run:
            print(f"\n--- Dry run: would delete {len(all_bad)} file(s) ---")
            for p_ in sorted(all_bad):
                print(f"  {p_}")
        else:
            print(f"\n--- Deleting {len(all_bad)} bad JSON file(s) ---")
            deleted = 0
            failed = 0
            for p_ in sorted(all_bad):
                try:
                    os.remove(p_)
                    print(f"  DELETED: {p_}")
                    deleted += 1
                except OSError as e:
                    print(f"  FAILED: {p_}  ({e})")
                    failed += 1
            print(f"Deleted: {deleted}, Failed: {failed}")

    if args.output and all_bad:
        with open(args.output, "w") as f:
            for path in sorted(all_bad):
                f.write(path + "\n")
        print(f"Paths written to: {args.output}")


if __name__ == "__main__":
    main()


