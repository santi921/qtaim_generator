#!/usr/bin/env python3
"""Re-run the ORCA .out parse for job folders whose orca.json is stale.

A folder is stale when its orca.json carries an ``orca_parser_version`` below
the current parser (absent key = version 1). For every stale folder the driver
locates an ``orca.out`` and calls ``_run_orca_parse`` from ``core/omol.py``,
which parses, rewrites orca.json, re-merges ORCA charges and bond orders into
charge.json / bond.json (idempotent), and records the timing.

Where orca.out comes from, in order:

1. ``<folder>/orca.out`` (or ``output.out``) already on disk.
2. ``<folder>/orca.tar.zst`` (extracted by ``_run_orca_parse`` itself, copy
   removed after the parse).
3. With ``--source_root``: the mirrored folder
   ``<source_root>/<relpath(folder, root_dir)>`` holding ``orca.out`` or
   ``orca.tar.zst`` (the layout ``clean_omol --purge-orca`` leaves behind).
   The file is staged into the job folder by symlink, parsed, and the staged
   copy removed.

After the campaign rebuild only orca.lmdb::

    json-to-lmdb --folder_list FILE ... --data_types orca

charge.lmdb and bond.lmdb do not change: the merged ``*_orca`` keys are the
same in version 1 and 2.

Usage:
    reparse-orca-json --folder_list jobs.txt --workers 32 --dry_run
    reparse-orca-json --folder_list jobs.txt --root_dir /res/OMol4M \\
        --source_root /src/OMol4M --workers 32 --report reparse.json
    reparse-orca-json --root_dir data/omol_full_test/omol_refs_no_clash/rmechdb
"""

import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from glob import glob
from typing import Dict, List, Optional

from qtaim_gen.source.core.parse_orca import (
    ORCA_PARSER_VERSION,
    find_orca_output_file,
    validate_parse_completeness,
)

STATUS_CURRENT = "current"
STATUS_REPARSED = "reparsed"
STATUS_PARTIAL = "partial"          # parsed, but truncated .out (no charges / energy)
STATUS_WOULD_REPARSE = "would_reparse"
STATUS_NO_SOURCE = "no_source"
STATUS_FAILED = "failed"


def _orca_json_path(folder: str, move_files: bool) -> Optional[str]:
    """Mirror _run_orca_parse / backfill_orca_into_json: generator/ first when
    move_files, then the folder root."""
    gen_path = os.path.join(folder, "generator", "orca.json")
    root_path = os.path.join(folder, "orca.json")
    if move_files and os.path.isfile(gen_path):
        return gen_path
    if os.path.isfile(root_path):
        return root_path
    if os.path.isfile(gen_path):
        return gen_path
    return None


def _orca_json_version(path: Optional[str]) -> Optional[int]:
    """Version of an existing orca.json; 1 when unversioned; None when absent
    or unreadable."""
    if not path:
        return None
    try:
        with open(path, "r") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(data, dict):
        return None
    return int(data.get("orca_parser_version", 1))


def _source_folder(folder: str, root_dir: Optional[str], source_root: Optional[str]) -> Optional[str]:
    if not source_root or not root_dir:
        return None
    rel = os.path.relpath(os.path.normpath(folder), os.path.normpath(root_dir))
    if rel == ".." or rel.startswith(".." + os.sep):
        return None
    return os.path.join(source_root, rel)


def locate_source(folder: str, source_folder: Optional[str]) -> Optional[str]:
    """Where orca.out can come from: folder_out, folder_archive, source_out,
    source_archive, or None."""
    if find_orca_output_file(folder):
        return "folder_out"
    if os.path.isfile(os.path.join(folder, "orca.tar.zst")):
        return "folder_archive"
    if source_folder and os.path.isdir(source_folder):
        if find_orca_output_file(source_folder):
            return "source_out"
        if os.path.isfile(os.path.join(source_folder, "orca.tar.zst")):
            return "source_archive"
    return None


def _stage_from_source(folder: str, source_folder: str, kind: str, logger: logging.Logger) -> List[str]:
    """Make orca.out available in *folder* from the mirrored source folder.
    Returns the paths created in *folder* that must be removed afterwards."""
    from qtaim_gen.source.core.omol import _extract_orca_out_from_archive

    created: List[str] = []
    if kind == "source_out":
        src = find_orca_output_file(source_folder)
        dst = os.path.join(folder, os.path.basename(src))
        os.symlink(os.path.abspath(src), dst)
        created.append(dst)
        return created
    # source_archive: link the archive in, reuse the pipeline's extraction
    link = os.path.join(folder, "orca.tar.zst")
    os.symlink(os.path.abspath(os.path.join(source_folder, "orca.tar.zst")), link)
    created.append(link)
    if _extract_orca_out_from_archive(folder, logger):
        created.append(os.path.join(folder, "orca.out"))
    return created


def _remove_staged(paths: List[str], logger: logging.Logger) -> None:
    for p in paths:
        try:
            if os.path.lexists(p):
                os.remove(p)
        except OSError as e:
            logger.warning("Could not remove staged %s: %s", p, e)


def _settle_orca_json_location(folder: str, move_files: bool) -> Optional[str]:
    """_run_orca_parse writes <folder>/orca.json. In the generator/ layout the
    live copy is <folder>/generator/orca.json, so move the fresh file there
    (replacing the stale one) and return the path that now holds the result."""
    root_path = os.path.join(folder, "orca.json")
    gen_dir = os.path.join(folder, "generator")
    if move_files and os.path.isdir(gen_dir) and os.path.isfile(root_path):
        gen_path = os.path.join(gen_dir, "orca.json")
        os.replace(root_path, gen_path)
        return gen_path
    return _orca_json_path(folder, move_files)


def process_folder(
    folder: str,
    move_files: bool,
    dry_run: bool,
    min_version: int,
    force: bool,
    root_dir: Optional[str] = None,
    source_root: Optional[str] = None,
) -> Dict[str, object]:
    result: Dict[str, object] = {
        "folder": folder,
        "status": "",
        "version_before": None,
        "version_after": None,
        "source": None,
        "error": "",
    }
    logger = logging.getLogger("reparse_orca_json")

    json_path = _orca_json_path(folder, move_files)
    version = _orca_json_version(json_path)
    result["version_before"] = version
    if version is not None and version >= min_version and not force:
        result["status"] = STATUS_CURRENT
        return result

    source_folder = _source_folder(folder, root_dir, source_root)
    kind = locate_source(folder, source_folder)
    result["source"] = kind
    if kind is None:
        result["status"] = STATUS_NO_SOURCE
        return result
    if dry_run:
        result["status"] = STATUS_WOULD_REPARSE
        return result

    from qtaim_gen.source.core.omol import _run_orca_parse

    staged: List[str] = []
    try:
        if kind in ("source_out", "source_archive"):
            staged = _stage_from_source(folder, source_folder, kind, logger)
            if find_orca_output_file(folder) is None:
                result["status"] = STATUS_FAILED
                result["error"] = f"could not stage orca.out from {source_folder}"
                return result
        _run_orca_parse(folder, move_files, logger)
        json_path = _settle_orca_json_location(folder, move_files)
        result["version_after"] = _orca_json_version(json_path)
        if result["version_after"] is None or result["version_after"] < min_version:
            result["status"] = STATUS_FAILED
            result["error"] = "orca.json not rewritten by the parser"
            return result
        with open(json_path, "r") as f:
            orca_dict = json.load(f)
        result["status"] = STATUS_REPARSED if validate_parse_completeness(orca_dict) else STATUS_PARTIAL
    except Exception as e:
        result["status"] = STATUS_FAILED
        result["error"] = f"{type(e).__name__}: {e}"
    finally:
        _remove_staged(staged, logger)
    return result


def discover_folders(root_dir: Optional[str], folder_list: Optional[str]) -> List[str]:
    if folder_list:
        with open(folder_list, "r") as f:
            raw = [line.strip() for line in f]
        candidates = [line for line in raw if line and not line.startswith("#")]
        return [p for p in candidates if os.path.isdir(p)]
    if root_dir:
        out = []
        for d in sorted(glob(os.path.join(root_dir, "*"))):
            if not os.path.isdir(d):
                continue
            if _orca_json_path(d, move_files=True) or find_orca_output_file(d) \
                    or os.path.isfile(os.path.join(d, "orca.tar.zst")):
                out.append(d)
        return out
    raise ValueError("Provide --root_dir or --folder_list")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root_dir", help="Root of the results tree. With --folder_list it is only used to "
                                       "compute the mirrored path under --source_root.")
    ap.add_argument("--folder_list", help="Text file with one absolute job folder path per line "
                                          "(blanks and '#' lines skipped).")
    ap.add_argument("--source_root", default=None,
                    help="Mirrored source tree holding orca.out / orca.tar.zst for folders that were "
                         "purged (clean_omol --purge-orca). Requires --root_dir.")
    ap.add_argument("--move_files", action="store_true", default=True,
                    help="JSON files live under <folder>/generator/ (default; OMol4M layout).")
    ap.add_argument("--no_move_files", dest="move_files", action="store_false",
                    help="Flat layout: json files at the job-folder root.")
    ap.add_argument("--min_version", type=int, default=ORCA_PARSER_VERSION,
                    help=f"Reparse folders below this orca_parser_version (default {ORCA_PARSER_VERSION}).")
    ap.add_argument("--force", action="store_true", help="Reparse even when orca.json is current.")
    ap.add_argument("--workers", type=int, default=0, help="Parallel workers (default: cpu_count, 1 disables).")
    ap.add_argument("--limit", type=int, default=None, help="Process at most N folders (debug).")
    ap.add_argument("--dry_run", action="store_true", help="Classify folders, write nothing.")
    ap.add_argument("--report", default=None, help="Write a JSON report of per-folder results.")
    ap.add_argument("--list_remaining", default=None,
                    help="Write the folders that are still stale after the run (no_source, partial, failed).")
    args = ap.parse_args()

    if not args.root_dir and not args.folder_list:
        ap.error("Provide --root_dir or --folder_list")
    if args.source_root and not args.root_dir:
        ap.error("--source_root requires --root_dir")

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

    folders = discover_folders(args.root_dir, args.folder_list)
    if args.limit is not None:
        folders = folders[: args.limit]
    n = len(folders)
    if n == 0:
        print("No job folders found.", file=sys.stderr)
        return 1

    workers = args.workers if args.workers > 0 else (os.cpu_count() or 1)
    workers = min(workers, n)
    print(f"Scanning {n} folders with {workers} worker(s); min_version={args.min_version}; "
          f"move_files={args.move_files}; dry_run={args.dry_run}; source_root={args.source_root}",
          file=sys.stderr)

    kwargs = dict(
        move_files=args.move_files, dry_run=args.dry_run, min_version=args.min_version,
        force=args.force, root_dir=args.root_dir, source_root=args.source_root,
    )
    results: List[Dict[str, object]] = []
    t0 = time.time()
    progress_every = max(1, n // 20)
    if workers <= 1:
        for i, folder in enumerate(folders):
            results.append(process_folder(folder, **kwargs))
            if (i + 1) % progress_every == 0 or (i + 1) == n:
                print(f"  {i + 1}/{n} ({(i + 1) / max(time.time() - t0, 1e-6):.1f}/s)", file=sys.stderr)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(process_folder, f, **kwargs): f for f in folders}
            done = 0
            for fut in as_completed(futs):
                results.append(fut.result())
                done += 1
                if done % progress_every == 0 or done == n:
                    print(f"  {done}/{n} ({done / max(time.time() - t0, 1e-6):.1f}/s)", file=sys.stderr)

    agg: Dict[str, object] = {"folders_total": n}
    for status in (STATUS_CURRENT, STATUS_WOULD_REPARSE, STATUS_REPARSED, STATUS_PARTIAL,
                   STATUS_NO_SOURCE, STATUS_FAILED):
        agg[status] = sum(1 for r in results if r["status"] == status)
    for kind in ("folder_out", "folder_archive", "source_out", "source_archive"):
        agg[f"source_{kind}"] = sum(1 for r in results if r["source"] == kind)
    agg["elapsed_sec"] = round(time.time() - t0, 2)

    print("\nReparse summary:", file=sys.stderr)
    for k, v in agg.items():
        print(f"  {k:<24} {v}", file=sys.stderr)
    failed = [r for r in results if r["status"] == STATUS_FAILED]
    if failed:
        print("\nFirst 10 failures:", file=sys.stderr)
        for r in failed[:10]:
            print(f"  {r['folder']}: {r['error']}", file=sys.stderr)

    if args.report:
        os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
        with open(args.report, "w") as f:
            json.dump({"aggregate": agg, "per_folder": results}, f, indent=2, default=str)
        print(f"\nReport written: {args.report}", file=sys.stderr)

    if args.list_remaining:
        os.makedirs(os.path.dirname(args.list_remaining) or ".", exist_ok=True)
        remaining = [r["folder"] for r in results
                     if r["status"] in (STATUS_NO_SOURCE, STATUS_PARTIAL, STATUS_FAILED, STATUS_WOULD_REPARSE)]
        with open(args.list_remaining, "w") as f:
            for p in remaining:
                f.write(f"{p}\n")
        print(f"List of {len(remaining)} folders still stale: {args.list_remaining}", file=sys.stderr)

    return 0 if not failed else 2


if __name__ == "__main__":
    sys.exit(main())
