#!/usr/bin/env python3
"""Scan a job-list file for folders that contain empty compressed source files.

Compressed files checked: *.gbw.zstd0, *.tar.zst, *.tgz

Usage
-----
find-empty-compressed --job_file /path/to/jobs.txt \
    --root_omol_inputs /p/lustre5/data/ \
    --output empty_compressed_jobs.txt
"""
import os
import logging
import argparse
from typing import Optional, List

logger = logging.getLogger("find_empty_compressed")

_COMPRESSED_EXTS = (".gbw.zstd0", ".tar.zst", ".tgz")


def has_empty_compressed(folder: str) -> List[str]:
    """Return list of empty compressed file names found in *folder*.

    Returns an empty list if none found or folder does not exist.
    """
    if not os.path.isdir(folder):
        return []
    empties = []
    try:
        for item in os.listdir(folder):
            if any(item.endswith(ext) for ext in _COMPRESSED_EXTS):
                fp = os.path.join(folder, item)
                try:
                    if os.path.getsize(fp) == 0:
                        empties.append(item)
                except OSError as e:
                    logger.warning("Could not stat %s: %s", fp, e)
    except OSError as e:
        logger.warning("Could not list %s: %s", folder, e)
    return empties


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="find-empty-compressed",
        description=(
            "Scan a job-list file for input folders that contain empty compressed "
            "files (*.gbw.zstd0, *.tar.zst, *.tgz). "
            "Writes the affected folder paths to --output."
        ),
    )

    parser.add_argument(
        "--job_file",
        type=str,
        required=True,
        help="file containing list of job folders (one per line)",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="empty_compressed_jobs.txt",
        help="path to write the list of affected folders (default: empty_compressed_jobs.txt)",
    )

    parser.add_argument(
        "--root_omol_inputs",
        type=str,
        default=None,
        help="root of the input data tree; each line in job_file is relative to this root",
    )

    parser.add_argument(
        "--num_folders",
        type=int,
        default=-1,
        help="maximum number of folders to scan (-1 = all)",
    )

    parser.add_argument(
        "--log_file",
        type=str,
        default=None,
        help="path to log file; if set, logger writes to file instead of stdout",
    )

    args = parser.parse_args(argv)

    log_file: Optional[str] = args.log_file
    handler = logging.FileHandler(log_file) if log_file else logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logging.basicConfig(level=logging.INFO, handlers=[handler])

    job_file: str = args.job_file
    output: str = args.output
    root_omol_inputs: Optional[str] = args.root_omol_inputs
    num_folders: int = args.num_folders

    if not os.path.exists(job_file):
        logger.error("job_file '%s' does not exist", job_file)
        return 2

    with open(job_file) as fh:
        lines = [ln.strip() for ln in fh if ln.strip()]

    if num_folders > 0:
        lines = lines[:num_folders]

    logger.info("Scanning %d folders from %s", len(lines), job_file)

    affected: List[str] = []
    for folder_path in lines:
        # job_file lines are normally absolute paths; root_omol_inputs only needed
        # if relative paths are present in the file
        if root_omol_inputs and not os.path.isabs(folder_path):
            folder_path = os.path.join(root_omol_inputs, folder_path)

        empties = has_empty_compressed(folder_path)
        if empties:
            logger.info("Empty compressed files in %s: %s", folder_path, empties)
            affected.append(folder_path)

    logger.info(
        "%d / %d folders have empty compressed files -> %s",
        len(affected),
        len(lines),
        output,
    )
    print(f"{len(affected)} folders with empty compressed files written to {output}")

    with open(output, "w") as fh:
        for p in affected:
            fh.write(p + "\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
