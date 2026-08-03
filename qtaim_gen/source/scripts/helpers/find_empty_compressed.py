#!/usr/bin/env python3
"""Scan a job-list file for folders that contain bad compressed source files.

Two checks per compressed file (*.gbw.zstd0, *.tar.zst, *.tgz):
  1. Size check  - fail immediately if 0 bytes.
  2. Decompress  - extract to a temp dir and verify the command succeeds.
     Skipped when --no_decompress_check is passed.

Exit codes: 0 = scan complete, 2 = job_file not found.

Usage
-----
find-empty-compressed \\
    --job_file /path/to/jobs.txt \\
    --output bad_compressed_jobs.txt \\
    --scratch /tmp \\
    --timeout 120
"""
import os
import shutil
import logging
import argparse
import subprocess
import tempfile
from typing import Optional, List, Dict

logger = logging.getLogger("find_empty_compressed")

_COMPRESSED_EXTS = (".gbw.zstd0", ".tar.zst", ".tgz")


def _check_file(fp: str, scratch: Optional[str], timeout: int) -> Optional[str]:
    """Return None if the file is OK, or a short reason string if it is bad.

    Checks:
      - size == 0  -> "empty (0 bytes)"
      - decompress fails or times out -> reason from stderr / timeout
    """
    # --- size check ---
    try:
        size = os.path.getsize(fp)
    except OSError as e:
        return f"cannot stat: {e}"

    if size == 0:
        return "empty (0 bytes)"

    # --- decompress check ---
    tmpdir = tempfile.mkdtemp(prefix="qtaim_check_", dir=scratch)
    try:
        fname = os.path.basename(fp)
        if fname.endswith(".gbw.zstd0"):
            out_file = os.path.join(tmpdir, fname[: -len(".zstd0")])
            cmd = ["unzstd", "-o", out_file, "-f", fp]
        elif fname.endswith(".tar.zst"):
            tar_path = os.path.join(tmpdir, fname[: -len(".zst")])
            decomp = subprocess.run(
                ["unzstd", "-o", tar_path, "-f", fp],
                capture_output=True,
                timeout=timeout,
            )
            if decomp.returncode != 0:
                stderr = decomp.stderr.decode(errors="replace").strip()
                return f"decompress failed (rc={decomp.returncode}): {stderr[:200]}"
            cmd = ["tar", "-xf", tar_path, "-C", tmpdir]
        elif fname.endswith(".tgz"):
            cmd = ["tar", "-xzf", fp, "-C", tmpdir]
        else:
            return None  # unknown extension - skip decompress check

        proc = subprocess.run(
            cmd,
            capture_output=True,
            timeout=timeout,
        )
        if proc.returncode != 0:
            stderr = proc.stderr.decode(errors="replace").strip()
            return f"decompress failed (rc={proc.returncode}): {stderr[:200]}"

    except subprocess.TimeoutExpired:
        return f"decompress timed out after {timeout}s"
    except FileNotFoundError as e:
        return f"decompression tool not found: {e}"
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    return None  # all good


def check_folder(
    folder: str,
    scratch: Optional[str],
    timeout: int,
    skip_decompress: bool,
) -> Dict[str, str]:
    """Return a dict mapping filename -> reason for each bad compressed file in *folder*.

    Empty dict means all files are OK (or no compressed files found).
    """
    bad: Dict[str, str] = {}
    if not os.path.isdir(folder):
        return bad

    try:
        items = os.listdir(folder)
    except OSError as e:
        logger.warning("Could not list %s: %s", folder, e)
        return bad

    for item in items:
        if not any(item.endswith(ext) for ext in _COMPRESSED_EXTS):
            continue

        fp = os.path.join(folder, item)

        if skip_decompress:
            # size-only path
            try:
                size = os.path.getsize(fp)
            except OSError as e:
                bad[item] = f"cannot stat: {e}"
                continue
            if size == 0:
                bad[item] = "empty (0 bytes)"
        else:
            reason = _check_file(fp, scratch, timeout)
            if reason is not None:
                bad[item] = reason

    return bad


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="find-empty-compressed",
        description=(
            "Scan a job-list file for input folders that contain bad compressed "
            "files (*.gbw.zstd0, *.tar.zst, *.tgz). "
            "Each file is size-checked, then decompressed into a temp dir to verify "
            "integrity. Affected folder paths are written to --output."
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
        default="bad_compressed_jobs.txt",
        help="path to write the list of affected folders (default: bad_compressed_jobs.txt)",
    )

    parser.add_argument(
        "--root_omol_inputs",
        type=str,
        default=None,
        help="prepended to relative paths in job_file (normally paths are already absolute)",
    )

    parser.add_argument(
        "--num_folders",
        type=int,
        default=-1,
        help="maximum number of folders to scan (-1 = all)",
    )

    parser.add_argument(
        "--scratch",
        type=str,
        default=None,
        help=(
            "directory to use as parent for temp extraction dirs "
            "(default: system tempdir). Use a fast local path on HPC."
        ),
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="per-file decompress timeout in seconds (default: 120)",
    )

    parser.add_argument(
        "--no_decompress_check",
        action="store_true",
        help="skip the decompress step; only check file size",
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
    scratch: Optional[str] = args.scratch
    timeout: int = args.timeout
    skip_decompress: bool = args.no_decompress_check

    if not os.path.exists(job_file):
        logger.error("job_file '%s' does not exist", job_file)
        return 2

    with open(job_file) as fh:
        lines = [ln.strip() for ln in fh if ln.strip()]

    if num_folders > 0:
        lines = lines[:num_folders]

    mode = "size-only" if skip_decompress else f"size + decompress (timeout={timeout}s)"
    logger.info("Scanning %d folders from %s [%s]", len(lines), job_file, mode)

    affected: List[str] = []
    n_empty = 0
    n_corrupt = 0

    for folder_path in lines:
        if root_omol_inputs and not os.path.isabs(folder_path):
            folder_path = os.path.join(root_omol_inputs, folder_path)

        bad = check_folder(folder_path, scratch, timeout, skip_decompress)
        if bad:
            for fname, reason in bad.items():
                logger.info("BAD %s/%s : %s", folder_path, fname, reason)
                if "empty" in reason:
                    n_empty += 1
                else:
                    n_corrupt += 1
            affected.append(folder_path)

    summary = (
        f"{len(affected)} / {len(lines)} folders have bad compressed files "
        f"({n_empty} empty, {n_corrupt} corrupt/failed) -> {output}"
    )
    logger.info(summary)
    print(summary)

    with open(output, "w") as fh:
        for p in affected:
            fh.write(p + "\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
