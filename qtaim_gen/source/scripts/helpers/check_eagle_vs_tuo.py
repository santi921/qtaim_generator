#!/usr/bin/env python3
"""Cross-HPC integrity check: Twoalome corrupt dirs -> Eagle.

Reads a list of job folder paths that were found corrupt on Twoalome,
remaps them to their equivalent locations on Eagle, then checks whether
the compressed source files (*.tar.zst, *.gbw.zstd0) are also corrupt
or missing on Eagle.

Checks performed per folder:
  1. Folder exists on Eagle.
  2. At least one compressed file (*.tar.zst or *.gbw.zstd0) is present.
  3. Size > 0 bytes.
  4. Decompression succeeds (unzstd / tar).

Output:
  - Summary stats printed to stdout.
  - --output_ok   : Eagle paths that are healthy (candidates for Globus migration).
  - --output_bad  : Eagle paths that are also corrupt / missing.

Usage
-----
python check_eagle_vs_tuo.py \\
    --corrupt_file /lus/eagle/projects/generator/corrupt_dirs.txt \\
    --scratch      /lus/eagle/projects/generator/tmp \\
    --output_ok    /lus/eagle/projects/generator/eagle_ok.txt \\
    --output_bad   /lus/eagle/projects/generator/eagle_also_bad.txt
"""
import os
import shutil
import logging
import argparse
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional, List, Dict, Tuple

from tqdm import tqdm

logger = logging.getLogger("check_eagle_vs_tuo")

TUO_PREFIX = "/p/lustre5/bennion1/Omol2025-4M-DiversitySet"
EAGLE_PREFIX = "/lus/eagle/projects/OMol25"

_EXTS = (".tar.zst", ".gbw.zstd0")


# ---------------------------------------------------------------------------
# Path remapping
# ---------------------------------------------------------------------------

def remap_path(tuo_path: str, tuo_prefix: str, eagle_prefix: str) -> str:
    """Strip tuo_prefix from tuo_path and prepend eagle_prefix."""
    tuo_path = tuo_path.rstrip("/")
    tuo_prefix = tuo_prefix.rstrip("/")
    if tuo_path.startswith(tuo_prefix):
        suffix = tuo_path[len(tuo_prefix):]
        return eagle_prefix.rstrip("/") + suffix
    # Path doesn't start with expected prefix - return as-is and let the
    # caller deal with a missing folder.
    logger.warning("Path does not start with expected prefix: %s", tuo_path)
    return tuo_path


# ---------------------------------------------------------------------------
# Per-file integrity check
# ---------------------------------------------------------------------------

def _check_file(fp: str, scratch: Optional[str], timeout: int) -> Optional[str]:
    """Return None if OK, or a short reason string if bad."""
    try:
        size = os.path.getsize(fp)
    except OSError as e:
        return f"cannot stat: {e}"

    if size == 0:
        return "empty (0 bytes)"

    tmpdir = tempfile.mkdtemp(prefix="eagle_check_", dir=scratch)
    try:
        fname = os.path.basename(fp)

        if fname.endswith(".gbw.zstd0"):
            out_file = os.path.join(tmpdir, fname[: -len(".zstd0")])
            cmd = ["unzstd", "-o", out_file, "-f", fp]
            proc = subprocess.run(cmd, capture_output=True, timeout=timeout)
            if proc.returncode != 0:
                stderr = proc.stderr.decode(errors="replace").strip()
                return f"unzstd failed (rc={proc.returncode}): {stderr[:200]}"

        elif fname.endswith(".tar.zst"):
            # Step 1: decompress .tar.zst -> .tar
            tar_path = os.path.join(tmpdir, fname[: -len(".zst")])
            decomp = subprocess.run(
                ["unzstd", "-o", tar_path, "-f", fp],
                capture_output=True,
                timeout=timeout,
            )
            if decomp.returncode != 0:
                stderr = decomp.stderr.decode(errors="replace").strip()
                return f"unzstd failed (rc={decomp.returncode}): {stderr[:200]}"
            # Step 2: list tar contents (no extraction needed for integrity)
            tar_check = subprocess.run(
                ["tar", "-tf", tar_path],
                capture_output=True,
                timeout=timeout,
            )
            if tar_check.returncode != 0:
                stderr = tar_check.stderr.decode(errors="replace").strip()
                return f"tar check failed (rc={tar_check.returncode}): {stderr[:200]}"

        else:
            return None  # unexpected extension, skip

    except subprocess.TimeoutExpired:
        return f"timed out after {timeout}s"
    except FileNotFoundError as e:
        return f"decompression tool not found: {e}"
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    return None  # all good


# ---------------------------------------------------------------------------
# Per-folder check
# ---------------------------------------------------------------------------

def check_folder(
    eagle_path: str,
    scratch: Optional[str],
    timeout: int,
) -> Tuple[str, Dict[str, str]]:
    """Check one folder on Eagle.

    Returns:
        (status, bad_files)

        status is one of:
          'not_found'    - folder does not exist on Eagle
          'no_compressed'- folder exists but has no *.tar.zst / *.gbw.zstd0
          'bad'          - one or more compressed files failed
          'ok'           - all compressed files passed

        bad_files maps filename -> failure reason (empty when status != 'bad')
    """
    if not os.path.isdir(eagle_path):
        return "not_found", {}

    try:
        items = os.listdir(eagle_path)
    except OSError as e:
        logger.warning("Cannot list %s: %s", eagle_path, e)
        return "not_found", {}

    compressed = [f for f in items if any(f.endswith(ext) for ext in _EXTS)]

    if not compressed:
        return "no_compressed", {}

    bad: Dict[str, str] = {}
    for fname in compressed:
        fp = os.path.join(eagle_path, fname)
        reason = _check_file(fp, scratch, timeout)
        if reason is not None:
            bad[fname] = reason

    if bad:
        return "bad", bad
    return "ok", {}


# ---------------------------------------------------------------------------
# Worker (top-level for pickling)
# ---------------------------------------------------------------------------

def _worker(args: Tuple[str, str, str, Optional[str], int]) -> Tuple[str, str, Dict[str, str]]:
    """Map one tuo_path -> (eagle_path, status, bad_files). No logging."""
    tuo_path, tuo_prefix, eagle_prefix, scratch, timeout = args
    eagle_path = remap_path(tuo_path, tuo_prefix, eagle_prefix)
    status, bad_files = check_folder(eagle_path, scratch, timeout)
    return eagle_path, status, bad_files


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="check-eagle-vs-tuo",
        description=(
            "Check whether job folders flagged as corrupt on Twoalome are "
            "also corrupt (or missing) on Eagle. Remaps paths automatically."
        ),
    )
    parser.add_argument(
        "--corrupt_file",
        required=True,
        help="File containing Twoalome paths that were found corrupt (one per line)",
    )
    parser.add_argument(
        "--tuo_prefix",
        default=TUO_PREFIX,
        help=f"Twoalome path prefix to strip (default: {TUO_PREFIX})",
    )
    parser.add_argument(
        "--eagle_prefix",
        default=EAGLE_PREFIX,
        help=f"Eagle path prefix to prepend (default: {EAGLE_PREFIX})",
    )
    parser.add_argument(
        "--scratch",
        default=None,
        help="Scratch dir for temp decompression (use fast local path on HPC)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Per-file decompress timeout in seconds (default: 120)",
    )
    parser.add_argument(
        "--output_ok",
        default="eagle_ok.txt",
        help="Write Eagle paths that are healthy here (default: eagle_ok.txt)",
    )
    parser.add_argument(
        "--output_bad",
        default="eagle_also_bad.txt",
        help="Write Eagle paths that are also corrupt/missing here (default: eagle_also_bad.txt)",
    )
    parser.add_argument(
        "--num_folders",
        type=int,
        default=-1,
        help="Limit scan to first N folders (-1 = all)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel worker processes (default: 8)",
    )
    parser.add_argument(
        "--log_file",
        default=None,
        help="Log to file instead of stdout",
    )
    args = parser.parse_args(argv)

    handler = (
        logging.FileHandler(args.log_file)
        if args.log_file
        else logging.StreamHandler()
    )
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logging.basicConfig(level=logging.INFO, handlers=[handler])

    if not os.path.exists(args.corrupt_file):
        logger.error("corrupt_file not found: %s", args.corrupt_file)
        return 2

    with open(args.corrupt_file) as fh:
        tuo_paths = [ln.strip() for ln in fh if ln.strip()]

    if args.num_folders > 0:
        tuo_paths = tuo_paths[: args.num_folders]

    logger.info(
        "Loaded %d paths from %s", len(tuo_paths), args.corrupt_file
    )

    counts = {
        "not_found": 0,
        "no_compressed": 0,
        "bad": 0,
        "ok": 0,
    }
    ok_paths: List[str] = []
    bad_paths: List[str] = []  # covers not_found, no_compressed, bad

    worker_args = [
        (p, args.tuo_prefix, args.eagle_prefix, args.scratch, args.timeout)
        for p in tuo_paths
    ]

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(_worker, a): a for a in worker_args}
        for future in tqdm(as_completed(futures), total=len(futures), desc="checking folders", unit="folder"):
            eagle_path, status, bad_files = future.result()
            counts[status] += 1

            if status == "ok":
                ok_paths.append(eagle_path)
                logger.info("OK         %s", eagle_path)
            else:
                bad_paths.append(eagle_path)
                if status == "not_found":
                    logger.info("NOT FOUND  %s", eagle_path)
                elif status == "no_compressed":
                    logger.info("NO FILES   %s", eagle_path)
                else:
                    for fname, reason in bad_files.items():
                        logger.info("BAD        %s/%s : %s", eagle_path, fname, reason)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    total = len(tuo_paths)
    print()
    print("=" * 60)
    print("Eagle integrity check - summary")
    print("=" * 60)
    print(f"  Total folders from corrupt_dirs.txt : {total}")
    print(f"  Not found on Eagle                  : {counts['not_found']}")
    print(f"  Found, no compressed files present  : {counts['no_compressed']}")
    print(f"  Found, also corrupt/failed on Eagle  : {counts['bad']}")
    print(f"  Found, healthy on Eagle (migrate?)   : {counts['ok']}")
    print("=" * 60)
    print(f"  Healthy Eagle paths -> {args.output_ok}")
    print(f"  Problem Eagle paths -> {args.output_bad}")
    print("=" * 60)

    with open(args.output_ok, "w") as fh:
        for p in ok_paths:
            fh.write(p + "\n")

    with open(args.output_bad, "w") as fh:
        for p in bad_paths:
            fh.write(p + "\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
