#!/usr/bin/env python3
"""Stage healthy compressed files for Globus transfer.

Reads eagle_ok.txt (output of check_eagle_vs_tuo.py), copies only the
compressed source files (*.tar.zst, *.gbw.zstd0) from each folder into
a staging directory, preserving the path structure relative to the Eagle
root so the Globus transfer destination maps directly back to Twoalome.

Example layout produced under --dest_dir:
  <dest_dir>/5A_elytes/625_BC4O8-1_5_group_45_shell_51_0_1/orca.tar.zst
  <dest_dir>/omol/metal_organics/restart5to6/job_.../orca.gbw.zstd0

Usage
-----
python stage_ok_for_transfer.py \\
    --ok_file   /lus/eagle/projects/generator/eagle_ok.txt \\
    --dest_dir  /tmp/eagle_ok \\
    --workers   16
"""
import os
import shutil
import argparse
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Tuple

from tqdm import tqdm

logger = logging.getLogger("stage_ok_for_transfer")

EAGLE_PREFIX = "/lus/eagle/projects/OMol25"
_EXTS = (".tar.zst", ".gbw.zstd0")


def _copy_folder(
    eagle_path: str,
    eagle_prefix: str,
    dest_dir: str,
) -> Tuple[str, int, List[str]]:
    """Copy compressed files from one folder into dest_dir.

    Returns:
        (eagle_path, n_copied, errors)
    """
    eagle_prefix = eagle_prefix.rstrip("/")
    if eagle_path.startswith(eagle_prefix):
        rel = eagle_path[len(eagle_prefix):].lstrip("/")
    else:
        # fallback: use the last two path components as relative path
        parts = eagle_path.rstrip("/").split("/")
        rel = os.path.join(*parts[-2:]) if len(parts) >= 2 else parts[-1]

    dest_folder = os.path.join(dest_dir, rel)
    os.makedirs(dest_folder, exist_ok=True)

    try:
        items = os.listdir(eagle_path)
    except OSError as e:
        return eagle_path, 0, [f"cannot list: {e}"]

    compressed = [f for f in items if any(f.endswith(ext) for ext in _EXTS)]

    errors: List[str] = []
    n_copied = 0
    for fname in compressed:
        src = os.path.join(eagle_path, fname)
        dst = os.path.join(dest_folder, fname)
        try:
            shutil.copy2(src, dst)
            n_copied += 1
        except OSError as e:
            errors.append(f"{fname}: {e}")

    return eagle_path, n_copied, errors


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="stage-ok-for-transfer",
        description=(
            "Copy compressed files from eagle_ok.txt folders into a staging "
            "directory, preserving relative path structure for Globus transfer."
        ),
    )
    parser.add_argument(
        "--ok_file",
        required=True,
        help="eagle_ok.txt produced by check_eagle_vs_tuo.py (Eagle paths, one per line)",
    )
    parser.add_argument(
        "--dest_dir",
        required=True,
        help="Staging directory to copy files into (e.g. /tmp/eagle_ok)",
    )
    parser.add_argument(
        "--eagle_prefix",
        default=EAGLE_PREFIX,
        help=f"Eagle root to strip when building relative paths (default: {EAGLE_PREFIX})",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=16,
        help="Number of parallel copy threads (default: 16)",
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

    if not os.path.exists(args.ok_file):
        logger.error("ok_file not found: %s", args.ok_file)
        return 2

    with open(args.ok_file) as fh:
        eagle_paths = [ln.strip() for ln in fh if ln.strip()]

    logger.info("Staging %d folders -> %s", len(eagle_paths), args.dest_dir)
    os.makedirs(args.dest_dir, exist_ok=True)

    total_copied = 0
    total_errors = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(_copy_folder, p, args.eagle_prefix, args.dest_dir): p
            for p in eagle_paths
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="copying", unit="folder"):
            eagle_path, n_copied, errors = future.result()
            total_copied += n_copied
            if errors:
                total_errors += len(errors)
                for err in errors:
                    logger.warning("COPY ERROR %s/%s", eagle_path, err)
            elif n_copied == 0:
                logger.warning("NO FILES   %s", eagle_path)

    print()
    print("=" * 60)
    print("Staging complete")
    print("=" * 60)
    print(f"  Folders processed : {len(eagle_paths)}")
    print(f"  Files copied      : {total_copied}")
    print(f"  Copy errors       : {total_errors}")
    print(f"  Staged under      : {args.dest_dir}")
    print("=" * 60)
    print()
    print("Globus transfer tip:")
    print(f"  source endpoint dir : {args.dest_dir}/")
    print(f"  dest endpoint dir   : /p/lustre5/bennion1/Omol2025-4M-DiversitySet/")
    print("  Use recursive transfer to preserve the subdirectory structure.")
    print("=" * 60)

    return 0 if total_errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
