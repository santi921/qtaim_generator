"""horton-charges: batch HORTON charge computation over job folders.

Runs run_horton_analysis (wfx -> horton.json + charge.json merge) for each
folder in a folder list, in parallel. The HORTON stack lives in its own
python environment; pass its interpreter via --horton_python.

Example:
    horton-charges --folder_list folders.txt \
        --horton_python ~/miniconda3/envs/horton/bin/python
"""

import argparse
import concurrent.futures
import logging
import os
import sys
from typing import List, Optional

from tqdm import tqdm

from qtaim_gen.source.core.horton import run_horton_analysis

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("horton_charges")


def read_folder_list(path: str) -> List[str]:
    folders = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                folders.append(line)
    return folders


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--folder_list",
        required=True,
        help="file with one absolute job-folder path per line (# comments ok)",
    )
    parser.add_argument(
        "--horton_python",
        required=True,
        help="python interpreter of the horton environment",
    )
    parser.add_argument("--schemes", default="becke,hirshfeld,is")
    parser.add_argument("--grid", default="fine", help="MolGrid preset")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=3600, help="per-folder seconds")
    parser.add_argument(
        "--overwrite", action="store_true", help="recompute existing horton.json"
    )
    args = parser.parse_args(argv)

    if not os.path.isfile(args.horton_python):
        logger.error("horton_python not found: %s", args.horton_python)
        return 2

    folders = read_folder_list(args.folder_list)
    if not folders:
        logger.error("no folders in %s", args.folder_list)
        return 2
    logger.info("%d folders, %d workers, schemes=%s", len(folders), args.workers, args.schemes)

    def run_one(folder: str) -> bool:
        return run_horton_analysis(
            folder=folder,
            horton_python=args.horton_python,
            schemes=args.schemes,
            grid=args.grid,
            timeout=args.timeout,
            overwrite=args.overwrite,
            logger=logger,
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        results = list(
            tqdm(executor.map(run_one, folders), total=len(folders), desc="horton")
        )

    n_ok = sum(results)
    logger.info("done: %d ok, %d failed", n_ok, len(results) - n_ok)
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
