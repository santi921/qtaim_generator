"""critic2-qtaim: batch Critic2 bond-critical-point runs over job folders.

Runs run_critic2_analysis (orca.wfx -> critic2.json) for each folder in a
folder list, in parallel. Critic2 must be on PATH (or pass --critic2_cmd).

Example:
    critic2-qtaim --folder_list folders.txt --workers 6
"""

import argparse
import concurrent.futures
import logging
import shutil
import sys
from typing import List, Optional

from tqdm import tqdm

from qtaim_gen.source.core.critic2 import DEFAULT_POINTPROPS, run_critic2_analysis

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("critic2_qtaim")


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
    parser.add_argument("--critic2_cmd", default="critic2")
    parser.add_argument(
        "--discard",
        type=float,
        default=1e-5,
        help="prune CPs below this density (a.u.); Critic2 manual's molecular default",
    )
    parser.add_argument(
        "--pointprops",
        default=",".join(DEFAULT_POINTPROPS),
        help="comma-separated Critic2 POINTPROP shorthands",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=1800, help="per-folder seconds")
    parser.add_argument(
        "--overwrite", action="store_true", help="recompute existing critic2.json"
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="delete the .cri/.cro/cpreport intermediates after parsing",
    )
    args = parser.parse_args(argv)

    if shutil.which(args.critic2_cmd) is None:
        logger.error("critic2 not found on PATH: %s", args.critic2_cmd)
        return 2

    folders = read_folder_list(args.folder_list)
    if not folders:
        logger.error("no folders in %s", args.folder_list)
        return 2
    pointprops = tuple(p.strip() for p in args.pointprops.split(",") if p.strip())
    logger.info(
        "%d folders, %d workers, discard=%g, pointprops=%s",
        len(folders),
        args.workers,
        args.discard,
        ",".join(pointprops),
    )

    def run_one(folder: str) -> bool:
        return run_critic2_analysis(
            folder=folder,
            critic2_cmd=args.critic2_cmd,
            discard=args.discard,
            pointprops=pointprops,
            timeout=args.timeout,
            overwrite=args.overwrite,
            keep_intermediates=not args.clean,
            logger=logger,
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        results = list(
            tqdm(executor.map(run_one, folders), total=len(folders), desc="critic2")
        )

    n_ok = sum(results)
    logger.info("done: %d ok, %d failed", n_ok, len(results) - n_ok)
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
