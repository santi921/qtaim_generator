#!/usr/bin/env python3
import os
import logging
import argparse
from typing import Optional, List
from qtaim_gen.source.utils.io import get_folders_from_file

should_stop = False

logger = logging.getLogger("refine_list_of_jobs")


def handle_signal(signum, _frame):
    global should_stop
    print(f"Received signal {signum}, initiating graceful shutdown...")
    should_stop = True


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point for launching a batch of Parsl jobs.

    Args:
        argv: Optional list of CLI args (excludes program name). If None,
            arguments are read from sys.argv.

    Returns:
        Exit code (0 on success).
    """

    parser = argparse.ArgumentParser(
        prog="job-filter",
        description=(
            "Filter and refine a list of job folders for QTAIM analysis. "
            "This script processes the list of folders in --job_file and outputs a refined list."
        ),
    )

    parser.add_argument(
        "--num_folders",
        type=int,
        default=-1,
        help="number of folders to check and try to run",
    )

    parser.add_argument(
        "--job_file",
        type=str,
        help="file containing list of folders to run analysis on",
        default="./out.txt",
    )

    parser.add_argument(
        "--refined_job_file",
        type=str,
        help="file to save refined list of folders",
        default="refined_jobs.txt",
    )

    parser.add_argument(
        "--full_set",
        type=int,
        default=0,
        help="level of calculation detail (0-baseline, 1-modest, 2-full)",
    )

    parser.add_argument(
        "--root_omol_results",
        type=str,
        default=None,
        help="root where to store results, should mimic root_omol_inputs",
    )

    parser.add_argument(
        "--root_omol_inputs",
        type=str,
        default=None,
        help="root where input folders are located",
    )

    parser.add_argument(
        "--move_results",
        action="store_true",
        help="whether to move results from temp to final location",
    )

    parser.add_argument(
        "--check_orphaned",
        action="store_true",
        help="whether to check for orphaned jobs",
    )

    parser.add_argument(
        "--check_orca",
        action="store_true",
        help="require orca.json during validation (for retroactive ORCA .out parsing)",
    )

    parser.add_argument(
        "--log_file",
        type=str,
        default=None,
        help="path to log file; if set, logger writes to file instead of stdout",
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="suppress all log output",
    )

    parser.add_argument(
        "--check_ecp",
        action="store_true",
        help="filter out jobs where ECP loaded successfully; keep only ECP_FAILED and ECP_NO_ZIP jobs",
    )

    args = parser.parse_args(argv)

    log_file: Optional[str] = getattr(args, "log_file", None)
    quiet: bool = bool(getattr(args, "quiet", False))
    if quiet:
        handler = logging.NullHandler()
    elif log_file:
        handler = logging.FileHandler(log_file)
        handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    else:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logging.basicConfig(level=logging.INFO, handlers=[handler])

    for key, value in vars(args).items():
        logger.info(f"{key}: {value}")

    full_set: int = int(getattr(args, "full_set", 0))
    move_results: bool = bool(getattr(args, "move_results", False))
    job_file: str = getattr(args, "job_file")
    orphaned_check: bool = bool(getattr(args, "check_orphaned", False))
    check_orca: bool = bool(getattr(args, "check_orca", False))
    check_ecp: bool = bool(getattr(args, "check_ecp", False))
    refined_job_file: str = getattr(args, "refined_job_file", "refined_jobs.txt")
    root_omol_results: Optional[str] = getattr(args, "root_omol_results", None)
    root_omol_inputs: Optional[str] = getattr(args, "root_omol_inputs", None)
    # if not set default to the entire length of the job file

    num_folders: int = int(getattr(args, "num_folders", -1))
    if num_folders <= 0:
        with open(job_file, "r") as f:
            num_folders = sum(1 for _ in f)

    # Basic static checks before launching heavy work
    if not os.path.exists(job_file):
        logger.error(f"job_file '{job_file}' does not exist")
        return 2

    folders_run = get_folders_from_file(
        job_file,
        num_folders,
        root_omol_results,
        root_omol_inputs,
        pre_validate=True,
        move_results=move_results,
        full_set=full_set,
        check_orca=check_orca,
        check_ecp=check_ecp,
        logger=logger,
    )

    if not folders_run:
        msg = f"No folders found in {job_file}"
        logger.info(msg)
        print(msg)
        return []

    summary = f"{len(folders_run)} folders remaining after refinement -> {refined_job_file}"
    logger.info(summary)
    print(summary)

    # save refined job file
    with open(refined_job_file, "w") as f:
        for folder in folders_run:
            f.write(f"{folder}\n")

    if orphaned_check:
        count_orphaned = 0
        for folder_inputs in folders_run:
            # find jobs in root_omol_results that have
            if (
                root_omol_inputs
                and root_omol_results
                and folder_inputs.startswith(root_omol_inputs)
            ):
                folder_relative = folder_inputs[len(root_omol_inputs) :].lstrip(os.sep)
                folder_outputs = root_omol_results + os.sep + folder_relative
            else:
                folder_outputs = folder_inputs
            # check if folder exists with more than 3 files
            if os.path.exists(folder_outputs):
                # check if there is no wavefunction file in the folder and that there is no gbw_analysis.log file
                from qtaim_gen.source.utils.io import has_wavefunction_file

                if not has_wavefunction_file(folder_outputs) and os.path.exists(
                    os.path.join(folder_outputs, "gbw_analysis.log")
                ):
                    logger.info(
                        f"Orphaned job detected: {folder_outputs} missing wfn/wfx w log present"
                    )
                    count_orphaned += 1

        logger.info(f"Total orphaned jobs detected: {count_orphaned}")


if __name__ == "__main__":
    raise SystemExit(main())
