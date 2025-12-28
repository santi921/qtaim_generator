#!/usr/bin/env python3
import os
import argparse
from typing import Optional, List
from qtaim_gen.source.utils.io import get_folders_from_file

should_stop = False
def handle_signal(signum, frame):
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

    args = parser.parse_args(argv)
    # print(args)
    for key, value in vars(args).items():
        print(f"{key}: {value}")
    # Safely extract boolean flags and other values using getattr
    # overrun_running: bool = bool(getattr(args, "overrun_running", False))

    full_set: int = int(getattr(args, "full_set", 0))
    move_results: bool = bool(getattr(args, "move_results", False))
    job_file: str = getattr(args, "job_file")
    refined_job_file: str = getattr(args, "refined_job_file", "refined_jobs.txt")
    root_omol_results: Optional[str] = getattr(args, "root_omol_results", None)
    root_omol_inputs: Optional[str] = getattr(args, "root_omol_inputs", None)
    # if not set default to the entire length of the job file
    
    num_folders: int = int(getattr(args, "num_folders", -1))
    if num_folders <= 0:
        with open(job_file, "r") as f:
            num_folders = sum(1 for _ in f)

    # set mem
    # Basic static checks before launching heavy work
    if not os.path.exists(job_file):
        print(f"Error: job_file '{job_file}' does not exist")
        return 2

    
    folders_run = get_folders_from_file(
        job_file, 
        num_folders, 
        root_omol_results,
        root_omol_inputs,
        pre_validate=True, 
        move_results=move_results, 
        full_set=full_set
    )

    if not folders_run:
        print(f"No folders found in {job_file}")
        return []

    # save refined job file
    with open(refined_job_file, "w") as f:
        for folder in folders_run:
            f.write(f"{folder}\n")
    

if __name__ == "__main__":
    raise SystemExit(main())

"""
python refine_list_of_jobs.py --root_omol_inputs /lus/eagle/projects/OMol25/ \
--job_file /lus/eagle/projects/generator/jobs_by_topdir/orbnet_denali.txt --move_results \
--refined_job_file /lus/eagle/projects/generator/jobs_by_topdir/orbnet_denali_refined.txt \
    --full_set 0 --root_omol_results /lus/eagle/projects/generator/OMol25_postprocessing/

python refine_list_of_jobs.py --root_omol_inputs /lus/eagle/projects/OMol25/ \
--job_file /lus/eagle/projects/generator/jobs_by_topdir/trans1x.txt --move_results \
--refined_job_file /lus/eagle/projects/generator/jobs_by_topdir/trans1x_refined.txt \
    --full_set 0 --root_omol_results /lus/eagle/projects/generator/OMol25_postprocessing/


    python refine_list_of_jobs.py --root_omol_inputs /lus/eagle/projects/OMol25/ \
--job_file /lus/eagle/projects/generator/jobs_by_topdir/ani1xbb.txt --move_results \
--refined_job_file /lus/eagle/projects/generator/jobs_by_topdir/ani1xbb_refined.txt \
    --full_set 0 --root_omol_results /lus/eagle/projects/generator/OMol25_postprocessing/




python refine_list_of_jobs.py --root_omol_inputs /lus/eagle/projects/OMol25/ \
--job_file /lus/eagle/projects/generator/jobs_by_topdir/droplet.txt --move_results \
--refined_job_file /lus/eagle/projects/generator/jobs_by_topdir/droplet_refined.txt \
    --full_set 0 --root_omol_results /lus/eagle/projects/generator/OMol25_postprocessing/



"""