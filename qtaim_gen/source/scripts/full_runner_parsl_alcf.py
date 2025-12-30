#!/usr/bin/env python3
from asyncio.log import logger
import os
import signal
import random
import argparse
import parsl
import resource
from typing import Optional, List


from qtaim_gen.source.core.workflow import run_folder_task_alcf
from qtaim_gen.source.utils.parsl_configs import (
    alcf_config,
    base_config
)

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
        prog="full_runner_parsl",
        description=(
            "Run QTAIM analysis on multiple folders concurrently using Parsl. "
            "This script submits one job per folder listed in --job_file."
        ),
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="debug mode",
    )
    parser.add_argument(
        "--multiwfn_cmd", type=str, help="absolute path to Multiwfn_noGUI executable"
    )

    parser.add_argument(
        "--orca_2mkl_cmd", type=str, help="absolute path to orca_6_2mkl executable"
    )

    parser.add_argument(
        "--preprocess_compressed",
        action="store_true",
        help="Whether to preprocess compressed files before analysis (default: False).",
    )

    parser.add_argument(
        "--restart",
        action="store_true",
        help="whether or not to restart failed calculations",
    )

    parser.add_argument(
        "--clean",
        action="store_true",
        help="whether or not to clean up intermediate files",
    )

    parser.add_argument(
        "--parse_only",
        action="store_true",
        help="only parse existing files, do not run analysis",
    )

    parser.add_argument(
        "--num_folders",
        type=int,
        default=100,
        help="number of folders to check and try to run",
    )

    parser.add_argument(
        "--job_file",
        type=str,
        help="file containing list of folders to run analysis on",
        default="./out.txt",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would run and exit without executing jobs (Parsl not invoked).",
    )

    parser.add_argument(
        "--prevalidate",
        action="store_true",
        help="Pre-validate folder jobs before running (checks completion).",
    )

    parser.add_argument(
        "--n_threads",
        type=int,
        default=4,
        help="Only used in local mode: total number of threads to use in threadpool",
    )

    parser.add_argument(
        "--n_threads_per_job",
        type=int,
        default=4,
        help="Number of threads to allocate per job",
    )

    parser.add_argument(
        "--full_set",
        type=int,
        default=0,
        help="level of calculation detail (0-baseline, 1-modest, 2-full)",
    )

    parser.add_argument(
        "--overwrite", action="store_true", help="overwrite existing analysis files"
    )

    parser.add_argument(
        "--move_results", action="store_true", help="move results to a separate folder"
    )

    # parsl args
    parser.add_argument(
        "--queue", type=str, default="debug", help="PBS queue to use (HPC)"
    )

    parser.add_argument(
        "--timeout_hr", type=float, default=0.5, help="Walltime for each PBS job (HPC)"
    )

    parser.add_argument(
        "--safety_factor",
        type=float,
        default=1.0,
        help="Safety factor for worker allocation. In local it's ratio between total workers and threads per job (HPC)",
    )

    parser.add_argument(
        "--type_runner",
        type=str,
        default="local",
        help="local or hpc/qsub submission via parsl (HPC)",
    )

    parser.add_argument(
        "--n_nodes", type=int, default=1, help="number of nodes to use (HPC)"
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

    args = parser.parse_args(argv)
    # print(args)
    for key, value in vars(args).items():
        print(f"{key}: {value}")
    # Safely extract boolean flags and other values using getattr
    # overrun_running: bool = bool(getattr(args, "overrun_running", False))
    restart: bool = bool(getattr(args, "restart", False))
    multiwfn_cmd: Optional[str] = getattr(args, "multiwfn_cmd", None)
    orca6_2mkl: Optional[str] = getattr(args, "orca_2mkl_cmd", None)
    preprocess_compressed: bool = bool(getattr(args, "preprocess_compressed", False))
    debug: bool = bool(getattr(args, "debug", False))
    clean: bool = bool(getattr(args, "clean", False))
    parse_only: bool = bool(getattr(args, "parse_only", False))
    num_folders: int = int(getattr(args, "num_folders", 100))
    n_threads: int = int(getattr(args, "n_threads", 128))
    n_threads_per_job: int = int(getattr(args, "n_threads_per_job", 4))
    full_set: int = int(getattr(args, "full_set", 0))
    move_results: bool = bool(getattr(args, "move_results", False))
    job_file: str = getattr(args, "job_file")
    prevalidate: bool = bool(getattr(args, "prevalidate", False))
    dry_run: bool = bool(getattr(args, "dry_run", False))
    overwrite = bool(args.overwrite) if "overwrite" in args else False
    root_omol_results: Optional[str] = getattr(args, "root_omol_results", None)
    root_omol_inputs: Optional[str] = getattr(args, "root_omol_inputs", None)

    # parsl args
    type_runner: str = str(getattr(args, "type_runner", "local"))
    queue: str = str(getattr(args, "queue", "debug"))
    timeout_hr: float = float(getattr(args, "timeout_hr", 0.5))
    safety_factor: float = float(getattr(args, "safety_factor", 1.0))
    n_nodes: int = int(getattr(args, "n_nodes", 1))
    # convert timeout_hr to str
    timeout_str: str = (
        f"{int(timeout_hr)}:{int((timeout_hr - int(timeout_hr)) * 60):02d}:00"
    )
    assert type_runner in ["local", "hpc"], "type_runner must be 'local' or 'hpc'"
    if type_runner == "local":
        n_threads_per_job = n_threads_per_job
        parsl_config = base_config(n_workers=int(n_threads / n_threads_per_job))

    else:
        parsl_config, n_threads_per_job = alcf_config(
            queue=queue,
            walltime=timeout_str,
            threads_per_task=n_threads_per_job,
            safety_factor=safety_factor,
            n_jobs=n_nodes,
            monitoring=False,
        )

    ##################### Gather Configs for Parsl
    parsl.clear()
    parsl.load(parsl_config)
    print("Parsl config loaded. Submitting jobs...")
    print(parsl_config)
    ####################

    # set env vars - this is only for the main process; workers set their own envs
    if resource == "local":
        os.environ["OMP_NUM_THREADS"] = "{}".format(n_threads_per_job)
        #threads_per = n_threads_per_job

    # to handle early stops on the pilot job
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    # set mem
    # Basic static checks before launching heavy work
    if not os.path.exists(job_file):
        print(f"Error: job_file '{job_file}' does not exist")
        return 2

    # make dir for results root
    if not os.path.exists(root_omol_results):
        os.makedirs(root_omol_results, exist_ok=True)

    
    folders_run = get_folders_from_file(
        job_file, 
        num_folders, 
        root_omol_results,
        root_omol_inputs,
        pre_validate=prevalidate, 
        move_results=move_results, 
        full_set=full_set
    )
    
    if not folders_run:
        print(f"No folders found in {job_file}")
        return []

    # If dry-run requested, print a short plan and exit
    if dry_run:
        print("Dry-run: the following folders WOULD be processed (sample):")
        for p in folders_run[: min(20, len(folders_run))]:
            print("  ", p)
        print(f"Total folders listed: {len(folders_run)}")
        return 0
    
    # Submit one Parsl job per selected folder. We do not pass a logger
    # into the remote app; each worker will create its own per-folder logger.
    futures = [
        run_folder_task_alcf(
            folder=f,
            orca_2mkl_cmd=orca6_2mkl,
            multiwfn_cmd=multiwfn_cmd,
            parse_only=parse_only,
            separate=True,  # default to true b/c this is how restarts work best
            overwrite=overwrite,
            orca_6=True,
            clean=clean,
            n_threads=n_threads_per_job,
            full_set=full_set,
            restart=restart,
            debug=debug,
            preprocess_compressed=preprocess_compressed,
            move_results=move_results,
            root_omol_results=root_omol_results,
            root_omol_inputs=root_omol_inputs,
        )
        for f in folders_run
    ]

    try:
        # block until all done
        for f in futures:
            if should_stop:
                print("Graceful shutdown requested. Exiting before all jobs complete.")
                break
            f.result()
    finally:
        # ensure we cleanup even on exceptions
        try:
            dfk = parsl.dfk()
            if dfk is not None:
                dfk.cleanup()  # shutdown workers/executors
        except Exception as e:
            # log warning, don't crash on cleanup failure
            print("Warning: cleanup failed:", e)
        # remove Parsl DFK from module so subsequent imports/config changes are possible
        try:
            parsl.clear()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())


"""
- running 12/29
555961 / 737210
full-runner-parsl-alcf --num_folders 20000 --orca_2mkl_cmd $HOME/orca_6_0_0/orca_2mkl    \
      --multiwfn_cmd $HOME/Multiwfn_3_8/Multiwfn_noGUI --clean --full_set 0 --type_runner hpc \
        --n_threads 220 --n_threads_per_job 1 --safety_factor 1.0 --move_results \
        --timeout_hr 5             --queuef workq-route --restart --n_nodes 1 --type_runner hpc --job_file ../jobs_by_topdir/ani1xbb_refined.txt \
        --preprocess_compressed --root_omol_results /lus/eagle/projects/generator/OMol25_postprocessing/  \
        --root_omol_inputs /lus/eagle/projects/OMol25/ 

                    

        
- running 12/29
25218 / 51102
full-runner-parsl-alcf --num_folders 10000 --orca_2mkl_cmd $HOME/orca_6_0_0/orca_2mkl    \
      --multiwfn_cmd $HOME/Multiwfn_3_8/Multiwfn_noGUI --clean --full_set 0 --type_runner hpc \
        --n_threads 220 --n_threads_per_job 1 --safety_factor 1.0 --move_results  \
        --timeout_hr 5             --queue workq-route --restart --n_nodes 2 --type_runner hpc \
        --job_file /lus/eagle/projects/generator/jobs_by_topdir/orbnet_denali_refined.txt \
        --preprocess_compressed --root_omol_results /lus/eagle/projects/generator/OMol25_postprocessing/ \
        --root_omol_inputs /lus/eagle/projects/OMol25/                

- running 12/29
full-runner-parsl-alcf --num_folders 25000 --orca_2mkl_cmd $HOME/orca_6_0_0/orca_2mkl    \
      --multiwfn_cmd $HOME/Multiwfn_3_8/Multiwfn_noGUI --clean --full_set 0 --type_runner hpc \
        --n_threads 220 --n_threads_per_job 1 --safety_factor 1.0 --move_results \
        --timeout_hr 4             --queue workq-route --restart --n_nodes 4 --type_runner hpc \
        --job_file /lus/eagle/projects/generator/jobs_by_topdir/geom_orca6.txt \
        --preprocess_compressed --root_omol_results /lus/eagle/projects/generator/OMol25_postprocessing/ \
        --root_omol_inputs /lus/eagle/projects/OMol25/ 



"""


