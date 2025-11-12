#!/usr/bin/env python3
import os
import logging
import random
import argparse
import resource
from qtaim_gen.source.core.omol import gbw_analysis


def main(argv=None):

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--overrun_running",
        action="store_true",
        help="overrun folders that are currently running (multiple .mwfn files)",
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
        help="whether or not to preprocess compressed files",
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
        "--num_jobs",
        type=int,
        default=100,
        help="number of jobs to check and try to run",
    )

    parser.add_argument(
        "--job_file",
        type=str,
        help="file containing list of folders to run analysis on",
        default="./out.txt",
    )

    parser.add_argument(
        "--n_threads", type=int, default=4, help="number of threads to use"
    )

    parser.add_argument(
        "--full_set",
        type=int,
        default=0,
        help="level of calculation detail (0-baseline, 1-baseline)",
    )

    parser.add_argument(
        "--overwrite", action="store_true", help="overwrite existing analysis files"
    )

    args = parser.parse_args(argv)

    overrun_running = bool(args.overrun_running) if "overrun_running" in args else False
    restart = bool(args.restart) if "restart" in args else False
    multiwfn_cmd = args.multiwfn_cmd
    orca6_2mkl = args.orca_2mkl_cmd
    preprocess_compressed = (
        bool(args.preprocess_compressed) if "preprocess_compressed" in args else False
    )
    debug = bool(args.debug) if "debug" in args else False
    clean = bool(args.clean) if "clean" in args else False
    parse_only = bool(args.parse_only) if "parse_only" in args else False
    num_jobs = int(args.num_jobs) if "num_jobs" in args else 1
    n_threads = int(args.n_threads)
    full_set = int(args.full_set) if "full_set" in args else 0
    overwrite = bool(args.overwrite) if "overwrite" in args else False
    job_file = args.job_file

    # set env vars
    # os.environ["OMP_STACKSIZE"] = args.OMP_STACKSIZE
    # set mem
    # os.system(
    #    "ulimit -s unlimited"
    # )  # this sometimes doesn't work and I need to manually set this in cmdline
    resource.setrlimit(
        resource.RLIMIT_STACK, (resource.RLIM_INFINITY, resource.RLIM_INFINITY)
    )

    # this is the running list of folders to run analysis on - change this to your own file
    folder_file = os.path.join(job_file)
    # read folder file and randomly select a folder
    with open(folder_file, "r") as f:
        folders = f.readlines()
    folders = [f.strip() for f in folders if f.strip()]  # remove empty lines
    if not folders:
        print("No folders found in out.txt")
        return

    # randomly sample num_jobs folders without replacement
    if num_jobs > len(folders):
        print(
            f"Requested num_jobs {num_jobs} exceeds available folders {len(folders)}. Reducing to {len(folders)}."
        )
        num_jobs = len(folders)

    # shuffle folders
    random.shuffle(folders)
    folders_run = folders[:num_jobs]

    for i in range(num_jobs):
        run_root = folders_run[i]
        print(f"Selected folder: {run_root}")
        # check three states - not started, running, finished
        # check if there is a file called timings.json, qtaim.json, other.json, fuzzy_full,json, and charge.json
        if (
            os.path.exists(os.path.join(run_root, "timings.json"))
            and os.path.exists(os.path.join(run_root, "qtaim.json"))
            and os.path.exists(os.path.join(run_root, "other.json"))
            and os.path.exists(os.path.join(run_root, "fuzzy_full.json"))
            and os.path.exists(os.path.join(run_root, "charge.json"))
            and not overwrite
        ):
            print(f"Skipping {run_root} - already processed")
            continue

        # check if there are multiple .*mwfn files
        mwfn_files = [f for f in os.listdir(run_root) if f.endswith(".mwfn")]

        if len(mwfn_files) > 1 and not overrun_running:
            print(f"Skipping {run_root} - multiple mwfn files found")
            continue

        else:
            print(f"Processing {run_root} - {i+1}/{num_jobs}")
            try:
                # create logger in target folder
                logging.basicConfig(
                    filename=os.path.join(run_root, "gbw_analysis.log"),
                    level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s",
                )

                gbw_analysis(
                    folder=run_root,
                    orca_2mkl_cmd=orca6_2mkl,
                    multiwfn_cmd=multiwfn_cmd,
                    parse_only=parse_only,
                    separate=True,  # default to true b/c this is how restarts work best
                    overwrite=overwrite,
                    orca_6=True,
                    clean=clean,
                    n_threads=n_threads,
                    full_set=full_set,
                    restart=restart,
                    debug=debug,
                    logger=logging.getLogger("gbw_analysis"),
                    preprocess_compressed=preprocess_compressed,
                )  # works!
            except Exception as e:
                print(f"Error in gbw_analysis for {run_root}: {e}")

    pass


if __name__ == "__main__":
    raise SystemExit(main())
