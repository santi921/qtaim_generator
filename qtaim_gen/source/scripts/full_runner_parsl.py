#!/usr/bin/env python3
from qtaim_gen.source.core.omol import gbw_analysis
import os
import logging
import random
import argparse
import time
import resource
from typing import Optional, List, Dict, Any


from parsl import python_app
from parsl.configs.local_threads import config  # or your config
import parsl

def setup_logger_for_folder(folder: str, name: str = "gbw_analysis") -> logging.Logger:
    logger = logging.getLogger(f"{name}-{folder}")
    # Avoid duplicate handlers
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        fh = logging.FileHandler(os.path.join(folder, "gbw_analysis.log"))
        fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger

def acquire_lock(folder: str) -> bool:
    lockfile = os.path.join(folder, ".processing.lock")
    try:
        fd = os.open(lockfile, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        return True
    except FileExistsError:
        return False

def release_lock(folder: str) -> None:
    lockfile = os.path.join(folder, ".processing.lock")
    try:
        os.remove(lockfile)
    except FileNotFoundError:
        pass

def process_folder(
    folder: str,
    multiwfn_cmd: str = None,
    orca_2mkl_cmd: str = None,
    parse_only: bool = False,
    restart: bool = False,
    clean: bool = False,
    debug: bool = False,
    overrun_running: bool = False,
    preprocess_compressed: bool = False,
    omp_stacksize: str = "64000000",
    n_threads: int = 4,
    overwrite: bool = False,
) -> dict:
    """Process a single folder and return a small status dict.

    Args:
        folder: path to folder
        ...: same flags you used before

    Returns:
        dict with keys: folder, status ('ok'|'error'|'skipped'), elapsed, error (opt)
    """
    result = {"folder": folder, "status": "unknown", "elapsed": None, "error": None}
    logger = setup_logger_for_folder(folder)

    try:
        # pre-checks (idempotency)
        # e.g. skip if outputs exist and not restart
        outputs_present = all(
            os.path.exists(os.path.join(folder, fn))
            for fn in ("timings.json", "qtaim.json", "other.json", "fuzzy_full.json", "charge.json")
        )
        if outputs_present and not overwrite:
            logger.info("Skipping %s: already processed", folder)
            result["status"] = "skipped"
            return result

        # optional: check mwfn files, multiple mwfn guard
        mwfn_files = [f for f in os.listdir(folder) if f.endswith(".mwfn")]
        if len(mwfn_files) > 1 and not overrun_running:
            logger.info("Skipping %s: multiple mwfn files found", folder)
            result["status"] = "skipped"
            return result

        # set env
        os.environ["OMP_STACKSIZE"] = omp_stacksize

        # call the existing function (pass logger)
        t0 = time.time()
        gbw_analysis(
            folder=folder,
            orca_2mkl_cmd=orca_2mkl_cmd,
            multiwfn_cmd=multiwfn_cmd,
            parse_only=parse_only,
            separate=True,
            overwrite=overwrite,
            orca_6=True,
            clean=clean,
            n_threads=n_threads,
            restart=restart,
            debug=debug,
            logger=logger,
            preprocess_compressed=preprocess_compressed,
        )
        t1 = time.time()

        result["elapsed"] = t1 - t0
        result["status"] = "ok"
        logger.info("Completed folder %s in %.2f s", folder, result["elapsed"])
        return result

    except Exception as exc:
        logger.exception("Error processing %s: %s", folder, exc)
        result["status"] = "error"
        result["error"] = str(exc)
        return result

@python_app
def run_folder_task(folder: str, multiwfn_cmd: Optional[str] = None, orca_2mkl_cmd: Optional[str] = None, **kwargs) -> Dict[str, Any]:
    """Parsl python_app wrapper that runs process_folder on a worker.

    The real processing function is imported inside the app so the worker
    process imports the correct package layout and environment.
    """
    # this runs inside remote worker; import inside to ensure worker env has package
    from qtaim_gen.source.scripts.full_runner import process_folder
    return process_folder(folder, multiwfn_cmd=multiwfn_cmd, orca_2mkl_cmd=orca_2mkl_cmd, **kwargs)

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
        "--overrun_running",
        action="store_true",
        help="overrun folders that are currently running (multiple .mwfn files)",
    )

    # parser.add_argument(
    #    "--OMP_STACKSIZE", type=str, default="64000000", help="set OMP_STACKSIZE environment variable"
    # )

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
        action="store_true", help="only parse existing files, do not run analysis",
    )

    parser.add_argument(
        "--num_jobs", type=int, default=100, help="number of jobs to check and try to run"
    )

    parser.add_argument(
        "--job_file", type=str, help="file containing list of folders to run analysis on", default="./out.txt"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would run and exit without executing jobs (Parsl not invoked).",
    )

    parser.add_argument(
        "--test-count",
        type=int,
        default=0,
        help="Run the first N folders locally (no Parsl) for quick debugging. 0 means disabled.",
    )

    parser.add_argument(
        "--n_threads", type=int, default=4, help="number of threads to use"
    )

    parser.add_argument(
        "--full_set", action="store_true", help="run full set of analysis or refined set"

    )

    args = parser.parse_args(argv)

    # Safely extract boolean flags and other values using getattr
    overrun_running: bool = bool(getattr(args, "overrun_running", False))
    restart: bool = bool(getattr(args, "restart", False))
    multiwfn_cmd: Optional[str] = getattr(args, "multiwfn_cmd", None)
    orca6_2mkl: Optional[str] = getattr(args, "orca_2mkl_cmd", None)
    preprocess_compressed: bool = bool(getattr(args, "preprocess_compressed", False))
    debug: bool = bool(getattr(args, "debug", False))
    clean: bool = bool(getattr(args, "clean", False))
    parse_only: bool = bool(getattr(args, "parse_only", False))
    num_jobs: int = int(getattr(args, "num_jobs", 1))
    n_threads: int = int(getattr(args, "n_threads", 4))
    full_set: bool = bool(getattr(args, "full_set", False))
    job_file: str = getattr(args, "job_file")
    dry_run: bool = bool(getattr(args, "dry_run", False))
    test_count: int = int(getattr(args, "test_count", 0))


    # set env vars
    #os.environ["OMP_STACKSIZE"] = args.OMP_STACKSIZE
    resource.setrlimit(resource.RLIMIT_STACK, (resource.RLIM_INFINITY, resource.RLIM_INFINITY))
    # set mem
    # Basic static checks before launching heavy work
    if not os.path.exists(job_file):
        print(f"Error: job_file '{job_file}' does not exist")
        return 2

    # read folder file and randomly select a folder
    folder_file = os.path.join(job_file)
    with open(folder_file, "r") as f:
        folders = f.readlines()
    folders = [f.strip() for f in folders if f.strip()]  # remove empty lines
    if not folders:
        print(f"No folders found in {job_file}")
        return 0

    # verify listed folders exist, warn & skip missing entries
    existing_folders = []
    missing = []
    for f in folders:
        if os.path.exists(f) and os.path.isdir(f):
            existing_folders.append(f)
        else:
            missing.append(f)
    if missing:
        print(f"Warning: {len(missing)} listed paths do not exist or are not directories; they will be skipped.")
        for m in missing[:10]:
            print("  missing:", m)
        if len(missing) > 10:
            print("  ...")
    folders = existing_folders
    if not folders:
        print("No valid folders to run after filtering missing entries.")
        return 0

    # If dry-run requested, print a short plan and exit
    if dry_run:
        print("Dry-run: the following folders WOULD be processed (sample):")
        for p in folders[: min(20, len(folders))]:
            print("  ", p)
        print(f"Total folders listed: {len(folders)}")
        return 0

    
    parsl.load(config)


    #randomly sample num_jobs folders without replacement
    if num_jobs > len(folders):
        num_jobs = len(folders)

    # shuffle folders
    random.shuffle(folders)
    folders_run = folders[:num_jobs]

    # Local-run fallback for quick testing: run first `test_count` folders directly
    if test_count > 0:
        to_run = folders_run[:min(test_count, len(folders_run))]
        print(f"Running {len(to_run)} folders locally (test mode)")
        local_results = []
        for p in to_run:
            r = process_folder(
                p,
                orca_2mkl_cmd=orca6_2mkl,
                multiwfn_cmd=multiwfn_cmd,
                parse_only=parse_only,
                restart=restart,
                clean=clean,
                n_threads=n_threads,
                full_set=full_set,
                debug=debug,
                preprocess_compressed=preprocess_compressed,
            )
            print(r)
            local_results.append(r)
        return 0

    # Submit one Parsl job per selected folder. We do not pass a logger
    # into the remote app; each worker will create its own per-folder logger.
    futures = [
        run_folder_task(
            folder=f,
            orca_2mkl_cmd=orca6_2mkl,
            multiwfn_cmd=multiwfn_cmd,
            parse_only=parse_only,
            separate=True,  # default to true b/c this is how restarts work best
            overwrite=False,
            orca_6=True,
            clean=clean,
            n_threads=n_threads,
            full_set=full_set,
            restart=restart,
            debug=debug,
            preprocess_compressed=preprocess_compressed,
        )
        for f in folders_run
    ]
    
    
    for fut in futures:
        res = fut.result()
        print(res)

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
