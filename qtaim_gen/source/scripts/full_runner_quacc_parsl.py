from quacc import change_settings
from qtaim_gen.source.quacc.runner import GeneratorRunner
from quacc import get_settings
import os
import time
import argparse

from typing import Optional, Dict, Any, List
from parsl import python_app
import parsl
from parsl.config import Config
import resource


def acquire_lock(folder: str) -> bool:
    lockfile: str = os.path.join(folder, ".processing.lock")
    try:
        fd: int = os.open(lockfile, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        return True
    except FileExistsError:
        return False


def release_lock(folder: str) -> None:
    lockfile: str = os.path.join(folder, ".processing.lock")
    try:
        os.remove(lockfile)
    except FileNotFoundError:
        pass


def process_folder(
    folder: str,
    multiwfn_cmd: Optional[str] = None,
    orca_2mkl_cmd: Optional[str] = None,
    parse_only: bool = False,
    restart: bool = False,
    clean: bool = False,
    debug: bool = False,
    overrun_running: bool = False,
    preprocess_compressed: bool = False,
    omp_stacksize: str = "64000000",
    n_threads: int = 3,
    overwrite: bool = False,
    separate: bool = True,
    orca_6: bool = True,
    full_set: bool = False,
    move_results: bool = True,
) -> Dict[str, Any]:
    """Process a single folder and return a small status dict.

    Args:
        folder: path to folder
        ...: same flags you used before

    Returns:
        dict with keys: folder, status ('ok'|'error'|'skipped'), elapsed, error (opt)
    """

    # multiwfn_cmd = "/home/santiagovargas/dev/Multiwfn_3.8_dev_bin_Linux_noGUI/Multiwfn_noGUI"
    # orca_2mkl_cmd = "/home/santiagovargas/orca_6_0_0/orca_2mkl"
    with change_settings(
        {"RESULTS_DIR": folder, "SCRATCH_DIR": "./scratch/", "CREATE_UNIQUE_DIR": False}
    ):
        print("Running GeneratorRunner test...")
        print("Setting RESULTS_DIR to ./results/ and SCRATCH_DIR to ./scratch/")

        full_set = 0
        n_threads = 5
        clean = True
        move_results = True
        overwrite = True
        overrun_running = True
        separate = True
        debug = False
        preprocess_compressed = False
        parse_only = False
        restart = False
        dry_run = False
        # print settings

        # not getting passed through
        print("Current SETTINGS:", get_settings())
        runner = GeneratorRunner(
            command="generator-single-runner",
            folder=folder,
            multiwfn_cmd=multiwfn_cmd,
            orca_2mkl_cmd=orca_2mkl_cmd,
            n_threads=n_threads,
            parse_only=parse_only,
            restart=restart,
            clean=clean,
            debug=debug,
            overrun_running=overrun_running,
            preprocess_compressed=preprocess_compressed,
            overwrite=overwrite,
            separate=separate,
            orca_6=True,
            full_set=full_set,
            move_results=move_results,
            dry_run=dry_run,
        )
        result = runner.run_cmd()
        # print("Command Output:", result.stdout)


@python_app
def run_folder_task_quacc(
    folder: str,
    multiwfn_cmd: Optional[str] = None,
    orca_2mkl_cmd: Optional[str] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Parsl python_app wrapper that runs process_folder on a worker.

    The real processing function is imported inside the app so the worker
    process imports the correct package layout and environment.
    """
    # this runs inside remote worker; import inside to ensure worker env has package

    return process_folder(
        folder, multiwfn_cmd=multiwfn_cmd, orca_2mkl_cmd=orca_2mkl_cmd, **kwargs
    )


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
        "--dry-run",
        action="store_true",
        help="Print what would run and exit without executing jobs (Parsl not invoked).",
    )

    parser.add_argument(
        "--n_threads", type=int, default=4, help="number of threads to use per job"
    )

    parser.add_argument(
        "--n_workers", type=int, default=4, help="number of workers total"
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

    parser.add_argument(
        "--move_results", action="store_true", help="move results to a separate folder"
    )

    parser.add_argument(
        "--exhaustive_qtaim",
        action="store_true",
        help="use exhaustive QTAIM critical point search (spherical search around atoms)",
    )

    args = parser.parse_args(argv)

    # Safely extract boolean flags and other values using getattr
    # overrun_running: bool = bool(getattr(args, "overrun_running", False))
    restart: bool = bool(getattr(args, "restart", False))
    multiwfn_cmd: Optional[str] = getattr(args, "multiwfn_cmd", None)
    orca6_2mkl: Optional[str] = getattr(args, "orca_2mkl_cmd", None)
    preprocess_compressed: bool = bool(getattr(args, "preprocess_compressed", False))
    debug: bool = bool(getattr(args, "debug", False))
    clean: bool = bool(getattr(args, "clean", False))
    parse_only: bool = bool(getattr(args, "parse_only", False))
    num_jobs: int = int(getattr(args, "num_jobs", 1))
    n_threads: int = int(getattr(args, "n_threads", 4))
    n_workers: int = int(getattr(args, "n_workers", 10))
    full_set: int = int(getattr(args, "full_set", 0))
    move_results: bool = bool(getattr(args, "move_results", False))
    job_file: str = getattr(args, "job_file")
    dry_run: bool = bool(getattr(args, "dry_run", False))
    overwrite = bool(args.overwrite) if "overwrite" in args else False
    exhaustive_qtaim: bool = bool(getattr(args, "exhaustive_qtaim", False))

    # set env vars
    resource.setrlimit(
        resource.RLIMIT_STACK, (resource.RLIM_INFINITY, resource.RLIM_INFINITY)
    )
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
        print(
            f"Warning: {len(missing)} listed paths do not exist or are not directories; they will be skipped."
        )
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

    local_threads = Config(
        executors=[ThreadPoolExecutor(max_threads=n_workers, label="local_threads")]
    )

    parsl.clear()
    parsl.load(local_threads)

    print("Parsl config loaded. Submitting jobs...")
    print(local_threads)

    # randomly sample num_jobs folders without replacement
    if num_jobs > len(folders):
        print(
            f"Requested num_jobs {num_jobs} exceeds available folders {len(folders)}. Reducing to {len(folders)}."
        )
        num_jobs = len(folders)

    # shuffle folders
    random.shuffle(folders)
    folders_run = folders[:num_jobs]

    # Submit one Parsl job per selected folder. We do not pass a logger
    # into the remote app; each worker will create its own per-folder logger.
    futures = [
        run_folder_task(
            folder=f,
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
            preprocess_compressed=preprocess_compressed,
            move_results=move_results,
            exhaustive_qtaim=exhaustive_qtaim,
        )
        for f in folders_run
    ]

    try:
        # block until all done
        for f in futures:
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
