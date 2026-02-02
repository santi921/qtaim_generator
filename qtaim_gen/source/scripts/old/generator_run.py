#!/usr/bin/env python3

# runs a single folder analysis. Mainly for quacc exe compatibility

import os
import logging
import argparse
import resource
import time
from typing import Optional, Dict, Any, List

from qtaim_gen.source.core.omol import gbw_analysis
from qtaim_gen.source.utils.validation import validation_checks


def setup_logger_for_folder(folder: str, name: str = "gbw_analysis") -> logging.Logger:
    logger: logging.Logger = logging.getLogger(f"{name}-{folder}")
    # Avoid duplicate handlers
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        fh: logging.FileHandler = logging.FileHandler(
            os.path.join(folder, "gbw_analysis.log")
        )
        fmt: logging.Formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s"
        )
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


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

    parser.add_argument(
        "--move_results", action="store_true", help="move results to a separate folder"
    )

    parser.add_argument(
        "--run_root", type=str, help="absolute path to the folder to run analysis on"
    )

    parser.add_argument(
        "--move_results_to_folder",
        type=str,
        help="absolute path to the folder to move results to",
        default=None,
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
    n_threads = int(args.n_threads)
    full_set = int(args.full_set) if "full_set" in args else 0
    overwrite = bool(args.overwrite) if "overwrite" in args else False
    move_results = bool(args.move_results) if "move_results" in args else False
    run_root = args.run_root
    move_results_to_folder = args.move_results_to_folder

    # set env vars
    resource.setrlimit(
        resource.RLIMIT_STACK, (resource.RLIM_INFINITY, resource.RLIM_INFINITY)
    )

    run_root = args.run_root

    result: Dict[str, Any] = {
        "folder": run_root,
        "status": "unknown",
        "elapsed": None,
        "error": None,
    }
    folder = os.path.abspath(run_root)
    logger: logging.Logger = setup_logger_for_folder(folder)

    print(f"Selected folder: {run_root}")
    # check three states - not started, running, finished
    # check if there is a file called timings.json, qtaim.json, other.json, fuzzy_full,json, and charge.json
    outputs_present: bool = all(
        os.path.exists(os.path.join(folder, fn))
        for fn in (
            "timings.json",
            "qtaim.json",
            "other.json",
            "fuzzy_full.json",
            "charge.json",
        )
    )
    tf_validation = validation_checks(
        folder,
        full_set=full_set,
        verbose=False,
        move_results=move_results,
        logger=logger,
    )
    if outputs_present and not overwrite and tf_validation:
        logger.info("Skipping %s: already processed and validated", folder)
        result["status"] = "skipped"
        return result

    # check if there are multiple .*mwfn files
    mwfn_files: List[str] = [f for f in os.listdir(folder) if f.endswith(".mwfn")]
    if len(mwfn_files) > 1 and not overrun_running:
        logger.info("Skipping %s: multiple mwfn files found", folder)
        result["status"] = "skipped"
        return result

    else:
        try:
            # create logger in target folder
            logging.basicConfig(
                filename=os.path.join(run_root, "gbw_analysis.log"),
                level=logging.INFO,
                format="%(asctime)s - %(levelname)s - %(message)s",
            )

            t0: float = time.time()

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
                logger=logger,
                preprocess_compressed=preprocess_compressed,
                move_results=move_results,
            )  # works!
            t1: float = time.time()
            result["elapsed"] = t1 - t0
            result["status"] = "ok"
            logger.info("Completed folder %s in %.2f s", folder, result["elapsed"])

            if move_results and move_results_to_folder is not None:
                # dest_folder = os.path.join(move_results_to_folder, os.path.basename(run_root))
                if not os.path.exists(move_results_to_folder):
                    os.makedirs(move_results_to_folder)

                for file_name in [
                    "timings.json",
                    "qtaim.json",
                    "other.json",
                    "fuzzy_full.json",
                    "charge.json",
                    "gbw_analysis.log",
                ]:
                    src_file = os.path.join(run_root, file_name)
                    if os.path.exists(src_file):
                        print(f"Moving {src_file} to {move_results_to_folder}")
                        dest_file = os.path.join(move_results_to_folder, file_name)
                        os.rename(src_file, dest_file)

        except Exception as exc:
            logger.exception("Error processing %s: %s", folder, exc)
            result["status"] = "error"
            result["error"] = str(exc)
            return result

    pass


if __name__ == "__main__":
    raise SystemExit(main())
