import os
import logging
import time
import zipfile
from typing import Optional, Dict, Any, List
import shutil

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


_LOCK_MAX_AGE_S: float = 28800.0  # 8 hours


def acquire_lock(folder: str, max_age_s: float = _LOCK_MAX_AGE_S) -> bool:
    """Acquire a processing lock on a folder.

    Uses mtime-based stale detection.  If an existing lock's mtime is older
    than *max_age_s*, it is considered stale and broken.  Atomic
    ``O_CREAT | O_EXCL`` prevents races between concurrent workers.
    """
    lockfile: str = os.path.join(folder, ".processing.lock")

    if os.path.exists(lockfile):
        try:
            age: float = time.time() - os.path.getmtime(lockfile)
        except OSError:
            age = float("inf")  # can't stat → treat as stale

        if age < max_age_s:
            return False  # not stale, genuinely locked

        # Stale → break it
        logging.getLogger("lock").warning(
            "Breaking stale lock in %s (age=%.0fs, threshold=%.0fs)",
            folder, age, max_age_s,
        )
        try:
            os.remove(lockfile)
        except FileNotFoundError:
            pass  # another worker already broke it

    # Atomic create-or-fail
    try:
        fd: int = os.open(lockfile, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode())  # PID for debugging only
        os.close(fd)
        return True
    except FileExistsError:
        return False  # lost race to another worker


def release_lock(folder: str) -> None:
    """Release processing lock.  Safe to call even if lock doesn't exist."""
    lockfile: str = os.path.join(folder, ".processing.lock")
    try:
        os.remove(lockfile)
    except FileNotFoundError:
        pass


def teardown_logger(folder: str, name: str = "gbw_analysis") -> None:
    """Close and detach all handlers from a folder's logger to prevent fd accumulation."""
    logger = logging.getLogger(f"{name}-{folder}")
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)


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
    clean_first: bool = False,
    orca_6: bool = True,
    full_set: bool = False,
    move_results: bool = True,
    wfx: bool = False,
    check_orca: bool = False,
    exhaustive_qtaim: bool = False,
) -> Dict[str, Any]:
    """Process a single folder and return a small status dict.

    Args:
        folder: path to folder
        ...: same flags you used before

    Returns:
        dict with keys: folder, status ('ok'|'error'|'skipped'), elapsed, error (opt)
    """
    result: Dict[str, Any] = {
        "folder": folder,
        "status": "unknown",
        "elapsed": None,
        "error": None,
    }
    # normalize to absolute path and set up logger
    folder = os.path.abspath(folder)
    logger: logging.Logger = setup_logger_for_folder(folder)

    # Acquire lock before any work
    if not acquire_lock(folder):
        logger.info("Skipping %s: folder locked by active process", folder)
        result["status"] = "skipped"
        result["error"] = "folder locked by active process"
        return result

    try:
        if clean_first:
            # clean everything except gbw_analysis.log and .processing.lock
            for item in os.listdir(folder):
                if item not in ("gbw_analysis.log", ".processing.lock"):
                    item_path = os.path.join(folder, item)
                    try:
                        if os.path.isfile(item_path) or os.path.islink(item_path):
                            os.unlink(item_path)
                            logger.info(
                                f"Removed file {item_path} due to clean_first flag"
                            )
                        elif os.path.isdir(item_path):
                            shutil.rmtree(item_path)
                            logger.info(
                                f"Removed directory {item_path} due to clean_first flag"
                            )
                    except Exception as e:
                        logger.error(f"Failed to remove {item_path}. Reason: {e}")

        # pre-checks (idempotency)
        # e.g. skip if outputs exist and not restart
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

        if outputs_present and not overwrite:
            logger.info("Skipping %s: already processed", folder)

            try:
                tf_validation = validation_checks(
                    folder,
                    full_set=full_set,
                    verbose=False,
                    move_results=move_results,
                    logger=logger,
                )

                if not tf_validation:
                    logger.info("Validation failed for %s: reprocessing", folder)
                else:
                    logger.info("Validation passed for %s: skipping", folder)
                    result["status"] = "skipped"
                    return result
            except Exception as e:
                logger.warning("Validation check failed for %s: %s", folder, str(e))
                # continue processing

        # optional: check mwfn files, multiple mwfn guard
        mwfn_files: List[str] = [f for f in os.listdir(folder) if f.endswith(".mwfn")]
        if len(mwfn_files) > 1 and not overrun_running:
            logger.info("Skipping %s: multiple mwfn files found", folder)
            result["status"] = "skipped"
            return result
        subprocess_env = {**os.environ, "OMP_STACKSIZE": omp_stacksize}

        t0: float = time.time()
        gbw_analysis(
            folder=folder,
            orca_2mkl_cmd=orca_2mkl_cmd,
            multiwfn_cmd=multiwfn_cmd,
            parse_only=parse_only,
            separate=separate,
            overwrite=overwrite,
            orca_6=orca_6,
            clean=clean,
            n_threads=n_threads,
            restart=restart,
            debug=debug,
            logger=logger,
            full_set=full_set,
            preprocess_compressed=preprocess_compressed,
            move_results=move_results,
            wfx=wfx,
            check_orca=check_orca,
            exhaustive_qtaim=exhaustive_qtaim,
            subprocess_env=subprocess_env,
        )
        t1: float = time.time()

        files_to_remove = [
            "density_mat.npz",
            "orca.gbw.zstd0",
            "orca.gbw",
            "orca.tar.zst",
            "orca.inp.orig",
            "orca.property.txt",
            "orca.engrad",
            "orca_stderr",
        ]
        for fn in files_to_remove:
            fp = os.path.join(folder, fn)
            if os.path.exists(fp):
                os.remove(fp)
                # add log
                logger.info("Removed file %s to save space", fp)

        result["elapsed"] = t1 - t0
        result["status"] = "ok"
        logger.info("Completed folder %s in %.2f s", folder, result["elapsed"])

        return result

    except Exception as exc:
        logger.exception("Error processing %s: %s", folder, exc)
        result["status"] = "error"
        result["error"] = str(exc)
        return result

    finally:
        release_lock(folder)
        teardown_logger(folder)


def process_folder_alcf(
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
    clean_first: bool = False,
    full_set: bool = False,
    move_results: bool = True,
    patch_path: bool = False,
    root_omol_results: Optional[
        str
    ] = None,  # root where to store results, should mimic root_omol_inputs
    root_omol_inputs: Optional[str] = None,  # root where input folders are located
    wfx: bool = False,
    check_orca: bool = False,
    exhaustive_qtaim: bool = False,
) -> Dict[str, Any]:
    """Process a single folder and return a small status dict.

    Args:
        folder: path to folder
        ...: same flags you used before

    Returns:
        dict with keys: folder, status ('ok'|'error'|'skipped'), elapsed, error (opt)
    """
    result: Dict[str, Any] = {
        "folder": folder,
        "status": "unknown",
        "elapsed": None,
        "error": None,
    }

    files_to_remove = [
        "density_mat.npz",
        "orca.gbw.zstd0",
        "orca.gbw",
        "orca.tar.zst",
        "orca.inp.orig",
        "orca.property.txt",
        "orca.engrad",
        "orca_stderr",
        "orca.wfx", 
        "orca.wfn", # this is specific to HPC where we are moving wfns to process
        #"orca.inp"  # this is specific to HPC where we are moving wfns to process
    ]

    # normalize to absolute path and set up logger
    folder_inputs = folder
    if not root_omol_inputs or not root_omol_results:
        result["status"] = "error"
        result["error"] = "root_omol_inputs and root_omol_results are required"
        return result
    if not folder_inputs.startswith(root_omol_inputs):
        result["status"] = "error"
        result["error"] = f"folder {folder_inputs!r} does not start with root_omol_inputs {root_omol_inputs!r}"
        return result
    folder_relative = folder_inputs[len(root_omol_inputs):].lstrip(os.sep)
    folder_outputs = os.path.join(root_omol_results, folder_relative)

    if not os.path.exists(folder_outputs):
        os.makedirs(folder_outputs)

    folder = os.path.abspath(folder_outputs)
    logger: logging.Logger = setup_logger_for_folder(folder)

    # Acquire lock before any work
    if not acquire_lock(folder):
        logger.info("Skipping %s: folder locked by active process", folder)
        result["status"] = "skipped"
        result["error"] = "folder locked by active process"
        return result

    try:
        if clean_first:
            # clean everything except gbw_analysis.log and .processing.lock
            for item in os.listdir(folder):
                if item not in ("gbw_analysis.log", ".processing.lock"):
                    item_path = os.path.join(folder, item)
                    try:
                        if os.path.isfile(item_path) or os.path.islink(item_path):
                            os.unlink(item_path)
                            logger.info(f"Removed file {item_path} due to clean_first flag")
                        elif os.path.isdir(item_path):
                            shutil.rmtree(item_path)
                            logger.info(
                                f"Removed directory {item_path} due to clean_first flag"
                            )
                    except Exception as e:
                        logger.error(f"Failed to remove {item_path}. Reason: {e}")

        _COMPRESSED_EXTS = (".gbw.zstd0", ".tar.zst", ".tgz")
        empty_compressed: list = []

        for item in os.listdir(folder_inputs):
            # skip "density_mat.npz"
            if item != "density_mat.npz":
                s = os.path.join(folder_inputs, item)
                d = os.path.join(folder_outputs, item)

                if os.path.isdir(s):
                    if not os.path.exists(d):
                        os.makedirs(d)
                else:
                    if any(item.endswith(ext) for ext in _COMPRESSED_EXTS):
                        try:
                            src_size = os.path.getsize(s)
                        except OSError as _e:
                            logger.error("Could not stat compressed file %s: %s", s, _e)
                            empty_compressed.append(item)
                            continue
                        if src_size == 0:
                            logger.error(
                                "Empty compressed file detected: %s (0 bytes) - skipping copy",
                                s,
                            )
                            empty_compressed.append(item)
                            continue

                    if not os.path.exists(d):
                        shutil.copy2(s, d)
                        logger.info(f"Copied {s} to {d}")

        if empty_compressed:
            result["status"] = "error"
            result["error"] = f"empty compressed files in source: {empty_compressed}"
            return result

        try:
            tf_validation = validation_checks(
                folder,
                full_set=full_set,
                verbose=False,
                move_results=move_results,
                logger=logger,
                check_orca=check_orca,
            )

            if not overwrite and tf_validation:
                logger.info("Skipping %s: already processed and validated", folder)
                result["status"] = "skipped"

                try:
                    zip_file_out = os.path.join(folder, "out_files.zip")
                    if not os.path.exists(zip_file_out):
                        with zipfile.ZipFile(zip_file_out, "w") as zipf:
                            for file in os.listdir(folder):
                                # skip orca.out
                                if file.endswith(".out") and file != "orca.out":
                                    zipf.write(os.path.join(folder, file), arcname=file)
                                    os.remove(os.path.join(folder, file))
                                    logger.info(f"Zipped and removed {file}")

                    if move_results:
                        results_folder = os.path.join(folder, "generator")
                        if not os.path.exists(results_folder):
                            os.makedirs(results_folder)
                        shutil.move(zip_file_out, os.path.join(results_folder, "out_files.zip"))
                        logger.info(f"Moved out_files.zip to results folder {results_folder}")

                except Exception as e:
                    logger.info(f"Couldn't zip .out files in {folder}: {e}")

                for fn in files_to_remove:
                    fp = os.path.join(folder, fn)
                    if os.path.exists(fp):
                        os.remove(fp)
                        logger.info("Removed file %s to save space", fp)

                return result

        except Exception as e:
            logger.warning("Validation check failed for %s: %s", folder, str(e))
            # continue processing

        subprocess_env = {**os.environ, "OMP_STACKSIZE": omp_stacksize}

        t0: float = time.time()
        gbw_analysis(
            folder=folder,
            orca_2mkl_cmd=orca_2mkl_cmd,
            multiwfn_cmd=multiwfn_cmd,
            parse_only=parse_only,
            separate=separate,
            overwrite=overwrite,
            orca_6=orca_6,
            clean=clean,
            n_threads=n_threads,
            restart=restart,
            debug=debug,
            logger=logger,
            full_set=full_set,
            preprocess_compressed=preprocess_compressed,
            move_results=move_results,
            patch_path=patch_path,
            wfx=wfx,
            check_orca=check_orca,
            exhaustive_qtaim=exhaustive_qtaim,
            subprocess_env=subprocess_env,
        )
        t1: float = time.time()

        if clean:
            for fn in files_to_remove:
                fp = os.path.join(folder, fn)
                if os.path.exists(fp):
                    os.remove(fp)
                    logger.info("Removed file %s to save space", fp)

        result["elapsed"] = t1 - t0
        result["status"] = "ok"
        logger.info("Completed folder %s in %.2f s", folder, result["elapsed"])

        return result

    except Exception as exc:
        logger.exception("Error processing %s: %s", folder, exc)
        result["status"] = "error"
        result["error"] = str(exc)
        return result

    finally:
        release_lock(folder)
        teardown_logger(folder)


try:
    from parsl import python_app
except ImportError:
    # parsl is optional; only needed for HPC batch submission
    python_app = None


if python_app is not None:

    @python_app
    def run_folder_task(
        folder: str,
        multiwfn_cmd: Optional[str] = None,
        orca_2mkl_cmd: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Parsl python_app wrapper that runs process_folder on a worker.

        The real processing function is imported inside the app so the worker
        process imports the correct package layout and environment.
        """
        return process_folder(
            folder, multiwfn_cmd=multiwfn_cmd, orca_2mkl_cmd=orca_2mkl_cmd, **kwargs
        )

    @python_app
    def run_folder_task_alcf(
        folder: str,
        multiwfn_cmd: Optional[str] = None,
        orca_2mkl_cmd: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Parsl python_app wrapper that runs process_folder on a worker.

        The real processing function is imported inside the app so the worker
        process imports the correct package layout and environment.
        """
        return process_folder_alcf(
            folder, multiwfn_cmd=multiwfn_cmd, orca_2mkl_cmd=orca_2mkl_cmd, **kwargs
        )
