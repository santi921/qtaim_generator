import os
import logging
import time
from parsl import python_app
from qtaim_gen.source.core.omol import gbw_analysis
from typing import Optional, Dict, Any, List
import shutil
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
    result: Dict[str, Any] = {
        "folder": folder,
        "status": "unknown",
        "elapsed": None,
        "error": None,
    }
    # normalize to absolute path and set up logger
    folder = os.path.abspath(folder)
    logger: logging.Logger = setup_logger_for_folder(folder)

    # remember current working directory and switch into the folder while processing
    orig_cwd: str = os.getcwd()

    try:
        os.chdir(folder)
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

            tf_validation = validation_checks(
                folder, 
                full_set=full_set, 
                verbose=False,
                move_results=move_results,
                logger=logger
            )

            if not tf_validation:
                logger.info("Validation failed for %s: reprocessing", folder)
            else:
                logger.info("Validation passed for %s: skipping", folder)
                result["status"] = "skipped"
                return result

        # optional: check mwfn files, multiple mwfn guard
        mwfn_files: List[str] = [f for f in os.listdir(folder) if f.endswith(".mwfn")]
        if len(mwfn_files) > 1 and not overrun_running:
            logger.info("Skipping %s: multiple mwfn files found", folder)
            result["status"] = "skipped"
            return result
        # set env
        # set omp stacksize
        os.environ["OMP_STACKSIZE"] = omp_stacksize

        # call the existing function (pass logger)
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
        )
        t1: float = time.time()

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
        # restore original working directory
        try:
            os.chdir(orig_cwd)
        except Exception:
            # best-effort restore; if this fails, log to stderr
            print(f"Warning: failed to restore cwd to {orig_cwd}")


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
    full_set: bool = False,
    move_results: bool = True,
    root_omol_results: Optional[str] = None, # root where to store results, should mimic root_omol_inputs
    root_omol_inputs: Optional[str] = None, # root where input folders are located
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
    # remove root_omol_status from folder name 
    folder_inputs = folder 
    if root_omol_inputs and folder_inputs.startswith(root_omol_inputs):
        folder_relative = folder_inputs[len(root_omol_inputs) :].lstrip(os.sep)
    
    folder_outputs = root_omol_results + os.sep + folder_relative
    # copy files from folder_inputs to folder_outputs if they don't exist
    if not os.path.exists(folder_outputs):
        os.makedirs(folder_outputs)
    
    for item in os.listdir(folder_inputs):
        s = os.path.join(folder_inputs, item)
        d = os.path.join(folder_outputs, item)
        if os.path.isdir(s):
            if not os.path.exists(d):
                os.makedirs(d)
        else:
            if not os.path.exists(d):
                shutil.copy2(s, d)
    
    folder = os.path.abspath(folder_outputs)
    logger: logging.Logger = setup_logger_for_folder(folder)

    # remember current working directory and switch into the folder while processing
    orig_cwd: str = os.getcwd()

    try:
        os.chdir(folder)
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

        tf_validation = validation_checks(
            folder, 
            full_set=full_set, 
            verbose=False,
            move_results=move_results,
            logger=logger
        )

        if outputs_present and not overwrite and tf_validation:
            logger.info("Skipping %s: already processed and validated", folder)
            result["status"] = "skipped"
            return result

        # optional: check mwfn files, multiple mwfn guard
        mwfn_files: List[str] = [f for f in os.listdir(folder) if f.endswith(".mwfn")]
        if len(mwfn_files) > 1 and not overrun_running:
            logger.info("Skipping %s: multiple mwfn files found", folder)
            result["status"] = "skipped"
            return result
        # set env
        # set omp stacksize
        os.environ["OMP_STACKSIZE"] = omp_stacksize

        # call the existing function (pass logger)
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
        )
        t1: float = time.time()

        result["elapsed"] = t1 - t0
        result["status"] = "ok"
        logger.info("Completed folder %s in %.2f s", folder, result["elapsed"])
        

        # remove density_mat.npz, orca.gbw.zstd0, orca.gbw, orca.tar.zst from results folder
        files_to_remove = [
            "density_mat.npz",
            "orca.gbw.zstd0",
            "orca.gbw",
            "orca.tar.zst",
            "orca.inp.orig", 
            "orca.property.txt", 
            "orca.out",
            "orca.engrad", 
            "orca_stderr",

        ]
        results_folder = os.path.join(folder, "generator")
        for fn in files_to_remove:
            fp = os.path.join(folder, fn)
            if os.path.exists(fp):
                os.remove(fp)
                # add log
                logger.info("Removed file %s to save space", fp)

        return result
    
    except Exception as exc:
        logger.exception("Error processing %s: %s", folder, exc)
        result["status"] = "error"
        result["error"] = str(exc)
        return result

    finally:
        # restore original working directory
        try:
            os.chdir(orig_cwd)
        except Exception:
            # best-effort restore; if this fails, log to stderr
            print(f"Warning: failed to restore cwd to {orig_cwd}")



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
    # this runs inside remote worker; import inside to ensure worker env has package

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
    # this runs inside remote worker; import inside to ensure worker env has package

    return process_folder_alcf(
        folder, multiwfn_cmd=multiwfn_cmd, orca_2mkl_cmd=orca_2mkl_cmd, **kwargs
    )
