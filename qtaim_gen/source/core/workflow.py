import os
import json
import logging
import time
import zipfile
from typing import Optional, Dict, Any, List
import shutil

from qtaim_gen.source.core.omol import gbw_analysis
from qtaim_gen.source.utils.atomic_write import atomic_json_write
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

        if age < max_age_s or max_age_s == float("inf"):
            return False  # not stale, genuinely locked (or the caller never breaks locks)

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


_STASH = "generator.pre_clean"
_STASH_DONE = "generator.pre_clean.done"
_STASH_FAILED = "generator.failed_rerun"
_CLEAN_FIRST_KEEP = ("gbw_analysis.log", ".processing.lock", _STASH)
# top-level keys are steps (or a step's flattened properties in other.json)
_STEP_KEYED = ("charge.json", "bond.json", "fuzzy_full.json", "other.json", "timings.json")
_CLEAN_FIRST_NEEDS_MOVE ="clean_first requires move_results: the results sit in the job folder and would be deleted"
# ORCA outputs the local runner cannot re-stage from anywhere else
_LOCAL_INPUT_SUFFIXES = (".inp", ".gbw", ".gbw.zstd0", ".tar.zst", ".tgz")
_LOCAL_INPUT_NAMES = ("orca.out", "orca.property.txt", "orca.engrad", "density_mat.npz")


def _is_local_input(name: str) -> bool:
    return name.endswith(_LOCAL_INPUT_SUFFIXES) or name in _LOCAL_INPUT_NAMES


def _carry_forward(stash: str, gen: str, logger: logging.Logger) -> None:
    """Copy into a validated rerun's generator/ what it did not recompute:
    steps above its full_set (a level-0 rerun of a level-1 folder must not
    drop mbis/elf), whole files it never wrote (horton.json, or orca.json
    without check_orca), and archived .out entries it did not replace.
    qtaim.json and orca.json are never merged key by key: their keys are CPs
    and parsed fields, not steps. A kill part way leaves the stash in place,
    and the next pass restores it and reruns."""
    from qtaim_gen.source.utils.io import merge_zip_into

    carried = []
    for name in sorted(os.listdir(stash)):
        old_p, new_p = os.path.join(stash, name), os.path.join(gen, name)
        try:
            if name == "out_files.zip":
                tmp = new_p + ".carry"
                shutil.copy2(old_p, tmp)
                if os.path.exists(new_p):
                    merge_zip_into(new_p, tmp, logger=logger)  # new entries win
                os.replace(tmp, new_p)
                carried.append(name)
            elif not name.endswith(".json"):
                continue
            elif not os.path.exists(new_p):
                shutil.copy2(old_p, new_p)
                carried.append(name)
            elif name in _STEP_KEYED:
                with open(old_p) as f:
                    old = json.load(f)
                with open(new_p) as f:
                    new = json.load(f)
                missing = {k: v for k, v in old.items() if k not in new}
                if missing:
                    atomic_json_write(new_p, {**new, **missing})
                    carried.append(f"{name}[{','.join(sorted(missing))}]")
        except (OSError, ValueError, zipfile.BadZipFile) as e:
            # an unreadable old file has nothing to carry; failing here would
            # restore the stash over a validated rerun
            logger.warning("clean_first: could not carry forward %s: %s", old_p, e)
    if carried:
        logger.info("clean_first: carried forward from %s: %s", stash, "; ".join(carried))


def _settle_stash(folder: str, keep_new: bool, logger: logging.Logger) -> None:
    """Resolve a generator.pre_clean/ stash: drop it once the rerun validated,
    otherwise put it back over whatever partial generator/ the rerun wrote.
    Every step is a rename first, so a kill at any point leaves either the
    stash or the old results in place, never neither."""
    gen = os.path.join(folder, "generator")
    stash = os.path.join(folder, _STASH)
    for leftover in (_STASH_DONE, _STASH_FAILED):
        if os.path.isdir(os.path.join(folder, leftover)):
            shutil.rmtree(os.path.join(folder, leftover))
    if not os.path.isdir(stash):
        return
    if keep_new:
        _carry_forward(stash, gen, logger)
        os.rename(stash, os.path.join(folder, _STASH_DONE))
        shutil.rmtree(os.path.join(folder, _STASH_DONE))
        logger.info("clean_first rerun validated; dropped %s", stash)
        return
    if os.path.isdir(gen):
        os.rename(gen, os.path.join(folder, _STASH_FAILED))
    os.rename(stash, gen)
    shutil.rmtree(os.path.join(folder, _STASH_FAILED), ignore_errors=True)
    logger.warning("clean_first rerun did not validate; restored previous generator/ from %s", stash)


def _clean_first(folder: str, logger: logging.Logger, keep_inputs: bool = False) -> None:
    """Move generator/ aside to generator.pre_clean/ and remove the working
    files, so every step recomputes into an empty generator/: no stale QTAIM
    CP or step result can survive the merge or satisfy validation. The caller
    settles the stash when the rerun ends, and restores an orphaned stash
    from a killed rerun before calling this (_settle_stash).

    keep_inputs keeps the ORCA inputs/outputs in place (local runner, where
    the job folder is the only copy); the ALCF runner re-copies them from the
    input tree."""
    gen = os.path.join(folder, "generator")
    if os.path.isdir(gen):
        os.rename(gen, os.path.join(folder, _STASH))
        logger.info("clean_first: moved %s aside to %s", gen, _STASH)
    for item in os.listdir(folder):
        if item in _CLEAN_FIRST_KEEP or (keep_inputs and _is_local_input(item)):
            continue
        item_path = os.path.join(folder, item)
        try:
            if os.path.isfile(item_path) or os.path.islink(item_path):
                os.unlink(item_path)
                logger.info(f"Removed file {item_path} due to clean_first flag")
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)
                logger.info(f"Removed directory {item_path} due to clean_first flag")
        except Exception as e:
            logger.error(f"Failed to remove {item_path}. Reason: {e}")


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
    wfx: bool = True,
    check_orca: bool = False,
    check_bcp_count: bool = False,
    bcp_tolerance: int = 2,
    require_qtaim_provenance: bool = False,
    recheck_allalpha_qtaim: bool = False,
    recheck_cp_labels: bool = False,
    enforce_poincare_hopf: bool = False,
    exhaustive_qtaim: bool = False,
    patch_timings: bool = False,
    horton_python: str = "",
    recheck_fuzzy: bool = False,
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
    if clean_first and not move_results:
        result["status"] = "error"
        result["error"] = _CLEAN_FIRST_NEEDS_MOVE
        return result
    # normalize to absolute path and set up logger
    folder = os.path.abspath(folder)
    logger: logging.Logger = setup_logger_for_folder(folder)

    # Acquire lock before any work
    if not acquire_lock(folder):
        logger.info("Skipping %s: folder locked by active process", folder)
        result["status"] = "skipped"
        result["error"] = "folder locked by active process"
        return result

    rerun_ok = False
    try:
        _settle_stash(folder, keep_new=False, logger=logger)
        if clean_first:
            _clean_first(folder, logger, keep_inputs=True)
            overwrite, restart = True, False

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
                    recheck_fuzzy=recheck_fuzzy,
                    recheck_allalpha_qtaim=recheck_allalpha_qtaim,
                    recheck_cp_labels=recheck_cp_labels,
                    enforce_poincare_hopf=enforce_poincare_hopf,
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
        tf_validation = gbw_analysis(
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
            check_bcp_count=check_bcp_count,
            bcp_tolerance=bcp_tolerance,
            require_qtaim_provenance=require_qtaim_provenance,
            recheck_allalpha_qtaim=recheck_allalpha_qtaim,
            recheck_cp_labels=recheck_cp_labels,
            enforce_poincare_hopf=enforce_poincare_hopf,
            exhaustive_qtaim=exhaustive_qtaim,
            subprocess_env=subprocess_env,
            patch_timings=patch_timings,
            horton_python=horton_python,
            recheck_fuzzy=recheck_fuzzy,
        )
        t1: float = time.time()
        rerun_ok = bool(tf_validation)

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
        # orca.gbw.zstd0/orca.tar.zst are the only wavefunction source a retry
        # has. Deleting them after a failed validation left the folder unable
        # to be reprocessed at all without re-staging from the source tree.
        if tf_validation:
            for fn in files_to_remove:
                fp = os.path.join(folder, fn)
                if os.path.exists(fp):
                    os.remove(fp)
                    # add log
                    logger.info("Removed file %s to save space", fp)
        else:
            logger.warning(
                "Validation failed for %s - keeping compressed sources for retry",
                folder,
            )

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
        if clean_first:
            try:
                _settle_stash(folder, keep_new=rerun_ok, logger=logger)
            except Exception as e:
                logger.error("Could not settle %s in %s: %s", _STASH, folder, e)
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
    wfx: bool = True,
    check_orca: bool = False,
    check_bcp_count: bool = False,
    bcp_tolerance: int = 2,
    require_qtaim_provenance: bool = False,
    recheck_allalpha_qtaim: bool = False,
    recheck_cp_labels: bool = False,
    enforce_poincare_hopf: bool = False,
    exhaustive_qtaim: bool = False,
    patch_timings: bool = False,
    horton_python: str = "",
    recheck_fuzzy: bool = False,
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
    if clean_first and not move_results:
        result["status"] = "error"
        result["error"] = _CLEAN_FIRST_NEEDS_MOVE
        return result

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

    rerun_ok = False
    try:
        _settle_stash(folder, keep_new=False, logger=logger)
        if clean_first:
            _clean_first(folder, logger)
            overwrite, restart = True, False

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
                check_bcp_count=check_bcp_count,
                bcp_tolerance=bcp_tolerance,
                require_qtaim_provenance=require_qtaim_provenance,
                recheck_allalpha_qtaim=recheck_allalpha_qtaim,
                recheck_cp_labels=recheck_cp_labels,
                enforce_poincare_hopf=enforce_poincare_hopf,
                recheck_fuzzy=recheck_fuzzy,
            )

            if not overwrite and tf_validation:
                logger.info("Skipping %s: already processed and validated", folder)
                result["status"] = "skipped"

                try:
                    zip_file_out = os.path.join(folder, "out_files.zip")
                    if not os.path.exists(zip_file_out):
                        files_to_zip = [
                            f for f in os.listdir(folder)
                            if f.endswith(".out") and f != "orca.out"
                        ]
                        with zipfile.ZipFile(zip_file_out, "w") as zipf:
                            for file in files_to_zip:
                                zipf.write(os.path.join(folder, file), arcname=file)
                                logger.info(f"Zipped {file}")
                    else:
                        files_to_zip = []

                    if move_results:
                        from qtaim_gen.source.utils.io import merge_zip_into
                        results_folder = os.path.join(folder, "generator")
                        merge_zip_into(
                            zip_file_out,
                            os.path.join(results_folder, "out_files.zip"),
                            logger=logger,
                        )

                    for file in files_to_zip:
                        fp = os.path.join(folder, file)
                        if os.path.exists(fp):
                            os.remove(fp)
                            logger.info(f"Removed {file} after zip")

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
        tf_validation = gbw_analysis(
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
            check_bcp_count=check_bcp_count,
            bcp_tolerance=bcp_tolerance,
            require_qtaim_provenance=require_qtaim_provenance,
            recheck_allalpha_qtaim=recheck_allalpha_qtaim,
            recheck_cp_labels=recheck_cp_labels,
            enforce_poincare_hopf=enforce_poincare_hopf,
            exhaustive_qtaim=exhaustive_qtaim,
            subprocess_env=subprocess_env,
            patch_timings=patch_timings,
            horton_python=horton_python,
            recheck_fuzzy=recheck_fuzzy,
        )
        t1: float = time.time()
        rerun_ok = bool(tf_validation)

        # See process_folder: the compressed sources are the only thing a
        # retry can rebuild the wavefunction from, so a failed validation
        # keeps them.
        if clean and tf_validation:
            for fn in files_to_remove:
                fp = os.path.join(folder, fn)
                if os.path.exists(fp):
                    os.remove(fp)
                    logger.info("Removed file %s to save space", fp)
        elif clean:
            logger.warning(
                "Validation failed for %s - keeping compressed sources for retry",
                folder,
            )

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
        if clean_first:
            try:
                _settle_stash(folder, keep_new=rerun_ok, logger=logger)
            except Exception as e:
                logger.error("Could not settle %s in %s: %s", _STASH, folder, e)
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
