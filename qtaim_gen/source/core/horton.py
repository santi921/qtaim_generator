"""HORTON charge engine orchestrator.

Runs the self-contained HORTON worker (scripts/helpers/horton_worker.py) in a
separate python environment (horton-part needs scipy<1.15, incompatible with
the main env) and merges the resulting *_horton charge schemes into charge.json,
mirroring the ORCA post-step (_run_orca_parse / merge_orca_into_charge_json).
"""

import json
import logging
import os
import subprocess
import tempfile
import time
from typing import Optional

from qtaim_gen.source.utils.atomic_write import atomic_json_write
from qtaim_gen.source.utils.io import find_wfx

HORTON_SCHEMES = ("becke", "hirshfeld", "is")

EDF_TAG = "<Additional Electron Density Function (EDF)>"
EDF_END = "</Additional Electron Density Function (EDF)>"


def strip_edf(wfx_text: str) -> str:
    """Remove the EDF block Multiwfn writes into ECP wfx files.

    iodata's unknown-section skipper cannot step over the nested EDF tags and
    raises a LoadError. HORTON cannot use the EDF core density anyway (it
    partitions the valence density against effective core charges), so the
    block is dropped. The trailing newline must go with it -- a blank line
    outside a section is also a parse error.
    """
    i = wfx_text.find(EDF_TAG)
    if i < 0:
        return wfx_text
    j = wfx_text.index(EDF_END) + len(EDF_END)
    while j < len(wfx_text) and wfx_text[j] in "\r\n":
        j += 1
    return wfx_text[:i] + wfx_text[j:]


def find_horton_json(folder: str) -> Optional[str]:
    """Locate a non-empty horton.json in the folder root or generator/."""
    for cand in (
        os.path.join(folder, "horton.json"),
        os.path.join(folder, "generator", "horton.json"),
    ):
        if os.path.isfile(cand) and os.path.getsize(cand) > 0:
            return cand
    return None


def resolve_charge_json(folder: str, move_results: bool) -> str:
    """Charge.json path to merge into: generator/ when results were moved."""
    charge_path = os.path.join(folder, "charge.json")
    if move_results:
        gen_charge = os.path.join(folder, "generator", "charge.json")
        if os.path.isfile(gen_charge):
            return gen_charge
    return charge_path


def merge_horton_into_charge_json(horton_dict: dict, charge_json_path: str) -> None:
    """Merge HORTON charge schemes into existing charge.json.

    Adds *_horton method keys. Idempotent. Skips if charge.json doesn't exist
    (rebuttal-style folders carry horton.json only).
    """
    if not os.path.isfile(charge_json_path):
        return

    try:
        with open(charge_json_path, "r") as f:
            charge_data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return

    modified = False
    for key, entry in horton_dict.items():
        if key.endswith("_horton") and isinstance(entry, dict) and "charge" in entry:
            charge_data[key] = entry
            modified = True

    if modified:
        atomic_json_write(charge_json_path, charge_data)


def _write_horton_timing(folder: str, elapsed: float) -> None:
    """Append a 'horton' timing key to timings.json (root wins over generator/)."""
    gen_timings = os.path.join(folder, "generator", "timings.json")
    root_timings = os.path.join(folder, "timings.json")
    timings = {}
    for path in (gen_timings, root_timings):
        if os.path.isfile(path) and os.path.getsize(path) > 0:
            try:
                with open(path, "r") as f:
                    timings.update(json.load(f))
            except json.JSONDecodeError:
                pass
    if timings:
        timings["horton"] = elapsed
        atomic_json_write(root_timings, timings)


def run_horton_analysis(
    folder: str,
    horton_python: str,
    schemes: str = "becke,hirshfeld,is",
    grid: str = "fine",
    timeout: int = 3600,
    subprocess_env: Optional[dict] = None,
    move_results: bool = False,
    overwrite: bool = False,
    logger: Optional[logging.Logger] = None,
) -> bool:
    """Compute HORTON charges for a job folder and merge into charge.json.

    Locates orca.wfx, strips the EDF block if present, runs the horton worker
    in the given python environment, writes horton.json next to the wfx's
    folder root, merges *_horton keys into charge.json (root or generator/),
    and records a 'horton' timing. Returns True on success.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    horton_json_path = os.path.join(folder, "horton.json")
    existing = None if overwrite else find_horton_json(folder)
    if existing is not None:
        # Skip the expensive recompute, but always (re)attempt the merge: it is
        # cheap and idempotent, and a run interrupted between writing
        # horton.json and merging would otherwise never integrate its results.
        logger.info("horton.json already present in %s -- merging only", folder)
        try:
            with open(existing, "r") as f:
                merge_horton_into_charge_json(
                    json.load(f), resolve_charge_json(folder, move_results)
                )
        except (json.JSONDecodeError, OSError) as e:
            logger.error("Could not merge existing %s: %s", existing, e)
            return False
        return True

    wfx_path = find_wfx(folder)
    if wfx_path is None:
        logger.info("No orca.wfx found in %s -- skipping HORTON analysis", folder)
        return False

    worker_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "scripts",
        "helpers",
        "horton_worker.py",
    )

    env = dict(subprocess_env) if subprocess_env else dict(os.environ)
    env.setdefault("OMP_NUM_THREADS", "4")

    t_start = time.time()
    tmp_wfx = None
    try:
        with open(wfx_path, "r") as f:
            text = f.read()
        stripped = strip_edf(text)
        run_wfx = wfx_path
        if stripped is not text:
            fd, tmp_wfx = tempfile.mkstemp(suffix=".wfx", prefix="horton_noedf_")
            with os.fdopen(fd, "w") as f:
                f.write(stripped)
            run_wfx = tmp_wfx

        cmd = [
            horton_python,
            worker_path,
            "--wfx",
            run_wfx,
            "--out",
            horton_json_path,
            "--schemes",
            schemes,
            "--grid",
            grid,
        ]
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, env=env
        )
        if result.returncode != 0:
            logger.error(
                "HORTON worker failed for %s (rc=%d): %s",
                folder,
                result.returncode,
                (result.stderr or result.stdout or "").strip()[-500:],
            )
            return False

        with open(horton_json_path, "r") as f:
            horton_dict = json.load(f)

        merge_horton_into_charge_json(
            horton_dict, resolve_charge_json(folder, move_results)
        )

        elapsed = round(time.time() - t_start, 2)
        _write_horton_timing(folder, elapsed)
        skipped = [
            s.get("scheme", "?")
            for s in horton_dict.get("_meta", {}).get("schemes_skipped", [])
        ]
        logger.info(
            "HORTON charges for %s in %.2f s%s",
            folder,
            elapsed,
            f" (skipped: {','.join(skipped)})" if skipped else "",
        )
        return True

    except subprocess.TimeoutExpired:
        logger.error("HORTON worker timed out after %d s for %s", timeout, folder)
        return False
    except Exception as e:
        logger.error("Error in HORTON analysis for %s: %s", folder, e)
        return False
    finally:
        if tmp_wfx is not None:
            try:
                os.remove(tmp_wfx)
            except OSError:
                pass
