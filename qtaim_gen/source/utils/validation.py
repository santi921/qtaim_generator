from asyncio.log import logger
import os
import json
import re
import shutil
import tempfile
import zipfile
from typing import Optional
from qtaim_gen.source.core.parse_qtaim import dft_inp_to_dict
from qtaim_gen.source.core.parse_orca import ORCA_PARSER_VERSION
import numpy as np
from datetime import datetime


# Shared contract between validate_timing_dict (consumer) and
# patch_timings_from_log in core/omol.py (producer). Keep in sync at one place
# so a rename of either the marker key or the sentinel is a one-line change.
TIMINGS_PATCHED_KEY = "_timings_patched"
TIMING_PLACEHOLDER = -1.0


def _safe_json_load(path: str, logger=None):
    """Load JSON from *path*, returning None on missing/empty/malformed file.

    Validators must surface bad-JSON as a False validation result, not raise
    out of validation_checks -- otherwise a single corrupted file (e.g.
    trailing-comma charge.json from a pre-atomic-write era) kills the
    whole gbw_analysis caller and the folder loops on HPC.
    """
    if not os.path.isfile(path):
        if logger:
            logger.error("Missing JSON file: %s", path)
        return None
    try:
        if os.path.getsize(path) == 0:
            if logger:
                logger.error("Empty JSON file: %s", path)
            return None
        with open(path, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        if logger:
            logger.error("Cannot read JSON %s: %s", path, e)
        return None


# ORCA writes orca.property.inp beside orca.inp and the pipeline never deletes
# it (clean_jobs sweeps .txt/.mfwn/molden/wavefunctions, not .inp). It carries
# no "* xyz" block, so it is not a geometry input. Picking it silently reports
# the wrong atom count, or crashes the parser outright.
NON_GEOMETRY_INPUTS = ("convert.in",)
NON_GEOMETRY_SUFFIXES = (".property.inp",)
# Canonical geometry inputs, best first.
PREFERRED_INPUTS = ("orca.inp", "input.inp", "input.in")


def geometry_input_candidates(folder: str) -> list:
    """Geometry input files in folder, best candidate first.

    Deterministic by construction. os.listdir order is arbitrary and shifts
    whenever a file is added to the directory, so selecting its first entry
    made the parsed molecule depend on unrelated folder contents.
    """
    names = [
        f
        for f in os.listdir(folder)
        if f.endswith((".inp", ".in"))
        and f not in NON_GEOMETRY_INPUTS
        and not f.endswith(NON_GEOMETRY_SUFFIXES)
    ]
    preferred = [f for f in PREFERRED_INPUTS if f in names]
    rest = sorted(f for f in names if f not in preferred)
    return preferred + rest


def get_charge_spin_n_atoms_from_folder(
    folder: str, logger=None, verbose=False
) -> tuple:
    inp_files = geometry_input_candidates(folder)

    if not inp_files:
        if logger:
            logger.error(f"No .inp file found in folder: {folder}.")
        if verbose:
            print(f"No .inp file found in folder: {folder}.")
        return False

    # Try candidates in order: a folder may hold a stale or truncated input
    # alongside the real one, and the first name is a preference, not a promise.
    last_error = None
    for inp_file in inp_files:
        orca_inp_path = os.path.join(folder, inp_file)
        try:
            parsed = dft_inp_to_dict(orca_inp_path, parse_charge_spin=True)
        except Exception as e:
            last_error = f"{inp_file}: {type(e).__name__}: {e}"
            if logger:
                logger.warning(f"Could not parse input {orca_inp_path}: {e}")
            continue

        if logger:
            logger.info(f'Using input file "{inp_file}" for validation.')
        if verbose:
            print(f'Using input file "{inp_file}" for validation.')
        return parsed

    if logger:
        logger.error(
            f"No parsable geometry input in {folder} "
            f"(tried {inp_files}); last error: {last_error}"
        )
    if verbose:
        print(
            f"No parsable geometry input in {folder} "
            f"(tried {inp_files}); last error: {last_error}"
        )
    return False


def get_val_breakdown_from_folder(
    folder: str, full_set: int, spin_tf: bool, n_atoms: int
) -> dict:

    info = {
        "total_time": None,
        "t_qtaim": None,
        "t_charge": None,
        "t_bond": None,
        "t_fuzzy": None,
        "t_other": None,
        "val_time": None,
        "val_qtaim": None,
        "val_charge": None,
        "val_bond": None,
        "val_fuzzy": None,
        "val_other": None,
        "has_orca_json": False,
        "val_orca": None,
    }

    # check timings
    timings_file = os.path.join(folder, "timings.json")
    if os.path.exists(timings_file) and os.path.getsize(timings_file) > 0:
        with open(timings_file, "r") as f:
            timings = json.load(f)
        total_time = np.array(list(timings.values())).sum()
        info["total_time"] = total_time

        for col in timings.keys():
            info[f"t_{col}"] = timings[col]
        val_time = validate_timing_dict(
            timings_file, logger=None, full_set=full_set, spin_tf=spin_tf
        )
        info["val_time"] = val_time

    # check fuzzy
    fuzzy_file = os.path.join(folder, "fuzzy_full.json")
    if os.path.exists(fuzzy_file) and os.path.getsize(fuzzy_file) > 0:
        tf_fuzzy = validate_fuzzy_dict(
            fuzzy_file,
            logger=None,
            n_atoms=n_atoms,
            spin_tf=spin_tf,
            full_set=full_set,
        )
        info["val_fuzzy"] = tf_fuzzy

    # check charge
    charge_file = os.path.join(folder, "charge.json")
    if os.path.exists(charge_file) and os.path.getsize(charge_file) > 0:
        tf_charge = validate_charge_dict(charge_file, logger=None)
        info["val_charge"] = tf_charge

    # check bond
    bond_file = os.path.join(folder, "bond.json")
    if os.path.exists(bond_file) and os.path.getsize(bond_file) > 0:
        tf_bond = validate_bond_dict(bond_file, logger=None)
        info["val_bond"] = tf_bond

    # check qtaim
    qtaim_file = os.path.join(folder, "qtaim.json")
    if os.path.exists(qtaim_file) and os.path.getsize(qtaim_file) > 0:
        tf_qtaim = validate_qtaim_dict(qtaim_file, n_atoms=n_atoms, logger=None)
        info["val_qtaim"] = tf_qtaim

    # echeck other
    other_file = os.path.join(folder, "other.json")
    if os.path.exists(other_file) and os.path.getsize(other_file) > 0:
        tf_other = validate_other_dict(
            other_file,
            logger=None,
            full_set=full_set
        )
        info["val_other"] = tf_other

    # check orca (optional)
    orca_file = os.path.join(folder, "orca.json")
    if os.path.exists(orca_file):
        info["has_orca_json"] = True
        if os.path.getsize(orca_file) > 0:
            info["val_orca"] = validate_orca_dict(orca_file, n_atoms=n_atoms, logger=None)
        else:
            info["val_orca"] = False

    return info


def get_expected_timing_keys(full_set: int = 0, spin_tf: bool = False) -> tuple:
    """Return (expected_keys, expected_spin_keys) for the given analysis level.

    Single source of truth shared by validate_timing_dict (consumer) and
    patch_timings_from_log (producer-side recovery in omol.py).

    Note: 'other' in expected_keys is satisfied by either 'other' or
    'other_alie' in the timings dict — see validate_timing_dict for that
    aliasing logic.
    """
    expected_keys = [
        "qtaim",
        "other",
        "hirshfeld",
        "becke",
        "adch",
        "cm5",
        "fuzzy_bond",
        "becke_fuzzy_density",
        "hirsh_fuzzy_density",
    ]
    expected_spin_keys = ["hirsh_fuzzy_spin", "becke_fuzzy_spin"]

    if full_set > 0:
        expected_keys += [
            "vdd",
            "mbis",
            "chelpg",
            "ibsi_bond",
            "elf_fuzzy",
            "mbis_fuzzy_density",
        ]
        expected_spin_keys += ["mbis_fuzzy_spin"]

    if full_set > 1:
        expected_keys += [
            "bader",
            "laplacian_bond",
            "grad_norm_rho_fuzzy",
            "laplacian_rho_fuzzy",
            "ESP_Volume",
        ]

    if spin_tf:
        expected_keys = expected_keys + expected_spin_keys

    return expected_keys, expected_spin_keys


def validate_timing_dict(
    timing_json_loc: str,
    verbose: bool = False,
    full_set: int = 0,
    spin_tf: bool = False,
    logger: any = None,
    n_atoms: int = None,
):
    """
    Basic check that the timing json file has the expected structure.
    Check that it has the keys 'total', 'qtaim', 'charge', 'bond', and 'fuzzy_full'.
    """
    timing_dict = _safe_json_load(timing_json_loc, logger=logger)
    if timing_dict is None:
        return False

    expected_keys, excepted_spin_keys = get_expected_timing_keys(
        full_set=full_set, spin_tf=False
    )

    # Keys patched by patch_timings_from_log carry a TIMING_PLACEHOLDER (-1.0)
    # when log-scrape couldn't find them; accept those here so cleanup runs.
    patched_keys = set((timing_dict.get(TIMINGS_PATCHED_KEY) or {}).keys())

    for key in expected_keys:
        if key not in timing_dict:
            if key == "other" and "other_alie" not in timing_dict: 
                if logger:
                    logger.error(f"Missing expected key '{key}' or 'other_alie' in timing json.")
                if verbose:
                    print(f"Missing expected key '{key}' or 'other_alie' in timing json.")
                return False
            
            elif key == "other" and "other_alie" in timing_dict:
                key = "other_alie"

            else: 
                if logger:
                    logger.error(f"Missing expected key '{key}' in timing json.")
                if verbose:
                    print(f"Missing expected key '{key}' in timing json.")
                return False

        # check that the times aren't tiny
        # For small molecules (n_atoms <= 2), bond-related timings may be
        # legitimately near-zero since there are few or no bonds to analyze
        bond_related_keys = {
            "fuzzy_bond", "ibsi_bond", "laplacian_bond",
            "becke_fuzzy_density", "hirsh_fuzzy_density",
            "elf_fuzzy", "mbis_fuzzy_density",
            "grad_norm_rho_fuzzy", "laplacian_rho_fuzzy",
        }
        is_small_molecule = n_atoms is not None and n_atoms <= 2
        if timing_dict[key] < 1e-6 and key != "convert":
            if is_small_molecule and key in bond_related_keys:
                continue  # acceptable for small molecules
            if key in patched_keys:
                if logger:
                    logger.warning(
                        f"Timing for '{key}' is patched ({timing_dict[key]}); "
                        "accepting via _timings_patched marker."
                    )
                continue
            if logger:
                logger.error(
                    f"Timing for '{key}' is too small: {timing_dict[key]} seconds."
                )
            print(f"Timing for '{key}' is too small: {timing_dict[key]} seconds.")
            return False

    if spin_tf:
        for key in excepted_spin_keys:
            if key not in timing_dict:
                if verbose:
                    print(f"Missing expected spin key '{key}' in timing json.")
                if logger:
                    logger.error(f"Missing expected spin key '{key}' in timing json.")
                return False

    if verbose:
        print("Timing json structure is valid.")
    return True


def validate_bond_dict(
    bond_json_loc: str, verbose: bool = False, full_set: int = 0, logger: any = None,
    n_atoms: int = None,
):
    """
    Basic check that the bond json file has the expected structure.
    Check that it has the keys 'fuzzy_bond', 'ibsi_bond', and 'laplacian_bond'.

    For small molecules (n_atoms <= 2), missing bond keys are acceptable
    since there may be no bonds to analyze.
    """
    bond_dict = _safe_json_load(bond_json_loc, logger=logger)
    if bond_dict is None:
        return False

    expected_keys = ["fuzzy_bond"]

    if full_set > 0:
        expected_keys += ["ibsi_bond"]
    if full_set > 1:
        expected_keys += ["laplacian_bond"]

    is_small_molecule = n_atoms is not None and n_atoms <= 2

    for key in expected_keys:
        if key not in bond_dict:
            if is_small_molecule:
                continue  # acceptable for small molecules
            if verbose:
                print(f"Missing expected key '{key}' in bond json.")
            if logger:
                logger.error(f"Missing expected key '{key}' in bond json.")
            return False

    if verbose:
        print("Bond json structure is valid.")

    return True


def validate_fuzzy_dict(
    fuzzy_json_loc: str,
    n_atoms: int = None,
    spin_tf: bool = False,
    verbose: bool = False,
    full_set: int = 0,
    logger: any = None,
):
    """
    Basic check that the fuzzy json file has the expected structure.
    Check that it has the keys 'fuzzy', 'fuzzy_bonds', 'fuzzy_bcp', 'fuzzy_ncp'.
    """
    fuzzy_dict = _safe_json_load(fuzzy_json_loc, logger=logger)
    if fuzzy_dict is None:
        return False

    expected_keys = [
        "becke_fuzzy_density",
        "hirsh_fuzzy_density",
    ]

    if full_set > 0:
        expected_keys += ["elf_fuzzy", "mbis_fuzzy_density"]

    if full_set > 1:
        expected_keys += ["grad_norm_rho_fuzzy", "laplacian_rho_fuzzy"]

    if spin_tf:
        expected_keys += ["hirsh_fuzzy_spin", "becke_fuzzy_spin"]

        if full_set > 0:
            expected_keys += ["mbis_fuzzy_spin"]

    for key in expected_keys:
        if key not in fuzzy_dict:
            if logger:
                logger.error(f"Missing expected key '{key}' in fuzzy json.")
            return False
        if n_atoms is not None:
            if len(fuzzy_dict[key]) != n_atoms + 2:
                if verbose:
                    print(
                        f"Number of fuzzy points ({len(fuzzy_dict[key])}) does not match expected ({n_atoms})."
                    )
                if logger:
                    logger.error(
                        f"Number of fuzzy points ({len(fuzzy_dict[key])}) does not match expected ({n_atoms})."
                    )
                return False
    if verbose:
        print("Fuzzy json structure is valid.")
    return True


def validate_other_dict(other_dict_loc: str, verbose: bool = False, logger: any = None, full_set: int = 0):
    """
    Basic check that the other json file has the expected structure.
    Check that it has the keys 'atoms', 'bonds', 'charges', and 'fuzzy'.
    """
    other_dict = _safe_json_load(other_dict_loc, logger=logger)
    if other_dict is None:
        return False

    expected_keys = [
        "mpp_full",
        "sdp_full",
        "mpp_heavy",
        "sdp_heavy",
        "ALIE_Volume",
        "ALIE_Surface_Density",
        "ALIE_Minimal_value",
        "ALIE_Maximal_value",
        "ALIE_Overall_surface_area",
        "ALIE_Positive_surface_area",
        "ALIE_Negative_surface_area",
        "ALIE_Overall_skewness",
    ]
    if full_set > 1:
        expected_keys += [
            "ESP_Volume",
            "ESP_Surface_Density",
            "ESP_Minimal_value",
            "ESP_Maximal_value",
            "ESP_Overall_surface_area",
            "ESP_Positive_surface_area",
            "ESP_Negative_surface_area",
            "ESP_Overall_skewness",
        ]

    for key in expected_keys:
        if key not in other_dict:
            if verbose:
                print(
                    f"Warning: Missing expected key '{key}' in other json. This may not be critical."
                )
            if logger:
                logger.warning(
                    f"Missing expected key '{key}' in other json. This may not be critical."
                )
            return False

    if verbose:
        print("Other json structure is valid.")
    return True


def validate_charge_dict(
    charge_json_loc: str,
    n_atoms: int = None,
    verbose: bool = False,
    full_set: int = 0,
    logger: any = None,
):
    """
    Basic check that the charge json file has the expected structure.
    Check that it has the keys 'mbis', 'adch', 'chelpg', 'becke',  'hirshfeld', 'cm5', 'bader', 'vdd'
    Check each one of these keys has a key "charge" with n_atoms entries.
    """
    charge_dict = _safe_json_load(charge_json_loc, logger=logger)
    if charge_dict is None:
        return False

    expected_keys = ["adch", "becke", "hirshfeld", "cm5"]

    if full_set > 0:
        expected_keys += ["mbis", "vdd", "chelpg"]
    if full_set > 1:
        expected_keys += ["bader"]

    for key in expected_keys:
        if key not in charge_dict:
            if verbose:
                print(f"Warning: Missing expected key '{key}' in charge json. ")
            if logger:
                logger.warning(
                    f"Warning: Missing expected key '{key}' in charge json. "
                )
            return False

    for key in expected_keys:
        if "charge" not in charge_dict[key]:
            if verbose:
                print(f"Missing 'charge' key in '{key}' of charge json.")
            if logger:
                logger.error(f"Missing 'charge' key in '{key}' of charge json.")
            return False

        if n_atoms is not None:
            if len(charge_dict[key]["charge"]) != n_atoms:
                if verbose:
                    print(
                        f"Number of charges in '{key}' ({len(charge_dict[key]['charge'])}) does not match expected ({n_atoms})."
                    )
                if logger:
                    logger.error(
                        f"Number of charges in '{key}' ({len(charge_dict[key]['charge'])}) does not match expected ({n_atoms})."
                    )
                return False
    if verbose:
        print("Charge json structure is valid.")
    return True


QTAIM_EXPORT_MARKER = "have been outputted to CPprop.txt"
QTAIM_COUNT_PATTERN = re.compile(r"Number of \(3,-1\) CPs:\s*(\d+)")


def _multiwfn_out_texts(folder: str, name: str):
    """Every copy of a Multiwfn `<step>.out`, most recent location first: root, generator/,
    generator/out_files.zip (move_results), then the root out_files.zip (move_results=False)."""
    for rel in (name, os.path.join("generator", name)):
        path = os.path.join(folder, rel)
        if os.path.isfile(path) and os.path.getsize(path) > 0:
            try:
                with open(path, "r", errors="replace") as f:
                    yield f.read()
            except OSError:
                pass
    for zip_path in (os.path.join(folder, "generator", "out_files.zip"), os.path.join(folder, "out_files.zip")):
        if os.path.isfile(zip_path):
            try:
                with zipfile.ZipFile(zip_path, "r") as zf:
                    if name in zf.namelist():
                        yield zf.read(name).decode("utf-8", errors="replace")
            except (zipfile.BadZipFile, OSError, KeyError):
                pass


def read_multiwfn_out(folder: str, name: str) -> Optional[str]:
    """Text of a Multiwfn `<step>.out`, or None: the first copy _multiwfn_out_texts finds."""
    return next(_multiwfn_out_texts(folder, name), None)


def read_qtaim_out(folder: str) -> Optional[str]:
    """Multiwfn's qtaim.out text, or None (see read_multiwfn_out)."""
    return read_multiwfn_out(folder, "qtaim.out")


def qtaim_run_status(folder: str) -> dict:
    """Whether Multiwfn's QTAIM step actually ran to completion.

    Multiwfn prints the CP count at the end of the *search*, then exports the
    per-CP properties to CPprop.txt and prints a completion line. Both markers
    together distinguish the ways the step can end early:

    - no qtaim.out            -> unknown; absence of evidence, not evidence of
                                 completeness
    - no count line           -> killed during the CP search
    - count but no export line -> killed during the CPprop.txt write, so the
                                 file that got parsed is partial

    Only the last case is visible by comparing counts; the others need these
    markers, which is why "count matched" alone must not be read as "complete".
    """
    text = read_qtaim_out(folder)
    if text is None:
        return {"have_qtaim_out": False, "search_done": None, "export_done": None,
                "reported_bcp": None}
    found = QTAIM_COUNT_PATTERN.findall(text)
    return {
        "have_qtaim_out": True,
        "search_done": bool(found),
        "export_done": QTAIM_EXPORT_MARKER in text,
        "reported_bcp": int(found[-1]) if found else None,
    }


# How many bond CPs may be missing before a record counts as defective.
# Measured on 19 residual jobs from a repair test: 17 were missing exactly one
# CP and 2 were missing two, and the count did not scale with system size (one
# missing out of 299 blocks, one out of 27). That flat 1-2 is a single
# pathological CP per molecule whose gradient path cannot be traced to two
# nuclei, so it has no storable atom pair -- unrepairable by definition. An
# absolute tolerance matches that; a fractional one would be lenient on large
# systems and strict on small ones, which is backwards.
DEFAULT_BCP_TOLERANCE = 2


def as_tristate(value) -> Optional[bool]:
    """Read a bool-ish audit field that may have been through a CSV.

    Audit rows are consumed two ways -- straight from audit_folder (real bools,
    None for unknown) and out of a CSV via DictReader (the strings "True",
    "False", ""). Comparing against one form silently mishandles the other, and
    for these fields the wrong answer is the dangerous direction: a job with no
    QTAIM output at all reads as a known-good control and never gets requeued.
    """
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in ("true", "1"):
        return True
    if text in ("false", "0"):
        return False
    return None


def storable_bcp_count(folder: str) -> Optional[int]:
    """How many bond CPs the atom-pair-keyed schema can actually hold.

    Multiwfn's reported (3,-1) count is an upper bound, not a target. Three
    kinds of CP are legitimately unstorable, and measured on a 100-job repair
    test they accounted for 21 of 28 residual shortfalls:

    - no "Connected atoms:" line, so the CP has no attributable atom pair
      (18 of 28 -- by far the most common)
    - two CPs resolving to the same pair, which the merge collapses because it
      keys a plain dict on that pair (2 of 28)
    - an attractor with no nuclear-CP match (1 of 28)

    Comparing against this instead of the raw count is what keeps the
    completeness check from rejecting records that are already as complete as
    the schema permits -- which would otherwise livelock, since the restart
    path reruns anything that fails validation and the rerun reproduces the
    same result exactly.

    Returns None when CPprop.txt is unavailable (it is only archived by runs
    after the fix that stopped deleting it before the zip was built).
    """
    from qtaim_gen.source.core.parse_qtaim import get_qtaim_descs, only_atom_cps

    text_path = None
    tmpdir = None
    for rel in ("CPprop.txt", os.path.join("generator", "CPprop.txt")):
        cand = os.path.join(folder, rel)
        if os.path.isfile(cand) and os.path.getsize(cand) > 0:
            text_path = cand
            break
    if text_path is None:
        zip_path = os.path.join(folder, "generator", "out_files.zip")
        if os.path.isfile(zip_path):
            try:
                with zipfile.ZipFile(zip_path, "r") as zf:
                    if "CPprop.txt" in zf.namelist():
                        tmpdir = tempfile.mkdtemp(prefix="cpprop_val_")
                        zf.extract("CPprop.txt", tmpdir)
                        text_path = os.path.join(tmpdir, "CPprop.txt")
            except (zipfile.BadZipFile, OSError, KeyError):
                return None
    if text_path is None:
        return None

    try:
        _atoms, bonds = only_atom_cps(get_qtaim_descs(text_path))
        pairs = {
            tuple(sorted(v["connected_bond_paths"]))
            for v in bonds.values()
            if v.get("connected_bond_paths")
        }
        return len(pairs)
    except Exception:
        return None
    finally:
        if tmpdir:
            shutil.rmtree(tmpdir, ignore_errors=True)


def count_reported_bcps(folder: str) -> Optional[int]:
    """Number of (3,-1) CPs Multiwfn *reported*, or None if unavailable.

    Comparing this to the bond-CP count in qtaim.json detects critical points
    lost between the search and the stored record - most importantly a
    truncated CPprop.txt, which validate_qtaim_dict's nuclear-CP check cannot
    see because Multiwfn numbers nuclear CPs first, so any surviving prefix
    still satisfies it. See qtaim_run_status for the completeness markers this
    count cannot provide.
    """
    return qtaim_run_status(folder)["reported_bcp"]


QTAIM_ALL_ALPHA, QTAIM_ALL_ALPHA_ECP = "all_alpha", "all_alpha_ecp"
QTAIM_RESOLVED, QTAIM_AMBIGUOUS = "resolved", "ambiguous"
# Multiwfn adds an ECP atom's EDF core density to alpha and beta evenly, so even an all-alpha read
# has beta ~ alpha at the nucleus of an atom at or beyond Rb (the first def2 ECP element)
_FIRST_ECP_Z = 37


def _num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


_PERIODIC_TABLE = None


def _atomic_number(element) -> int:
    global _PERIODIC_TABLE
    if _PERIODIC_TABLE is None:
        from rdkit import Chem

        _PERIODIC_TABLE = Chem.GetPeriodicTable()
    try:
        return _PERIODIC_TABLE.GetAtomicNumber(re.match(r"[A-Za-z]+", element or "").group(0))
    except Exception:
        return 0


def _beta_zero(cp: dict) -> bool:
    # rounding noise in a stored zero scales with the density (up to ~3e-12 relative seen)
    return abs(cp["density_beta"]) <= 1e-9 * max(1.0, abs(cp["density_all"]))


def qtaim_spin_class(record: dict) -> str:
    """How a qtaim.json record resolves alpha and beta density.

    Among CPs with density_all > 1e-6, leaving out the nuclear CPs of ECP atoms (Z >= 37) whose
    beta is non-zero (the EDF core density, split evenly):
    all_alpha: density_beta == 0 (to rounding) and density_alpha == density_all at every CP --
        what Multiwfn writes when it reads an unrestricted .wfn as all-alpha
    all_alpha_ecp: the same, in a record whose ECP nuclei carry the split core density
    resolved: no such CP has density_beta == 0 (or the record has no spin fields)
    ambiguous: anything in between, e.g. stale all-alpha CPs merged into a resolved record
    """
    cps = {k: v for k, v in record.items() if isinstance(v, dict) and _num(v.get("density_all"))
           and v["density_all"] > 1e-6}
    # the common case, decided without element lookups: no CP is missing beta or has it at zero
    if cps and all(_num(v.get("density_beta")) and not _beta_zero(v) for v in cps.values()):
        return QTAIM_RESOLVED
    split = {k for k, v in cps.items() if "_" not in k and _num(v.get("density_beta")) and not _beta_zero(v)
             and _atomic_number(v.get("element")) >= _FIRST_ECP_Z}
    rest = [v for k, v in cps.items() if k not in split]
    if not rest or not any("density_beta" in v for v in rest):
        return QTAIM_RESOLVED
    if not all(_num(v.get("density_beta")) for v in rest):
        return QTAIM_AMBIGUOUS
    zero = [_beta_zero(v) for v in rest]
    if not any(zero):
        return QTAIM_RESOLVED
    if not all(zero):
        return QTAIM_AMBIGUOUS
    for v in rest:
        alpha = v.get("density_alpha")
        if not _num(alpha) or abs(alpha - v["density_all"]) > 1e-8 * max(1.0, abs(v["density_all"])):
            return QTAIM_AMBIGUOUS
    return QTAIM_ALL_ALPHA_ECP if split else QTAIM_ALL_ALPHA


def all_electron_count(dft_dict) -> Optional[int]:
    """sum(Z) - net charge of the parsed geometry input (core electrons included); None if unknown."""
    try:
        zs = [_atomic_number(a["element"]) for a in dft_dict["mol"].values()]
        if not zs or 0 in zs:
            return None
        return sum(zs) - int(dft_dict.get("charge", 0))
    except Exception:
        return None


_QTAIM_BANNER = re.compile(r"Total/Alpha/Beta electrons:\s*(\S+)\s+(\S+)\s+(\S+)")


def qtaim_out_banner(folder: Optional[str]) -> Optional[tuple]:
    """(alpha, beta) electrons Multiwfn loaded for the QTAIM run; None if unknown.

    Taken from the first qtaim.out copy whose run finished (CP count and CPprop.txt export both
    printed). Multiwfn prints the banner as soon as it loads the wavefunction, so a rerun killed
    mid-search leaves a beta > 0 banner next to the old all-alpha record; that copy is skipped and
    the archived one decides. An unrestricted .wfn read as all-alpha shows beta == 0.
    """
    if not folder:
        return None
    for text in _multiwfn_out_texts(folder, "qtaim.out"):
        if not (QTAIM_COUNT_PATTERN.search(text) and QTAIM_EXPORT_MARKER in text):
            continue
        m = _QTAIM_BANNER.search(text)
        if m:
            try:
                return float(m.group(2)), float(m.group(3))
            except ValueError:
                return None
    return None


def qtaim_fixed_in_place(record: dict) -> bool:
    """fix-allalpha-qtaim's signature: density_alpha == density_beta and spin_density == 0 exactly at
    every CP. Such a record keeps the all-alpha banner in its archived qtaim.out; no unrestricted run
    gives exact zeros, and a restricted one shows beta > 0 in the banner."""
    cps = [v for v in record.values() if isinstance(v, dict) and _num(v.get("density_all"))]
    return bool(cps) and all(
        _num(v.get("density_alpha")) and v.get("density_alpha") == v.get("density_beta")
        and v.get("spin_density") == 0.0 for v in cps)


def qtaim_all_alpha_defect(
    record: dict,
    n_electrons: Optional[int] = None,
    mult: Optional[int] = None,
    banner: Optional[tuple] = None,
) -> bool:
    """True if the record needs a QTAIM rerun from a .wfx because Multiwfn read the wavefunction as all-alpha.

    With the banner of a finished QTAIM run (qtaim_out_banner: alpha, beta electrons):
    - beta > 0, a resolved run: a defect only if the record is all_alpha or all_alpha_ecp, i.e. not from
      that run. A resolved .wfx run has beta at every CP, so its own record always passes (no loop).
      An ambiguous record (stale all-alpha CPs in a resolved one) passes: a density cutoff there could
      reject a correct, strongly spin-polarized record after every rerun; those go to a rerun list.
    - beta == 0, an all-alpha run: a defect unless every electron is alpha (alpha == mult - 1),
      fix-allalpha-qtaim already repaired the record, or the multiplicity is unknown (a rerun of a
      genuinely all-alpha system would reproduce beta == 0 forever).
    Without a banner the densities decide: all_alpha, all_alpha_ecp and ambiguous are defects, unless
    every electron is alpha (n_electrons == mult - 1, e.g. an H atom); unknown counts count as a defect.
    """
    if banner is not None:
        alpha_e, beta_e = banner
        if beta_e > 0:
            return qtaim_spin_class(record) in (QTAIM_ALL_ALPHA, QTAIM_ALL_ALPHA_ECP)
        if mult is None or alpha_e == mult - 1:
            return False
        return not qtaim_fixed_in_place(record)
    spin_class = qtaim_spin_class(record)
    if spin_class in (QTAIM_AMBIGUOUS, QTAIM_ALL_ALPHA_ECP):
        return True
    if spin_class != QTAIM_ALL_ALPHA:
        return False
    return not (n_electrons is not None and mult is not None and n_electrons == mult - 1)


def qtaim_copy_has_all_alpha_defect(folder: str, n_electrons: Optional[int] = None,
                                     mult: Optional[int] = None) -> bool:
    """True if either qtaim.json copy (folder root or generator/) is an all-alpha defect.

    One rule for the validator, the restart gate and the pre-extraction cleanup: with
    clean=False a stale root copy survives next to generator/, and checking only one copy
    let the gate skip QTAIM while the validator failed the folder, every pass.
    """
    banner = qtaim_out_banner(folder)
    for base in (folder, os.path.join(folder, "generator")):
        try:
            with open(os.path.join(base, "qtaim.json"), "r") as f:
                record = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(record, dict) and qtaim_all_alpha_defect(
                record, n_electrons=n_electrons, mult=mult, banner=banner):
            return True
    return False


def validate_qtaim_dict(
    qtaim_json_loc: str,
    n_atoms: int = None,
    verbose: bool = False,
    logger: any = None,
    folder: str = None,
    check_bcp_count: bool = False,
    bcp_tolerance: int = DEFAULT_BCP_TOLERANCE,
    require_provenance: bool = False,
):
    """
    Basic check that the qtaim json file has the expected structure
    Check that it has the keys 'atoms', 'bonds', 'charges', and 'fuzzy'.
    If n_atoms is provided, check that the number of non-bonded critical points matches n_atoms.
    If harsh_check is True, also check that the number of nuclear critical points matches n_atoms.
    """
    qtaim_dict = _safe_json_load(qtaim_json_loc, logger=logger)
    if qtaim_dict is None:
        return False
    # check it isn't empty
    if not qtaim_dict:
        if verbose:
            print("QTAIM json file is empty.")
        return False

    # dict_ncps = {qtaim_dict[key] for key in qtaim_dict if "_" not in key}
    dict_ncps = [qtaim_dict[key] for key in list(qtaim_dict.keys()) if "_" not in key]
    # dict_bcps = {qtaim_dict[key] for key in qtaim_dict if "_" in key}
    dict_bcps = [qtaim_dict[key] for key in list(qtaim_dict.keys()) if "_" in key]

    if n_atoms is not None:
        if len(dict_ncps) != n_atoms:
            if verbose:
                print(
                    f"Number of nuclear critical points ({len(dict_ncps)}) does not match expected ({n_atoms})."
                )
            if logger:
                logger.error(
                    f"Number of nuclear critical points ({len(dict_ncps)}) does not match expected ({n_atoms})."
                )
            return False
    status = None
    if folder is not None and (check_bcp_count or require_provenance):
        status = qtaim_run_status(folder)

    # A bound multi-atom system must have at least one bond critical point.
    # Logged unconditionally (cheap, and this class of failure is otherwise
    # invisible), and only fatal under check_bcp_count, since a genuinely
    # non-interacting pair of atoms legitimately has none. Even then, an empty
    # BCP set backed by a *complete* run defers to the reported/storable
    # shortfall logic below: a run that itself found zero (or only unstorable)
    # bond CPs is deterministic, and failing it would requeue a job that no
    # rerun can change.
    if not dict_bcps and n_atoms is not None and n_atoms > 1:
        msg = (
            f"QTAIM json has no bond critical points for {n_atoms} atoms: "
            f"{qtaim_json_loc}"
        )
        if verbose:
            print(msg)
        if logger:
            logger.error(msg)
        if check_bcp_count:
            run_complete = (
                status is not None
                and status["have_qtaim_out"]
                and status["search_done"]
                and status["export_done"]
            )
            if not run_complete:
                return False

    if require_provenance and folder is not None:
        # Absent qtaim.out means the record's completeness cannot be established
        # from anything on disk. check_bcp_count cannot catch this on its own:
        # with no reported count there is no shortfall to measure, so the record
        # passes and the runner skips the folder -- while the audit classifies it
        # no_provenance and selects it for rerun. This is what makes the two
        # agree, at the cost of rerunning records that may well be fine.
        if not status["have_qtaim_out"]:
            msg = (
                f"No qtaim.out for {folder}, so the bond-CP count cannot be "
                f"verified; treating as incomplete because --require_qtaim_"
                f"provenance is set ({qtaim_json_loc})"
            )
            if verbose:
                print(msg)
            if logger:
                logger.error(msg)
            return False

    if check_bcp_count and folder is not None:
        # An incomplete run is a defect even when the counts happen to agree:
        # if the search or the export never finished, the record cannot be
        # complete regardless of what it contains.
        if status["have_qtaim_out"] and not status["search_done"]:
            msg = f"QTAIM search never completed (no CP count in qtaim.out): {folder}"
            if verbose:
                print(msg)
            if logger:
                logger.error(msg)
            return False
        if status["have_qtaim_out"] and not status["export_done"]:
            msg = (
                f"QTAIM CPprop.txt export never completed, so the parsed record "
                f"is partial: {folder}"
            )
            if verbose:
                print(msg)
            if logger:
                logger.error(msg)
            return False

        reported = status["reported_bcp"]
        raw_deficit = reported - len(dict_bcps) if reported is not None else 0
        # 0 < raw_deficit <= tolerance is deliberately silent. Those records are
        # almost all schema-complete, and saying so would mean paying the
        # CPprop.txt parse on the majority of folders just to emit a line.
        if raw_deficit > bcp_tolerance:
            # Only now is the exact storable count worth computing. Multiwfn's
            # reported count is an upper bound on what the atom-pair-keyed
            # schema can hold, so storable <= reported and the raw deficit is
            # an upper bound on the real one -- if that already fits inside the
            # tolerance, the real one does too. Checking it first is what keeps
            # the common clean case from extracting CPprop.txt out of
            # out_files.zip and reparsing every CP block.
            storable = storable_bcp_count(folder)
            expected = storable if storable is not None else reported
            basis = "storable" if storable is not None else "reported (upper bound)"
            deficit = expected - len(dict_bcps)

            if deficit <= 0:
                note = (
                    f"QTAIM json holds {len(dict_bcps)} of {reported} reported "
                    f"bond critical points; the rest have no storable atom pair, "
                    f"so the record is complete for this schema ({qtaim_json_loc})"
                )
                if verbose:
                    print(note)
                if logger:
                    logger.info(note)
            elif deficit > bcp_tolerance:
                msg = (
                    f"QTAIM json holds {len(dict_bcps)} bond critical points but "
                    f"{expected} are {basis} (of {reported} reported) -- "
                    f"{deficit} lost, above the tolerance of {bcp_tolerance} "
                    f"({qtaim_json_loc})"
                )
                if verbose:
                    print(msg)
                if logger:
                    logger.error(msg)
                return False
            else:
                # Within tolerance: almost certainly CPs with no traceable bond
                # path. Failing here would queue a job that no rerun can fix.
                msg = (
                    f"QTAIM json is {deficit} bond critical point(s) short of "
                    f"{expected} {basis}, within the tolerance of {bcp_tolerance} "
                    f"({qtaim_json_loc})"
                )
                if verbose:
                    print(msg)
                if logger:
                    logger.warning(msg)

    if verbose:
        print(f"Number of nuclear critical points: {len(dict_ncps)}")
        print(f"Number of bond critical points: {len(dict_bcps)}")
        print("QTAIM json structure is valid.")
    return True


def validate_orca_dict(
    orca_json_loc: str,
    n_atoms: int = None,
    verbose: bool = False,
    logger=None,
    min_parser_version: Optional[int] = None,
) -> bool:
    """Validate orca.json structure and data integrity.

    Returns True if valid, False if malformed, or older than min_parser_version
    (orca_parser_version, 1 when absent) when that is set.
    Returns True if file is absent (backward compat -- absence is valid).
    """
    if not os.path.exists(orca_json_loc):
        return True  # Absent orca.json is valid (backward compat)

    if os.path.getsize(orca_json_loc) == 0:
        if logger:
            logger.error("Empty orca.json at %s", orca_json_loc)
        return False

    try:
        with open(orca_json_loc, "r") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        if logger:
            logger.error("Invalid orca.json at %s: %s", orca_json_loc, e)
        return False

    if not isinstance(data, dict):
        if logger:
            logger.error("orca.json is not a dict at %s", orca_json_loc)
        return False

    if len(data) == 0:
        if logger:
            logger.error("orca.json is empty dict at %s", orca_json_loc)
        return False

    if min_parser_version:
        version = data.get("orca_parser_version", 1)
        if version < min_parser_version:
            if logger:
                logger.error(
                    "orca.json parser version %s < %s at %s (stale, needs reparse)",
                    version,
                    min_parser_version,
                    orca_json_loc,
                )
            if verbose:
                print(f"orca.json parser version {version} < {min_parser_version}: stale")
            return False

    # Type checks for known keys
    type_checks = {
        "final_energy_eh": (int, float),
        "scf_converged": (bool,),
        "scf_cycles": (int,),
        "total_run_time_s": (int, float),
        "gradient_norm": (int, float),
        "gradient_rms": (int, float),
        "gradient_max": (int, float),
    }
    for key, types in type_checks.items():
        if key in data and data[key] is not None:
            if not isinstance(data[key], types):
                if logger:
                    logger.error(
                        "orca.json key '%s' has wrong type at %s", key, orca_json_loc
                    )
                return False

    # Array length checks
    array_checks = {
        "dipole_au": 3,
        "quadrupole_au": 6,
        "rotational_constants_cm1": 3,
    }
    for key, expected_len in array_checks.items():
        if key in data:
            if not isinstance(data[key], list) or len(data[key]) != expected_len:
                if logger:
                    logger.error(
                        "orca.json key '%s' must be %d-element list at %s",
                        key,
                        expected_len,
                        orca_json_loc,
                    )
                return False

    # Atom count checks for charge dicts (warning only)
    if n_atoms is not None:
        charge_keys = [
            "mulliken_charges",
            "loewdin_charges",
            "hirshfeld_charges",
            "mbis_charges",
        ]
        for key in charge_keys:
            if key in data and isinstance(data[key], dict):
                if len(data[key]) != n_atoms:
                    if verbose:
                        print(
                            f"orca.json key '{key}' has {len(data[key])} atoms, "
                            f"expected {n_atoms}"
                        )
                    if logger:
                        logger.warning(
                            "orca.json key '%s' has %d atoms, expected %d at %s",
                            key,
                            len(data[key]),
                            n_atoms,
                            orca_json_loc,
                        )

    if verbose:
        print("orca.json structure is valid.")
    return True


def validation_checks(
    folder: str,
    verbose: bool = False,
    full_set: int = 0,
    move_results: bool = True,
    logger=None,
    check_orca: bool = False,
    check_bcp_count: bool = False,
    bcp_tolerance: int = DEFAULT_BCP_TOLERANCE,
    require_qtaim_provenance: bool = False,
    recheck_fuzzy: bool = False,
    orca_min_parser_version: Optional[int] = ORCA_PARSER_VERSION,
    recheck_allalpha_qtaim: bool = False,
):
    """
    Run all validation checks on the json files in the given folder.
    Arguments:
        folder (str): Path to the folder containing the json files.
        verbose (bool): If True, print detailed validation messages.
        full_set (int): Level of calculation detail (0-baseline, 1-baseline, 2-full).
        move_results (bool): Adjust if files have been moved during cleaning.
        bcp_tolerance (int): how many bond CPs may be missing before the record
            is treated as defective. Guards against queueing jobs no rerun can
            fix, since a CP with no traceable bond path has no storable atom
            pair. Default DEFAULT_BCP_TOLERANCE.
        require_qtaim_provenance (bool): fail records with no qtaim.out. Their
            completeness is unverifiable rather than verified, and without this
            the runner skips them while the audit selects them for rerun.
        check_bcp_count (bool): cross-check qtaim.json's bond-CP count against
            the count Multiwfn reported in qtaim.out, and reject records whose
            critical points were lost between the search and the stored file.
            Off by default: it needs qtaim.out, which older runs may not retain.
        recheck_fuzzy (bool): also fail when fuzzy integrations or fuzzy bond
            orders of an unrestricted wavefunction (open shell or UKS singlet)
            are present but physically wrong (all-zero densities, spin not
            summing to multiplicity - 1, all-alpha or alpha-only fuzzy bonds).
            Dry run: nothing is written.
        recheck_allalpha_qtaim (bool): fail an all-alpha or partly all-alpha qtaim.json
            (an unrestricted .wfn read as all-alpha), unless every electron is alpha.
        orca_min_parser_version (Optional[int]): with check_orca, also fail when
            orca.json predates this parser version (orca_parser_version, 1 when
            absent), so the runner reparses it. None or 0 disables the gate.
    Returns:
        bool: True if all validation checks pass, False otherwise.
    """
    # check that all the json files are present
    required_files = [
        "timings.json",
        "fuzzy_full.json",
        "other.json",
        "charge.json",
        "qtaim.json",
        "bond.json",
    ]
    tf = True

    if move_results:
        folder_check_res = os.path.join(folder, "generator")
    else:
        folder_check_res = folder

    for file in required_files:
        if not os.path.exists(os.path.join(folder_check_res, file)):
            if logger:
                logger.error(
                    f"Missing required file: {file} in folder: {folder_check_res}"
                )
            if verbose:
                print(f"Missing required file: {file} in folder: {folder_check_res}")
            tf = False

    if not tf:
        return False

    dft_dict = get_charge_spin_n_atoms_from_folder(
        folder, logger=logger, verbose=verbose
    )
    if not dft_dict:
        return False
    # print("log dict: ", str(dft_dict))
    n_atoms = len(dft_dict["mol"])
    spin = dft_dict.get("spin", None)
    charge = dft_dict.get("charge", None)

    if verbose:
        print(f"n_atoms: {n_atoms}, spin: {spin}, charge: {charge}")
    if logger:
        logger.info(f"n_atoms: {n_atoms}, spin: {spin}, charge: {charge}")

    if spin != 1:
        spin_tf = True
    else:
        spin_tf = False

    timing_json_loc = os.path.join(folder_check_res, "timings.json")
    fuzzy_json_loc = os.path.join(folder_check_res, "fuzzy_full.json")
    other_dict_loc = os.path.join(folder_check_res, "other.json")
    charge_json_loc = os.path.join(folder_check_res, "charge.json")
    qtaim_json_loc = os.path.join(folder_check_res, "qtaim.json")
    bond_json_loc = os.path.join(folder_check_res, "bond.json")
    # bonding_json_loc = os.path.join(folder, "bonding.json")
    tf_cond = True

    if not validate_timing_dict(
        timing_json_loc, verbose=verbose, full_set=full_set, spin_tf=spin_tf
    ):
        if logger:
            logger.error(f"Timing json validation failed in folder: {folder}")
        tf_cond = False

    if not validate_fuzzy_dict(
        fuzzy_json_loc,
        n_atoms=n_atoms,
        spin_tf=spin_tf,
        verbose=verbose,
        full_set=full_set,
        logger=logger,
    ):
        if logger:
            logger.error(f"Fuzzy json validation failed in folder: {folder}")
        tf_cond = False

    if not validate_other_dict(other_dict_loc, verbose=verbose, logger=logger, full_set=full_set):
        if logger:
            logger.error(f"Other json validation failed in folder: {folder}")
        tf_cond = False

    if not validate_charge_dict(
        charge_json_loc, n_atoms=n_atoms, verbose=verbose, logger=logger
    ):
        if logger:
            logger.error(f"Charge json validation failed in folder: {folder}")
        tf_cond = False

    if not validate_qtaim_dict(
        qtaim_json_loc,
        n_atoms=n_atoms,
        verbose=verbose,
        logger=logger,
        folder=folder,
        check_bcp_count=check_bcp_count,
        bcp_tolerance=bcp_tolerance,
        require_provenance=require_qtaim_provenance,
    ):
        if logger:
            logger.error(f"QTAIM json validation failed in folder: {folder}")
        tf_cond = False
    elif recheck_allalpha_qtaim and qtaim_copy_has_all_alpha_defect(
        folder, n_electrons=all_electron_count(dft_dict), mult=spin
    ):
        # both copies (root and generator/) and the qtaim.out banner, the restart gate's rule
        msg = (f"QTAIM json is all-alpha or partly all-alpha (an unrestricted .wfn read as "
               f"all-alpha); rerun QTAIM from a .wfx: {folder}")
        if verbose:
            print(msg)
        if logger:
            logger.error(msg)
        tf_cond = False

    if not validate_bond_dict(
        bond_json_loc, verbose=verbose, full_set=full_set, logger=logger
    ):
        if logger:
            logger.error(f"Bond json validation failed in folder: {folder}")
        tf_cond = False

    # ORCA validation -- optional unless check_orca=True
    orca_json_loc = os.path.join(folder_check_res, "orca.json")
    if check_orca:
        # When check_orca is set, orca.json is REQUIRED
        if not os.path.exists(orca_json_loc):
            if logger:
                logger.error(
                    f"Missing required orca.json in folder: {folder_check_res}"
                )
            if verbose:
                print(f"Missing required orca.json in folder: {folder_check_res}")
            tf_cond = False
        elif not validate_orca_dict(
            orca_json_loc,
            n_atoms=n_atoms,
            verbose=verbose,
            logger=logger,
            min_parser_version=orca_min_parser_version,
        ):
            if logger:
                logger.error(f"ORCA json validation failed in folder: {folder}")
            tf_cond = False
    elif os.path.exists(orca_json_loc):
        # When check_orca is NOT set, still validate if present (but don't require it)
        if not validate_orca_dict(
            orca_json_loc, n_atoms=n_atoms, verbose=verbose, logger=logger
        ):
            if logger:
                logger.error(f"ORCA json validation failed in folder: {folder}")
            tf_cond = False

    if recheck_fuzzy and tf_cond and spin is not None:
        from qtaim_gen.source.utils.fuzzy_recheck import recheck_fuzzy as _recheck

        report = _recheck(folder, int(spin), logger=logger, dry_run=True)
        if report["reparse"] or report["rerun"]:
            if logger:
                logger.error(
                    f"recheck_fuzzy: reparse {report['reparse']}, rerun {report['rerun']} "
                    f"in folder: {folder}"
                )
            tf_cond = False

    if verbose:
        print("All validation checks passed.")

    return tf_cond


def get_information_from_job_folder(folder: str, full_set: int) -> dict:
    """Extracts relevant information from the job folder name."""

    # check if folder has /generator/ subdirectory, if so get timings.json in that folder

    info = {
        "validation_level_0": None,
        "validation_level_1": None,
        "validation_level_2": None,
        "total_time": None,
        "t_qtaim": None,
        "t_other": None,
        "last_edit_time": None,
        "val_time": None,
        "val_qtaim": None,
        "val_charge": None,
        "val_bond": None,
        "val_fuzzy": None,
        "val_other": None,
        "has_orca_json": False,
        "val_orca": None,
        "n_atoms": None,
        "spin": None,
        "charge": None,
    }

    # get .inp file in the folder for spin, charge, n_atoms
    dft_dict = get_charge_spin_n_atoms_from_folder(folder, logger=None, verbose=False)
    # print("DFT dict: ", dft_dict)
    if not dft_dict:
        return info  # return empty info if dft_dict is None or empty

    n_atoms = len(dft_dict["mol"])
    spin = dft_dict.get("spin", None)
    charge = dft_dict.get("charge", None)

    info["n_atoms"] = n_atoms
    info["spin"] = spin
    info["charge"] = charge

    if spin != 1:
        spin_tf = True
    else:
        spin_tf = False

    # check is there is a generator subfolder
    # print("Folder to analyze: ", folder)

    if "generator" in os.listdir(folder):
        # print("Found generator subfolder.")
        gen_folder = folder + "/generator/"
        timings_file = os.path.join(gen_folder, "timings.json")

        if os.path.exists(timings_file) and os.path.getsize(timings_file) > 0:
            with open(timings_file, "r") as f:
                timings = json.load(f)
            total_time = float(np.array(list(timings.values())).sum())
            info["total_time"] = total_time

            for col in timings.keys():
                info[f"t_{col}"] = timings[col]
        else:
            return info  # return empty info if timings file is missing or empty

        tf_validation_level_0 = validation_checks(
            folder, full_set=0, verbose=False, move_results=True, logger=None
        )

        tf_validation_level_1 = validation_checks(
            folder, full_set=1, verbose=False, move_results=True, logger=None
        )

        tf_validation_level_2 = validation_checks(
            folder, full_set=2, verbose=False, move_results=True, logger=None
        )

        # set val_qtaim, val_charge, val_bond, val_fuzzy, val_other to True for corresponding level
        if full_set == 0:
            status_val = tf_validation_level_0
        elif full_set == 1:
            status_val = tf_validation_level_1
        elif full_set == 2:
            status_val = tf_validation_level_2

        if status_val:
            info["val_qtaim"] = True
            info["val_charge"] = True
            info["val_bond"] = True
            info["val_fuzzy"] = True
            info["val_other"] = True
            info["val_time"] = True
        else:
            dict_val = get_val_breakdown_from_folder(
                gen_folder, n_atoms=n_atoms, full_set=full_set, spin_tf=spin_tf
            )
            info.update(dict_val)

        # check edit date of timings.json
        mtime_timestamp = os.path.getmtime(timings_file)
        # Convert the timestamp to a datetime object
        mtime_datetime = datetime.fromtimestamp(mtime_timestamp)
        # Format the datetime object into a human-readable string
        # Example format: YYYY-MM-DD HH:MM:SS
        human_readable_mtime = mtime_datetime.strftime("%Y-%m-%d %H:%M:%S")
        info["last_edit_time"] = human_readable_mtime

        info.update(
            {
                "validation_level_0": tf_validation_level_0,
                "validation_level_1": tf_validation_level_1,
                "validation_level_2": tf_validation_level_2,
            }
        )

    else:
        timings_file = os.path.join(folder, "timings.json")

        if os.path.exists(timings_file):
            with open(timings_file, "r") as f:
                timings = json.load(f)
            total_time = float(np.array(list(timings.values())).sum())
            info["total_time"] = total_time

            for col in timings.keys():
                info[f"t_{col}"] = timings[col]

            edit_time = os.path.getmtime(timings_file)
            # Convert the timestamp to a datetime object
            mtime_datetime = datetime.fromtimestamp(edit_time)
            # Format the datetime object into a human-readable string
            human_readable_mtime = mtime_datetime.strftime("%Y-%m-%d %H:%M:%S")
            info["last_edit_time"] = human_readable_mtime

        dict_val = get_val_breakdown_from_folder(
            folder, n_atoms=n_atoms, full_set=full_set, spin_tf=spin_tf
        )
        info.update(dict_val)

    return info
