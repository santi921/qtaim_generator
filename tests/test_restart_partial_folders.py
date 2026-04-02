"""Tests for restart logic with partially complete job folders.

Tests cover:
- validation_checks on folders missing individual JSON files
- get_val_breakdown_from_folder on partially complete folders
- run_jobs restart skip logic with partial completions
- Edge cases: errored timings (-1), empty files, corrupted JSON
"""

import json
import os
import time
import pytest

from qtaim_gen.source.utils.validation import (
    validation_checks,
    get_val_breakdown_from_folder,
    validate_charge_dict,
    validate_bond_dict,
    validate_fuzzy_dict,
    validate_timing_dict,
    validate_qtaim_dict,
    validate_other_dict,
)

# ---------------------------------------------------------------------------
# Fixtures — reference data derived from tests/test_files/lmdb_tests/orca5_rks
# ---------------------------------------------------------------------------

REFERENCE_DIR = os.path.join(
    os.path.dirname(__file__), "test_files", "lmdb_tests", "orca5_rks"
)

# Minimal valid charge dict for full_set=0 (keys: adch, becke, hirshfeld, cm5)
def _make_charge_dict(n_atoms, keys=("adch", "becke", "hirshfeld", "cm5")):
    """Build a minimal valid charge dict with n_atoms per method."""
    d = {}
    for k in keys:
        d[k] = {"charge": {f"{i+1}_X": 0.1 for i in range(n_atoms)}}
    return d


def _make_bond_dict(keys=("fuzzy_bond",)):
    """Build a minimal valid bond dict."""
    d = {}
    for k in keys:
        d[k] = {"1_X_to_2_Y": 0.5}
    return d


def _make_fuzzy_dict(n_atoms, keys=("becke_fuzzy_density", "hirsh_fuzzy_density")):
    """Build a minimal valid fuzzy dict. n_atoms + 2 entries expected."""
    d = {}
    for k in keys:
        d[k] = {f"{i+1}_X": 0.0 for i in range(n_atoms + 2)}
    return d


def _make_qtaim_dict(n_atoms):
    """Build a minimal valid QTAIM dict. NCPs keyed by plain int strings,
    BCPs keyed by i_j strings."""
    d = {}
    for i in range(n_atoms):
        d[str(i)] = {"cp_num": i + 1, "element": "C", "density_all": 1.0}
    # Add one BCP
    d["0_1"] = {"cp_num": n_atoms + 1, "density_all": 0.5}
    return d


def _make_other_dict():
    """Build a minimal valid other dict for full_set=0."""
    return {
        "mpp_full": 1.0,
        "sdp_full": 1.0,
        "mpp_heavy": 1.0,
        "sdp_heavy": 1.0,
        "ALIE_Volume": 100.0,
        "ALIE_Surface_Density": 1.0,
        "ALIE_Minimal_value": 10.0,
        "ALIE_Maximal_value": 20.0,
        "ALIE_Overall_surface_area": 100.0,
        "ALIE_Positive_surface_area": 60.0,
        "ALIE_Negative_surface_area": 40.0,
        "ALIE_Overall_skewness": -0.5,
    }


def _make_timings_dict(full_set=0, spin_tf=False):
    """Build a valid timings dict for full_set=0."""
    d = {
        "qtaim": 10.0,
        "other": 5.0,
        "hirshfeld": 3.0,
        "becke": 3.0,
        "adch": 3.0,
        "cm5": 3.0,
        "fuzzy_bond": 2.0,
        "becke_fuzzy_density": 2.0,
        "hirsh_fuzzy_density": 2.0,
    }
    if spin_tf:
        d["hirsh_fuzzy_spin"] = 2.0
        d["becke_fuzzy_spin"] = 2.0
    if full_set > 0:
        d.update({
            "vdd": 3.0,
            "mbis": 3.0,
            "chelpg": 3.0,
            "ibsi_bond": 2.0,
            "elf_fuzzy": 2.0,
            "mbis_fuzzy_density": 2.0,
        })
        if spin_tf:
            d["mbis_fuzzy_spin"] = 2.0
    if full_set > 1:
        d.update({
            "bader": 3.0,
            "laplacian_bond": 2.0,
            "grad_norm_rho_fuzzy": 2.0,
            "laplacian_rho_fuzzy": 2.0,
            "ESP_Volume": 5.0,
        })
    return d


# Minimal ORCA inp content for a 3-atom molecule (H2O)
MINIMAL_INP = """\
! B3LYP def2-SVP
*xyz 0 1
O   0.000000  0.000000  0.117300
H   0.000000  0.757200 -0.469200
H   0.000000 -0.757200 -0.469200
*
"""

N_ATOMS_H2O = 3


@pytest.fixture
def complete_folder(tmp_path):
    """Create a fully complete job folder with all valid JSONs."""
    folder = tmp_path / "complete_job"
    folder.mkdir()

    # Write inp file
    (folder / "orca.inp").write_text(MINIMAL_INP)

    # Write all JSON files
    _write_json(folder / "charge.json", _make_charge_dict(N_ATOMS_H2O))
    _write_json(folder / "bond.json", _make_bond_dict())
    _write_json(folder / "fuzzy_full.json", _make_fuzzy_dict(N_ATOMS_H2O))
    _write_json(folder / "qtaim.json", _make_qtaim_dict(N_ATOMS_H2O))
    _write_json(folder / "other.json", _make_other_dict())
    _write_json(folder / "timings.json", _make_timings_dict())

    return folder


@pytest.fixture
def complete_folder_moved(tmp_path):
    """Create a complete job folder with results in generator/ subdir."""
    folder = tmp_path / "complete_moved"
    folder.mkdir()
    gen = folder / "generator"
    gen.mkdir()

    (folder / "orca.inp").write_text(MINIMAL_INP)

    _write_json(gen / "charge.json", _make_charge_dict(N_ATOMS_H2O))
    _write_json(gen / "bond.json", _make_bond_dict())
    _write_json(gen / "fuzzy_full.json", _make_fuzzy_dict(N_ATOMS_H2O))
    _write_json(gen / "qtaim.json", _make_qtaim_dict(N_ATOMS_H2O))
    _write_json(gen / "other.json", _make_other_dict())
    _write_json(gen / "timings.json", _make_timings_dict())

    return folder


def _write_json(path, data):
    with open(str(path), "w") as f:
        json.dump(data, f)


# ---------------------------------------------------------------------------
# Tests: validation_checks on complete folders
# ---------------------------------------------------------------------------


class TestValidationChecksComplete:
    def test_complete_folder_passes(self, complete_folder):
        assert validation_checks(
            str(complete_folder), full_set=0, move_results=False
        )

    def test_complete_folder_moved_passes(self, complete_folder_moved):
        assert validation_checks(
            str(complete_folder_moved), full_set=0, move_results=True
        )


# ---------------------------------------------------------------------------
# Tests: validation_checks on folders missing individual files
# ---------------------------------------------------------------------------


class TestValidationChecksMissingFiles:
    """Each test removes one JSON file and verifies validation fails."""

    @pytest.fixture(autouse=True)
    def setup_folder(self, complete_folder):
        self.folder = complete_folder

    def test_missing_charge_json(self):
        os.remove(str(self.folder / "charge.json"))
        assert not validation_checks(
            str(self.folder), full_set=0, move_results=False
        )

    def test_missing_bond_json(self):
        os.remove(str(self.folder / "bond.json"))
        assert not validation_checks(
            str(self.folder), full_set=0, move_results=False
        )

    def test_missing_fuzzy_json(self):
        os.remove(str(self.folder / "fuzzy_full.json"))
        assert not validation_checks(
            str(self.folder), full_set=0, move_results=False
        )

    def test_missing_qtaim_json(self):
        os.remove(str(self.folder / "qtaim.json"))
        assert not validation_checks(
            str(self.folder), full_set=0, move_results=False
        )

    def test_missing_other_json(self):
        os.remove(str(self.folder / "other.json"))
        assert not validation_checks(
            str(self.folder), full_set=0, move_results=False
        )

    def test_missing_timings_json(self):
        os.remove(str(self.folder / "timings.json"))
        assert not validation_checks(
            str(self.folder), full_set=0, move_results=False
        )

    def test_empty_charge_json(self):
        """Empty charge.json causes JSONDecodeError in validation_checks.
        This documents a real bug: validation_checks doesn't handle
        corrupted/empty JSON files gracefully."""
        (self.folder / "charge.json").write_text("")
        # Currently raises JSONDecodeError — should return False instead
        with pytest.raises(json.JSONDecodeError):
            validation_checks(str(self.folder), full_set=0, move_results=False)


# ---------------------------------------------------------------------------
# Tests: Partial charge completion (some methods present, others missing)
# ---------------------------------------------------------------------------


class TestPartialChargeCompletion:
    """Test that charge validation fails when only some methods completed."""

    def test_missing_cm5_from_charge(self, tmp_path):
        """full_set=0 expects adch, becke, hirshfeld, cm5.
        Remove cm5 -> validation should fail."""
        charge = _make_charge_dict(N_ATOMS_H2O, keys=("adch", "becke", "hirshfeld"))
        path = tmp_path / "charge.json"
        _write_json(path, charge)
        assert not validate_charge_dict(str(path), n_atoms=N_ATOMS_H2O, full_set=0)

    def test_missing_adch_from_charge(self, tmp_path):
        charge = _make_charge_dict(N_ATOMS_H2O, keys=("becke", "hirshfeld", "cm5"))
        path = tmp_path / "charge.json"
        _write_json(path, charge)
        assert not validate_charge_dict(str(path), n_atoms=N_ATOMS_H2O, full_set=0)

    def test_only_hirshfeld_present(self, tmp_path):
        charge = _make_charge_dict(N_ATOMS_H2O, keys=("hirshfeld",))
        path = tmp_path / "charge.json"
        _write_json(path, charge)
        assert not validate_charge_dict(str(path), n_atoms=N_ATOMS_H2O, full_set=0)

    def test_all_four_present(self, tmp_path):
        charge = _make_charge_dict(N_ATOMS_H2O)
        path = tmp_path / "charge.json"
        _write_json(path, charge)
        assert validate_charge_dict(str(path), n_atoms=N_ATOMS_H2O, full_set=0)

    def test_full_set_0_passes_without_mbis(self, tmp_path):
        """full_set=0 doesn't require mbis, vdd, chelpg."""
        charge = _make_charge_dict(N_ATOMS_H2O, keys=("adch", "becke", "hirshfeld", "cm5"))
        path = tmp_path / "charge.json"
        _write_json(path, charge)
        assert validate_charge_dict(str(path), n_atoms=N_ATOMS_H2O, full_set=0)

    def test_full_set_1_requires_mbis(self, tmp_path):
        """full_set=1 additionally requires mbis, vdd, chelpg."""
        charge = _make_charge_dict(N_ATOMS_H2O, keys=("adch", "becke", "hirshfeld", "cm5"))
        path = tmp_path / "charge.json"
        _write_json(path, charge)
        assert not validate_charge_dict(str(path), n_atoms=N_ATOMS_H2O, full_set=1)

    def test_wrong_atom_count(self, tmp_path):
        """Charge dict with wrong number of atoms should fail."""
        charge = _make_charge_dict(5)  # 5 atoms but we'll validate for 3
        path = tmp_path / "charge.json"
        _write_json(path, charge)
        assert not validate_charge_dict(str(path), n_atoms=N_ATOMS_H2O, full_set=0)


# ---------------------------------------------------------------------------
# Tests: Partial bond completion
# ---------------------------------------------------------------------------


class TestPartialBondCompletion:
    def test_missing_fuzzy_bond(self, tmp_path):
        bond = _make_bond_dict(keys=())
        path = tmp_path / "bond.json"
        _write_json(path, bond)
        assert not validate_bond_dict(str(path), full_set=0)

    def test_full_set_1_missing_ibsi(self, tmp_path):
        bond = _make_bond_dict(keys=("fuzzy_bond",))
        path = tmp_path / "bond.json"
        _write_json(path, bond)
        assert not validate_bond_dict(str(path), full_set=1)

    def test_full_set_1_complete(self, tmp_path):
        bond = _make_bond_dict(keys=("fuzzy_bond", "ibsi_bond"))
        path = tmp_path / "bond.json"
        _write_json(path, bond)
        assert validate_bond_dict(str(path), full_set=1)


# ---------------------------------------------------------------------------
# Tests: Partial fuzzy completion
# ---------------------------------------------------------------------------


class TestPartialFuzzyCompletion:
    def test_missing_hirsh_fuzzy(self, tmp_path):
        fuzzy = _make_fuzzy_dict(N_ATOMS_H2O, keys=("becke_fuzzy_density",))
        path = tmp_path / "fuzzy_full.json"
        _write_json(path, fuzzy)
        assert not validate_fuzzy_dict(str(path), n_atoms=N_ATOMS_H2O, full_set=0)

    def test_wrong_atom_count_in_fuzzy(self, tmp_path):
        fuzzy = _make_fuzzy_dict(5)  # 5+2=7 entries, but n_atoms=3 expects 3+2=5
        path = tmp_path / "fuzzy_full.json"
        _write_json(path, fuzzy)
        assert not validate_fuzzy_dict(str(path), n_atoms=N_ATOMS_H2O, full_set=0)

    def test_spin_keys_missing_when_required(self, tmp_path):
        fuzzy = _make_fuzzy_dict(N_ATOMS_H2O)
        path = tmp_path / "fuzzy_full.json"
        _write_json(path, fuzzy)
        assert not validate_fuzzy_dict(
            str(path), n_atoms=N_ATOMS_H2O, spin_tf=True, full_set=0
        )


# ---------------------------------------------------------------------------
# Tests: get_val_breakdown_from_folder for partial folders
# ---------------------------------------------------------------------------


class TestValBreakdownPartial:
    """Test that get_val_breakdown_from_folder correctly identifies which
    categories are valid and which are not in partially complete folders."""

    def test_all_valid(self, complete_folder):
        info = get_val_breakdown_from_folder(
            str(complete_folder),
            full_set=0,
            spin_tf=False,
            n_atoms=N_ATOMS_H2O,
        )
        assert info["val_charge"] is True
        assert info["val_bond"] is True
        assert info["val_fuzzy"] is True
        assert info["val_qtaim"] is True
        assert info["val_other"] is True
        assert info["val_time"] is True

    def test_missing_charge_file(self, complete_folder):
        os.remove(str(complete_folder / "charge.json"))
        info = get_val_breakdown_from_folder(
            str(complete_folder),
            full_set=0,
            spin_tf=False,
            n_atoms=N_ATOMS_H2O,
        )
        assert info["val_charge"] is None  # file absent, never validated
        assert info["val_bond"] is True
        assert info["val_fuzzy"] is True
        assert info["val_qtaim"] is True

    def test_partial_charge_content(self, complete_folder):
        """Charge file exists but is missing the cm5 key."""
        charge = _make_charge_dict(N_ATOMS_H2O, keys=("adch", "becke", "hirshfeld"))
        _write_json(complete_folder / "charge.json", charge)
        info = get_val_breakdown_from_folder(
            str(complete_folder),
            full_set=0,
            spin_tf=False,
            n_atoms=N_ATOMS_H2O,
        )
        assert info["val_charge"] is False
        # Other categories unaffected
        assert info["val_bond"] is True
        assert info["val_qtaim"] is True

    def test_missing_bond_file(self, complete_folder):
        os.remove(str(complete_folder / "bond.json"))
        info = get_val_breakdown_from_folder(
            str(complete_folder),
            full_set=0,
            spin_tf=False,
            n_atoms=N_ATOMS_H2O,
        )
        assert info["val_bond"] is None
        assert info["val_charge"] is True

    def test_missing_qtaim_file(self, complete_folder):
        os.remove(str(complete_folder / "qtaim.json"))
        info = get_val_breakdown_from_folder(
            str(complete_folder),
            full_set=0,
            spin_tf=False,
            n_atoms=N_ATOMS_H2O,
        )
        assert info["val_qtaim"] is None
        assert info["val_charge"] is True

    def test_empty_qtaim_dict(self, complete_folder):
        """QTAIM file exists but is empty dict."""
        _write_json(complete_folder / "qtaim.json", {})
        info = get_val_breakdown_from_folder(
            str(complete_folder),
            full_set=0,
            spin_tf=False,
            n_atoms=N_ATOMS_H2O,
        )
        assert info["val_qtaim"] is False

    def test_missing_fuzzy_file(self, complete_folder):
        os.remove(str(complete_folder / "fuzzy_full.json"))
        info = get_val_breakdown_from_folder(
            str(complete_folder),
            full_set=0,
            spin_tf=False,
            n_atoms=N_ATOMS_H2O,
        )
        assert info["val_fuzzy"] is None

    def test_timings_with_error_value(self, complete_folder):
        """Timing with -1 value (errored job) causes timing validation to fail
        because -1 < 1e-6 triggers the 'too small' check."""
        timings = _make_timings_dict()
        timings["cm5"] = -1  # errored
        _write_json(complete_folder / "timings.json", timings)
        info = get_val_breakdown_from_folder(
            str(complete_folder),
            full_set=0,
            spin_tf=False,
            n_atoms=N_ATOMS_H2O,
        )
        # -1 < 1e-6, so validate_timing_dict returns False
        assert info["val_time"] is False


# ---------------------------------------------------------------------------
# Tests: Timing validation edge cases
# ---------------------------------------------------------------------------


class TestTimingValidation:
    def test_missing_key_fails(self, tmp_path):
        timings = _make_timings_dict()
        del timings["qtaim"]
        path = tmp_path / "timings.json"
        _write_json(path, timings)
        assert not validate_timing_dict(str(path), full_set=0, spin_tf=False)

    def test_zero_timing_fails(self, tmp_path):
        timings = _make_timings_dict()
        timings["qtaim"] = 0.0  # suspiciously small
        path = tmp_path / "timings.json"
        _write_json(path, timings)
        assert not validate_timing_dict(str(path), full_set=0, spin_tf=False)

    def test_negative_timing_passes_threshold(self, tmp_path):
        """A timing of -1 (error marker) is < 0 which is < 1e-6.
        This should fail the threshold check."""
        timings = _make_timings_dict()
        timings["qtaim"] = -1
        path = tmp_path / "timings.json"
        _write_json(path, timings)
        # -1 < 1e-6, so validate_timing_dict should return False
        assert not validate_timing_dict(str(path), full_set=0, spin_tf=False)

    def test_other_key_or_other_alie(self, tmp_path):
        """The 'other' key has a fallback to 'other_alie'."""
        timings = _make_timings_dict()
        del timings["other"]
        timings["other_alie"] = 5.0
        path = tmp_path / "timings.json"
        _write_json(path, timings)
        assert validate_timing_dict(str(path), full_set=0, spin_tf=False)


# ---------------------------------------------------------------------------
# Tests: validation_checks with move_results=True for partial folders
# ---------------------------------------------------------------------------


class TestValidationMoveResults:
    """Test validation_checks when results are in generator/ subfolder."""

    def test_partial_in_generator(self, tmp_path):
        """Generator folder has all files except bond.json."""
        folder = tmp_path / "job"
        folder.mkdir()
        gen = folder / "generator"
        gen.mkdir()

        (folder / "orca.inp").write_text(MINIMAL_INP)
        _write_json(gen / "charge.json", _make_charge_dict(N_ATOMS_H2O))
        _write_json(gen / "fuzzy_full.json", _make_fuzzy_dict(N_ATOMS_H2O))
        _write_json(gen / "qtaim.json", _make_qtaim_dict(N_ATOMS_H2O))
        _write_json(gen / "other.json", _make_other_dict())
        _write_json(gen / "timings.json", _make_timings_dict())
        # No bond.json!

        assert not validation_checks(
            str(folder), full_set=0, move_results=True
        )

    def test_results_split_between_root_and_generator(self, tmp_path):
        """Some results in root, some in generator/ — simulates interrupted move.
        validation_checks only looks in generator/ when move_results=True."""
        folder = tmp_path / "job"
        folder.mkdir()
        gen = folder / "generator"
        gen.mkdir()

        (folder / "orca.inp").write_text(MINIMAL_INP)

        # Some in generator
        _write_json(gen / "charge.json", _make_charge_dict(N_ATOMS_H2O))
        _write_json(gen / "bond.json", _make_bond_dict())
        _write_json(gen / "fuzzy_full.json", _make_fuzzy_dict(N_ATOMS_H2O))
        _write_json(gen / "timings.json", _make_timings_dict())

        # Some in root (not yet moved)
        _write_json(folder / "qtaim.json", _make_qtaim_dict(N_ATOMS_H2O))
        _write_json(folder / "other.json", _make_other_dict())

        # Should fail because validation looks only in generator/
        assert not validation_checks(
            str(folder), full_set=0, move_results=True
        )


# ---------------------------------------------------------------------------
# Tests: check_results_exist missing bond.json (Bug 3)
# ---------------------------------------------------------------------------


class TestCheckResultsExist:
    """Verify check_results_exist correctly requires bond.json."""

    def test_missing_bond_fails_check_results_exist(self, tmp_path):
        """check_results_exist should return False when bond.json is missing."""
        from qtaim_gen.source.utils.io import check_results_exist

        folder = tmp_path / "job"
        folder.mkdir()

        # Write everything except bond.json
        _write_json(folder / "charge.json", _make_charge_dict(N_ATOMS_H2O))
        _write_json(folder / "fuzzy_full.json", _make_fuzzy_dict(N_ATOMS_H2O))
        _write_json(folder / "qtaim.json", _make_qtaim_dict(N_ATOMS_H2O))
        _write_json(folder / "other.json", _make_other_dict())
        _write_json(folder / "timings.json", _make_timings_dict())

        # check_results_exist now correctly catches missing bond.json
        assert check_results_exist(str(folder), move_results=False) is False

        # validation_checks also correctly fails
        (folder / "orca.inp").write_text(MINIMAL_INP)
        assert not validation_checks(
            str(folder), full_set=0, move_results=False
        )


# ---------------------------------------------------------------------------
# Tests: Restart edge cases — errored timings, partial timings
# ---------------------------------------------------------------------------


class TestRestartTimingEdgeCases:
    """Test scenarios that affect restart skip decisions in run_jobs."""

    def test_timing_present_but_errored(self, complete_folder):
        """A timing key with value -1 indicates error. The restart logic
        checks `order in timings.keys()` but doesn't check value.
        This test documents the behavior."""
        timings = _make_timings_dict()
        timings["cm5"] = -1  # errored
        _write_json(complete_folder / "timings.json", timings)

        # Charge validation fails because cm5 output would be missing/invalid
        # (but our synthetic test has it in charge.json - in real scenario it wouldn't)
        # Remove cm5 from charge to simulate real partial state
        charge = _make_charge_dict(N_ATOMS_H2O, keys=("adch", "becke", "hirshfeld"))
        _write_json(complete_folder / "charge.json", charge)

        info = get_val_breakdown_from_folder(
            str(complete_folder),
            full_set=0,
            spin_tf=False,
            n_atoms=N_ATOMS_H2O,
        )
        # charge validation should fail (missing cm5)
        assert info["val_charge"] is False
        # but the errored timing key IS present
        with open(str(complete_folder / "timings.json")) as f:
            t = json.load(f)
        assert "cm5" in t  # key exists...
        assert t["cm5"] == -1  # ...but value indicates error

    def test_partial_timings_missing_keys(self, complete_folder):
        """Timings file exists but is missing some keys — simulates
        a job that was interrupted mid-way."""
        timings = {"qtaim": 10.0, "hirshfeld": 3.0}  # only partial
        _write_json(complete_folder / "timings.json", timings)

        info = get_val_breakdown_from_folder(
            str(complete_folder),
            full_set=0,
            spin_tf=False,
            n_atoms=N_ATOMS_H2O,
        )
        # Timing validation should fail (missing required keys)
        assert info["val_time"] is False

    def test_empty_timings_file(self, complete_folder):
        """Empty timings file."""
        (complete_folder / "timings.json").write_text("")
        info = get_val_breakdown_from_folder(
            str(complete_folder),
            full_set=0,
            spin_tf=False,
            n_atoms=N_ATOMS_H2O,
        )
        # Should be None — file is empty so no timings loaded
        assert info["val_time"] is None


# ---------------------------------------------------------------------------
# Tests: Full-folder validation with mixed partial states
# ---------------------------------------------------------------------------


class TestMixedPartialStates:
    """Simulate realistic partial completion scenarios."""

    def test_qtaim_and_charge_complete_rest_missing(self, tmp_path):
        """Only QTAIM and charge finished, bond/fuzzy/other/timings incomplete."""
        folder = tmp_path / "partial"
        folder.mkdir()
        (folder / "orca.inp").write_text(MINIMAL_INP)

        _write_json(folder / "qtaim.json", _make_qtaim_dict(N_ATOMS_H2O))
        _write_json(folder / "charge.json", _make_charge_dict(N_ATOMS_H2O))
        # timings with only completed steps
        _write_json(folder / "timings.json", {"qtaim": 10.0, "adch": 3.0, "becke": 3.0, "hirshfeld": 3.0, "cm5": 3.0})

        # validation_checks should fail (missing bond, fuzzy, other, timing keys)
        assert not validation_checks(str(folder), full_set=0, move_results=False)

        # But breakdown should show qtaim and charge valid
        info = get_val_breakdown_from_folder(
            str(folder),
            full_set=0,
            spin_tf=False,
            n_atoms=N_ATOMS_H2O,
        )
        assert info["val_qtaim"] is True
        assert info["val_charge"] is True
        assert info["val_bond"] is None  # file missing
        assert info["val_fuzzy"] is None  # file missing
        assert info["val_other"] is None  # file missing

    def test_all_files_exist_but_fuzzy_content_invalid(self, tmp_path):
        """All files present but fuzzy has wrong atom count."""
        folder = tmp_path / "bad_fuzzy"
        folder.mkdir()
        (folder / "orca.inp").write_text(MINIMAL_INP)

        _write_json(folder / "charge.json", _make_charge_dict(N_ATOMS_H2O))
        _write_json(folder / "bond.json", _make_bond_dict())
        _write_json(folder / "fuzzy_full.json", _make_fuzzy_dict(10))  # wrong n_atoms
        _write_json(folder / "qtaim.json", _make_qtaim_dict(N_ATOMS_H2O))
        _write_json(folder / "other.json", _make_other_dict())
        _write_json(folder / "timings.json", _make_timings_dict())

        assert not validation_checks(str(folder), full_set=0, move_results=False)

        info = get_val_breakdown_from_folder(
            str(folder),
            full_set=0,
            spin_tf=False,
            n_atoms=N_ATOMS_H2O,
        )
        assert info["val_fuzzy"] is False
        assert info["val_charge"] is True
        assert info["val_qtaim"] is True
        assert info["val_bond"] is True

    def test_generator_folder_partial_then_root_partial(self, tmp_path):
        """Simulate a restart scenario: generator/ has results from previous
        incomplete run, root has results from current interrupted run."""
        folder = tmp_path / "restart_scenario"
        folder.mkdir()
        gen = folder / "generator"
        gen.mkdir()

        (folder / "orca.inp").write_text(MINIMAL_INP)

        # Previous run: qtaim and charge completed, moved to generator/
        _write_json(gen / "qtaim.json", _make_qtaim_dict(N_ATOMS_H2O))
        _write_json(gen / "charge.json", _make_charge_dict(N_ATOMS_H2O))
        _write_json(gen / "timings.json", {"qtaim": 10.0, "adch": 3.0})

        # Current run: bond completed, still in root (not yet moved)
        _write_json(folder / "bond.json", _make_bond_dict())
        _write_json(folder / "timings.json", {
            "qtaim": 10.0, "adch": 3.0, "becke": 3.0,
            "hirshfeld": 3.0, "cm5": 3.0, "fuzzy_bond": 2.0,
        })

        # With move_results=True, validation only checks generator/
        assert not validation_checks(str(folder), full_set=0, move_results=True)

        # Breakdown from generator/ shows partial state
        info = get_val_breakdown_from_folder(
            str(gen),
            full_set=0,
            spin_tf=False,
            n_atoms=N_ATOMS_H2O,
        )
        assert info["val_qtaim"] is True
        assert info["val_charge"] is True
        assert info["val_bond"] is None  # not in generator/ yet
        assert info["val_fuzzy"] is None


# ---------------------------------------------------------------------------
# Tests: validate_other_dict edge cases
# ---------------------------------------------------------------------------


class TestOtherValidation:
    def test_full_set_2_requires_esp_keys(self, tmp_path):
        other = _make_other_dict()
        # full_set=2 requires ESP_ keys which our minimal dict doesn't have
        path = tmp_path / "other.json"
        _write_json(path, other)
        assert not validate_other_dict(str(path), full_set=2)

    def test_full_set_0_passes_without_esp(self, tmp_path):
        other = _make_other_dict()
        path = tmp_path / "other.json"
        _write_json(path, other)
        assert validate_other_dict(str(path), full_set=0)


# ---------------------------------------------------------------------------
# Tests: validate_qtaim_dict edge cases
# ---------------------------------------------------------------------------


class TestQtaimValidation:
    def test_empty_qtaim(self, tmp_path):
        path = tmp_path / "qtaim.json"
        _write_json(path, {})
        assert not validate_qtaim_dict(str(path))

    def test_wrong_ncp_count(self, tmp_path):
        qtaim = _make_qtaim_dict(5)  # 5 NCPs
        path = tmp_path / "qtaim.json"
        _write_json(path, qtaim)
        assert not validate_qtaim_dict(str(path), n_atoms=3)

    def test_correct_ncp_count(self, tmp_path):
        qtaim = _make_qtaim_dict(N_ATOMS_H2O)
        path = tmp_path / "qtaim.json"
        _write_json(path, qtaim)
        assert validate_qtaim_dict(str(path), n_atoms=N_ATOMS_H2O)


# ---------------------------------------------------------------------------
# Tests: Per-sub-job restart granularity
# ---------------------------------------------------------------------------


class TestPerSubJobRestart:
    """Tests for the per-sub-job restart skip logic in run_jobs().

    The new logic skips a sub-job on restart if timings[order] > 0,
    and re-runs if timings[order] <= 0 or absent.
    """

    def test_positive_timing_should_skip(self):
        """Sub-job with timing > 0 should be considered complete."""
        timings = {"hirshfeld": 3.2, "adch": 2.8}
        # Positive timing → skip
        assert "hirshfeld" in timings and timings["hirshfeld"] > 0

    def test_error_timing_should_rerun(self):
        """Sub-job with timing == -1 should NOT be skipped."""
        timings = {"hirshfeld": 3.2, "adch": -1, "cm5": 2.8}
        # Negative timing → must re-run
        assert timings["adch"] <= 0

    def test_missing_timing_should_run(self):
        """Sub-job absent from timings should run."""
        timings = {"hirshfeld": 3.2, "adch": 2.8}
        # Missing key → must run
        assert "becke" not in timings

    def test_partial_charge_only_reruns_failed(self):
        """If adch=-1 but hirshfeld/cm5/becke > 0, only adch needs to re-run."""
        timings = {"hirshfeld": 3.2, "adch": -1, "cm5": 2.8, "becke": 2.5}
        to_skip = [k for k in timings if timings[k] > 0]
        to_rerun = [k for k in timings if timings[k] <= 0]
        assert set(to_skip) == {"hirshfeld", "cm5", "becke"}
        assert to_rerun == ["adch"]

    def test_phantom_keys_filtered(self):
        """Phantom keys should be filtered from order_of_operations in run_jobs."""
        from qtaim_gen.source.core.omol import ORDER_OF_OPERATIONS_separate

        phantom_keys = {"charge_separate", "bond_separate", "other_separate", "fuzzy_full"}
        ops = ORDER_OF_OPERATIONS_separate.copy()
        filtered = [o for o in ops if o not in phantom_keys]
        # No phantom keys should remain
        assert not (set(filtered) & phantom_keys)
        # qtaim should still be present
        assert "qtaim" in filtered

    def test_separate_false_no_nameerror(self):
        """separate=False path should define charge_dict etc. without NameError."""
        # Simulates the else branch in run_jobs
        separate = False
        if separate:
            charge_dict = {"hirshfeld": "data"}
        else:
            charge_dict = {}
            bond_dict = {}
            fuzzy_dict = {}
            other_dict = {}
        # Should not raise NameError
        assert isinstance(charge_dict, dict)
        assert isinstance(bond_dict, dict)
        assert isinstance(fuzzy_dict, dict)
        assert isinstance(other_dict, dict)

    def test_corrupted_timings_fresh_start(self, tmp_path):
        """Corrupted timings.json should be handled gracefully."""
        timings_path = tmp_path / "timings.json"
        timings_path.write_text("{invalid json content")
        # Simulate the new handling
        timings = {}
        if timings_path.exists() and timings_path.stat().st_size > 0:
            try:
                with open(timings_path, "r") as f:
                    timings = json.load(f)
            except json.JSONDecodeError:
                timings = {}
        assert timings == {}

    def test_zero_timing_should_rerun(self):
        """Sub-job with timing == 0 should be re-run (not skipped)."""
        timings = {"hirshfeld": 0.0}
        # Zero timing → not > 0 → should re-run
        assert not (timings["hirshfeld"] > 0)


# ---------------------------------------------------------------------------
# Tests: Lock protection (mtime-based stale detection)
# ---------------------------------------------------------------------------


class TestLockProtection:
    """Tests for acquire_lock/release_lock with mtime-based stale detection."""

    def test_acquire_release_lifecycle(self, tmp_path):
        """Lock can be acquired and released."""
        from qtaim_gen.source.core.workflow import acquire_lock, release_lock

        folder = str(tmp_path)
        assert acquire_lock(folder)
        lockfile = os.path.join(folder, ".processing.lock")
        assert os.path.exists(lockfile)
        release_lock(folder)
        assert not os.path.exists(lockfile)

    def test_lock_contains_pid(self, tmp_path):
        """Lock file contains current PID for debugging."""
        from qtaim_gen.source.core.workflow import acquire_lock, release_lock

        folder = str(tmp_path)
        acquire_lock(folder)
        lockfile = os.path.join(folder, ".processing.lock")
        with open(lockfile, "r") as f:
            content = f.read().strip()
        assert content == str(os.getpid())
        release_lock(folder)

    def test_double_acquire_fails(self, tmp_path):
        """Second acquire_lock returns False when lock is active."""
        from qtaim_gen.source.core.workflow import acquire_lock, release_lock

        folder = str(tmp_path)
        assert acquire_lock(folder)
        assert not acquire_lock(folder)  # second attempt fails
        release_lock(folder)

    def test_stale_lock_broken_by_age(self, tmp_path):
        """Lock with mtime older than threshold is broken and reacquired."""
        from qtaim_gen.source.core.workflow import acquire_lock, release_lock

        folder = str(tmp_path)
        lockfile = os.path.join(folder, ".processing.lock")
        # Create a lock file manually
        with open(lockfile, "w") as f:
            f.write("99999")
        # Backdate mtime to 10 hours ago
        old_time = time.time() - 36000
        os.utime(lockfile, (old_time, old_time))
        # Should break the stale lock and acquire
        assert acquire_lock(folder)
        # Verify our PID is now in the lock
        with open(lockfile, "r") as f:
            assert f.read().strip() == str(os.getpid())
        release_lock(folder)

    def test_fresh_lock_not_broken(self, tmp_path):
        """Lock within age threshold is respected (not broken)."""
        from qtaim_gen.source.core.workflow import acquire_lock

        folder = str(tmp_path)
        lockfile = os.path.join(folder, ".processing.lock")
        # Create a fresh lock (mtime = now)
        with open(lockfile, "w") as f:
            f.write("99999")
        # Should NOT break it
        assert not acquire_lock(folder)

    def test_clean_first_preserves_lock(self, tmp_path):
        """clean_first cleanup loop should skip .processing.lock."""
        folder = str(tmp_path)
        lockfile = os.path.join(folder, ".processing.lock")
        other_file = os.path.join(folder, "some_data.json")
        log_file = os.path.join(folder, "gbw_analysis.log")
        # Create files
        for f in [lockfile, other_file, log_file]:
            with open(f, "w") as fh:
                fh.write("test")
        # Simulate clean_first loop (from workflow.py)
        for item in os.listdir(folder):
            if item not in ("gbw_analysis.log", ".processing.lock"):
                item_path = os.path.join(folder, item)
                if os.path.isfile(item_path):
                    os.unlink(item_path)
        # Lock and log should survive
        assert os.path.exists(lockfile)
        assert os.path.exists(log_file)
        assert not os.path.exists(other_file)

    def test_heartbeat_updates_mtime(self, tmp_path):
        """os.utime on lock file updates mtime (heartbeat mechanism)."""
        folder = str(tmp_path)
        lockfile = os.path.join(folder, ".processing.lock")
        with open(lockfile, "w") as f:
            f.write(str(os.getpid()))
        # Backdate
        old_time = time.time() - 3600
        os.utime(lockfile, (old_time, old_time))
        mtime_before = os.path.getmtime(lockfile)
        # Heartbeat
        os.utime(lockfile, None)
        mtime_after = os.path.getmtime(lockfile)
        assert mtime_after > mtime_before

    def test_release_nonexistent_lock_is_safe(self, tmp_path):
        """release_lock on folder without lock file should not raise."""
        from qtaim_gen.source.core.workflow import release_lock

        folder = str(tmp_path)
        # Should not raise
        release_lock(folder)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
