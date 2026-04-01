"""Tests for empty bond validation edge cases (two-atom molecules)."""
import json
import pytest

from qtaim_gen.source.utils.validation import validate_bond_dict, validate_timing_dict
from qtaim_gen.source.utils.lmdbs import parse_bond_data


class TestValidateBondDictTwoAtom:
    """validate_bond_dict with n_atoms parameter for small molecules."""

    def test_empty_values_n_atoms_2(self, tmp_path):
        """Empty bond values accepted for n_atoms=2."""
        bond_file = tmp_path / "bond.json"
        bond_file.write_text('{"fuzzy_bond": {}}')
        assert validate_bond_dict(str(bond_file), n_atoms=2) is True

    def test_missing_key_n_atoms_2(self, tmp_path):
        """Missing bond key accepted for n_atoms=2."""
        bond_file = tmp_path / "bond.json"
        bond_file.write_text('{}')
        assert validate_bond_dict(str(bond_file), n_atoms=2) is True

    def test_missing_key_n_atoms_5(self, tmp_path):
        """Missing bond key rejected for n_atoms=5."""
        bond_file = tmp_path / "bond.json"
        bond_file.write_text('{}')
        assert validate_bond_dict(str(bond_file), n_atoms=5) is False

    def test_empty_values_n_atoms_5(self, tmp_path):
        """Empty bond values still accepted for n_atoms=5 (key-presence-only check)."""
        bond_file = tmp_path / "bond.json"
        bond_file.write_text('{"fuzzy_bond": {}}')
        assert validate_bond_dict(str(bond_file), n_atoms=5) is True

    def test_n_atoms_none_preserves_behavior(self, tmp_path):
        """Default n_atoms=None preserves existing behavior (missing key -> False)."""
        bond_file = tmp_path / "bond.json"
        bond_file.write_text('{}')
        assert validate_bond_dict(str(bond_file)) is False

    def test_n_atoms_1_accepted(self, tmp_path):
        """Single-atom molecule with no bond keys accepted."""
        bond_file = tmp_path / "bond.json"
        bond_file.write_text('{}')
        assert validate_bond_dict(str(bond_file), n_atoms=1) is True


class TestValidateTimingDictTwoAtom:
    """validate_timing_dict with n_atoms parameter for small molecules."""

    @pytest.fixture
    def base_timings(self):
        return {
            "qtaim": 1.0, "other": 1.0, "hirshfeld": 1.0, "becke": 1.0,
            "adch": 1.0, "cm5": 1.0, "fuzzy_bond": 1.0,
            "becke_fuzzy_density": 1.0, "hirsh_fuzzy_density": 1.0,
        }

    def test_near_zero_bond_timing_n_atoms_2(self, tmp_path, base_timings):
        """Near-zero fuzzy_bond timing passes for n_atoms=2."""
        base_timings["fuzzy_bond"] = 1e-9
        timing_file = tmp_path / "timings.json"
        timing_file.write_text(json.dumps(base_timings))
        assert validate_timing_dict(str(timing_file), n_atoms=2) is True

    def test_near_zero_bond_timing_n_atoms_5(self, tmp_path, base_timings):
        """Near-zero fuzzy_bond timing fails for n_atoms=5."""
        base_timings["fuzzy_bond"] = 1e-9
        timing_file = tmp_path / "timings.json"
        timing_file.write_text(json.dumps(base_timings))
        assert validate_timing_dict(str(timing_file), n_atoms=5) is False

    def test_near_zero_non_bond_timing_still_fails_n_atoms_2(self, tmp_path, base_timings):
        """Near-zero non-bond timing (e.g. qtaim) still fails even for n_atoms=2."""
        base_timings["qtaim"] = 1e-9
        timing_file = tmp_path / "timings.json"
        timing_file.write_text(json.dumps(base_timings))
        assert validate_timing_dict(str(timing_file), n_atoms=2) is False

    def test_n_atoms_none_preserves_behavior(self, tmp_path, base_timings):
        """Default n_atoms=None preserves existing behavior (tiny timing -> False)."""
        base_timings["fuzzy_bond"] = 1e-9
        timing_file = tmp_path / "timings.json"
        timing_file.write_text(json.dumps(base_timings))
        assert validate_timing_dict(str(timing_file)) is False


class TestParseBondDataEmpty:
    """parse_bond_data with empty bond dicts."""

    def test_empty_fuzzy_bond(self):
        """Empty fuzzy_bond dict returns empty results without error."""
        bond_feats, bond_list = parse_bond_data(
            {"fuzzy_bond": {}}, bond_list_definition="fuzzy", as_lists=False
        )
        assert bond_feats == {}
        assert bond_list == []

    def test_empty_multiple_bond_types(self):
        """Multiple empty bond types return empty results."""
        bond_feats, bond_list = parse_bond_data(
            {"fuzzy_bond": {}, "ibsi_bond": {}},
            bond_list_definition="fuzzy", as_lists=False,
        )
        assert bond_feats == {}
        assert bond_list == []
