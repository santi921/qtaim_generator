"""Tests for core/lmdb_analysis.py — LMDB descriptor analysis utilities."""
from __future__ import annotations

import math
import os
import pytest
import numpy as np

from qtaim_gen.source.core.lmdb_analysis import (
    iter_lmdb,
    get_lmdb_entry_count,
    discover_lmdbs,
    compute_stats,
    flatten_entry,
    flatten_bond_entry,
    extract_structure_info,
    split_qtaim_atom_bond,
    METADATA_KEYS,
)

# Path to merged LMDB test fixtures
FIXTURE_DIR = os.path.join(
    os.path.dirname(__file__), "test_files", "lmdb_tests", "generator_lmdbs_merged"
)


# ---- iter_lmdb ----


class TestIterLmdb:
    def test_yields_correct_count(self):
        """iter_lmdb yields all non-metadata entries."""
        path = os.path.join(FIXTURE_DIR, "merged_charge.lmdb")
        entries = list(iter_lmdb(path))
        assert len(entries) == 4

    def test_skips_metadata_keys(self):
        """No yielded key should be in METADATA_KEYS."""
        path = os.path.join(FIXTURE_DIR, "merged_charge.lmdb")
        for key, _data in iter_lmdb(path):
            assert key not in METADATA_KEYS

    def test_yields_string_keys(self):
        """Keys should be decoded strings, not bytes."""
        path = os.path.join(FIXTURE_DIR, "merged_charge.lmdb")
        for key, _data in iter_lmdb(path):
            assert isinstance(key, str)

    def test_yields_dict_values(self):
        """Values should be dicts (for charge LMDB)."""
        path = os.path.join(FIXTURE_DIR, "merged_charge.lmdb")
        for _key, data in iter_lmdb(path):
            assert isinstance(data, dict)

    def test_sample_size_limits_output(self):
        """With sample_size < total, should yield fewer entries."""
        path = os.path.join(FIXTURE_DIR, "merged_charge.lmdb")
        # 4 entries, sample_size=2 -> step=2, should yield ~2
        entries = list(iter_lmdb(path, sample_size=2))
        assert len(entries) <= 4
        assert len(entries) >= 1

    def test_sample_size_none_yields_all(self):
        """sample_size=None should yield all entries."""
        path = os.path.join(FIXTURE_DIR, "merged_charge.lmdb")
        entries = list(iter_lmdb(path, sample_size=None))
        assert len(entries) == 4

    def test_works_on_all_lmdb_types(self):
        """Should successfully iterate all 6 merged LMDB types."""
        for name in [
            "merged_charge.lmdb",
            "merged_bond.lmdb",
            "merged_fuzzy.lmdb",
            "merged_other.lmdb",
            "merged_qtaim.lmdb",
            "merged_geom.lmdb",
        ]:
            path = os.path.join(FIXTURE_DIR, name)
            entries = list(iter_lmdb(path))
            assert len(entries) == 4, f"{name} should have 4 entries"


# ---- get_lmdb_entry_count ----


class TestGetLmdbEntryCount:
    def test_returns_correct_count(self):
        path = os.path.join(FIXTURE_DIR, "merged_charge.lmdb")
        assert get_lmdb_entry_count(path) == 4


# ---- discover_lmdbs ----


class TestDiscoverLmdbs:
    def test_finds_merged_lmdbs(self):
        """Should find all 6 types using merged_ prefix variants."""
        found = discover_lmdbs(FIXTURE_DIR)
        assert len(found) == 6
        for type_name in ["structure", "charge", "qtaim", "bond", "fuzzy", "other"]:
            assert type_name in found
            assert found[type_name].exists()

    def test_returns_path_objects(self):
        found = discover_lmdbs(FIXTURE_DIR)
        from pathlib import Path

        for _name, path in found.items():
            assert isinstance(path, Path)

    def test_empty_dir_returns_empty(self, tmp_path):
        found = discover_lmdbs(tmp_path)
        assert len(found) == 0


# ---- compute_stats ----


class TestComputeStats:
    def test_basic_stats(self):
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = compute_stats(values)
        assert result["count"] == 5
        assert result["nan_count"] == 0
        assert result["inf_count"] == 0
        assert result["mean"] == pytest.approx(3.0)
        assert result["std"] == pytest.approx(np.std(values))
        assert result["min"] == pytest.approx(1.0)
        assert result["max"] == pytest.approx(5.0)

    def test_nan_handling(self):
        values = np.array([1.0, float("nan"), 3.0, float("nan"), 5.0])
        result = compute_stats(values)
        assert result["count"] == 5
        assert result["nan_count"] == 2
        assert result["mean"] == pytest.approx(3.0)

    def test_inf_handling(self):
        values = np.array([1.0, float("inf"), 3.0, float("-inf")])
        result = compute_stats(values)
        assert result["inf_count"] == 2
        assert result["mean"] == pytest.approx(2.0)

    def test_all_nan(self):
        values = np.array([float("nan"), float("nan"), float("nan")])
        result = compute_stats(values)
        assert result["count"] == 3
        assert result["nan_count"] == 3
        assert result["mean"] is None
        assert result["std"] is None
        assert result["min"] is None
        assert result["max"] is None
        assert result["percentiles"] == {}

    def test_single_value(self):
        values = np.array([42.0])
        result = compute_stats(values)
        assert result["count"] == 1
        assert result["mean"] == pytest.approx(42.0)
        assert result["std"] == pytest.approx(0.0)

    def test_empty_array(self):
        values = np.array([])
        result = compute_stats(values)
        assert result["count"] == 0
        assert result["mean"] is None

    def test_percentiles(self):
        values = np.arange(100, dtype=np.float64)
        result = compute_stats(values)
        assert 5 in result["percentiles"]
        assert 50 in result["percentiles"]
        assert result["percentiles"][50] == pytest.approx(49.5)

    def test_mixed_nan_inf(self):
        values = np.array([1.0, float("nan"), float("inf"), 3.0])
        result = compute_stats(values)
        assert result["nan_count"] == 1
        assert result["inf_count"] == 1
        assert result["mean"] == pytest.approx(2.0)


# ---- flatten_entry ----


class TestFlattenEntry:
    def test_flat_dict(self):
        entry = {"a": 1.0, "b": 2.0, "c": 3.0}
        result = flatten_entry(entry)
        assert result == {"a": 1.0, "b": 2.0, "c": 3.0}

    def test_nested_dict(self):
        entry = {"adch": {"charge": {"1_C": -0.1, "2_H": 0.05}}}
        result = flatten_entry(entry)
        assert result["adch.charge.1_C"] == pytest.approx(-0.1)
        assert result["adch.charge.2_H"] == pytest.approx(0.05)

    def test_skips_non_numeric(self):
        entry = {"a": 1.0, "b": "string", "c": [1, 2, 3], "d": True}
        result = flatten_entry(entry)
        assert result == {"a": 1.0}

    def test_charge_lmdb_entry(self):
        """Flatten a real charge LMDB entry structure."""
        path = os.path.join(FIXTURE_DIR, "merged_charge.lmdb")
        for _key, data in iter_lmdb(path):
            result = flatten_entry(data)
            # Should have charge values like adch.charge.1_C
            charge_keys = [k for k in result if k.startswith("adch.charge.")]
            assert len(charge_keys) > 0
            # Should have dipole values
            dipole_keys = [k for k in result if "dipole.mag" in k]
            assert len(dipole_keys) > 0
            break

    def test_other_lmdb_entry(self):
        """Other LMDB has flat structure — flatten should preserve it."""
        path = os.path.join(FIXTURE_DIR, "merged_other.lmdb")
        for _key, data in iter_lmdb(path):
            result = flatten_entry(data)
            assert "mpp_full" in result
            assert "ESP_Volume" in result
            break

    def test_fuzzy_separates_sum_abs_sum(self):
        """Fuzzy entries have sum/abs_sum as numeric leaves — they should be included."""
        path = os.path.join(FIXTURE_DIR, "merged_fuzzy.lmdb")
        for _key, data in iter_lmdb(path):
            result = flatten_entry(data)
            # sum/abs_sum should appear as e.g. mbis_fuzzy_spin.sum
            sum_keys = [k for k in result if k.endswith(".sum")]
            abs_sum_keys = [k for k in result if k.endswith(".abs_sum")]
            assert len(sum_keys) > 0
            assert len(abs_sum_keys) > 0
            break


# ---- flatten_bond_entry ----


class TestFlattenBondEntry:
    def test_normalizes_bond_suffix(self):
        entry = {"fuzzy_bond": {"1_C_to_2_H": 0.86}, "ibsi_bond": {"1_C_to_2_H": 0.75}}
        result = flatten_bond_entry(entry)
        assert "fuzzy.1_C_to_2_H" in result
        assert "ibsi.1_C_to_2_H" in result
        # Should NOT have _bond suffix
        assert not any("_bond" in k for k in result)

    def test_handles_no_suffix(self):
        """Keys without _bond suffix should pass through unchanged."""
        entry = {"fuzzy": {"1_C_to_2_H": 0.86}, "ibsi": {"1_C_to_2_H": 0.75}}
        result = flatten_bond_entry(entry)
        assert "fuzzy.1_C_to_2_H" in result
        assert "ibsi.1_C_to_2_H" in result

    def test_real_bond_lmdb(self):
        """Test on actual bond LMDB fixture."""
        path = os.path.join(FIXTURE_DIR, "merged_bond.lmdb")
        for _key, data in iter_lmdb(path):
            result = flatten_bond_entry(data)
            assert len(result) > 0
            # All keys should be normalized (no _bond suffix)
            assert not any("_bond" in k for k in result)
            break


# ---- extract_structure_info ----


class TestExtractStructureInfo:
    def test_extracts_from_real_entry(self):
        """Test on actual structure LMDB fixture."""
        path = os.path.join(FIXTURE_DIR, "merged_geom.lmdb")
        for _key, data in iter_lmdb(path):
            info = extract_structure_info(data)
            assert info["n_atoms"] > 0
            assert info["n_bonds"] >= 0
            assert isinstance(info["charge"], int)
            assert isinstance(info["spin"], int)
            assert isinstance(info["elements"], list)
            assert len(info["elements"]) == info["n_atoms"]
            assert all(isinstance(e, str) for e in info["elements"])
            break

    def test_handles_missing_molecule(self):
        """Should not crash on an entry without molecule key."""
        info = extract_structure_info({"ids": "test", "bonds": [], "charge": 0, "spin": 1})
        assert info["n_atoms"] == 0
        assert info["elements"] == []


# ---- split_qtaim_atom_bond ----


class TestSplitQtaimAtomBond:
    def test_splits_correctly(self):
        entry = {
            "0": {"density_all": 1.0, "spin_density": 0.5},
            "1": {"density_all": 2.0, "spin_density": 0.3},
            "0_1": {"density_all": 0.5, "connected_bond_paths": [0, 1]},
        }
        atoms, bonds = split_qtaim_atom_bond(entry)
        assert "0" in atoms
        assert "1" in atoms
        assert "0_1" in bonds
        assert len(atoms) == 2
        assert len(bonds) == 1

    def test_does_not_misclassify_sub_keys(self):
        """Sub-keys like 'spin_density' (with underscore) should not affect
        the top-level split — only top-level keys matter."""
        entry = {
            "0": {"spin_density": 0.5, "energy_density": -1.0},
            "1_2": {"spin_density": 0.1, "connected_bond_paths": [1, 2]},
        }
        atoms, bonds = split_qtaim_atom_bond(entry)
        assert "0" in atoms
        assert "1_2" in bonds
        # The spin_density sub-key should be preserved in both
        assert "spin_density" in atoms["0"]
        assert "spin_density" in bonds["1_2"]

    def test_real_qtaim_lmdb(self):
        """Test on actual QTAIM LMDB fixture."""
        path = os.path.join(FIXTURE_DIR, "merged_qtaim.lmdb")
        for _key, data in iter_lmdb(path):
            atoms, bonds = split_qtaim_atom_bond(data)
            assert len(atoms) > 0
            assert len(bonds) > 0
            # Atom keys should be plain numbers
            for ak in atoms:
                assert "_" not in ak
            # Bond keys should contain underscore
            for bk in bonds:
                assert "_" in bk
            break

    def test_skips_non_dict_values(self):
        """Non-dict values (metadata) should be skipped."""
        entry = {
            "0": {"density_all": 1.0},
            "some_metadata": "not a dict",
            "0_1": {"density_all": 0.5},
        }
        atoms, bonds = split_qtaim_atom_bond(entry)
        assert "0" in atoms
        assert "0_1" in bonds
        assert len(atoms) == 1
        assert len(bonds) == 1


# ---- NaN detection on real fixtures ----


class TestNanDetection:
    def test_other_lmdb_has_known_nan(self):
        """ESP_Positive_average_value should be NaN in at least one entry."""
        path = os.path.join(FIXTURE_DIR, "merged_other.lmdb")
        found_nan = False
        for _key, data in iter_lmdb(path):
            result = flatten_entry(data)
            val = result.get("ESP_Positive_average_value")
            if val is not None and math.isnan(val):
                found_nan = True
                break
        assert found_nan, "Expected NaN in ESP_Positive_average_value"

    def test_compute_stats_reports_nan_from_other(self):
        """compute_stats should report nan_count > 0 for ESP_Positive_average_value."""
        path = os.path.join(FIXTURE_DIR, "merged_other.lmdb")
        values = []
        for _key, data in iter_lmdb(path):
            result = flatten_entry(data)
            val = result.get("ESP_Positive_average_value")
            if val is not None:
                values.append(val)
        stats = compute_stats(np.array(values))
        assert stats["nan_count"] > 0
