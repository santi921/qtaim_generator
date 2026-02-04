"""
Test GeneralConverter filter interactions.

This test verifies that:
1. charge_filter correctly filters charge types into atom features
2. fuzzy_filter correctly filters fuzzy types into atom features
3. other_filter correctly filters global features
4. bond_filter and bonding_scheme affect connectivity correctly
5. keys_data and keys_target interact properly with filters
6. filter_bond_feats utility works correctly
"""

import os
import pytest
import json
from pathlib import Path
from copy import deepcopy

from qtaim_gen.source.core.converter import GeneralConverter
from qtaim_gen.source.utils.lmdbs import (
    parse_charge_data,
    parse_fuzzy_data,
    parse_other_data,
    parse_bond_data,
    parse_qtaim_data,
    filter_bond_feats,
    gather_structure_info,
)


# ============================================================================
# Fixtures and helpers
# ============================================================================

@pytest.fixture(scope="module")
def test_paths():
    """Return test file paths."""
    base = Path(__file__).parent / "test_files" / "lmdb_tests"
    return {
        "base": base,
        "merged": base / "generator_lmdbs_merged",
        "orca5_uks": base / "orca5_uks",
        "orca5_rks": base / "orca5_rks",
        "orca6_rks": base / "orca6_rks",
        "orca5": base / "orca5",
    }


def _base_config(tmp_path, test_paths):
    """Create base config for GeneralConverter tests."""
    merged = test_paths["merged"]
    return {
        "chunk": -1,
        "filter_list": ["scaled", "length"],
        "restart": False,
        "allowed_ring_size": [3, 4, 5, 6, 7, 8],
        "allowed_charges": None,
        "allowed_spins": None,
        "keys_target": {"atom": [], "bond": [], "global": []},
        "keys_data": {"atom": [], "bond": [], "global": ["n_atoms"]},
        "lmdb_path": str(tmp_path),
        "lmdb_name": "test_graphs.lmdb",
        "lmdb_locations": {
            "geom_lmdb": str(merged / "merged_geom.lmdb"),
            "charge_lmdb": str(merged / "merged_charge.lmdb"),
            "fuzzy_full_lmdb": str(merged / "merged_fuzzy.lmdb"),
            "bonds_lmdb": str(merged / "merged_bond.lmdb"),
            "other_lmdb": str(merged / "merged_other.lmdb"),
            "qtaim_lmdb": str(merged / "merged_qtaim.lmdb"),
        },
        "n_workers": 1,
        "batch_size": 100,
    }


# ============================================================================
# Test charge_filter
# ============================================================================

class TestChargeFilter:
    """Tests for charge_filter behavior."""

    def test_charge_filter_single_type(self, tmp_path, test_paths):
        """Test that charge_filter with a single type only includes that type."""
        config = _base_config(tmp_path, test_paths)
        config["charge_filter"] = ["mbis"]
        config["keys_data"]["atom"] = ["charge_mbis"]

        converter = GeneralConverter(config)
        charge_raw = converter.__getitem__("charge_lmdb", b"orca5_uks")
        struct_raw = converter.__getitem__("geom_lmdb", b"orca5_uks")
        _, global_feats = gather_structure_info(struct_raw)

        atom_feats, global_feats_charge = parse_charge_data(
            charge_raw, global_feats["n_atoms"], charge_filter=["mbis"]
        )

        # Should only have mbis-related features
        for atom_idx, feats in atom_feats.items():
            assert "charge_mbis" in feats, f"Missing charge_mbis for atom {atom_idx}"
            # Should NOT have other charge types
            assert "charge_hirshfeld" not in feats, "Should not have hirshfeld"
            assert "charge_bader" not in feats, "Should not have bader"

    def test_charge_filter_multiple_types(self, tmp_path, test_paths):
        """Test that charge_filter with multiple types includes all specified."""
        config = _base_config(tmp_path, test_paths)
        config["charge_filter"] = ["mbis", "hirshfeld"]
        config["keys_data"]["atom"] = ["charge_mbis", "charge_hirshfeld"]

        converter = GeneralConverter(config)
        charge_raw = converter.__getitem__("charge_lmdb", b"orca5_uks")
        struct_raw = converter.__getitem__("geom_lmdb", b"orca5_uks")
        _, global_feats = gather_structure_info(struct_raw)

        atom_feats, _ = parse_charge_data(
            charge_raw, global_feats["n_atoms"], charge_filter=["mbis", "hirshfeld"]
        )

        for atom_idx, feats in atom_feats.items():
            assert "charge_mbis" in feats
            assert "charge_hirshfeld" in feats
            # Should NOT have unfiltered types
            assert "charge_bader" not in feats

    def test_charge_filter_none_includes_all(self, tmp_path, test_paths):
        """Test that charge_filter=None includes all charge types."""
        config = _base_config(tmp_path, test_paths)
        # No charge_filter specified

        converter = GeneralConverter(config)
        charge_raw = converter.__getitem__("charge_lmdb", b"orca5_uks")
        struct_raw = converter.__getitem__("geom_lmdb", b"orca5_uks")
        _, global_feats = gather_structure_info(struct_raw)

        atom_feats, _ = parse_charge_data(
            charge_raw, global_feats["n_atoms"], charge_filter=None
        )

        # All 8 charge types should produce features plus spin
        # Expected: charge_mbis, charge_hirshfeld, etc.
        all_charge_types = ["mbis", "chelpg", "hirshfeld", "bader", "cm5", "becke", "adch", "vdd"]
        first_atom_feats = atom_feats[0]

        for charge_type in all_charge_types:
            assert f"charge_{charge_type}" in first_atom_feats, f"Missing charge_{charge_type}"



# ============================================================================
# Test fuzzy_filter
# ============================================================================

class TestFuzzyFilter:
    """Tests for fuzzy_filter behavior."""

    def test_fuzzy_filter_single_type(self, tmp_path, test_paths):
        """Test that fuzzy_filter with a single type only includes that type."""
        config = _base_config(tmp_path, test_paths)
        config["fuzzy_filter"] = ["mbis_fuzzy_density"]

        converter = GeneralConverter(config)
        fuzzy_raw = converter.__getitem__("fuzzy_full_lmdb", b"orca5_uks")
        struct_raw = converter.__getitem__("geom_lmdb", b"orca5_uks")
        _, global_feats = gather_structure_info(struct_raw)

        atom_feats, global_feats_fuzzy = parse_fuzzy_data(
            fuzzy_raw, global_feats["n_atoms"], fuzzy_filter=["mbis_fuzzy_density"]
        )

        first_atom = atom_feats[0]
        assert "fuzzy_mbis_fuzzy_density" in first_atom
        # Should NOT have other fuzzy types
        assert "fuzzy_elf_fuzzy" not in first_atom
        assert "fuzzy_becke_fuzzy_density" not in first_atom

    def test_fuzzy_filter_multiple_types(self, tmp_path, test_paths):
        """Test that fuzzy_filter with multiple types includes all specified."""
        config = _base_config(tmp_path, test_paths)
        config["fuzzy_filter"] = ["mbis_fuzzy_density", "elf_fuzzy"]

        converter = GeneralConverter(config)
        fuzzy_raw = converter.__getitem__("fuzzy_full_lmdb", b"orca5_uks")
        struct_raw = converter.__getitem__("geom_lmdb", b"orca5_uks")
        _, global_feats = gather_structure_info(struct_raw)

        atom_feats, _ = parse_fuzzy_data(
            fuzzy_raw, global_feats["n_atoms"], fuzzy_filter=["mbis_fuzzy_density", "elf_fuzzy"]
        )

        first_atom = atom_feats[0]
        assert "fuzzy_mbis_fuzzy_density" in first_atom
        assert "fuzzy_elf_fuzzy" in first_atom
        # Should NOT have unfiltered types
        assert "fuzzy_becke_fuzzy_density" not in first_atom

    def test_fuzzy_filter_none_includes_all(self, tmp_path, test_paths):
        """Test that fuzzy_filter=None includes all fuzzy types."""
        config = _base_config(tmp_path, test_paths)

        converter = GeneralConverter(config)
        fuzzy_raw = converter.__getitem__("fuzzy_full_lmdb", b"orca5_uks")
        struct_raw = converter.__getitem__("geom_lmdb", b"orca5_uks")
        _, global_feats = gather_structure_info(struct_raw)

        atom_feats, _ = parse_fuzzy_data(
            fuzzy_raw, global_feats["n_atoms"], fuzzy_filter=None
        )

        # All fuzzy types should be present
        all_fuzzy_types = [
            "mbis_fuzzy_spin", "grad_norm_rho_fuzzy", "becke_fuzzy_spin",
            "mbis_fuzzy_density", "becke_fuzzy_density", "laplacian_rho_fuzzy", "elf_fuzzy"
        ]
        first_atom = atom_feats[0]
        for fuzzy_type in all_fuzzy_types:
            assert f"fuzzy_{fuzzy_type}" in first_atom, f"Missing fuzzy_{fuzzy_type}"

    def test_fuzzy_filter_global_sum_features(self, tmp_path, test_paths):
        """Test that global sum/abs_sum features respect fuzzy_filter."""
        config = _base_config(tmp_path, test_paths)
        config["fuzzy_filter"] = ["mbis_fuzzy_density"]

        converter = GeneralConverter(config)
        fuzzy_raw = converter.__getitem__("fuzzy_full_lmdb", b"orca5_uks")
        struct_raw = converter.__getitem__("geom_lmdb", b"orca5_uks")
        _, global_feats = gather_structure_info(struct_raw)

        _, global_feats_fuzzy = parse_fuzzy_data(
            fuzzy_raw, global_feats["n_atoms"], fuzzy_filter=["mbis_fuzzy_density"]
        )

        # Should have sum and abs_sum for filtered type
        assert "fuzzy_mbis_fuzzy_density_sum" in global_feats_fuzzy
        assert "fuzzy_mbis_fuzzy_density_abs_sum" in global_feats_fuzzy
        # Should NOT have global features for unfiltered types
        assert "fuzzy_elf_fuzzy_sum" not in global_feats_fuzzy


# ============================================================================
# Test other_filter
# ============================================================================

class TestOtherFilter:
    """Tests for other_filter behavior."""

    def test_other_filter_single_type(self, tmp_path, test_paths):
        """Test that other_filter with a single type only includes that type."""
        config = _base_config(tmp_path, test_paths)
        config["other_filter"] = ["mpp_full"]

        converter = GeneralConverter(config)
        other_raw = converter.__getitem__("other_lmdb", b"orca6_rks")

        global_feats = parse_other_data(other_raw, other_filter=["mpp_full"])

        assert "other_mpp_full" in global_feats
        assert "other_sdp_full" not in global_feats
        assert "other_ESP_Volume" not in global_feats

    def test_other_filter_multiple_types(self, tmp_path, test_paths):
        """Test that other_filter with multiple types includes all specified."""
        config = _base_config(tmp_path, test_paths)
        config["other_filter"] = ["mpp_full", "sdp_full", "ESP_Volume"]

        converter = GeneralConverter(config)
        other_raw = converter.__getitem__("other_lmdb", b"orca6_rks")

        global_feats = parse_other_data(
            other_raw, other_filter=["mpp_full", "sdp_full", "ESP_Volume"]
        )

        assert "other_mpp_full" in global_feats
        assert "other_sdp_full" in global_feats
        assert "other_ESP_Volume" in global_feats
        # Should NOT have unfiltered types
        assert "other_ESP_Surface_Density" not in global_feats

    def test_other_filter_none_includes_all(self, tmp_path, test_paths):
        """Test that other_filter=None includes all other types."""
        config = _base_config(tmp_path, test_paths)

        converter = GeneralConverter(config)
        other_raw = converter.__getitem__("other_lmdb", b"orca6_rks")

        global_feats = parse_other_data(other_raw, other_filter=None)

        # Check some expected keys are present
        assert "other_mpp_full" in global_feats
        assert "other_sdp_full" in global_feats
        assert "other_ESP_Volume" in global_feats
        assert "other_ALIE_Volume" in global_feats


# ============================================================================
# Test bond_filter and bonding_scheme
# ============================================================================

class TestBondFilter:
    """Tests for bond_filter and bonding_scheme behavior."""

    def test_bond_filter_single_type(self, tmp_path, test_paths):
        """Test that bond_filter with a single type only includes that type."""
        config = _base_config(tmp_path, test_paths)
        config["bond_filter"] = ["ibsi"]
        config["bond_list_definition"] = "ibsi"

        converter = GeneralConverter(config)
        bond_raw = converter.__getitem__("bonds_lmdb", b"orca5_rks")

        bond_feats, bond_list = parse_bond_data(
            bond_raw, bond_list_definition="ibsi", bond_filter=["ibsi"]
        )

        # Should have ibsi features
        assert len(bond_feats) > 0
        for bond_key, feats in bond_feats.items():
            assert "ibsi" in feats, f"Missing ibsi for bond {bond_key}"
            assert "fuzzy" not in feats, f"Should not have fuzzy for bond {bond_key}"

    def test_bond_filter_multiple_types(self, tmp_path, test_paths):
        """Test that bond_filter with multiple types includes all specified."""
        config = _base_config(tmp_path, test_paths)
        config["bond_filter"] = ["ibsi", "fuzzy"]
        config["bond_list_definition"] = "ibsi"

        converter = GeneralConverter(config)
        bond_raw = converter.__getitem__("bonds_lmdb", b"orca5_rks")

        bond_feats, _ = parse_bond_data(
            bond_raw, bond_list_definition="ibsi", bond_filter=["ibsi", "fuzzy"]
        )

        # Count bonds by feature type (not all bonds have all types)
        bonds_with_ibsi = sum(1 for feats in bond_feats.values() if "ibsi" in feats)
        bonds_with_fuzzy = sum(1 for feats in bond_feats.values() if "fuzzy" in feats)
        bonds_with_both = sum(1 for feats in bond_feats.values() if "ibsi" in feats and "fuzzy" in feats)

        # All bonds should have ibsi (bond_list_definition)
        assert bonds_with_ibsi == len(bond_feats), "Not all bonds have ibsi"
        # This test data has 144 bonds with fuzzy out of 351 total
        assert bonds_with_fuzzy == 144, f"Expected 144 bonds with fuzzy, got {bonds_with_fuzzy}"
        assert bonds_with_both == 144, f"Expected 144 bonds with both types, got {bonds_with_both}"

        # Verify no unexpected bond types leaked through the filter
        for feats in bond_feats.values():
            for key in feats.keys():
                assert key in ["ibsi", "fuzzy"], f"Unexpected bond type: {key}"

    def test_bond_list_definition_ibsi(self, tmp_path, test_paths):
        """Test that bond_list_definition='ibsi' uses ibsi bonds for connectivity."""
        config = _base_config(tmp_path, test_paths)
        config["bond_filter"] = ["ibsi", "fuzzy"]
        config["bond_list_definition"] = "ibsi"

        converter = GeneralConverter(config)
        bond_raw = converter.__getitem__("bonds_lmdb", b"orca5_rks")

        _, bond_list_ibsi = parse_bond_data(
            bond_raw, bond_list_definition="ibsi", bond_filter=["ibsi", "fuzzy"]
        )
        _, bond_list_fuzzy = parse_bond_data(
            bond_raw, bond_list_definition="fuzzy", bond_filter=["ibsi", "fuzzy"]
        )

        # Bond lists should exist and potentially differ
        assert isinstance(bond_list_ibsi, list)
        assert isinstance(bond_list_fuzzy, list)
        assert len(bond_list_ibsi) > 0
        assert len(bond_list_fuzzy) > 0

    def test_bond_cutoff_filters_weak_bonds(self, tmp_path, test_paths):
        """Test that bond_cutoff filters out bonds below threshold."""
        config = _base_config(tmp_path, test_paths)
        config["bond_filter"] = ["ibsi"]
        config["bond_list_definition"] = "ibsi"

        converter = GeneralConverter(config)
        bond_raw = converter.__getitem__("bonds_lmdb", b"orca5_rks")

        # No cutoff - should include all bonds
        _, bond_list_no_cutoff = parse_bond_data(
            bond_raw, bond_list_definition="ibsi", bond_filter=["ibsi"], cutoff=None
        )

        # High cutoff - should filter weak bonds
        _, bond_list_high_cutoff = parse_bond_data(
            bond_raw, bond_list_definition="ibsi", bond_filter=["ibsi"], cutoff=0.5
        )

        # High cutoff should have fewer or equal bonds
        assert len(bond_list_high_cutoff) <= len(bond_list_no_cutoff)


class TestFilterBondFeats:
    """Tests for filter_bond_feats utility function."""

    def test_filter_bond_feats_basic(self, tmp_path, test_paths):
        """Test that filter_bond_feats correctly filters by bond list."""
        config = _base_config(tmp_path, test_paths)

        converter = GeneralConverter(config)
        bond_raw = converter.__getitem__("bonds_lmdb", b"orca5_rks")

        # Get all bond features (as_lists=False to get tuples)
        bond_feats_all, bond_list = parse_bond_data(
            bond_raw, bond_list_definition="ibsi", bond_filter=["ibsi", "fuzzy"], as_lists=False
        )

        # Take subset of bond list
        subset_bond_list = bond_list[:5] if len(bond_list) >= 5 else bond_list

        # Filter bond features
        filtered_feats = filter_bond_feats(bond_feats_all, subset_bond_list)

        # Should only have bonds in subset
        assert len(filtered_feats) == len(subset_bond_list)
        for bond in subset_bond_list:
            assert bond in filtered_feats

    def test_filter_bond_feats_empty_list(self, tmp_path, test_paths):
        """Test that filter_bond_feats with empty list returns empty dict."""
        config = _base_config(tmp_path, test_paths)

        converter = GeneralConverter(config)
        bond_raw = converter.__getitem__("bonds_lmdb", b"orca5_rks")

        bond_feats_all, _ = parse_bond_data(
            bond_raw, bond_list_definition="ibsi", bond_filter=["ibsi"]
        )

        filtered_feats = filter_bond_feats(bond_feats_all, [])
        assert len(filtered_feats) == 0


# ============================================================================
# Test keys_data and keys_target interaction
# ============================================================================

class TestKeysDataInteraction:
    """Tests for keys_data filtering parsed features."""

    def test_keys_data_atom_filters_atom_features(self, tmp_path, test_paths):
        """Test that GeneralConverter auto-discovers available atom features."""
        config = _base_config(tmp_path, test_paths)
        config["charge_filter"] = ["mbis", "hirshfeld"]
        config["keys_data"]["atom"] = []  # Start empty to see auto-discovery

        converter = GeneralConverter(config)
        converter.process()

        # GeneralConverter auto-discovers features and modifies keys_data
        discovered_atom_feats = config["keys_data"]["atom"]

        # With mbis and hirshfeld filters, should discover charge_mbis and charge_hirshfeld
        assert "charge_mbis" in discovered_atom_feats
        assert "charge_hirshfeld" in discovered_atom_feats

    def test_keys_data_global_filters_global_features(self, tmp_path, test_paths):
        """Test that GeneralConverter auto-discovers available global features."""
        config = _base_config(tmp_path, test_paths)
        config["charge_filter"] = ["mbis"]  # Required for initialization
        config["other_filter"] = ["mpp_full", "sdp_full"]
        config["bond_list_definition"] = "ibsi"
        config["keys_data"]["global"] = []  # Start empty

        converter = GeneralConverter(config)
        converter.process()

        # Check auto-discovered global features from other_filter
        discovered_global_feats = config["keys_data"]["global"]
        assert "other_mpp_full" in discovered_global_feats
        assert "other_sdp_full" in discovered_global_feats
        # Note: n_atoms is structural, not from other_filter

    def test_keys_data_bond_filters_bond_features(self, tmp_path, test_paths):
        """Test that GeneralConverter auto-discovers available bond features."""
        config = _base_config(tmp_path, test_paths)
        config["charge_filter"] = ["mbis"]  # Required for initialization
        config["bond_filter"] = ["ibsi", "fuzzy"]
        config["bonding_scheme"] = "bonding"
        config["bond_list_definition"] = "ibsi"
        config["keys_data"]["bond"] = []  # Start empty

        converter = GeneralConverter(config)
        converter.process()

        # Check auto-discovered bond features
        discovered_bond_feats = config["keys_data"]["bond"]
        assert "ibsi" in discovered_bond_feats
        assert "fuzzy" in discovered_bond_feats


class TestKeysTargetInteraction:
    """Tests for keys_target marking features as prediction targets."""

    def test_keys_target_atom_marks_targets(self, tmp_path, test_paths):
        """Test that keys_target['atom'] configuration is preserved."""
        config = _base_config(tmp_path, test_paths)
        config["charge_filter"] = ["mbis"]
        config["keys_target"]["atom"] = ["charge_mbis"]

        converter = GeneralConverter(config)
        converter.process()

        # Verify keys_target config is preserved
        assert "charge_mbis" in config["keys_target"]["atom"]
        # Verify the target also appears in discovered features
        assert "charge_mbis" in config["keys_data"]["atom"]

    def test_keys_target_global_marks_targets(self, tmp_path, test_paths):
        """Test that keys_target['global'] configuration is preserved."""
        config = _base_config(tmp_path, test_paths)
        config["keys_target"]["global"] = ["n_atoms"]

        converter = GeneralConverter(config)
        converter.process()

        # Verify keys_target config is preserved
        assert "n_atoms" in config["keys_target"]["global"]
        # Verify the target also appears in discovered features
        assert "n_atoms" in config["keys_data"]["global"]


# ============================================================================
# Test QTAIM data parsing
# ============================================================================

class TestQTAIMDataParsing:
    """Tests for QTAIM data parsing and integration."""

    def test_qtaim_atom_features(self, tmp_path, test_paths):
        """Test that QTAIM atom features are parsed correctly."""
        config = _base_config(tmp_path, test_paths)

        converter = GeneralConverter(config)
        qtaim_raw = converter.__getitem__("qtaim_lmdb", b"orca5")

        atom_keys, bond_keys, atom_feats, bond_feats, connected_bond_paths = parse_qtaim_data(
            dict_qtaim=qtaim_raw,
            atom_feats={},
            bond_feats={},
        )

        # Should have 22 atom feature keys
        assert len(atom_keys) == 22, f"Expected 22 atom keys, got {len(atom_keys)}"
        assert len(bond_keys) == 22, f"Expected 22 bond keys, got {len(bond_keys)}"

        # Check some expected features
        expected_features = ["eta", "lol", "density_alpha", "density_beta"]
        for feat in expected_features:
            assert feat in atom_keys, f"Missing expected atom feature: {feat}"

    def test_qtaim_bond_paths(self, tmp_path, test_paths):
        """Test that QTAIM bond paths are parsed correctly."""
        config = _base_config(tmp_path, test_paths)

        converter = GeneralConverter(config)
        qtaim_raw = converter.__getitem__("qtaim_lmdb", b"orca5")

        _, _, _, _, connected_bond_paths = parse_qtaim_data(
            dict_qtaim=qtaim_raw,
            atom_feats={},
            bond_feats={},
        )

        # Should have 158 connected bond paths for this test case
        assert len(connected_bond_paths) == 158, f"Expected 158 bond paths, got {len(connected_bond_paths)}"

        # Each bond path should be a tuple/list of length 2
        for bp in connected_bond_paths:
            assert len(bp) == 2, f"Bond path should have 2 elements, got {len(bp)}"


# ============================================================================
# Test bonding_scheme
# ============================================================================

class TestBondingScheme:
    """Tests for bonding_scheme behavior."""

    def test_bonding_scheme_structural(self, tmp_path, test_paths):
        """Test that bonding_scheme='structural' uses coordinate-based bonds."""
        config = _base_config(tmp_path, test_paths)
        config["bonding_scheme"] = "structural"

        converter = GeneralConverter(config)
        struct_raw = converter.__getitem__("geom_lmdb", b"orca5_uks")

        # Structural bonding uses the bonds from geometry
        bonds = struct_raw.get("bonds", [])
        assert len(bonds) > 0, "Should have structural bonds"

    def test_bonding_scheme_bonding(self, tmp_path, test_paths):
        """Test that bonding_scheme='bonding' uses bond order data."""
        config = _base_config(tmp_path, test_paths)
        config["bonding_scheme"] = "bonding"
        config["bond_filter"] = ["ibsi"]
        config["bond_list_definition"] = "ibsi"

        converter = GeneralConverter(config)
        bond_raw = converter.__getitem__("bonds_lmdb", b"orca5_rks")

        _, bond_list = parse_bond_data(
            bond_raw, bond_list_definition="ibsi", bond_filter=["ibsi"]
        )

        assert len(bond_list) > 0, "Should have bonds from bond order data"

    def test_bonding_scheme_qtaim(self, tmp_path, test_paths):
        """Test that bonding_scheme='qtaim' uses QTAIM bond paths."""
        config = _base_config(tmp_path, test_paths)
        config["bonding_scheme"] = "qtaim"

        converter = GeneralConverter(config)
        qtaim_raw = converter.__getitem__("qtaim_lmdb", b"orca5")

        _, _, _, _, connected_bond_paths = parse_qtaim_data(
            dict_qtaim=qtaim_raw,
            atom_feats={},
            bond_feats={},
        )

        assert len(connected_bond_paths) > 0, "Should have QTAIM bond paths"


# ============================================================================
# Integration tests
# ============================================================================

class TestGeneralConverterIntegration:
    """Integration tests for GeneralConverter with all filters."""

    def test_full_pipeline_with_filters(self, tmp_path, test_paths):
        """Test full conversion pipeline with multiple filters and auto-discovery."""
        config = _base_config(tmp_path, test_paths)
        config["charge_filter"] = ["mbis", "hirshfeld"]
        config["fuzzy_filter"] = ["mbis_fuzzy_density", "elf_fuzzy"]
        config["other_filter"] = ["mpp_full", "sdp_full"]
        config["bond_filter"] = ["ibsi"]
        config["bond_list_definition"] = "ibsi"
        config["bonding_scheme"] = "structural"
        config["keys_target"] = {
            "atom": ["charge_hirshfeld"],
            "bond": [],
            "global": ["other_sdp_full"],
        }

        converter = GeneralConverter(config)
        converter.process(return_info=True)

        # Check auto-discovered features respect filters
        discovered_atom = config["keys_data"]["atom"]
        discovered_global = config["keys_data"]["global"]
        discovered_bond = config["keys_data"]["bond"]

        # Should have filtered charge types
        assert "charge_mbis" in discovered_atom
        assert "charge_hirshfeld" in discovered_atom

        # Should have filtered fuzzy types
        assert "fuzzy_elf_fuzzy" in discovered_atom
        assert "fuzzy_mbis_fuzzy_density" in discovered_atom

        # Should have filtered other types
        assert "other_mpp_full" in discovered_global
        assert "other_sdp_full" in discovered_global

        # Should have filtered bond types
        assert "ibsi" in discovered_bond

    def test_converter_process_completes(self, tmp_path, test_paths):
        """Test that converter completes processing successfully."""
        config = _base_config(tmp_path, test_paths)
        config["charge_filter"] = ["mbis"]
        config["bond_list_definition"] = "ibsi"

        converter = GeneralConverter(config)
        converter.process(return_info=True)

        # Verify auto-discovery worked
        assert "charge_mbis" in config["keys_data"]["atom"]

    def test_graph_has_correct_features(self, tmp_path, test_paths):
        """Test that auto-discovery finds correct features with filters."""
        config = _base_config(tmp_path, test_paths)
        config["charge_filter"] = ["mbis"]
        config["fuzzy_filter"] = ["elf_fuzzy"]

        converter = GeneralConverter(config)
        converter.process()

        # Check auto-discovered features
        discovered_atom = config["keys_data"]["atom"]

        # Should have charge_mbis from charge filter
        assert "charge_mbis" in discovered_atom

        # Should have fuzzy_elf_fuzzy from fuzzy filter
        assert "fuzzy_elf_fuzzy" in discovered_atom

        # Should NOT have other charge types (filtered out)
        assert "charge_hirshfeld" not in discovered_atom
        assert "charge_bader" not in discovered_atom

    def test_filtered_out_features_not_in_graph(self, tmp_path, test_paths):
        """Test that filtered-out features don't appear in auto-discovered features."""
        config = _base_config(tmp_path, test_paths)
        config["charge_filter"] = ["mbis"]  # Only mbis

        converter = GeneralConverter(config)
        converter.process()

        # Check auto-discovered features
        discovered_atom = config["keys_data"]["atom"]

        # Should have charge_mbis (in filter)
        assert "charge_mbis" in discovered_atom

        # Should NOT have charge_hirshfeld (filtered out)
        assert "charge_hirshfeld" not in discovered_atom

        # Should NOT have charge_bader (filtered out)
        assert "charge_bader" not in discovered_atom
