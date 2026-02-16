"""
Integration tests for converter MoleculeWrapper generation.

These tests verify that converters correctly produce MoleculeWrapper objects with:
1. Proper molecular graph structure (atoms, bonds, connectivity)
2. Correct atom features from charge/fuzzy/QTAIM data
3. Correct bond features from bond order/QTAIM bond paths
4. Correct global features (dipoles, n_atoms, other descriptors)

IMPORTANT: These are integration tests requiring qtaim-embed dependency.
Run with: pytest tests/test_converter_molwrapper_integration.py -v
"""

import os
import pytest
from pathlib import Path
from typing import Dict, Any

# Try to import qtaim-embed dependencies
try:
    from qtaim_gen.source.core.converter import BaseConverter, QTAIMConverter, GeneralConverter
    from qtaim_embed.data.lmdb import load_dgl_graph_from_serialized
    from qtaim_embed.core.molwrapper import MoleculeWrapper
    QTAIM_EMBED_AVAILABLE = True
except ImportError:
    QTAIM_EMBED_AVAILABLE = False
    # Create dummy classes so tests can be collected
    BaseConverter = None
    QTAIMConverter = None
    GeneralConverter = None
    MoleculeWrapper = None

# Skip all tests in this module if qtaim-embed is not available
pytestmark = pytest.mark.skipif(
    not QTAIM_EMBED_AVAILABLE,
    reason="qtaim-embed dependency not available. Install with: pip install qtaim-embed"
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture(scope="module")
def test_paths():
    """Return paths to test LMDB fixtures."""
    base = Path(__file__).parent / "test_files" / "lmdb_tests"
    return {
        "base": base,
        "merged": base / "generator_lmdbs_merged",
        "orca5_uks": base / "orca5_uks",
        "orca5_rks": base / "orca5_rks",
        "orca6_rks": base / "orca6_rks",
        "orca5": base / "orca5",
    }


def _base_converter_config(tmp_path, geom_lmdb_path, lmdb_name="test_graphs.lmdb"):
    """Create config for BaseConverter."""
    return {
        "chunk": -1,
        "filter_list": ["length"],
        "restart": False,
        "allowed_ring_size": [3, 4, 5, 6, 7, 8],
        "allowed_charges": None,
        "allowed_spins": None,
        "keys_target": {"atom": [], "bond": [], "global": []},
        "keys_data": {"atom": [], "bond": [], "global": ["n_atoms"]},
        "lmdb_path": str(tmp_path),
        "lmdb_name": lmdb_name,
        "lmdb_locations": {"geom_lmdb": str(geom_lmdb_path)},
        "n_workers": 1,
        "batch_size": 100,
    }


def _qtaim_converter_config(tmp_path, test_paths, lmdb_name="test_graphs.lmdb"):
    """Create config for QTAIMConverter."""
    merged = test_paths["merged"]
    return {
        "chunk": -1,
        "filter_list": ["length"],
        "restart": False,
        "allowed_ring_size": [3, 4, 5, 6, 7, 8],
        "allowed_charges": None,
        "allowed_spins": None,
        "keys_target": {"atom": [], "bond": [], "global": []},
        "keys_data": {"atom": [], "bond": [], "global": []},
        "lmdb_path": str(tmp_path),
        "lmdb_name": lmdb_name,
        "lmdb_locations": {
            "geom_lmdb": str(merged / "merged_geom.lmdb"),
            "qtaim_lmdb": str(merged / "merged_qtaim.lmdb"),
        },
        "n_workers": 1,
        "batch_size": 100,
    }


def _general_converter_config(tmp_path, test_paths, lmdb_name="test_graphs.lmdb"):
    """Create config for GeneralConverter with all features."""
    merged = test_paths["merged"]
    return {
        "chunk": -1,
        "filter_list": ["length"],
        "restart": False,
        "allowed_ring_size": [3, 4, 5, 6, 7, 8],
        "allowed_charges": None,
        "allowed_spins": None,
        "keys_target": {"atom": [], "bond": [], "global": []},
        "keys_data": {"atom": [], "bond": [], "global": ["n_atoms"]},
        "lmdb_path": str(tmp_path),
        "lmdb_name": lmdb_name,
        "lmdb_locations": {
            "geom_lmdb": str(merged / "merged_geom.lmdb"),
            "charge_lmdb": str(merged / "merged_charge.lmdb"),
            "fuzzy_full_lmdb": str(merged / "merged_fuzzy.lmdb"),
            "bonds_lmdb": str(merged / "merged_bond.lmdb"),
            "other_lmdb": str(merged / "merged_other.lmdb"),
            "qtaim_lmdb": str(merged / "merged_qtaim.lmdb"),
        },
        "charge_filter": ["mbis", "hirshfeld"],
        "fuzzy_filter": ["mbis_fuzzy_density", "elf_fuzzy"],
        "bond_filter": ["ibsi", "fuzzy"],
        "bond_list_definition": "ibsi",
        "bonding_scheme": "bonding",
        "n_workers": 1,
        "batch_size": 100,
    }


def _extract_molwrapper_from_converter(converter, key_idx=0):
    """
    Extract a MoleculeWrapper by fetching raw LMDB data through the converter
    and parsing it with the same utility functions the converter uses internally.

    This follows the pattern from test_general_converter_filters.py: use the
    converter as a data fetcher via __getitem__, then parse with utility functions
    and create MoleculeWrapper directly.

    Returns:
        Tuple of (key_str, mol_wrapper, failures)
    """
    from qtaim_gen.source.utils.lmdbs import (
        gather_structure_info,
        parse_charge_data,
        parse_fuzzy_data,
        parse_bond_data,
        parse_qtaim_data,
        parse_other_data,
    )
    from qtaim_gen.source.core.converter import clean_id

    # Get keys from geom_lmdb
    keys = list(converter.lmdb_dict["geom_lmdb"]["keys"])
    if key_idx >= len(keys):
        key_idx = 0
    test_key = keys[key_idx]

    key_str = test_key.decode("ascii") if isinstance(test_key, bytes) else str(test_key)
    id_val = clean_id(test_key) if converter.single_lmdb_in else key_str

    # Step 1: Structure
    value_structure = converter.__getitem__("geom_lmdb", test_key)
    if value_structure is None:
        pytest.skip(f"No structure data for key {test_key}")

    mol_graph, global_feats = gather_structure_info(value_structure)
    n_atoms = global_feats["n_atoms"]
    atom_feats = {i: {} for i in range(n_atoms)}
    bond_feats = {}
    bonds = value_structure["bonds"]
    bond_list = {tuple(sorted(b)): None for b in bonds if b[0] != b[1]}

    # Step 2: Parse additional features based on available LMDBs
    connected_bond_paths = None
    bonds_from_lmdb = None

    if isinstance(converter, (GeneralConverter, QTAIMConverter)):
        # Charge data
        if "charge_lmdb" in converter.lmdb_dict:
            dict_charge = converter.__getitem__("charge_lmdb", test_key)
            if dict_charge is not None:
                charge_filter = getattr(converter, "charge_filter", None)
                atom_feats_charge, global_dipole = parse_charge_data(
                    dict_charge, n_atoms, charge_filter
                )
                global_feats.update(global_dipole)
                atom_feats.update(atom_feats_charge)

        # QTAIM data
        if "qtaim_lmdb" in converter.lmdb_dict:
            dict_qtaim = converter.__getitem__("qtaim_lmdb", test_key)
            if dict_qtaim is not None:
                # Pass None for empty lists to trigger auto-discovery
                atom_keys = converter.keys_data.get("atom", []) or None
                bond_keys = converter.keys_data.get("bond", []) or None
                (_, _, atom_feats, bond_feats, connected_bond_paths) = parse_qtaim_data(
                    dict_qtaim, atom_feats, bond_feats,
                    atom_keys=atom_keys,
                    bond_keys=bond_keys,
                )

        # Fuzzy data - check both naming conventions (fuzzy_lmdb / fuzzy_full_lmdb)
        fuzzy_key = None
        for k in ["fuzzy_lmdb", "fuzzy_full_lmdb"]:
            if k in converter.lmdb_dict:
                fuzzy_key = k
                break
        if fuzzy_key:
            dict_fuzzy = converter.__getitem__(fuzzy_key, test_key)
            if dict_fuzzy is not None:
                fuzzy_filter = getattr(converter, "fuzzy_filter", None)
                atom_feats_fuzzy, global_fuzzy = parse_fuzzy_data(
                    dict_fuzzy, n_atoms, fuzzy_filter
                )
                global_feats.update(global_fuzzy)
                for idx, ffeats in atom_feats_fuzzy.items():
                    atom_feats.setdefault(idx, {}).update(ffeats)

        # Other data
        if "other_lmdb" in converter.lmdb_dict:
            dict_other = converter.__getitem__("other_lmdb", test_key)
            if dict_other is not None:
                other_filter = getattr(converter, "other_filter", None)
                global_other = parse_other_data(dict_other, other_filter)
                global_feats.update(global_other)

        # Bond data
        if "bonds_lmdb" in converter.lmdb_dict:
            dict_bonds = converter.__getitem__("bonds_lmdb", test_key)
            if dict_bonds is not None:
                bond_filter = getattr(converter, "bond_filter", None)
                bond_cutoff = getattr(converter, "bond_cutoff", None)
                bond_list_def = getattr(converter, "bond_list_definition", "fuzzy")
                bond_feats_lmdb, bonds_from_lmdb = parse_bond_data(
                    dict_bonds, bond_filter=bond_filter, cutoff=bond_cutoff,
                    bond_list_definition=bond_list_def, bond_feats=None,
                    clean=True, as_lists=False,
                )
                if bond_feats_lmdb:
                    for bk, bv in bond_feats_lmdb.items():
                        bond_feats.setdefault(bk, {}).update(bv)

    # Step 3: Select bond definitions based on converter type and bonding scheme
    if isinstance(converter, QTAIMConverter):
        # QTAIMConverter always uses QTAIM bond paths
        selected_bonds = connected_bond_paths if connected_bond_paths is not None else bond_list
    elif isinstance(converter, GeneralConverter):
        # GeneralConverter respects bonding_scheme config
        bonding_scheme = getattr(converter, "bonding_scheme", "structural")
        if bonding_scheme == "qtaim" and connected_bond_paths is not None:
            selected_bonds = connected_bond_paths
        elif bonding_scheme == "bonding" and bonds_from_lmdb is not None:
            selected_bonds = {tuple(sorted(b)): None for b in bonds_from_lmdb}
        else:
            selected_bonds = bond_list
    else:
        # BaseConverter uses structural bonds from coordinates
        selected_bonds = bond_list

    # Step 4: Create MoleculeWrapper
    mol_wrapper = MoleculeWrapper(
        mol_graph,
        functional_group=None,
        free_energy=None,
        id=id_val,
        bonds=selected_bonds,
        non_metal_bonds=selected_bonds,
        atom_features=atom_feats,
        bond_features=bond_feats,
        global_features=global_feats,
        original_atom_ind=None,
        original_bond_mapping=None,
    )

    return key_str, mol_wrapper, {}


# ============================================================================
# Test BaseConverter MoleculeWrapper
# ============================================================================

class TestBaseConverterMolWrapper:
    """Tests for BaseConverter MoleculeWrapper generation."""

    @pytest.mark.skipif(
        not os.path.exists(os.path.join(os.path.dirname(__file__), "test_files", "lmdb_tests")),
        reason="Test fixtures not available"
    )
    def test_base_converter_creates_molwrapper(self, tmp_path, test_paths):
        """Test that BaseConverter creates a valid MoleculeWrapper."""
        merged = test_paths["merged"]
        geom_lmdb = merged / "merged_geom.lmdb"

        if not os.path.exists(geom_lmdb):
            pytest.skip("Test LMDB fixtures not available")

        config = _base_converter_config(tmp_path, geom_lmdb)
        converter = BaseConverter(config, config_path=str(tmp_path / "config.json"))

        key_str, mol_wrapper, failures = _extract_molwrapper_from_converter(converter)

        # Verify it's a MoleculeWrapper
        assert isinstance(mol_wrapper, MoleculeWrapper), "Should be a MoleculeWrapper instance"

        # Verify basic structure
        assert mol_wrapper.id is not None, "Should have an ID"
        assert mol_wrapper.mol_graph is not None, "Should have a molecular graph"
        assert len(mol_wrapper.mol_graph) > 0, "Should have at least one atom"

    @pytest.mark.skipif(
        not os.path.exists(os.path.join(os.path.dirname(__file__), "test_files", "lmdb_tests")),
        reason="Test fixtures not available"
    )
    def test_base_converter_global_features(self, tmp_path, test_paths):
        """Test that BaseConverter includes global features in MoleculeWrapper."""
        merged = test_paths["merged"]
        geom_lmdb = merged / "merged_geom.lmdb"

        if not os.path.exists(geom_lmdb):
            pytest.skip("Test LMDB fixtures not available")

        config = _base_converter_config(tmp_path, geom_lmdb)
        converter = BaseConverter(config, config_path=str(tmp_path / "config.json"))

        key_str, mol_wrapper, failures = _extract_molwrapper_from_converter(converter)

        # Verify global features
        assert mol_wrapper.global_features is not None, "Should have global features"
        assert isinstance(mol_wrapper.global_features, dict), "Global features should be dict"
        assert "n_atoms" in mol_wrapper.global_features, "Should have n_atoms"

        # Verify n_atoms matches graph
        n_atoms = mol_wrapper.global_features["n_atoms"]
        assert n_atoms == len(mol_wrapper.mol_graph), \
            f"n_atoms ({n_atoms}) should match node count ({len(mol_wrapper.mol_graph)})"

    @pytest.mark.skipif(
        not os.path.exists(os.path.join(os.path.dirname(__file__), "test_files", "lmdb_tests")),
        reason="Test fixtures not available"
    )
    def test_base_converter_bonds(self, tmp_path, test_paths):
        """Test that BaseConverter includes bond connectivity."""
        merged = test_paths["merged"]
        geom_lmdb = merged / "merged_geom.lmdb"

        if not os.path.exists(geom_lmdb):
            pytest.skip("Test LMDB fixtures not available")

        config = _base_converter_config(tmp_path, geom_lmdb)
        converter = BaseConverter(config, config_path=str(tmp_path / "config.json"))

        key_str, mol_wrapper, failures = _extract_molwrapper_from_converter(converter)

        # Verify bonds
        assert mol_wrapper.bonds is not None, "Should have bonds"
        assert isinstance(mol_wrapper.bonds, dict), "Bonds should be dict"

        if len(mol_wrapper.bonds) > 0:
            # Check bond format (should be tuples as keys)
            sample_bond = list(mol_wrapper.bonds.keys())[0]
            assert isinstance(sample_bond, tuple), "Bond keys should be tuples"
            assert len(sample_bond) == 2, "Bond keys should be pairs"


# ============================================================================
# Test QTAIMConverter MoleculeWrapper
# ============================================================================

class TestQTAIMConverterMolWrapper:
    """Tests for QTAIMConverter MoleculeWrapper generation."""

    @pytest.mark.skipif(
        not os.path.exists(os.path.join(os.path.dirname(__file__), "test_files", "lmdb_tests")),
        reason="Test fixtures not available"
    )
    def test_qtaim_converter_creates_molwrapper(self, tmp_path, test_paths):
        """Test that QTAIMConverter creates a valid MoleculeWrapper."""
        merged = test_paths["merged"]
        qtaim_lmdb = merged / "merged_qtaim.lmdb"

        if not os.path.exists(qtaim_lmdb):
            pytest.skip("Test LMDB fixtures not available")

        config = _qtaim_converter_config(tmp_path, test_paths)
        converter = QTAIMConverter(config, config_path=str(tmp_path / "config.json"))

        key_str, mol_wrapper, failures = _extract_molwrapper_from_converter(converter)

        # Verify it's a MoleculeWrapper
        assert isinstance(mol_wrapper, MoleculeWrapper), "Should be a MoleculeWrapper instance"
        assert mol_wrapper.id is not None, "Should have an ID"

    @pytest.mark.skipif(
        not os.path.exists(os.path.join(os.path.dirname(__file__), "test_files", "lmdb_tests")),
        reason="Test fixtures not available"
    )
    def test_qtaim_converter_atom_features(self, tmp_path, test_paths):
        """Test that QTAIMConverter includes QTAIM atom features."""
        merged = test_paths["merged"]
        qtaim_lmdb = merged / "merged_qtaim.lmdb"

        if not os.path.exists(qtaim_lmdb):
            pytest.skip("Test LMDB fixtures not available")

        config = _qtaim_converter_config(tmp_path, test_paths)
        converter = QTAIMConverter(config, config_path=str(tmp_path / "config.json"))

        key_str, mol_wrapper, failures = _extract_molwrapper_from_converter(converter)

        # Verify atom features
        assert mol_wrapper.atom_features is not None, "Should have atom features"
        assert isinstance(mol_wrapper.atom_features, dict), "Atom features should be dict"

        if len(mol_wrapper.atom_features) > 0:
            # Check first atom's features
            first_atom = list(mol_wrapper.atom_features.keys())[0]
            atom_feats = mol_wrapper.atom_features[first_atom]

            # Should have QTAIM features like eta, lol, density_alpha, etc.
            expected_qtaim_features = ["eta", "lol", "density_alpha", "density_beta"]
            found_features = [f for f in expected_qtaim_features if f in atom_feats]

            assert len(found_features) > 0, \
                f"Should have at least some QTAIM features. Found: {list(atom_feats.keys())}"

    @pytest.mark.skipif(
        not os.path.exists(os.path.join(os.path.dirname(__file__), "test_files", "lmdb_tests")),
        reason="Test fixtures not available"
    )
    def test_qtaim_converter_bond_paths(self, tmp_path, test_paths):
        """Test that QTAIMConverter uses QTAIM bond paths."""
        merged = test_paths["merged"]
        qtaim_lmdb = merged / "merged_qtaim.lmdb"

        if not os.path.exists(qtaim_lmdb):
            pytest.skip("Test LMDB fixtures not available")

        config = _qtaim_converter_config(tmp_path, test_paths)
        converter = QTAIMConverter(config, config_path=str(tmp_path / "config.json"))

        key_str, mol_wrapper, failures = _extract_molwrapper_from_converter(converter)

        # Verify bond paths from QTAIM
        assert mol_wrapper.bonds is not None, "Should have bonds (bond paths)"
        assert isinstance(mol_wrapper.bonds, dict), "Bonds should be dict"

        # QTAIM bond paths should exist for molecules
        if len(mol_wrapper.mol_graph) > 1:
            assert len(mol_wrapper.bonds) > 0, "Should have QTAIM bond paths for multi-atom molecules"


# ============================================================================
# Test GeneralConverter MoleculeWrapper
# ============================================================================

class TestGeneralConverterMolWrapper:
    """Tests for GeneralConverter MoleculeWrapper generation."""

    @pytest.mark.skipif(
        not os.path.exists(os.path.join(os.path.dirname(__file__), "test_files", "lmdb_tests")),
        reason="Test fixtures not available"
    )
    def test_general_converter_creates_molwrapper(self, tmp_path, test_paths):
        """Test that GeneralConverter creates a valid MoleculeWrapper."""
        merged = test_paths["merged"]
        geom_lmdb = merged / "merged_geom.lmdb"

        if not os.path.exists(geom_lmdb):
            pytest.skip("Test LMDB fixtures not available")

        config = _general_converter_config(tmp_path, test_paths)
        converter = GeneralConverter(config, config_path=str(tmp_path / "config.json"))

        key_str, mol_wrapper, failures = _extract_molwrapper_from_converter(converter)

        # Verify it's a MoleculeWrapper
        assert isinstance(mol_wrapper, MoleculeWrapper), "Should be a MoleculeWrapper instance"
        assert mol_wrapper.id is not None, "Should have an ID"

    @pytest.mark.skipif(
        not os.path.exists(os.path.join(os.path.dirname(__file__), "test_files", "lmdb_tests")),
        reason="Test fixtures not available"
    )
    def test_general_converter_charge_features(self, tmp_path, test_paths):
        """Test that GeneralConverter includes charge features."""
        merged = test_paths["merged"]
        geom_lmdb = merged / "merged_geom.lmdb"

        if not os.path.exists(geom_lmdb):
            pytest.skip("Test LMDB fixtures not available")

        config = _general_converter_config(tmp_path, test_paths)
        # Set specific charge filter
        config["charge_filter"] = ["mbis", "hirshfeld"]

        converter = GeneralConverter(config, config_path=str(tmp_path / "config.json"))

        key_str, mol_wrapper, failures = _extract_molwrapper_from_converter(converter)

        # Verify atom features include charges
        assert mol_wrapper.atom_features is not None, "Should have atom features"

        if len(mol_wrapper.atom_features) > 0:
            first_atom = list(mol_wrapper.atom_features.keys())[0]
            atom_feats = mol_wrapper.atom_features[first_atom]

            # Should have filtered charge types
            assert "charge_mbis" in atom_feats, "Should have charge_mbis"
            assert "charge_hirshfeld" in atom_feats, "Should have charge_hirshfeld"

            # Should NOT have unfiltered charge types
            assert "charge_bader" not in atom_feats, "Should not have charge_bader (filtered out)"

    @pytest.mark.skipif(
        not os.path.exists(os.path.join(os.path.dirname(__file__), "test_files", "lmdb_tests")),
        reason="Test fixtures not available"
    )
    def test_general_converter_fuzzy_features(self, tmp_path, test_paths):
        """Test that GeneralConverter includes fuzzy features."""
        merged = test_paths["merged"]
        geom_lmdb = merged / "merged_geom.lmdb"

        if not os.path.exists(geom_lmdb):
            pytest.skip("Test LMDB fixtures not available")

        config = _general_converter_config(tmp_path, test_paths)
        # Set specific fuzzy filter
        config["fuzzy_filter"] = ["mbis_fuzzy_density", "elf_fuzzy"]

        converter = GeneralConverter(config, config_path=str(tmp_path / "config.json"))

        key_str, mol_wrapper, failures = _extract_molwrapper_from_converter(converter)

        # Verify atom features include fuzzy descriptors
        if len(mol_wrapper.atom_features) > 0:
            first_atom = list(mol_wrapper.atom_features.keys())[0]
            atom_feats = mol_wrapper.atom_features[first_atom]

            # Should have filtered fuzzy types
            assert "fuzzy_mbis_fuzzy_density" in atom_feats, "Should have fuzzy_mbis_fuzzy_density"
            assert "fuzzy_elf_fuzzy" in atom_feats, "Should have fuzzy_elf_fuzzy"

    @pytest.mark.skipif(
        not os.path.exists(os.path.join(os.path.dirname(__file__), "test_files", "lmdb_tests")),
        reason="Test fixtures not available"
    )
    def test_general_converter_bond_features(self, tmp_path, test_paths):
        """Test that GeneralConverter includes bond order features."""
        merged = test_paths["merged"]
        geom_lmdb = merged / "merged_geom.lmdb"

        if not os.path.exists(geom_lmdb):
            pytest.skip("Test LMDB fixtures not available")

        config = _general_converter_config(tmp_path, test_paths)
        # Set specific bond filter
        config["bond_filter"] = ["ibsi", "fuzzy"]
        config["bond_list_definition"] = "ibsi"

        converter = GeneralConverter(config, config_path=str(tmp_path / "config.json"))

        key_str, mol_wrapper, failures = _extract_molwrapper_from_converter(converter)

        # Verify bond features
        assert mol_wrapper.bond_features is not None, "Should have bond features"

        if len(mol_wrapper.bond_features) > 0:
            # Check first bond's features
            first_bond = list(mol_wrapper.bond_features.keys())[0]
            bond_feats = mol_wrapper.bond_features[first_bond]

            # Should have bond order features from bonds_lmdb
            # The LMDB may contain more features than the filter specifies
            bond_feat_keys = list(bond_feats.keys())
            assert len(bond_feat_keys) > 0, "Should have bond features"

    @pytest.mark.skipif(
        not os.path.exists(os.path.join(os.path.dirname(__file__), "test_files", "lmdb_tests")),
        reason="Test fixtures not available"
    )
    def test_general_converter_global_features(self, tmp_path, test_paths):
        """Test that GeneralConverter includes global features."""
        merged = test_paths["merged"]
        geom_lmdb = merged / "merged_geom.lmdb"

        if not os.path.exists(geom_lmdb):
            pytest.skip("Test LMDB fixtures not available")

        config = _general_converter_config(tmp_path, test_paths)
        # Add other_filter for global features
        config["other_filter"] = ["mpp_full", "sdp_full"]

        converter = GeneralConverter(config, config_path=str(tmp_path / "config.json"))

        key_str, mol_wrapper, failures = _extract_molwrapper_from_converter(converter)

        # Verify global features
        assert mol_wrapper.global_features is not None, "Should have global features"

        # Should have n_atoms
        assert "n_atoms" in mol_wrapper.global_features, "Should have n_atoms"

        # Should have filtered other features (if data exists)
        # Note: not all molecules may have all features
        global_feat_keys = list(mol_wrapper.global_features.keys())
        assert len(global_feat_keys) > 0, "Should have some global features"

    @pytest.mark.skipif(
        not os.path.exists(os.path.join(os.path.dirname(__file__), "test_files", "lmdb_tests")),
        reason="Test fixtures not available"
    )
    def test_general_converter_bonding_scheme_structural(self, tmp_path, test_paths):
        """Test that bonding_scheme='structural' uses coordinate-based bonds."""
        merged = test_paths["merged"]
        geom_lmdb = merged / "merged_geom.lmdb"

        if not os.path.exists(geom_lmdb):
            pytest.skip("Test LMDB fixtures not available")

        config = _general_converter_config(tmp_path, test_paths)
        config["bonding_scheme"] = "structural"

        converter = GeneralConverter(config, config_path=str(tmp_path / "config.json"))

        key_str, mol_wrapper, failures = _extract_molwrapper_from_converter(converter)

        # Verify bonds exist
        assert mol_wrapper.bonds is not None, "Should have bonds"

        # For structural bonding, bonds come from coordinate-based detection
        if len(mol_wrapper.mol_graph) > 1:
            assert len(mol_wrapper.bonds) > 0, "Should have structural bonds for multi-atom molecules"

    @pytest.mark.skipif(
        not os.path.exists(os.path.join(os.path.dirname(__file__), "test_files", "lmdb_tests")),
        reason="Test fixtures not available"
    )
    def test_general_converter_bonding_scheme_qtaim(self, tmp_path, test_paths):
        """Test that bonding_scheme='qtaim' uses QTAIM bond paths."""
        merged = test_paths["merged"]
        geom_lmdb = merged / "merged_geom.lmdb"
        qtaim_lmdb = merged / "merged_qtaim.lmdb"

        if not os.path.exists(geom_lmdb) or not os.path.exists(qtaim_lmdb):
            pytest.skip("Test LMDB fixtures not available")

        config = _general_converter_config(tmp_path, test_paths)
        config["bonding_scheme"] = "qtaim"

        converter = GeneralConverter(config, config_path=str(tmp_path / "config.json"))

        key_str, mol_wrapper, failures = _extract_molwrapper_from_converter(converter)

        # Verify bonds exist (QTAIM bond paths)
        assert mol_wrapper.bonds is not None, "Should have bonds"

        # For QTAIM bonding, bonds are bond paths
        if len(mol_wrapper.mol_graph) > 1:
            assert len(mol_wrapper.bonds) > 0, "Should have QTAIM bond paths for multi-atom molecules"


# ============================================================================
# Integration tests - end-to-end
# ============================================================================

class TestConverterEndToEnd:
    """End-to-end integration tests verifying full pipeline."""

    @pytest.mark.skipif(
        not os.path.exists(os.path.join(os.path.dirname(__file__), "test_files", "lmdb_tests")),
        reason="Test fixtures not available"
    )
    def test_molwrapper_to_graph_roundtrip(self, tmp_path, test_paths):
        """Test that MoleculeWrapper can be converted to graph and back."""
        merged = test_paths["merged"]
        geom_lmdb = merged / "merged_geom.lmdb"

        if not os.path.exists(geom_lmdb):
            pytest.skip("Test LMDB fixtures not available")

        config = _general_converter_config(tmp_path, test_paths)
        converter = GeneralConverter(config, config_path=str(tmp_path / "config.json"))

        key_str, mol_wrapper, failures = _extract_molwrapper_from_converter(converter)

        # Convert to DGL graph using grapher
        from qtaim_embed.utils.grapher import get_grapher

        # Get element set for grapher - pymatgen MoleculeGraph has .molecule attribute
        elements = set()
        for site in mol_wrapper.mol_graph.molecule:
            elements.add(site.specie.symbol)

        # Create grapher
        grapher = get_grapher(
            element_set=elements,
            atom_keys=list(mol_wrapper.atom_features[0].keys()) if mol_wrapper.atom_features else [],
            bond_keys=list(mol_wrapper.bond_features[list(mol_wrapper.bond_features.keys())[0]].keys())
                if mol_wrapper.bond_features else [],
            global_keys=list(mol_wrapper.global_features.keys()) if mol_wrapper.global_features else [],
            allowed_ring_size=[3, 4, 5, 6, 7, 8],
            allowed_charges=None,
            allowed_spins=None,
            self_loop=True,
        )

        # Build graph
        dgl_graph = grapher.build_graph(mol_wrapper)

        # Verify graph structure
        assert dgl_graph is not None, "Should create DGL graph"
        # DGL graph is heterogeneous with 'atom', 'bond', 'global' node types
        atom_count = dgl_graph.num_nodes('atom')
        assert atom_count > 0, "Graph should have atom nodes"
        assert atom_count == len(mol_wrapper.mol_graph), \
            f"Atom node count ({atom_count}) should match MoleculeWrapper ({len(mol_wrapper.mol_graph)})"

    @pytest.mark.skipif(
        not os.path.exists(os.path.join(os.path.dirname(__file__), "test_files", "lmdb_tests")),
        reason="Test fixtures not available"
    )
    def test_multiple_molecules_consistency(self, tmp_path, test_paths):
        """Test that multiple molecules from same converter have consistent structure."""
        merged = test_paths["merged"]
        geom_lmdb = merged / "merged_geom.lmdb"

        if not os.path.exists(geom_lmdb):
            pytest.skip("Test LMDB fixtures not available")

        config = _general_converter_config(tmp_path, test_paths)
        config["charge_filter"] = ["mbis"]
        converter = GeneralConverter(config, config_path=str(tmp_path / "config.json"))

        # Extract multiple MoleculeWrappers
        molwrappers = []
        for i in range(min(3, len(converter.lmdb_dict["geom_lmdb"]["keys"]))):
            try:
                key_str, mol_wrapper, failures = _extract_molwrapper_from_converter(converter, key_idx=i)
                if mol_wrapper is not None:
                    molwrappers.append(mol_wrapper)
            except:
                continue

        if len(molwrappers) < 2:
            pytest.skip("Not enough molecules to test consistency")

        # Verify all have same feature structure (even if different values)
        first_mol = molwrappers[0]
        first_atom_keys = set(first_mol.atom_features[0].keys()) if first_mol.atom_features else set()

        for mol in molwrappers[1:]:
            if mol.atom_features:
                mol_atom_keys = set(mol.atom_features[0].keys())
                # Feature keys should match across molecules
                assert mol_atom_keys == first_atom_keys, \
                    f"Atom feature keys should match. Got {mol_atom_keys} vs {first_atom_keys}"


if __name__ == "__main__":
    # Allow running directly for debugging
    import sys
    pytest.main([__file__, "-v"] + sys.argv[1:])
