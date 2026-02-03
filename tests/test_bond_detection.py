"""Tests for bond detection from coordinates (utils/io.py)."""

from pathlib import Path

import numpy as np
import pytest

from qtaim_gen.source.utils.io import (
    get_bonds_from_coords,
    parse_orca_inp_to_molecule_data,
    _PERIODIC_TABLE,
)

TEST_FILES = Path(__file__).parent / "test_files"


class TestPeriodicTable:
    """Tests for RDKit periodic table integration."""

    def test_periodic_table_exists(self):
        """Verify RDKit periodic table is initialized."""
        assert _PERIODIC_TABLE is not None

    def test_covalent_radii_common_elements(self):
        """Test covalent radii for common organic elements."""
        # Values should be positive and reasonable (0.2-2.5 Angstroms)
        for symbol in ["H", "C", "N", "O", "F", "S", "Cl", "Br"]:
            z = _PERIODIC_TABLE.GetAtomicNumber(symbol)
            r = _PERIODIC_TABLE.GetRcovalent(z)
            assert 0.2 < r < 2.5, f"Unexpected covalent radius for {symbol}: {r}"

    def test_covalent_radii_heavy_elements(self):
        """Test covalent radii for heavy elements (actinides)."""
        for symbol in ["U", "Th", "Pu"]:
            z = _PERIODIC_TABLE.GetAtomicNumber(symbol)
            r = _PERIODIC_TABLE.GetRcovalent(z)
            assert r > 0, f"Missing covalent radius for {symbol}"

    def test_atomic_number_lookup(self):
        """Test atomic number lookups."""
        assert _PERIODIC_TABLE.GetAtomicNumber("C") == 6
        assert _PERIODIC_TABLE.GetAtomicNumber("Fe") == 26
        assert _PERIODIC_TABLE.GetAtomicNumber("U") == 92


class TestGetBondsFromCoords:
    """Tests for get_bonds_from_coords function."""

    def test_methane_bonds(self):
        """Test bond detection for methane (CH4)."""
        species = ["C", "H", "H", "H", "H"]
        # Tetrahedral geometry with ~1.09 A C-H bonds
        coords = np.array([
            [0.0, 0.0, 0.0],
            [0.63, 0.63, 0.63],
            [-0.63, -0.63, 0.63],
            [-0.63, 0.63, -0.63],
            [0.63, -0.63, -0.63],
        ])

        bonds = get_bonds_from_coords(species, coords)

        # Should have 4 C-H bonds
        assert len(bonds) == 4, f"Expected 4 bonds, got {len(bonds)}"
        # All bonds should involve atom 0 (carbon)
        for bond in bonds:
            assert 0 in bond, f"Bond {bond} doesn't include carbon (index 0)"

    def test_ethane_bonds(self):
        """Test bond detection for ethane (C2H6)."""
        species = ["C", "C", "H", "H", "H", "H", "H", "H"]
        # Simplified ethane geometry
        coords = np.array([
            [0.0, 0.0, 0.0],      # C1
            [1.54, 0.0, 0.0],     # C2 (C-C bond ~1.54 A)
            [-0.5, 0.9, 0.0],     # H on C1
            [-0.5, -0.9, 0.0],    # H on C1
            [-0.5, 0.0, 0.9],     # H on C1
            [2.04, 0.9, 0.0],     # H on C2
            [2.04, -0.9, 0.0],    # H on C2
            [2.04, 0.0, 0.9],     # H on C2
        ])

        bonds = get_bonds_from_coords(species, coords)

        # Should have 7 bonds: 1 C-C + 6 C-H
        assert len(bonds) == 7, f"Expected 7 bonds, got {len(bonds)}"

        # Check C-C bond exists
        cc_bond = [0, 1] if [0, 1] in bonds else [1, 0]
        assert [0, 1] in bonds or [1, 0] in bonds, "C-C bond not detected"

    def test_water_bonds(self):
        """Test bond detection for water (H2O)."""
        species = ["O", "H", "H"]
        # Water geometry with ~0.96 A O-H bonds, 104.5 degree angle
        coords = np.array([
            [0.0, 0.0, 0.0],
            [0.76, 0.59, 0.0],
            [-0.76, 0.59, 0.0],
        ])

        bonds = get_bonds_from_coords(species, coords)

        # Should have 2 O-H bonds
        assert len(bonds) == 2, f"Expected 2 bonds, got {len(bonds)}"
        # Both bonds should involve oxygen (index 0)
        for bond in bonds:
            assert 0 in bond

    def test_no_bonds_distant_atoms(self):
        """Test that distant atoms have no bonds."""
        species = ["H", "H"]
        coords = np.array([
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],  # 10 A apart - too far for a bond
        ])

        bonds = get_bonds_from_coords(species, coords)

        assert len(bonds) == 0, "Should have no bonds for distant atoms"

    def test_h2_molecule(self):
        """Test H2 molecule detection."""
        species = ["H", "H"]
        coords = np.array([
            [0.0, 0.0, 0.0],
            [0.74, 0.0, 0.0],  # H-H bond ~0.74 A
        ])

        bonds = get_bonds_from_coords(species, coords)

        assert len(bonds) == 1, f"Expected 1 bond, got {len(bonds)}"
        assert bonds[0] == [0, 1]

    def test_covalent_factor_adjustment(self):
        """Test that covalent_factor affects bond detection."""
        species = ["C", "C"]
        coords = np.array([
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],  # 2.0 A apart
        ])

        # With default factor (1.3), this should be a bond
        # C covalent radius ~0.76 A, so threshold = 2 * 0.76 * 1.3 = 1.98 A
        # 2.0 A is slightly above threshold
        bonds_default = get_bonds_from_coords(species, coords, covalent_factor=1.3)

        # With larger factor, should detect bond
        bonds_large = get_bonds_from_coords(species, coords, covalent_factor=1.5)

        # With smaller factor, should not detect bond
        bonds_small = get_bonds_from_coords(species, coords, covalent_factor=1.0)

        assert len(bonds_large) == 1, "Should detect bond with larger covalent factor"
        assert len(bonds_small) == 0, "Should not detect bond with smaller covalent factor"

    def test_bond_indices_zero_indexed(self):
        """Verify bond indices are 0-indexed."""
        species = ["C", "H"]
        coords = np.array([
            [0.0, 0.0, 0.0],
            [1.09, 0.0, 0.0],
        ])

        bonds = get_bonds_from_coords(species, coords)

        assert bonds[0] == [0, 1], "Bond indices should be 0-indexed"

    def test_bond_list_format(self):
        """Verify bonds are returned as list of lists."""
        species = ["O", "H", "H"]
        coords = np.array([
            [0.0, 0.0, 0.0],
            [0.96, 0.0, 0.0],
            [-0.24, 0.93, 0.0],
        ])

        bonds = get_bonds_from_coords(species, coords)

        assert isinstance(bonds, list)
        for bond in bonds:
            assert isinstance(bond, list)
            assert len(bond) == 2

    def test_transition_metal_complex(self):
        """Test bond detection with transition metal."""
        # Simple Fe-CO fragment
        species = ["Fe", "C", "O"]
        coords = np.array([
            [0.0, 0.0, 0.0],
            [1.8, 0.0, 0.0],   # Fe-C bond
            [2.95, 0.0, 0.0],  # C-O bond
        ])

        bonds = get_bonds_from_coords(species, coords)

        # Should detect Fe-C and C-O bonds
        assert len(bonds) >= 1, "Should detect at least Fe-C bond"


class TestParseOrcaInpToMoleculeData:
    """Tests for parse_orca_inp_to_molecule_data function."""

    def test_parse_orca_inp(self):
        """Test parsing ORCA input file."""
        species, coords, charge, spin = parse_orca_inp_to_molecule_data(
            str(TEST_FILES / "orca" / "orca.inp")
        )

        # Check species
        assert isinstance(species, list)
        assert len(species) > 0
        assert all(isinstance(s, str) for s in species)

        # Check coords
        assert isinstance(coords, np.ndarray)
        assert coords.shape[1] == 3
        assert coords.shape[0] == len(species)

        # Check charge and spin (from file: *xyz 2 1)
        assert charge == 2
        assert spin == 1

    def test_parse_orca5_inp(self):
        """Test parsing ORCA 5 format input file."""
        species, coords, charge, spin = parse_orca_inp_to_molecule_data(
            str(TEST_FILES / "orca" / "orca5.inp")
        )

        assert len(species) == 118
        assert coords.shape == (118, 3)

    def test_species_coords_consistency(self):
        """Test that species and coords arrays have matching lengths."""
        species, coords, charge, spin = parse_orca_inp_to_molecule_data(
            str(TEST_FILES / "orca" / "orca.inp")
        )

        assert len(species) == coords.shape[0]

    def test_coords_are_numeric(self):
        """Test that all coordinates are numeric."""
        species, coords, charge, spin = parse_orca_inp_to_molecule_data(
            str(TEST_FILES / "orca" / "orca.inp")
        )

        assert not np.isnan(coords).any(), "Coordinates contain NaN"
        assert not np.isinf(coords).any(), "Coordinates contain Inf"


class TestIntegration:
    """Integration tests combining parsing and bond detection."""

    def test_orca_to_bonds_pipeline(self):
        """Test full pipeline from ORCA input to bond detection."""
        species, coords, charge, spin = parse_orca_inp_to_molecule_data(
            str(TEST_FILES / "orca" / "orca.inp")
        )

        bonds = get_bonds_from_coords(species, coords)

        # Should detect reasonable number of bonds for an organic molecule
        assert len(bonds) > 0, "Should detect some bonds"

        # All bond indices should be valid
        n_atoms = len(species)
        for bond in bonds:
            assert 0 <= bond[0] < n_atoms
            assert 0 <= bond[1] < n_atoms
            assert bond[0] != bond[1], "Self-bonds should not exist"

    def test_pymatgen_molecule_construction(self):
        """Test that parsed data can construct pymatgen Molecule."""
        from pymatgen.core import Molecule

        species, coords, charge, spin = parse_orca_inp_to_molecule_data(
            str(TEST_FILES / "orca" / "orca.inp")
        )

        # Should be able to construct Molecule directly
        mol = Molecule(
            species=species,
            coords=coords,
            charge=charge,
            spin_multiplicity=spin,
        )

        assert mol.num_sites == len(species)
        assert mol.charge == charge
        assert mol.spin_multiplicity == spin
