"""Tests for qtaim_gen.source.core.parse_orca."""

import json
import os
import tempfile
from pathlib import Path

import pytest

from qtaim_gen.source.utils.atomic_write import atomic_json_write
from qtaim_gen.source.core.parse_orca import (
    OrcaParseState,
    _atom_key,
    _bond_key,
    _parse_bond_pairs,
    parse_orca_float,
    parse_orca_output,
    write_orca_json,
    merge_orca_into_charge_json,
    merge_orca_into_bond_json,
    find_orca_output_file,
    validate_parse_completeness,
)
from qtaim_gen.source.utils.validation import validate_orca_dict, validation_checks

TEST_FILES = Path(__file__).parent / "test_files" / "orca_outs"

FIXTURE_RKS = str(TEST_FILES / "minimal_rks.out")
FIXTURE_TRUNCATED = str(TEST_FILES / "minimal_truncated.out")
FIXTURE_DUPLICATE = str(TEST_FILES / "minimal_duplicate_energy.out")


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def rks_result():
    return parse_orca_output(FIXTURE_RKS)


@pytest.fixture
def truncated_result():
    return parse_orca_output(FIXTURE_TRUNCATED)


@pytest.fixture
def duplicate_result():
    return parse_orca_output(FIXTURE_DUPLICATE)


@pytest.fixture
def tmp_job_dir():
    """Create a temporary directory that acts like a job folder."""
    with tempfile.TemporaryDirectory() as d:
        yield d


# ── Utility function tests ───────────────────────────────────────────


class TestParseOrcaFloat:

    @pytest.mark.parametrize(
        "value_str, expected",
        [
            ("1.234", 1.234),
            ("-0.5", -0.5),
            ("1.0D+02", 100.0),
            ("1.0d-03", 0.001),
            ("1.2276e-09", 1.2276e-09),
            ("-1342.54535275551711", -1342.54535275551711),
        ],
    )
    def test_valid_floats(self, value_str, expected):
        result = parse_orca_float(value_str)
        assert result == pytest.approx(expected, rel=1e-10)

    @pytest.mark.parametrize(
        "value_str",
        [
            "",
            "   ",
            "***",
            "****",
            "abc",
        ],
    )
    def test_returns_none(self, value_str):
        assert parse_orca_float(value_str) is None


class TestAtomKey:

    def test_basic(self):
        assert _atom_key(0, "O") == "1_O"
        assert _atom_key(52, "H") == "53_H"

    def test_strips_whitespace(self):
        assert _atom_key(0, "  Ac  ") == "1_Ac"


class TestBondKey:

    def test_basic(self):
        assert _bond_key(0, "O", 1, "C") == "1_O_to_2_C"

    def test_strips_whitespace(self):
        assert _bond_key(0, " N ", 2, " C ") == "1_N_to_3_C"


class TestParseBondPairs:

    def test_single_pair(self):
        line = "B(  0-O ,  1-C ) :   2.5719 B(  0-O ,  2-C ) :   0.3283"
        pairs = _parse_bond_pairs(line)
        assert len(pairs) == 2
        assert pairs[0] == ("1_O_to_2_C", pytest.approx(2.5719))
        assert pairs[1] == ("1_O_to_3_C", pytest.approx(0.3283))

    def test_empty_line(self):
        assert _parse_bond_pairs("") == []
        assert _parse_bond_pairs("no bonds here") == []


# ── Truncated file handling ──────────────────────────────────────────


class TestTruncatedFile:

    def test_returns_partial_dict(self, truncated_result):
        """Truncated file should return whatever was parsed before cutoff."""
        # Truncated file has energy block, convergence, orbitals, partial Mulliken
        # but NOT FINAL SINGLE POINT ENERGY (that's later in the full file)
        assert "energy_components" in truncated_result
        assert "scf_convergence" in truncated_result
        assert "homo_eh" in truncated_result

    def test_partial_mulliken(self, truncated_result):
        """File truncates mid-Mulliken; should have partial charges."""
        mc = truncated_result["mulliken_charges"]
        assert 0 < len(mc) < 53
        assert mc["1_O"] == pytest.approx(-0.740774, rel=1e-5)

    def test_missing_sections(self, truncated_result):
        """Sections after truncation point should be absent."""
        assert "loewdin_charges" not in truncated_result
        assert "loewdin_bond_orders" not in truncated_result
        assert "mayer_population" not in truncated_result
        assert "mayer_bond_orders" not in truncated_result
        assert "gradient" not in truncated_result
        assert "dipole_au" not in truncated_result
        assert "quadrupole_au" not in truncated_result

    def test_no_exception(self):
        """Parsing truncated file should not raise."""
        result = parse_orca_output(FIXTURE_TRUNCATED)
        assert isinstance(result, dict)


# ── Duplicate section handling (last occurrence wins) ────────────────


class TestDuplicateSections:

    def test_last_energy_wins(self, duplicate_result):
        assert duplicate_result["final_energy_eh"] == pytest.approx(
            -200.987654321099, rel=1e-12
        )

    def test_last_energy_components_wins(self, duplicate_result):
        ec = duplicate_result["energy_components"]
        assert ec["nuclear_repulsion_eh"] == pytest.approx(
            60.12345678900000, rel=1e-10
        )
        assert ec["virial_ratio"] == pytest.approx(2.00527654321098, rel=1e-10)

    def test_last_scf_convergence_wins(self, duplicate_result):
        sc = duplicate_result["scf_convergence"]
        assert sc["energy_change"] == pytest.approx(1.1111e-09, rel=1e-3)
        assert sc["diis_error"] == pytest.approx(6.6666e-07, rel=1e-3)

    def test_mulliken_charges_present(self, duplicate_result):
        mc = duplicate_result["mulliken_charges"]
        assert len(mc) == 3
        assert mc["1_O"] == pytest.approx(-0.500000, abs=1e-6)
        assert mc["2_C"] == pytest.approx(0.300000, abs=1e-6)
        assert mc["3_H"] == pytest.approx(0.200000, abs=1e-6)


# ── Missing / nonexistent file handling ──────────────────────────────


class TestEdgeCases:

    def test_nonexistent_file(self):
        result = parse_orca_output("/nonexistent/path/orca.out")
        assert result == {}

    def test_empty_file(self, tmp_job_dir):
        empty = os.path.join(tmp_job_dir, "empty.out")
        with open(empty, "w") as f:
            f.write("")
        result = parse_orca_output(empty)
        assert result == {}

    def test_no_relevant_sections(self, tmp_job_dir):
        noise = os.path.join(tmp_job_dir, "noise.out")
        with open(noise, "w") as f:
            f.write("This file has no ORCA sections\n" * 100)
        result = parse_orca_output(noise)
        assert result == {}


# ── JSON I/O tests ───────────────────────────────────────────────────


class TestAtomicJsonWrite:

    def test_writes_valid_json(self, tmp_job_dir):
        path = os.path.join(tmp_job_dir, "test.json")
        data = {"key": "value", "number": 42}
        atomic_json_write(path, data)
        with open(path, "r") as f:
            loaded = json.load(f)
        assert loaded == data

    def test_overwrites_existing(self, tmp_job_dir):
        path = os.path.join(tmp_job_dir, "test.json")
        atomic_json_write(path, {"old": True})
        atomic_json_write(path, {"new": True})
        with open(path, "r") as f:
            loaded = json.load(f)
        assert loaded == {"new": True}


class TestWriteOrcaJson:

    def test_creates_orca_json(self, tmp_job_dir, rks_result):
        write_orca_json(tmp_job_dir, rks_result)
        path = os.path.join(tmp_job_dir, "orca.json")
        assert os.path.isfile(path)
        with open(path, "r") as f:
            loaded = json.load(f)
        assert loaded["final_energy_eh"] == pytest.approx(
            rks_result["final_energy_eh"], rel=1e-12
        )


# Real Multiwfn job folder for integration merge tests
LMDB_TEST_ORCA6 = Path(__file__).parent / "test_files" / "lmdb_tests" / "orca6_rks"


@pytest.fixture
def real_job_dir(tmp_path):
    """Copy real Multiwfn charge.json + bond.json into a tmp dir for merge testing."""
    import shutil
    dest = tmp_path / "job"
    dest.mkdir()
    for name in ("charge.json", "bond.json"):
        shutil.copy2(str(LMDB_TEST_ORCA6 / name), str(dest / name))
    return str(dest)


class TestMergeOrcaIntoChargeJson:
    """Merge ORCA-parsed charges into real Multiwfn charge.json (53-atom orca6_rks)."""

    def test_preserves_all_multiwfn_methods(self, real_job_dir, rks_result):
        charge_path = os.path.join(real_job_dir, "charge.json")
        with open(charge_path, "r") as f:
            original_keys = set(json.load(f).keys())

        merge_orca_into_charge_json(rks_result, charge_path)

        with open(charge_path, "r") as f:
            merged = json.load(f)

        # Every original Multiwfn method should still be present
        for key in original_keys:
            assert key in merged, f"Multiwfn key '{key}' was lost after merge"

    def test_multiwfn_values_unchanged(self, real_job_dir, rks_result):
        charge_path = os.path.join(real_job_dir, "charge.json")
        with open(charge_path, "r") as f:
            before = json.load(f)

        merge_orca_into_charge_json(rks_result, charge_path)

        with open(charge_path, "r") as f:
            after = json.load(f)

        # Spot-check Multiwfn values are untouched
        assert after["becke"]["charge"]["1_O"] == pytest.approx(
            before["becke"]["charge"]["1_O"], rel=1e-12
        )
        assert after["adch"]["charge"]["2_C"] == pytest.approx(
            before["adch"]["charge"]["2_C"], rel=1e-12
        )
        assert after["bader"]["charge"]["53_H"] == pytest.approx(
            before["bader"]["charge"]["53_H"], rel=1e-12
        )

    def test_orca_charge_keys_added(self, real_job_dir, rks_result):
        charge_path = os.path.join(real_job_dir, "charge.json")
        merge_orca_into_charge_json(rks_result, charge_path)

        with open(charge_path, "r") as f:
            merged = json.load(f)

        assert "mulliken_orca" in merged
        assert "loewdin_orca" in merged
        assert "mayer_orca" in merged

    def test_orca_mulliken_values(self, real_job_dir, rks_result):
        charge_path = os.path.join(real_job_dir, "charge.json")
        merge_orca_into_charge_json(rks_result, charge_path)

        with open(charge_path, "r") as f:
            merged = json.load(f)

        orca_mull = merged["mulliken_orca"]["charge"]
        assert len(orca_mull) == 53
        assert orca_mull["1_O"] == pytest.approx(-0.740774, rel=1e-5)
        assert orca_mull["53_H"] == pytest.approx(0.292385, rel=1e-5)

    def test_orca_mayer_values(self, real_job_dir, rks_result):
        charge_path = os.path.join(real_job_dir, "charge.json")
        merge_orca_into_charge_json(rks_result, charge_path)

        with open(charge_path, "r") as f:
            merged = json.load(f)

        orca_mayer = merged["mayer_orca"]["charge"]
        assert len(orca_mayer) == 53
        assert orca_mayer["1_O"] == pytest.approx(-0.7408, rel=1e-4)

    def test_orca_vs_multiwfn_same_atoms(self, real_job_dir, rks_result):
        """ORCA and Multiwfn charges should have the same atom keys."""
        charge_path = os.path.join(real_job_dir, "charge.json")
        merge_orca_into_charge_json(rks_result, charge_path)

        with open(charge_path, "r") as f:
            merged = json.load(f)

        mfwn_atoms = set(merged["becke"]["charge"].keys())
        orca_atoms = set(merged["mulliken_orca"]["charge"].keys())
        assert mfwn_atoms == orca_atoms

    def test_idempotent(self, real_job_dir, rks_result):
        charge_path = os.path.join(real_job_dir, "charge.json")

        merge_orca_into_charge_json(rks_result, charge_path)
        with open(charge_path, "r") as f:
            first = json.load(f)

        merge_orca_into_charge_json(rks_result, charge_path)
        with open(charge_path, "r") as f:
            second = json.load(f)

        assert first == second

    def test_skips_missing_file(self, tmp_job_dir, rks_result):
        charge_path = os.path.join(tmp_job_dir, "charge.json")
        merge_orca_into_charge_json(rks_result, charge_path)
        assert not os.path.isfile(charge_path)

    def test_skips_corrupt_json(self, tmp_job_dir, rks_result):
        charge_path = os.path.join(tmp_job_dir, "charge.json")
        with open(charge_path, "w") as f:
            f.write("{invalid json")
        merge_orca_into_charge_json(rks_result, charge_path)
        with open(charge_path, "r") as f:
            assert f.read() == "{invalid json"


class TestMergeOrcaIntoBondJson:
    """Merge ORCA-parsed bond orders into real Multiwfn bond.json (53-atom orca6_rks)."""

    def test_preserves_multiwfn_bond_orders(self, real_job_dir, rks_result):
        bond_path = os.path.join(real_job_dir, "bond.json")
        with open(bond_path, "r") as f:
            before = json.load(f)

        merge_orca_into_bond_json(rks_result, bond_path)

        with open(bond_path, "r") as f:
            after = json.load(f)

        # Multiwfn fuzzy_bond and ibsi_bond should be untouched
        assert after["fuzzy_bond"]["1_O_to_2_C"] == pytest.approx(
            before["fuzzy_bond"]["1_O_to_2_C"], rel=1e-12
        )
        assert after["ibsi_bond"]["1_O_to_2_C"] == pytest.approx(
            before["ibsi_bond"]["1_O_to_2_C"], rel=1e-12
        )

    def test_orca_bond_keys_added(self, real_job_dir, rks_result):
        bond_path = os.path.join(real_job_dir, "bond.json")
        merge_orca_into_bond_json(rks_result, bond_path)

        with open(bond_path, "r") as f:
            merged = json.load(f)

        assert "fuzzy_bond" in merged  # Multiwfn preserved
        assert "ibsi_bond" in merged   # Multiwfn preserved
        assert "mayer_orca" in merged   # ORCA added
        assert "loewdin_orca" in merged # ORCA added

    def test_orca_mayer_bond_values(self, real_job_dir, rks_result):
        bond_path = os.path.join(real_job_dir, "bond.json")
        merge_orca_into_bond_json(rks_result, bond_path)

        with open(bond_path, "r") as f:
            merged = json.load(f)

        mayer = merged["mayer_orca"]
        assert len(mayer) == 90
        assert mayer["10_C_to_11_C"] == pytest.approx(0.1624, rel=1e-4)

    def test_orca_loewdin_bond_values(self, real_job_dir, rks_result):
        bond_path = os.path.join(real_job_dir, "bond.json")
        merge_orca_into_bond_json(rks_result, bond_path)

        with open(bond_path, "r") as f:
            merged = json.load(f)

        loewdin = merged["loewdin_orca"]
        assert len(loewdin) == 179
        assert loewdin["10_C_to_11_C"] == pytest.approx(1.0424, rel=1e-4)

    def test_idempotent(self, real_job_dir, rks_result):
        bond_path = os.path.join(real_job_dir, "bond.json")

        merge_orca_into_bond_json(rks_result, bond_path)
        with open(bond_path, "r") as f:
            first = json.load(f)

        merge_orca_into_bond_json(rks_result, bond_path)
        with open(bond_path, "r") as f:
            second = json.load(f)

        assert first == second

    def test_skips_missing_file(self, tmp_job_dir, rks_result):
        bond_path = os.path.join(tmp_job_dir, "bond.json")
        merge_orca_into_bond_json(rks_result, bond_path)
        assert not os.path.isfile(bond_path)


# ── Enum state test ──────────────────────────────────────────────────


def test_orca_parse_state_enum():
    """Sanity check that all states are defined."""
    assert len(OrcaParseState) >= 15
    assert OrcaParseState.IDLE.value != OrcaParseState.SCF_ENERGY_BLOCK.value


# ── Full reference file tests ─────────────────────────────────────────
# Parse real ORCA output files from data/orca_outs_4_reference/ and validate
# actual numeric values against known-good outputs.
# Skipped if the reference directory is not present.

REFERENCE_DIR = Path(__file__).parent.parent / "data" / "orca_outs_4_reference"
_ref_available = REFERENCE_DIR.is_dir()

_skip_no_ref = pytest.mark.skipif(not _ref_available, reason="Reference files not present")


@pytest.fixture(scope="module")
def ref_orca6_rks():
    """Parse orca6_rks.out once per module (RKS, 53 atoms, no MBIS)."""
    path = REFERENCE_DIR / "orca6_rks.out"
    if not path.is_file():
        pytest.skip("orca6_rks.out not found")
    return parse_orca_output(str(path))


@pytest.fixture(scope="module")
def ref_uks_mbis():
    """Parse orca_mbis_nbo_non_act_1.out once per module (UKS, 64 atoms, MBIS, Po/Te)."""
    path = REFERENCE_DIR / "orca_mbis_nbo_non_act_1.out"
    if not path.is_file():
        pytest.skip("orca_mbis_nbo_non_act_1.out not found")
    return parse_orca_output(str(path))


@pytest.fixture(scope="module")
def ref_rks_mbis():
    """Parse orca_mbis_nbo_non_act_2.out once per module (RKS, 40 atoms, MBIS converged, Po)."""
    path = REFERENCE_DIR / "orca_mbis_nbo_non_act_2.out"
    if not path.is_file():
        pytest.skip("orca_mbis_nbo_non_act_2.out not found")
    return parse_orca_output(str(path))


@_skip_no_ref
class TestRefOrca6RKS:
    """Value-based tests against real orca6_rks.out (RKS, 53-atom organic)."""

    def test_final_energy(self, ref_orca6_rks):
        assert ref_orca6_rks["final_energy_eh"] == pytest.approx(
            -1342.545352755517, rel=1e-12
        )

    def test_scf_metadata(self, ref_orca6_rks):
        assert ref_orca6_rks["scf_converged"] is True
        assert ref_orca6_rks["scf_cycles"] == 15
        assert ref_orca6_rks["n_electrons"] == pytest.approx(210.0)
        assert ref_orca6_rks["n_orbitals"] == 1301

    def test_energy_components(self, ref_orca6_rks):
        ec = ref_orca6_rks["energy_components"]
        assert ec["nuclear_repulsion_eh"] == pytest.approx(2557.63336953413, rel=1e-10)
        assert ec["electronic_energy_eh"] == pytest.approx(-3900.9588409530616, rel=1e-10)
        assert ec["virial_ratio"] == pytest.approx(2.00292811861515, rel=1e-10)
        assert ec["xc_energy_eh"] == pytest.approx(-130.730757296629, rel=1e-10)

    def test_scf_convergence(self, ref_orca6_rks):
        sc = ref_orca6_rks["scf_convergence"]
        assert sc["energy_change"] == pytest.approx(1.2276e-09, rel=1e-3)
        assert sc["diis_error"] == pytest.approx(8.0031e-06, rel=1e-3)

    def test_orbital_energies(self, ref_orca6_rks):
        assert ref_orca6_rks["homo_eh"] == pytest.approx(-0.311606, rel=1e-5)
        assert ref_orca6_rks["lumo_eh"] == pytest.approx(-0.009184, rel=1e-5)
        assert ref_orca6_rks["homo_lumo_gap_eh"] == pytest.approx(0.302422, rel=1e-5)

    def test_mulliken_charges(self, ref_orca6_rks):
        mc = ref_orca6_rks["mulliken_charges"]
        assert len(mc) == 53
        assert mc["1_O"] == pytest.approx(-0.740774, rel=1e-5)
        assert mc["53_H"] == pytest.approx(0.292385, rel=1e-5)

    def test_loewdin_charges(self, ref_orca6_rks):
        lc = ref_orca6_rks["loewdin_charges"]
        assert len(lc) == 53
        assert lc["1_O"] == pytest.approx(0.048565, rel=1e-5)
        assert lc["53_H"] == pytest.approx(0.142150, rel=1e-5)

    def test_mayer_charges(self, ref_orca6_rks):
        mc = ref_orca6_rks["mayer_charges"]
        assert len(mc) == 53
        assert mc["1_O"] == pytest.approx(-0.7408, rel=1e-4)
        assert mc["53_H"] == pytest.approx(0.2924, rel=1e-4)

    def test_no_spin_columns(self, ref_orca6_rks):
        """RKS should not produce spin columns."""
        assert "mulliken_spins" not in ref_orca6_rks
        assert "loewdin_spins" not in ref_orca6_rks

    def test_no_mbis(self, ref_orca6_rks):
        """RKS-only file should not have MBIS or Hirshfeld sections."""
        assert "mbis_charges" not in ref_orca6_rks
        assert "hirshfeld_charges" not in ref_orca6_rks

    def test_loewdin_bond_orders(self, ref_orca6_rks):
        lb = ref_orca6_rks["loewdin_bond_orders"]
        assert len(lb) == 179
        assert lb["10_C_to_11_C"] == pytest.approx(1.0424, rel=1e-4)

    def test_mayer_bond_orders(self, ref_orca6_rks):
        mb = ref_orca6_rks["mayer_bond_orders"]
        assert len(mb) == 90
        assert mb["10_C_to_11_C"] == pytest.approx(0.1624, rel=1e-4)

    def test_gradient(self, ref_orca6_rks):
        g = ref_orca6_rks["gradient"]
        assert len(g) == 53
        assert g["1_O"] == pytest.approx(
            [0.000385692, 0.000381587, 0.000766511], rel=1e-5
        )

    def test_gradient_stats(self, ref_orca6_rks):
        assert ref_orca6_rks["gradient_norm"] == pytest.approx(0.1700783337, rel=1e-8)
        assert ref_orca6_rks["gradient_rms"] == pytest.approx(0.0134880892, rel=1e-8)
        assert ref_orca6_rks["gradient_max"] == pytest.approx(0.0542779604, rel=1e-8)

    def test_dipole(self, ref_orca6_rks):
        assert ref_orca6_rks["dipole_au"] == pytest.approx(
            [1.363229353, 0.571139804, -0.220405496], rel=1e-8
        )
        assert ref_orca6_rks["dipole_magnitude_au"] == pytest.approx(1.49438065, rel=1e-8)

    def test_quadrupole(self, ref_orca6_rks):
        assert ref_orca6_rks["quadrupole_au"] == pytest.approx(
            [-126.887126542, -116.801679527, -133.912552927,
             -3.526753699, 6.943015886, 1.386549503],
            rel=1e-8,
        )

    def test_rotational_constants(self, ref_orca6_rks):
        assert ref_orca6_rks["rotational_constants_cm1"] == pytest.approx(
            [0.013421, 0.001535, 0.001508], rel=1e-4
        )

    def test_total_run_time(self, ref_orca6_rks):
        assert ref_orca6_rks["total_run_time_s"] == pytest.approx(2454.733, rel=1e-6)

    def test_charge_atom_counts_consistent(self, ref_orca6_rks):
        """All charge-type dicts should have the same atom count."""
        n = 53
        assert len(ref_orca6_rks["mulliken_charges"]) == n
        assert len(ref_orca6_rks["loewdin_charges"]) == n
        assert len(ref_orca6_rks["mayer_charges"]) == n
        assert len(ref_orca6_rks["mayer_population"]) == n


@_skip_no_ref
class TestRefUKSMBIS:
    """Value-based tests against real orca_mbis_nbo_non_act_1.out
    (UKS, 64 atoms, Po/Te heavy elements, MBIS + Hirshfeld, spin columns)."""

    def test_final_energy(self, ref_uks_mbis):
        assert ref_uks_mbis["final_energy_eh"] == pytest.approx(
            -2671.064484741533, rel=1e-12
        )

    def test_scf_metadata(self, ref_uks_mbis):
        assert ref_uks_mbis["scf_converged"] is True
        assert ref_uks_mbis["scf_cycles"] == 23
        assert ref_uks_mbis["n_electrons"] == pytest.approx(158.0)
        assert ref_uks_mbis["n_orbitals"] == 1368

    def test_energy_components(self, ref_uks_mbis):
        ec = ref_uks_mbis["energy_components"]
        assert ec["nuclear_repulsion_eh"] == pytest.approx(5853.314934011381, rel=1e-10)
        assert ec["electronic_energy_eh"] == pytest.approx(-8525.490670378018, rel=1e-10)
        assert ec["virial_ratio"] == pytest.approx(2.97867610849658, rel=1e-10)
        assert ec["xc_energy_eh"] == pytest.approx(-175.863000371855, rel=1e-10)

    def test_scf_convergence(self, ref_uks_mbis):
        sc = ref_uks_mbis["scf_convergence"]
        assert sc["energy_change"] == pytest.approx(5.3224e-09, rel=1e-3)
        assert sc["diis_error"] == pytest.approx(0.00071014, rel=1e-3)

    def test_orbital_energies(self, ref_uks_mbis):
        assert ref_uks_mbis["homo_eh"] == pytest.approx(-0.559159, rel=1e-5)
        assert ref_uks_mbis["lumo_eh"] == pytest.approx(-0.41149, rel=1e-5)
        assert ref_uks_mbis["homo_lumo_gap_eh"] == pytest.approx(0.147669, rel=1e-4)

    def test_mulliken_charges(self, ref_uks_mbis):
        mc = ref_uks_mbis["mulliken_charges"]
        assert len(mc) == 64
        assert mc["1_Po"] == pytest.approx(-0.465047, rel=1e-5)
        assert mc["64_H"] == pytest.approx(0.461158, rel=1e-5)

    def test_loewdin_charges(self, ref_uks_mbis):
        lc = ref_uks_mbis["loewdin_charges"]
        assert len(lc) == 64
        assert lc["1_Po"] == pytest.approx(-0.074407, rel=1e-5)
        assert lc["64_H"] == pytest.approx(0.047861, rel=1e-5)

    def test_mayer_charges(self, ref_uks_mbis):
        mc = ref_uks_mbis["mayer_charges"]
        assert len(mc) == 64
        assert mc["1_Po"] == pytest.approx(-0.465, rel=1e-3)

    def test_mulliken_spins(self, ref_uks_mbis):
        """UKS should produce non-zero spin populations on heavy atoms."""
        ms = ref_uks_mbis["mulliken_spins"]
        assert len(ms) == 64
        assert ms["1_Po"] == pytest.approx(0.342488, rel=1e-5)
        assert ms["21_Te"] == pytest.approx(0.176223, rel=1e-5)
        # All values should be non-zero for this open-shell system
        assert all(abs(v) > 1e-6 for v in ms.values())

    def test_loewdin_spins(self, ref_uks_mbis):
        ls = ref_uks_mbis["loewdin_spins"]
        assert len(ls) == 64
        assert ls["1_Po"] == pytest.approx(0.326781, rel=1e-5)
        assert all(abs(v) > 1e-6 for v in ls.values())

    def test_hirshfeld_charges(self, ref_uks_mbis):
        hc = ref_uks_mbis["hirshfeld_charges"]
        assert len(hc) == 64
        assert hc["1_Po"] == pytest.approx(0.246689, rel=1e-5)
        assert hc["64_H"] == pytest.approx(0.068135, rel=1e-5)

    def test_hirshfeld_spins(self, ref_uks_mbis):
        hs = ref_uks_mbis["hirshfeld_spins"]
        assert len(hs) == 64
        assert hs["1_Po"] == pytest.approx(0.335693, rel=1e-5)
        assert all(abs(v) > 1e-6 for v in hs.values())

    def test_mbis_charges_all_nan(self, ref_uks_mbis):
        """MBIS fitting diverged (144 iters) — all atoms should be NaN, not just heavy ones."""
        import math
        mc = ref_uks_mbis["mbis_charges"]
        assert len(mc) == 64
        assert all(math.isnan(v) for v in mc.values())

    def test_mbis_spins_all_nan(self, ref_uks_mbis):
        """MBIS fitting diverged — all spins should be NaN."""
        import math
        mbs = ref_uks_mbis["mbis_spins"]
        assert len(mbs) == 64
        assert all(math.isnan(v) for v in mbs.values())

    def test_mbis_populations(self, ref_uks_mbis):
        mp = ref_uks_mbis["mbis_populations"]
        assert len(mp) == 64

    def test_mbis_valence_widths(self, ref_uks_mbis):
        vw = ref_uks_mbis["mbis_valence_widths"]
        assert len(vw) == 64

    def test_loewdin_bond_orders(self, ref_uks_mbis):
        lb = ref_uks_mbis["loewdin_bond_orders"]
        assert len(lb) == 212
        assert lb["11_C_to_12_Te"] == pytest.approx(1.2767, rel=1e-4)

    def test_mayer_bond_orders(self, ref_uks_mbis):
        mb = ref_uks_mbis["mayer_bond_orders"]
        assert len(mb) == 87
        assert mb["11_C_to_12_Te"] == pytest.approx(0.8021, rel=1e-4)

    def test_gradient(self, ref_uks_mbis):
        g = ref_uks_mbis["gradient"]
        assert len(g) == 64
        assert g["1_Po"] == pytest.approx(
            [-0.013658833, -0.013986185, -0.03158535], rel=1e-5
        )

    def test_gradient_stats(self, ref_uks_mbis):
        assert ref_uks_mbis["gradient_norm"] == pytest.approx(0.1962350659, rel=1e-8)
        assert ref_uks_mbis["gradient_rms"] == pytest.approx(0.014162046, rel=1e-8)
        assert ref_uks_mbis["gradient_max"] == pytest.approx(0.0596251334, rel=1e-8)

    def test_dipole(self, ref_uks_mbis):
        assert ref_uks_mbis["dipole_au"] == pytest.approx(
            [-0.230531505, -0.507879878, 0.775276307], rel=1e-8
        )
        assert ref_uks_mbis["dipole_magnitude_au"] == pytest.approx(0.955060259, rel=1e-8)

    def test_quadrupole(self, ref_uks_mbis):
        assert ref_uks_mbis["quadrupole_au"] == pytest.approx(
            [-140.223601754, -135.06209933, -148.577865221,
             -3.751992159, 0.472254585, 2.442162234],
            rel=1e-8,
        )

    def test_rotational_constants(self, ref_uks_mbis):
        assert ref_uks_mbis["rotational_constants_cm1"] == pytest.approx(
            [0.002189, 0.002155, 0.001979], rel=1e-4
        )

    def test_total_run_time(self, ref_uks_mbis):
        assert ref_uks_mbis["total_run_time_s"] == pytest.approx(5374.756, rel=1e-6)

    def test_charge_atom_counts_consistent(self, ref_uks_mbis):
        """All charge/spin dicts should have the same atom count (64)."""
        n = 64
        assert len(ref_uks_mbis["mulliken_charges"]) == n
        assert len(ref_uks_mbis["loewdin_charges"]) == n
        assert len(ref_uks_mbis["mayer_charges"]) == n
        assert len(ref_uks_mbis["hirshfeld_charges"]) == n
        assert len(ref_uks_mbis["mbis_charges"]) == n
        assert len(ref_uks_mbis["mulliken_spins"]) == n
        assert len(ref_uks_mbis["loewdin_spins"]) == n

    def test_all_keys_present(self, ref_uks_mbis):
        """UKS+MBIS should produce the full set of 36 keys."""
        expected = {
            "final_energy_eh", "scf_converged", "scf_cycles", "scf_convergence",
            "energy_components", "homo_eh", "homo_ev", "lumo_eh", "lumo_ev",
            "homo_lumo_gap_eh", "n_electrons", "n_orbitals",
            "mulliken_charges", "mulliken_spins",
            "loewdin_charges", "loewdin_spins",
            "mayer_charges", "mayer_population", "mayer_bond_orders",
            "loewdin_bond_orders",
            "hirshfeld_charges", "hirshfeld_spins",
            "mbis_charges", "mbis_populations", "mbis_spins",
            "mbis_valence_populations", "mbis_valence_widths",
            "gradient", "gradient_norm", "gradient_rms", "gradient_max",
            "dipole_au", "dipole_magnitude_au", "quadrupole_au",
            "rotational_constants_cm1", "total_run_time_s",
        }
        assert expected.issubset(set(ref_uks_mbis.keys()))


@_skip_no_ref
class TestRefRKSMBIS:
    """Value-based tests for real MBIS values from orca_mbis_nbo_non_act_2.out
    (RKS, 40 atoms, Po, MBIS converged in 73 iterations)."""

    def test_final_energy(self, ref_rks_mbis):
        assert ref_rks_mbis["final_energy_eh"] == pytest.approx(
            -1546.295655240122, rel=1e-12
        )

    def test_mbis_charges(self, ref_rks_mbis):
        mc = ref_rks_mbis["mbis_charges"]
        assert len(mc) == 40
        assert mc["1_Po"] == pytest.approx(1.644959, rel=1e-5)
        assert mc["2_O"] == pytest.approx(-0.485639, rel=1e-5)
        assert mc["3_C"] == pytest.approx(0.626927, rel=1e-5)
        assert mc["10_N"] == pytest.approx(-0.353352, rel=1e-5)
        assert mc["40_H"] == pytest.approx(0.136429, rel=1e-5)

    def test_mbis_populations(self, ref_rks_mbis):
        mp = ref_rks_mbis["mbis_populations"]
        assert len(mp) == 40
        assert mp["1_Po"] == pytest.approx(22.355041, rel=1e-5)
        assert mp["2_O"] == pytest.approx(8.485639, rel=1e-5)

    def test_mbis_spins_zero_for_rks(self, ref_rks_mbis):
        """RKS MBIS should have all-zero spins (not NaN)."""
        ms = ref_rks_mbis["mbis_spins"]
        assert len(ms) == 40
        assert all(v == pytest.approx(0.0, abs=1e-6) for v in ms.values())

    def test_mbis_valence_populations(self, ref_rks_mbis):
        vp = ref_rks_mbis["mbis_valence_populations"]
        assert len(vp) == 40
        assert vp["1_Po"] == pytest.approx(9.290057, rel=1e-5)
        assert vp["10_N"] == pytest.approx(5.732071, rel=1e-5)

    def test_mbis_valence_widths(self, ref_rks_mbis):
        vw = ref_rks_mbis["mbis_valence_widths"]
        assert len(vw) == 40
        assert vw["1_Po"] == pytest.approx(0.433629, rel=1e-5)
        assert vw["10_N"] == pytest.approx(0.440995, rel=1e-5)

    def test_hirshfeld_charges(self, ref_rks_mbis):
        hc = ref_rks_mbis["hirshfeld_charges"]
        assert len(hc) == 40
        assert hc["1_Po"] == pytest.approx(0.714931, rel=1e-5)
        assert hc["40_H"] == pytest.approx(0.057653, rel=1e-5)

    def test_no_spin_columns(self, ref_rks_mbis):
        """RKS should not produce Mulliken/Loewdin spin columns."""
        assert "mulliken_spins" not in ref_rks_mbis
        assert "loewdin_spins" not in ref_rks_mbis

    def test_charge_counts_consistent(self, ref_rks_mbis):
        n = 40
        assert len(ref_rks_mbis["mulliken_charges"]) == n
        assert len(ref_rks_mbis["loewdin_charges"]) == n
        assert len(ref_rks_mbis["mbis_charges"]) == n
        assert len(ref_rks_mbis["hirshfeld_charges"]) == n
        assert len(ref_rks_mbis["mbis_populations"]) == n
        assert len(ref_rks_mbis["mbis_valence_populations"]) == n
        assert len(ref_rks_mbis["mbis_valence_widths"]) == n


# ── find_orca_output_file tests ──────────────────────────────────────


class TestFindOrcaOutputFile:

    def test_finds_orca_out(self, tmp_job_dir):
        path = os.path.join(tmp_job_dir, "orca.out")
        with open(path, "w") as f:
            f.write("dummy")
        assert find_orca_output_file(tmp_job_dir) == path

    def test_finds_output_out(self, tmp_job_dir):
        path = os.path.join(tmp_job_dir, "output.out")
        with open(path, "w") as f:
            f.write("dummy")
        assert find_orca_output_file(tmp_job_dir) == path

    def test_prefers_orca_out_over_output_out(self, tmp_job_dir):
        for name in ("orca.out", "output.out"):
            with open(os.path.join(tmp_job_dir, name), "w") as f:
                f.write("dummy")
        result = find_orca_output_file(tmp_job_dir)
        assert result.endswith("orca.out")

    def test_returns_none_when_missing(self, tmp_job_dir):
        assert find_orca_output_file(tmp_job_dir) is None


# ── validate_parse_completeness tests ───────────────────────────────


class TestValidateParseCompleteness:

    def test_complete_dict(self, rks_result):
        assert validate_parse_completeness(rks_result) is True

    def test_energy_only_is_incomplete(self):
        assert validate_parse_completeness({"final_energy_eh": -100.0}) is False

    def test_charges_only_is_incomplete(self):
        assert validate_parse_completeness({"mulliken_charges": {"1_O": -0.5}}) is False

    def test_empty_dict_is_incomplete(self):
        assert validate_parse_completeness({}) is False

    def test_energy_plus_loewdin_is_complete(self):
        d = {"final_energy_eh": -100.0, "loewdin_charges": {"1_O": -0.5}}
        assert validate_parse_completeness(d) is True

    def test_truncated_fixture_is_incomplete(self, truncated_result):
        # truncated fixture has energy_components but NOT final_energy_eh
        assert validate_parse_completeness(truncated_result) is False


# ── validate_orca_dict tests ─────────────────────────────────────────


class TestValidateOrcaDict:

    def test_absent_file_is_valid(self, tmp_job_dir):
        path = os.path.join(tmp_job_dir, "orca.json")
        assert validate_orca_dict(path) is True

    def test_empty_file_is_invalid(self, tmp_job_dir):
        path = os.path.join(tmp_job_dir, "orca.json")
        with open(path, "w") as f:
            f.write("")
        assert validate_orca_dict(path) is False

    def test_corrupt_json_is_invalid(self, tmp_job_dir):
        path = os.path.join(tmp_job_dir, "orca.json")
        with open(path, "w") as f:
            f.write("{invalid")
        assert validate_orca_dict(path) is False

    def test_empty_dict_is_invalid(self, tmp_job_dir):
        path = os.path.join(tmp_job_dir, "orca.json")
        with open(path, "w") as f:
            json.dump({}, f)
        assert validate_orca_dict(path) is False

    def test_valid_minimal(self, tmp_job_dir):
        path = os.path.join(tmp_job_dir, "orca.json")
        with open(path, "w") as f:
            json.dump({"final_energy_eh": -100.0}, f)
        assert validate_orca_dict(path) is True

    def test_wrong_type_fails(self, tmp_job_dir):
        path = os.path.join(tmp_job_dir, "orca.json")
        with open(path, "w") as f:
            json.dump({"final_energy_eh": "not_a_number"}, f)
        assert validate_orca_dict(path) is False

    def test_bad_array_length_fails(self, tmp_job_dir):
        path = os.path.join(tmp_job_dir, "orca.json")
        with open(path, "w") as f:
            json.dump({"final_energy_eh": -100.0, "dipole_au": [1.0, 2.0]}, f)
        assert validate_orca_dict(path) is False

    def test_correct_array_length_passes(self, tmp_job_dir):
        path = os.path.join(tmp_job_dir, "orca.json")
        with open(path, "w") as f:
            json.dump({"final_energy_eh": -100.0, "dipole_au": [1.0, 2.0, 3.0]}, f)
        assert validate_orca_dict(path) is True

    def test_roundtrip_with_rks(self, tmp_job_dir, rks_result):
        """Write parsed result, then validate it."""
        write_orca_json(tmp_job_dir, rks_result)
        path = os.path.join(tmp_job_dir, "orca.json")
        assert validate_orca_dict(path) is True


# ── validation_checks check_orca flag tests ──────────────────────────
# Uses a copy of tests/test_files/lmdb_tests/orca6_rks with the timings.json
# patched to include the fuzzy_bond key so base validation passes.

LMDB_TEST_FOLDER = Path(__file__).parent / "test_files" / "lmdb_tests" / "orca6_rks"


@pytest.fixture
def valid_job_dir(tmp_path):
    """Copy a real test job folder and patch timings.json so validation passes."""
    import shutil
    dest = tmp_path / "job"
    shutil.copytree(str(LMDB_TEST_FOLDER), str(dest))
    # Patch timings.json to include all keys that validate_timing_dict expects
    # for full_set=0 (the test folder's timings.json predates some newer keys)
    timings_path = dest / "timings.json"
    with open(timings_path, "r") as f:
        timings = json.load(f)
    for key in ("fuzzy_bond", "becke_fuzzy_density", "hirsh_fuzzy_density"):
        if key not in timings:
            timings[key] = 5.0
    with open(timings_path, "w") as f:
        json.dump(timings, f)
    # Patch fuzzy_full.json — add hirsh_fuzzy_density if missing
    # (test fixture predates this key; validator requires it at full_set=0)
    fuzzy_path = dest / "fuzzy_full.json"
    with open(fuzzy_path, "r") as f:
        fuzzy = json.load(f)
    if "hirsh_fuzzy_density" not in fuzzy:
        # Mirror the becke_fuzzy_density structure (same atom keys)
        fuzzy["hirsh_fuzzy_density"] = dict(fuzzy["becke_fuzzy_density"])
    with open(fuzzy_path, "w") as f:
        json.dump(fuzzy, f)
    return str(dest)


class TestValidationChecksOrcaFlag:

    def test_passes_without_orca_json_when_not_required(self, valid_job_dir):
        result = validation_checks(
            valid_job_dir, full_set=0, move_results=False, check_orca=False
        )
        assert result is True

    def test_fails_without_orca_json_when_required(self, valid_job_dir):
        result = validation_checks(
            valid_job_dir, full_set=0, move_results=False, check_orca=True
        )
        assert result is False

    def test_passes_with_valid_orca_json_when_required(self, valid_job_dir):
        orca_path = os.path.join(valid_job_dir, "orca.json")
        with open(orca_path, "w") as f:
            json.dump({"final_energy_eh": -100.0}, f)
        result = validation_checks(
            valid_job_dir, full_set=0, move_results=False, check_orca=True
        )
        assert result is True

    def test_fails_with_corrupt_orca_json_when_required(self, valid_job_dir):
        orca_path = os.path.join(valid_job_dir, "orca.json")
        with open(orca_path, "w") as f:
            f.write("{corrupt")
        result = validation_checks(
            valid_job_dir, full_set=0, move_results=False, check_orca=True
        )
        assert result is False

    def test_validates_present_orca_even_when_not_required(self, valid_job_dir):
        """When check_orca=False but orca.json is present and corrupt, should fail."""
        orca_path = os.path.join(valid_job_dir, "orca.json")
        with open(orca_path, "w") as f:
            f.write("{corrupt")
        result = validation_checks(
            valid_job_dir, full_set=0, move_results=False, check_orca=False
        )
        assert result is False
