"""Tests for qtaim_gen.source.core.parse_orca."""

import json
import os
import tempfile
from pathlib import Path

import pytest

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
    _atomic_json_write,
)

TEST_FILES = Path(__file__).parent / "test_files" / "orca_outs"

FIXTURE_RKS = str(TEST_FILES / "minimal_rks.out")
FIXTURE_MBIS = str(TEST_FILES / "minimal_mbis.out")
FIXTURE_TRUNCATED = str(TEST_FILES / "minimal_truncated.out")
FIXTURE_DUPLICATE = str(TEST_FILES / "minimal_duplicate_energy.out")


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def rks_result():
    return parse_orca_output(FIXTURE_RKS)


@pytest.fixture
def mbis_result():
    return parse_orca_output(FIXTURE_MBIS)


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


# ── Section extraction tests (minimal_rks.out) ──────────────────────


class TestRKSParsing:

    def test_keys_present(self, rks_result):
        expected_keys = {
            "final_energy_eh",
            "scf_converged",
            "scf_cycles",
            "energy_components",
            "scf_convergence",
            "homo_eh",
            "homo_ev",
            "lumo_eh",
            "lumo_ev",
            "homo_lumo_gap_eh",
            "n_electrons",
            "n_orbitals",
            "mulliken_charges",
            "loewdin_charges",
            "loewdin_bond_orders",
            "mayer_population",
            "mayer_charges",
            "mayer_bond_orders",
            "gradient",
            "gradient_norm",
            "gradient_rms",
            "gradient_max",
            "dipole_au",
            "dipole_magnitude_au",
            "quadrupole_au",
            "rotational_constants_cm1",
            "total_run_time_s",
        }
        assert expected_keys.issubset(set(rks_result.keys()))

    def test_final_energy(self, rks_result):
        assert rks_result["final_energy_eh"] == pytest.approx(
            -1342.545352755517, rel=1e-12
        )
        assert rks_result["scf_converged"] is True

    def test_energy_components(self, rks_result):
        ec = rks_result["energy_components"]
        assert ec["nuclear_repulsion_eh"] == pytest.approx(2557.63336953413, rel=1e-10)
        assert ec["electronic_energy_eh"] == pytest.approx(
            -3900.95884095306155, rel=1e-10
        )
        assert ec["one_electron_energy_eh"] == pytest.approx(
            -6910.10794669760344, rel=1e-10
        )
        assert ec["two_electron_energy_eh"] == pytest.approx(
            3009.14910574454188, rel=1e-10
        )
        assert ec["virial_ratio"] == pytest.approx(2.00292811861515, rel=1e-10)
        assert ec["nl_energy_eh"] == pytest.approx(0.780114040230, rel=1e-10)
        assert ec["xc_energy_eh"] == pytest.approx(-130.730757296629, rel=1e-10)

    def test_scf_convergence(self, rks_result):
        sc = rks_result["scf_convergence"]
        assert sc["energy_change"] == pytest.approx(1.2276e-09, rel=1e-3)
        assert sc["max_density_change"] == pytest.approx(1.7607e-03, rel=1e-3)
        assert sc["rms_density_change"] == pytest.approx(1.0580e-05, rel=1e-3)
        assert sc["diis_error"] == pytest.approx(8.0031e-06, rel=1e-3)

    def test_orbital_energies(self, rks_result):
        assert rks_result["homo_eh"] == pytest.approx(-0.311606, rel=1e-5)
        assert rks_result["homo_ev"] == pytest.approx(-8.4792, rel=1e-4)
        assert rks_result["lumo_eh"] == pytest.approx(-0.009184, rel=1e-5)
        assert rks_result["lumo_ev"] == pytest.approx(-0.2499, rel=1e-3)
        assert rks_result["homo_lumo_gap_eh"] == pytest.approx(0.302422, rel=1e-5)

    def test_mulliken_charges(self, rks_result):
        mc = rks_result["mulliken_charges"]
        assert len(mc) == 53
        assert mc["1_O"] == pytest.approx(-0.740774, rel=1e-5)
        assert mc["2_C"] == pytest.approx(0.978726, rel=1e-5)
        assert mc["53_H"] == pytest.approx(0.292385, rel=1e-5)

    def test_loewdin_charges(self, rks_result):
        lc = rks_result["loewdin_charges"]
        assert len(lc) == 53
        assert lc["1_O"] == pytest.approx(0.048565, rel=1e-5)
        assert lc["9_N"] == pytest.approx(0.081577, rel=1e-5)

    def test_loewdin_bond_orders(self, rks_result):
        lb = rks_result["loewdin_bond_orders"]
        assert lb["1_O_to_2_C"] == pytest.approx(2.5719, rel=1e-4)
        assert lb["1_O_to_3_C"] == pytest.approx(0.3283, rel=1e-4)
        assert len(lb) > 50

    def test_mayer_population(self, rks_result):
        mp = rks_result["mayer_population"]
        assert len(mp) == 53
        assert mp["1_O"]["va"] == pytest.approx(1.5029, rel=1e-4)
        assert mp["1_O"]["bva"] == pytest.approx(1.5029, rel=1e-4)

    def test_mayer_bond_orders(self, rks_result):
        mb = rks_result["mayer_bond_orders"]
        assert mb["1_O_to_2_C"] == pytest.approx(1.3593, rel=1e-4)
        assert len(mb) > 30

    def test_gradient(self, rks_result):
        g = rks_result["gradient"]
        assert len(g) == 53
        assert g["1_O"] == pytest.approx(
            [0.000385692, 0.000381587, 0.000766511], rel=1e-5
        )
        assert g["2_C"] == pytest.approx(
            [0.020861570, 0.020973062, -0.008697032], rel=1e-5
        )

    def test_dipole(self, rks_result):
        assert rks_result["dipole_au"] == pytest.approx(
            [1.363229353, 0.571139804, -0.220405496], rel=1e-8
        )
        assert rks_result["dipole_magnitude_au"] == pytest.approx(
            1.494380650, rel=1e-8
        )

    def test_quadrupole(self, rks_result):
        assert rks_result["quadrupole_au"] == pytest.approx(
            [
                -126.887126542,
                -116.801679527,
                -133.912552927,
                -3.526753699,
                6.943015886,
                1.386549503,
            ],
            rel=1e-8,
        )

    def test_scf_cycles(self, rks_result):
        assert rks_result["scf_cycles"] == 15

    def test_gradient_stats(self, rks_result):
        assert rks_result["gradient_norm"] == pytest.approx(0.1700783337, rel=1e-8)
        assert rks_result["gradient_rms"] == pytest.approx(0.0134880892, rel=1e-8)
        assert rks_result["gradient_max"] == pytest.approx(0.0542779604, rel=1e-8)

    def test_mayer_charges(self, rks_result):
        mc = rks_result["mayer_charges"]
        assert len(mc) == 53
        # QA column: "  0 O      8.7408     8.0000    -0.7408 ..."
        assert mc["1_O"] == pytest.approx(-0.7408, rel=1e-4)
        assert mc["2_C"] == pytest.approx(0.9787, rel=1e-4)
        assert mc["9_N"] == pytest.approx(-0.0592, rel=1e-3)

    def test_total_run_time(self, rks_result):
        # 0 days 0 hours 24 minutes 5 seconds 906 msec = 1445.906 s
        assert rks_result["total_run_time_s"] == pytest.approx(1445.906, rel=1e-6)

    def test_rotational_constants(self, rks_result):
        assert rks_result["rotational_constants_cm1"] == pytest.approx(
            [0.013421, 0.001535, 0.001508], rel=1e-4
        )


# ── Section extraction tests (minimal_mbis.out) ─────────────────────


class TestMBISParsing:

    def test_keys_present(self, mbis_result):
        expected_keys = {
            "final_energy_eh",
            "scf_converged",
            "scf_convergence",
            "homo_eh",
            "lumo_eh",
            "mulliken_charges",
            "loewdin_charges",
            "loewdin_bond_orders",
            "mayer_population",
            "mayer_bond_orders",
            "dipole_au",
            "dipole_magnitude_au",
            "quadrupole_au",
            "hirshfeld_charges",
            "hirshfeld_spins",
            "mbis_charges",
            "mbis_populations",
            "mbis_spins",
            "mbis_valence_populations",
            "mbis_valence_widths",
        }
        assert expected_keys.issubset(set(mbis_result.keys()))

    def test_final_energy(self, mbis_result):
        assert mbis_result["final_energy_eh"] == pytest.approx(
            -1219.951729570202, rel=1e-12
        )

    def test_scf_convergence(self, mbis_result):
        sc = mbis_result["scf_convergence"]
        assert sc["energy_change"] == pytest.approx(9.3337e-09, rel=1e-3)
        assert sc["max_density_change"] == pytest.approx(1.4386e-02, rel=1e-3)

    def test_orbital_energies(self, mbis_result):
        assert mbis_result["homo_eh"] == pytest.approx(-0.329493, rel=1e-5)
        assert mbis_result["lumo_eh"] == pytest.approx(-0.022592, rel=1e-5)

    def test_mulliken_charges(self, mbis_result):
        mc = mbis_result["mulliken_charges"]
        assert len(mc) == 49
        assert mc["1_N"] == pytest.approx(-0.560555, rel=1e-5)

    def test_loewdin_charges(self, mbis_result):
        lc = mbis_result["loewdin_charges"]
        assert len(lc) == 49
        assert lc["1_N"] == pytest.approx(0.168557, rel=1e-5)

    def test_mayer_population(self, mbis_result):
        mp = mbis_result["mayer_population"]
        assert len(mp) == 49
        assert mp["1_N"]["va"] == pytest.approx(1.8457, rel=1e-4)
        assert mp["1_N"]["bva"] == pytest.approx(1.8457, rel=1e-4)

    def test_mayer_charges(self, mbis_result):
        mc = mbis_result["mayer_charges"]
        assert len(mc) == 49
        # QA column: "  0 N      7.5606     7.0000    -0.5606 ..."
        assert mc["1_N"] == pytest.approx(-0.5606, rel=1e-4)
        assert mc["2_O"] == pytest.approx(-0.6223, rel=1e-4)

    def test_mayer_bond_orders(self, mbis_result):
        mb = mbis_result["mayer_bond_orders"]
        assert mb["1_N_to_3_C"] == pytest.approx(1.5313, rel=1e-4)

    def test_dipole(self, mbis_result):
        assert mbis_result["dipole_au"] == pytest.approx(
            [0.508615974, 0.146156735, -1.868354990], rel=1e-8
        )
        assert mbis_result["dipole_magnitude_au"] == pytest.approx(
            1.941855393, rel=1e-8
        )

    def test_quadrupole(self, mbis_result):
        assert mbis_result["quadrupole_au"] == pytest.approx(
            [
                -101.956832521,
                -107.601885343,
                -132.499347966,
                6.456454883,
                1.201925623,
                2.017424645,
            ],
            rel=1e-8,
        )

    def test_hirshfeld_charges(self, mbis_result):
        hc = mbis_result["hirshfeld_charges"]
        assert len(hc) == 112
        assert hc["1_Ac"] == pytest.approx(0.187646, rel=1e-5)
        assert hc["112_H"] == pytest.approx(0.025076, rel=1e-5)

    def test_hirshfeld_spins(self, mbis_result):
        hs = mbis_result["hirshfeld_spins"]
        assert len(hs) == 112
        assert hs["1_Ac"] == pytest.approx(0.0, abs=1e-6)

    def test_mbis_charges(self, mbis_result):
        mc = mbis_result["mbis_charges"]
        assert len(mc) == 112
        assert mc["1_Ac"] == pytest.approx(2.439318, rel=1e-5)
        assert mc["2_C"] == pytest.approx(-1.111020, rel=1e-5)

    def test_mbis_populations(self, mbis_result):
        mp = mbis_result["mbis_populations"]
        assert len(mp) == 112
        assert mp["1_Ac"] == pytest.approx(26.560682, rel=1e-5)

    def test_mbis_spins(self, mbis_result):
        ms = mbis_result["mbis_spins"]
        assert len(ms) == 112
        assert ms["1_Ac"] == pytest.approx(0.0, abs=1e-6)

    def test_mbis_valence_populations(self, mbis_result):
        vp = mbis_result["mbis_valence_populations"]
        assert len(vp) == 112
        assert vp["1_Ac"] == pytest.approx(6.581989, rel=1e-5)

    def test_mbis_valence_widths(self, mbis_result):
        vw = mbis_result["mbis_valence_widths"]
        assert len(vw) == 112
        assert vw["1_Ac"] == pytest.approx(0.408576, rel=1e-5)


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


# ── Parametrized section presence across fixtures ────────────────────


@pytest.mark.parametrize(
    "fixture_path, expected_keys",
    [
        (
            FIXTURE_RKS,
            [
                "final_energy_eh",
                "scf_cycles",
                "energy_components",
                "scf_convergence",
                "homo_eh",
                "mulliken_charges",
                "loewdin_charges",
                "loewdin_bond_orders",
                "mayer_population",
                "mayer_charges",
                "mayer_bond_orders",
                "gradient",
                "gradient_norm",
                "gradient_rms",
                "gradient_max",
                "dipole_au",
                "quadrupole_au",
                "rotational_constants_cm1",
                "total_run_time_s",
            ],
        ),
        (
            FIXTURE_MBIS,
            [
                "final_energy_eh",
                "scf_convergence",
                "homo_eh",
                "mulliken_charges",
                "loewdin_charges",
                "mayer_charges",
                "hirshfeld_charges",
                "mbis_charges",
                "mbis_valence_populations",
            ],
        ),
        (
            FIXTURE_DUPLICATE,
            ["final_energy_eh", "energy_components", "scf_convergence", "mulliken_charges"],
        ),
        (
            FIXTURE_TRUNCATED,
            ["energy_components", "scf_convergence", "homo_eh"],
        ),
    ],
    ids=["rks", "mbis", "duplicate", "truncated"],
)
def test_section_presence(fixture_path, expected_keys):
    result = parse_orca_output(fixture_path)
    for key in expected_keys:
        assert key in result, f"Missing key {key!r} in parse result"


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
        _atomic_json_write(path, data)
        with open(path, "r") as f:
            loaded = json.load(f)
        assert loaded == data

    def test_overwrites_existing(self, tmp_job_dir):
        path = os.path.join(tmp_job_dir, "test.json")
        _atomic_json_write(path, {"old": True})
        _atomic_json_write(path, {"new": True})
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


class TestMergeOrcaIntoChargeJson:

    def test_merge_adds_orca_keys(self, tmp_job_dir, rks_result):
        charge_path = os.path.join(tmp_job_dir, "charge.json")
        existing = {"mulliken": {"charge": {"1_O": -0.5}}}
        with open(charge_path, "w") as f:
            json.dump(existing, f)

        merge_orca_into_charge_json(rks_result, charge_path)

        with open(charge_path, "r") as f:
            merged = json.load(f)

        # Original data preserved
        assert "mulliken" in merged
        # ORCA data added
        assert "mulliken_orca" in merged
        assert "loewdin_orca" in merged
        assert "mayer_orca" in merged
        assert merged["mulliken_orca"]["charge"]["1_O"] == pytest.approx(
            -0.740774, rel=1e-5
        )
        assert merged["mayer_orca"]["charge"]["1_O"] == pytest.approx(
            -0.7408, rel=1e-4
        )

    def test_merge_with_mbis(self, tmp_job_dir, mbis_result):
        charge_path = os.path.join(tmp_job_dir, "charge.json")
        with open(charge_path, "w") as f:
            json.dump({}, f)

        merge_orca_into_charge_json(mbis_result, charge_path)

        with open(charge_path, "r") as f:
            merged = json.load(f)

        assert "hirshfeld_orca" in merged
        assert "mbis_orca" in merged
        assert "mayer_orca" in merged
        assert merged["mayer_orca"]["charge"]["1_N"] == pytest.approx(
            -0.5606, rel=1e-4
        )
        assert merged["hirshfeld_orca"]["charge"]["1_Ac"] == pytest.approx(
            0.187646, rel=1e-5
        )
        assert merged["mbis_orca"]["charge"]["1_Ac"] == pytest.approx(
            2.439318, rel=1e-5
        )
        assert "spin" in merged["hirshfeld_orca"]
        assert "spin" in merged["mbis_orca"]
        assert "population" in merged["mbis_orca"]

    def test_idempotent(self, tmp_job_dir, rks_result):
        """Merging twice should produce same result."""
        charge_path = os.path.join(tmp_job_dir, "charge.json")
        with open(charge_path, "w") as f:
            json.dump({"original": True}, f)

        merge_orca_into_charge_json(rks_result, charge_path)
        with open(charge_path, "r") as f:
            first = json.load(f)

        merge_orca_into_charge_json(rks_result, charge_path)
        with open(charge_path, "r") as f:
            second = json.load(f)

        assert first == second

    def test_skips_missing_file(self, tmp_job_dir, rks_result):
        """Should not raise when charge.json doesn't exist."""
        charge_path = os.path.join(tmp_job_dir, "charge.json")
        merge_orca_into_charge_json(rks_result, charge_path)
        assert not os.path.isfile(charge_path)

    def test_skips_corrupt_json(self, tmp_job_dir, rks_result):
        """Should not raise when charge.json is corrupt."""
        charge_path = os.path.join(tmp_job_dir, "charge.json")
        with open(charge_path, "w") as f:
            f.write("{invalid json")
        merge_orca_into_charge_json(rks_result, charge_path)
        # File should remain unchanged (corrupt)
        with open(charge_path, "r") as f:
            assert f.read() == "{invalid json"


class TestMergeOrcaIntoBondJson:

    def test_merge_adds_orca_keys(self, tmp_job_dir, rks_result):
        bond_path = os.path.join(tmp_job_dir, "bond.json")
        existing = {"fuzzy": {"1_O_to_2_C": 1.5}}
        with open(bond_path, "w") as f:
            json.dump(existing, f)

        merge_orca_into_bond_json(rks_result, bond_path)

        with open(bond_path, "r") as f:
            merged = json.load(f)

        assert "fuzzy" in merged  # original preserved
        assert "mayer_orca" in merged
        assert "loewdin_orca" in merged
        assert merged["mayer_orca"]["1_O_to_2_C"] == pytest.approx(
            1.3593, rel=1e-4
        )

    def test_idempotent(self, tmp_job_dir, rks_result):
        bond_path = os.path.join(tmp_job_dir, "bond.json")
        with open(bond_path, "w") as f:
            json.dump({"original": True}, f)

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


# ── Optional full reference file tests ────────────────────────────────
# These parse real (large) ORCA output files from data/orca_outs_4_reference/.
# Skipped if the reference directory is not present.

REFERENCE_DIR = Path(__file__).parent.parent / "data" / "orca_outs_4_reference"

REFERENCE_FILES = {
    "orca6_rks": REFERENCE_DIR / "orca6_rks.out",
    "orca_mbis_nbo_act_ecp_1": REFERENCE_DIR / "orca_mbis_nbo_act_ecp_1.out",
    "orca_mbis_nbo_act_ecp_2": REFERENCE_DIR / "orca_mbis_nbo_act_ecp_2.out",
    "orca_mbis_nbo_non_act_1": REFERENCE_DIR / "orca_mbis_nbo_non_act_1.out",
    "orca_mbis_nbo_non_act_2": REFERENCE_DIR / "orca_mbis_nbo_non_act_2.out",
    "orca_spice_omol": REFERENCE_DIR / "orca_spice_omol.out",
}

_ref_available = REFERENCE_DIR.is_dir()


@pytest.mark.skipif(not _ref_available, reason="Reference files not present")
@pytest.mark.parametrize(
    "name, path",
    [(k, str(v)) for k, v in REFERENCE_FILES.items()],
    ids=list(REFERENCE_FILES.keys()),
)
class TestFullReferenceFiles:
    """Parse full (28-114MB) ORCA output files and validate structure."""

    def test_parses_without_error(self, name, path):
        if not os.path.isfile(path):
            pytest.skip(f"Reference file {path} not found")
        result = parse_orca_output(path)
        assert isinstance(result, dict)
        assert len(result) > 0, f"{name} produced empty dict"

    def test_has_final_energy(self, name, path):
        if not os.path.isfile(path):
            pytest.skip(f"Reference file {path} not found")
        result = parse_orca_output(path)
        assert "final_energy_eh" in result, f"{name} missing final_energy_eh"
        assert isinstance(result["final_energy_eh"], float)
        assert result["final_energy_eh"] < 0

    def test_has_core_sections(self, name, path):
        if not os.path.isfile(path):
            pytest.skip(f"Reference file {path} not found")
        result = parse_orca_output(path)
        # All reference files should have these core sections
        for key in ["scf_converged", "energy_components", "homo_eh",
                     "mulliken_charges", "loewdin_charges",
                     "dipole_au", "dipole_magnitude_au"]:
            assert key in result, f"{name} missing {key}"

    def test_charge_counts_consistent(self, name, path):
        if not os.path.isfile(path):
            pytest.skip(f"Reference file {path} not found")
        result = parse_orca_output(path)
        n_mulliken = len(result.get("mulliken_charges", {}))
        n_loewdin = len(result.get("loewdin_charges", {}))
        if n_mulliken > 0 and n_loewdin > 0:
            assert n_mulliken == n_loewdin, (
                f"{name}: Mulliken ({n_mulliken}) != Loewdin ({n_loewdin}) atom count"
            )

    def test_mbis_present_for_mbis_files(self, name, path):
        """Files with 'mbis' in the name should have MBIS sections."""
        if "mbis" not in name:
            pytest.skip("Not an MBIS file")
        if not os.path.isfile(path):
            pytest.skip(f"Reference file {path} not found")
        result = parse_orca_output(path)
        assert "mbis_charges" in result, f"{name} missing mbis_charges"
        assert "mbis_populations" in result, f"{name} missing mbis_populations"
        assert "hirshfeld_charges" in result, f"{name} missing hirshfeld_charges"
