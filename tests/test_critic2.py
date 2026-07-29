"""Tests for the Critic2 QTAIM bond-critical-point engine (core/critic2.py).

The parser tests run unconditionally against the committed cpreport JSON
fixture, so no critic2 binary is needed. Tests that shell out to critic2 are
gated on it being on PATH.
"""

import json
import os
import shutil

import numpy as np
import pytest

from qtaim_gen.source.core.critic2 import (
    BOHR_TO_ANG,
    CONVENTION_DIVERGENT,
    DEFAULT_POINTPROPS,
    POINTPROP_MAP,
    parse_critic2_cps,
    run_critic2_analysis,
    write_critic2_deck,
)

TEST_FILES = os.path.join(os.path.dirname(__file__), "test_files")
CPREPORT_FIXTURE = os.path.join(
    TEST_FILES, "critic2", "molecule", "phenol_phenol.json"
)
ECP_WFX_FOLDER = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "cross_validation_wfns",
    "wfx_pull",
    "rmechdb",
    "rmechdb_1463_step2_0_2",
)
AE_WFX_FOLDER = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "cross_validation_wfns",
    "wfx_pull",
    "rmechdb",
    "rmechdb_1056_step0_0_1",
)
HAS_CRITIC2 = shutil.which("critic2") is not None


class TestWriteDeck:
    def test_contains_required_commands(self):
        deck = write_critic2_deck("orca.wfx")
        assert "molecule orca.wfx" in deck
        assert "load orca.wfx id rho" in deck
        # reference must be explicit: field 0 is the promolecular density
        assert "reference rho" in deck
        assert "cpreport" in deck
        assert deck.rstrip().endswith("end")

    def test_discard_threshold_formatted(self):
        assert 'auto discard "$rho < 1e-05"' in write_critic2_deck(
            "m.wfx", discard=1e-5
        )
        assert 'auto discard "$rho < 0.001"' in write_critic2_deck(
            "m.wfx", discard=1e-3
        )

    def test_pointprops_emitted(self):
        deck = write_critic2_deck("m.wfx", pointprops=("gkin", "elf"))
        assert "pointprop gkin" in deck
        assert "pointprop elf" in deck
        assert "pointprop lol" not in deck

    def test_pointprops_precede_auto(self):
        """POINTPROP must be registered before AUTO evaluates the CPs."""
        deck = write_critic2_deck("m.wfx")
        assert deck.index("pointprop") < deck.index("auto ")

    def test_relative_paths_only(self):
        """Deck is run with cwd=job folder, so no absolute paths may leak in."""
        deck = write_critic2_deck("orca.wfx")
        assert "/" not in deck.replace("$rho", "")


class TestParseCpreportFixture:
    @pytest.fixture(scope="class")
    def parsed(self):
        return parse_critic2_cps(CPREPORT_FIXTURE)

    @pytest.fixture(scope="class")
    def raw(self):
        with open(CPREPORT_FIXTURE) as f:
            return json.load(f)

    def test_only_bcp_keys_plus_meta(self, parsed):
        for k in parsed:
            if k == "_meta":
                continue
            i, j = k.split("_")
            assert i.isdigit() and j.isdigit()

    def test_bcp_count_matches_signature_minus_one(self, parsed, raw):
        n_bcp = sum(
            1
            for c in raw["critical_points"]["nonequivalent_cps"]
            if c["signature"] == -1
        )
        # fixture is a well-behaved neutral molecule: every BCP resolves
        assert parsed["_meta"]["cp_counts"]["bond"] == n_bcp
        assert parsed["_meta"]["n_bcps_resolved"] == len(parsed) - 1

    def test_pair_keys_sorted_ascending(self, parsed):
        for k in parsed:
            if k == "_meta":
                continue
            i, j = (int(x) for x in k.split("_"))
            assert i < j

    def test_atom_indices_in_range(self, parsed):
        n_atoms = parsed["_meta"]["n_atoms"]
        for k in parsed:
            if k == "_meta":
                continue
            for idx in (int(x) for x in k.split("_")):
                assert 0 <= idx < n_atoms

    def test_derived_hessian_quantities(self, parsed):
        """lap/eig_hess/det/ellip/eta must follow from the eigenvalues."""
        for k, v in parsed.items():
            if k == "_meta":
                continue
            lam = v["hessian_eigenvalues"]
            assert v["eig_hess"] == pytest.approx(sum(lam))
            assert v["lap_e_density"] == pytest.approx(sum(lam), rel=1e-6)
            assert v["det_hessian"] == pytest.approx(lam[0] * lam[1] * lam[2])
            assert v["ellip_e_dens"] == pytest.approx(lam[0] / lam[1] - 1)
            assert v["eta"] == pytest.approx(abs(lam[0]) / lam[2])

    def test_bcp_is_a_density_saddle(self, parsed):
        """(3,-1): two negative curvatures, one positive."""
        for k, v in parsed.items():
            if k == "_meta":
                continue
            lam = v["hessian_eigenvalues"]
            assert lam[0] < 0 and lam[1] < 0 and lam[2] > 0

    def test_positions_converted_to_angstrom(self, parsed, raw):
        cv = raw["structure"]["molecule_centering_vector"]
        neq = {c["id"]: c for c in raw["critical_points"]["nonequivalent_cps"]}
        cell = raw["critical_points"]["cell_cps"]
        bcp = next(c for c in cell if c["signature"] == -1)
        expected = [
            (c + cv[i]) * BOHR_TO_ANG
            for i, c in enumerate(neq[bcp["nonequivalent_id"]]["cartesian_coordinates"])
        ]
        key = "_".join(
            str(x) for x in sorted(a["cell_id"] - 1 for a in bcp["attractors"])
        )
        assert parsed[key]["pos_ang"] == pytest.approx(expected)

    def test_bcp_lies_between_its_atoms(self, parsed, raw):
        """Sanity on the pair assignment: a BCP sits between the two atoms."""
        cv = raw["structure"]["molecule_centering_vector"]
        atoms = [
            [(c + cv[i]) * BOHR_TO_ANG for i, c in enumerate(a["cartesian_coordinates"])]
            for a in raw["structure"]["cell_atoms"]
        ]
        for k, v in parsed.items():
            if k == "_meta":
                continue
            i, j = (int(x) for x in k.split("_"))
            pos = np.array(v["pos_ang"])
            d_atoms = np.linalg.norm(np.array(atoms[i]) - np.array(atoms[j]))
            d_i = np.linalg.norm(pos - np.array(atoms[i]))
            d_j = np.linalg.norm(pos - np.array(atoms[j]))
            # allow bond-path curvature, but the BCP must not be far outside
            assert d_i + d_j < 1.5 * d_atoms + 0.5

    def test_meta_shape(self, parsed):
        meta = parsed["_meta"]
        assert meta["engine"] == "critic2"
        assert set(meta["cp_counts"]) == {"nucleus", "bond", "ring", "cage"}
        assert meta["source_units"] == "bohr"
        assert isinstance(meta["poincare_hopf_ok"], bool)
        assert meta["nna_remapped"] == []

    def test_poincare_hopf_from_counts(self, parsed):
        c = parsed["_meta"]["cp_counts"]
        assert parsed["_meta"]["poincare_hopf_sum"] == (
            c["nucleus"] - c["bond"] + c["ring"] - c["cage"]
        )
        # the fixture is a valid molecular topology
        assert parsed["_meta"]["poincare_hopf_ok"]

    def test_no_pointprops_in_fixture(self, parsed):
        """Fixture was produced without POINTPROP, so those fields are absent."""
        for k, v in parsed.items():
            if k == "_meta":
                continue
            for name in POINTPROP_MAP.values():
                assert name not in v


class TestNonNuclearAttractorRemap:
    """ECP jobs grow a phantom attractor; the BCP must survive, remapped."""

    def _fixture_with_phantom(self, tmp_path):
        with open(CPREPORT_FIXTURE) as f:
            data = json.load(f)
        neq = data["critical_points"]["nonequivalent_cps"]
        cell = data["critical_points"]["cell_cps"]
        n_atoms = len(data["structure"]["cell_atoms"])
        bcp = next(c for c in cell if c["signature"] == -1)
        real_pair = sorted(a["cell_id"] - 1 for a in bcp["attractors"])
        # invent a phantom nuclear CP sitting near the first attractor atom
        atom = data["structure"]["cell_atoms"][real_pair[0]]
        phantom_id = max(c["id"] for c in neq) + 1
        phantom = dict(neq[0])
        phantom["id"] = phantom_id
        phantom["signature"] = -3
        phantom["is_nucleus"] = False
        phantom["cartesian_coordinates"] = [
            c + 0.05 for c in atom["cartesian_coordinates"]
        ]
        neq.append(phantom)
        # point one attractor of the BCP at the phantom instead of the atom
        bcp["attractors"][0]["cell_id"] = phantom_id
        cell.append(
            {
                "id": phantom_id,
                "rank": 3,
                "signature": -3,
                "nonequivalent_id": phantom_id,
                "cartesian_coordinates": phantom["cartesian_coordinates"],
                "attractors": [],
            }
        )
        path = tmp_path / "phantom.json"
        path.write_text(json.dumps(data))
        return path, real_pair, n_atoms

    def test_bcp_kept_and_pair_recovered(self, tmp_path):
        path, real_pair, _ = self._fixture_with_phantom(tmp_path)
        parsed = parse_critic2_cps(str(path))
        key = f"{real_pair[0]}_{real_pair[1]}"
        assert key in parsed, "remap must recover the physical pair, not drop the BCP"

    def test_remap_recorded_in_meta(self, tmp_path):
        path, real_pair, _ = self._fixture_with_phantom(tmp_path)
        meta = parse_critic2_cps(str(path))["_meta"]
        assert len(meta["nna_remapped"]) == 1
        entry = meta["nna_remapped"][0]
        assert entry["mapped_to_atom"] == real_pair[0]
        assert entry["distance_ang"] < 0.1

    def test_atom_indices_still_valid_after_remap(self, tmp_path):
        path, _, n_atoms = self._fixture_with_phantom(tmp_path)
        parsed = parse_critic2_cps(str(path))
        for k in parsed:
            if k == "_meta":
                continue
            for idx in (int(x) for x in k.split("_")):
                assert idx < n_atoms


class TestPairValidation:
    """Critic2's gradient-path terminator can be wrong in metal/ECP systems.

    Regression: a Dy/La cluster reported a BCP as La-O at 3.47 A with
    rho = 0.69 when it was really the C-O bond CP (0.45 A from O, 0.76 A from
    C, with the La 3.05 A away).
    """

    def _redirect_attractor(self, tmp_path, target_offset=0):
        """Point one attractor of one BCP at a deliberately distant atom."""
        with open(CPREPORT_FIXTURE) as f:
            data = json.load(f)
        cv = data["structure"]["molecule_centering_vector"]
        atoms = np.array(
            [
                [(c + cv[i]) * BOHR_TO_ANG for i, c in enumerate(a["cartesian_coordinates"])]
                for a in data["structure"]["cell_atoms"]
            ]
        )
        neq = {c["id"]: c for c in data["critical_points"]["nonequivalent_cps"]}
        bcp = next(
            c for c in data["critical_points"]["cell_cps"] if c["signature"] == -1
        )
        pos = np.array(
            [
                (c + cv[i]) * BOHR_TO_ANG
                for i, c in enumerate(neq[bcp["nonequivalent_id"]]["cartesian_coordinates"])
            ]
        )
        d = np.linalg.norm(atoms - pos, axis=1)
        true_pair = sorted(a["cell_id"] - 1 for a in bcp["attractors"])
        farthest = int(np.argsort(d)[-1 - target_offset])
        bcp["attractors"][0]["cell_id"] = farthest + 1
        path = tmp_path / "redirected.json"
        path.write_text(json.dumps(data))
        return path, true_pair, farthest

    def test_implausible_pair_is_corrected(self, tmp_path):
        path, true_pair, farthest = self._redirect_attractor(tmp_path)
        parsed = parse_critic2_cps(str(path))
        key = f"{true_pair[0]}_{true_pair[1]}"
        assert key in parsed, "correction should restore the geometric pair"
        assert f"{min(true_pair[1], farthest)}_{max(true_pair[1], farthest)}" not in parsed

    def test_correction_recorded_in_meta(self, tmp_path):
        path, true_pair, _ = self._redirect_attractor(tmp_path)
        corrections = parse_critic2_cps(str(path))["_meta"]["pair_corrections"]
        assert len(corrections) == 1
        c = corrections[0]
        assert c["corrected_pair"] == true_pair
        assert max(c["claimed_distances_ang"]) > max(c["nearest_distances_ang"])

    def test_validation_can_be_disabled(self, tmp_path):
        path, true_pair, farthest = self._redirect_attractor(tmp_path)
        parsed = parse_critic2_cps(str(path), validate_pairs=False)
        assert parsed["_meta"]["pair_corrections"] == []
        bogus = sorted([true_pair[1], farthest])
        assert f"{bogus[0]}_{bogus[1]}" in parsed

    def test_untouched_fixture_needs_no_corrections(self):
        """A well-behaved all-electron molecule must not trip the heuristic."""
        parsed = parse_critic2_cps(CPREPORT_FIXTURE)
        assert parsed["_meta"]["pair_corrections"] == []


class TestConstants:
    def test_pointprop_map_covers_defaults(self):
        assert set(DEFAULT_POINTPROPS) == set(POINTPROP_MAP)

    def test_convention_divergent_are_mapped_fields(self):
        assert set(CONVENTION_DIVERGENT) <= set(POINTPROP_MAP.values())


@pytest.mark.critic2
@pytest.mark.skipif(not HAS_CRITIC2, reason="critic2 not on PATH")
class TestEndToEnd:
    @staticmethod
    def _fresh_job(tmp_path, src):
        folder = tmp_path / "job"
        folder.mkdir()
        shutil.copy(os.path.join(src, "orca.wfx"), folder)
        return folder

    @pytest.mark.skipif(
        not os.path.isfile(os.path.join(AE_WFX_FOLDER, "orca.wfx")),
        reason="all-electron wfx fixture not available",
    )
    def test_all_electron_job(self, tmp_path):
        folder = self._fresh_job(tmp_path, AE_WFX_FOLDER)
        assert run_critic2_analysis(str(folder))
        result = json.loads((folder / "critic2.json").read_text())
        # Cl2: two nuclei, one bond, healthy topology
        assert result["_meta"]["poincare_hopf_ok"]
        assert result["_meta"]["nna_remapped"] == []
        assert "0_1" in result
        for name in POINTPROP_MAP.values():
            assert name in result["0_1"]

    @pytest.mark.skipif(
        not os.path.isfile(os.path.join(ECP_WFX_FOLDER, "orca.wfx")),
        reason="ECP wfx fixture not available",
    )
    def test_ecp_job_remaps_phantom_attractor(self, tmp_path):
        """Critic2 ignores wfx EDF core density, so an ECP atom grows a phantom
        attractor. The BCP set must still come out physical."""
        folder = self._fresh_job(tmp_path, ECP_WFX_FOLDER)
        assert run_critic2_analysis(str(folder))
        result = json.loads((folder / "critic2.json").read_text())
        assert len(result["_meta"]["nna_remapped"]) >= 1
        bcps = {k for k in result if k != "_meta"}
        assert bcps == {"0_1", "1_3", "2_3"}

    def test_skip_if_done(self, tmp_path):
        folder = self._fresh_job(tmp_path, AE_WFX_FOLDER)
        assert run_critic2_analysis(str(folder))
        mtime = (folder / "critic2.json").stat().st_mtime_ns
        assert run_critic2_analysis(str(folder))
        assert (folder / "critic2.json").stat().st_mtime_ns == mtime

    def test_missing_wfx_returns_false(self, tmp_path):
        assert run_critic2_analysis(str(tmp_path)) is False

    def test_clean_removes_intermediates(self, tmp_path):
        folder = self._fresh_job(tmp_path, AE_WFX_FOLDER)
        assert run_critic2_analysis(str(folder), keep_intermediates=False)
        assert (folder / "critic2.json").exists()
        assert not (folder / "critic2_run.cri").exists()
        assert not (folder / "critic2_cps.json").exists()
