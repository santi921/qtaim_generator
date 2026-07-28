"""Tests for the HORTON charge engine orchestrator (core/horton.py).

The compute worker itself lives in a separate python environment; tests here
cover everything that runs in the main env (EDF stripping, charge.json merge,
wfx discovery). An end-to-end worker test is gated on HORTON_PYTHON pointing
at the horton environment's interpreter.
"""

import importlib.util
import json
import os
import shutil

import pytest

from qtaim_gen.source.core.horton import (
    find_horton_json,
    find_wfx,
    merge_horton_into_charge_json,
    resolve_charge_json,
    run_horton_analysis,
    strip_edf,
)

# The worker runs in a separate env, but its module-level tables import with
# stdlib + numpy only, so they are testable here.
_WORKER_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "qtaim_gen",
    "source",
    "scripts",
    "helpers",
    "horton_worker.py",
)
_spec = importlib.util.spec_from_file_location("horton_worker", _WORKER_PATH)
horton_worker = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(horton_worker)

TEST_FILES = os.path.join(os.path.dirname(__file__), "test_files")
HORTON_FIXTURE = os.path.join(TEST_FILES, "horton", "horton.json")
CHARGE_FIXTURE = os.path.join(TEST_FILES, "lmdb_tests", "orca6_rks", "charge.json")

HORTON_PYTHON = os.environ.get("HORTON_PYTHON", "")
ECP_WFX_FOLDER = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "cross_validation_wfns",
    "wfx_pull",
    "rmechdb",
    "rmechdb_1463_step2_0_2",
)

WFX_WITH_EDF = """<Number of Nuclei>
 4
</Number of Nuclei>
<Primitive Exponents>
  0.1 0.2
</Primitive Exponents>
<Additional Electron Density Function (EDF)>
<Number of EDF Primitives>
    30
</Number of EDF Primitives>
<EDF Primitive Coefficients>
  1.0 2.0
</EDF Primitive Coefficients>
</Additional Electron Density Function (EDF)>
<Molecular Orbital Occupation Numbers>
  2.0
</Molecular Orbital Occupation Numbers>
"""


class TestStripEdf:
    def test_removes_block_and_trailing_newline(self):
        stripped = strip_edf(WFX_WITH_EDF)
        assert "EDF" not in stripped
        # no blank line left at the splice point
        assert "</Primitive Exponents>\n<Molecular Orbital" in stripped

    def test_preserves_non_edf_content(self):
        stripped = strip_edf(WFX_WITH_EDF)
        assert "<Number of Nuclei>" in stripped
        assert "<Molecular Orbital Occupation Numbers>" in stripped

    def test_no_edf_returns_same_object(self):
        text = "<Number of Nuclei>\n 4\n</Number of Nuclei>\n"
        assert strip_edf(text) is text


class TestFindWfx:
    def test_root_preferred_over_generator(self, tmp_path):
        os.makedirs(tmp_path / "generator")
        (tmp_path / "orca.wfx").write_text("root")
        (tmp_path / "generator" / "orca.wfx").write_text("gen")
        assert find_wfx(str(tmp_path)) == str(tmp_path / "orca.wfx")

    def test_generator_fallback(self, tmp_path):
        os.makedirs(tmp_path / "generator")
        (tmp_path / "generator" / "orca.wfx").write_text("gen")
        assert find_wfx(str(tmp_path)) == str(tmp_path / "generator" / "orca.wfx")

    def test_missing_returns_none(self, tmp_path):
        assert find_wfx(str(tmp_path)) is None

    def test_empty_file_ignored(self, tmp_path):
        (tmp_path / "orca.wfx").write_text("")
        assert find_wfx(str(tmp_path)) is None


class TestMergeHortonIntoChargeJson:
    @pytest.fixture
    def charge_json(self, tmp_path):
        dest = tmp_path / "charge.json"
        shutil.copy(CHARGE_FIXTURE, dest)
        return str(dest)

    @pytest.fixture
    def horton_dict(self):
        with open(HORTON_FIXTURE) as f:
            return json.load(f)

    def test_adds_horton_keys(self, charge_json, horton_dict):
        merge_horton_into_charge_json(horton_dict, charge_json)
        with open(charge_json) as f:
            merged = json.load(f)
        assert "becke_horton" in merged
        assert "hirshfeld_horton" in merged
        assert "is_horton" in merged

    def test_meta_not_merged(self, charge_json, horton_dict):
        merge_horton_into_charge_json(horton_dict, charge_json)
        with open(charge_json) as f:
            merged = json.load(f)
        assert "_meta" not in merged

    def test_preserves_multiwfn_values(self, charge_json, horton_dict):
        with open(charge_json) as f:
            before = json.load(f)
        merge_horton_into_charge_json(horton_dict, charge_json)
        with open(charge_json) as f:
            merged = json.load(f)
        for scheme in before:
            assert merged[scheme] == before[scheme]

    def test_idempotent(self, charge_json, horton_dict):
        merge_horton_into_charge_json(horton_dict, charge_json)
        with open(charge_json) as f:
            first = json.load(f)
        merge_horton_into_charge_json(horton_dict, charge_json)
        with open(charge_json) as f:
            second = json.load(f)
        assert first == second

    def test_skips_absent_charge_json(self, tmp_path, horton_dict):
        target = str(tmp_path / "charge.json")
        merge_horton_into_charge_json(horton_dict, target)
        assert not os.path.exists(target)

    def test_charge_values_match_fixture(self, charge_json, horton_dict):
        merge_horton_into_charge_json(horton_dict, charge_json)
        with open(charge_json) as f:
            merged = json.load(f)
        assert (
            merged["becke_horton"]["charge"]
            == horton_dict["becke_horton"]["charge"]
        )


class TestFindHortonJson:
    def test_root_preferred(self, tmp_path):
        os.makedirs(tmp_path / "generator")
        (tmp_path / "horton.json").write_text("{}x")
        (tmp_path / "generator" / "horton.json").write_text("{}y")
        assert find_horton_json(str(tmp_path)) == str(tmp_path / "horton.json")

    def test_generator_fallback(self, tmp_path):
        """After move_results, horton.json lives in generator/."""
        os.makedirs(tmp_path / "generator")
        (tmp_path / "generator" / "horton.json").write_text("{}y")
        assert find_horton_json(str(tmp_path)) == str(
            tmp_path / "generator" / "horton.json"
        )

    def test_missing_and_empty(self, tmp_path):
        assert find_horton_json(str(tmp_path)) is None
        (tmp_path / "horton.json").write_text("")
        assert find_horton_json(str(tmp_path)) is None


class TestResolveChargeJson:
    def test_root_when_not_moved(self, tmp_path):
        assert resolve_charge_json(str(tmp_path), False) == str(
            tmp_path / "charge.json"
        )

    def test_generator_when_moved_and_present(self, tmp_path):
        os.makedirs(tmp_path / "generator")
        (tmp_path / "generator" / "charge.json").write_text("{}")
        assert resolve_charge_json(str(tmp_path), True) == str(
            tmp_path / "generator" / "charge.json"
        )

    def test_root_when_moved_but_absent(self, tmp_path):
        assert resolve_charge_json(str(tmp_path), True) == str(
            tmp_path / "charge.json"
        )


class TestSkipPathStillMerges:
    """A pre-existing horton.json must not prevent the charge.json merge.

    Regression: an interrupted run (or a charge.json produced after the HORTON
    run) would otherwise leave the *_horton keys permanently unmerged.
    """

    @pytest.fixture
    def folder(self, tmp_path):
        shutil.copy(HORTON_FIXTURE, tmp_path / "horton.json")
        shutil.copy(CHARGE_FIXTURE, tmp_path / "charge.json")
        return str(tmp_path)

    def test_merges_without_running_worker(self, folder):
        # bogus interpreter: proves no subprocess is spawned on this path
        assert run_horton_analysis(folder, "/nonexistent/python") is True
        with open(os.path.join(folder, "charge.json")) as f:
            merged = json.load(f)
        assert "becke_horton" in merged
        assert "hirshfeld_horton" in merged

    def test_idempotent_on_repeat(self, folder):
        run_horton_analysis(folder, "/nonexistent/python")
        with open(os.path.join(folder, "charge.json")) as f:
            first = json.load(f)
        run_horton_analysis(folder, "/nonexistent/python")
        with open(os.path.join(folder, "charge.json")) as f:
            assert json.load(f) == first

    def test_no_charge_json_is_not_an_error(self, tmp_path):
        shutil.copy(HORTON_FIXTURE, tmp_path / "horton.json")
        assert run_horton_analysis(str(tmp_path), "/nonexistent/python") is True
        assert not os.path.exists(tmp_path / "charge.json")

    def test_corrupt_horton_json_reports_failure(self, tmp_path):
        (tmp_path / "horton.json").write_text("{not json")
        assert run_horton_analysis(str(tmp_path), "/nonexistent/python") is False


class TestWorkerTables:
    """The radii/multiplicity tables are hand-transcribed; guard the values.

    A typo here yields plausible-but-wrong charges rather than an error, so
    these spot checks matter more than usual.
    """

    def test_radii_cover_h_through_lr(self):
        missing = [z for z in range(1, 104) if z not in horton_worker.COVR_TIANLU]
        assert missing == []

    def test_radii_spot_values(self):
        # from Multiwfn's covr_tianlu (Angstrom)
        expected = {
            1: 0.31, 2: 0.28, 6: 0.76, 15: 1.11, 26: 1.32, 53: 1.39,
            71: 1.87, 79: 1.36, 92: 1.96, 96: 1.69,
        }
        for z, radius in expected.items():
            assert horton_worker.COVR_TIANLU[z] == pytest.approx(radius)

    def test_radii_row_uniformity(self):
        """Main-group rows take the group-IVA radius (except H/He)."""
        table = horton_worker.COVR_TIANLU
        assert len({table[z] for z in range(3, 11)}) == 1  # Li-Ne
        assert len({table[z] for z in range(11, 19)}) == 1  # Na-Ar
        assert len({table[z] for z in range(31, 37)}) == 1  # Ga-Kr
        assert len({table[z] for z in range(49, 55)}) == 1  # In-Xe

    def test_transition_metals_not_uniform(self):
        """TMs carry individual radii - catches an over-broad range fill."""
        assert len({horton_worker.COVR_TIANLU[z] for z in range(21, 31)}) > 5

    def test_beyond_cm_uses_multiwfn_default(self):
        for z in range(97, 104):
            assert horton_worker.COVR_TIANLU[z] == pytest.approx(
                horton_worker.COVR_TIANLU_DEFAULT
            )

    def test_multiplicities_cover_h_through_lr(self):
        missing = [
            z for z in range(1, 104) if z not in horton_worker.GROUND_STATE_MULT
        ]
        assert missing == []

    def test_multiplicity_spot_values(self):
        # ground-state multiplicities; these must match qc-AtomDB's dataset
        expected = {
            1: 2, 6: 3, 8: 3, 24: 7, 26: 5, 36: 1, 54: 1, 64: 9, 79: 2,
            92: 5, 95: 8, 102: 1,
        }
        for z, mult in expected.items():
            assert horton_worker.GROUND_STATE_MULT[z] == mult

    def test_schemes_registry(self):
        assert set(horton_worker.SCHEMES) == {
            "becke",
            "becke_csd",
            "hirshfeld",
            "is",
        }


@pytest.mark.skipif(
    not (HORTON_PYTHON and os.path.isfile(HORTON_PYTHON)),
    reason="HORTON_PYTHON env var not set to the horton env interpreter",
)
@pytest.mark.skipif(
    not os.path.isfile(os.path.join(ECP_WFX_FOLDER, "orca.wfx")),
    reason="cross_validation_wfns ECP wfx fixture not available",
)
class TestWorkerEndToEnd:
    @staticmethod
    def _fresh_job(tmp_path):
        """Copy the wfx fixture without outputs a prior local run may have left.

        The source folder is a real data directory, so horton.json may already
        be present there; keeping it would send these tests down the
        merge-only skip path instead of exercising the worker.
        """
        folder = tmp_path / "job"
        shutil.copytree(
            ECP_WFX_FOLDER, folder, ignore=shutil.ignore_patterns("horton*.json")
        )
        return folder

    def test_ecp_folder(self, tmp_path):
        folder = self._fresh_job(tmp_path)
        ok = run_horton_analysis(str(folder), HORTON_PYTHON)
        assert ok
        with open(folder / "horton.json") as f:
            result = json.load(f)
        assert "becke_horton" in result
        assert "is_horton" in result
        # ECP job: hirshfeld must be skipped, not silently wrong
        assert "hirshfeld_horton" not in result
        skipped = [s["scheme"] for s in result["_meta"]["schemes_skipped"]]
        assert "hirshfeld" in skipped
        # neutral radical: charges sum to ~0
        total = sum(result["becke_horton"]["charge"].values())
        assert abs(total) < 0.02

    def test_becke_csd_matches_multiwfn_radii_convention(self, tmp_path):
        """becke_csd must reproduce Multiwfn's Becke charges closely.

        Native HORTON Becke (Bragg-Slater radii, 0.45 clip) differs from
        Multiwfn by ~0.2 e; with the modified-CSD radii and 0.5 clip the two
        codes agree to ~0.01 e, so a loose bound here still catches a broken
        radii table or a dropped clip patch.
        """
        folder = self._fresh_job(tmp_path)
        ok = run_horton_analysis(
            str(folder), HORTON_PYTHON, schemes="becke,becke_csd"
        )
        assert ok
        with open(folder / "horton.json") as f:
            result = json.load(f)
        native = result["becke_horton"]["charge"]
        csd = result["becke_csd_horton"]["charge"]
        assert set(native) == set(csd)
        # the two radii conventions must actually produce different numbers
        assert max(abs(csd[k] - native[k]) for k in csd) > 0.01
        assert abs(sum(csd.values())) < 0.02

    def test_unknown_scheme_is_rejected(self, tmp_path):
        folder = self._fresh_job(tmp_path)
        assert (
            run_horton_analysis(str(folder), HORTON_PYTHON, schemes="not_a_scheme")
            is False
        )
        assert not (folder / "horton.json").exists()

    def test_surviving_schemes_are_kept_when_one_is_skipped(self, tmp_path):
        """hirshfeld skips on this ECP job; becke/is must still be written."""
        folder = self._fresh_job(tmp_path)
        assert run_horton_analysis(
            str(folder), HORTON_PYTHON, schemes="hirshfeld,becke"
        )
        with open(folder / "horton.json") as f:
            result = json.load(f)
        assert "becke_horton" in result
        assert "hirshfeld_horton" not in result
