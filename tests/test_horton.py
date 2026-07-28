"""Tests for the HORTON charge engine orchestrator (core/horton.py).

The compute worker itself lives in a separate python environment; tests here
cover everything that runs in the main env (EDF stripping, charge.json merge,
wfx discovery). An end-to-end worker test is gated on HORTON_PYTHON pointing
at the horton environment's interpreter.
"""

import json
import os
import shutil

import pytest

from qtaim_gen.source.core.horton import (
    find_wfx,
    merge_horton_into_charge_json,
    run_horton_analysis,
    strip_edf,
)

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


@pytest.mark.skipif(
    not (HORTON_PYTHON and os.path.isfile(HORTON_PYTHON)),
    reason="HORTON_PYTHON env var not set to the horton env interpreter",
)
@pytest.mark.skipif(
    not os.path.isfile(os.path.join(ECP_WFX_FOLDER, "orca.wfx")),
    reason="cross_validation_wfns ECP wfx fixture not available",
)
class TestWorkerEndToEnd:
    def test_ecp_folder(self, tmp_path):
        folder = tmp_path / "job"
        shutil.copytree(ECP_WFX_FOLDER, folder)
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
