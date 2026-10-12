"""--libreta_esp: QTAIM stdin prefix that makes Multiwfn evaluate the CP ESP with
LIBRETA, its plumbing from the runners to qtaim.txt, and the esp-code provenance
helper. Fixtures are Multiwfn 3.8 qtaim.out files of rmechdb_264_step6_0_2 run with
the production stdin and with the prefix. See docs/LIBRETA_ESP.md."""

import os
import shutil

import pytest

from qtaim_gen.source.data.multiwfn import LIBRETA_ESP_PREFIX, qtaim_data
from qtaim_gen.source.utils.validation import qtaim_esp_code, qtaim_run_status

FIXTURES = os.path.join(os.path.dirname(__file__), "test_files", "libreta_esp")
DEFAULT = "2\n2\n3\n4\n5\n8\n7\n0\n-10\nq\n"
EXHAUSTIVE = "2\n2\n3\n4\n5\n6\n-1\n-9\n8\n7\n0\n-10\nq\n"


def _read(name):
    with open(os.path.join(FIXTURES, name), encoding="latin-1") as f:
        return f.read()


class TestQtaimData:
    def test_off_by_default_is_the_production_stdin(self):
        assert qtaim_data() == DEFAULT
        assert qtaim_data(exhaustive=True) == EXHAUSTIVE

    @pytest.mark.parametrize("exhaustive,tail", [(False, DEFAULT), (True, EXHAUSTIVE)])
    def test_prefix_leaves_the_topology_stdin_unchanged(self, exhaustive, tail):
        assert qtaim_data(exhaustive=exhaustive, libreta_esp=True) == LIBRETA_ESP_PREFIX + tail

    def test_prefix_is_the_2x2_esp_plane_map(self):
        # main function 4, ESP (12), color-filled map, 2x2 grid, XY plane, Z=0, return
        assert LIBRETA_ESP_PREFIX == "4\n12\n1\n2,2\n1\n0\n-5\n"


class TestEspCode:
    def test_production_output_is_eleesp1(self):
        assert qtaim_esp_code(_read("qtaim_eleesp1.out")) == "eleesp1"

    def test_prefixed_output_is_libreta_fast(self):
        assert qtaim_esp_code(_read("qtaim_libreta_fast.out")) == "libreta_fast"

    def test_slow_version_and_missing_output(self):
        assert qtaim_esp_code(" Initializing LIBRETA library (slow version) for ESP evaluation ...") == "libreta_slow"
        assert qtaim_esp_code(None) is None


class TestPrefixedOutputStillValidates:
    """The restart gate and --check_bcp_count read qtaim.out; the plane map's
    output ahead of the topology menu must not change what they see."""

    def test_same_run_status(self, tmp_path):
        status = {}
        for name in ("qtaim_eleesp1.out", "qtaim_libreta_fast.out"):
            d = tmp_path / name.split(".")[0]
            d.mkdir()
            shutil.copy(os.path.join(FIXTURES, name), d / "qtaim.out")
            status[name] = qtaim_run_status(str(d))
        assert status["qtaim_libreta_fast.out"] == status["qtaim_eleesp1.out"]
        assert status["qtaim_eleesp1.out"]["export_done"] is True


class TestPlumbing:
    def _job(self, tmp_path):
        job = tmp_path / "job"
        job.mkdir()
        (job / "orca.gbw").write_text("")
        (job / "orca.inp").write_text("! B3LYP def2-SVP\n\n*xyz 0 1\nC 0.0 0.0 0.0\nH 1.0 0.0 0.0\n*\n")
        return job

    @pytest.mark.parametrize("flag", [False, True])
    def test_create_jobs_writes_the_prefix_only_under_the_flag(self, tmp_path, flag):
        from qtaim_gen.source.core.omol import create_jobs

        job = self._job(tmp_path)
        create_jobs(folder=str(job), multiwfn_cmd="Multiwfn", orca_2mkl_cmd="orca_2mkl",
                    separate=False, debug=True, libreta_esp=flag)
        assert (job / "qtaim.txt").read_text() == qtaim_data(libreta_esp=flag)

    def test_gbw_analysis_passes_the_flag(self, tmp_path, monkeypatch):
        from qtaim_gen.source.core import omol

        seen = {}
        monkeypatch.setattr(omol, "create_jobs", lambda *a, **k: seen.update(k))
        monkeypatch.setattr(omol, "run_jobs", lambda *a, **k: None)
        job = self._job(tmp_path)
        # validation fails on the empty folder, after create_jobs ran
        assert omol.gbw_analysis(str(job), multiwfn_cmd="x", orca_2mkl_cmd="y", move_results=False,
                                 libreta_esp=True) is False
        assert seen.get("libreta_esp") is True

    def test_process_folder_passes_the_flag(self, tmp_path, monkeypatch):
        from qtaim_gen.source.core import workflow

        seen = {}
        monkeypatch.setattr(workflow, "gbw_analysis", lambda *a, **k: seen.update(k) or True)
        monkeypatch.setattr(workflow, "validation_checks", lambda *a, **k: True)
        job = tmp_path / "job"
        job.mkdir()
        workflow.process_folder(str(job), move_results=False, libreta_esp=True)
        assert seen.get("libreta_esp") is True

    @pytest.mark.parametrize("module", ["full_runner_parsl", "full_runner_parsl_alcf"])
    def test_runner_cli_exposes_the_flag(self, module, capsys):
        import importlib

        main = importlib.import_module(f"qtaim_gen.source.scripts.{module}").main
        with pytest.raises(SystemExit):
            main(["--help"])
        assert "--libreta_esp" in capsys.readouterr().out
