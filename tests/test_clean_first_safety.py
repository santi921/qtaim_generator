"""--clean_first never destroys completed results; --check_ecp only queues proven ECP failures."""

import importlib
import json
import zipfile

import pytest

from qtaim_gen.source.core import workflow
from qtaim_gen.source.utils import io as qio

_INP_LIGHT = "! wB97M-V def2-TZVPD\n*xyz 0 1\nC 0.0 0.0 0.0\nO 0.0 0.0 1.2\n*\n"
_INP_ECP = "! wB97M-V def2-TZVPD\n*xyz 0 1\nH 0.0 0.0 0.0\nI 0.0 0.0 1.6\n*\n"
_INP_EMPTY = "! wB97M-V def2-TZVPD\n*xyz 0 1\n*\n"
_EDF_LINE = " Loading EDF library finished!\n"
_LOG = workflow.logging.getLogger("t")


def _results_folder(path, inp=_INP_LIGHT):
    path.mkdir(parents=True)
    gen = path / "generator"
    gen.mkdir()
    (gen / "charge.json").write_text(json.dumps({"hirshfeld": {"charge": {"1_C": 0.1}}}))
    (gen / "timings.json").write_text(json.dumps({"hirshfeld": 3.0}))
    (path / "orca.inp").write_text(inp)
    (path / "hirshfeld.out").write_text("scratch\n")
    (path / "props_hirshfeld.mfwn").write_text("scratch\n")
    (path / "scratch_dir").mkdir()
    (path / "gbw_analysis.log").write_text("old log\n")
    return path


def _names(path):
    return sorted(p.name for p in path.iterdir())


class TestCleanFirstHelper:
    def test_moves_generator_aside_and_keeps_log_and_lock(self, tmp_path):
        job = _results_folder(tmp_path / "job")
        (job / ".processing.lock").write_text("123")
        before = {p.name: p.read_bytes() for p in (job / "generator").iterdir()}
        workflow._clean_first(str(job), _LOG)
        assert _names(job) == [".processing.lock", "gbw_analysis.log", workflow._STASH]
        assert {p.name: p.read_bytes() for p in (job / workflow._STASH).iterdir()} == before

    def test_keep_inputs_keeps_orca_files(self, tmp_path):
        job = _results_folder(tmp_path / "job")
        for name in ("orca.gbw.zstd0", "orca.tar.zst", "orca.out", "orca.wfx", "adch.out"):
            (job / name).write_text("x")
        workflow._clean_first(str(job), _LOG, keep_inputs=True)
        assert _names(job) == [
            "gbw_analysis.log", workflow._STASH, "orca.gbw.zstd0", "orca.inp", "orca.out", "orca.tar.zst",
        ]


class TestSettleStash:
    def _stashed(self, tmp_path):
        job = _results_folder(tmp_path / "job")
        workflow._clean_first(str(job), _LOG)
        (job / "generator").mkdir()
        (job / "generator" / "charge.json").write_text(json.dumps({"hirshfeld": "new"}))
        return job

    def test_success_drops_stash(self, tmp_path):
        job = self._stashed(tmp_path)
        workflow._settle_stash(str(job), keep_new=True, logger=_LOG)
        assert not (job / workflow._STASH).exists()
        assert json.loads((job / "generator" / "charge.json").read_text()) == {"hirshfeld": "new"}

    def test_success_carries_forward_what_the_rerun_did_not_compute(self, tmp_path):
        job = _results_folder(tmp_path / "job")
        old = job / "generator"
        (old / "fuzzy_full.json").write_text(json.dumps({"becke_fuzzy_density": "old", "mbis_fuzzy_spin": "L1"}))
        (old / "timings.json").write_text(json.dumps({"becke_fuzzy_density": 1.0, "mbis_fuzzy_spin": 2.0}))
        (old / "qtaim.json").write_text(json.dumps({"0": {}, "4_31": {"stale": 1}}))
        (old / "horton.json").write_text(json.dumps({"mbis": 1}))
        (old / "other.json").write_text("{not json")
        with zipfile.ZipFile(old / "out_files.zip", "w") as zf:
            zf.writestr("becke_fuzzy_density.out", "old")
            zf.writestr("mbis_fuzzy_spin.out", "L1")
        workflow._clean_first(str(job), _LOG)
        new = job / "generator"
        new.mkdir()
        (new / "fuzzy_full.json").write_text(json.dumps({"becke_fuzzy_density": "new"}))
        (new / "timings.json").write_text(json.dumps({"becke_fuzzy_density": 9.0}))
        (new / "qtaim.json").write_text(json.dumps({"0": {}}))
        (new / "other.json").write_text(json.dumps({"ALIE_Volume": 1.0}))
        with zipfile.ZipFile(new / "out_files.zip", "w") as zf:
            zf.writestr("becke_fuzzy_density.out", "new")

        workflow._settle_stash(str(job), keep_new=True, logger=_LOG)
        load = lambda n: json.loads((new / n).read_text())  # noqa: E731
        assert load("fuzzy_full.json") == {"becke_fuzzy_density": "new", "mbis_fuzzy_spin": "L1"}
        assert load("timings.json") == {"becke_fuzzy_density": 9.0, "mbis_fuzzy_spin": 2.0}
        assert load("qtaim.json") == {"0": {}}
        assert load("horton.json") == {"mbis": 1}
        assert load("charge.json") == {"hirshfeld": {"charge": {"1_C": 0.1}}}
        assert load("other.json") == {"ALIE_Volume": 1.0}
        with zipfile.ZipFile(new / "out_files.zip") as zf:
            assert {n: zf.read(n).decode() for n in zf.namelist()} == {
                "becke_fuzzy_density.out": "new", "mbis_fuzzy_spin.out": "L1",
            }
        assert not (new / "out_files.zip.carry").exists()
        assert not (job / workflow._STASH).exists()

    def test_failure_restores_stash_over_partial(self, tmp_path):
        job = self._stashed(tmp_path)
        workflow._settle_stash(str(job), keep_new=False, logger=_LOG)
        assert not (job / workflow._STASH).exists()
        assert _names(job / "generator") == ["charge.json", "timings.json"]
        assert "1_C" in (job / "generator" / "charge.json").read_text()

    def test_interrupted_restore_is_finished(self, tmp_path):
        job = self._stashed(tmp_path)
        # killed after the partial generator/ was renamed away, before the stash came back
        (job / "generator").rename(job / workflow._STASH_FAILED)
        workflow._settle_stash(str(job), keep_new=False, logger=_LOG)
        assert "1_C" in (job / "generator" / "charge.json").read_text()
        assert not (job / workflow._STASH_FAILED).exists()

    def test_interrupted_drop_never_restores_old_results(self, tmp_path):
        job = self._stashed(tmp_path)
        # killed mid-rmtree of a validated rerun's stash
        (job / workflow._STASH).rename(job / workflow._STASH_DONE)
        workflow._settle_stash(str(job), keep_new=False, logger=_LOG)
        assert not (job / workflow._STASH_DONE).exists()
        assert json.loads((job / "generator" / "charge.json").read_text()) == {"hirshfeld": "new"}


class TestCleanFirstRunners:
    def test_alcf_failed_rerun_keeps_generator(self, tmp_path, monkeypatch):
        inputs = tmp_path / "in"
        (inputs / "v" / "job").mkdir(parents=True)
        (inputs / "v" / "job" / "orca.inp").write_text(_INP_LIGHT)
        (inputs / "v" / "job" / "orca.gbw.zstd0").write_bytes(b"gbw")
        job = _results_folder(tmp_path / "res" / "v" / "job")
        charge_before = (job / "generator" / "charge.json").read_text()
        calls = {}

        def fake_gbw(**kwargs):
            calls.update(kwargs)
            assert not (job / "generator").exists(), "rerun must start from an empty generator/"
            raise RuntimeError("walltime")

        # validation passing would skip the folder unless clean_first forces overwrite
        monkeypatch.setattr(workflow, "validation_checks", lambda *a, **k: True)
        monkeypatch.setattr(workflow, "gbw_analysis", fake_gbw)
        result = workflow.process_folder_alcf(
            str(inputs / "v" / "job"),
            root_omol_inputs=str(inputs),
            root_omol_results=str(tmp_path / "res"),
            clean_first=True,
            restart=True,
            move_results=True,
        )
        assert result["status"] == "error"
        assert calls["overwrite"] is True and calls["restart"] is False
        assert (job / "generator" / "charge.json").read_text() == charge_before
        assert not (job / workflow._STASH).exists()
        assert not (job / "hirshfeld.out").exists() and not (job / "scratch_dir").exists()
        # inputs were copied back in fresh
        assert (job / "orca.gbw.zstd0").read_bytes() == b"gbw"

    def test_local_runner_failed_rerun_keeps_generator_and_inputs(self, tmp_path, monkeypatch):
        job = _results_folder(tmp_path / "job")
        (job / "orca.gbw.zstd0").write_bytes(b"gbw")
        charge_before = (job / "generator" / "charge.json").read_text()
        calls = {}

        def fake_gbw(**kwargs):
            calls.update(kwargs)
            return False

        monkeypatch.setattr(workflow, "gbw_analysis", fake_gbw)
        workflow.process_folder(str(job), clean_first=True, restart=True, move_results=True)
        assert calls["overwrite"] is True and calls["restart"] is False
        assert (job / "generator" / "charge.json").read_text() == charge_before
        assert not (job / workflow._STASH).exists()
        assert (job / "orca.inp").read_text() == _INP_LIGHT
        assert (job / "orca.gbw.zstd0").read_bytes() == b"gbw"

    def test_validated_rerun_replaces_generator_whole(self, tmp_path, monkeypatch):
        job = _results_folder(tmp_path / "job")
        (job / "generator" / "qtaim.json").write_text(json.dumps({"0": {}, "4_31": {"stale": 1}}))

        def fake_gbw(**kwargs):
            gen = job / "generator"
            assert not gen.exists(), "rerun must start from an empty generator/"
            gen.mkdir()
            (gen / "qtaim.json").write_text(json.dumps({"0": {}}))
            return True

        monkeypatch.setattr(workflow, "gbw_analysis", fake_gbw)
        workflow.process_folder(str(job), clean_first=True, move_results=True)
        assert json.loads((job / "generator" / "qtaim.json").read_text()) == {"0": {}}
        # charge/timings were not rewritten by the fake rerun, so they carry forward whole
        assert _names(job / "generator") == ["charge.json", "qtaim.json", "timings.json"]
        assert not (job / workflow._STASH).exists()

    def test_unvalidated_rerun_restores_old_results(self, tmp_path, monkeypatch):
        job = _results_folder(tmp_path / "job")

        def fake_gbw(**kwargs):
            # a step failed: the fresh generator/ holds only part of the results
            (job / "generator").mkdir()
            (job / "generator" / "timings.json").write_text("{}")
            return False

        monkeypatch.setattr(workflow, "gbw_analysis", fake_gbw)
        workflow.process_folder(str(job), clean_first=True, move_results=True)
        assert _names(job / "generator") == ["charge.json", "timings.json"]
        assert json.loads((job / "generator" / "timings.json").read_text()) == {"hirshfeld": 3.0}

    def test_orphaned_stash_restored_before_plain_restart(self, tmp_path, monkeypatch):
        job = _results_folder(tmp_path / "job")
        workflow._clean_first(str(job), _LOG)
        (job / "generator").mkdir()  # partial output of a rerun killed by walltime
        seen = {}

        def fake_gbw(**kwargs):
            seen["files"] = _names(job / "generator")
            return True

        monkeypatch.setattr(workflow, "gbw_analysis", fake_gbw)
        workflow.process_folder(str(job), restart=True, overwrite=True, move_results=True)
        assert seen["files"] == ["charge.json", "timings.json"]
        assert not (job / workflow._STASH).exists()


class TestCleanFirstNeedsMoveResults:
    def test_runners_refuse_and_touch_nothing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(workflow, "gbw_analysis", lambda **k: pytest.fail("must not run"))
        job = _results_folder(tmp_path / "res" / "v" / "job")
        (job / "charge.json").write_text("{}")
        before = _names(job)
        r = workflow.process_folder(str(job), clean_first=True, move_results=False)
        assert r["status"] == "error" and "move_results" in r["error"]
        (tmp_path / "in" / "v" / "job").mkdir(parents=True)
        r = workflow.process_folder_alcf(
            str(tmp_path / "in" / "v" / "job"),
            root_omol_inputs=str(tmp_path / "in"),
            root_omol_results=str(tmp_path / "res"),
            clean_first=True,
            move_results=False,
        )
        assert r["status"] == "error" and "move_results" in r["error"]
        assert _names(job) == before

    @pytest.mark.parametrize("module", ["full_runner_parsl_alcf", "full_runner_parsl"])
    def test_cli_rejects_combination(self, module, capsys):
        pytest.importorskip("parsl")
        mod = importlib.import_module(f"qtaim_gen.source.scripts.{module}")
        with pytest.raises(SystemExit) as exc:
            mod.main(["--clean_first"])
        assert exc.value.code == 2
        assert "--clean_first requires --move_results" in capsys.readouterr().err


def _ecp_folder(path, inp, adch=None):
    path.mkdir(parents=True)
    (path / "orca.inp").write_text(inp)
    gen = path / "generator"
    gen.mkdir()
    if adch is not None:
        with zipfile.ZipFile(gen / "out_files.zip", "w") as zf:
            zf.writestr("adch.out", adch)
    return str(path)


class TestCheckEcp:
    def test_statuses(self, tmp_path):
        assert qio.check_ecp_for_folder(_ecp_folder(tmp_path / "a", _INP_LIGHT)) == qio.ECP_NOT_APPLICABLE
        assert qio.check_ecp_for_folder(_ecp_folder(tmp_path / "b", _INP_ECP)) == qio.ECP_NO_ZIP
        assert qio.check_ecp_for_folder(_ecp_folder(tmp_path / "c", _INP_ECP, "no edf\n")) == qio.ECP_FAILED
        assert qio.check_ecp_for_folder(_ecp_folder(tmp_path / "d", _INP_ECP, _EDF_LINE)) == qio.ECP_PASSED

    def test_unreadable_geometry_falls_back_to_zip_check(self, tmp_path):
        folder = _ecp_folder(tmp_path / "e", "garbage\n", "no edf\n")
        assert qio.check_ecp_for_folder(folder) == qio.ECP_FAILED

    def test_zero_atom_geometry_is_unknown_not_light(self, tmp_path):
        assert qio.check_ecp_for_folder(_ecp_folder(tmp_path / "f", _INP_EMPTY)) == qio.ECP_NO_ZIP
        assert qio.check_ecp_for_folder(_ecp_folder(tmp_path / "g", _INP_EMPTY, "no edf\n")) == qio.ECP_FAILED

    def test_prevalidation_queues_only_proven_failures(self, tmp_path, monkeypatch):
        folders = {
            "light_no_zip": _ecp_folder(tmp_path / "light", _INP_LIGHT),
            "ecp_no_zip": _ecp_folder(tmp_path / "ecp_nozip", _INP_ECP),
            "ecp_failed": _ecp_folder(tmp_path / "ecp_failed", _INP_ECP, "no edf\n"),
            "ecp_passed": _ecp_folder(tmp_path / "ecp_passed", _INP_ECP, _EDF_LINE),
        }
        job_file = tmp_path / "jobs.txt"
        job_file.write_text("\n".join(folders.values()) + "\n")
        monkeypatch.setattr(qio, "validation_checks", lambda *a, **k: True)
        queued = qio.get_folders_from_file(
            str(job_file), num_folders=len(folders), pre_validate=True, check_ecp=True, max_workers=1
        )
        assert queued == [folders["ecp_failed"]]
