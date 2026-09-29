"""--clean_first never destroys completed results; --check_ecp only queues proven ECP failures."""

import json
import zipfile

from qtaim_gen.source.core import workflow
from qtaim_gen.source.utils import io as qio

_INP_LIGHT = "! wB97M-V def2-TZVPD\n*xyz 0 1\nC 0.0 0.0 0.0\nO 0.0 0.0 1.2\n*\n"
_INP_ECP = "! wB97M-V def2-TZVPD\n*xyz 0 1\nH 0.0 0.0 0.0\nI 0.0 0.0 1.6\n*\n"
_EDF_LINE = " Loading EDF library finished!\n"


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


class TestCleanFirstHelper:
    def test_keeps_generator_log_and_lock(self, tmp_path):
        job = _results_folder(tmp_path / "job")
        (job / ".processing.lock").write_text("123")
        before = {p.name: p.read_bytes() for p in (job / "generator").iterdir()}
        workflow._clean_first(str(job), workflow.logging.getLogger("t"))
        assert sorted(p.name for p in job.iterdir()) == [".processing.lock", "gbw_analysis.log", "generator"]
        assert {p.name: p.read_bytes() for p in (job / "generator").iterdir()} == before


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
        assert not (job / "hirshfeld.out").exists() and not (job / "scratch_dir").exists()
        # inputs were copied back in fresh
        assert (job / "orca.gbw.zstd0").read_bytes() == b"gbw"

    def test_local_runner_failed_rerun_keeps_generator(self, tmp_path, monkeypatch):
        job = _results_folder(tmp_path / "job")
        charge_before = (job / "generator" / "charge.json").read_text()
        calls = {}

        def fake_gbw(**kwargs):
            calls.update(kwargs)
            return False

        monkeypatch.setattr(workflow, "gbw_analysis", fake_gbw)
        workflow.process_folder(str(job), clean_first=True, restart=True, move_results=True)
        assert calls["overwrite"] is True and calls["restart"] is False
        assert (job / "generator" / "charge.json").read_text() == charge_before


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
