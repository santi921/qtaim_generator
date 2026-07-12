"""Tests for the restart-skip dry-run sweep (sweep_truncated_steps helper)."""
import json

import pytest

from qtaim_gen.source.scripts.helpers import sweep_truncated_steps as sweep


_ORCA_INP = (
    "! wB97M-V def2-TZVPD EnGrad\n"
    "*xyz 0 1 \n"
    "C   0.000 0.000 0.000\n"
    "O   0.000 0.000 1.200\n"
    "H   0.900 0.000 -0.500\n"
    "*\n"
)

_MULTIWFN_HEAD = (
    " Multiwfn -- A Multifunctional Wavefunction Analyzer\n"
    "                    ************ Main function menu ************\n"
)

_MULTIWFN_MENU_TAIL = (
    "                    ************ Main function menu ************\n"
)


def _make_started_folder(tmp_path):
    """Folder that has begun processing: log + orca.inp present."""
    folder = tmp_path / "job"
    folder.mkdir()
    (folder / "gbw_analysis.log").write_text("started\n")
    (folder / "orca.inp").write_text(_ORCA_INP)
    return folder


class TestResolveResultsFolder:
    def test_mapping_applied(self):
        out = sweep.resolve_results_folder(
            "/inputs/cat/job_1", "/inputs", "/results"
        )
        assert out == "/results/cat/job_1"

    def test_passthrough_without_roots(self):
        assert sweep.resolve_results_folder("/x/job", None, None) == "/x/job"

    def test_passthrough_when_prefix_mismatch(self):
        assert (
            sweep.resolve_results_folder("/other/job", "/inputs", "/results")
            == "/other/job"
        )


class TestRoutineSets:
    def test_full_set_1_contains_extended_routines(self):
        order, compiled_map, fuzzy_routines = sweep.routine_sets(1, spin_tf=False)
        for op in ("vdd", "chelpg", "mbis", "elf_fuzzy", "mbis_fuzzy_density", "qtaim"):
            assert op in order
        assert compiled_map["vdd"] == ("charge.json", "vdd", "charge")
        assert compiled_map["ibsi_bond"] == ("bond.json", "ibsi_bond")
        assert compiled_map["elf_fuzzy"] == ("fuzzy_full.json", "elf_fuzzy")
        assert compiled_map["other_alie"] == ("other.json", None)
        assert "elf_fuzzy" in fuzzy_routines
        assert "qtaim" not in compiled_map

    def test_spin_adds_spin_routines(self):
        order, _, fuzzy_routines = sweep.routine_sets(1, spin_tf=True)
        assert "hirsh_fuzzy_spin" in order
        assert "mbis_fuzzy_spin" in fuzzy_routines


class TestClassifyFolder:
    def test_no_outputs_when_folder_missing(self, tmp_path):
        rec = sweep.classify_folder(
            str(tmp_path / "absent"), None, None, full_set=1, move_results=False
        )
        assert rec["class"] == "no_outputs"

    def test_no_outputs_without_log(self, tmp_path):
        folder = tmp_path / "job"
        folder.mkdir()
        rec = sweep.classify_folder(
            str(folder), None, None, full_set=1, move_results=False
        )
        assert rec["class"] == "no_outputs"

    def test_complete_when_validation_passes(self, tmp_path, monkeypatch):
        folder = _make_started_folder(tmp_path)
        monkeypatch.setattr(sweep, "validation_checks", lambda *a, **k: True)
        rec = sweep.classify_folder(
            str(folder), None, None, full_set=1, move_results=False
        )
        assert rec["class"] == "complete"
        assert rec["n_atoms"] == 3

    def test_validation_loop_when_all_steps_skip(self, tmp_path, monkeypatch):
        """Validation fails but every step looks skippable -- the stuck class."""
        folder = _make_started_folder(tmp_path)
        monkeypatch.setattr(sweep, "validation_checks", lambda *a, **k: False)
        monkeypatch.setattr(sweep, "_has_usable_step_output", lambda *a, **k: True)
        rec = sweep.classify_folder(
            str(folder), None, None, full_set=1, move_results=False
        )
        assert rec["class"] == "validation_loop"
        assert rec["rerun_steps"] == []

    def test_needs_rerun_flags_truncated_out(self, tmp_path):
        """Real files, no mocks: truncated elf_fuzzy.out (one banner) must be
        classified needs_rerun with elf_fuzzy in truncated_steps."""
        folder = _make_started_folder(tmp_path)
        (folder / "elf_fuzzy.out").write_text(
            _MULTIWFN_HEAD + " Progress: [####------]  38.0 %\n"
        )
        rec = sweep.classify_folder(
            str(folder), None, None, full_set=1, move_results=False
        )
        assert rec["class"] == "needs_rerun"
        assert "elf_fuzzy" in rec["rerun_steps"]
        assert "elf_fuzzy" in rec["truncated_steps"]
        # steps with no artifacts at all rerun but are not "truncated"
        assert "hirshfeld" in rec["rerun_steps"]
        assert "hirshfeld" not in rec["truncated_steps"]

    def test_complete_out_not_rerun(self, tmp_path):
        """A .out with two banners is trusted; the step is skipped."""
        folder = _make_started_folder(tmp_path)
        (folder / "hirshfeld.out").write_text(
            _MULTIWFN_HEAD
            + " Final atomic charges:\n"
            + _MULTIWFN_MENU_TAIL
        )
        rec = sweep.classify_folder(
            str(folder), None, None, full_set=1, move_results=False
        )
        assert rec["class"] == "needs_rerun"
        assert "hirshfeld" not in rec["rerun_steps"]

    def test_timings_sum_reported(self, tmp_path):
        folder = _make_started_folder(tmp_path)
        (folder / "timings.json").write_text(
            json.dumps({"qtaim": 100.0, "hirshfeld": 50.0, "adch": -1})
        )
        rec = sweep.classify_folder(
            str(folder), None, None, full_set=1, move_results=False
        )
        assert rec["timings_sum_s"] == 150.0


class TestMainCli:
    def test_end_to_end_report_and_requeue(self, tmp_path):
        folder = _make_started_folder(tmp_path)
        (folder / "elf_fuzzy.out").write_text(
            _MULTIWFN_HEAD + " Progress: [##--------]  20.0 %\n"
        )
        job_file = tmp_path / "jobs.txt"
        job_file.write_text(f"# comment\n\n{folder}\n")
        report = tmp_path / "report.jsonl"
        requeue = tmp_path / "requeue.txt"
        rc = sweep.main([
            "--job_file", str(job_file),
            "--full_set", "1",
            "--report_file", str(report),
            "--requeue_file", str(requeue),
            "--n_workers", "2",
        ])
        assert rc == 0
        recs = [json.loads(line) for line in report.read_text().splitlines()]
        assert len(recs) == 1
        assert recs[0]["class"] == "needs_rerun"
        assert requeue.read_text().strip() == str(folder)

    def test_empty_job_file_returns_2(self, tmp_path):
        job_file = tmp_path / "jobs.txt"
        job_file.write_text("# only a comment\n")
        rc = sweep.main([
            "--job_file", str(job_file),
            "--report_file", str(tmp_path / "r.jsonl"),
        ])
        assert rc == 2
