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
        """A .out with two banners and a parseable, full charge table is
        trusted; the step is skipped."""
        folder = _make_started_folder(tmp_path)
        (folder / "hirshfeld.out").write_text(
            _MULTIWFN_HEAD
            + " Final atomic charges:\n"
            + " Atom    1(C ):   0.10000000\n"
            + " Atom    2(O ):  -0.20000000\n"
            + " Atom    3(H ):   0.10000000\n"
            + "\n"
            + _MULTIWFN_MENU_TAIL
        )
        rec = sweep.classify_folder(
            str(folder), None, None, full_set=1, move_results=False
        )
        assert rec["class"] == "needs_rerun"
        assert "hirshfeld" not in rec["rerun_steps"]

    def test_banner_complete_but_empty_table_reruns(self, tmp_path):
        """Two banners are no longer enough: a charge table with no rows (or
        overflowed values) is what a run on a broken wavefunction leaves."""
        folder = _make_started_folder(tmp_path)
        (folder / "hirshfeld.out").write_text(
            _MULTIWFN_HEAD + " Final atomic charges:\n" + _MULTIWFN_MENU_TAIL
        )
        rec = sweep.classify_folder(
            str(folder), None, None, full_set=1, move_results=False
        )
        assert "hirshfeld" in rec["rerun_steps"]

    def test_timings_sum_reported(self, tmp_path):
        folder = _make_started_folder(tmp_path)
        (folder / "timings.json").write_text(
            json.dumps({"qtaim": 100.0, "hirshfeld": 50.0, "adch": -1})
        )
        rec = sweep.classify_folder(
            str(folder), None, None, full_set=1, move_results=False
        )
        assert rec["timings_sum_s"] == 150.0


class TestCheckOrca:
    @staticmethod
    def _write_orca_json(folder, version=None):
        data = {"final_energy_eh": -1.0}
        if version is not None:
            data["orca_parser_version"] = version
        (folder / "orca.json").write_text(json.dumps(data))

    def _classify(self, folder, check_orca=True):
        return sweep.classify_folder(
            str(folder), None, None, full_set=0, move_results=False, check_orca=check_orca
        )

    def test_current_orca_json_is_complete(self, tmp_path, monkeypatch):
        folder = _make_started_folder(tmp_path)
        self._write_orca_json(folder, sweep.ORCA_PARSER_VERSION)
        monkeypatch.setattr(sweep, "validation_checks", lambda *a, **k: True)
        rec = self._classify(folder)
        assert rec["class"] == "complete"
        assert rec["orca_stale"] is False

    def test_stale_orca_json_with_archive_is_orca_reparse(self, tmp_path, monkeypatch):
        folder = _make_started_folder(tmp_path)
        self._write_orca_json(folder)
        (folder / "orca.tar.zst").write_bytes(b"x")
        monkeypatch.setattr(sweep, "validation_checks", lambda *a, **k: True)
        rec = self._classify(folder)
        assert rec["class"] == "orca_reparse"
        assert rec["orca_stale"] is True

    def test_missing_orca_json_with_out_is_orca_reparse(self, tmp_path, monkeypatch):
        folder = _make_started_folder(tmp_path)
        (folder / "orca.out").write_text("ORCA\n")
        monkeypatch.setattr(sweep, "validation_checks", lambda *a, **k: True)
        assert self._classify(folder)["class"] == "orca_reparse"

    def test_source_in_inputs_folder_counts(self, tmp_path, monkeypatch):
        inputs = tmp_path / "in"
        results = tmp_path / "res"
        (inputs / "job").mkdir(parents=True)
        (inputs / "job" / "orca.tar.zst").write_bytes(b"x")
        results.mkdir()
        folder = _make_started_folder(results)
        self._write_orca_json(folder)
        monkeypatch.setattr(sweep, "validation_checks", lambda *a, **k: True)
        rec = sweep.classify_folder(
            str(inputs / "job"), str(inputs), str(results),
            full_set=0, move_results=False, check_orca=True,
        )
        assert rec["class"] == "orca_reparse"

    def test_stale_without_source_is_orca_no_source(self, tmp_path, monkeypatch):
        folder = _make_started_folder(tmp_path)
        self._write_orca_json(folder)
        monkeypatch.setattr(sweep, "validation_checks", lambda *a, **k: True)
        assert self._classify(folder)["class"] == "orca_no_source"

    def test_stale_ignored_without_flag(self, tmp_path, monkeypatch):
        folder = _make_started_folder(tmp_path)
        self._write_orca_json(folder)
        monkeypatch.setattr(sweep, "validation_checks", lambda *a, **k: True)
        rec = self._classify(folder, check_orca=False)
        assert rec["class"] == "complete"
        assert "orca_stale" not in rec

    def test_stale_flag_rides_along_on_rerun(self, tmp_path, monkeypatch):
        folder = _make_started_folder(tmp_path)
        self._write_orca_json(folder)
        (folder / "elf_fuzzy.out").write_text(
            _MULTIWFN_HEAD + " Progress: [##--------]  20.0 %\n"
        )
        rec = sweep.classify_folder(
            str(folder), None, None, full_set=1, move_results=False, check_orca=True
        )
        assert rec["class"] == "needs_rerun"
        assert rec["orca_stale"] is True

    def test_requeue_skips_orca_no_source(self, tmp_path, monkeypatch):
        (tmp_path / "a").mkdir()
        (tmp_path / "b").mkdir()
        a = _make_started_folder(tmp_path / "a")
        b = _make_started_folder(tmp_path / "b")
        for f in (a, b):
            self._write_orca_json(f)
        (a / "orca.tar.zst").write_bytes(b"x")
        monkeypatch.setattr(sweep, "validation_checks", lambda *a, **k: True)
        job_file = tmp_path / "jobs.txt"
        job_file.write_text(f"{a}\n{b}\n")
        requeue = tmp_path / "requeue.txt"
        rc = sweep.main([
            "--job_file", str(job_file), "--check_orca",
            "--report_file", str(tmp_path / "r.jsonl"),
            "--requeue_file", str(requeue), "--n_workers", "1",
        ])
        assert rc == 0
        assert requeue.read_text().split() == [str(a)]



class TestHirshDensityRebuild:
    def test_closed_shell_without_archive_is_reparse_only(self, tmp_path, monkeypatch):
        folder = _make_started_folder(tmp_path)
        gen = folder / "generator"
        gen.mkdir()
        atoms = ["1_C", "2_O", "3_H"]
        zeros = dict.fromkeys(atoms, 0.0)
        zeros.update(sum=0.0, abs_sum=0.0)
        (gen / "fuzzy_full.json").write_text(json.dumps({"hirsh_fuzzy_density": zeros}))
        (gen / "charge.json").write_text(json.dumps(
            {"hirshfeld": {"charge": {"1_C": 0.1, "2_O": -0.2, "3_H": 0.1}}}))
        monkeypatch.setattr(sweep, "validation_checks", lambda *a, **k: True)
        rec = sweep.classify_folder(
            str(folder), None, None, full_set=0, move_results=True, recheck_fuzzy=True
        )
        assert rec["class"] == "reparse_only"
        assert rec["recheck"]["derived"] == ["hirsh_fuzzy_density"]


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
