"""Tests for the recheck-fuzzy driver (scripts/helpers/recheck_fuzzy_folders.py)."""

import json
import sys

import pytest

from qtaim_gen.source.scripts.helpers import recheck_fuzzy_folders as rf
from qtaim_gen.source.utils import validation
from tests.test_fuzzy_recheck import (
    ALL_ALPHA_SPIN,
    GOOD_DENSITY,
    GOOD_SPIN,
    HIRSH_Q,
    ZEROS,
    _job,
    _write_charges,
)

_ELEMENTS = ["H", "Br", "H", "C", "C", "C", "Br", "H", "H", "H", "H", "H"]


def _inp(mult):
    atoms = "".join(f"{e} 0.0 0.0 {1.5 * i:.1f}\n" for i, e in enumerate(_ELEMENTS))
    return f"! wB97M-V def2-TZVPD\n*xyz 0 {mult}\n{atoms}*\n"


def _folder(tmp_path, fuzzy, mult=1, inp=True):
    tmp_path.mkdir(parents=True, exist_ok=True)
    job = _job(tmp_path, fuzzy)
    _write_charges(job, HIRSH_Q)
    if inp:
        (job / "orca.inp").write_text(_inp(mult))
    return job


def _run(job, **kw):
    args = dict(root_inputs=None, root_results=None, full_set=0, move_results=True,
                check_orca=False, dry_run=False)
    args.update(kw)
    return rf.process_folder(str(job), **args)


def _fuzzy(job):
    return json.loads((job / "generator" / "fuzzy_full.json").read_text())


@pytest.fixture
def validates(monkeypatch):
    monkeypatch.setattr(validation, "validation_checks", lambda *a, **k: True)


class TestProcessFolder:

    def test_reparse_only_folder_is_fixed_in_place(self, tmp_path, validates):
        job = _folder(tmp_path, {"becke_fuzzy_density": GOOD_DENSITY, "hirsh_fuzzy_density": ZEROS})
        r = _run(job)
        assert r["status"] == rf.STATUS_FIXED
        assert r["reparse"] == ["hirsh_fuzzy_density"] and r["derived"] == ["hirsh_fuzzy_density"]
        assert _fuzzy(job)["hirsh_fuzzy_density"]["sum"] > 90
        assert not (job / ".processing.lock").exists()

    def test_rerun_needed_is_left_untouched(self, tmp_path, validates):
        fuzzy = {"hirsh_fuzzy_density": ZEROS, "hirsh_fuzzy_spin": ALL_ALPHA_SPIN}
        job = _folder(tmp_path, fuzzy, mult=2)
        (job / "orca.gbw").write_text("gbw")
        r = _run(job)
        assert r["status"] == rf.STATUS_NEEDS_MULTIWFN and r["rerun"] == ["hirsh_fuzzy_spin"]
        assert _fuzzy(job) == fuzzy
        assert (job / "orca.gbw").exists()

    def test_clean_folder(self, tmp_path, validates):
        job = _folder(tmp_path, {"hirsh_fuzzy_density": GOOD_DENSITY, "hirsh_fuzzy_spin": GOOD_SPIN}, mult=2)
        assert _run(job)["status"] == rf.STATUS_CLEAN

    def test_dry_run_writes_nothing_and_takes_no_lock(self, tmp_path, validates):
        fuzzy = {"hirsh_fuzzy_density": ZEROS}
        job = _folder(tmp_path, fuzzy)
        (job / ".processing.lock").write_text("other job")
        assert _run(job, dry_run=True)["status"] == rf.STATUS_WOULD_FIX
        assert _fuzzy(job) == fuzzy

    def test_locked_folder_is_left_alone(self, tmp_path, validates):
        fuzzy = {"hirsh_fuzzy_density": ZEROS}
        job = _folder(tmp_path, fuzzy)
        (job / ".processing.lock").write_text("other job")
        assert _run(job)["status"] == rf.STATUS_LOCKED
        assert _fuzzy(job) == fuzzy
        assert (job / ".processing.lock").read_text() == "other job"

    def test_still_invalid_when_validation_fails(self, tmp_path, monkeypatch):
        monkeypatch.setattr(validation, "validation_checks", lambda *a, **k: False)
        job = _folder(tmp_path, {"hirsh_fuzzy_density": ZEROS})
        assert _run(job)["status"] == rf.STATUS_STILL_INVALID
        assert _fuzzy(job)["hirsh_fuzzy_density"]["sum"] > 90

    def test_no_multiplicity(self, tmp_path, validates):
        job = _folder(tmp_path, {"hirsh_fuzzy_density": ZEROS}, inp=False)
        assert _run(job)["status"] == rf.STATUS_NO_MULT

    def test_input_path_maps_to_results_and_reads_inp_there(self, tmp_path, validates):
        inputs, results = tmp_path / "in", tmp_path / "res"
        (inputs / "v" / "job").mkdir(parents=True)
        (inputs / "v" / "job" / "orca.inp").write_text(_inp(1))
        res_job = results / "v" / "job"
        res_job.mkdir(parents=True)
        _job(res_job, {"hirsh_fuzzy_density": ZEROS})
        _write_charges(res_job, HIRSH_Q)
        r = rf.process_folder(str(inputs / "v" / "job"), root_inputs=str(inputs), root_results=str(results),
                              full_set=0, move_results=True, check_orca=False, dry_run=False)
        assert r["folder"] == str(res_job) and r["status"] == rf.STATUS_FIXED


class TestMain:

    def test_remaining_lists_entries_that_need_the_runner(self, tmp_path, monkeypatch, validates):
        fixed = _folder(tmp_path / "a", {"hirsh_fuzzy_density": ZEROS})
        rerun = _folder(tmp_path / "b", {"hirsh_fuzzy_spin": ALL_ALPHA_SPIN}, mult=2)
        (rerun / "orca.gbw").write_text("gbw")
        lst = tmp_path / "jobs.txt"
        lst.write_text(f"{fixed}\n{rerun}\n")
        report, left = tmp_path / "rep.json", tmp_path / "left.txt"
        monkeypatch.setattr(sys, "argv", ["recheck-fuzzy", "--folder_list", str(lst), "--workers", "1",
                                          "--report", str(report), "--list_remaining", str(left)])
        assert rf.main() == 0
        agg = json.loads(report.read_text())["aggregate"]
        assert agg[rf.STATUS_FIXED] == 1 and agg[rf.STATUS_NEEDS_MULTIWFN] == 1
        assert left.read_text().split() == [str(rerun)]
