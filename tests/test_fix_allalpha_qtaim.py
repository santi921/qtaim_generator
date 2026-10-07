"""Tests for fix-allalpha-qtaim (scripts/helpers/fix_allalpha_qtaim.py)."""

import json
import math
import os
import sys
import time

import pytest

from qtaim_gen.source.scripts.helpers import fix_allalpha_qtaim as fa

INP = "! UKS wB97M-V\n*xyz 0 {mult}\nH 0.0 0.0 0.0\nH 0.0 0.0 0.74\n*\n"
ALLALPHA = {
    "0": {"cp_num": 1, "element": "H", "density_all": 0.40, "density_alpha": 0.40, "density_beta": 0.0,
          "spin_density": 0.40, "e_loc_func": 0.98, "lol": 0.90, "lap_e_density": -1.2},
    "1": {"cp_num": 2, "element": "H", "density_all": 0.41, "density_alpha": 0.41, "density_beta": 0.0,
          "spin_density": 0.41, "e_loc_func": 0.97, "lol": 0.88, "lap_e_density": -1.1},
    "0_1": {"cp_num": 3, "connected_bond_paths": [0, 1], "density_all": 0.25, "density_alpha": 0.25,
            "density_beta": 0.0, "spin_density": 0.25, "e_loc_func": 0.50, "lol": 0.50, "lap_e_density": -0.9},
}


def _job(tmp_path, record=ALLALPHA, mult=1, s2=0.0, inp=True, orca=True):
    job = tmp_path / "job"
    gen = job / "generator"
    gen.mkdir(parents=True)
    (gen / "qtaim.json").write_text(json.dumps(record))
    if orca:
        (gen / "orca.json").write_text(json.dumps({} if s2 is None else {"s_squared": s2}))
    if inp:
        (job / "orca.inp").write_text(INP.format(mult=mult))
    return job


def _stored(job, rel="generator/qtaim.json"):
    return json.loads((job / rel).read_text())


def _run(job, dry_run=False, max_s2=0.05):
    return fa.process_folder(str(job), None, None, dry_run=dry_run, max_s2=max_s2)


def _with(record, cp, **fields):
    out = json.loads(json.dumps(record))
    out[cp].update(fields)
    return out


class TestTransforms:

    def test_constant_and_hand_computed_values(self):
        assert fa.C == pytest.approx(1.5874010519681994, rel=1e-15)
        # ELF 0.5 -> 1 / (1 + 2^(4/3)); LOL 0.5 -> t = 2^(-2/3), t / (1 + t)
        assert fa.elf_fix(0.5) == pytest.approx(0.2841036534166501, rel=1e-12)
        assert fa.lol_fix(0.5) == pytest.approx(0.38648820956430935, rel=1e-12)

    @pytest.mark.parametrize("chi", [0.01, 0.3, 1.0, 2.5, 40.0])
    def test_elf_undoes_the_all_alpha_uniform_gas_term(self, chi):
        elf_aa = 1 / (1 + (chi / 1.5874010519681994) ** 2)
        assert math.isclose(fa.elf_fix(elf_aa), 1 / (1 + chi ** 2), rel_tol=1e-12)

    @pytest.mark.parametrize("t", [0.001, 0.2, 1.0, 3.0, 500.0])
    def test_lol_undoes_the_all_alpha_uniform_gas_term(self, t):
        t_aa = 1.5874010519681994 * t
        assert math.isclose(fa.lol_fix(t_aa / (1 + t_aa)), t / (1 + t), rel_tol=1e-12)

    def test_bounds_are_left_alone(self):
        assert fa.elf_fix(0.0) == 0.0 and fa.elf_fix(1.0) == 1.0
        assert fa.lol_fix(0.0) == 0.0 and fa.lol_fix(1.0) == 1.0

    def test_cp_without_elf_or_lol_keeps_the_rest(self):
        record = {"0": {"density_all": 0.4, "density_alpha": 0.4, "density_beta": 0.0, "spin_density": 0.4}}
        assert fa.fix_record(record)["0"] == {"density_all": 0.4, "density_alpha": 0.2, "density_beta": 0.2,
                                              "spin_density": 0.0}


class TestClassify:

    def test_all_alpha_and_resolved(self):
        assert fa.classify(ALLALPHA) == fa.ALL_ALPHA
        assert fa.classify(fa.fix_record(ALLALPHA)) == fa.RESOLVED
        assert fa.classify({"0": {"density_all": 0.4}}) == fa.RESOLVED

    @pytest.mark.parametrize("fields", [
        {"density_beta": None},
        {"density_beta": 0.125, "density_alpha": 0.125},
        {"density_alpha": 0.0},
    ], ids=["beta_missing_value", "beta_mixed", "alpha_not_total"])
    def test_anything_in_between_is_ambiguous(self, fields):
        assert fa.classify(_with(ALLALPHA, "0_1", **fields)) == fa.AMBIGUOUS

    def test_missing_beta_key_is_ambiguous(self):
        record = json.loads(json.dumps(ALLALPHA))
        del record["0_1"]["density_beta"]
        assert fa.classify(record) == fa.AMBIGUOUS


class TestProcessFolder:

    def test_singlet_is_fixed(self, tmp_path):
        job = _job(tmp_path, s2=0.01)
        r = _run(job)
        assert r["status"] == fa.STATUS_FIXED and r["s_squared"] == 0.01
        assert r["copies"] == ["generator/qtaim.json"]
        out = _stored(job)
        for key, cp in ALLALPHA.items():
            fixed = out[key]
            assert fixed["density_alpha"] == fixed["density_beta"] == cp["density_all"] / 2
            assert fixed["spin_density"] == 0.0
            for f in ("cp_num", "density_all", "lap_e_density"):
                assert fixed[f] == cp[f]
        assert out["0_1"]["e_loc_func"] == pytest.approx(0.2841036534166501, rel=1e-12)
        assert out["0_1"]["lol"] == pytest.approx(0.38648820956430935, rel=1e-12)
        assert out["0_1"]["connected_bond_paths"] == [0, 1]
        assert sorted(os.listdir(job / "generator")) == ["orca.json", "qtaim.json"]
        assert not (job / ".processing.lock").exists()

    def test_second_pass_has_nothing_to_do(self, tmp_path):
        job = _job(tmp_path)
        assert _run(job)["status"] == fa.STATUS_FIXED
        once = _stored(job)
        assert _run(job)["status"] == fa.STATUS_NOT_ALLALPHA and _stored(job) == once

    def test_every_all_alpha_copy_is_fixed(self, tmp_path):
        job = _job(tmp_path)
        (job / "qtaim.json").write_text(json.dumps(ALLALPHA))
        r = _run(job)
        assert r["status"] == fa.STATUS_FIXED and r["copies"] == ["generator/qtaim.json", "qtaim.json"]
        assert _stored(job) == _stored(job, "qtaim.json") == fa.fix_record(ALLALPHA)

    def test_a_resolved_copy_is_left_alone(self, tmp_path):
        resolved = fa.fix_record(_with(ALLALPHA, "0", lap_e_density=-7.0))
        job = _job(tmp_path, record=resolved)
        (job / "qtaim.json").write_text(json.dumps(ALLALPHA))
        r = _run(job)
        assert r["status"] == fa.STATUS_FIXED and r["copies"] == ["qtaim.json"]
        assert _stored(job) == resolved

    def test_an_ambiguous_copy_stops_the_folder(self, tmp_path):
        job = _job(tmp_path)
        bad = _with(ALLALPHA, "0_1", density_beta=0.125)
        (job / "qtaim.json").write_text(json.dumps(bad))
        assert _run(job)["status"] == fa.STATUS_AMBIGUOUS
        assert _stored(job) == ALLALPHA and _stored(job, "qtaim.json") == bad

    def test_ecp_record_is_left_for_a_rerun(self, tmp_path):
        ecp = _with(ALLALPHA, "0", element="Pt", density_all=1.6e5, density_alpha=8.0e4, density_beta=8.0e4)
        job = _job(tmp_path, record=ecp)
        assert _run(job)["status"] == fa.STATUS_ALL_ALPHA_ECP and _stored(job) == ecp

    def test_all_alpha_record_from_a_resolved_run_is_not_fixed(self, tmp_path):
        job = _job(tmp_path)
        (job / "generator" / "qtaim.out").write_text(" Total/Alpha/Beta electrons:  2.0  1.0  1.0\n")
        assert _run(job)["status"] == fa.STATUS_STALE_MISMATCH and _stored(job) == ALLALPHA

    def test_all_alpha_banner_is_fixed(self, tmp_path):
        job = _job(tmp_path)
        (job / "generator" / "qtaim.out").write_text(" Total/Alpha/Beta electrons:  2.0  2.0  0.0\n")
        assert _run(job)["status"] == fa.STATUS_FIXED

    def test_dry_run_writes_nothing_and_takes_no_lock(self, tmp_path):
        job = _job(tmp_path)
        (job / ".processing.lock").write_text("other job")
        r = _run(job, dry_run=True)
        assert r["status"] == fa.STATUS_WOULD_FIX and r["copies"] == ["generator/qtaim.json"]
        assert _stored(job) == ALLALPHA
        assert (job / ".processing.lock").read_text() == "other job"

    @pytest.mark.parametrize("kwargs, status", [
        (dict(mult=3), fa.STATUS_OPEN_SHELL),
        (dict(s2=0.05), fa.STATUS_HIGH_S2),
        (dict(s2=None), fa.STATUS_NO_S2),
        (dict(orca=False), fa.STATUS_NO_S2),
        (dict(inp=False), fa.STATUS_NO_INP),
    ])
    def test_records_outside_the_validated_case_are_left_alone(self, tmp_path, kwargs, status):
        job = _job(tmp_path, **kwargs)
        assert _run(job)["status"] == status and _stored(job) == ALLALPHA

    def test_resolved_record_is_left_alone(self, tmp_path):
        resolved = fa.fix_record(ALLALPHA)
        job = _job(tmp_path, record=resolved)
        assert _run(job)["status"] == fa.STATUS_NOT_ALLALPHA and _stored(job) == resolved

    def test_input_folder_supplies_multiplicity_and_s2(self, tmp_path):
        inputs, results = tmp_path / "in", tmp_path / "res"
        (inputs / "v" / "job").mkdir(parents=True)
        (inputs / "v" / "job" / "orca.inp").write_text(INP.format(mult=1))
        (inputs / "v" / "job" / "orca.json").write_text(json.dumps({"s_squared": 0.0}))
        job = _job(results / "v", inp=False, s2=None)  # results orca.json has no s_squared
        r = fa.process_folder(str(inputs / "v" / "job"), str(inputs), str(results), dry_run=False, max_s2=0.05)
        assert r["folder"] == str(job) and r["status"] == fa.STATUS_FIXED and r["s_squared"] == 0.0

    def test_a_live_lock_is_never_broken_however_old(self, tmp_path):
        job = _job(tmp_path)
        lock = job / ".processing.lock"
        lock.write_text("stalled heavy job")
        old = time.time() - 30 * 86400
        os.utime(lock, (old, old))
        assert _run(job)["status"] == fa.STATUS_LOCKED
        assert _stored(job) == ALLALPHA and lock.read_text() == "stalled heavy job"

    def test_missing(self, tmp_path):
        assert _run(tmp_path / "gone")["status"] == fa.STATUS_MISSING

    def test_an_error_is_failed_and_releases_the_lock(self, tmp_path, monkeypatch):
        job = _job(tmp_path)

        def boom(*a, **k):
            raise OSError("ESTALE")
        monkeypatch.setattr(fa, "_plan", boom)
        r = _run(job)
        assert r["status"] == fa.STATUS_FAILED and "ESTALE" in r["error"]
        assert not (job / ".processing.lock").exists()

    def test_a_lock_error_is_failed_not_raised(self, tmp_path, monkeypatch):
        from qtaim_gen.source.core import workflow
        job = _job(tmp_path)

        def denied(*a, **k):
            raise PermissionError("EACCES")
        monkeypatch.setattr(workflow, "acquire_lock", denied)
        r = _run(job)
        assert r["status"] == fa.STATUS_FAILED and "EACCES" in r["error"] and _stored(job) == ALLALPHA


class TestMain:

    def _list(self, tmp_path):
        fixed = _job(tmp_path / "a")
        triplet = _job(tmp_path / "b", mult=3)
        locked = _job(tmp_path / "c")
        (locked / ".processing.lock").write_text("other job")
        lst = tmp_path / "jobs.txt"
        lst.write_text(f"{fixed}\n{triplet}\n{locked}\n")
        return lst, fixed, triplet, locked

    @pytest.mark.parametrize("workers", ["1", "2"])
    def test_remaining_lists_only_folders_needing_a_rerun(self, tmp_path, monkeypatch, workers):
        lst, fixed, triplet, locked = self._list(tmp_path)
        report, left = tmp_path / "rep.json", tmp_path / "left.txt"
        monkeypatch.setattr(sys, "argv", ["fix-allalpha-qtaim", "--folder_list", str(lst), "--workers", workers,
                                          "--report", str(report), "--list_remaining", str(left)])
        assert fa.main() == 0
        agg = json.loads(report.read_text())["aggregate"]
        assert agg[fa.STATUS_FIXED] == 1 and agg[fa.STATUS_OPEN_SHELL] == 1 and agg[fa.STATUS_LOCKED] == 1
        assert sorted(left.read_text().split()) == sorted([str(triplet), str(locked)])
        assert _stored(fixed) == fa.fix_record(ALLALPHA) and _stored(locked) == ALLALPHA

    def test_dry_run_through_main_writes_nothing(self, tmp_path, monkeypatch):
        lst, fixed, _, _ = self._list(tmp_path)
        report = tmp_path / "rep.json"
        monkeypatch.setattr(sys, "argv", ["fix-allalpha-qtaim", "--folder_list", str(lst), "--workers", "1",
                                          "--dry_run", "--report", str(report)])
        assert fa.main() == 0
        rep = json.loads(report.read_text())
        assert rep["aggregate"][fa.STATUS_WOULD_FIX] == 2 and rep["dry_run"] is True
        assert _stored(fixed) == ALLALPHA
