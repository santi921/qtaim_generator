"""Tests for fix-allalpha-qtaim (scripts/helpers/fix_allalpha_qtaim.py)."""

import json
import math
import sys

import pytest

from qtaim_gen.source.scripts.helpers import fix_allalpha_qtaim as fa

INP = "! UKS wB97M-V\n*xyz 0 {mult}\nH 0.0 0.0 0.0\nH 0.0 0.0 0.74\n*\n"
ALLALPHA = {
    "0": {"cp_num": 1, "element": "H", "density_all": 0.40, "density_alpha": 0.40, "density_beta": 0.0,
          "spin_density": 0.40, "e_loc_func": 0.98, "lol": 0.90, "lap_e_density": -1.2},
    "1": {"cp_num": 2, "element": "H", "density_all": 0.41, "density_alpha": 0.41, "density_beta": 0.0,
          "spin_density": 0.41, "e_loc_func": 0.97, "lol": 0.88, "lap_e_density": -1.1},
    "0_1": {"cp_num": 3, "connected_bond_paths": [0, 1], "density_all": 0.25, "density_alpha": 0.25,
            "density_beta": 0.0, "spin_density": 0.25, "e_loc_func": 0.50, "lol": 0.40, "lap_e_density": -0.9},
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


def _stored(job):
    return json.loads((job / "generator" / "qtaim.json").read_text())


def _run(job, dry_run=False, max_s2=0.05, **kw):
    return fa.process_folder(str(job), kw.get("root_inputs"), kw.get("root_results"), dry_run=dry_run,
                             max_s2=max_s2)


class TestTransforms:

    @pytest.mark.parametrize("chi", [0.01, 0.3, 1.0, 2.5, 40.0])
    def test_elf_undoes_the_all_alpha_uniform_gas_term(self, chi):
        # all-alpha D0 is 2^(2/3) too big, so chi_aa = chi / 2^(2/3)
        elf_aa = 1 / (1 + (chi / fa.C) ** 2)
        assert math.isclose(fa.elf_fix(elf_aa), 1 / (1 + chi ** 2), rel_tol=1e-12)

    @pytest.mark.parametrize("t", [0.001, 0.2, 1.0, 3.0, 500.0])
    def test_lol_undoes_the_all_alpha_uniform_gas_term(self, t):
        t_aa = fa.C * t
        assert math.isclose(fa.lol_fix(t_aa / (1 + t_aa)), t / (1 + t), rel_tol=1e-12)

    def test_bounds_are_left_alone(self):
        assert fa.elf_fix(0.0) == 0.0 and fa.elf_fix(1.0) == 1.0
        assert fa.lol_fix(0.0) == 0.0 and fa.lol_fix(1.0) == 1.0

    def test_all_alpha_detection(self):
        assert fa.is_all_alpha(ALLALPHA)
        assert not fa.is_all_alpha(fa.fix_record(ALLALPHA))
        assert not fa.is_all_alpha({"0": {"density_all": 0.4}})


class TestProcessFolder:

    def test_singlet_is_fixed_and_recorded(self, tmp_path):
        job = _job(tmp_path, s2=0.01)
        r = _run(job)
        assert r["status"] == fa.STATUS_FIXED and r["s_squared"] == 0.01
        out = _stored(job)
        for key, cp in ALLALPHA.items():
            fixed = out[key]
            assert fixed["density_alpha"] == fixed["density_beta"] == cp["density_all"] / 2
            assert fixed["spin_density"] == 0.0
            assert fixed["e_loc_func"] == fa.elf_fix(cp["e_loc_func"]) < cp["e_loc_func"]
            assert fixed["lol"] == fa.lol_fix(cp["lol"]) < cp["lol"]
            for f in ("cp_num", "density_all", "lap_e_density"):
                assert fixed[f] == cp[f]
        assert out["0_1"]["connected_bond_paths"] == [0, 1]
        side = json.loads((job / "generator" / fa.SIDECAR).read_text())
        assert side["s_squared"] == 0.01 and side["n_cps"] == 3 and side["fields"] == list(fa.FIELDS)
        assert not (job / ".processing.lock").exists()

    def test_second_pass_has_nothing_to_do(self, tmp_path):
        job = _job(tmp_path)
        assert _run(job)["status"] == fa.STATUS_FIXED
        once = _stored(job)
        assert _run(job)["status"] == fa.STATUS_NOT_ALLALPHA and _stored(job) == once

    def test_dry_run_writes_nothing_and_takes_no_lock(self, tmp_path):
        job = _job(tmp_path)
        (job / ".processing.lock").write_text("other job")
        assert _run(job, dry_run=True)["status"] == fa.STATUS_WOULD_FIX
        assert _stored(job) == ALLALPHA and not (job / "generator" / fa.SIDECAR).exists()

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
        job = _job(results / "v", inp=False, orca=False)
        r = fa.process_folder(str(inputs / "v" / "job"), str(inputs), str(results), dry_run=False, max_s2=0.05)
        assert r["folder"] == str(job) and r["status"] == fa.STATUS_FIXED

    def test_locked_and_missing(self, tmp_path):
        job = _job(tmp_path)
        (job / ".processing.lock").write_text("other job")
        assert _run(job)["status"] == fa.STATUS_LOCKED and _stored(job) == ALLALPHA
        assert _run(tmp_path / "gone")["status"] == fa.STATUS_MISSING


class TestMain:

    def test_remaining_lists_only_folders_needing_a_rerun(self, tmp_path, monkeypatch):
        fixed = _job(tmp_path / "a")
        triplet = _job(tmp_path / "b", mult=3)
        lst = tmp_path / "jobs.txt"
        lst.write_text(f"{fixed}\n{triplet}\n")
        report, left = tmp_path / "rep.json", tmp_path / "left.txt"
        monkeypatch.setattr(sys, "argv", ["fix-allalpha-qtaim", "--folder_list", str(lst), "--workers", "1",
                                          "--report", str(report), "--list_remaining", str(left)])
        assert fa.main() == 0
        agg = json.loads(report.read_text())["aggregate"]
        assert agg[fa.STATUS_FIXED] == 1 and agg[fa.STATUS_OPEN_SHELL] == 1
        assert left.read_text().split() == [str(triplet)]
