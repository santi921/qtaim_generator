"""--recheck_cp_labels: a qtaim.json whose nuclear CP sits on another atom is rerun (validator and restart gate agree)."""

import json
import logging
import re

import pytest

from qtaim_gen.source.core.omol import _prepare_allalpha_qtaim_rerun, _qtaim_output_complete
from qtaim_gen.source.utils.validation import (
    CP_MOVED_A, CP_ON_ATOM_A, misplaced_nuclear_cps, qtaim_copy_has_all_alpha_defect,
    qtaim_copy_has_mislabeled_cps)

LOG = logging.getLogger("test_recheck_cp_labels")
INP = "! RKS wB97M-V\n*xyz 0 1\nC 0.0 0.0 0.0\nH 1.09 0.0 0.0\nH 1.5 0.62 0.0\nO -1.2 0.0 0.0\n*\n"
# the same geometry as a doublet cation (15 electrons)
INP_OPEN = INP.replace("RKS", "UKS").replace("*xyz 0 1", "*xyz 1 2")
POS = {0: [0.0, 0.0, 0.0], 1: [1.09, 0.0, 0.0], 2: [1.5, 0.62, 0.0], 3: [-1.2, 0.0, 0.0]}


def _ncp(atom, rho, dx=0.01):
    return {"number": str(atom + 1), "pos_ang": [POS[atom][0] + dx, POS[atom][1], POS[atom][2]], "density_all": rho}


def _bcp(a, b, rho):
    return {"connected_bond_paths": [a + 1, b + 1], "density_all": rho}


BONDS = {"0_1": _bcp(0, 1, 0.28), "0_2": _bcp(0, 2, 0.20), "0_3": _bcp(0, 3, 0.40)}
RIGHT = {"0": _ncp(0, 120.0), "1": _ncp(1, 0.42), "2": _ncp(2, 0.31), "3": _ncp(3, 300.0), **BONDS}
# the old mapper: two close H atoms swapped their CPs
SWAPPED = dict(RIGHT, **{"1": RIGHT["2"], "2": RIGHT["1"]})
# the pre-February mapper: both H atoms claimed H2's CP, H1's own CP is lost
DUPLICATE = dict(RIGHT, **{"1": RIGHT["2"]})
# a CP well off its atom but not on any other (the current mapper's distance fallback can do this)
OFF_EVERY_ATOM = dict(RIGHT, **{"1": dict(RIGHT["1"], pos_ang=[1.3, 0.3, 0.0])})
# nearest to another atom, but 0.13 A from it: not sitting on it, so not a mislabel
NEAR_ANOTHER = dict(RIGHT, **{"1": dict(RIGHT["1"], pos_ang=[1.45, 0.5, 0.0])})


def _folder(tmp_path, record, root_copy=None, inp=INP):
    job = tmp_path / "job"
    (job / "generator").mkdir(parents=True)
    (job / "generator" / "qtaim.json").write_text(json.dumps(record))
    if root_copy is not None:
        (job / "qtaim.json").write_text(json.dumps(root_copy))
    (job / "orca.inp").write_text(inp)
    return job


def _resolved(record):
    """The record as a sound unrestricted run stores it (alpha and beta both present)."""
    return {k: dict(v, density_alpha=0.6 * v["density_all"], density_beta=0.4 * v["density_all"],
                    spin_density=0.2 * v["density_all"]) for k, v in record.items()}


class TestDetection:

    @pytest.mark.parametrize("record, moved", [
        (RIGHT, {}), (SWAPPED, {1: 2, 2: 1}), (DUPLICATE, {1: 2}), (OFF_EVERY_ATOM, {}), (NEAR_ANOTHER, {})])
    def test_misplaced(self, record, moved):
        assert misplaced_nuclear_cps(record, POS) == moved

    def test_limits(self):
        assert (CP_MOVED_A, CP_ON_ATOM_A) == (0.1, 0.05)
        # 0.04 A from its own atom is still its own
        near = dict(RIGHT, **{"1": dict(RIGHT["1"], pos_ang=[1.13, 0.0, 0.0])})
        assert misplaced_nuclear_cps(near, POS) == {}

    def test_bad_positions_are_ignored(self):
        bad = dict(RIGHT, **{"1": dict(RIGHT["1"], pos_ang=[float("nan"), 0.0, 0.0]), "2": dict(RIGHT["2"], pos_ang=[1.5])})
        assert misplaced_nuclear_cps(bad, POS) == {}

    def test_cp_on_an_atom_of_another_element_is_flagged(self):
        # relabel-qtaim-cps rejects this one (element check), so the rerun is what repairs it
        on_o = dict(RIGHT, **{"1": dict(RIGHT["1"], pos_ang=POS[3])})
        assert misplaced_nuclear_cps(on_o, POS) == {1: 3}

    def test_cp_key_missing_from_the_geometry_is_ignored(self):
        extra = dict(RIGHT, **{"7": dict(RIGHT["1"], pos_ang=POS[2])})
        assert misplaced_nuclear_cps(extra, POS) == {}

    @pytest.mark.parametrize("bad", [[1.5, 0.62], [1.5, 0.62, 0.0, 1.0], [float("nan"), 0.62, 0.0]])
    def test_malformed_geometry_line_does_not_raise(self, bad):
        # a 2- or 4-number line made math.dist raise inside run_jobs' restart gate
        assert misplaced_nuclear_cps(SWAPPED, {**POS, 2: bad}) == {}
        assert misplaced_nuclear_cps(SWAPPED, {**POS, 0: bad}) == {1: 2, 2: 1}

    def test_relabel_tool_uses_the_same_limits(self):
        from qtaim_gen.source.scripts.helpers import relabel_qtaim_cps as rq
        assert (rq.MOVED_A, rq.ON_ATOM_A) == (CP_MOVED_A, CP_ON_ATOM_A)


class TestValidatorAndRestartGateAgree:

    @pytest.mark.parametrize("gen, root, rejected", [
        (RIGHT, None, False), (SWAPPED, None, True), (DUPLICATE, None, True), (OFF_EVERY_ATOM, None, False),
        (RIGHT, SWAPPED, True), (SWAPPED, RIGHT, True), (RIGHT, RIGHT, False)])
    def test_same_verdict(self, tmp_path, monkeypatch, gen, root, rejected):
        from qtaim_gen.source.utils import validation
        job = _folder(tmp_path, gen, root_copy=root)
        for name in ("timings", "fuzzy_full", "other", "charge", "bond"):
            (job / "generator" / f"{name}.json").write_text("{}")
        for check in ("validate_timing_dict", "validate_fuzzy_dict", "validate_other_dict",
                      "validate_charge_dict", "validate_bond_dict"):
            monkeypatch.setattr(validation, check, lambda *a, **k: True)
        assert qtaim_copy_has_mislabeled_cps(str(job)) is rejected
        assert _qtaim_output_complete(str(job), n_atoms=4, recheck_cp_labels=True) is not rejected
        assert validation.validation_checks(str(job), move_results=True, recheck_cp_labels=True) is not rejected

    def test_gate_uses_the_geometry_it_is_given(self, tmp_path, monkeypatch):
        from qtaim_gen.source.utils import validation

        def no_reparse(*a, **k):
            raise AssertionError("geometry input parsed again")
        monkeypatch.setattr(validation, "get_charge_spin_n_atoms_from_folder", no_reparse)
        job = _folder(tmp_path, SWAPPED)
        assert not _qtaim_output_complete(str(job), n_atoms=4, recheck_cp_labels=True, atoms=POS)
        assert _qtaim_output_complete(str(job), n_atoms=4, recheck_cp_labels=True,
                                      atoms={0: POS[0], 1: POS[2], 2: POS[1], 3: POS[3]})

    def test_off_by_default(self, tmp_path, monkeypatch):
        from qtaim_gen.source.utils import validation
        job = _folder(tmp_path, SWAPPED)
        for name in ("timings", "fuzzy_full", "other", "charge", "bond"):
            (job / "generator" / f"{name}.json").write_text("{}")
        for check in ("validate_timing_dict", "validate_fuzzy_dict", "validate_other_dict",
                      "validate_charge_dict", "validate_bond_dict"):
            monkeypatch.setattr(validation, check, lambda *a, **k: True)
        assert _qtaim_output_complete(str(job), n_atoms=4)
        assert validation.validation_checks(str(job), move_results=True)


class TestRunJobsGate:
    """run_jobs' per-step restart decision for qtaim, through the real call."""

    def _run(self, tmp_path, caplog, record, flag):
        from qtaim_gen.source.core.omol import run_jobs
        job = _folder(tmp_path, record)
        (job / "qtaim.json").write_text(json.dumps(record))
        (job / "orca.wfn").write_text("wfn")
        (job / "timings.json").write_text(json.dumps({"qtaim": 1.0}))
        with caplog.at_level(logging.INFO, logger=LOG.name):
            run_jobs(str(job), separate=False, restart=True, debug=True, logger=LOG, recheck_cp_labels=flag)
        return any("Skipping qtaim" in r.getMessage() for r in caplog.records)

    def test_mislabeled_is_skipped_without_the_flag(self, tmp_path, caplog):
        assert self._run(tmp_path, caplog, DUPLICATE, flag=False)

    def test_mislabeled_reruns_under_the_flag(self, tmp_path, caplog):
        assert not self._run(tmp_path, caplog, DUPLICATE, flag=True)

    def test_a_correct_record_is_still_skipped(self, tmp_path, caplog):
        # what a fresh parse writes passes, so the rerun cannot loop
        assert self._run(tmp_path, caplog, RIGHT, flag=True)


class TestUnrestrictedRerun:
    """A mislabeled record of an unrestricted calculation reruns from a .wfx, never the .wfn."""

    def _job(self, tmp_path, record, inp=INP_OPEN, gbw=True):
        job = _folder(tmp_path, _resolved(record), inp=inp)
        for rel in ("orca.wfn", "generator/orca.wfn", "CPprop.txt", "generator/CPprop.txt"):
            (job / rel).write_text("old")
        if gbw:
            (job / "orca.gbw").write_bytes(b"gbw")
        assert not qtaim_copy_has_all_alpha_defect(str(job), n_electrons=15, mult=2)
        return job

    @pytest.mark.parametrize("allalpha", [True, False])
    def test_wfn_set_aside(self, tmp_path, allalpha):
        job = self._job(tmp_path, SWAPPED)
        assert _prepare_allalpha_qtaim_rerun(str(job), wfx=True, preprocess_compressed=False, logger=LOG,
                                             recheck_allalpha_qtaim=allalpha, recheck_cp_labels=True)
        for rel in ("orca.wfn", "generator/orca.wfn", "CPprop.txt", "generator/CPprop.txt"):
            assert not (job / rel).exists()
        assert (job / "orca.wfn.allalpha").exists() and (job / "orca.gbw").exists()

    @pytest.mark.parametrize("record, inp, flag", [
        (SWAPPED, INP, True),          # restricted: the .wfn is fine
        (RIGHT, INP_OPEN, True),       # nothing mislabeled
        (SWAPPED, INP_OPEN, False),    # flag off
    ], ids=["restricted", "sound", "flag_off"])
    def test_left_alone(self, tmp_path, record, inp, flag):
        job = self._job(tmp_path, record, inp=inp)
        assert _prepare_allalpha_qtaim_rerun(str(job), wfx=True, preprocess_compressed=False, logger=LOG,
                                             recheck_cp_labels=flag)
        assert (job / "orca.wfn").exists() and (job / "CPprop.txt").exists()

    @pytest.mark.parametrize("wfx, gbw", [(False, True), (True, False)], ids=["no_wfx_flag", "no_gbw_source"])
    def test_refused_when_it_cannot_rerun_from_a_wfx(self, tmp_path, wfx, gbw):
        job = self._job(tmp_path, SWAPPED, gbw=gbw)
        assert not _prepare_allalpha_qtaim_rerun(str(job), wfx=wfx, preprocess_compressed=False, logger=LOG,
                                                 recheck_allalpha_qtaim=False, recheck_cp_labels=True)
        assert (job / "orca.wfn").exists() and (job / "CPprop.txt").exists()

    def test_run_jobs_refuses_the_unrestricted_wfn_under_this_flag_alone(self, tmp_path, caplog):
        from qtaim_gen.source.core.omol import run_jobs
        job = self._job(tmp_path, SWAPPED)
        (job / "qtaim.json").write_text(json.dumps(_resolved(SWAPPED)))
        (job / "timings.json").write_text(json.dumps({"qtaim": 1.0}))
        with caplog.at_level(logging.INFO, logger=LOG.name):
            run_jobs(str(job), separate=False, restart=True, debug=True, logger=LOG, recheck_cp_labels=True)
        assert any("refusing to run qtaim" in r.getMessage() for r in caplog.records)
        assert json.loads((job / "timings.json").read_text())["qtaim"] == -1


class TestPlumbing:

    def test_gbw_analysis_implies_restart(self, tmp_path, monkeypatch):
        from qtaim_gen.source.core import omol
        seen = {}
        monkeypatch.setattr(omol, "run_jobs", lambda *a, **k: seen.update(k))
        monkeypatch.setattr(omol, "create_jobs", lambda *a, **k: None)
        monkeypatch.setattr(omol, "parse_multiwfn", lambda *a, **k: None)
        monkeypatch.setattr(omol, "validation_checks", lambda *a, **k: False)
        job = _folder(tmp_path, SWAPPED)
        (job / "orca.wfn").write_text("wfn")
        omol.gbw_analysis(str(job), multiwfn_cmd="x", orca_2mkl_cmd="y", restart=False, overwrite=False,
                          logger=LOG, recheck_cp_labels=True, move_results=True)
        assert seen.get("restart") is True and seen.get("recheck_cp_labels") is True

    def test_gbw_analysis_prepares_the_rerun_under_this_flag_alone(self, tmp_path, monkeypatch):
        from qtaim_gen.source.core import omol
        seen = {}
        monkeypatch.setattr(omol, "_prepare_allalpha_qtaim_rerun", lambda *a, **k: seen.update(k) or False)
        monkeypatch.setattr(omol, "run_jobs", lambda *a, **k: None)
        job = _folder(tmp_path, SWAPPED)
        assert omol.gbw_analysis(str(job), multiwfn_cmd="x", orca_2mkl_cmd="y", restart=True, overwrite=False,
                                 logger=LOG, recheck_cp_labels=True, move_results=True) is False
        assert seen == {"recheck_allalpha_qtaim": False, "recheck_cp_labels": True}

    def test_process_folder_pre_skip_passes_the_flag(self, tmp_path, monkeypatch):
        from qtaim_gen.source.core import workflow
        seen = {}

        def fake_validation(folder, **kwargs):
            seen.update(kwargs)
            return True
        monkeypatch.setattr(workflow, "validation_checks", fake_validation)
        monkeypatch.setattr(workflow, "gbw_analysis", lambda *a, **k: True)
        job = tmp_path / "job"
        job.mkdir()
        for name in ("timings", "qtaim", "other", "fuzzy_full", "charge"):
            (job / f"{name}.json").write_text("{}")
        workflow.process_folder(str(job), move_results=False, recheck_cp_labels=True)
        assert seen.get("recheck_cp_labels") is True

    def test_folder_prevalidation_passes_the_flag(self, tmp_path, monkeypatch):
        from qtaim_gen.source.utils import io
        seen = {}

        def fake(folder, **kwargs):
            seen.update(kwargs)
            return True
        monkeypatch.setattr(io, "validation_checks", fake)
        lst = tmp_path / "jobs.txt"
        (tmp_path / "a").mkdir()
        lst.write_text(f"{tmp_path / 'a'}\n")
        io.get_folders_from_file(str(lst), num_folders=10, pre_validate=True, recheck_cp_labels=True)
        assert seen.get("recheck_cp_labels") is True

    def test_sweep_predicts_with_the_flag_and_the_geometry(self, tmp_path, monkeypatch):
        from qtaim_gen.source.scripts.helpers import sweep_truncated_steps as sweep
        seen = []
        monkeypatch.setattr(sweep, "validation_checks", lambda *a, **k: seen.append(k) or False)
        monkeypatch.setattr(sweep, "_compiled_data_present", lambda *a, **k: False)
        monkeypatch.setattr(sweep, "_has_usable_step_output", lambda *a, **k: seen.append(k) or True)
        job = _folder(tmp_path, SWAPPED)
        sweep.classify_folder(str(job), None, None, full_set=0, move_results=True, recheck_cp_labels=True)
        assert seen and all(k.get("recheck_cp_labels") is True for k in seen)
        assert seen[-1].get("atoms") == POS

    @pytest.mark.parametrize("module", ["full_runner", "full_runner_parsl", "full_runner_parsl_alcf",
                                        "helpers.refine_list_of_jobs", "helpers.sweep_truncated_steps"])
    def test_cli_flag(self, module):
        import subprocess
        import sys
        out = subprocess.run([sys.executable, "-m", f"qtaim_gen.source.scripts.{module}", "--help"],
                             capture_output=True, text=True)
        # the option line itself (2-space indent): wrapped help text may also start with the flag
        assert re.search(r"^  --recheck_cp_labels\b", out.stdout, re.M), out.stderr[-500:]
