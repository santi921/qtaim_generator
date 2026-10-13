"""--recheck_allalpha_qtaim: all-alpha qtaim.json records are rerun from a .wfx (validator and restart gate agree)."""

import json
import logging

import pytest

from qtaim_gen.source.core.omol import _prepare_allalpha_qtaim_rerun, _qtaim_output_complete, gbw_analysis
from qtaim_gen.source.utils.validation import (
    QTAIM_ALL_ALPHA, QTAIM_ALL_ALPHA_ECP, QTAIM_AMBIGUOUS, QTAIM_RESOLVED, all_electron_count,
    qtaim_all_alpha_defect, qtaim_copy_has_all_alpha_defect, qtaim_spin_class, validate_qtaim_dict)

LOG = logging.getLogger("test_recheck_allalpha")


def _cp(total, alpha, beta):
    return {"density_all": total, "density_alpha": alpha, "density_beta": beta, "spin_density": alpha - beta}


ALL_ALPHA = {"0": _cp(0.4, 0.4, 0.0), "1": _cp(0.3, 0.3, 0.0), "0_1": _cp(0.25, 0.25, 0.0)}
RESOLVED = {"0": _cp(0.4, 0.2, 0.2), "1": _cp(0.3, 0.15, 0.15), "0_1": _cp(0.25, 0.125, 0.125)}
PARTLY = {"0": _cp(0.4, 0.4, 0.0), "1": _cp(0.3, 0.15, 0.15), "0_1": _cp(0.25, 0.125, 0.125)}
# OH: 9 electrons; as a "mult 10" every electron would be alpha (only the arithmetic matters here)
INP = "! {ref} wB97M-V\n*xyz {charge} {mult}\nO 0.0 0.0 0.0\nH 0.0 0.0 0.97\n*\n"
SET_ASIDE = ".allalpha"


# a well-formed CPprop.txt (validation.cpprop_integrity checks loose copies) with the one
# (3,-1) block _banner reports
CPPROP_TEXT = "".join(
    f" ----------------   CP{n:>6},     Type {kind}   ----------------\n"
    " Position (Bohr):      0.000000000000    0.000000000000    0.000000000000\n"
    " Density of all electrons:  0.1000000000E+00\n"
    " Norm of gradient is:  0.1000000000E-14\n"
    f" Eigenvalues of Hessian: {eig}\n"
    " Determinant of Hessian:  0.6000000000E-02\n"
    for n, kind, eig in ((1, "(3,-3)", "-0.3E+00 -0.2E+00 -0.1E+00"), (2, "(3,-3)", "-0.3E+00 -0.2E+00 -0.1E+00"),
                         (3, "(3,-1)", "-0.3E+00 -0.2E+00  0.1E+00"))
)


def _folder(tmp_path, record, mult=1, charge=0, gbw=True, wfn=True, cpprop=True, ref="UKS"):
    job = tmp_path / "job"
    (job / "generator").mkdir(parents=True)
    (job / "generator" / "qtaim.json").write_text(json.dumps(record))
    (job / "orca.inp").write_text(INP.format(ref=ref, charge=charge, mult=mult))
    if gbw:
        (job / "orca.gbw").write_bytes(b"gbw")
    if wfn:
        (job / "orca.wfn").write_text("wfn")
        (job / "generator" / "orca.wfn").write_text("wfn")
    if cpprop:
        (job / "CPprop.txt").write_text(CPPROP_TEXT)
        (job / "generator" / "CPprop.txt").write_text(CPPROP_TEXT)
    return job


class TestClassification:

    def test_classes(self):
        assert qtaim_spin_class(ALL_ALPHA) == QTAIM_ALL_ALPHA
        assert qtaim_spin_class(RESOLVED) == QTAIM_RESOLVED
        assert qtaim_spin_class(PARTLY) == QTAIM_AMBIGUOUS

    def test_rounding_noise_counts_as_zero(self):
        # seen in production: 5.7e-10 at an N nucleus (density 199), 1.6e-11 at Br (28,789)
        noisy = {"0": dict(_cp(199.24, 199.24, 5.7e-10), element="N"),
                 "1": dict(_cp(28789.3, 28789.3, 1.6e-11), element="Br"), "0_1": _cp(0.25, 0.25, -3e-17)}
        assert qtaim_spin_class(noisy) == QTAIM_ALL_ALPHA

    def test_ecp_nucleus_with_split_core_density(self):
        w = {"0": dict(_cp(3.1295e6, 1564749.281, 1564749.281), element="W"),
             "1": dict(_cp(122.6, 122.6, 3.5e-14), element="C"), "0_1": _cp(0.12, 0.12, 0.0)}
        assert qtaim_spin_class(w) == QTAIM_ALL_ALPHA_ECP
        assert qtaim_all_alpha_defect(w, n_electrons=80, mult=1)

    def test_ecp_nucleus_without_edf_is_plain_all_alpha(self):
        w = {"0": dict(_cp(5000.0, 5000.0, 0.0), element="W"), "1": dict(_cp(122.6, 122.6, 0.0), element="C")}
        assert qtaim_spin_class(w) == QTAIM_ALL_ALPHA

    def test_resolved_record_with_ecp_atoms_stays_resolved(self):
        w = {"0": dict(_cp(3.1e6, 1.55e6, 1.55e6), element="W"), "1": dict(_cp(122.6, 61.3, 61.3), element="C")}
        assert qtaim_spin_class(w) == QTAIM_RESOLVED

    def test_light_nucleus_with_beta_is_not_exempt(self):
        mixed = {"0": dict(_cp(122.6, 61.3, 61.3), element="C"), "1": dict(_cp(0.4, 0.4, 0.0), element="H")}
        assert qtaim_spin_class(mixed) == QTAIM_AMBIGUOUS

    def test_stale_all_alpha_self_pair_in_a_resolved_record(self):
        # rgd_uks: mapper-bug pairs like '10_10' carried over from an older all-alpha run
        stale = dict(RESOLVED, **{"1_1": _cp(0.215, 0.215, 2.8e-17)})
        assert qtaim_spin_class(stale) == QTAIM_AMBIGUOUS

    def test_defects(self):
        assert qtaim_all_alpha_defect(ALL_ALPHA, n_electrons=10, mult=1)
        assert qtaim_all_alpha_defect(PARTLY, n_electrons=10, mult=1)
        assert not qtaim_all_alpha_defect(RESOLVED, n_electrons=10, mult=1)

    def test_every_electron_alpha_is_right(self):
        assert not qtaim_all_alpha_defect(ALL_ALPHA, n_electrons=1, mult=2)  # H atom
        assert not qtaim_all_alpha_defect(ALL_ALPHA, n_electrons=2, mult=3)  # triplet H2
        assert qtaim_all_alpha_defect(PARTLY, n_electrons=1, mult=2)

    def test_unknown_counts_are_a_defect(self):
        assert qtaim_all_alpha_defect(ALL_ALPHA, n_electrons=None, mult=2)
        assert qtaim_all_alpha_defect(ALL_ALPHA, n_electrons=1, mult=None)

    def test_all_electron_count(self):
        mol = {0: {"element": "O"}, 1: {"element": "H"}, 2: {"element": "Pt"}}
        assert all_electron_count({"mol": mol, "charge": -1}) == 8 + 1 + 78 + 1
        assert all_electron_count({"mol": {0: {"element": "Fe1"}}, "charge": 2}) == 24
        assert all_electron_count({"mol": {0: {"element": "Xx"}}}) is None


class TestValidatorAndRestartGateAgree:

    @pytest.mark.parametrize("record, mult, rejected", [
        (ALL_ALPHA, 1, True),
        (PARTLY, 1, True),
        (RESOLVED, 1, False),
        (ALL_ALPHA, 3, True),
        (ALL_ALPHA, 10, False),  # 9 electrons, mult 10: every electron alpha
    ])
    def test_same_verdict(self, tmp_path, record, mult, rejected):
        (tmp_path / "qtaim.json").write_text(json.dumps(record))
        n_e = 9
        defect = qtaim_copy_has_all_alpha_defect(str(tmp_path), n_electrons=n_e, mult=mult)
        gate = _qtaim_output_complete(str(tmp_path), n_atoms=2, recheck_allalpha_qtaim=True,
                                      n_electrons=n_e, mult=mult)
        assert defect is rejected and gate is (not rejected)

    def test_off_by_default(self, tmp_path):
        path = tmp_path / "qtaim.json"
        path.write_text(json.dumps(ALL_ALPHA))
        assert validate_qtaim_dict(str(path), n_atoms=2)
        assert _qtaim_output_complete(str(tmp_path), n_atoms=2)


class TestPrepareRerun:

    def test_all_alpha_folder_sets_the_wfn_aside_and_loses_cpprop(self, tmp_path):
        job = _folder(tmp_path, ALL_ALPHA)
        assert _prepare_allalpha_qtaim_rerun(str(job), wfx=True, preprocess_compressed=False, logger=LOG)
        for rel in ("orca.wfn", "generator/orca.wfn", "CPprop.txt", "generator/CPprop.txt"):
            assert not (job / rel).exists()
        assert (job / ("orca.wfn" + SET_ASIDE)).read_text() == "wfn"
        assert (job / "generator" / ("orca.wfn" + SET_ASIDE)).exists()
        assert (job / "orca.gbw").exists() and (job / "generator" / "qtaim.json").exists()

    def test_sound_unrestricted_folder_is_untouched(self, tmp_path):
        # setting its .wfn aside would make extraction unpack a folder that then returns early as valid
        job = _folder(tmp_path, RESOLVED)
        assert _prepare_allalpha_qtaim_rerun(str(job), wfx=True, preprocess_compressed=False, logger=LOG)
        assert (job / "orca.wfn").exists() and (job / "CPprop.txt").exists()
        assert not (job / ("orca.wfn" + SET_ASIDE)).exists()

    def test_restricted_singlet_keeps_its_wfn(self, tmp_path):
        job = _folder(tmp_path, RESOLVED, ref="RKS")
        assert _prepare_allalpha_qtaim_rerun(str(job), wfx=True, preprocess_compressed=False, logger=LOG)
        assert (job / "orca.wfn").exists() and (job / "CPprop.txt").exists()

    def test_sound_unrestricted_folder_without_a_source_keeps_its_wfn(self, tmp_path):
        job = _folder(tmp_path, RESOLVED, gbw=False)
        assert _prepare_allalpha_qtaim_rerun(str(job), wfx=True, preprocess_compressed=False, logger=LOG)
        assert (job / "orca.wfn").exists()

    def test_every_electron_alpha_keeps_its_outputs(self, tmp_path):
        job = _folder(tmp_path, ALL_ALPHA, mult=10)
        assert _prepare_allalpha_qtaim_rerun(str(job), wfx=True, preprocess_compressed=False, logger=LOG)
        assert (job / "CPprop.txt").exists() and (job / "orca.wfn").exists()

    @pytest.mark.parametrize("kwargs, wfx", [(dict(gbw=False), True), (dict(), False)],
                             ids=["no_gbw_source", "no_wfx_flag"])
    def test_refuses_when_the_rerun_would_reproduce_the_record(self, tmp_path, kwargs, wfx):
        job = _folder(tmp_path, ALL_ALPHA, **kwargs)
        assert not _prepare_allalpha_qtaim_rerun(str(job), wfx=wfx, preprocess_compressed=False, logger=LOG)
        assert (job / "orca.wfn").exists() and (job / "CPprop.txt").exists()

    def test_existing_wfx_is_enough(self, tmp_path):
        job = _folder(tmp_path, ALL_ALPHA, gbw=False)
        (job / "orca.wfx").write_text("wfx")
        assert _prepare_allalpha_qtaim_rerun(str(job), wfx=True, preprocess_compressed=False, logger=LOG)
        assert (job / "orca.wfx").exists() and not (job / "orca.wfn").exists()
        assert (job / ("orca.wfn" + SET_ASIDE)).exists()

    def test_gbw_analysis_stops_before_running_anything(self, tmp_path):
        job = _folder(tmp_path, ALL_ALPHA, gbw=False)
        ok = gbw_analysis(str(job), multiwfn_cmd="/nonexistent/Multiwfn", orca_2mkl_cmd="/nonexistent/orca_2mkl",
                          restart=True, overwrite=False, logger=LOG, wfx=True, recheck_allalpha_qtaim=True)
        assert ok is False
        assert (job / "orca.wfn").exists() and (job / "generator" / "qtaim.json").exists()
        assert not (job / "settings.ini").exists()  # returned before any job was set up

    def test_gbw_analysis_clears_the_stale_inputs_before_running(self, tmp_path):
        job = _folder(tmp_path, ALL_ALPHA)
        gbw_analysis(str(job), multiwfn_cmd="/nonexistent/Multiwfn", orca_2mkl_cmd="/nonexistent/orca_2mkl",
                     restart=True, overwrite=False, logger=LOG, wfx=True, recheck_allalpha_qtaim=True,
                     move_results=True)
        # generator/ paths: clean_jobs never touches them, so only the preparation step explains these
        assert not (job / "generator" / "orca.wfn").exists()
        assert (job / "generator" / ("orca.wfn" + SET_ASIDE)).exists()
        assert not (job / "generator" / "CPprop.txt").exists()
        # orca_2mkl failed: the gbw is the only source left and must survive for the next pass
        assert (job / "orca.gbw").exists()
        assert _prepare_allalpha_qtaim_rerun(str(job), wfx=True, preprocess_compressed=False, logger=LOG)


class TestCopies:
    """A stale root copy (clean=False) next to generator/: every check reads both."""

    @pytest.mark.parametrize("root,gen", [(RESOLVED, ALL_ALPHA), (ALL_ALPHA, RESOLVED)])
    def test_any_defective_copy_rejects_everywhere(self, tmp_path, monkeypatch, root, gen):
        from qtaim_gen.source.utils import validation
        job = _folder(tmp_path, gen)
        (job / "qtaim.json").write_text(json.dumps(root))
        for name in ("timings", "fuzzy_full", "other", "charge", "bond"):
            (job / "generator" / f"{name}.json").write_text("{}")
        for check in ("validate_timing_dict", "validate_fuzzy_dict", "validate_other_dict",
                      "validate_charge_dict", "validate_bond_dict"):
            monkeypatch.setattr(validation, check, lambda *a, **k: True)
        assert not _qtaim_output_complete(str(job), n_atoms=2, recheck_allalpha_qtaim=True, n_electrons=9, mult=1)
        assert not validation.validation_checks(str(job), move_results=True, recheck_allalpha_qtaim=True)
        assert _prepare_allalpha_qtaim_rerun(str(job), wfx=True, preprocess_compressed=False, logger=LOG)
        assert not (job / "orca.wfn").exists() and not (job / "CPprop.txt").exists()

    def test_both_copies_resolved_pass(self, tmp_path, monkeypatch):
        from qtaim_gen.source.utils import validation
        job = _folder(tmp_path, RESOLVED)
        (job / "qtaim.json").write_text(json.dumps(RESOLVED))
        for name in ("timings", "fuzzy_full", "other", "charge", "bond"):
            (job / "generator" / f"{name}.json").write_text("{}")
        for check in ("validate_timing_dict", "validate_fuzzy_dict", "validate_other_dict",
                      "validate_charge_dict", "validate_bond_dict"):
            monkeypatch.setattr(validation, check, lambda *a, **k: True)
        assert _qtaim_output_complete(str(job), n_atoms=2, recheck_allalpha_qtaim=True, n_electrons=9, mult=1)
        assert validation.validation_checks(str(job), move_results=True, recheck_allalpha_qtaim=True)


class TestRunJobsGate:
    """run_jobs' per-step restart decision for qtaim, through the real call (n_electrons/mult forwarded)."""

    def _run(self, tmp_path, caplog, record, mult, flag):
        from qtaim_gen.source.core.omol import run_jobs
        job = _folder(tmp_path, record, mult=mult)
        (job / "qtaim.json").write_text(json.dumps(record))
        (job / "timings.json").write_text(json.dumps({"qtaim": 1.0}))
        with caplog.at_level(logging.INFO, logger=LOG.name):
            run_jobs(str(job), separate=False, restart=True, debug=True, logger=LOG,
                     recheck_allalpha_qtaim=flag)
        skipped = any("Skipping qtaim" in r.getMessage() for r in caplog.records)
        # a step that ran rewrites its timing; a skipped one keeps 1.0
        rewritten = json.loads((job / "timings.json").read_text()).get("qtaim") != 1.0
        assert skipped is not rewritten
        return skipped

    def test_all_alpha_is_skipped_without_the_flag(self, tmp_path, caplog):
        assert self._run(tmp_path, caplog, ALL_ALPHA, mult=1, flag=False)

    def test_all_alpha_reruns_under_the_flag(self, tmp_path, caplog):
        assert not self._run(tmp_path, caplog, ALL_ALPHA, mult=1, flag=True)

    @pytest.mark.parametrize("ref, mult, wfx, refused", [
        ("UKS", 1, False, True),    # UKS singlet, only the .wfn
        ("RKS", 3, False, True),    # open shell is unrestricted whatever the keyword
        ("RKS", 1, False, False),   # restricted .wfn is fine
        ("UKS", 1, True, False),    # a .wfx is read first
    ])
    def test_qtaim_never_runs_from_an_unrestricted_wfn(self, tmp_path, caplog, ref, mult, wfx, refused):
        # the record is sound but QTAIM reruns for another reason (here: no qtaim.json at all)
        from qtaim_gen.source.core.omol import run_jobs
        job = _folder(tmp_path, RESOLVED, mult=mult, ref=ref)
        (job / "generator" / "qtaim.json").unlink()
        if wfx:
            (job / "orca.wfx").write_text("wfx")
        (job / "timings.json").write_text(json.dumps({"qtaim": 1.0}))
        with caplog.at_level(logging.INFO, logger=LOG.name):
            run_jobs(str(job), separate=False, restart=True, debug=True, logger=LOG, recheck_allalpha_qtaim=True)
        said = any("refusing to run qtaim" in r.getMessage() for r in caplog.records)
        assert said is refused
        if refused:
            assert json.loads((job / "timings.json").read_text())["qtaim"] == -1

    def test_every_electron_alpha_is_still_skipped(self, tmp_path, caplog):
        # OH as "mult 10": all 9 electrons alpha, so the record is right (needs n_electrons and mult)
        assert self._run(tmp_path, caplog, ALL_ALPHA, mult=10, flag=True)


class TestPlumbing:

    def test_step_output_check_passes_the_flag(self, tmp_path):
        from qtaim_gen.source.core.omol import _has_usable_step_output
        (tmp_path / "qtaim.json").write_text(json.dumps(ALL_ALPHA))
        assert _has_usable_step_output(str(tmp_path), "qtaim", n_atoms=2)
        assert not _has_usable_step_output(str(tmp_path), "qtaim", n_atoms=2, recheck_allalpha_qtaim=True,
                                           n_electrons=9, mult=1)

    def test_validation_checks_rejects_under_the_flag(self, tmp_path, monkeypatch):
        from qtaim_gen.source.utils import validation
        job = _folder(tmp_path, ALL_ALPHA)
        for name in ("timings", "fuzzy_full", "other", "charge", "bond"):
            (job / "generator" / f"{name}.json").write_text("{}")
        for check in ("validate_timing_dict", "validate_fuzzy_dict", "validate_other_dict",
                      "validate_charge_dict", "validate_bond_dict"):
            monkeypatch.setattr(validation, check, lambda *a, **k: True)
        assert validation.validation_checks(str(job), move_results=True)
        assert not validation.validation_checks(str(job), move_results=True, recheck_allalpha_qtaim=True)

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
        io.get_folders_from_file(str(lst), num_folders=10, pre_validate=True, recheck_allalpha_qtaim=True)
        assert seen.get("recheck_allalpha_qtaim") is True

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
        workflow.process_folder(str(job), move_results=False, recheck_allalpha_qtaim=True)
        assert seen.get("recheck_allalpha_qtaim") is True

    @pytest.mark.parametrize("module", ["full_runner", "full_runner_parsl", "full_runner_parsl_alcf",
                                        "helpers.refine_list_of_jobs"])
    def test_cli_flag(self, module):
        import subprocess
        import sys
        out = subprocess.run([sys.executable, "-m", f"qtaim_gen.source.scripts.{module}", "--help"],
                             capture_output=True, text=True)
        assert "--recheck_allalpha_qtaim" in out.stdout, out.stderr[-500:]


def _banner(alpha, beta, finished=True):
    text = f" Total/Alpha/Beta electrons:  {alpha + beta:.4f}  {alpha:.4f}  {beta:.4f}\n"
    if finished:
        text += (" Number of (3,-1) CPs:     1\n"
                 " Done! The results have been outputted to CPprop.txt in current folder\n")
    return text


class TestBanner:
    """With a qtaim.out the banner decides (no loop: a .wfx rerun shows beta > 0)."""

    def test_banner_is_read_loose_and_from_the_archive(self, tmp_path):
        import zipfile
        from qtaim_gen.source.utils.validation import qtaim_out_banner
        (tmp_path / "generator").mkdir()
        assert qtaim_out_banner(str(tmp_path)) is None
        with zipfile.ZipFile(tmp_path / "generator" / "out_files.zip", "w") as zf:
            zf.writestr("qtaim.out", _banner(9, 0))
        assert qtaim_out_banner(str(tmp_path)) == (9.0, 0.0)
        (tmp_path / "qtaim.out").write_text(_banner(5, 4))
        assert qtaim_out_banner(str(tmp_path)) == (5.0, 4.0)

    def test_root_archive_is_read_without_move_results(self, tmp_path):
        import zipfile
        from qtaim_gen.source.utils.validation import qtaim_out_banner
        with zipfile.ZipFile(tmp_path / "out_files.zip", "w") as zf:
            zf.writestr("qtaim.out", _banner(9, 0))
        assert qtaim_out_banner(str(tmp_path)) == (9.0, 0.0)

    def test_unfinished_run_does_not_supply_the_banner(self, tmp_path):
        # a .wfx rerun killed mid-search printed beta > 0; the archived all-alpha run decides
        import zipfile
        from qtaim_gen.source.utils.validation import qtaim_out_banner
        (tmp_path / "generator").mkdir()
        (tmp_path / "generator" / "qtaim.json").write_text(json.dumps(ALL_ALPHA))
        with zipfile.ZipFile(tmp_path / "generator" / "out_files.zip", "w") as zf:
            zf.writestr("qtaim.out", _banner(9, 0))
        (tmp_path / "qtaim.out").write_text(_banner(5, 4, finished=False))
        assert qtaim_out_banner(str(tmp_path)) == (9.0, 0.0)
        assert qtaim_copy_has_all_alpha_defect(str(tmp_path), n_electrons=9, mult=1)
        assert not _qtaim_output_complete(str(tmp_path), recheck_allalpha_qtaim=True, n_electrons=9, mult=1)

    @pytest.mark.parametrize("record, banner, mult, rejected", [
        (ALL_ALPHA, (9, 0), 1, True),                       # all-alpha run
        (PARTLY, (9, 0), 1, True),
        ({"0": _cp(0.4, 0.2000001, 0.1999999)}, (9, 0), 1, True),  # resolved-looking, but not the fix's exact zeros
        (ALL_ALPHA, (9, 0), 10, False),                     # every electron alpha
        (ALL_ALPHA, (9, 0), None, False),                   # multiplicity unknown: a rerun could not settle it
        (ALL_ALPHA, (5, 4), 2, True),                       # resolved run, all-alpha record: not from that run
        (RESOLVED, (5, 4), 2, False),                       # the rerun's own record
        (PARTLY, (5, 4), 2, False),                         # stale CPs: a rerun list, never the gate
    ])
    def test_validator_and_gate_follow_the_banner(self, tmp_path, record, banner, mult, rejected):
        (tmp_path / "qtaim.json").write_text(json.dumps(record))
        (tmp_path / "qtaim.out").write_text(_banner(*banner))
        defect = qtaim_copy_has_all_alpha_defect(str(tmp_path), n_electrons=9, mult=mult)
        gate = _qtaim_output_complete(str(tmp_path), recheck_allalpha_qtaim=True, n_electrons=9, mult=mult)
        assert defect is rejected and gate is (not rejected)

    def test_records_fixed_in_place_are_not_rerun(self, tmp_path, monkeypatch):
        # the archived qtaim.out of a fixed record still shows beta == 0
        from qtaim_gen.source.scripts.helpers.fix_allalpha_qtaim import fix_record
        from qtaim_gen.source.utils import validation
        job = _folder(tmp_path, fix_record(ALL_ALPHA))
        (job / "generator" / "qtaim.out").write_text(_banner(9, 0))
        for name in ("timings", "fuzzy_full", "other", "charge", "bond"):
            (job / "generator" / f"{name}.json").write_text("{}")
        for check in ("validate_timing_dict", "validate_fuzzy_dict", "validate_other_dict",
                      "validate_charge_dict", "validate_bond_dict"):
            monkeypatch.setattr(validation, check, lambda *a, **k: True)
        assert validation.validation_checks(str(job), move_results=True, recheck_allalpha_qtaim=True)
        assert _qtaim_output_complete(str(job), recheck_allalpha_qtaim=True, n_electrons=9, mult=1)
        assert _prepare_allalpha_qtaim_rerun(str(job), wfx=True, preprocess_compressed=False, logger=LOG)
        # not a defect: nothing is touched
        assert (job / "generator" / "qtaim.out").exists() and (job / "CPprop.txt").exists()
        assert (job / "orca.wfn").exists()

    def test_rerun_preparation_removes_the_loose_qtaim_out(self, tmp_path):
        job = _folder(tmp_path, ALL_ALPHA)
        (job / "qtaim.out").write_text(_banner(9, 0))
        (job / "generator" / "qtaim.out").write_text(_banner(9, 0))
        assert _prepare_allalpha_qtaim_rerun(str(job), wfx=True, preprocess_compressed=False, logger=LOG)
        assert not (job / "qtaim.out").exists() and not (job / "generator" / "qtaim.out").exists()


class TestReviewFixes:
    """Pre-existing runner paths the set-aside .wfn made dangerous, and the cleanup tools."""

    def test_clean_jobs_removes_the_set_aside_wfn(self, tmp_path):
        from qtaim_gen.source.core.omol import clean_jobs
        job = _folder(tmp_path, RESOLVED)
        (job / ("orca.wfn" + SET_ASIDE)).write_text("wfn")
        (job / "generator" / ("orca.wfn" + SET_ASIDE)).write_text("wfn")
        clean_jobs(str(job), logger=LOG, move_results=True)
        assert not (job / ("orca.wfn" + SET_ASIDE)).exists()
        assert not (job / "generator" / ("orca.wfn" + SET_ASIDE)).exists()

    def test_clean_omol_deletes_the_set_aside_wfn(self):
        from qtaim_gen.source.scripts.helpers.clean_omol import should_delete
        assert should_delete("orca.wfn" + SET_ASIDE)
        assert not should_delete("orca.wfn") and not should_delete("orca.wfx")

    def test_failed_extraction_keeps_the_compressed_gbw(self, tmp_path):
        job = tmp_path / "job"
        job.mkdir()
        (job / "orca.inp").write_text(INP.format(ref="UKS", charge=0, mult=2))
        (job / "orca.gbw.zstd0").write_bytes(b"not a zstd stream")
        gbw_analysis(str(job), multiwfn_cmd="/nonexistent/Multiwfn", orca_2mkl_cmd="/nonexistent/orca_2mkl",
                     logger=LOG, preprocess_compressed=True, move_results=True)
        assert not (job / "orca.gbw").exists()
        assert (job / "orca.gbw.zstd0").read_bytes() == b"not a zstd stream"

    def test_patched_timings_do_not_pass_a_folder_that_fails_otherwise(self, tmp_path, monkeypatch):
        from qtaim_gen.source.core import omol
        calls = []

        def failing(*a, **k):
            calls.append(k)
            return False
        monkeypatch.setattr(omol, "validation_checks", failing)
        monkeypatch.setattr(omol, "patch_timings_from_log", lambda *a, **k: True)
        job = _folder(tmp_path, ALL_ALPHA)
        ok = gbw_analysis(str(job), multiwfn_cmd="/nonexistent/Multiwfn", orca_2mkl_cmd="/nonexistent/orca_2mkl",
                          logger=LOG, parse_only=True, patch_timings=True, move_results=True)
        assert ok is False and len(calls) >= 2  # validated again after the patch

    def test_sweep_predicts_with_the_flag(self, tmp_path, monkeypatch):
        from qtaim_gen.source.scripts.helpers import sweep_truncated_steps as sweep
        seen = []
        monkeypatch.setattr(sweep, "validation_checks", lambda *a, **k: seen.append(("validation", k)) or False)

        def step(folder, op, **k):
            seen.append((op, k))
            return True
        monkeypatch.setattr(sweep, "_has_usable_step_output", step)
        monkeypatch.setattr(sweep, "_compiled_data_present", lambda *a, **k: False)
        job = _folder(tmp_path, ALL_ALPHA)
        sweep.classify_folder(str(job), None, None, full_set=0, move_results=True, recheck_allalpha_qtaim=True)
        assert seen and all(k.get("recheck_allalpha_qtaim") is True for _, k in seen)
        qtaim_call = [k for op, k in seen if op == "qtaim"][0]
        assert qtaim_call["n_electrons"] == 9 and qtaim_call["mult"] == 1
