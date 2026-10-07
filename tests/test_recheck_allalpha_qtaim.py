"""--recheck_allalpha_qtaim: all-alpha qtaim.json records are rerun from a .wfx (validator and restart gate agree)."""

import json
import logging

import pytest

from qtaim_gen.source.core.omol import _prepare_allalpha_qtaim_rerun, _qtaim_output_complete, gbw_analysis
from qtaim_gen.source.utils.validation import (
    QTAIM_ALL_ALPHA, QTAIM_ALL_ALPHA_ECP, QTAIM_AMBIGUOUS, QTAIM_RESOLVED, all_electron_count,
    qtaim_all_alpha_defect, qtaim_spin_class, validate_qtaim_dict)

LOG = logging.getLogger("test_recheck_allalpha")


def _cp(total, alpha, beta):
    return {"density_all": total, "density_alpha": alpha, "density_beta": beta, "spin_density": alpha - beta}


ALL_ALPHA = {"0": _cp(0.4, 0.4, 0.0), "1": _cp(0.3, 0.3, 0.0), "0_1": _cp(0.25, 0.25, 0.0)}
RESOLVED = {"0": _cp(0.4, 0.2, 0.2), "1": _cp(0.3, 0.15, 0.15), "0_1": _cp(0.25, 0.125, 0.125)}
PARTLY = {"0": _cp(0.4, 0.4, 0.0), "1": _cp(0.3, 0.15, 0.15), "0_1": _cp(0.25, 0.125, 0.125)}
# OH: 9 electrons; as a "mult 10" every electron would be alpha (only the arithmetic matters here)
INP = "! UKS wB97M-V\n*xyz {charge} {mult}\nO 0.0 0.0 0.0\nH 0.0 0.0 0.97\n*\n"


def _folder(tmp_path, record, mult=1, charge=0, gbw=True, wfn=True, cpprop=True):
    job = tmp_path / "job"
    (job / "generator").mkdir(parents=True)
    (job / "generator" / "qtaim.json").write_text(json.dumps(record))
    (job / "orca.inp").write_text(INP.format(charge=charge, mult=mult))
    if gbw:
        (job / "orca.gbw").write_bytes(b"gbw")
    if wfn:
        (job / "orca.wfn").write_text("wfn")
        (job / "generator" / "orca.wfn").write_text("wfn")
    if cpprop:
        (job / "CPprop.txt").write_text("old")
        (job / "generator" / "CPprop.txt").write_text("old")
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
        path = tmp_path / "qtaim.json"
        path.write_text(json.dumps(record))
        n_e = 9
        valid = validate_qtaim_dict(str(path), n_atoms=2, reject_all_alpha=True, n_electrons=n_e, mult=mult)
        gate = _qtaim_output_complete(str(tmp_path), n_atoms=2, recheck_allalpha_qtaim=True,
                                      n_electrons=n_e, mult=mult)
        assert valid is (not rejected) and gate is (not rejected)

    def test_off_by_default(self, tmp_path):
        path = tmp_path / "qtaim.json"
        path.write_text(json.dumps(ALL_ALPHA))
        assert validate_qtaim_dict(str(path), n_atoms=2)
        assert _qtaim_output_complete(str(tmp_path), n_atoms=2)


class TestPrepareRerun:

    def test_all_alpha_folder_loses_wfn_and_cpprop(self, tmp_path):
        job = _folder(tmp_path, ALL_ALPHA)
        assert _prepare_allalpha_qtaim_rerun(str(job), wfx=True, preprocess_compressed=False, logger=LOG)
        for rel in ("orca.wfn", "generator/orca.wfn", "CPprop.txt", "generator/CPprop.txt"):
            assert not (job / rel).exists()
        assert (job / "orca.gbw").exists() and (job / "generator" / "qtaim.json").exists()

    def test_resolved_folder_is_untouched(self, tmp_path):
        job = _folder(tmp_path, RESOLVED)
        assert _prepare_allalpha_qtaim_rerun(str(job), wfx=True, preprocess_compressed=False, logger=LOG)
        assert (job / "orca.wfn").exists() and (job / "CPprop.txt").exists()

    def test_every_electron_alpha_is_untouched(self, tmp_path):
        job = _folder(tmp_path, ALL_ALPHA, mult=10)
        assert _prepare_allalpha_qtaim_rerun(str(job), wfx=True, preprocess_compressed=False, logger=LOG)
        assert (job / "orca.wfn").exists()

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
        for rel in ("orca.wfn", "generator/orca.wfn", "CPprop.txt", "generator/CPprop.txt"):
            assert not (job / rel).exists(), rel


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

    @pytest.mark.parametrize("module", ["full_runner", "full_runner_parsl", "full_runner_parsl_alcf",
                                        "helpers.refine_list_of_jobs"])
    def test_cli_flag(self, module):
        import subprocess
        import sys
        out = subprocess.run([sys.executable, "-m", f"qtaim_gen.source.scripts.{module}", "--help"],
                             capture_output=True, text=True)
        assert "--recheck_allalpha_qtaim" in out.stdout, out.stderr[-500:]
