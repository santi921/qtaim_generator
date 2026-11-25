"""DDEC6 in the runner: omol._run_ddec6 / _merge_ddec6, the ddec6.json move, and
validation.validate_ddec6. The fast tests stub ddec6.analyze; the output-layout test
runs the real port (DDEC6_SLOW=1).
"""

import json
import logging
import os
import shutil

import pytest

from qtaim_gen.source.core import ddec6, omol
from qtaim_gen.source.data.engine_versions import ENGINE_VERSION, last_change
from qtaim_gen.source.utils import provenance
from qtaim_gen.source.utils.validation import validate_ddec6

FIXTURE = os.path.join(os.path.dirname(__file__), "test_files", "charge_engine", "rmechdb_652_step10_0_2")
LABELS = ["1_Br", "2_O", "3_O", "4_O"]
LOG = logging.getLogger("test_ddec6_runner")
DDEC6_FUZZY = ("ddec6_atomic_dipole_mag", "ddec6_quadrupole_mag", "ddec6_tail_decay", "ddec6_sbo", "ddec6_r2",
               "ddec6_r3", "ddec6_r4")


def _fake_result():
    q = {k: v for k, v in zip(LABELS, (0.15, -0.14, 0.13, -0.14))}
    feats = {name: {**{k: 0.1 for k in LABELS}, "sum": 0.4, "abs_sum": 0.4} for name in DDEC6_FUZZY}
    bonds = {"ddec6_bond": {"1_Br_to_2_O": 0.82}, "ddec6_overlap_population": {"1_Br_to_2_O": 0.54}}
    return {"ddec6": {"charge": q, "dipole": {"mag": 0.0, "xyz": [0.0, 0.0, 0.0]}}, "fuzzy": feats, "bond": bonds,
            "_meta": {"n_iterations": 7, "converged": True, "natoms": 4}}


@pytest.fixture
def folder(tmp_path):
    shutil.copy(os.path.join(FIXTURE, "orca.wfx"), tmp_path / "orca.wfx")
    return str(tmp_path)


def _timings(folder):
    with open(os.path.join(folder, "timings.json")) as f:
        return json.load(f)


def test_run_writes_result_timing_and_version(folder, monkeypatch):
    calls = []
    monkeypatch.setattr(ddec6, "analyze", lambda wfx, **kw: calls.append(wfx) or _fake_result())
    assert omol._run_ddec6(folder, n_threads=1, logger=LOG)
    with open(os.path.join(folder, omol.DDEC6_JSON)) as f:
        assert json.load(f)["ddec6"]["charge"]["1_Br"] == 0.15
    assert _timings(folder)["ddec6"] > 0
    prov = provenance.load(folder)
    entry = prov["routines"]["ddec6"]
    assert entry["by"] == "ddec6" and entry["engine_version"] == ENGINE_VERSION
    assert entry["spec"] == {"chargemol": "09_26_2017", "grid_spacing_bohr": 0.14, "max_atoms": ddec6.MAX_ATOMS}
    assert prov["engines"]["ddec6"] == {"n_iterations": 7, "converged": True, "natoms": 4}
    # restart finds the completed run and does not recompute
    assert omol._run_ddec6(folder, n_threads=1, logger=LOG, restart=True)
    assert len(calls) == 1


def test_restart_recomputes_a_result_from_an_older_version(folder, monkeypatch):
    calls = []
    monkeypatch.setattr(ddec6, "analyze", lambda wfx, **kw: calls.append(wfx) or _fake_result())
    assert omol._run_ddec6(folder, n_threads=1, logger=LOG)
    provenance.record_routines(folder, ["ddec6"], "ddec6", last_change("ddec6") - 1)
    assert omol._run_ddec6(folder, n_threads=1, logger=LOG, restart=True)
    assert len(calls) == 2
    assert provenance.load(folder)["routines"]["ddec6"]["engine_version"] == ENGINE_VERSION


def test_size_cap_is_a_permanent_skip(folder, monkeypatch):
    monkeypatch.setattr(ddec6, "MAX_ATOMS", 3)
    monkeypatch.setattr(ddec6, "analyze", lambda wfx, **kw: pytest.fail("analyze must not run above the cap"))
    assert omol._run_ddec6(folder, n_threads=1, logger=LOG)
    assert provenance.load(folder)["skipped"]["ddec6"]["reason"] == "natoms_above_3"
    # a recorded skip is final: later passes do nothing
    assert omol._run_ddec6(folder, n_threads=1, logger=LOG)


def test_chargemol_stop_is_a_permanent_skip(folder, monkeypatch):
    def stop(wfx, **kw):
        raise ddec6.ChargemolStop("grid spacing check failed")
    monkeypatch.setattr(ddec6, "analyze", stop)
    assert omol._run_ddec6(folder, n_threads=1, logger=LOG)
    rec = provenance.load(folder)["skipped"]["ddec6"]
    assert rec["reason"] == "chargemol_stop: grid spacing check failed" and rec["by"] == "ddec6"


def test_other_errors_fail_without_a_skip(folder, monkeypatch):
    def boom(wfx, **kw):
        raise RuntimeError("numba exploded")
    monkeypatch.setattr(ddec6, "analyze", boom)
    assert not omol._run_ddec6(folder, n_threads=1, logger=LOG)
    assert not os.path.exists(os.path.join(folder, "timings.json"))
    assert "ddec6" not in provenance.load(folder)["routines"]


def test_merge_into_generator_copy_and_idempotent(folder):
    gen = os.path.join(folder, "generator")
    os.makedirs(gen)
    for name, data in (("charge.json", {"hirshfeld": {"charge": {k: 0.0 for k in LABELS}}}),
                       ("fuzzy_full.json", {"becke_fuzzy_density": {"1_Br": 35.0}}),
                       ("bond.json", {"fuzzy_bond": {"1_Br_to_2_O": 1.1}})):
        with open(os.path.join(gen, name), "w") as f:
            json.dump(data, f)
    with open(os.path.join(folder, omol.DDEC6_JSON), "w") as f:
        json.dump(_fake_result(), f)
    provenance.record_routines(folder, ["ddec6"], "ddec6", ENGINE_VERSION)
    for _ in range(2):
        omol._merge_ddec6(folder, move_results=True, logger=LOG)
    with open(os.path.join(gen, "charge.json")) as f:
        charge = json.load(f)
    with open(os.path.join(gen, "fuzzy_full.json")) as f:
        fuzzy = json.load(f)
    with open(os.path.join(gen, "bond.json")) as f:
        bond = json.load(f)
    assert set(charge) == {"hirshfeld", "ddec6"} and charge["ddec6"]["charge"]["2_O"] == -0.14
    assert set(fuzzy) == {"becke_fuzzy_density", *DDEC6_FUZZY}
    assert bond == {"fuzzy_bond": {"1_Br_to_2_O": 1.1}, **_fake_result()["bond"]}


def test_merge_into_root_without_generator(folder):
    with open(os.path.join(folder, "charge.json"), "w") as f:
        json.dump({"cm5": {"charge": {}}}, f)
    with open(os.path.join(folder, "fuzzy_full.json"), "w") as f:
        json.dump({}, f)
    with open(os.path.join(folder, omol.DDEC6_JSON), "w") as f:
        json.dump(_fake_result(), f)
    provenance.record_routines(folder, ["ddec6"], "ddec6", ENGINE_VERSION)
    omol._merge_ddec6(folder, move_results=True, logger=LOG)
    with open(os.path.join(folder, "charge.json")) as f:
        assert "ddec6" in json.load(f)


def test_merge_skips_a_result_provenance_does_not_vouch_for(folder):
    with open(os.path.join(folder, "charge.json"), "w") as f:
        json.dump({"cm5": {"charge": {}}}, f)
    with open(os.path.join(folder, "fuzzy_full.json"), "w") as f:
        json.dump({}, f)
    with open(os.path.join(folder, omol.DDEC6_JSON), "w") as f:
        json.dump(_fake_result(), f)
    omol._merge_ddec6(folder, move_results=True, logger=LOG)  # no provenance entry
    provenance.record_routines(folder, ["ddec6"], "ddec6", last_change("ddec6") - 1)
    omol._merge_ddec6(folder, move_results=True, logger=LOG)  # an older version
    with open(os.path.join(folder, "charge.json")) as f:
        assert "ddec6" not in json.load(f)


def test_failed_recompute_removes_the_earlier_result(folder, monkeypatch):
    monkeypatch.setattr(ddec6, "analyze", lambda wfx, **kw: _fake_result())
    assert omol._run_ddec6(folder, n_threads=1, logger=LOG)
    gen = os.path.join(folder, "generator")
    os.makedirs(gen)
    shutil.copy(os.path.join(folder, omol.DDEC6_JSON), os.path.join(gen, omol.DDEC6_JSON))
    shutil.copy(os.path.join(folder, "timings.json"), os.path.join(gen, "timings.json"))

    def boom(wfx, **kw):
        raise RuntimeError("new wavefunction, new failure")
    monkeypatch.setattr(ddec6, "analyze", boom)
    assert not omol._run_ddec6(folder, n_threads=1, logger=LOG)
    assert omol._ddec6_result_path(folder) is None
    for base in (folder, gen):
        with open(os.path.join(base, "timings.json")) as f:
            assert "ddec6" not in json.load(f)
    assert "ddec6" not in provenance.load(folder)["routines"]


def test_unconverged_result_is_stored_with_a_warning(folder, monkeypatch, caplog):
    res = _fake_result()
    res["_meta"].update(converged=False, n_iterations=500)
    monkeypatch.setattr(ddec6, "analyze", lambda wfx, **kw: res)
    with caplog.at_level(logging.WARNING, logger=LOG.name):
        assert omol._run_ddec6(folder, n_threads=1, logger=LOG)
    assert "without convergence" in caplog.text
    assert provenance.load(folder)["engines"]["ddec6"]["converged"] is False


def test_ddec6_forces_restart(tmp_path, monkeypatch, caplog):
    class Reached(Exception):
        pass

    def reached(*a, **k):
        raise Reached
    monkeypatch.setattr(omol, "check_results_exist", reached)
    with caplog.at_level(logging.INFO, logger=LOG.name), pytest.raises(Reached):
        omol.gbw_analysis(str(tmp_path), "Multiwfn", "orca_2mkl", overwrite=False, logger=LOG, ddec6=True)
    assert "forcing restart" in caplog.text


def test_ddec6_json_moves_to_generator(folder):
    with open(os.path.join(folder, omol.DDEC6_JSON), "w") as f:
        json.dump(_fake_result(), f)
    omol.move_results_to_folder(folder, logger=LOG, clean=True)
    assert os.path.isfile(os.path.join(folder, "generator", omol.DDEC6_JSON))
    assert not os.path.exists(os.path.join(folder, omol.DDEC6_JSON))
    assert omol._ddec6_result_path(folder) == os.path.join(folder, "generator", omol.DDEC6_JSON)


def test_validate_ddec6(tmp_path):
    charge, timings = str(tmp_path / "charge.json"), str(tmp_path / "timings.json")
    with open(charge, "w") as f:
        json.dump({"ddec6": {"charge": {k: 0.0 for k in LABELS}}}, f)
    with open(timings, "w") as f:
        json.dump({"ddec6": 12.5}, f)
    assert validate_ddec6(charge, timings, n_atoms=4)
    assert not validate_ddec6(charge, timings, n_atoms=5)
    bond = str(tmp_path / "bond.json")
    with open(bond, "w") as f:
        json.dump({"fuzzy_bond": {}}, f)
    assert not validate_ddec6(charge, timings, n_atoms=4, bond_json_loc=bond)
    with open(bond, "w") as f:
        json.dump({"fuzzy_bond": {}, "ddec6_bond": {}}, f)  # empty: no pair at or above the print cutoff
    assert validate_ddec6(charge, timings, n_atoms=4, bond_json_loc=bond)
    fuzzy = str(tmp_path / "fuzzy_full.json")
    feats = {k: v for k, v in _fake_result()["fuzzy"].items()}
    with open(fuzzy, "w") as f:
        json.dump({k: v for k, v in feats.items() if k != "ddec6_r4"}, f)
    assert not validate_ddec6(charge, timings, n_atoms=4, fuzzy_json_loc=fuzzy)
    with open(fuzzy, "w") as f:
        json.dump(feats, f)
    assert validate_ddec6(charge, timings, n_atoms=4, bond_json_loc=bond, fuzzy_json_loc=fuzzy)
    assert not validate_ddec6(charge, timings, n_atoms=4, current=False)  # e.g. an older engine version
    with open(timings, "w") as f:
        json.dump({}, f)
    assert not validate_ddec6(charge, timings, n_atoms=4)
    assert validate_ddec6(charge, timings, n_atoms=4, skipped={"ddec6"})


def test_validation_checks_requires_current_ddec6(folder, monkeypatch):
    from qtaim_gen.source.utils import validation

    seen = {}
    os.makedirs(os.path.join(folder, "generator"))
    for name in ("timings", "fuzzy_full", "other", "charge", "qtaim", "bond"):
        with open(os.path.join(folder, "generator", f"{name}.json"), "w") as f:
            f.write("{}")
    for check in ("validate_timing_dict", "validate_fuzzy_dict", "validate_other_dict", "validate_charge_dict",
                  "validate_qtaim_dict", "validate_bond_dict"):
        monkeypatch.setattr(validation, check, lambda *a, **k: True)
    monkeypatch.setattr(validation, "get_charge_spin_n_atoms_from_folder",
                        lambda *a, **k: {"mol": LABELS, "charge": 0, "spin": 1})
    monkeypatch.setattr(validation, "validate_ddec6", lambda *a, **k: seen.update(k) or True)
    validation.validation_checks(folder, ddec6=True, logger=LOG)
    assert seen["current"] is False  # no provenance entry
    assert seen["bond_json_loc"] == os.path.join(folder, "generator", "bond.json")
    assert seen["fuzzy_json_loc"] == os.path.join(folder, "generator", "fuzzy_full.json")
    provenance.record_routines(folder, ["ddec6"], "ddec6", ENGINE_VERSION)
    validation.validation_checks(folder, ddec6=True, logger=LOG)
    assert seen["current"] is True


def _gbw_stubbed(monkeypatch, folder, **kwargs):
    seen = {"merge": 0}
    monkeypatch.setattr(omol, "create_jobs", lambda *a, **k: None)
    monkeypatch.setattr(omol, "run_jobs", lambda *a, **k: None)
    monkeypatch.setattr(omol, "parse_multiwfn", lambda *a, **k: None)
    monkeypatch.setattr(omol, "_run_orca_parse", lambda *a, **k: None)
    monkeypatch.setattr(omol, "_run_engines", lambda *a, **k: seen.update(engines=k))
    monkeypatch.setattr(omol, "validation_checks", lambda *a, **k: seen.update(validate=k) or False)
    monkeypatch.setattr(omol, "_merge_ddec6", lambda *a, **k: seen.update(merge=seen["merge"] + 1))
    omol.gbw_analysis(folder, multiwfn_cmd="x", orca_2mkl_cmd="y", restart=False, overwrite=False, logger=LOG,
                      move_results=True, **kwargs)
    return seen


def test_parse_merges_ddec6_without_the_flag(folder, monkeypatch):
    # parse_multiwfn rewrites charge.json; a current ddec6.json must be merged back either way
    seen = _gbw_stubbed(monkeypatch, folder)
    assert seen["merge"] >= 1 and seen["engines"]["ddec6"] is False


def test_parse_only_does_not_require_ddec6(folder, monkeypatch, caplog):
    with caplog.at_level(logging.WARNING, logger=LOG.name):
        seen = _gbw_stubbed(monkeypatch, folder, parse_only=True, ddec6=True)
    assert seen["validate"]["ddec6"] is False and seen["merge"] >= 1
    assert "computes nothing with parse_only" in caplog.text


def test_ddec6_reads_generator_wfx_behind_a_root_wfn(folder, monkeypatch):
    gen = os.path.join(folder, "generator")
    os.makedirs(gen)
    shutil.move(os.path.join(folder, "orca.wfx"), os.path.join(gen, "orca.wfx"))
    with open(os.path.join(folder, "orca.wfn"), "w") as f:
        f.write("wfn")
    seen = []
    monkeypatch.setattr(ddec6, "analyze", lambda wfx, **kw: seen.append(1) or _fake_result())
    assert omol._run_ddec6(folder, n_threads=1, logger=LOG)
    assert seen == [1]


def test_runner_refuses_ddec6_with_wfn(tmp_path):
    from qtaim_gen.source.scripts.full_runner_parsl_alcf import main

    with pytest.raises(SystemExit):
        main(["--ddec6", "--wfn", "--job_file", str(tmp_path / "jobs.txt")])


@pytest.mark.skipif(not os.environ.get("DDEC6_SLOW"), reason="about a minute; set DDEC6_SLOW=1")
def test_analyze_layout():
    out = ddec6.analyze(os.path.join(FIXTURE, "orca.wfx"))
    d = out["ddec6"]
    assert list(d["charge"]) == LABELS
    assert set(d) == {"charge", "spin", "dipole", "atomic_dipole", "atomic_quadrupole",
                      "atomic_quadrupole_eigenvalues"}
    assert all(len(v) == 3 for v in d["atomic_dipole"].values())
    assert all(len(v) == 5 for v in d["atomic_quadrupole"].values())
    assert abs(sum(d["charge"].values())) < 1e-12
    assert set(out["fuzzy"]) == set(DDEC6_FUZZY)
    for feats in out["fuzzy"].values():
        assert set(feats) == set(LABELS) | {"sum", "abs_sum"}
        assert feats["sum"] == pytest.approx(sum(feats[k] for k in LABELS))
    bond = out["bond"]
    assert set(bond) == {"ddec6_bond", "ddec6_overlap_population"}
    assert set(bond["ddec6_bond"]) == set(bond["ddec6_overlap_population"])
    assert len(bond["ddec6_bond"]) == 5 and min(bond["ddec6_bond"].values()) >= ddec6.BO_PRINT_CUTOFF
    meta = out["_meta"]
    assert meta["n_iterations"] == 7 and meta["natoms"] == 4
    assert meta["converged"] is True and meta["spin_converged"] is True and meta["est_peak_gb"] > 0
    assert meta["n_bond_pairs"] == 6 and meta["bo_prep_passes"] >= 1
    assert meta["summed_contact_exchange_error"] < 1e-10


def test_killed_child_drops_the_ddec6_timing(folder, monkeypatch):
    import stat
    import sys

    os.makedirs(os.path.join(folder, "generator"))
    old = {"other_geometry": 0.2, "ddec6": 9.0}
    for path in (os.path.join(folder, "timings.json"), os.path.join(folder, "generator", "timings.json")):
        with open(path, "w") as f:
            json.dump(old, f)
    fake = os.path.join(folder, "python_killed")
    with open(fake, "w") as f:
        f.write("#!/bin/sh\nkill -9 $$\n")
    os.chmod(fake, os.stat(fake).st_mode | stat.S_IEXEC)
    monkeypatch.setattr(sys, "executable", fake)
    provenance.record_routines(folder, ["ddec6"], "ddec6", ENGINE_VERSION)
    omol._run_engines(folder, n_threads=1, logger=LOG, ddec6=True)
    for path in (os.path.join(folder, "timings.json"), os.path.join(folder, "generator", "timings.json")):
        with open(path) as f:
            assert json.load(f) == {"other_geometry": 0.2}
    assert "ddec6" not in provenance.load(folder)["routines"]


def test_killed_ddec6_child_keeps_the_other_engines(folder, monkeypatch):
    import stat
    import sys

    from qtaim_gen.source.data.engine_versions import ROUTINE_ENGINE

    old = {"charge_engine": 5.0, "surface_engine": 2.0, "ddec6": 9.0}
    with open(os.path.join(folder, "timings.json"), "w") as f:
        json.dump(old, f)
    fake = os.path.join(folder, "python_killed_in_ddec6")
    with open(fake, "w") as f:
        f.write('#!/bin/sh\ncase "$*" in *--ddec6*) kill -9 $$ ;; esac\nexit 0\n')
    os.chmod(fake, os.stat(fake).st_mode | stat.S_IEXEC)
    monkeypatch.setattr(sys, "executable", fake)
    charge_routine = next(r for r, e in ROUTINE_ENGINE.items() if e == "charge_engine")
    provenance.record_routines(folder, [charge_routine], "charge_engine", ENGINE_VERSION)
    provenance.record_routines(folder, ["ddec6"], "ddec6", ENGINE_VERSION)
    omol._run_engines(folder, n_threads=1, logger=LOG, charge_engine=True, ddec6=True)
    assert _timings(folder) == {"charge_engine": 5.0, "surface_engine": 2.0}
    routines = provenance.load(folder)["routines"]
    assert charge_routine in routines and "ddec6" not in routines


def test_peak_memory_above_limit_is_a_permanent_skip(folder, monkeypatch):
    def too_big(wfx, **kw):
        raise ddec6.PeakMemoryAboveLimit("estimated peak 30.0 GB > limit 16.0 GB")
    monkeypatch.setattr(ddec6, "analyze", too_big)
    assert omol._run_ddec6(folder, n_threads=1, logger=LOG)
    rec = provenance.load(folder)["skipped"]["ddec6"]
    assert rec == {"reason": "est_peak_above_limit: estimated peak 30.0 GB > limit 16.0 GB", "by": "ddec6"}
    assert validate_ddec6(os.path.join(folder, "charge.json"), os.path.join(folder, "timings.json"), 4,
                          skipped=provenance.skipped(provenance.load(folder)))


def test_skip_on_recompute_removes_merged_ddec6_data(folder, monkeypatch):
    monkeypatch.setattr(ddec6, "analyze", lambda wfx, **kw: _fake_result())
    assert omol._run_ddec6(folder, n_threads=1, logger=LOG)
    gen = os.path.join(folder, "generator")
    os.makedirs(gen)
    for base in (folder, gen):
        for name, data in (("charge.json", {"hirshfeld": {"charge": {k: 0.0 for k in LABELS}}}),
                           ("fuzzy_full.json", {"becke_fuzzy_density": {"1_Br": 35.0}}),
                           ("bond.json", {"fuzzy_bond": {"1_Br_to_2_O": 1.1}})):
            with open(os.path.join(base, name), "w") as f:
                json.dump(data, f)
    shutil.move(os.path.join(folder, omol.DDEC6_JSON), os.path.join(gen, omol.DDEC6_JSON))
    omol._merge_ddec6(folder, move_results=True, logger=LOG)
    omol._merge_ddec6(folder, move_results=False, logger=LOG)
    monkeypatch.setattr(ddec6, "MAX_ATOMS", 3)
    assert omol._run_ddec6(folder, n_threads=1, logger=LOG)
    for base in (folder, gen):
        with open(os.path.join(base, "charge.json")) as f:
            assert set(json.load(f)) == {"hirshfeld"}
        with open(os.path.join(base, "fuzzy_full.json")) as f:
            assert set(json.load(f)) == {"becke_fuzzy_density"}
        with open(os.path.join(base, "bond.json")) as f:
            assert set(json.load(f)) == {"fuzzy_bond"}
    assert provenance.load(folder)["skipped"]["ddec6"]["reason"] == "natoms_above_3"
    assert validate_ddec6(os.path.join(folder, "charge.json"), os.path.join(folder, "timings.json"), 4,
                          skipped=provenance.skipped(provenance.load(folder)))


@pytest.mark.skipif(not os.environ.get("DDEC6_SLOW"), reason="a few minutes; set DDEC6_SLOW=1")
def test_child_process_matches_in_process(tmp_path):
    # _run_engines always runs DDEC6 in a child; the reference calls _run_ddec6 in this process
    xecl4 = os.path.join(os.path.dirname(__file__), "test_files", "charge_engine", "XeCl4_step2_0_1")
    runs = {}
    for sub in (True, False):
        d = tmp_path / ("child" if sub else "inproc")
        d.mkdir()
        shutil.copy(os.path.join(xecl4, "orca.wfx"), d / "orca.wfx")
        if sub:
            omol._run_engines(str(d), n_threads=4, logger=LOG, ddec6=True)
        else:
            omol._run_ddec6(str(d), n_threads=4, logger=LOG)
        with open(d / omol.DDEC6_JSON) as f:
            runs[sub] = json.load(f)
        assert _timings(str(d))["ddec6"] > 0
    assert runs[True]["ddec6"] == runs[False]["ddec6"] and runs[True]["fuzzy"] == runs[False]["fuzzy"]
