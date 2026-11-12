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
from qtaim_gen.source.data.engine_versions import ENGINE_VERSION
from qtaim_gen.source.utils import provenance
from qtaim_gen.source.utils.validation import validate_ddec6

FIXTURE = os.path.join(os.path.dirname(__file__), "test_files", "charge_engine", "rmechdb_652_step10_0_2")
LABELS = ["1_Br", "2_O", "3_O", "4_O"]
LOG = logging.getLogger("test_ddec6_runner")


def _fake_result():
    q = {k: v for k, v in zip(LABELS, (0.15, -0.14, 0.13, -0.14))}
    feats = {name: {**{k: 0.1 for k in LABELS}, "sum": 0.4, "abs_sum": 0.4}
             for name in ("ddec6_atomic_dipole_mag", "ddec6_quadrupole_mag", "ddec6_tail_decay")}
    return {"ddec6": {"charge": q, "dipole": {"mag": 0.0, "xyz": [0.0, 0.0, 0.0]}}, "fuzzy": feats,
            "_meta": {"n_iterations": 7, "natoms": 4}}


@pytest.fixture
def folder(tmp_path):
    shutil.copy(os.path.join(FIXTURE, "orca.wfx"), tmp_path / "orca.wfx")
    return str(tmp_path)


def _timings(folder):
    with open(os.path.join(folder, "timings.json")) as f:
        return json.load(f)


def test_run_writes_result_timing_and_version(folder, monkeypatch):
    calls = []
    monkeypatch.setattr(ddec6, "analyze", lambda path: calls.append(path) or _fake_result())
    assert omol._run_ddec6(folder, n_threads=1, logger=LOG)
    with open(os.path.join(folder, omol.DDEC6_JSON)) as f:
        assert json.load(f)["ddec6"]["charge"]["1_Br"] == 0.15
    assert _timings(folder)["ddec6"] > 0
    prov = provenance.load(folder)
    entry = prov["routines"]["ddec6"]
    assert entry["by"] == "ddec6" and entry["engine_version"] == ENGINE_VERSION
    assert entry["spec"] == {"chargemol": "09_26_2017", "grid_spacing_bohr": 0.14, "max_atoms": ddec6.MAX_ATOMS}
    assert prov["engines"]["ddec6"] == {"n_iterations": 7, "natoms": 4}
    # restart finds the completed run and does not recompute
    assert omol._run_ddec6(folder, n_threads=1, logger=LOG, restart=True)
    assert len(calls) == 1


def test_restart_recomputes_a_result_from_an_older_version(folder, monkeypatch):
    calls = []
    monkeypatch.setattr(ddec6, "analyze", lambda path: calls.append(path) or _fake_result())
    assert omol._run_ddec6(folder, n_threads=1, logger=LOG)
    provenance.record_routines(folder, ["ddec6"], "ddec6", ENGINE_VERSION - 1)
    assert omol._run_ddec6(folder, n_threads=1, logger=LOG, restart=True)
    assert len(calls) == 2
    assert provenance.load(folder)["routines"]["ddec6"]["engine_version"] == ENGINE_VERSION


def test_size_cap_is_a_permanent_skip(folder, monkeypatch):
    monkeypatch.setattr(ddec6, "MAX_ATOMS", 3)
    monkeypatch.setattr(ddec6, "analyze", lambda path: pytest.fail("analyze must not run above the cap"))
    assert omol._run_ddec6(folder, n_threads=1, logger=LOG)
    assert provenance.load(folder)["skipped"]["ddec6"]["reason"] == "natoms_above_3"
    # a recorded skip is final: later passes do nothing
    assert omol._run_ddec6(folder, n_threads=1, logger=LOG)


def test_chargemol_stop_is_a_permanent_skip(folder, monkeypatch):
    def stop(path):
        raise ddec6.ChargemolStop("grid spacing check failed")
    monkeypatch.setattr(ddec6, "analyze", stop)
    assert omol._run_ddec6(folder, n_threads=1, logger=LOG)
    rec = provenance.load(folder)["skipped"]["ddec6"]
    assert rec["reason"] == "chargemol_stop: grid spacing check failed" and rec["by"] == "ddec6"


def test_other_errors_fail_without_a_skip(folder, monkeypatch):
    def boom(path):
        raise RuntimeError("numba exploded")
    monkeypatch.setattr(ddec6, "analyze", boom)
    assert not omol._run_ddec6(folder, n_threads=1, logger=LOG)
    assert not os.path.exists(os.path.join(folder, "timings.json"))
    assert "ddec6" not in provenance.load(folder)["routines"]


def test_merge_into_generator_copy_and_idempotent(folder):
    gen = os.path.join(folder, "generator")
    os.makedirs(gen)
    for name, data in (("charge.json", {"hirshfeld": {"charge": {k: 0.0 for k in LABELS}}}),
                       ("fuzzy_full.json", {"becke_fuzzy_density": {"1_Br": 35.0}})):
        with open(os.path.join(gen, name), "w") as f:
            json.dump(data, f)
    with open(os.path.join(folder, omol.DDEC6_JSON), "w") as f:
        json.dump(_fake_result(), f)
    for _ in range(2):
        omol._merge_ddec6(folder, move_results=True, logger=LOG)
    with open(os.path.join(gen, "charge.json")) as f:
        charge = json.load(f)
    with open(os.path.join(gen, "fuzzy_full.json")) as f:
        fuzzy = json.load(f)
    assert set(charge) == {"hirshfeld", "ddec6"} and charge["ddec6"]["charge"]["2_O"] == -0.14
    assert set(fuzzy) == {"becke_fuzzy_density", "ddec6_atomic_dipole_mag", "ddec6_quadrupole_mag", "ddec6_tail_decay"}


def test_merge_into_root_without_generator(folder):
    with open(os.path.join(folder, "charge.json"), "w") as f:
        json.dump({"cm5": {"charge": {}}}, f)
    with open(os.path.join(folder, "fuzzy_full.json"), "w") as f:
        json.dump({}, f)
    with open(os.path.join(folder, omol.DDEC6_JSON), "w") as f:
        json.dump(_fake_result(), f)
    omol._merge_ddec6(folder, move_results=True, logger=LOG)
    with open(os.path.join(folder, "charge.json")) as f:
        assert "ddec6" in json.load(f)


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
    with open(timings, "w") as f:
        json.dump({}, f)
    assert not validate_ddec6(charge, timings, n_atoms=4)
    assert validate_ddec6(charge, timings, n_atoms=4, skipped={"ddec6"})


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
    for name in ("ddec6_atomic_dipole_mag", "ddec6_quadrupole_mag", "ddec6_tail_decay"):
        feats = out["fuzzy"][name]
        assert set(feats) == set(LABELS) | {"sum", "abs_sum"}
        assert feats["sum"] == pytest.approx(sum(feats[k] for k in LABELS))
    assert out["_meta"]["n_iterations"] == 7 and out["_meta"]["natoms"] == 4


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


@pytest.mark.skipif(not os.environ.get("DDEC6_SLOW"), reason="a few minutes; set DDEC6_SLOW=1")
def test_child_process_matches_in_process(tmp_path):
    xecl4 = os.path.join(os.path.dirname(__file__), "test_files", "charge_engine", "XeCl4_step2_0_1")
    runs = {}
    for sub in (True, False):
        d = tmp_path / ("child" if sub else "inproc")
        d.mkdir()
        shutil.copy(os.path.join(xecl4, "orca.wfx"), d / "orca.wfx")
        omol._run_engines(str(d), n_threads=4, logger=LOG, ddec6=True, engine_subprocess=sub)
        with open(d / omol.DDEC6_JSON) as f:
            runs[sub] = json.load(f)
        assert _timings(str(d))["ddec6"] > 0
    assert runs[True]["ddec6"] == runs[False]["ddec6"] and runs[True]["fuzzy"] == runs[False]["fuzzy"]
