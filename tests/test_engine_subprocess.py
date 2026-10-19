"""omol._run_engines: the engine steps of gbw_analysis in a child process (core/engine_worker.py)."""

import importlib.util
import json
import logging
import os
import shutil
import stat
import sys

import pytest

pytestmark = pytest.mark.skipif(importlib.util.find_spec("numba") is None, reason="needs numba")

FIXTURE = os.path.join(os.path.dirname(__file__), "test_files", "charge_engine", "XeCl4_step2_0_1")


def _folder(path):
    for f in ("orca.inp", "orca.wfx"):
        shutil.copy(os.path.join(FIXTURE, f), path / f)
    return str(path)


def _logger(folder):
    from qtaim_gen.source.core.omol import setup_logger

    return setup_logger(folder)


def _timings(folder):
    with open(os.path.join(folder, "timings.json")) as f:
        return json.load(f)


def _log(folder):
    for h in logging.getLogger(f"gbw_analysis-{folder}").handlers:
        h.flush()
    with open(os.path.join(folder, "gbw_analysis.log")) as f:
        return f.read()


def test_child_process_writes_what_the_in_process_run_writes(tmp_path):
    from qtaim_gen.source.core.omol import _run_engines

    (tmp_path / "child").mkdir()
    (tmp_path / "inproc").mkdir()
    child, inproc = _folder(tmp_path / "child"), _folder(tmp_path / "inproc")
    for folder, sub in ((child, True), (inproc, False)):
        _run_engines(folder, n_threads=2, logger=_logger(folder), qtaim_engine=True, charge_engine=True,
                     engine_subprocess=sub)
    for folder in (child, inproc):
        t = _timings(folder)
        assert all(t.get(k, 0) > 0 for k in ("qtaim_engine", "charge_engine", "surface_engine")), t
    assert "engine child finished" in _log(child)
    assert "engine child" not in _log(inproc)
    with open(os.path.join(child, "CPprop.txt")) as a, open(os.path.join(inproc, "CPprop.txt")) as b:
        assert a.read() == b.read()
    names = sorted(n for n in os.listdir(inproc) if n.endswith(".json") and n != "timings.json")
    assert "hirshfeld.json" in names and "other_alie.json" in names, names
    for name in names:
        with open(os.path.join(child, name)) as a, open(os.path.join(inproc, name)) as b:
            assert json.load(a) == json.load(b), name


def test_killed_child_is_logged_and_leaves_no_engine_timing(tmp_path, monkeypatch):
    from qtaim_gen.source.core import omol

    folder = _folder(tmp_path)
    fake = tmp_path / "python_killed"
    fake.write_text("#!/bin/sh\necho dying >&2\nkill -9 $$\n")
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setattr(sys, "executable", str(fake))
    omol._run_engines(folder, n_threads=1, logger=_logger(folder), qtaim_engine=True, charge_engine=True)
    log = _log(folder)
    assert "engine child exited with status -9" in log and "dying" in log
    assert not os.path.exists(os.path.join(folder, "timings.json"))


def test_step_failure_is_reported(tmp_path):
    from qtaim_gen.source.core.omol import _run_engines

    folder = str(tmp_path)
    shutil.copy(os.path.join(FIXTURE, "orca.inp"), tmp_path / "orca.inp")
    with open(tmp_path / "orca.wfx", "w") as f:
        f.write("not a wavefunction\n")
    _run_engines(folder, n_threads=1, logger=_logger(folder), qtaim_engine=True)
    log = _log(folder)
    assert "qtaim_engine failed" in log and "engine child: a step failed" in log
    assert not os.path.exists(os.path.join(folder, "timings.json"))


def test_no_engine_flag_launches_nothing(tmp_path, monkeypatch):
    # Multiwfn mode: gbw_analysis calls _run_engines with both flags off
    from qtaim_gen.source.core import omol

    def boom(*a, **k):
        raise AssertionError("no child process in Multiwfn mode")

    monkeypatch.setattr(omol.subprocess, "run", boom)
    omol._run_engines(str(tmp_path), n_threads=1, logger=_logger(str(tmp_path)))
