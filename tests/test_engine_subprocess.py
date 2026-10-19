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


def test_killed_child_is_logged_and_drops_stale_engine_timings(tmp_path, monkeypatch):
    # an earlier run's engine timings, here and in generator/, must not vouch for what a
    # killed child half-wrote; timings of other steps stay
    from qtaim_gen.source.core import omol

    folder = _folder(tmp_path)
    os.makedirs(tmp_path / "generator")
    old = {"other_geometry": 0.2, "qtaim_engine": 5.0, "charge_engine": 3.0, "surface_engine": 4.0}
    for path in (tmp_path / "timings.json", tmp_path / "generator" / "timings.json"):
        path.write_text(json.dumps(old))
    fake = tmp_path / "python_killed"
    fake.write_text("#!/bin/sh\necho dying >&2\nkill -9 $$\n")
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setattr(sys, "executable", str(fake))
    omol._run_engines(folder, n_threads=1, logger=_logger(folder), qtaim_engine=True, charge_engine=True)
    log = _log(folder)
    assert "engine child exited with status -9" in log and "dying" in log
    for path in (tmp_path / "timings.json", tmp_path / "generator" / "timings.json"):
        assert json.loads(path.read_text()) == {"other_geometry": 0.2}, path


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


def test_child_command_carries_flags_and_environment(tmp_path, monkeypatch):
    from qtaim_gen.source.core import omol

    seen = {}

    class Done:
        returncode, stderr = 0, ""

    def fake_run(cmd, env, cwd, **kw):
        seen.update(cmd=cmd, env=env, cwd=cwd)
        return Done()

    monkeypatch.setattr(omol.subprocess, "run", fake_run)
    monkeypatch.setenv("NUMBA_NUM_THREADS", "3")
    monkeypatch.chdir(tmp_path)
    os.makedirs("job")
    omol._run_engines("job", n_threads=8, logger=_logger(str(tmp_path / "job")), restart=True, full_set=1,
                      qtaim_engine=True, charge_engine=True)
    job = str(tmp_path / "job")
    assert seen["cmd"][1:4] == ["-m", "qtaim_gen.source.core.engine_worker", job]
    assert seen["cmd"][4:] == ["--n_threads", "8", "--full_set", "1", "--qtaim", "--charge", "--restart"]
    assert seen["cwd"] == job
    env = seen["env"]
    assert env["NUMBA_NUM_THREADS"] == "3" and env["PYTHONSAFEPATH"] == "1"
    assert env["OPENBLAS_NUM_THREADS"] == "1"


def test_worker_passes_restart_and_full_set_to_the_engines(monkeypatch):
    from qtaim_gen.source.core import engine_worker, omol

    calls = []

    def record(name, ok=True):
        def run(folder, **kw):
            calls.append((name, folder, kw.get("restart"), kw.get("full_set"), kw.get("n_threads")))
            return ok
        return run

    monkeypatch.setattr(omol, "_run_qtaim_engine", record("qtaim"))
    monkeypatch.setattr(omol, "_run_charge_engine", record("charge", ok=False))
    monkeypatch.setattr(omol, "_run_surface_engine", record("surface"))
    monkeypatch.setattr(omol, "setup_logger", lambda folder: logging.getLogger("engine_worker_test"))
    rc = engine_worker.main(["/x", "--qtaim", "--charge", "--n_threads", "6", "--full_set", "1", "--restart"])
    assert rc == engine_worker.STEP_FAILED
    assert calls == [("qtaim", "/x", True, None, 6), ("charge", "/x", True, 1, 6), ("surface", "/x", True, None, 6)]


def test_restart_skips_a_completed_qtaim_run(tmp_path):
    from qtaim_gen.source.core.omol import _run_engines

    folder = _folder(tmp_path)
    _run_engines(folder, n_threads=2, logger=_logger(folder), qtaim_engine=True)
    _run_engines(folder, n_threads=2, logger=_logger(folder), qtaim_engine=True, restart=True)
    log = _log(folder)
    assert "qtaim_engine: completed run found - skipping" in log
    assert log.count("engine child finished") == 2


def test_child_ignores_a_qtaim_gen_package_in_the_job_folder(tmp_path):
    # python -m would otherwise import it ahead of PYTHONPATH (needs Python 3.11 for
    # PYTHONSAFEPATH; cwd=folder alone is exactly the case it guards)
    from qtaim_gen.source.core.omol import _run_engines

    if sys.version_info < (3, 11):
        pytest.skip("PYTHONSAFEPATH needs Python 3.11")
    folder = _folder(tmp_path)
    os.makedirs(tmp_path / "qtaim_gen")
    (tmp_path / "qtaim_gen" / "__init__.py").write_text("raise ImportError('wrong checkout')\n")
    _run_engines(folder, n_threads=2, logger=_logger(folder), qtaim_engine=True)
    log = _log(folder)
    assert "engine child finished" in log and "wrong checkout" not in log
    assert _timings(folder)["qtaim_engine"] > 0
