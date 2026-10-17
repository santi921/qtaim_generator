"""The QTAIM engine as the production QTAIM step (gbw_analysis qtaim_engine=True):
CPprop.txt and qtaim.out the gates and parser read unchanged, _meta.qtaim_engine,
the qtaim_engine timing, and no Poincare-Hopf escalation for engine records."""

import importlib.util
import json
import logging
import os
import shutil

import pytest

from qtaim_gen.source.utils.validation import (
    QTAIM_META_KEY,
    QTAIM_SEARCH_ENGINE,
    poincare_hopf_needs_exhaustive,
    qtaim_cpprop_problems,
    qtaim_run_status,
    validate_qtaim_dict,
    validate_timing_dict,
)

LOG = logging.getLogger("test_qtaim_engine_wiring")
FIXTURE = os.path.join(os.path.dirname(__file__), "test_files", "charge_engine", "XeCl4_step2_0_1")
HAS_NUMBA = importlib.util.find_spec("numba") is not None


@pytest.mark.skipif(not HAS_NUMBA, reason="numba not installed")
def test_engine_qtaim_step_end_to_end(tmp_path):
    from qtaim_gen.source.core.cp_search import QTAIM_ENGINE_VERSION
    from qtaim_gen.source.core.omol import _run_qtaim_engine, parse_multiwfn
    from qtaim_gen.source.core.parse_multiwfn import parse_qtaim

    for f in ("orca.wfx", "orca.inp"):
        shutil.copy(os.path.join(FIXTURE, f), tmp_path / f)
    assert _run_qtaim_engine(str(tmp_path), n_threads=2, logger=LOG)
    assert qtaim_cpprop_problems(str(tmp_path)) == []
    status = qtaim_run_status(str(tmp_path))
    assert status["search_done"] and status["export_done"]
    assert json.loads((tmp_path / "timings.json").read_text())["qtaim_engine"] > 0

    parse_multiwfn(str(tmp_path), separate=True, logger=LOG, full_set=0)
    rec = json.loads((tmp_path / "qtaim.json").read_text())
    meta = rec[QTAIM_META_KEY]
    assert meta["qtaim_search"] == QTAIM_SEARCH_ENGINE and meta["poincare_hopf"] == 1
    assert meta["qtaim_engine"]["version"] == QTAIM_ENGINE_VERSION
    assert meta["qtaim_engine"]["seeds"] == "superset" and meta["qtaim_engine"]["poincare_hopf_full_set"] == 1
    assert status["reported_bcp"] == meta["cp_counts"]["BCP"]

    # same CPs as Multiwfn's CPprop.txt for this molecule, parsed by the same parser
    ref = parse_qtaim(cprop_file=os.path.join(FIXTURE, "multiwfn_cpprop.txt"),
                      inp_loc=os.path.join(FIXTURE, "orca.inp"), orca_tf=True)
    assert {k for k in rec if k != QTAIM_META_KEY} == {str(k) for k in ref}
    assert validate_qtaim_dict(str(tmp_path / "qtaim.json"), n_atoms=5, folder=str(tmp_path),
                               check_bcp_count=True, require_provenance=True, enforce_poincare_hopf=True)


def _record(tmp_path, search, ph):
    rec = {"0": {"element": "O"}, "1": {"element": "H"}, "2": {"element": "H"},
           "0_1": {"cp_num": 4}, "0_2": {"cp_num": 5},
           QTAIM_META_KEY: {"poincare_hopf": ph, "cp_counts": {"NCP": 3, "BCP": 2, "RCP": 0, "CCP": 0},
                            "qtaim_search": search}}
    (tmp_path / "qtaim.json").write_text(json.dumps(rec))
    return str(tmp_path / "qtaim.json")


def test_engine_record_is_never_escalated(tmp_path):
    # an engine record whose kept CPs do not sum to 1 (a closing CP below the weak cut, by
    # design; both sums are in _meta.qtaim_engine) is accepted, and gbw_analysis does not
    # rerun it with Multiwfn's exhaustive search
    from qtaim_gen.source.core.omol import _poincare_hopf_needs_escalation

    p = _record(tmp_path, QTAIM_SEARCH_ENGINE, 0)
    assert not poincare_hopf_needs_exhaustive(json.loads(open(p).read())[QTAIM_META_KEY])
    assert validate_qtaim_dict(p, n_atoms=3, folder=str(tmp_path), enforce_poincare_hopf=True)
    assert not _poincare_hopf_needs_escalation(str(tmp_path), LOG)

    p = _record(tmp_path, "standard", 0)
    assert not validate_qtaim_dict(p, n_atoms=3, folder=str(tmp_path), enforce_poincare_hopf=True)
    assert _poincare_hopf_needs_escalation(str(tmp_path), LOG)


def test_qtaim_engine_timing_stands_in_for_qtaim(tmp_path):
    from qtaim_gen.source.core.omol import patch_timings_from_log
    from qtaim_gen.source.utils.validation import get_expected_timing_keys

    expected, _ = get_expected_timing_keys(full_set=0, spin_tf=False)
    timings = {k: 1.0 for k in expected if k != "qtaim"}
    path = tmp_path / "timings.json"
    path.write_text(json.dumps(timings))
    assert not validate_timing_dict(str(path))
    timings["qtaim_engine"] = 2.0
    path.write_text(json.dumps(timings))
    assert validate_timing_dict(str(path)) and validate_timing_dict(str(path), qtaim_engine=True)
    del timings["qtaim_engine"]
    timings["qtaim"] = 1.0
    path.write_text(json.dumps(timings))
    assert not validate_timing_dict(str(path), qtaim_engine=True)  # a Multiwfn QTAIM record is not done

    timings = {k: 1.0 for k in expected if k != "qtaim"}
    timings["qtaim_engine"] = 2.0
    path.write_text(json.dumps(timings))
    (tmp_path / "gbw_analysis.log").write_text("")
    patch_timings_from_log(str(tmp_path), full_set=0, spin_tf=False, move_results=False, logger=LOG)
    assert "qtaim" not in json.loads(path.read_text())
