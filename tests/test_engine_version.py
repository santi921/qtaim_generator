"""Engine version in provenance.json: which routines a folder's engine records vouch for
depends on the ENGINE_VERSION they were computed at (data/engine_versions.py)."""

import json

import pytest

from qtaim_gen.source.data.engine_versions import ENGINE_VERSION, ROUTINE_ENGINE, is_current, last_change, routines_at
from qtaim_gen.source.utils import provenance
from qtaim_gen.source.utils.validation import (TIMINGS_PATCHED_KEY, get_expected_timing_keys, timing_total,
                                                 validate_timing_dict)


def _write_provenance(folder, version, routines=None, generator=True):
    """Engine entries at this version for every engine routine it computed; a routine a
    later version changed (MBIS at 5) carries that version, as after the restart that
    recomputed it, so only the routine under test is stale."""
    prov = provenance.empty()
    for r in sorted(routines if routines is not None else routines_at(version)):
        prov["routines"][r] = {"by": ROUTINE_ENGINE[r], "engine_version": max(version, last_change(r))}
    path = (folder / "generator") if generator else folder
    path.mkdir(parents=True, exist_ok=True)
    (path / provenance.PROVENANCE_FILE).write_text(json.dumps(prov))


def test_mbis_recorded_before_version_5_is_stale():
    assert not is_current("mbis", 4) and not is_current("mbis_fuzzy_spin", 4)
    assert is_current("mbis", 5) and is_current("hirshfeld", 4)


def test_timing_total_skips_metadata():
    timings = {"qtaim": 2.0, "charge_engine": 3.0, TIMINGS_PATCHED_KEY: {"qtaim": "log"}}
    assert timing_total(timings) == 5.0


def _timings_folder(tmp_path, version, full_set):
    """generator/ with every step timing of the level (as Multiwfn would leave them),
    both engine timings and provenance from this engine version."""
    keys, spin_keys = get_expected_timing_keys(full_set=full_set, spin_tf=False)
    timings = {k: 1.0 for k in list(keys) + list(spin_keys)}
    timings.update({"charge_engine": 2.0, "surface_engine": 1.0})
    _write_provenance(tmp_path, version, routines_at(version) - {"qtaim"})
    path = tmp_path / "generator" / "timings.json"
    path.write_text(json.dumps(timings))
    return path


@pytest.mark.parametrize("version, full_set, done", [
    (1, 0, True), (1, 1, False),                    # version 1 left elf_fuzzy to Multiwfn
    (2, 0, True), (2, 1, False), (5, 1, False),     # versions 2-5 left chelpg to Multiwfn
    (2, 2, False),                                  # version 2 left other_esp to Multiwfn
    (ENGINE_VERSION, 1, True), (ENGINE_VERSION, 2, True),
])
def test_engine_validation_requires_current_engine_output(tmp_path, version, full_set, done):
    """With charge_engine=True a folder is done only when every charge- and surface-engine
    routine of its level is current engine output: Multiwfn's timing for a routine an
    older engine version left out does not count, so the restart recomputes it."""
    path = _timings_folder(tmp_path, version, full_set)
    assert validate_timing_dict(str(path), full_set=full_set, charge_engine=True) is done
    # the stock runner (charge_engine=False) still accepts Multiwfn's timings
    assert validate_timing_dict(str(path), full_set=full_set)


def test_engine_validation_accepts_permanently_skipped_routines(tmp_path):
    path = _timings_folder(tmp_path, 5, full_set=1)
    assert not validate_timing_dict(str(path), full_set=1, charge_engine=True)
    provenance.record_skips(str(tmp_path), {"chelpg": {"reason": "test", "by": "charge_engine"}})
    assert validate_timing_dict(str(path), full_set=1, charge_engine=True)


def test_validation_without_provenance_expects_every_step_timing(tmp_path):
    """No provenance: the folder is Multiwfn's, so the routines need their timings even
    with a charge_engine timing present."""
    gen = tmp_path / "generator"
    gen.mkdir()
    (gen / "timings.json").write_text(json.dumps({"qtaim": 5.0, "other": 3.0, "charge_engine": 2.0,
                                                  "surface_engine": 1.0}))
    assert not validate_timing_dict(str(gen / "timings.json"), full_set=0, charge_engine=True)


def _engine_folder(tmp_path, version):
    from qtaim_gen.source.core.omol import _engine_routines

    gen = tmp_path / "generator"
    gen.mkdir(parents=True)
    (gen / "timings.json").write_text(json.dumps({"charge_engine": 2.0}))
    routines = _engine_routines(2, spin_tf=True)
    charge = {r: {"charge": {"1_H": 0.1}} for r in ("hirshfeld", "adch", "cm5", "becke", "vdd", "mbis", "chelpg")}
    (gen / "charge.json").write_text(json.dumps(charge))
    (gen / "bond.json").write_text(json.dumps({"fuzzy_bond": {"1_H_to_2_H": 1.0}}))
    fuzzy = {r: {"1_H": 1.0} for r in routines if r not in charge and r != "fuzzy_bond"}
    (gen / "fuzzy_full.json").write_text(json.dumps(fuzzy))
    _write_provenance(tmp_path, version, routines_at(version) & routines)
    return str(tmp_path)


def test_restart_reruns_an_older_engine_only_where_it_left_routines_to_multiwfn(tmp_path):
    from qtaim_gen.source.core.omol import _engine_outputs_present

    old = _engine_folder(tmp_path / "v1", version=1)
    v2 = _engine_folder(tmp_path / "v2", version=2)
    new = _engine_folder(tmp_path / "v6", version=6)
    # full_set 0: version 1 computed every level-0 routine
    assert _engine_outputs_present(old, spin_tf=True, full_set=0)
    # full_set 1 and 2: version 1 left elf_fuzzy (and the level-2 fuzzy) to Multiwfn
    assert not _engine_outputs_present(old, spin_tf=True, full_set=1)
    assert not _engine_outputs_present(old, spin_tf=True, full_set=2)
    # versions 2-5 left chelpg (full_set 1) to Multiwfn
    assert _engine_outputs_present(v2, spin_tf=True, full_set=0)
    assert not _engine_outputs_present(v2, spin_tf=True, full_set=1)
    assert _engine_outputs_present(new, spin_tf=True, full_set=2)


def test_restart_reruns_a_routine_a_later_version_changed(tmp_path, monkeypatch):
    from qtaim_gen.source.core.omol import _engine_outputs_present
    from qtaim_gen.source.data import engine_versions as ev

    folder = _engine_folder(tmp_path / "v", version=ENGINE_VERSION)
    assert _engine_outputs_present(folder, spin_tf=True, full_set=2)
    monkeypatch.setitem(ev.ENGINE_CHANGES, ENGINE_VERSION + 1,
                        {"added": frozenset(), "changed": frozenset({"mbis"}), "note": "test"})
    assert not _engine_outputs_present(folder, spin_tf=True, full_set=2)
