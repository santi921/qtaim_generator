"""Engine version in provenance.json: which routines a folder's engine records vouch for
depends on the ENGINE_VERSION they were computed at (data/engine_versions.py)."""

import json

import pytest

from qtaim_gen.source.data.engine_versions import ENGINE_VERSION, ROUTINE_ENGINE, is_current, last_change, routines_at
from qtaim_gen.source.utils import provenance
from qtaim_gen.source.utils.validation import TIMINGS_PATCHED_KEY, timing_total, validate_timing_dict


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


@pytest.mark.parametrize("version, needs_elf_timing", [(1, True), (2, False), (ENGINE_VERSION, False)])
def test_validation_expects_elf_timing_from_a_v1_engine(tmp_path, version, needs_elf_timing):
    """A version-1 engine left elf_fuzzy to Multiwfn, so its folders still need the
    elf_fuzzy timing; version 2 and later computed it."""
    timings = {"qtaim": 5.0, "other": 3.0, "charge_engine": 2.0, "surface_engine": 1.0,
               "chelpg": 1.0, "ibsi_bond": 1.0}
    _write_provenance(tmp_path, version, routines_at(version) - {"qtaim"})
    path = tmp_path / "generator" / "timings.json"
    path.write_text(json.dumps(timings))
    assert validate_timing_dict(str(path), full_set=1, charge_engine=True) is not needs_elf_timing
    timings["elf_fuzzy"] = 1.0
    path.write_text(json.dumps(timings))
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
