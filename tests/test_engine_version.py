"""Charge-engine version stamp (TIMINGS_ENGINE_VERSIONS_KEY): which routines a
folder's charge_engine timing vouches for depends on the engine version that ran."""

import json

import pytest

from qtaim_gen.source.data.multiwfn import (
    CHARGE_ENGINE_ROUTINES_ADDED,
    CHARGE_ENGINE_VERSION,
    ENGINE_ROUTINES,
    engine_routines_at,
)
from qtaim_gen.source.utils.validation import (
    TIMINGS_ENGINE_VERSIONS_KEY,
    TIMINGS_SKIPPED_KEY,
    TIMINGS_SURFACE_ROUTINES_KEY,
    charge_engine_version,
    engine_routines_of,
    surface_engine_routines_of,
    timing_total,
    validate_timing_dict,
)

V2_ROUTINES = {"elf_fuzzy", "grad_norm_rho_fuzzy", "laplacian_rho_fuzzy"}
V3_ROUTINES = {"other_esp"}


def test_routines_by_version():
    assert engine_routines_at(CHARGE_ENGINE_VERSION) == ENGINE_ROUTINES
    assert engine_routines_at(1) == ENGINE_ROUTINES - V2_ROUTINES - V3_ROUTINES
    assert engine_routines_at(2) == ENGINE_ROUTINES - V3_ROUTINES
    assert CHARGE_ENGINE_ROUTINES_ADDED[2] == V2_ROUTINES
    assert CHARGE_ENGINE_ROUTINES_ADDED[3] == V3_ROUTINES


def test_version_of_timings():
    assert charge_engine_version({"qtaim": 1.0}) is None
    assert charge_engine_version({"charge_engine": 0.0}) is None
    # a charge_engine timing without a stamp predates it
    assert charge_engine_version({"charge_engine": 2.0}) == 1
    stamped = {"charge_engine": 2.0, TIMINGS_ENGINE_VERSIONS_KEY: {"charge_engine": CHARGE_ENGINE_VERSION}}
    assert charge_engine_version(stamped) == CHARGE_ENGINE_VERSION
    assert engine_routines_of(stamped) == ENGINE_ROUTINES
    assert engine_routines_of({"charge_engine": 2.0}) == ENGINE_ROUTINES - V2_ROUTINES - V3_ROUTINES
    assert engine_routines_of({"qtaim": 1.0}) == frozenset()


def test_surface_routines_of_timings():
    assert surface_engine_routines_of({"qtaim": 1.0}) == frozenset()
    # no routine record: the routines of the stamped version (no stamp: version 1)
    assert surface_engine_routines_of({"surface_engine": 1.0}) == {"other_alie", "other_geometry"}
    v3 = {"surface_engine": 1.0, TIMINGS_ENGINE_VERSIONS_KEY: {"surface_engine": 3}}
    assert surface_engine_routines_of(v3) == {"other_alie", "other_geometry", "other_esp"}
    # the record says what was computed, whatever the stamp
    assert surface_engine_routines_of({**v3, TIMINGS_SURFACE_ROUTINES_KEY: ["other_alie"]}) == {"other_alie"}
    assert surface_engine_routines_of({TIMINGS_SURFACE_ROUTINES_KEY: ["other_alie"]}) == frozenset()


def test_timing_total_skips_metadata():
    timings = {"qtaim": 2.0, "charge_engine": 3.0, TIMINGS_ENGINE_VERSIONS_KEY: {"charge_engine": 2},
               TIMINGS_SKIPPED_KEY: {"mbis": {"reason": "x", "by": "charge_engine"}}}
    assert timing_total(timings) == 5.0


@pytest.mark.parametrize("stamp, needs_elf_timing", [(None, True), (2, False)])
def test_validation_expects_elf_timing_from_a_v1_engine(tmp_path, stamp, needs_elf_timing):
    """A version-1 engine left elf_fuzzy to Multiwfn, so its folders still need the
    elf_fuzzy timing; a version-2 engine computed it."""
    timings = {"qtaim": 5.0, "other": 3.0, "charge_engine": 2.0, "surface_engine": 1.0,
               "chelpg": 1.0, "ibsi_bond": 1.0}
    if stamp:
        timings[TIMINGS_ENGINE_VERSIONS_KEY] = {"charge_engine": stamp}
    path = tmp_path / "timings.json"
    path.write_text(json.dumps(timings))
    assert validate_timing_dict(str(path), full_set=1, charge_engine=True) is not needs_elf_timing
    timings["elf_fuzzy"] = 1.0
    path.write_text(json.dumps(timings))
    assert validate_timing_dict(str(path), full_set=1, charge_engine=True)


def _engine_folder(tmp_path, stamp):
    from qtaim_gen.source.core.omol import _engine_routines

    gen = tmp_path / "generator"
    gen.mkdir()
    timings = {"charge_engine": 2.0}
    if stamp:
        timings[TIMINGS_ENGINE_VERSIONS_KEY] = {"charge_engine": stamp}
    (gen / "timings.json").write_text(json.dumps(timings))
    routines = _engine_routines(2, spin_tf=True)
    charge = {r: {"charge": {"1_H": 0.1}} for r in ("hirshfeld", "adch", "cm5", "becke", "vdd", "mbis")}
    (gen / "charge.json").write_text(json.dumps(charge))
    (gen / "bond.json").write_text(json.dumps({"fuzzy_bond": {"1_H_to_2_H": 1.0}}))
    fuzzy = {r: {"1_H": 1.0} for r in routines if r not in charge and r != "fuzzy_bond"}
    (gen / "fuzzy_full.json").write_text(json.dumps(fuzzy))
    return str(tmp_path)


def test_restart_reruns_an_older_engine_only_where_it_left_routines_to_multiwfn(tmp_path):
    from qtaim_gen.source.core.omol import _engine_outputs_present

    (tmp_path / "v1").mkdir()
    (tmp_path / "v2").mkdir()
    old = _engine_folder(tmp_path / "v1", stamp=None)
    new = _engine_folder(tmp_path / "v2", stamp=2)
    # full_set 0: version 1 computed every level-0 routine
    assert _engine_outputs_present(old, spin_tf=True, full_set=0)
    # full_set 1 and 2: version 1 left elf_fuzzy (and the level-2 fuzzy) to Multiwfn
    assert not _engine_outputs_present(old, spin_tf=True, full_set=1)
    assert not _engine_outputs_present(old, spin_tf=True, full_set=2)
    assert _engine_outputs_present(new, spin_tf=True, full_set=2)
