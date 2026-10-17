"""Routines that can never be computed for a wavefunction are recorded in timings.json
(TIMINGS_SKIPPED_KEY) and no longer expected by validation, so the folder completes
instead of rerunning them on every pass."""

import json
import logging

from qtaim_gen.source.data.multiwfn import MBIS_FAMILY
from qtaim_gen.source.utils.validation import (
    TIMINGS_SKIPPED_KEY,
    skipped_routines,
    validate_charge_dict,
    validate_fuzzy_dict,
    validate_timing_dict,
)

LOG = logging.getLogger("test_permanent_skips")


def _inp(folder, atoms):
    lines = ["! PBE0 def2-SVP", "* xyz 0 1"] + [f"{el} 0.0 0.0 {i:.1f}" for i, el in enumerate(atoms)] + ["*"]
    (folder / "orca.inp").write_text("\n".join(lines) + "\n")


def test_engine_skip_is_recorded_and_the_run_stamped(tmp_path, monkeypatch):
    from qtaim_gen.source.core import charge_engine, omol

    (tmp_path / "orca.wfx").write_text("wfx")
    monkeypatch.setattr(omol, "check_spin", lambda folder: False)
    wanted = omol._engine_routines(1, False)
    mbis = {"mbis", "mbis_fuzzy_density"} & wanted
    out = {r: {"x": 1.0} for r in wanted - mbis}
    out["_meta"] = {"skipped": [{"schemes": "mbis family", "reason": "ecp_disabled", "routines": list(MBIS_FAMILY)}]}
    monkeypatch.setattr(charge_engine, "run", lambda path, full_set=0: out)
    assert omol._run_charge_engine(str(tmp_path), n_threads=1, logger=LOG, full_set=1) is True
    timings = json.loads((tmp_path / "timings.json").read_text())
    assert timings["charge_engine"] > 0
    assert timings[TIMINGS_SKIPPED_KEY] == {r: {"reason": "ecp_disabled", "by": "charge_engine"} for r in mbis}


def test_a_routine_missing_without_a_skip_still_blocks_the_stamp(tmp_path, monkeypatch):
    from qtaim_gen.source.core import charge_engine, omol

    (tmp_path / "orca.wfx").write_text("wfx")
    monkeypatch.setattr(omol, "check_spin", lambda folder: False)
    wanted = sorted(omol._engine_routines(1, False))
    out = {r: {"x": 1.0} for r in wanted[1:]}
    out["_meta"] = {"skipped": [{"schemes": "mbis family", "reason": "ecp_disabled", "routines": list(MBIS_FAMILY)}]}
    monkeypatch.setattr(charge_engine, "run", lambda path, full_set=0: out)
    if wanted[0] in MBIS_FAMILY:  # the dropped routine must be one the skip does not cover
        out.pop(wanted[1])
    assert omol._run_charge_engine(str(tmp_path), n_threads=1, logger=LOG, full_set=1) is False
    assert "charge_engine" not in json.loads((tmp_path / "timings.json").read_text())


def test_multiwfn_skips_mbis_beyond_rn_at_level_1(tmp_path):
    from qtaim_gen.source.core.omol import _multiwfn_permanent_skips, _skip_routines
    from qtaim_gen.source.data.multiwfn import ENGINE_ROUTINES

    act = tmp_path / "act"
    act.mkdir()
    _inp(act, ["Pu", "Cl", "O"])
    assert set(_multiwfn_permanent_skips(str(act), 1)) == set(MBIS_FAMILY)
    assert _multiwfn_permanent_skips(str(act), 0) == {}
    assert _skip_routines(str(act), 1, charge_engine=False) == frozenset(MBIS_FAMILY)
    assert _skip_routines(str(act), 1, charge_engine=True) == ENGINE_ROUTINES

    rn = tmp_path / "rn"
    rn.mkdir()
    _inp(rn, ["Rn", "Pt"])
    assert _multiwfn_permanent_skips(str(rn), 1) == {}
    assert _skip_routines(str(rn), 1, charge_engine=False) is None


def test_validators_stop_expecting_skipped_routines(tmp_path):
    skipped = set(MBIS_FAMILY) | {"hirshfeld"}
    charges = {k: {"charge": {"1_U": 1.0}} for k in ("adch", "becke", "cm5")}
    (tmp_path / "charge.json").write_text(json.dumps(charges))
    assert not validate_charge_dict(str(tmp_path / "charge.json"), n_atoms=1)
    assert validate_charge_dict(str(tmp_path / "charge.json"), n_atoms=1, skipped=skipped)

    fz = {k: {"1_U": 1.0, "sum": 1.0, "abs_sum": 1.0} for k in ("becke_fuzzy_density", "hirsh_fuzzy_density", "elf_fuzzy")}
    (tmp_path / "fuzzy_full.json").write_text(json.dumps(fz))
    assert not validate_fuzzy_dict(str(tmp_path / "fuzzy_full.json"), n_atoms=1, full_set=1)
    assert validate_fuzzy_dict(str(tmp_path / "fuzzy_full.json"), n_atoms=1, full_set=1, skipped=skipped)


def test_timing_validation_and_patching_ignore_skipped_routines(tmp_path):
    from qtaim_gen.source.core.omol import patch_timings_from_log
    from qtaim_gen.source.utils.validation import get_expected_timing_keys

    expected, _ = get_expected_timing_keys(full_set=1, spin_tf=False)
    timings = {k: 1.0 for k in expected if k not in MBIS_FAMILY}
    timings[TIMINGS_SKIPPED_KEY] = {r: {"reason": "z_above_86", "by": "multiwfn"} for r in MBIS_FAMILY}
    (tmp_path / "timings.json").write_text(json.dumps(timings))
    assert skipped_routines(timings) == set(MBIS_FAMILY)
    assert validate_timing_dict(str(tmp_path / "timings.json"), full_set=1)
    del timings[TIMINGS_SKIPPED_KEY]
    (tmp_path / "timings.json").write_text(json.dumps(timings))
    assert not validate_timing_dict(str(tmp_path / "timings.json"), full_set=1)

    timings[TIMINGS_SKIPPED_KEY] = {r: {"reason": "z_above_86", "by": "multiwfn"} for r in MBIS_FAMILY}
    (tmp_path / "timings.json").write_text(json.dumps(timings))
    (tmp_path / "gbw_analysis.log").write_text("")
    assert patch_timings_from_log(str(tmp_path), full_set=1, spin_tf=False, move_results=False, logger=LOG) is not None
    patched = json.loads((tmp_path / "timings.json").read_text())
    assert not set(MBIS_FAMILY) & set(patched) - {TIMINGS_SKIPPED_KEY}


def test_parsers_fill_missing_schemes_with_nan_and_flag():
    import math

    from qtaim_gen.source.utils.lmdbs import parse_charge_data, parse_fuzzy_data

    rec = {"hirshfeld": {"charge": {"1_U": 1.2, "2_O": -0.6}, "dipole": {"mag": 0.3}}}
    atoms, glob = parse_charge_data(rec, 2, charge_filter=["hirshfeld", "mbis"], fill_missing=float("nan"))
    assert atoms[0]["charge_hirshfeld"] == 1.2 and math.isnan(atoms[1]["charge_mbis"])
    assert glob["missing_charge_mbis"] == 1.0 and glob["missing_charge_hirshfeld"] == 0.0
    assert "charge_mbis_dipole_mag" not in glob  # MBIS records carry no dipole
    atoms, glob = parse_charge_data(rec, 2, charge_filter=["hirshfeld", "mbis"])
    assert "charge_mbis" not in atoms[0] and "missing_charge_mbis" not in glob  # default: unchanged

    fz = {"becke_fuzzy_density": {"1_U": 90.0, "2_O": 8.0, "sum": 98.0, "abs_sum": 98.0}}
    atoms, glob = parse_fuzzy_data(fz, 2, fuzzy_filter=["becke_fuzzy_density", "mbis_fuzzy_density"],
                                   fill_missing=float("nan"))
    assert math.isnan(atoms[0]["fuzzy_mbis_fuzzy_density"]) and math.isnan(glob["fuzzy_mbis_fuzzy_density_sum"])
    assert glob["missing_fuzzy_mbis_fuzzy_density"] == 1.0 and glob["missing_fuzzy_becke_fuzzy_density"] == 0.0


def test_converter_keeps_a_molecule_whose_scheme_is_missing(tmp_path):
    import math
    import pickle
    import shutil
    from pathlib import Path

    import lmdb

    from qtaim_gen.source.core.converter import GeneralConverter
    from qtaim_gen.source.utils.scaling import _deserialize_graph

    base = Path(__file__).parent / "test_files" / "lmdb_tests" / "generator_lmdbs_merged"

    def run(fill):
        work = tmp_path / f"fill_{fill}"
        work.mkdir()
        shutil.copy(base / "merged_charge.lmdb", work / "charge.lmdb")
        env = lmdb.open(str(work / "charge.lmdb"), subdir=False, map_size=1 << 30)
        with env.begin(write=True) as txn:
            rec = pickle.loads(txn.get(b"orca5_uks"))
            rec.pop("mbis")
            txn.put(b"orca5_uks", pickle.dumps(rec))
        env.close()
        cfg = {"chunk": -1, "filter_list": ["scaled", "length"], "restart": False,
               "allowed_ring_size": [3, 4, 5, 6, 7, 8], "allowed_charges": None, "allowed_spins": None,
               "keys_target": {"atom": [], "bond": [], "global": []},
               "keys_data": {"atom": [], "bond": [], "global": ["n_atoms"]}, "lmdb_path": str(work), "lmdb_name": "g.lmdb",
               "lmdb_locations": {"geom_lmdb": str(base / "merged_geom.lmdb"), "charge_lmdb": str(work / "charge.lmdb")},
               "n_workers": 1, "batch_size": 100, "charge_filter": ["mbis", "hirshfeld"], "fill_missing_schemes": fill}
        GeneralConverter(cfg).process()
        out = lmdb.open(str(work / "g.lmdb"), subdir=False, readonly=True, lock=False)
        with out.begin() as txn:
            graphs = [_deserialize_graph(v) for k, v in txn.cursor() if k.isdigit()]
            names = pickle.loads(txn.get(b"feature_names"))
        out.close()
        return graphs, names

    dropped, _ = run(False)
    graphs, names = run(True)
    assert len(graphs) == len(dropped) + 1  # the molecule without MBIS is kept, not dropped
    i_mbis = names["atom"].index("charge_mbis")
    i_flag = names["global"].index("missing_charge_mbis")
    flagged = [g for g in graphs if g["global"].feat[0, i_flag] == 1.0]
    assert len(flagged) == 1
    assert all(math.isnan(v) for v in flagged[0]["atom"].feat[:, i_mbis].tolist())
    others = [g for g in graphs if g["global"].feat[0, i_flag] == 0.0]
    assert others and all(not math.isnan(v) for g in others for v in g["atom"].feat[:, i_mbis].tolist())
