"""ENGINE_VERSION / ENGINE_CHANGES (data/engine_versions.py) and provenance.json
(utils/provenance.py): which engine routines a folder holds current values for."""

import json

from qtaim_gen.source.data.engine_versions import (
    ENGINE_CHANGES, ENGINE_VERSION, ROUTINE_ENGINE, is_current, last_change, routines_at,
)
from qtaim_gen.source.data.multiwfn import ENGINE_ROUTINES
from qtaim_gen.source.utils import provenance as P


def test_version_table():
    assert max(ENGINE_CHANGES) == ENGINE_VERSION
    assert routines_at(ENGINE_VERSION) == frozenset(ROUTINE_ENGINE)
    assert set(ROUTINE_ENGINE) == ENGINE_ROUTINES | {"qtaim", "convert"}
    # every routine was added exactly once
    added = [r for c in ENGINE_CHANGES.values() for r in c["added"]]
    assert len(added) == len(set(added))
    assert last_change("elf_fuzzy") == 2 and last_change("other_esp") == 3 and last_change("hirshfeld") == 1
    assert last_change("chelpg") == 0
    assert "other_esp" not in routines_at(2) and "elf_fuzzy" in routines_at(2)


def test_is_current():
    assert is_current("other_esp", 3) and is_current("other_esp", ENGINE_VERSION)
    assert not is_current("other_esp", 2)
    assert not is_current("chelpg", ENGINE_VERSION)  # not an engine routine
    assert not is_current("hirshfeld", None)


def test_a_change_makes_older_records_stale(monkeypatch):
    from qtaim_gen.source.data import engine_versions as ev

    monkeypatch.setitem(ev.ENGINE_CHANGES, ENGINE_VERSION + 1,
                        {"added": frozenset(), "changed": frozenset({"mbis"}), "note": "test"})
    assert not ev.is_current("mbis", ENGINE_VERSION)
    assert ev.is_current("hirshfeld", ENGINE_VERSION)


def test_record_load_merge_and_drop(tmp_path):
    folder = tmp_path / "job"
    (folder / "generator").mkdir(parents=True)
    # a previous run, moved to generator/
    gen = P.empty()
    gen["routines"] = {"hirshfeld": {"by": "charge_engine", "engine_version": 3},
                       "chelpg": {"by": "multiwfn"}}
    (folder / "generator" / P.PROVENANCE_FILE).write_text(json.dumps(gen))

    P.record_routines(str(folder), ["hirshfeld", "adch"], "charge_engine", ENGINE_VERSION, spec={"grid": [75, 434, 10.0]})
    P.record_skips(str(folder), {"mbis": {"reason": "z_above_118", "by": "charge_engine"}})
    P.record_engine(str(folder), "charge_engine", {"nelec_wfx": 10})
    P.record_converter(str(folder), "native", "fixed")
    prov = P.load(str(folder))
    assert prov["routines"]["hirshfeld"]["engine_version"] == ENGINE_VERSION  # root on top
    assert prov["routines"]["chelpg"] == {"by": "multiwfn"}  # kept from generator/
    assert prov["routines"]["adch"]["spec"] == {"grid": [75, 434, 10.0]}
    assert P.skipped(prov) == {"mbis"}
    assert prov["engines"]["charge_engine"] == {"nelec_wfx": 10}
    assert prov["converter"] == {"by": "native", "mode": "fixed", "engine_version": ENGINE_VERSION}
    assert P.engine_routines(prov) == {"hirshfeld", "adch"}
    assert P.produced_by(prov, "chelpg") == "multiwfn" and not P.engine_current(prov, "chelpg")

    # a routine recorded later is no longer skipped
    P.record_routines(str(folder), ["mbis"], "charge_engine", ENGINE_VERSION)
    assert "mbis" not in P.skipped(P.load(str(folder)))

    P.drop(str(folder), ["hirshfeld"], converter=True)
    prov = P.load(str(folder))
    assert "hirshfeld" not in prov["routines"] and "converter" not in prov
    assert "chelpg" in prov["routines"]


def test_load_without_files(tmp_path):
    assert P.load(str(tmp_path)) == P.empty()


def test_multiwfn_build(tmp_path):
    out = tmp_path / "hirshfeld.out"
    out.write_text(" Multiwfn -- A Multifunctional Wavefunction Analyzer\n Version 3.8(dev), update date: 2024-Oct-6\n")
    assert P.multiwfn_build(str(out), "/opt/Multiwfn") == {"binary": "/opt/Multiwfn", "version": "3.8(dev)",
                                                            "date": "2024-Oct-6"}
    assert P.multiwfn_build(str(tmp_path / "missing.out")) == {}


def test_json_to_lmdb_provenance(tmp_path):
    """json_2_lmdbs turns generator/provenance.json into provenance.lmdb, keyed like the other types."""
    import os
    import pickle

    import lmdb

    from qtaim_gen.source.utils.lmdbs import json_2_lmdbs

    for name in ("job_a", "job_b"):
        gen = tmp_path / "jobs" / name / "generator"
        gen.mkdir(parents=True)
        prov = P.empty()
        prov["routines"]["hirshfeld"] = {"by": "charge_engine", "engine_version": ENGINE_VERSION}
        (gen / P.PROVENANCE_FILE).write_text(json.dumps(prov))
    out = tmp_path / "out"
    out.mkdir()
    json_2_lmdbs(root_dir=str(tmp_path / "jobs") + os.sep, out_dir=str(out) + os.sep, data_type="provenance",
                 out_lmdb="provenance.lmdb", chunk_size=10, clean=True, merge=True, move_files=True)
    env = lmdb.open(str(out / "provenance.lmdb"), subdir=False, readonly=True, lock=False)
    with env.begin() as txn:
        assert sorted(k.decode() for k, _ in txn.cursor() if k.decode() != "length") == ["job_a", "job_b"]
        assert pickle.loads(txn.get(b"job_a"))["routines"]["hirshfeld"]["by"] == "charge_engine"
    env.close()
