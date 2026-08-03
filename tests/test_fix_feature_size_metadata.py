"""Tests for the feature_size/feature_names metadata patch helper."""
import pickle

import lmdb
import pytest

from qtaim_gen.source.scripts.helpers.fix_lmdb_feature_size_metadata import (
    _corrected_schema,
    fix_one,
)


def _write_lmdb(path, feature_names, feature_size, target_dict, n_graphs=3):
    db = lmdb.open(str(path), subdir=False, map_size=10 ** 7)
    with db.begin(write=True) as txn:
        for i in range(n_graphs):
            txn.put(f"{i}".encode("ascii"), pickle.dumps({"molecule_graph": b"x"}, protocol=-1))
        txn.put(b"length", pickle.dumps(n_graphs, protocol=-1))
        txn.put(b"feature_names", pickle.dumps(feature_names, protocol=-1))
        txn.put(b"feature_size", pickle.dumps(feature_size, protocol=-1))
        txn.put(b"target_dict", pickle.dumps(target_dict, protocol=-1))
    db.sync()
    db.close()


def _read_meta(path):
    e = lmdb.open(str(path), subdir=False, readonly=True, lock=False)
    with e.begin() as t:
        fn = pickle.loads(t.get(b"feature_names"))
        fs = pickle.loads(t.get(b"feature_size"))
    e.close()
    return fn, fs


def test_corrected_schema_trims_trailing_targets():
    feature_names = {
        "atom": ["sym_C", "sym_H", "charge_adch", "charge_cm5"],
        "bond": ["len", "boo"],
        "global": ["n_atoms"],
    }
    target_dict = {"atom": ["charge_adch", "charge_cm5"], "bond": [], "global": ["n_atoms"]}
    new_names, new_size, changed = _corrected_schema(feature_names, target_dict)
    assert changed
    assert new_names["atom"] == ["sym_C", "sym_H"]
    assert new_size == {"atom": 2, "bond": 2, "global": 0}


def test_corrected_schema_noop_without_targets():
    feature_names = {"atom": ["sym_C", "sym_H"], "bond": ["len"], "global": ["n_atoms"]}
    target_dict = {"atom": [], "bond": [], "global": []}
    new_names, new_size, changed = _corrected_schema(feature_names, target_dict)
    assert not changed
    assert new_size == {"atom": 2, "bond": 1, "global": 1}


def test_corrected_schema_rejects_mismatched_trailing():
    # targets not at the tail -> refuse to trim
    feature_names = {"atom": ["charge_adch", "sym_C", "sym_H"]}
    target_dict = {"atom": ["charge_adch"]}
    with pytest.raises(ValueError, match="do not match targets"):
        _corrected_schema(feature_names, target_dict)


def test_fix_one_patches_inflated_metadata(tmp_path):
    path = tmp_path / "data.lmdb"
    # inflated: feature_size includes the 2 charge targets that are not in .feat
    _write_lmdb(
        path,
        feature_names={"atom": ["sym_C", "sym_H", "charge_adch", "charge_cm5"],
                       "bond": ["len"], "global": ["n_atoms"]},
        feature_size={"atom": 4, "bond": 1, "global": 1},
        target_dict={"atom": ["charge_adch", "charge_cm5"], "bond": [], "global": []},
    )
    fix_one(str(path))
    fn, fs = _read_meta(path)
    assert fs == {"atom": 2, "bond": 1, "global": 1}
    assert fn["atom"] == ["sym_C", "sym_H"]


def test_fix_one_dry_run_does_not_write(tmp_path):
    path = tmp_path / "data.lmdb"
    _write_lmdb(
        path,
        feature_names={"atom": ["sym_C", "charge_adch"], "bond": [], "global": []},
        feature_size={"atom": 2, "bond": 0, "global": 0},
        target_dict={"atom": ["charge_adch"], "bond": [], "global": []},
    )
    fix_one(str(path), dry_run=True)
    _, fs = _read_meta(path)
    assert fs == {"atom": 2, "bond": 0, "global": 0}  # unchanged
