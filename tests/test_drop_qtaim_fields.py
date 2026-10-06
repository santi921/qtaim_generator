"""delta_g_promolecular is left out of qtaim.lmdb and of graph features (mixed Multiwfn builds)."""

import json
import os
import pickle

import lmdb

from qtaim_gen.source.utils.lmdbs import DROPPED_QTAIM_FIELDS, json_2_lmdbs, parse_qtaim_data

QTAIM = {
    "0": {"cp_num": 1, "element": "H", "density_all": 0.4, "delta_g_promolecular": 0.04, "delta_g_hirsh": 0.05},
    "1": {"cp_num": 2, "element": "H", "density_all": 0.4, "delta_g_promolecular": 0.04, "delta_g_hirsh": 0.05},
    "0_1": {"cp_num": 3, "connected_bond_paths": [0, 1], "density_all": 0.25,
            "delta_g_promolecular": 0.11, "delta_g_hirsh": 0.22},
}


def _read(path):
    env = lmdb.open(path, subdir=False, readonly=True, lock=False)
    with env.begin() as txn:
        out = {k.decode(): pickle.loads(v) for k, v in txn.cursor() if k != b"length"}
    env.close()
    return out


def test_json_to_lmdb_leaves_the_field_out(tmp_path):
    root = tmp_path / "root"
    (root / "job").mkdir(parents=True)
    (root / "job" / "qtaim.json").write_text(json.dumps(QTAIM))
    out = str(tmp_path / "out") + os.sep
    os.makedirs(out)
    json_2_lmdbs(root_dir=str(root) + os.sep, out_dir=out, data_type="qtaim", out_lmdb="qtaim.lmdb",
                 chunk_size=10, clean=True, merge=True)
    record = _read(os.path.join(out, "qtaim.lmdb"))["job"]
    assert set(record) == set(QTAIM)
    for cp, values in record.items():
        assert "delta_g_promolecular" not in values
        assert values["delta_g_hirsh"] == QTAIM[cp]["delta_g_hirsh"]
    # the qtaim.json on disk is not touched
    assert json.loads((root / "job" / "qtaim.json").read_text()) == QTAIM


def test_auto_discovered_and_explicit_keys_skip_the_field():
    for keys in (None, ["density_all", "delta_g_promolecular", "delta_g_hirsh"]):
        atom_keys, bond_keys, atom_feats, bond_feats, _ = parse_qtaim_data(
            QTAIM, {}, {}, atom_keys=keys, bond_keys=keys)
        for field in DROPPED_QTAIM_FIELDS:
            assert field not in atom_keys and field not in bond_keys
            assert all(field not in v for v in atom_feats.values())
            assert all(field not in v for v in bond_feats.values())
        assert atom_feats[0]["delta_g_hirsh"] == 0.05 and bond_feats[(0, 1)]["delta_g_hirsh"] == 0.22


def test_explicit_key_list_is_not_mutated():
    keys = ["density_all", "delta_g_promolecular"]
    parse_qtaim_data(QTAIM, {}, {}, atom_keys=keys, bond_keys=keys)
    assert keys == ["density_all", "delta_g_promolecular"]
