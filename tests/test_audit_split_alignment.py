import gzip
import os
import pickle

import lmdb
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")
pytest.importorskip("qtaim_embed")

from torch_geometric.data import HeteroData  # noqa: E402
from qtaim_embed.data.lmdb import serialize_graph  # noqa: E402

from qtaim_gen.source.scripts.helpers.audit_split_alignment import main  # noqa: E402

MAPPING = {
    "v__a": "train", "v__b": "train", "v__c": "train",
    "v__d": "val", "v__e": "test",
    "v__h": "H7,H8",
}


def _write_mapping(path):
    with gzip.open(path, "wt") as f:
        f.write("key\tdestination\n")
        for k, d in MAPPING.items():
            f.write(f"{k}\t{d}\n")


def _write_descriptor(set_dir, keys):
    os.makedirs(set_dir, exist_ok=True)
    for dt in ("structure", "charge"):
        env = lmdb.open(os.path.join(set_dir, f"{dt}.lmdb"), subdir=False, map_size=1 << 24)
        with env.begin(write=True) as txn:
            for k in keys:
                txn.put(k.encode(), pickle.dumps({}))
            txn.put(b"length", pickle.dumps(len(keys)))
        env.close()


def _write_graphs(path, mol_names):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    env = lmdb.open(path, subdir=False, map_size=1 << 24)
    with env.begin(write=True) as txn:
        for i, name in enumerate(mol_names):
            g = HeteroData()
            g["atom"].feat = torch.zeros(1, 1)
            g.mol_name = name
            txn.put(str(i).encode(), pickle.dumps({"molecule_graph": serialize_graph(g)}))
        txn.put(b"length", pickle.dumps(len(mol_names)))
    env.close()


def _run(tmp_path, graphs, workers="1", extra=()):
    mapping = tmp_path / "map.tsv.gz"
    _write_mapping(mapping)
    desc = tmp_path / "desc"
    _write_descriptor(desc / "train", ["v__a", "v__b", "v__c"])
    _write_descriptor(desc / "val", ["v__d"])
    _write_descriptor(desc / "test", ["v__e"])
    gr = tmp_path / "graphs"
    for rel, names in graphs.items():
        _write_graphs(str(gr / rel / "merged" / "merged.lmdb"), names)
    report = tmp_path / "report.json"
    moves = tmp_path / "moves.tsv"
    rc = main(["--mapping", str(mapping), "--descriptor_root", str(desc), "--graph_root", str(gr),
               "--workers", workers, "--report", str(report), "--moves_tsv", str(moves), *extra])
    import json
    return rc, json.loads(report.read_text()), moves.read_text().splitlines()[1:]


def test_aligned(tmp_path):
    rc, rep, moves = _run(tmp_path, {
        "train": ["v__a", "v__b"],  # v__c missing = build drop, not a failure
        "val": ["v__d"], "test": ["v__e"],
        "holdouts/H7": ["v__h"], "holdouts/H8": ["v__h"],  # dual-suite key is legal in both
    })
    assert rc == 0
    assert rep["problems"] == []
    assert moves == []
    cm = rep["sets"]["train"]["graph"]["canonical_missing"]
    assert cm["absent_but_descriptor_present"] == 1
    assert rep["graph_cross_set_duplicates"] == {}


@pytest.mark.parametrize("workers", ["1", "2"])
def test_misplaced_duplicate_unknown(tmp_path, workers):
    rc, rep, moves = _run(tmp_path, {
        "train": ["v__a", "v__b", "v__c", "v__d", "v__a"],  # v__d belongs in val; v__a duplicated
        "val": ["v__zz"],  # not in the mapping
        "test": ["v__e"],
        "holdouts/H7": ["v__e"],  # test key also in a holdout: cross-set duplicate
    }, workers=workers)
    assert rc == 1
    tr = rep["sets"]["train"]["graph"]
    assert tr["vs_canonical"]["misplaced_to"] == {"val": 1}
    assert tr["duplicate_within_set"] == 1
    assert tr["not_in_same_set_descriptor"] == 1
    val = rep["sets"]["val"]["graph"]
    assert val["vs_canonical"]["not_in_mapping"] == 1
    assert val["canonical_missing"]["in_other_graph_set"] == 1
    assert rep["sets"]["H7"]["graph"]["vs_canonical"]["misplaced_to"] == {"test": 1}
    assert rep["graph_cross_set_duplicates"] == {"H7&test": 1}
    rows = {tuple(r.split("\t")[i] for i in (0, 1, 4)) for r in moves}
    assert ("v__d", "train", "val") in rows
    assert ("v__zz", "val", "NOT_IN_MAPPING") in rows
    assert rep["sets"]["train"]["descriptor"]["vs_canonical"]["misplaced"] == 0


@pytest.mark.parametrize("flag", ["--descriptor_root", "--graph_root"])
def test_missing_or_empty_root_fails(tmp_path, flag):
    mapping = tmp_path / "map.tsv.gz"
    _write_mapping(mapping)
    with pytest.raises(SystemExit):
        main(["--mapping", str(mapping), flag, str(tmp_path / "does_not_exist")])
    (tmp_path / "empty").mkdir()
    with pytest.raises(SystemExit):
        main(["--mapping", str(mapping), flag, str(tmp_path / "empty")])


def test_prefix_vertical_from_parent(tmp_path):
    mapping = tmp_path / "map.tsv.gz"
    _write_mapping(mapping)
    gr = tmp_path / "splits"
    _write_graphs(str(gr / "v" / "train" / "shard_0.lmdb"), ["a", "b"])
    _write_graphs(str(gr / "v" / "train" / "shard_1.lmdb"), ["d"])
    _write_graphs(str(gr / "v" / "val" / "shard_0.lmdb"), ["d"])
    report = tmp_path / "r.json"
    rc = main(["--mapping", str(mapping), "--graph_root", str(gr), "--workers", "1",
               "--prefix_vertical_from_parent", "--report", str(report)])
    import json
    rep = json.loads(report.read_text())
    assert rc == 1
    assert rep["sets"]["train"]["graph"]["n_records"] == 3
    assert rep["sets"]["train"]["graph"]["vs_canonical"]["misplaced_to"] == {"val": 1}
    assert rep["graph_cross_set_duplicates"] == {"train&val": 1}
