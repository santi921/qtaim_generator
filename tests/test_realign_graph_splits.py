import json
import os
import pickle

import lmdb
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")
pytest.importorskip("qtaim_embed")

from torch_geometric.data import HeteroData  # noqa: E402
from qtaim_embed.data.lmdb import load_graph_from_serialized, serialize_graph  # noqa: E402

from qtaim_gen.source.scripts.helpers.realign_graph_splits import main  # noqa: E402
from qtaim_gen.source.utils.scaling import (  # noqa: E402
    apply_scalers_to_lmdb_inplace,
    fit_scalers_on_lmdbs,
    save_scalers,
)

SKIP = {"length", "scaled", "split_name", "scaler_finalized"}
OLD = {"train": [f"tr{i}" for i in range(12)], "val": ["va0", "va1", "va2"], "test": ["te0", "te1", "te2"]}
MOVES = [("tr0", "train", "test"), ("tr1", "train", "val"), ("va0", "val", "train"), ("te2", "test", "train")]
HOLDOUT = ["h0", "h1"]


def _raw_graph(name, dtype=torch.float64):
    seed = sum(ord(c) * (i + 1) for i, c in enumerate(name))
    gen = torch.Generator().manual_seed(seed)
    n = 2 + seed % 3
    g = HeteroData()
    g["atom"].feat = (torch.randn(n, 4, generator=gen) * 4 + 2).to(dtype)
    g["atom"].feat[:, 2] = 1.0  # constant column: guarded to std 1.0, zero shift
    g["atom"].feat[:, 3] = (torch.randn(n, generator=gen) * 1e-8).to(dtype)  # near-constant column
    g["atom"].labels = (torch.randn(n, 2, generator=gen) * 0.5).to(dtype)
    g["global"].feat = torch.randn(1, 2, generator=gen).to(dtype)
    g.mol_name = name
    return g


def _write(path, names):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    env = lmdb.open(path, subdir=False, map_size=1 << 26)
    with env.begin(write=True) as txn:
        for i, name in enumerate(names):
            txn.put(str(i).encode(), pickle.dumps({"molecule_graph": serialize_graph(_raw_graph(name))}))
        txn.put(b"length", pickle.dumps(len(names)))
        txn.put(b"scaled", pickle.dumps(False))
        txn.put(b"processed_source_keys", pickle.dumps(set(names)))
    env.close()


def _read(path):
    env = lmdb.open(path, subdir=False, readonly=True, lock=False)
    out, meta = {}, {}
    with env.begin() as txn:
        for k, v in txn.cursor():
            if k.isdigit():
                obj = pickle.loads(v)
                g = load_graph_from_serialized(obj["molecule_graph"])
                out[g.mol_name] = (int(k), g)
            else:
                meta[k] = pickle.loads(v)
    env.close()
    return out, meta


def _merged(root, rel):
    return str(root / rel / "merged" / "merged.lmdb")


@pytest.fixture
def old_tree(tmp_path):
    root = tmp_path / "old"
    for s, names in OLD.items():
        _write(_merged(root, s), names)
    _write(_merged(root, "holdouts/H1"), HOLDOUT)
    f, lab = fit_scalers_on_lmdbs([_merged(root, "train")], SKIP)
    save_scalers(f, lab, str(root / "train" / "merged"))
    for rel in ("train", "val", "test", "holdouts/H1"):
        apply_scalers_to_lmdb_inplace(_merged(root, rel), f, lab, SKIP)
    moves = tmp_path / "moves.tsv"
    idx = {s: {n: i for i, n in enumerate(names)} for s, names in OLD.items()}
    with open(moves, "w") as fh:
        fh.write("mol_name\tcurrent_set\tlmdb\trecord_key\tcanonical\n")
        for name, cur, dst in MOVES:
            fh.write(f"{name}\t{cur}\t{_merged(root, cur)}\t{idx[cur][name]}\t{dst}\n")
    return root, moves


def _expected_membership():
    sets = {s: set(n) for s, n in OLD.items()}
    for name, cur, dst in MOVES:
        sets[cur].discard(name)
        sets[dst].add(name)
    return sets


def test_move_scaler_check_rescale(tmp_path, old_tree):
    root, moves = old_tree
    new = tmp_path / "new"
    assert main(["--phase", "move", "--graph_root", str(root), "--out_root", str(new),
                 "--moves_tsv", str(moves)]) == 0

    want = _expected_membership()
    old_graphs = {n: g for s in OLD for n, (_, g) in _read(_merged(root, s))[0].items()}
    for s in OLD:
        graphs, meta = _read(_merged(new, s))
        assert set(graphs) == want[s]
        assert sorted(i for i, _ in graphs.values()) == list(range(len(want[s])))
        assert meta[b"length"] == len(want[s])
        assert meta[b"processed_source_keys"] == want[s]
        assert meta[b"scaled"] is True
        for n, (_, g) in graphs.items():  # bytes copied through unchanged
            assert torch.equal(g["atom"].feat, old_graphs[n]["atom"].feat)
    assert os.path.samefile(_merged(root, "holdouts/H1"), _merged(new, "holdouts/H1"))
    rep = json.loads((new / "realign_report.json").read_text())
    assert rep["splits"]["train"] == {"old_length": 12, "kept": 10, "moved_out": 2, "moved_in": 2, "new_length": 12}

    assert main(["--phase", "scaler_check", "--out_root", str(new)]) == 0
    chk = json.loads((new / "scaler_check.json").read_text())
    assert chk["train_graphs"] == 12
    assert chk["inverse_roundtrip_max_abs_err"] < 1e-5
    atom = chk["features"]["atom"]
    assert list(atom["near_constant_cols"]) == ["3"]  # reported with raw values, not ranked
    assert 3 not in atom["worst_cols"]

    # refit must equal a scaler fit directly on the raw new-train graphs
    ref_dir = tmp_path / "ref"
    _write(str(ref_dir / "raw.lmdb"), sorted(want["train"]))
    ref_f, ref_l = fit_scalers_on_lmdbs([str(ref_dir / "raw.lmdb")], SKIP)
    refit = torch.load(new / "train" / "merged" / "refit" / "feature_scaler_iterative.pt", weights_only=False)
    for nt in ref_f.mean:
        assert torch.allclose(refit["mean"][nt].double(), ref_f.mean[nt].double(), atol=1e-5)
        assert torch.allclose(refit["std"][nt].double(), ref_f.std[nt].double(), atol=1e-5)

    graphs_dtype = {rel: next(iter(_read(_merged(new, rel))[0].values()))[1]["atom"].feat.dtype
                    for rel in list(OLD) + ["holdouts/H1"]}
    assert main(["--phase", "rescale", "--out_root", str(new)]) == 0
    for rel, names in [(s, want[s]) for s in OLD] + [("holdouts/H1", set(HOLDOUT))]:
        graphs, _ = _read(_merged(new, rel))
        for n in names:
            expect = ref_l([ref_f([_raw_graph(n)])[0]])[0]
            got = graphs[n][1]
            assert torch.allclose(got["atom"].feat, expect["atom"].feat, atol=1e-4)
            assert torch.allclose(got["atom"].labels, expect["atom"].labels, atol=1e-4)
            assert torch.allclose(got["global"].feat, expect["global"].feat, atol=1e-4)
            assert got["atom"].feat.dtype == graphs_dtype[rel]
    # source holdout untouched by the rescale (hardlink replaced, not written through)
    src_h, _ = _read(_merged(root, "holdouts/H1"))
    assert not os.path.samefile(_merged(root, "holdouts/H1"), _merged(new, "holdouts/H1"))
    assert torch.equal(src_h["h0"][1]["atom"].feat, old_graphs_h0(root))
    assert os.path.isdir(new / "train" / "merged" / "pre_refit")

    # re-running rescale is a no-op (markers)
    before = _read(_merged(new, "val"))[0]["va1"][1]["atom"].feat.clone()
    assert main(["--phase", "rescale", "--out_root", str(new)]) == 0
    assert torch.equal(_read(_merged(new, "val"))[0]["va1"][1]["atom"].feat, before)


def old_graphs_h0(root):
    return _read(_merged(root, "holdouts/H1"))[0]["h0"][1]["atom"].feat


def test_move_rejects_wrong_mol_name(tmp_path, old_tree):
    root, moves = old_tree
    lines = moves.read_text().splitlines()
    name, cur, path, key, dst = lines[1].split("\t")
    lines[1] = "\t".join([name, cur, path, str((int(key) + 1) % 12), dst])
    moves.write_text("\n".join(lines) + "\n")
    with pytest.raises(RuntimeError):
        main(["--phase", "move", "--graph_root", str(root), "--out_root", str(tmp_path / "new"),
              "--moves_tsv", str(moves)])


def test_move_rejects_holdout_destination(tmp_path, old_tree):
    root, moves = old_tree
    with open(moves, "a") as fh:
        fh.write(f"tr5\ttrain\t{_merged(root, 'train')}\t5\tH7\n")
    with pytest.raises(SystemExit):
        main(["--phase", "move", "--graph_root", str(root), "--out_root", str(tmp_path / "new"),
              "--moves_tsv", str(moves)])


def test_rescale_keeps_float32(tmp_path, old_tree):
    """Stored float32 graphs stay float32 after new(old.inverse(x))."""
    root, moves = old_tree
    from qtaim_gen.source.scripts.helpers import realign_graph_splits as rg
    old_f = rg._load_scaler(str(root / "train" / "merged" / "feature_scaler_iterative.pt"), True)
    g = _raw_graph("tr3", dtype=torch.float32)
    out = rg._Rescale(old_f, old_f)([g])[0]
    assert out["atom"].feat.dtype == torch.float32
    assert out["global"].feat.dtype == torch.float32
