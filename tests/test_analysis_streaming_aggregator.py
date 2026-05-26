"""Tests for qtaim_gen.source.analysis.streaming_aggregator (Stream B5a-B5l)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import pytest

from qtaim_gen.source.analysis import stream, stream_to_parquet


# B5a -------------------------------------------------------------------

def test_stream_single_lmdb(tiny_vertical):
    root, keys = tiny_vertical()
    df = stream(
        root,
        ["structure"],
        per_record_fn=lambda k, v: {"n_atoms": len(v["atomic_numbers"])},
        progress=False,
    )
    assert len(df) == len(keys)
    assert list(df["key"]) == keys
    assert "n_atoms" in df.columns
    assert (df["n_atoms"] > 0).all()


# B5b -------------------------------------------------------------------

def test_stream_two_lmdb_inner_join(tiny_vertical):
    root, keys = tiny_vertical()
    df = stream(
        root,
        ["charge", "orca"],
        per_record_fn=lambda k, c, o: {"dipole": o["dipole_magnitude_au"], "n_charges": len(c["hirshfeld"])},
        progress=False,
    )
    assert len(df) == len(keys)
    assert set(df["key"]) == set(keys)
    assert df["dipole"].notna().all()


# B5c -------------------------------------------------------------------

def test_stream_three_lmdb_join(tiny_vertical):
    root, keys = tiny_vertical()
    df = stream(
        root,
        ["structure", "charge", "orca"],
        per_record_fn=lambda k, s, c, o: {"n_atoms": len(s["atomic_numbers"]), "energy": o["energy"]},
        progress=False,
    )
    assert len(df) == len(keys)
    assert {"n_atoms", "energy", "key"}.issubset(df.columns)


# B5d -------------------------------------------------------------------

def test_stream_inner_join_drops_orphans(tiny_vertical):
    root, _ = tiny_vertical(orphan_lmdb={"charge": ["k_orphan"]})
    df = stream(
        root,
        ["charge", "orca"],
        per_record_fn=lambda k, c, o: {"x": o["energy"]},
        progress=False,
    )
    assert "k_orphan" not in set(df["key"])
    assert df.attrs["dropped_keys"] == ["k_orphan"]


# B5e -------------------------------------------------------------------

def test_stream_keys_filter_subset(tiny_vertical):
    root, keys = tiny_vertical()
    df = stream(
        root,
        ["structure"],
        per_record_fn=lambda k, v: {"n_atoms": len(v["atomic_numbers"])},
        keys=[keys[0]],
        progress=False,
    )
    assert len(df) == 1
    assert df.iloc[0]["key"] == keys[0]

    df_empty = stream(
        root,
        ["structure"],
        per_record_fn=lambda k, v: {"n_atoms": len(v["atomic_numbers"])},
        keys=["nope_does_not_exist"],
        progress=False,
    )
    assert len(df_empty) == 0
    assert df_empty.attrs["dropped_keys"] == ["nope_does_not_exist"]


# B5f -------------------------------------------------------------------

def test_stream_keys_filter_missing(tiny_vertical):
    root, keys = tiny_vertical(orphan_lmdb={"charge": ["k_orphan"]})
    df = stream(
        root,
        ["charge", "orca"],
        per_record_fn=lambda k, c, o: {"x": o["energy"]},
        keys=[keys[0], "k_orphan"],
        progress=False,
    )
    assert list(df["key"]) == [keys[0]]
    assert df.attrs["dropped_keys"] == ["k_orphan"]


# B5g -------------------------------------------------------------------

def test_per_record_fn_returns_dict(tiny_vertical):
    root, keys = tiny_vertical()
    df = stream(
        root,
        ["structure"],
        per_record_fn=lambda k, v: {"label": "L"},
        progress=False,
    )
    assert len(df) == len(keys)
    assert (df["label"] == "L").all()


# B5h -------------------------------------------------------------------

def test_per_record_fn_returns_list_of_dicts(tiny_vertical):
    root, keys = tiny_vertical()

    def per_atom(k, v):
        return [{"atom_idx": i, "z": z} for i, z in enumerate(v["atomic_numbers"])]

    df = stream(root, ["structure"], per_record_fn=per_atom, progress=False)
    # Two structures of size 3, one of size 5: expected total 11
    assert len(df) == 11
    assert {"key", "atom_idx", "z"}.issubset(df.columns)
    # Every row has the correct group key
    grouped = df.groupby("key")["atom_idx"].count().to_dict()
    for k in keys:
        assert grouped[k] >= 3


# B5i -------------------------------------------------------------------

def test_per_record_fn_returns_empty_list(tiny_vertical):
    root, _ = tiny_vertical()
    df = stream(root, ["structure"], per_record_fn=lambda k, v: [], progress=False)
    assert len(df) == 0
    assert isinstance(df, pd.DataFrame)


# B5j -------------------------------------------------------------------

def test_stream_to_parquet_roundtrip(tiny_vertical, tmp_path):
    root, keys = tiny_vertical()
    out = tmp_path / "out.parquet"

    def fn(k, v):
        return {"n_atoms": len(v["atomic_numbers"]), "atom_indices": list(range(len(v["atomic_numbers"])))}

    stream_to_parquet(root, ["structure"], fn, out, progress=False)
    assert out.exists()
    df = pd.read_parquet(out)
    assert len(df) == len(keys)
    assert "atom_indices" in df.columns
    # List-valued column survives roundtrip
    assert all(isinstance(v, (list, tuple)) or hasattr(v, "__iter__") for v in df["atom_indices"])
    sample = df.iloc[0]["atom_indices"]
    assert list(sample) == list(range(df.iloc[0]["n_atoms"]))


# B5k -------------------------------------------------------------------

def test_stream_to_parquet_chunked_writes(tiny_vertical, tmp_path, monkeypatch):
    five_keys = [f"k{i}" for i in range(5)]
    root, _ = tiny_vertical(keys=five_keys)
    out = tmp_path / "chunked.parquet"

    call_count = {"n": 0}
    real_write_table = pq.ParquetWriter.write_table

    def counting_write_table(self, table, *args, **kwargs):
        call_count["n"] += 1
        return real_write_table(self, table, *args, **kwargs)

    monkeypatch.setattr(pq.ParquetWriter, "write_table", counting_write_table)

    stream_to_parquet(
        root,
        ["structure"],
        per_record_fn=lambda k, v: {"n_atoms": len(v["atomic_numbers"])},
        output_path=out,
        chunk_size=2,
        progress=False,
    )

    df = pd.read_parquet(out)
    assert len(df) == 5
    # 5 rows / chunk_size 2 -> 3 flushes (2, 2, 1)
    assert call_count["n"] >= 2


# B5l -------------------------------------------------------------------

@pytest.mark.slow
def test_stream_noble_gas_smoke():
    root = Path("data/OMol4M_lmdbs/noble_gas")
    if not (root / "structure.lmdb").exists():
        pytest.skip(f"{root}/structure.lmdb not present; smoke test skipped")
    df = stream(
        root,
        ["structure"],
        per_record_fn=lambda k, v: {"n_atoms": len(v["molecule"]), "spin": v.get("spin")},
        progress=False,
    )
    assert len(df) > 0
    assert (df["n_atoms"] > 0).all()
    assert df["n_atoms"].dtype.kind in {"i", "u"}
