"""Tests for qtaim_gen.source.analysis.bond_agreement (Stream D)."""
from __future__ import annotations

import math
import pickle
import subprocess
import sys
from pathlib import Path

import lmdb
import pandas as pd
import pytest
from pymatgen.core import Molecule

from qtaim_gen.source.analysis.bond_agreement import (
    aggregate,
    aggregate_by_element_pair,
    classify_bond_order,
    classify_geom_bonded,
    classify_qtaim_bonded,
    enumerate_candidate_pairs,
    per_pair_classification,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# H2O geometry: H0=[0,0,0], H1=[0.96,0,0], O2=[0.48,0.83,0]
# H-O distances ~0.96 A  (r_cov H=0.31, O=0.66 -> sum=0.97; pool=1.358, geom=1.261)
# H-H distance   0.96 A  (r_cov H+H=0.62; pool=0.868) -> NOT a candidate pair
_H2O = Molecule(
    ["H", "H", "O"],
    [[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [0.48, 0.83, 0.0]],
)

# CH4: C-H bonds present, H-H far enough to be out of pool
_CH4 = Molecule(
    ["C", "H", "H", "H", "H"],
    [
        [0.0, 0.0, 0.0],
        [0.63, 0.63, 0.63],
        [-0.63, -0.63, 0.63],
        [-0.63, 0.63, -0.63],
        [0.63, -0.63, -0.63],
    ],
)


def _write_lmdb(path: Path, records: dict):
    env = lmdb.open(str(path), subdir=False, map_size=10 * 1024 * 1024, meminit=False)
    with env.begin(write=True) as txn:
        for k, v in records.items():
            txn.put(k.encode("ascii"), pickle.dumps(v, protocol=-1))
        txn.put(b"length", pickle.dumps(len(records), protocol=-1))
    env.sync()
    env.close()


def _make_bond_vertical(tmp_path: Path) -> Path:
    """Synthetic vertical with production-format bond/qtaim/structure keys.

    Two records:
      'h2o_full'    - H2O, both H-O pairs bonded by all schemes
      'h2o_partial' - H2O, H1-O pair has fuzzy_bond = 0.3 (below 0.5 threshold)
                      -> fuzzy_bonded=False while geom_bonded=True (disagreement)
    """
    root = tmp_path / "synth_vertical"
    root.mkdir(parents=True, exist_ok=True)

    structure_recs = {
        "h2o_full": {"molecule": _H2O, "spin": 1, "charge": 0},
        "h2o_partial": {"molecule": _H2O, "spin": 1, "charge": 0},
        "ch4": {"molecule": _CH4, "spin": 1, "charge": 0},
    }
    # qtaim: BCP keys are 0-indexed. H2O: atoms 0(H), 1(H), 2(O).
    # Bonds: H0-O2 = "0_2", H1-O2 = "1_2". No H0-H1 BCP.
    _bcp = {"density_all": 0.28, "lap_e_density": -0.1}
    # CH4: C=0, H=1,2,3,4. BCPs: 0_1, 0_2, 0_3, 0_4.
    qtaim_recs = {
        "h2o_full": {
            "0": _bcp, "1": _bcp, "2": _bcp,
            "0_2": _bcp, "1_2": _bcp,
        },
        "h2o_partial": {
            "0": _bcp, "1": _bcp, "2": _bcp,
            "0_2": _bcp, "1_2": _bcp,
        },
        "ch4": {
            "0": _bcp, "1": _bcp, "2": _bcp, "3": _bcp, "4": _bcp,
            "0_1": _bcp, "0_2": _bcp, "0_3": _bcp, "0_4": _bcp,
        },
    }
    # bond: 1-indexed keys "{i+1}_{elem}_to_{j+1}_{elem}"
    # H2O: H0=atom1_H, H1=atom2_H, O2=atom3_O
    bond_recs = {
        "h2o_full": {
            "mayer_orca": {"1_H_to_3_O": 0.85, "2_H_to_3_O": 0.83},
            "loewdin_orca": {"1_H_to_3_O": 0.78, "2_H_to_3_O": 0.76},
            "fuzzy_bond": {"1_H_to_3_O": 0.72, "2_H_to_3_O": 0.70},
        },
        "h2o_partial": {
            "mayer_orca": {"1_H_to_3_O": 0.85, "2_H_to_3_O": 0.83},
            "loewdin_orca": {"1_H_to_3_O": 0.78, "2_H_to_3_O": 0.76},
            # H1-O2 fuzzy is 0.3 -> below threshold -> fuzzy_bonded=False
            "fuzzy_bond": {"1_H_to_3_O": 0.72, "2_H_to_3_O": 0.30},
        },
        # CH4: C=1_C, H=2_H/3_H/4_H/5_H
        "ch4": {
            "mayer_orca": {
                "1_C_to_2_H": 0.93, "1_C_to_3_H": 0.93,
                "1_C_to_4_H": 0.93, "1_C_to_5_H": 0.93,
            },
            "loewdin_orca": {
                "1_C_to_2_H": 0.88, "1_C_to_3_H": 0.88,
                "1_C_to_4_H": 0.88, "1_C_to_5_H": 0.88,
            },
            "fuzzy_bond": {
                "1_C_to_2_H": 0.80, "1_C_to_3_H": 0.80,
                "1_C_to_4_H": 0.80, "1_C_to_5_H": 0.80,
            },
        },
    }

    _write_lmdb(root / "structure.lmdb", structure_recs)
    _write_lmdb(root / "qtaim.lmdb", qtaim_recs)
    _write_lmdb(root / "bond.lmdb", bond_recs)
    # Other LMDBs required by streaming_aggregator inner-join
    dummy = {k: {} for k in structure_recs}
    for t in ("charge", "fuzzy", "other", "orca", "timings"):
        _write_lmdb(root / f"{t}.lmdb", dummy)
    return root


# ---------------------------------------------------------------------------
# D2: classify_geom_bonded
# ---------------------------------------------------------------------------

def test_classify_geom_bonded_true():
    assert classify_geom_bonded(1.0, 1.0, k=1.3) is True


def test_classify_geom_bonded_false():
    assert classify_geom_bonded(1.4, 1.0, k=1.3) is False


def test_classify_geom_bonded_boundary():
    # exactly at threshold
    assert classify_geom_bonded(1.3, 1.0, k=1.3) is True


# ---------------------------------------------------------------------------
# D3: classify_qtaim_bonded
# ---------------------------------------------------------------------------

def test_classify_qtaim_bonded_present():
    rec = {"0_2": {}, "1_2": {}}
    assert classify_qtaim_bonded(rec, 0, 2) is True
    assert classify_qtaim_bonded(rec, 2, 0) is True  # reversed lookup


def test_classify_qtaim_bonded_absent():
    rec = {"0_2": {}, "1_2": {}}
    assert classify_qtaim_bonded(rec, 0, 1) is False


def test_classify_qtaim_bonded_empty_record():
    assert classify_qtaim_bonded({}, 0, 1) is False


# ---------------------------------------------------------------------------
# D4: classify_bond_order
# ---------------------------------------------------------------------------

def test_classify_bond_order_mayer_bonded():
    rec = {"mayer_orca": {"1_H_to_3_O": 0.85}}
    result = classify_bond_order(rec, "mayer", 0, 2, "H", "O", threshold=0.5)
    assert result is True


def test_classify_bond_order_below_threshold():
    rec = {"fuzzy_bond": {"2_H_to_3_O": 0.30}}
    result = classify_bond_order(rec, "fuzzy", 1, 2, "H", "O", threshold=0.5)
    assert result is False


def test_classify_bond_order_reverse_key():
    # Key stored as j_to_i order
    rec = {"mayer_orca": {"3_O_to_1_H": 0.85}}
    result = classify_bond_order(rec, "mayer", 0, 2, "H", "O", threshold=0.5)
    assert result is True


def test_classify_bond_order_absent_pair():
    rec = {"mayer_orca": {"1_C_to_2_C": 1.5}}
    result = classify_bond_order(rec, "mayer", 0, 2, "H", "O", threshold=0.5)
    assert result is None


def test_classify_bond_order_absent_scheme():
    rec = {"mayer_orca": {"1_H_to_3_O": 0.85}}
    result = classify_bond_order(rec, "loewdin", 0, 2, "H", "O", threshold=0.5)
    assert result is None


# ---------------------------------------------------------------------------
# D1: enumerate_candidate_pairs
# ---------------------------------------------------------------------------

def test_enumerate_candidate_pairs_h2o():
    rec = {"molecule": _H2O}
    pairs = enumerate_candidate_pairs(rec, pool_multiplier=1.4)
    # H0-H1 distance 0.96, r_cov_sum 0.62, pool limit 0.868 -> not included
    # H0-O2 and H1-O2 included
    assert len(pairs) == 2
    idxs = {(p[0], p[1]) for p in pairs}
    assert (0, 2) in idxs
    assert (1, 2) in idxs


def test_enumerate_candidate_pairs_ch4():
    rec = {"molecule": _CH4}
    pairs = enumerate_candidate_pairs(rec, pool_multiplier=1.4)
    # C-H pairs only (4); H-H too far
    assert len(pairs) == 4
    for i, j, d, _ in pairs:
        assert min(i, j) == 0  # all pairs involve C at index 0


def test_enumerate_candidate_pairs_indices_ordered():
    rec = {"molecule": _H2O}
    for i, j, _, _ in enumerate_candidate_pairs(rec):
        assert i < j


# ---------------------------------------------------------------------------
# D5: per_pair_classification
# ---------------------------------------------------------------------------

def test_per_pair_classification_h2o_full():
    structure = {"molecule": _H2O}
    qtaim = {"0": {}, "1": {}, "2": {}, "0_2": {}, "1_2": {}}
    bond = {
        "mayer_orca": {"1_H_to_3_O": 0.85, "2_H_to_3_O": 0.83},
        "loewdin_orca": {"1_H_to_3_O": 0.78, "2_H_to_3_O": 0.76},
        "fuzzy_bond": {"1_H_to_3_O": 0.72, "2_H_to_3_O": 0.70},
    }
    rows = per_pair_classification("k1", structure, qtaim, bond, vertical="v")
    assert len(rows) == 2
    for r in rows:
        assert r["vertical"] == "v"
        assert r["geom_bonded"] is True
        assert r["qtaim_bonded"] is True
        assert r["mayer_bonded"] is True
        assert r["loewdin_bonded"] is True
        assert r["fuzzy_bonded"] is True


def test_per_pair_classification_disagreement():
    structure = {"molecule": _H2O}
    qtaim = {"0": {}, "1": {}, "2": {}, "0_2": {}, "1_2": {}}
    bond = {
        "mayer_orca": {"1_H_to_3_O": 0.85, "2_H_to_3_O": 0.83},
        "loewdin_orca": {"1_H_to_3_O": 0.78, "2_H_to_3_O": 0.76},
        "fuzzy_bond": {"1_H_to_3_O": 0.72, "2_H_to_3_O": 0.30},
    }
    rows = per_pair_classification("k2", structure, qtaim, bond)
    df = pd.DataFrame(rows)
    # The H1-O2 pair (i=1, j=2) has fuzzy_bonded=False, geom_bonded=True
    dis = df[(df["i"] == 1) & (df["j"] == 2)]
    assert len(dis) == 1
    assert not dis.iloc[0]["fuzzy_bonded"]
    assert dis.iloc[0]["geom_bonded"]


def test_per_pair_classification_none_qtaim():
    structure = {"molecule": _H2O}
    rows = per_pair_classification("k3", structure, None, None)
    assert len(rows) == 2
    for r in rows:
        assert r["qtaim_bonded"] is None
        assert r["mayer_bonded"] is None


# ---------------------------------------------------------------------------
# D7: aggregate / aggregate_by_element_pair
# ---------------------------------------------------------------------------

def _make_simple_df():
    rows = [
        {"vertical": "v", "i": 0, "j": 2, "element_i": "H", "element_j": "O",
         "geom_bonded": True, "qtaim_bonded": True,
         "mayer_bonded": True, "loewdin_bonded": True, "fuzzy_bonded": True},
        {"vertical": "v", "i": 1, "j": 2, "element_i": "H", "element_j": "O",
         "geom_bonded": True, "qtaim_bonded": True,
         "mayer_bonded": True, "loewdin_bonded": True, "fuzzy_bonded": False},
    ]
    return pd.DataFrame(rows)


def test_aggregate_basic():
    df = _make_simple_df()
    agg = aggregate(df, "v")
    assert set(agg["scheme"]) == {"qtaim", "mayer", "loewdin", "fuzzy"}
    qtaim_row = agg[agg["scheme"] == "qtaim"].iloc[0]
    assert qtaim_row["tp"] == 2
    assert qtaim_row["fp"] == 0
    fuzzy_row = agg[agg["scheme"] == "fuzzy"].iloc[0]
    # fuzzy: 1 TP (pair 0,2) + 1 FN (pair 1,2 geom=True fuzzy=False)
    assert fuzzy_row["tp"] == 1
    assert fuzzy_row["fn"] == 1
    assert not math.isnan(fuzzy_row["f1"])


def test_aggregate_by_element_pair():
    df = _make_simple_df()
    agg = aggregate_by_element_pair(df, "v")
    assert "H_O" in agg["element_pair"].values
    row = agg[(agg["scheme"] == "mayer") & (agg["element_pair"] == "H_O")].iloc[0]
    assert row["tp"] == 2


def test_aggregate_empty():
    df = pd.DataFrame(columns=["geom_bonded", "qtaim_bonded", "mayer_bonded",
                                "loewdin_bonded", "fuzzy_bonded"])
    agg = aggregate(df, "v")
    assert len(agg) == 0


# ---------------------------------------------------------------------------
# D6 + D10: streaming integration via CLI
# ---------------------------------------------------------------------------

def test_cli_runs_end_to_end(tmp_path):
    root = _make_bond_vertical(tmp_path)
    out = tmp_path / "ba.parquet"
    rc = subprocess.call(
        [sys.executable, "-m",
         "qtaim_gen.source.scripts.analysis_bond_agreement",
         "--root", str(root),
         "--output", str(out),
         "--no-progress"],
        stderr=subprocess.DEVNULL,
    )
    assert rc == 0
    assert out.exists()
    df = pd.read_parquet(out)
    assert "geom_bonded" in df.columns
    assert "qtaim_bonded" in df.columns
    assert "mayer_bonded" in df.columns
    assert "loewdin_bonded" in df.columns
    assert "fuzzy_bonded" in df.columns
    assert "vertical" in df.columns
    assert len(df) > 0
    # agg output produced
    agg_path = out.with_name("ba_agg.parquet")
    assert agg_path.exists()
    agg = pd.read_parquet(agg_path)
    assert set(agg["scheme"]) >= {"mayer", "loewdin", "fuzzy", "qtaim"}


def test_cli_emit_disagreements(tmp_path):
    root = _make_bond_vertical(tmp_path)
    out = tmp_path / "ba.parquet"
    dis = tmp_path / "dis.parquet"
    rc = subprocess.call(
        [sys.executable, "-m",
         "qtaim_gen.source.scripts.analysis_bond_agreement",
         "--root", str(root),
         "--output", str(out),
         "--emit-disagreements", str(dis),
         "--no-progress"],
        stderr=subprocess.DEVNULL,
    )
    assert rc == 0
    assert dis.exists()
    df_dis = pd.read_parquet(dis)
    # h2o_partial has 1 disagreement row (fuzzy_bonded=False, geom_bonded=True)
    assert len(df_dis) >= 1
    assert (df_dis["geom_bonded"] != df_dis["fuzzy_bonded"].astype(bool)).any()
