"""Tests for qtaim_gen.source.analysis.noise_floors (Stream F)."""
from __future__ import annotations

import json
import math
import pickle
import subprocess
import sys
from pathlib import Path

import lmdb
import numpy as np
import pandas as pd
import pytest
from pymatgen.core import Molecule

from qtaim_gen.source.analysis.noise_floors import (
    BOND_SCHEMES,
    CHARGE_SCHEMES,
    QTAIM_DESCRIPTORS,
    aggregate_bond_noise_floor,
    aggregate_charge_noise_floor,
    aggregate_qtaim_redundancy,
    build_noise_floor_table,
    emit_exemplars,
    per_record_bond_pairs,
    per_record_charge_atoms,
    per_record_qtaim_bcps,
)

# ---------------------------------------------------------------------------
# Shared molecules
# ---------------------------------------------------------------------------

_H2O = Molecule(["H", "H", "O"], [[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [0.48, 0.83, 0.0]])
_CH4 = Molecule(
    ["C", "H", "H", "H", "H"],
    [[0.0, 0.0, 0.0], [0.63, 0.63, 0.63], [-0.63, -0.63, 0.63],
     [-0.63, 0.63, -0.63], [0.63, -0.63, -0.63]],
)
_NH3 = Molecule(
    ["N", "H", "H", "H"],
    [[0.0, 0.0, 0.0], [1.01, 0.0, 0.0], [-0.34, 0.96, 0.0], [-0.34, -0.48, 0.83]],
)


def _write_lmdb(path: Path, records: dict):
    env = lmdb.open(str(path), subdir=False, map_size=20 * 1024 * 1024, meminit=False)
    with env.begin(write=True) as txn:
        for k, v in records.items():
            txn.put(k.encode("ascii"), pickle.dumps(v, protocol=-1))
        txn.put(b"length", pickle.dumps(len(records), protocol=-1))
    env.sync()
    env.close()


# ---------------------------------------------------------------------------
# Synthetic vertical fixture
# ---------------------------------------------------------------------------

def _make_nf_vertical(tmp_path: Path) -> Path:
    """Four-record synthetic vertical for Stream F tests.

    Records:
      h2o_a / h2o_b : H2O with slightly different charges across schemes
      ch4_a / ch4_b : CH4 with similar variance
    All four records have structure, charge, bond, and qtaim LMDBs.
    """
    root = tmp_path / "synth_nf"
    root.mkdir(parents=True, exist_ok=True)

    structure_recs = {
        "h2o_a": {"molecule": _H2O, "spin": 1, "charge": 0},
        "h2o_b": {"molecule": _H2O, "spin": 1, "charge": 0},
        "ch4_a": {"molecule": _CH4, "spin": 1, "charge": 0},
        "ch4_b": {"molecule": _CH4, "spin": 1, "charge": 0},
    }

    # Charge records: 4 schemes present (subset of CHARGE_SCHEMES)
    # Atom keys: 1-indexed, element suffix
    def _h2o_charges(base_offset: float) -> dict:
        # H atoms: atoms 1_H, 2_H; O: 3_O
        h1 = 0.30 + base_offset
        h2 = 0.31 + base_offset
        o = -(h1 + h2)
        return {
            "hirshfeld": {"charge": {"1_H": h1, "2_H": h2, "3_O": o}},
            "cm5": {"charge": {"1_H": h1 + 0.05, "2_H": h2 + 0.05, "3_O": o - 0.10}},
            "adch": {"charge": {"1_H": h1 + 0.10, "2_H": h2 + 0.10, "3_O": o - 0.20}},
            "becke": {"charge": {"1_H": h1 + 0.02, "2_H": h2 + 0.02, "3_O": o - 0.04}},
        }

    def _ch4_charges(base_offset: float) -> dict:
        # C: 1_C; H: 2_H, 3_H, 4_H, 5_H
        c = -0.40 - base_offset
        h = 0.10 + base_offset / 4
        return {
            "hirshfeld": {"charge": {"1_C": c, "2_H": h, "3_H": h, "4_H": h, "5_H": h}},
            "cm5": {"charge": {"1_C": c + 0.08, "2_H": h - 0.02, "3_H": h - 0.02,
                               "4_H": h - 0.02, "5_H": h - 0.02}},
            "adch": {"charge": {"1_C": c + 0.15, "2_H": h - 0.04, "3_H": h - 0.04,
                                "4_H": h - 0.04, "5_H": h - 0.04}},
            "becke": {"charge": {"1_C": c + 0.03, "2_H": h - 0.01, "3_H": h - 0.01,
                                 "4_H": h - 0.01, "5_H": h - 0.01}},
        }

    charge_recs = {
        "h2o_a": _h2o_charges(0.00),
        "h2o_b": _h2o_charges(0.02),
        "ch4_a": _ch4_charges(0.00),
        "ch4_b": _ch4_charges(0.01),
    }

    # Bond records: 3 bond-order schemes; 1-indexed pair keys
    # H2O: H0=1_H, H1=2_H, O2=3_O  -> pairs 1_H_to_3_O, 2_H_to_3_O
    # CH4: C=1_C, H=2_H..5_H       -> pairs 1_C_to_2_H .. 1_C_to_5_H
    def _h2o_bond(mayer_base: float) -> dict:
        return {
            "mayer_orca": {"1_H_to_3_O": mayer_base, "2_H_to_3_O": mayer_base - 0.02},
            "loewdin_orca": {"1_H_to_3_O": mayer_base - 0.10, "2_H_to_3_O": mayer_base - 0.12},
            "fuzzy_bond": {"1_H_to_3_O": mayer_base - 0.20, "2_H_to_3_O": mayer_base - 0.22},
        }

    def _ch4_bond(mayer_base: float) -> dict:
        return {
            "mayer_orca": {"1_C_to_2_H": mayer_base, "1_C_to_3_H": mayer_base,
                           "1_C_to_4_H": mayer_base, "1_C_to_5_H": mayer_base},
            "loewdin_orca": {"1_C_to_2_H": mayer_base - 0.05, "1_C_to_3_H": mayer_base - 0.05,
                             "1_C_to_4_H": mayer_base - 0.05, "1_C_to_5_H": mayer_base - 0.05},
            "fuzzy_bond": {"1_C_to_2_H": mayer_base - 0.15, "1_C_to_3_H": mayer_base - 0.15,
                           "1_C_to_4_H": mayer_base - 0.15, "1_C_to_5_H": mayer_base - 0.15},
        }

    bond_recs = {
        "h2o_a": _h2o_bond(0.85),
        "h2o_b": _h2o_bond(0.83),
        "ch4_a": _ch4_bond(0.92),
        "ch4_b": _ch4_bond(0.90),
    }

    # QTAIM BCP records: 0-indexed keys "{i}_{j}"
    # H2O BCPs: 0_2 (H0-O2), 1_2 (H1-O2)
    # CH4 BCPs: 0_1, 0_2, 0_3, 0_4 (C-H bonds)
    def _bcp(rho: float, lap: float, ellip: float, eta: float) -> dict:
        return {
            "density_all": rho,
            "lap_e_density": lap,
            "ellip_e_dens": ellip,
            "eta": eta,
        }

    _atom_bcp = {"density_all": 0.1, "lap_e_density": 0.5}

    qtaim_recs = {
        "h2o_a": {
            "0": _atom_bcp, "1": _atom_bcp, "2": _atom_bcp,
            "0_2": _bcp(0.28, -1.10, 0.02, 0.85),
            "1_2": _bcp(0.27, -1.05, 0.03, 0.82),
        },
        "h2o_b": {
            "0": _atom_bcp, "1": _atom_bcp, "2": _atom_bcp,
            "0_2": _bcp(0.29, -1.15, 0.01, 0.88),
            "1_2": _bcp(0.28, -1.08, 0.02, 0.84),
        },
        "ch4_a": {
            "0": _atom_bcp, "1": _atom_bcp, "2": _atom_bcp, "3": _atom_bcp, "4": _atom_bcp,
            "0_1": _bcp(0.24, -0.75, 0.00, 0.90),
            "0_2": _bcp(0.24, -0.76, 0.00, 0.91),
            "0_3": _bcp(0.23, -0.74, 0.00, 0.89),
            "0_4": _bcp(0.24, -0.75, 0.00, 0.90),
        },
        "ch4_b": {
            "0": _atom_bcp, "1": _atom_bcp, "2": _atom_bcp, "3": _atom_bcp, "4": _atom_bcp,
            "0_1": _bcp(0.25, -0.78, 0.00, 0.92),
            "0_2": _bcp(0.25, -0.77, 0.00, 0.91),
            "0_3": _bcp(0.24, -0.76, 0.00, 0.90),
            "0_4": _bcp(0.25, -0.78, 0.00, 0.92),
        },
    }

    _write_lmdb(root / "structure.lmdb", structure_recs)
    _write_lmdb(root / "charge.lmdb", charge_recs)
    _write_lmdb(root / "bond.lmdb", bond_recs)
    _write_lmdb(root / "qtaim.lmdb", qtaim_recs)
    # Dummy LMDBs so streaming_aggregator does not fail on missing files
    dummy = {k: {} for k in structure_recs}
    for t in ("fuzzy", "other", "orca", "timings"):
        _write_lmdb(root / f"{t}.lmdb", dummy)
    return root


# ---------------------------------------------------------------------------
# Unit tests: parsing helpers
# ---------------------------------------------------------------------------

def test_per_record_charge_atoms_basic():
    rec = {
        "hirshfeld": {"charge": {"1_H": 0.30, "2_H": 0.31, "3_O": -0.61}},
        "cm5": {"charge": {"1_H": 0.35, "2_H": 0.36, "3_O": -0.71}},
    }
    rows = per_record_charge_atoms("k1", rec, vertical="v")
    assert len(rows) == 3
    elems = {r["element"] for r in rows}
    assert elems == {"H", "O"}
    # Charges populated
    for r in rows:
        if r["element"] == "H":
            assert not math.isnan(r["charge_hirshfeld"])
            assert not math.isnan(r["charge_cm5"])
            # Absent schemes are NaN
            assert math.isnan(r.get("charge_adch", math.nan))


def test_per_record_charge_atoms_empty():
    assert per_record_charge_atoms("k", None) == []
    assert per_record_charge_atoms("k", {}) == []


def test_per_record_charge_atoms_element_mismatch():
    # Two schemes disagree on element at same index -> skip record
    rec = {
        "hirshfeld": {"charge": {"1_H": 0.30}},
        "cm5": {"charge": {"1_N": 0.35}},  # different element at index 0
    }
    rows = per_record_charge_atoms("k", rec)
    assert rows == []


def test_per_record_bond_pairs_basic():
    rec = {
        "mayer_orca": {"1_H_to_3_O": 0.85, "2_H_to_3_O": 0.83},
        "loewdin_orca": {"1_H_to_3_O": 0.75, "2_H_to_3_O": 0.73},
        "fuzzy_bond": {"1_H_to_3_O": 0.65, "2_H_to_3_O": 0.63},
    }
    rows = per_record_bond_pairs("k", rec, vertical="v")
    assert len(rows) == 2
    for r in rows:
        assert not math.isnan(r["bo_mayer_orca"])
        assert not math.isnan(r["bo_loewdin_orca"])
        assert not math.isnan(r["bo_fuzzy_bond"])
        assert r["element_pair"] in {"H_O", "O_H"}  # sorted
    # element_pair should always be sorted
    for r in rows:
        parts = r["element_pair"].split("_")
        assert parts == sorted(parts)


def test_per_record_bond_pairs_empty():
    assert per_record_bond_pairs("k", None) == []
    assert per_record_bond_pairs("k", {}) == []


def test_per_record_bond_pairs_canonical_order():
    # Key stored with higher index first; should still produce i < j
    rec = {"mayer_orca": {"3_O_to_1_H": 0.85}}
    rows = per_record_bond_pairs("k", rec)
    assert len(rows) == 1
    assert rows[0]["i"] < rows[0]["j"]


def test_per_record_qtaim_bcps_basic():
    structure = {"molecule": _H2O}
    qtaim = {
        "0": {"density_all": 0.1},  # atom CP, no "_" -> skipped
        "0_2": {"density_all": 0.28, "lap_e_density": -1.1, "ellip_e_dens": 0.02, "eta": 0.85},
        "1_2": {"density_all": 0.27, "lap_e_density": -1.05, "ellip_e_dens": 0.03, "eta": 0.82},
    }
    rows = per_record_qtaim_bcps("k", qtaim, structure, vertical="v")
    assert len(rows) == 2
    for r in rows:
        assert r["element_i"] in {"H", "O"}
        assert r["element_j"] in {"H", "O"}
        assert not math.isnan(r["density_all"])
        assert not math.isnan(r["lap_e_density"])
        # delocalization_index absent -> NaN
        assert math.isnan(r.get("delocalization_index", math.nan))


def test_per_record_qtaim_bcps_no_structure():
    qtaim = {"0_1": {"density_all": 0.25, "lap_e_density": -0.80}}
    rows = per_record_qtaim_bcps("k", qtaim, None)
    assert len(rows) == 1
    assert rows[0]["element_i"] == "?"
    assert rows[0]["element_j"] == "?"


# ---------------------------------------------------------------------------
# Unit tests: aggregation
# ---------------------------------------------------------------------------

def _make_charge_df() -> pd.DataFrame:
    rows = [
        per_record_charge_atoms("h2o_a", {
            "hirshfeld": {"charge": {"1_H": 0.30, "2_H": 0.31, "3_O": -0.61}},
            "cm5": {"charge": {"1_H": 0.35, "2_H": 0.36, "3_O": -0.71}},
            "adch": {"charge": {"1_H": 0.40, "2_H": 0.41, "3_O": -0.81}},
        }, vertical="v"),
        per_record_charge_atoms("h2o_b", {
            "hirshfeld": {"charge": {"1_H": 0.32, "2_H": 0.33, "3_O": -0.65}},
            "cm5": {"charge": {"1_H": 0.37, "2_H": 0.38, "3_O": -0.75}},
            "adch": {"charge": {"1_H": 0.42, "2_H": 0.43, "3_O": -0.85}},
        }, vertical="v"),
    ]
    all_rows = [r for sublist in rows for r in sublist]
    return pd.DataFrame(all_rows)


def test_aggregate_charge_noise_floor_nonzero():
    df = _make_charge_df()
    nf = aggregate_charge_noise_floor(df, "v", topk=10)
    assert not nf.empty
    # Check columns
    for col in ["vertical", "analysis", "descriptor", "mar", "iqr", "n_obs", "top_keys"]:
        assert col in nf.columns
    # All analysis values should be "charge"
    assert (nf["analysis"] == "charge").all()
    # MAR should be positive for hirshfeld_vs_cm5
    h_cm5 = nf[nf["descriptor"] == "hirshfeld_vs_cm5"]
    assert len(h_cm5) > 0
    global_row = h_cm5[h_cm5["element"].isna()]
    assert len(global_row) == 1
    assert global_row.iloc[0]["mar"] > 0
    assert global_row.iloc[0]["n_obs"] > 0
    # top_keys is valid JSON list
    tk = json.loads(global_row.iloc[0]["top_keys"])
    assert isinstance(tk, list)
    assert len(tk) > 0


def test_aggregate_charge_noise_floor_per_element():
    df = _make_charge_df()
    nf = aggregate_charge_noise_floor(df, "v")
    elem_rows = nf[nf["element"].notna()]
    assert len(elem_rows) > 0
    assert set(elem_rows["element"].unique()) >= {"H", "O"}


def test_aggregate_bond_noise_floor_nonzero():
    rec = {
        "mayer_orca": {"1_H_to_3_O": 0.85, "2_H_to_3_O": 0.83},
        "loewdin_orca": {"1_H_to_3_O": 0.75, "2_H_to_3_O": 0.73},
        "fuzzy_bond": {"1_H_to_3_O": 0.65, "2_H_to_3_O": 0.63},
    }
    rows_a = per_record_bond_pairs("ka", rec, vertical="v")
    rows_b = per_record_bond_pairs("kb", rec, vertical="v")
    df = pd.DataFrame(rows_a + rows_b)
    nf = aggregate_bond_noise_floor(df, "v", topk=10)
    assert not nf.empty
    assert (nf["analysis"] == "bond_order").all()
    # mayer_orca vs loewdin_orca should have MAR ~ 0.10
    m_l = nf[(nf["descriptor"] == "mayer_orca_vs_loewdin_orca") & nf["element_pair"].notna()]
    assert len(m_l) > 0
    assert m_l.iloc[0]["mar"] == pytest.approx(0.10, abs=1e-3)


def test_aggregate_qtaim_redundancy_pearson():
    # density_all and lap_e_density should be strongly correlated in synthetic data
    rows = []
    for i, (rho, lap) in enumerate([(0.28, -1.10), (0.29, -1.15), (0.27, -1.05),
                                    (0.24, -0.75), (0.25, -0.78), (0.23, -0.74)]):
        rows.append({
            "key": f"k{i}", "vertical": "v",
            "i": 0, "j": 1, "element_i": "C", "element_j": "H",
            "element_pair": "C_H",
            "density_all": rho, "lap_e_density": lap,
            "ellip_e_dens": 0.01, "eta": 0.90, "delocalization_index": math.nan,
        })
    df = pd.DataFrame(rows)
    redund = aggregate_qtaim_redundancy(df, "v")
    assert not redund.empty
    # density_all vs lap_e_density should have non-NaN Pearson r
    row = redund[redund["descriptor"] == "density_all_vs_lap_e_density"]
    assert len(row) > 0
    global_r = row[row["element_pair"].isna()].iloc[0]["pearson_r"]
    assert not math.isnan(global_r)
    # rho and Laplacian tend to correlate
    assert abs(global_r) > 0.5


def test_build_noise_floor_table_columns():
    df = _make_charge_df()
    charge_nf = aggregate_charge_noise_floor(df, "v")
    bond_nf = pd.DataFrame(columns=["vertical", "analysis", "descriptor", "element",
                                     "element_pair", "mar", "iqr", "n_obs",
                                     "pearson_r", "top_keys"])
    qtaim_redund = pd.DataFrame(columns=["vertical", "analysis", "descriptor", "element",
                                          "element_pair", "mar", "iqr", "n_obs",
                                          "pearson_r", "top_keys"])
    table = build_noise_floor_table(charge_nf, bond_nf, qtaim_redund)
    assert not table.empty
    for col in ["vertical", "analysis", "descriptor", "mar", "n_obs", "pearson_r", "top_keys"]:
        assert col in table.columns


def test_emit_exemplars_nonempty():
    df = _make_charge_df()
    # Bond df
    bond_rows = per_record_bond_pairs("ka", {
        "mayer_orca": {"1_H_to_3_O": 0.85, "2_H_to_3_O": 0.83},
        "loewdin_orca": {"1_H_to_3_O": 0.75, "2_H_to_3_O": 0.73},
        "fuzzy_bond": {"1_H_to_3_O": 0.65, "2_H_to_3_O": 0.63},
    }, vertical="v")
    bond_df = pd.DataFrame(bond_rows)
    ex = emit_exemplars(df, bond_df, topk=5, vertical="v")
    assert not ex.empty
    # Required columns
    for col in ["vertical", "key", "atom_or_pair", "element_or_pair",
                "descriptor", "schemes_compared", "residual"]:
        assert col in ex.columns
    # All residuals must be positive
    assert (ex["residual"] >= 0).all()
    # Descriptors should include charge and/or bond_order
    assert set(ex["descriptor"].unique()) <= {"charge", "bond_order"}


def test_emit_exemplars_topk_limit():
    # With topk=1, at most 1 exemplar per (element_or_pair, descriptor)
    df = _make_charge_df()
    ex = emit_exemplars(df, pd.DataFrame(), topk=1, vertical="v")
    for (elem, desc), grp in ex.groupby(["element_or_pair", "descriptor"]):
        assert len(grp) <= 1


# ---------------------------------------------------------------------------
# Integration: CLI end-to-end
# ---------------------------------------------------------------------------

def test_cli_single_vertical(tmp_path):
    root = _make_nf_vertical(tmp_path)
    out = tmp_path / "nf.parquet"
    rc = subprocess.call(
        [sys.executable, "-m", "qtaim_gen.source.scripts.analysis_noise_floors",
         "--root", str(root),
         "--output", str(out),
         "--topk", "5",
         "--no-progress"],
        stderr=subprocess.DEVNULL,
    )
    assert rc == 0

    # Noise-floor table
    assert out.exists()
    nf = pd.read_parquet(out)
    assert not nf.empty
    assert "analysis" in nf.columns
    assert set(nf["analysis"].unique()) >= {"charge", "bond_order"}
    # Charge rows should have finite MAR for H and O
    charge_elem = nf[(nf["analysis"] == "charge") & nf["element"].notna()]
    assert set(charge_elem["element"].unique()) >= {"H", "O"}
    assert charge_elem["mar"].notna().all()

    # Exemplars
    exemplars_path = tmp_path / "nf_exemplars.parquet"
    assert exemplars_path.exists()
    ex = pd.read_parquet(exemplars_path)
    assert not ex.empty
    assert (ex["residual"] > 0).any()

    # Intermediate charge atoms: stem is "nf" (from nf.parquet), so "nf_charge_atoms.parquet"
    charge_atoms_path = tmp_path / "nf_charge_atoms.parquet"
    assert charge_atoms_path.exists()
    # Exemplars path
    assert (tmp_path / "nf_exemplars.parquet").exists()


def test_cli_b1_nonzero_residuals_multi_atom(tmp_path):
    """B1: multi-atom records produce non-zero pairwise residuals."""
    root = _make_nf_vertical(tmp_path)
    out = tmp_path / "nf.parquet"
    subprocess.call(
        [sys.executable, "-m", "qtaim_gen.source.scripts.analysis_noise_floors",
         "--root", str(root), "--output", str(out),
         "--topk", "5", "--no-progress"],
        stderr=subprocess.DEVNULL,
    )
    nf = pd.read_parquet(out)
    charge_global = nf[
        (nf["analysis"] == "charge") & nf["element"].isna() & nf["element_pair"].isna()
    ]
    # All global charge rows should have MAR > 0 (schemes differ by construction)
    assert (charge_global["mar"] > 0).all()


def test_cli_b5_exemplar_emitted(tmp_path):
    """B5: at least one exemplar emitted for charge and bond_order."""
    root = _make_nf_vertical(tmp_path)
    out = tmp_path / "nf.parquet"
    subprocess.call(
        [sys.executable, "-m", "qtaim_gen.source.scripts.analysis_noise_floors",
         "--root", str(root), "--output", str(out),
         "--topk", "5", "--no-progress"],
        stderr=subprocess.DEVNULL,
    )
    ex = pd.read_parquet(tmp_path / "nf_exemplars.parquet")
    assert "charge" in ex["descriptor"].values
    assert "bond_order" in ex["descriptor"].values
