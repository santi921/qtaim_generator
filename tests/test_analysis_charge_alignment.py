"""Tests for atom-level charge agreement analysis."""

from __future__ import annotations

import math
import pickle
from pathlib import Path

import lmdb
import numpy as np
import pandas as pd
import pytest
from scipy.stats import pearsonr

from qtaim_gen.source.analysis.charge_alignment import (
    SCHEMES,
    _parse_atom_key,
    aggregate_per_vertical,
    per_record_atoms,
    run_vertical,
)


def _write_lmdb(path: Path, records: dict) -> None:
    env = lmdb.open(str(path), subdir=False, map_size=10 * 1024 * 1024, meminit=False)
    with env.begin(write=True) as txn:
        for k, v in records.items():
            txn.put(k.encode("ascii"), pickle.dumps(v, protocol=-1))
        txn.put(b"length", pickle.dumps(len(records), protocol=-1))
    env.sync()
    env.close()


def _build_charge_only_fixture(tmp_path: Path) -> Path:
    """Three-record fixture with all 7 schemes on the complete record."""
    root = tmp_path / "vertical"
    root.mkdir()

    def _scheme(charge_dict, dipole=None):
        out = {"charge": charge_dict}
        if dipole is not None:
            out["dipole"] = dipole
        return out

    records = {
        # Full coverage: 3 atoms, all 7 schemes.
        "k_full": {
            s: _scheme({"1_C": -0.20 + i * 0.01, "2_H": 0.10, "3_O": 0.10 - i * 0.01})
            for i, s in enumerate(SCHEMES)
        },
        # Partial: 2 atoms, only 4 multiwfn schemes (no ORCA-merged).
        "k_partial": {
            "hirshfeld": _scheme({"1_O": -0.30, "2_H": 0.30}),
            "cm5":       _scheme({"1_O": -0.32, "2_H": 0.32}),
            "adch":      _scheme({"1_O": -0.31, "2_H": 0.31}),
            "becke":     _scheme({"1_O": -0.29, "2_H": 0.29}),
        },
        # Inconsistent: hirshfeld says atom 1 is C; cm5 says it is O. Should skip.
        "k_inconsistent": {
            "hirshfeld": _scheme({"1_C": -0.10, "2_H": 0.10}),
            "cm5":       _scheme({"1_O": -0.12, "2_H": 0.12}),
        },
    }
    _write_lmdb(root / "charge.lmdb", records)
    return root


def test_parse_atom_key():
    assert _parse_atom_key("1_C") == (0, "C")
    assert _parse_atom_key("12_Ag") == (11, "Ag")
    assert _parse_atom_key("3_") is None
    assert _parse_atom_key("foo") is None
    assert _parse_atom_key(None) is None


def test_per_record_emits_one_row_per_atom():
    charge = {
        "hirshfeld": {"charge": {"1_C": -0.20, "2_H": 0.10, "3_O": 0.10}},
        "cm5":       {"charge": {"1_C": -0.22, "2_H": 0.11, "3_O": 0.11}},
    }
    rows = per_record_atoms("k", charge, vertical="v")
    assert len(rows) == 3
    assert [r["element"] for r in rows] == ["C", "H", "O"]
    assert rows[0]["charge_hirshfeld"] == pytest.approx(-0.20)
    assert rows[0]["charge_cm5"] == pytest.approx(-0.22)
    # Schemes not present -> NaN
    assert math.isnan(rows[0]["charge_adch"])


def test_per_record_skips_inconsistent_element_assignment():
    charge = {
        "hirshfeld": {"charge": {"1_C": -0.10, "2_H": 0.10}},
        "cm5":       {"charge": {"1_O": -0.12, "2_H": 0.12}},  # idx 0 says O instead of C
    }
    rows = per_record_atoms("k", charge, vertical="v")
    assert rows == []  # whole record skipped, not silently merged


def test_per_record_handles_missing_record():
    assert per_record_atoms("k", None, vertical="v") == []
    assert per_record_atoms("k", {}, vertical="v") == []


def test_run_vertical_writes_both_parquets(tmp_path):
    root = _build_charge_only_fixture(tmp_path)
    out_dir = tmp_path / "out"
    atoms_path, sum_path = run_vertical(root, out_dir, vertical_name="v", progress=False)

    df_atoms = pd.read_parquet(atoms_path)
    # k_full = 3 atoms, k_partial = 2 atoms, k_inconsistent = skipped.
    assert len(df_atoms) == 5
    assert set(df_atoms["key"].unique()) == {"k_full", "k_partial"}
    for scheme in SCHEMES:
        assert f"charge_{scheme}" in df_atoms.columns
    # k_partial has no ORCA-merged charges -> NaN for those rows.
    partial = df_atoms[df_atoms["key"] == "k_partial"]
    assert partial["charge_mulliken_orca"].isna().all()
    assert partial["charge_hirshfeld"].notna().all()


def test_aggregate_pair_pearson_matches_scipy(tmp_path):
    root = _build_charge_only_fixture(tmp_path)
    out_dir = tmp_path / "out"
    atoms_path, sum_path = run_vertical(root, out_dir, vertical_name="v", progress=False)
    df = pd.read_parquet(atoms_path)
    summary = pd.read_parquet(sum_path)

    pair_rows = summary[summary.kind == "pair"].set_index(["scheme_a", "scheme_b"])
    a, b = "hirshfeld", "cm5"
    qa = df[f"charge_{a}"].to_numpy()
    qb = df[f"charge_{b}"].to_numpy()
    mask = np.isfinite(qa) & np.isfinite(qb)
    expected_n = int(mask.sum())
    expected_r, _ = pearsonr(qa[mask], qb[mask])

    row = pair_rows.loc[(a, b)]
    assert int(row["n"]) == expected_n
    assert float(row["pearson_vertical"]) == pytest.approx(expected_r, abs=1e-10)


def test_aggregate_per_element_rows_present(tmp_path):
    root = _build_charge_only_fixture(tmp_path)
    out_dir = tmp_path / "out"
    _, sum_path = run_vertical(root, out_dir, vertical_name="v", progress=False)
    summary = pd.read_parquet(sum_path)

    # Per-element scheme rows for hirshfeld
    e_rows = summary[(summary.kind == "scheme_element") & (summary.scheme_a == "hirshfeld")]
    elements = set(e_rows["element"].tolist())
    assert "C" in elements and "H" in elements and "O" in elements

    # Per-element pair rows for at least one pair
    pe_rows = summary[summary.kind == "pair_element"]
    assert len(pe_rows) > 0


# ---- Real-vertical smoke ----

_REAL = Path("/home/santiagovargas/dev/qtaim_generator/data/OMol4M_lmdbs/5A_elytes")


@pytest.mark.slow
def test_real_vertical_smoke(tmp_path):
    if not _REAL.exists():
        pytest.skip(f"{_REAL} not on disk")
    out_dir = tmp_path / "out"
    atoms_path, sum_path = run_vertical(_REAL, out_dir, progress=False)
    df = pd.read_parquet(atoms_path)
    assert len(df) > 0
    # Multiwfn level-0 schemes should have near-universal coverage on real data.
    for s in ("hirshfeld", "cm5", "adch", "becke"):
        cov = df[f"charge_{s}"].notna().mean()
        assert cov > 0.95, f"{s} coverage only {cov:.3f}"
