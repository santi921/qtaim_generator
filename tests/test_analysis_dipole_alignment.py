"""Tests for Stream E2 (pairwise dipole agreement).

CM5 is excluded from METHODS: Multiwfn prints the Hirshfeld promolecular dipole
in the CM5 section of the combined output, making cm5_dipole == hirshfeld_dipole.

Plan: docs/plans/2026-05-05-analysis-implementation-plan_E2_charge_dipole_comprehensive.md
"""

from __future__ import annotations

import math
import os
import pickle
from itertools import combinations
from pathlib import Path

import lmdb
import numpy as np
import pandas as pd
import pytest
from scipy.stats import pearsonr

from qtaim_gen.source.analysis.dipole_alignment import (
    METHODS,
    aggregate_per_vertical,
    cos_sim,
    extract_method_dipoles,
    per_record,
    run_vertical,
)


# Synthetic three-record fixture.
#
# k_full:    all 4 methods (adch/becke/hirshfeld multiwfn + ORCA); element_set = "C_H_O".
# k_partial: adch/hirshfeld with dipoles, becke without dipole field; ORCA present.
# k_orphan:  in orca.lmdb only; charge.lmdb has no entry. Inner join must drop it.
def _build_fixture(tmp_path: Path) -> Path:
    root = tmp_path / "vertical"
    root.mkdir()

    charge_records = {
        "k_full": {
            "hirshfeld": {"charge": {"1_C": -0.10, "2_H": 0.05, "3_O": 0.05},
                          "dipole": {"mag": 0.50, "xyz": [0.30, 0.40, 0.00]}},
            "becke":     {"charge": {"1_C": -0.09, "2_H": 0.04, "3_O": 0.05},
                          "dipole": {"mag": 0.48, "xyz": [0.28, 0.38, 0.00]}},
            "adch":      {"charge": {"1_C": -0.11, "2_H": 0.05, "3_O": 0.06},
                          "dipole": {"mag": 0.52, "xyz": [0.31, 0.41, 0.00]}},
        },
        "k_partial": {
            "hirshfeld": {"charge": {"1_C": -0.10, "2_H": 0.10},
                          "dipole": {"mag": 1.00, "xyz": [1.00, 0.00, 0.00]}},
            "adch":      {"charge": {"1_C": -0.10, "2_H": 0.10},
                          "dipole": {"mag": 1.05, "xyz": [1.05, 0.00, 0.00]}},
            "becke":     {"charge": {"1_C": -0.10, "2_H": 0.10}},
        },
    }
    orca_records = {
        "k_full":    {"dipole_au": [0.31, 0.42, 0.01], "dipole_magnitude_au": 0.523},
        "k_partial": {"dipole_au": [1.02, 0.0, 0.0],   "dipole_magnitude_au": 1.02},
        "k_orphan":  {"dipole_au": [0.0, 0.0, 1.0],    "dipole_magnitude_au": 1.0},
    }

    _write_lmdb(root / "charge.lmdb", charge_records)
    _write_lmdb(root / "orca.lmdb", orca_records)
    return root


def _write_lmdb(path: Path, records: dict) -> None:
    env = lmdb.open(str(path), subdir=False, map_size=10 * 1024 * 1024, meminit=False)
    with env.begin(write=True) as txn:
        for k, v in records.items():
            txn.put(k.encode("ascii"), pickle.dumps(v, protocol=-1))
        txn.put(b"length", pickle.dumps(len(records), protocol=-1))
    env.sync()
    env.close()


def test_extract_four_methods_complete_record():
    charge = {
        "hirshfeld": {"charge": {}, "dipole": {"mag": 1.0, "xyz": [1.0, 0.0, 0.0]}},
        "becke":     {"charge": {}, "dipole": {"mag": 3.0, "xyz": [0.0, 0.0, 3.0]}},
        "adch":      {"charge": {}, "dipole": {"mag": 4.0, "xyz": [4.0, 0.0, 0.0]}},
    }
    orca = {"dipole_au": [0.0, 0.0, 6.0], "dipole_magnitude_au": 6.0}
    out = extract_method_dipoles(charge, orca)
    assert set(out) == set(METHODS)
    assert out["hirshfeld_multiwfn"] == (1.0, [1.0, 0.0, 0.0])
    assert out["scf_orca"] == (6.0, [0.0, 0.0, 6.0])


def test_extract_returns_nan_when_method_missing():
    charge = {"hirshfeld": {"charge": {}, "dipole": {"mag": 1.0, "xyz": [1.0, 0.0, 0.0]}}}
    orca = {}
    out = extract_method_dipoles(charge, orca)
    mag, xyz = out["adch_multiwfn"]
    assert math.isnan(mag)
    assert all(math.isnan(c) for c in xyz)
    mag, xyz = out["scf_orca"]
    assert math.isnan(mag)
    assert all(math.isnan(c) for c in xyz)


def test_per_record_partial_nan_row_emitted(tmp_path):
    root = _build_fixture(tmp_path)
    with lmdb.open(str(root / "charge.lmdb"), subdir=False, readonly=True, lock=False) as e:
        with e.begin() as txn:
            charge = pickle.loads(txn.get(b"k_partial"))
    with lmdb.open(str(root / "orca.lmdb"), subdir=False, readonly=True, lock=False) as e:
        with e.begin() as txn:
            orca = pickle.loads(txn.get(b"k_partial"))
    row = per_record("k_partial", charge, orca, vertical="vertical")
    # becke has no dipole field on this record -> NaN col.
    assert math.isnan(row["dipole_mag_becke_multiwfn"])
    # hirshfeld present.
    assert row["dipole_mag_hirshfeld_multiwfn"] == pytest.approx(1.00)
    # ORCA present.
    assert row["dipole_mag_scf_orca"] == pytest.approx(1.02)


def test_cos_sim_zero_vector_returns_nan():
    assert math.isnan(cos_sim([0.0, 0.0, 0.0], [1.0, 0.0, 0.0]))
    assert math.isnan(cos_sim([1.0, 0.0, 0.0], [0.0, 0.0, 0.0]))
    # Sanity: orthogonal -> 0; parallel -> 1.
    assert cos_sim([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]) == pytest.approx(0.0)
    assert cos_sim([1.0, 2.0, 3.0], [2.0, 4.0, 6.0]) == pytest.approx(1.0)


def test_cos_sim_nan_input_returns_nan():
    assert math.isnan(cos_sim([math.nan, 0.0, 0.0], [1.0, 0.0, 0.0]))


def test_run_vertical_inner_join_drops_orphan(tmp_path):
    root = _build_fixture(tmp_path)
    out_dir = tmp_path / "out"
    rec_path, sum_path = run_vertical(root, out_dir, vertical_name="vertical", progress=False)

    df = pd.read_parquet(rec_path)
    keys = set(df["key"].tolist())
    # k_orphan only in orca.lmdb -> dropped.
    assert keys == {"k_full", "k_partial"}
    # Wide schema present.
    for method in METHODS:
        assert f"dipole_mag_{method}" in df.columns
        for axis in ("x", "y", "z"):
            assert f"dipole_{axis}_{method}" in df.columns


def test_aggregate_method_n_excludes_nan(tmp_path):
    root = _build_fixture(tmp_path)
    out_dir = tmp_path / "out"
    rec_path, sum_path = run_vertical(root, out_dir, vertical_name="v", progress=False)
    summary = pd.read_parquet(sum_path)

    method_rows = summary[summary["kind"] == "method"].set_index("method_a")
    # becke has NaN on k_partial -> n = 1; adch/hirshfeld/scf_orca = 2.
    assert int(method_rows.loc["becke_multiwfn", "n"]) == 1
    assert int(method_rows.loc["hirshfeld_multiwfn", "n"]) == 2
    assert int(method_rows.loc["adch_multiwfn", "n"]) == 2
    assert int(method_rows.loc["scf_orca", "n"]) == 2


def test_aggregate_pair_pearson_matches_scipy(tmp_path):
    """Closed-form Pearson r from sufficient stats matches scipy.stats.pearsonr."""
    root = _build_fixture(tmp_path)
    out_dir = tmp_path / "out"
    rec_path, sum_path = run_vertical(root, out_dir, vertical_name="v", progress=False)
    df = pd.read_parquet(rec_path)
    summary = pd.read_parquet(sum_path)

    pair_rows = summary[summary["kind"] == "pair"].set_index(["method_a", "method_b"])

    # Pick a pair both records share: hirshfeld_multiwfn and scf_orca.
    a, b = "hirshfeld_multiwfn", "scf_orca"
    ma = df[f"dipole_mag_{a}"].to_numpy()
    mb = df[f"dipole_mag_{b}"].to_numpy()
    mask = np.isfinite(ma) & np.isfinite(mb)
    expected_n = int(mask.sum())
    expected_r, _ = pearsonr(ma[mask], mb[mask])

    row = pair_rows.loc[(a, b)]
    assert int(row["n"]) == expected_n
    assert float(row["pearson_mag_vertical"]) == pytest.approx(expected_r, abs=1e-10)
    # Sufficient stats are also exact.
    assert float(row["sum_a"]) == pytest.approx(float(ma[mask].sum()))
    assert float(row["sum_ab"]) == pytest.approx(float((ma[mask] * mb[mask]).sum()))


# ---- Real-vertical sign-sanity smoke test ----

_REAL_VERTICAL = Path("/home/santiagovargas/dev/qtaim_generator/data/OMol4M_lmdbs/5A_elytes")


@pytest.mark.slow
def test_real_vertical_sign_sanity(tmp_path):
    """On a real vertical, find a record where Hirshfeld and ORCA SCF dipole
    both have finite vectors with non-trivial magnitude, and assert that their
    dot product is positive.

    Locks the convention that Multiwfn's reported dipole_xyz and ORCA's
    dipole_au use the same sign direction (charge displacement, not its
    inverse). If this ever fails, downstream cosine similarities are flipped
    and need a sign-fix pass.
    """
    if not _REAL_VERTICAL.exists():
        pytest.skip(f"{_REAL_VERTICAL} not on disk")

    out_dir = tmp_path / "out"
    rec_path, sum_path = run_vertical(_REAL_VERTICAL, out_dir, progress=False)
    df = pd.read_parquet(rec_path)
    assert len(df) > 0

    # Restrict to records where both methods have a non-trivial dipole.
    mask = (
        df["dipole_mag_hirshfeld_multiwfn"].notna()
        & df["dipole_mag_scf_orca"].notna()
        & (df["dipole_mag_hirshfeld_multiwfn"] > 0.5)
        & (df["dipole_mag_scf_orca"] > 0.5)
    )
    polar = df[mask].head(20)
    if len(polar) == 0:
        pytest.skip("no polar records with both Hirshfeld + ORCA dipoles in vertical")

    hirsh = polar[["dipole_x_hirshfeld_multiwfn", "dipole_y_hirshfeld_multiwfn", "dipole_z_hirshfeld_multiwfn"]].to_numpy()
    orca = polar[["dipole_x_scf_orca", "dipole_y_scf_orca", "dipole_z_scf_orca"]].to_numpy()
    dot = (hirsh * orca).sum(axis=1)
    # On polar molecules the sign convention should agree on at least the
    # majority. If half flip, that is a hard signal of convention drift.
    n_positive = int((dot > 0).sum())
    assert n_positive >= len(polar) * 0.8, (
        f"sign disagreement: {n_positive}/{len(polar)} positive dot products. "
        "Convention may be flipped."
    )
