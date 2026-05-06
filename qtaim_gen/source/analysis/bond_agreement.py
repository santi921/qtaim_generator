"""Stream D: Bond agreement.

Cross-vertical bond-classification analysis. For each candidate atom pair
within pool_multiplier * sum(r_cov), classify bonded/not by four schemes
(QTAIM BCP, Mayer, Loewdin, fuzzy) and measure agreement against a
distance-grounded geometric reference.

Schemes present in local LMDBs:
  - mayer   -> bond.lmdb key 'mayer_orca'
  - loewdin -> bond.lmdb key 'loewdin_orca'
  - fuzzy   -> bond.lmdb key 'fuzzy_bond'
  - qtaim   -> BCP keys '{i}_{j}' in qtaim.lmdb (0-indexed)

bond.lmdb pair keys use 1-indexed atom numbers: '{i+1}_{ei}_to_{j+1}_{ej}'.
qtaim.lmdb BCP keys use 0-indexed atom indices: '{i}_{j}'.
"""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd
from rdkit.Chem import GetPeriodicTable

_PT = GetPeriodicTable()

SCHEMES = ("qtaim", "mayer", "loewdin", "fuzzy")

_SCHEME_LMDB_KEY: dict[str, str] = {
    "mayer": "mayer_orca",
    "loewdin": "loewdin_orca",
    "fuzzy": "fuzzy_bond",
}


def _rcov(symbol: str) -> float:
    return _PT.GetRcovalent(_PT.GetAtomicNumber(symbol))


def _symbols(mol) -> list[str]:
    return [list(mol[i].species.keys())[0].symbol for i in range(len(mol))]


# ---------------------------------------------------------------------------
# Classifiers
# ---------------------------------------------------------------------------


def enumerate_candidate_pairs(
    structure_record: dict,
    pool_multiplier: float = 1.4,
) -> list[tuple[int, int, float, float]]:
    """Return (i, j, distance, r_cov_sum) for pairs within pool_multiplier * r_cov_sum.

    i, j are 0-indexed with i < j. Brute-force O(n^2); fine up to ~200 atoms.
    """
    mol = structure_record["molecule"]
    coords = mol.cart_coords
    syms = _symbols(mol)
    radii = [_rcov(s) for s in syms]
    n = len(syms)
    pairs: list[tuple[int, int, float, float]] = []
    for i in range(n):
        for j in range(i + 1, n):
            r_cov_sum = radii[i] + radii[j]
            d = float(np.linalg.norm(coords[i] - coords[j]))
            if d <= pool_multiplier * r_cov_sum:
                pairs.append((i, j, d, r_cov_sum))
    return pairs


def classify_geom_bonded(distance: float, r_cov_sum: float, k: float = 1.3) -> bool:
    return distance <= k * r_cov_sum


def classify_qtaim_bonded(qtaim_record: dict, i: int, j: int) -> bool:
    """True if a BCP exists for the pair. Keys in qtaim.lmdb are 0-indexed."""
    return f"{i}_{j}" in qtaim_record or f"{j}_{i}" in qtaim_record


def classify_bond_order(
    bond_record: dict,
    scheme: str,
    i: int,
    j: int,
    element_i: str,
    element_j: str,
    threshold: float = 0.5,
) -> bool | None:
    """Classify bond order for a scheme. i, j are 0-indexed.

    bond.lmdb stores 1-indexed keys: '{i+1}_{ei}_to_{j+1}_{ej}'.
    Returns None if scheme absent or pair not in the dict.
    """
    lmdb_key = _SCHEME_LMDB_KEY.get(scheme)
    if lmdb_key is None:
        return None
    bo_dict = bond_record.get(lmdb_key)
    if not isinstance(bo_dict, dict):
        return None
    k1 = f"{i + 1}_{element_i}_to_{j + 1}_{element_j}"
    k2 = f"{j + 1}_{element_j}_to_{i + 1}_{element_i}"
    val = bo_dict.get(k1)
    if val is None:
        val = bo_dict.get(k2)
    if val is None:
        return None
    return float(val) >= threshold


def per_pair_classification(
    key: str,
    structure_record: dict,
    qtaim_record: dict | None,
    bond_record: dict | None,
    pool_multiplier: float = 1.4,
    geom_k: float = 1.3,
    bo_threshold: float = 0.5,
    vertical: str = "",
) -> list[dict]:
    """One dict per candidate atom pair for a single molecule record."""
    mol = structure_record["molecule"]
    syms = _symbols(mol)
    pairs = enumerate_candidate_pairs(structure_record, pool_multiplier)
    rows: list[dict] = []
    for i, j, d, r_cov_sum in pairs:
        ei, ej = syms[i], syms[j]
        rows.append({
            "key": key,
            "vertical": vertical,
            "i": i,
            "j": j,
            "element_i": ei,
            "element_j": ej,
            "distance": d,
            "r_cov_sum": r_cov_sum,
            "geom_bonded": classify_geom_bonded(d, r_cov_sum, geom_k),
            "qtaim_bonded": (
                classify_qtaim_bonded(qtaim_record, i, j)
                if isinstance(qtaim_record, dict)
                else None
            ),
            "mayer_bonded": (
                classify_bond_order(bond_record, "mayer", i, j, ei, ej, bo_threshold)
                if isinstance(bond_record, dict)
                else None
            ),
            "loewdin_bonded": (
                classify_bond_order(bond_record, "loewdin", i, j, ei, ej, bo_threshold)
                if isinstance(bond_record, dict)
                else None
            ),
            "fuzzy_bonded": (
                classify_bond_order(bond_record, "fuzzy", i, j, ei, ej, bo_threshold)
                if isinstance(bond_record, dict)
                else None
            ),
        })
    return rows


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _binary_f1(pred_series: pd.Series, true_series: pd.Series) -> dict:
    pred = pred_series.astype(bool)
    true = true_series.astype(bool)
    tp = int(((pred) & (true)).sum())
    fp = int(((pred) & (~true)).sum())
    fn = int((~pred & true).sum())
    tn = int((~pred & ~true).sum())
    prec = tp / (tp + fp) if (tp + fp) > 0 else math.nan
    rec = tp / (tp + fn) if (tp + fn) > 0 else math.nan
    if math.isnan(prec) or math.isnan(rec) or (prec + rec) == 0:
        f1 = math.nan
    else:
        f1 = 2 * prec * rec / (prec + rec)
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": prec, "recall": rec, "f1": f1}


def aggregate(df: pd.DataFrame, vertical: str) -> pd.DataFrame:
    """Per-scheme binary F1 vs geom_bonded for a single vertical."""
    rows: list[dict] = []
    for scheme in SCHEMES:
        col = f"{scheme}_bonded"
        if col not in df.columns:
            continue
        sub = df[df[col].notna()].copy()
        if sub.empty:
            continue
        stats = _binary_f1(sub[col], sub["geom_bonded"])
        rows.append({"vertical": vertical, "scheme": scheme, "n_pairs": len(sub), **stats})
    return pd.DataFrame(rows)


def aggregate_by_element_pair(df: pd.DataFrame, vertical: str) -> pd.DataFrame:
    """Per-(scheme, element_pair) binary F1."""
    df = df.copy()
    df["element_pair"] = df.apply(
        lambda r: "_".join(sorted([r["element_i"], r["element_j"]])), axis=1
    )
    rows: list[dict] = []
    for scheme in SCHEMES:
        col = f"{scheme}_bonded"
        if col not in df.columns:
            continue
        for ep, grp in df.groupby("element_pair"):
            sub = grp[grp[col].notna()]
            if sub.empty:
                continue
            stats = _binary_f1(sub[col], sub["geom_bonded"])
            rows.append({
                "vertical": vertical,
                "scheme": scheme,
                "element_pair": ep,
                "n_pairs": len(sub),
                **stats,
            })
    return pd.DataFrame(rows)
