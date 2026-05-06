"""Stream F: Cross-method noise floors (B1-B5).

Reads charge.lmdb, bond.lmdb, and qtaim.lmdb per vertical to produce:
  B1/F1-F2: per-element charge residual distributions across schemes
  B2/F3-F4: per-element-pair bond-order residual distributions across schemes
  B3/F5-F6: QTAIM internal redundancy (pairwise Pearson r between BCP descriptors)
  B4/F7-F8: unified per-vertical noise-floor table (published T4 rows)
  B5/F9:    high-disagreement exemplars (top-k by cross-method spread)

Output schema (noise-floor table):
  vertical, analysis, descriptor, element, element_pair,
  mar, iqr, n_obs, pearson_r, top_keys (JSON string)

Output schema (exemplars):
  vertical, key, atom_or_pair, element_or_pair,
  descriptor, schemes_compared, residual
"""
from __future__ import annotations

import json
import math
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd

CHARGE_SCHEMES: tuple[str, ...] = (
    "hirshfeld",
    "cm5",
    "adch",
    "becke",
    "mulliken_orca",
    "loewdin_orca",
    # mayer_orca is excluded: ORCA's Mayer population QA column is labelled
    # "Mulliken gross atomic charge" in the ORCA output and is identical to
    # mulliken_orca up to 4-vs-6 digit truncation (observed MAR ~2.5e-5 e).
    # Treating it as an independent scheme inflates the scheme count and
    # produces a noise-floor row that measures parser precision, not method
    # disagreement. See parse_orca.py merge_orca_into_charge_json() note.
)

BOND_SCHEMES: tuple[str, ...] = (
    "fuzzy_bond",
    "mayer_orca",
    "loewdin_orca",
)

# Locked BCP descriptor set. delocalization_index included if present in record.
QTAIM_DESCRIPTORS: tuple[str, ...] = (
    "density_all",
    "lap_e_density",
    "ellip_e_dens",
    "eta",
    "delocalization_index",
)

_NF_COLUMNS = [
    "vertical", "analysis", "descriptor", "element", "element_pair",
    "mar", "iqr", "n_obs", "pearson_r", "top_keys",
]

_EXEMPLAR_COLUMNS = [
    "vertical", "key", "atom_or_pair", "element_or_pair",
    "descriptor", "schemes_compared", "residual",
]


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def _parse_atom_key(atom_key: str) -> tuple[int, str] | None:
    """'1_C' -> (0, 'C'). Returns None if unparseable."""
    if not isinstance(atom_key, str) or "_" not in atom_key:
        return None
    idx_str, _, elem = atom_key.partition("_")
    try:
        idx = int(idx_str) - 1
    except ValueError:
        return None
    if not elem or idx < 0:
        return None
    return idx, elem


def _parse_bond_pair_key(pair_key: str) -> tuple[int, int, str, str] | None:
    """'1_Ru_to_3_C' -> (0, 2, 'Ru', 'C') with i < j.

    Format: {i+1}_{ei}_to_{j+1}_{ej}. Returns None if unparseable.
    """
    if "_to_" not in pair_key:
        return None
    left, _, right = pair_key.partition("_to_")

    def _split(s: str) -> tuple[int, str] | None:
        parts = s.split("_", 1)
        if len(parts) != 2:
            return None
        try:
            idx = int(parts[0]) - 1
        except ValueError:
            return None
        if idx < 0 or not parts[1]:
            return None
        return idx, parts[1]

    lp = _split(left)
    rp = _split(right)
    if lp is None or rp is None:
        return None
    i, ei = lp
    j, ej = rp
    if i > j:
        i, j, ei, ej = j, i, ej, ei
    return i, j, ei, ej


def _symbols_from_structure(structure_rec: dict) -> list[str] | None:
    """Extract ordered element symbols from a structure record."""
    mol = structure_rec.get("molecule")
    if mol is None:
        return None
    try:
        return [list(mol[k].species.keys())[0].symbol for k in range(len(mol))]
    except Exception:
        return None


# ---------------------------------------------------------------------------
# B1: per-atom charge rows
# ---------------------------------------------------------------------------


def per_record_charge_atoms(
    key: str,
    charge_rec: dict | None,
    vertical: str = "",
) -> list[dict[str, Any]]:
    """One row per atom with charge_{scheme} columns (B1).

    Skips the entire record if two schemes report different elements at the
    same atom index.
    """
    if not isinstance(charge_rec, dict):
        return []

    by_atom: dict[tuple[int, str], dict[str, float]] = {}
    elem_by_idx: dict[int, str] = {}

    for scheme in CHARGE_SCHEMES:
        payload = charge_rec.get(scheme)
        if not isinstance(payload, dict):
            continue
        charge_dict = payload.get("charge")
        if not isinstance(charge_dict, dict):
            continue
        for atom_key, q in charge_dict.items():
            parsed = _parse_atom_key(str(atom_key))
            if parsed is None:
                continue
            idx, elem = parsed
            if idx in elem_by_idx and elem_by_idx[idx] != elem:
                return []
            elem_by_idx[idx] = elem
            try:
                qv = float(q)
            except (TypeError, ValueError):
                continue
            by_atom.setdefault((idx, elem), {})[scheme] = qv

    if not by_atom:
        return []

    rows: list[dict[str, Any]] = []
    for (idx, elem), per_scheme in sorted(by_atom.items()):
        row: dict[str, Any] = {
            "key": key,
            "vertical": vertical,
            "atom_idx": idx,
            "element": elem,
        }
        for scheme in CHARGE_SCHEMES:
            row[f"charge_{scheme}"] = per_scheme.get(scheme, math.nan)
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# B2: per-pair bond-order rows
# ---------------------------------------------------------------------------


def per_record_bond_pairs(
    key: str,
    bond_rec: dict | None,
    vertical: str = "",
) -> list[dict[str, Any]]:
    """One row per atom pair with bo_{scheme} columns (B2).

    Collects the union of pairs across all schemes; missing scheme values
    are NaN.
    """
    if not isinstance(bond_rec, dict):
        return []

    by_pair: dict[tuple[int, int, str, str], dict[str, float]] = {}

    for scheme in BOND_SCHEMES:
        scheme_dict = bond_rec.get(scheme)
        if not isinstance(scheme_dict, dict):
            continue
        for pair_key, val in scheme_dict.items():
            parsed = _parse_bond_pair_key(str(pair_key))
            if parsed is None:
                continue
            i, j, ei, ej = parsed
            try:
                fval = float(val)
            except (TypeError, ValueError):
                continue
            by_pair.setdefault((i, j, ei, ej), {})[scheme] = fval

    if not by_pair:
        return []

    rows: list[dict[str, Any]] = []
    for (i, j, ei, ej), per_scheme in sorted(by_pair.items()):
        ep = "_".join(sorted([ei, ej]))
        row: dict[str, Any] = {
            "key": key,
            "vertical": vertical,
            "i": i,
            "j": j,
            "element_i": ei,
            "element_j": ej,
            "element_pair": ep,
        }
        for scheme in BOND_SCHEMES:
            row[f"bo_{scheme}"] = per_scheme.get(scheme, math.nan)
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# B3: per-BCP descriptor rows
# ---------------------------------------------------------------------------


def per_record_qtaim_bcps(
    key: str,
    qtaim_rec: dict | None,
    structure_rec: dict | None,
    vertical: str = "",
) -> list[dict[str, Any]]:
    """One row per bond critical point with QTAIM descriptor columns (B3).

    BCP keys in qtaim.lmdb are 0-indexed integers joined by '_' (e.g. '0_2').
    Atom-only keys (single integer, no '_') are skipped.
    Element symbols come from structure_rec if available; '?' otherwise.
    """
    if not isinstance(qtaim_rec, dict):
        return []

    syms: list[str] | None = None
    if isinstance(structure_rec, dict):
        syms = _symbols_from_structure(structure_rec)

    rows: list[dict[str, Any]] = []
    for bcp_key, bcp_val in qtaim_rec.items():
        if not isinstance(bcp_val, dict) or "_" not in bcp_key:
            continue
        parts = bcp_key.split("_")
        if len(parts) != 2:
            continue
        try:
            i, j = int(parts[0]), int(parts[1])
        except ValueError:
            continue
        if i < 0 or j < 0:
            continue
        if i > j:
            i, j = j, i

        ei = syms[i] if syms and i < len(syms) else "?"
        ej = syms[j] if syms and j < len(syms) else "?"
        ep = "_".join(sorted([ei, ej]))

        row: dict[str, Any] = {
            "key": key,
            "vertical": vertical,
            "i": i,
            "j": j,
            "element_i": ei,
            "element_j": ej,
            "element_pair": ep,
        }
        for desc in QTAIM_DESCRIPTORS:
            val = bcp_val.get(desc)
            try:
                row[desc] = float(val) if val is not None else math.nan
            except (TypeError, ValueError):
                row[desc] = math.nan
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------


def _mar_iqr(residuals: np.ndarray) -> tuple[float, float]:
    if residuals.size == 0:
        return math.nan, math.nan
    mar = float(np.median(residuals))
    iqr = float(np.percentile(residuals, 75) - np.percentile(residuals, 25))
    return mar, iqr


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    n = int(mask.sum())
    if n < 2:
        return math.nan
    aa, bb = a[mask], b[mask]
    r = np.corrcoef(aa, bb)
    return float(r[0, 1]) if r.shape == (2, 2) else math.nan


def _top_keys(keys_arr: np.ndarray, residuals: np.ndarray, topk: int) -> str:
    idx = np.argsort(residuals)[::-1][:topk]
    seen: dict[str, None] = {}
    for k in keys_arr[idx]:
        seen[str(k)] = None
    return json.dumps(list(seen)[:topk])


def _top_keys_from_mask(
    keys: pd.Series, mask: np.ndarray, residuals: np.ndarray, topk: int,
) -> str:
    """Top-k keys without materializing a string array of all keys.

    `residuals` is aligned to `keys[mask]`; `mask` is a boolean array over `keys`.
    For a vertical with hundreds of millions of long string keys, calling
    `keys.to_numpy(dtype=str)` blows up memory (U91 dtype -> 29+ GB just for the
    array, and another copy for the masked slice). Only `topk` keys are ever used
    in the output, so we use argpartition on the small residuals array and look
    up only those k keys.
    """
    n = residuals.size
    if n == 0:
        return "[]"
    n_take = min(topk, n)
    if n_take < n:
        part = np.argpartition(residuals, n - n_take)[-n_take:]
        order = part[np.argsort(residuals[part])[::-1]]
    else:
        order = np.argsort(residuals)[::-1]
    masked_idx = np.flatnonzero(mask)
    orig_idx = masked_idx[order]
    seen: dict[str, None] = {}
    for k in keys.iloc[orig_idx].astype(str):
        seen[k] = None
    return json.dumps(list(seen)[:topk])


def _empty_nf() -> pd.DataFrame:
    return pd.DataFrame(columns=_NF_COLUMNS)


# ---------------------------------------------------------------------------
# B1: charge noise-floor aggregation
# ---------------------------------------------------------------------------


def aggregate_charge_noise_floor(
    df: pd.DataFrame,
    vertical: str,
    topk: int = 50,
) -> pd.DataFrame:
    """Per (element, scheme_pair) MAR, IQR, n_obs, top_keys (B1).

    Also emits global rows (element=None) for the corpus-level floor.
    """
    if df.empty:
        return _empty_nf()

    rows: list[dict[str, Any]] = []
    keys = df["key"]

    for sa, sb in combinations(CHARGE_SCHEMES, 2):
        ca, cb = f"charge_{sa}", f"charge_{sb}"
        if ca not in df.columns or cb not in df.columns:
            continue
        desc = f"{sa}_vs_{sb}"
        va = df[ca].to_numpy(dtype=float)
        vb = df[cb].to_numpy(dtype=float)
        mask = np.isfinite(va) & np.isfinite(vb)
        if not mask.any():
            continue
        res = np.abs(va[mask] - vb[mask])

        mar, iqr = _mar_iqr(res)
        rows.append({
            "vertical": vertical, "analysis": "charge",
            "descriptor": desc, "element": None, "element_pair": None,
            "mar": mar, "iqr": iqr, "n_obs": int(mask.sum()),
            "pearson_r": math.nan,
            "top_keys": _top_keys_from_mask(keys, mask, res, topk),
        })

        for elem, grp in df.groupby("element"):
            va_e = grp[ca].to_numpy(dtype=float)
            vb_e = grp[cb].to_numpy(dtype=float)
            m = np.isfinite(va_e) & np.isfinite(vb_e)
            if not m.any():
                continue
            res_e = np.abs(va_e[m] - vb_e[m])
            mar_e, iqr_e = _mar_iqr(res_e)
            rows.append({
                "vertical": vertical, "analysis": "charge",
                "descriptor": desc, "element": elem, "element_pair": None,
                "mar": mar_e, "iqr": iqr_e, "n_obs": int(m.sum()),
                "pearson_r": math.nan,
                "top_keys": _top_keys_from_mask(grp["key"].reset_index(drop=True), m, res_e, topk),
            })

    return pd.DataFrame(rows) if rows else _empty_nf()


# ---------------------------------------------------------------------------
# B2: bond-order noise-floor aggregation
# ---------------------------------------------------------------------------


def aggregate_bond_noise_floor(
    df: pd.DataFrame,
    vertical: str,
    topk: int = 50,
) -> pd.DataFrame:
    """Per (element_pair, scheme_pair) MAR, IQR, n_obs, top_keys (B2)."""
    if df.empty:
        return _empty_nf()

    rows: list[dict[str, Any]] = []
    keys = df["key"]

    for sa, sb in combinations(BOND_SCHEMES, 2):
        ca, cb = f"bo_{sa}", f"bo_{sb}"
        if ca not in df.columns or cb not in df.columns:
            continue
        desc = f"{sa}_vs_{sb}"
        va = df[ca].to_numpy(dtype=float)
        vb = df[cb].to_numpy(dtype=float)
        mask = np.isfinite(va) & np.isfinite(vb)
        if not mask.any():
            continue
        res = np.abs(va[mask] - vb[mask])

        mar, iqr = _mar_iqr(res)
        rows.append({
            "vertical": vertical, "analysis": "bond_order",
            "descriptor": desc, "element": None, "element_pair": None,
            "mar": mar, "iqr": iqr, "n_obs": int(mask.sum()),
            "pearson_r": math.nan,
            "top_keys": _top_keys_from_mask(keys, mask, res, topk),
        })

        for ep, grp in df.groupby("element_pair"):
            va_e = grp[ca].to_numpy(dtype=float)
            vb_e = grp[cb].to_numpy(dtype=float)
            m = np.isfinite(va_e) & np.isfinite(vb_e)
            if not m.any():
                continue
            res_e = np.abs(va_e[m] - vb_e[m])
            mar_e, iqr_e = _mar_iqr(res_e)
            rows.append({
                "vertical": vertical, "analysis": "bond_order",
                "descriptor": desc, "element": None, "element_pair": ep,
                "mar": mar_e, "iqr": iqr_e, "n_obs": int(m.sum()),
                "pearson_r": math.nan,
                "top_keys": _top_keys_from_mask(grp["key"].reset_index(drop=True), m, res_e, topk),
            })

    return pd.DataFrame(rows) if rows else _empty_nf()


# ---------------------------------------------------------------------------
# B3: QTAIM redundancy aggregation
# ---------------------------------------------------------------------------


def aggregate_qtaim_redundancy(
    df: pd.DataFrame,
    vertical: str,
) -> pd.DataFrame:
    """Per (element_pair, descriptor_pair) Pearson r (B3).

    Only descriptors actually present in df.columns are included. Rows with
    fewer than 2 finite observations are dropped.
    """
    if df.empty:
        return _empty_nf()

    present = [d for d in QTAIM_DESCRIPTORS if d in df.columns]
    if len(present) < 2:
        return _empty_nf()

    rows: list[dict[str, Any]] = []

    for da, db in combinations(present, 2):
        desc = f"{da}_vs_{db}"
        va = df[da].to_numpy(dtype=float)
        vb = df[db].to_numpy(dtype=float)
        n_global = int((np.isfinite(va) & np.isfinite(vb)).sum())
        r_global = _pearson(va, vb)
        if n_global >= 2:
            rows.append({
                "vertical": vertical, "analysis": "qtaim_redundancy",
                "descriptor": desc, "element": None, "element_pair": None,
                "mar": math.nan, "iqr": math.nan, "n_obs": n_global,
                "pearson_r": r_global, "top_keys": json.dumps([]),
            })

        if "element_pair" in df.columns:
            for ep, grp in df.groupby("element_pair"):
                va_e = grp[da].to_numpy(dtype=float)
                vb_e = grp[db].to_numpy(dtype=float)
                m = np.isfinite(va_e) & np.isfinite(vb_e)
                if m.sum() < 2:
                    continue
                rows.append({
                    "vertical": vertical, "analysis": "qtaim_redundancy",
                    "descriptor": desc, "element": None, "element_pair": ep,
                    "mar": math.nan, "iqr": math.nan, "n_obs": int(m.sum()),
                    "pearson_r": _pearson(va_e, vb_e),
                    "top_keys": json.dumps([]),
                })

    return pd.DataFrame(rows) if rows else _empty_nf()


# ---------------------------------------------------------------------------
# B4: Unified noise-floor table
# ---------------------------------------------------------------------------


def build_noise_floor_table(
    charge_nf: pd.DataFrame,
    bond_nf: pd.DataFrame,
    qtaim_redund: pd.DataFrame,
) -> pd.DataFrame:
    """Concatenate B1-B3 results into one noise-floor table (B4)."""
    parts = [df for df in [charge_nf, bond_nf, qtaim_redund] if not df.empty]
    if not parts:
        return _empty_nf()
    return pd.concat(parts, ignore_index=True)


# ---------------------------------------------------------------------------
# B5: High-disagreement exemplars
# ---------------------------------------------------------------------------


def emit_exemplars(
    charge_df: pd.DataFrame,
    bond_df: pd.DataFrame,
    topk: int = 50,
    vertical: str = "",
) -> pd.DataFrame:
    """Top-topk exemplars per (descriptor, element_or_pair) by cross-method spread (B5).

    Spread = nanmax - nanmin across scheme values for atoms/pairs with at
    least 2 finite scheme values. Separate ranking per element (charge) and
    per element-pair (bond order).
    """
    rows: list[dict[str, Any]] = []

    # --- Charge exemplars ---
    if not charge_df.empty:
        cdf = charge_df.reset_index(drop=True)
        scheme_cols = [f"charge_{s}" for s in CHARGE_SCHEMES if f"charge_{s}" in cdf.columns]
        if len(scheme_cols) >= 2:
            vals = cdf[scheme_cols].to_numpy(dtype=float)
            n_finite = np.sum(np.isfinite(vals), axis=1)
            spread = np.where(
                n_finite >= 2,
                np.nanmax(vals, axis=1) - np.nanmin(vals, axis=1),
                np.nan,
            )
            cdf = cdf.copy()
            cdf["_spread"] = spread
            schemes_str = ",".join(scheme_cols)
            for elem, grp in cdf.groupby("element"):
                top = grp[grp["_spread"].notna()].nlargest(topk, "_spread")
                for _, r in top.iterrows():
                    rows.append({
                        "vertical": vertical,
                        "key": r["key"],
                        "atom_or_pair": str(int(r["atom_idx"])),
                        "element_or_pair": str(elem),
                        "descriptor": "charge",
                        "schemes_compared": schemes_str,
                        "residual": float(r["_spread"]),
                    })

    # --- Bond-order exemplars ---
    if not bond_df.empty:
        bdf = bond_df.reset_index(drop=True)
        bo_cols = [f"bo_{s}" for s in BOND_SCHEMES if f"bo_{s}" in bdf.columns]
        if len(bo_cols) >= 2:
            vals = bdf[bo_cols].to_numpy(dtype=float)
            n_finite = np.sum(np.isfinite(vals), axis=1)
            spread = np.where(
                n_finite >= 2,
                np.nanmax(vals, axis=1) - np.nanmin(vals, axis=1),
                np.nan,
            )
            bdf = bdf.copy()
            bdf["_spread"] = spread
            schemes_str = ",".join(bo_cols)
            for ep, grp in bdf.groupby("element_pair"):
                top = grp[grp["_spread"].notna()].nlargest(topk, "_spread")
                for _, r in top.iterrows():
                    rows.append({
                        "vertical": vertical,
                        "key": r["key"],
                        "atom_or_pair": f"{int(r['i'])}_{int(r['j'])}",
                        "element_or_pair": str(ep),
                        "descriptor": "bond_order",
                        "schemes_compared": schemes_str,
                        "residual": float(r["_spread"]),
                    })

    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=_EXEMPLAR_COLUMNS)
