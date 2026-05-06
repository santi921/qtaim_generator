"""Per-atom partial-charge agreement across level-0 schemes.

Companion to ``dipole_alignment``. One row per (record, atom) instead of one
row per record. Scheme set is locked to what is actually present in the
OMol4M level-0 charge.lmdb: four Multiwfn schemes (hirshfeld, cm5, adch,
becke) plus two ORCA-merged schemes (mulliken_orca, loewdin_orca).
mayer_orca is excluded - ORCA's Mayer QA column is labelled "Mulliken gross
atomic charge" in the output and is a duplicate of mulliken_orca.
Sparse-coverage schemes (MBIS, Hirshfeld_orca, VDD) are
ignored; they would just be NaN columns.

Per-vertical summary stores sufficient statistics (n, sum_a, sum_b, sum_aa,
sum_bb, sum_ab) per (scheme-pair, optionally element) so corpus Pearson r
reconstructs in the notebook without re-streaming.

See docs/plans/2026-05-05-analysis-implementation-plan_E2_charge_dipole_comprehensive.md
(Phase 2 atom-level extension; not yet locked into a separate plan).
"""

from __future__ import annotations

import math
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from qtaim_gen.source.analysis.streaming_aggregator import stream_to_parquet

# Locked scheme set. Names match the keys in charge.lmdb. ORCA-merged schemes
# carry the "_orca" suffix per merge_orca_into_charge_json.
# mayer_orca is excluded: ORCA's Mayer population QA column is labelled
# "Mulliken gross atomic charge" in the ORCA output and is identical to
# mulliken_orca up to 4-vs-6 digit truncation (observed MAR ~2.5e-5 e).
# See parse_orca.py merge_orca_into_charge_json() and noise_floors.py.
SCHEMES: tuple[str, ...] = (
    "hirshfeld",
    "cm5",
    "adch",
    "becke",
    "mulliken_orca",
    "loewdin_orca",
)


def _parse_atom_key(atom_key: str) -> tuple[int, str] | None:
    """``"1_C"`` -> ``(0, "C")``. None if unparseable."""
    if not isinstance(atom_key, str) or "_" not in atom_key:
        return None
    idx_str, _, elem = atom_key.partition("_")
    try:
        idx = int(idx_str) - 1  # convert 1-based to 0-based
    except ValueError:
        return None
    if not elem or idx < 0:
        return None
    return idx, elem


def per_record_atoms(
    key: str,
    charge_record: dict | None,
    vertical: str = "",
) -> list[dict[str, Any]]:
    """Build one row per atom for the wide per-atom parquet.

    Atom identity (idx, element) is derived from the union of atom keys
    present across schemes. If two schemes disagree on the element at the
    same atom index for a given record, the record is skipped entirely
    (returned as ``[]`` with a logger warning rather than silently merged).
    """
    if not isinstance(charge_record, dict):
        return []

    # Build per-(idx, element) → {scheme: charge} map.
    by_atom: dict[tuple[int, str], dict[str, float]] = {}
    elem_by_idx: dict[int, str] = {}

    for scheme in SCHEMES:
        scheme_payload = charge_record.get(scheme)
        if not isinstance(scheme_payload, dict):
            continue
        charge_dict = scheme_payload.get("charge")
        if not isinstance(charge_dict, dict):
            continue
        for atom_key, q in charge_dict.items():
            parsed = _parse_atom_key(str(atom_key))
            if parsed is None:
                continue
            idx, elem = parsed
            if idx in elem_by_idx and elem_by_idx[idx] != elem:
                # Inconsistent element at same index across schemes -> skip record.
                return []
            elem_by_idx[idx] = elem
            try:
                qv = float(q)
            except (TypeError, ValueError):
                continue
            by_atom.setdefault((idx, elem), {})[scheme] = qv

    if not by_atom:
        return []

    n_atoms = max((idx for idx, _ in by_atom), default=-1) + 1

    rows: list[dict[str, Any]] = []
    for (idx, elem), per_scheme in sorted(by_atom.items()):
        row: dict[str, Any] = {
            "key": key,
            "vertical": vertical,
            "atom_idx": idx,
            "element": elem,
            "n_atoms": n_atoms,
        }
        for scheme in SCHEMES:
            row[f"charge_{scheme}"] = per_scheme.get(scheme, math.nan)
        rows.append(row)
    return rows


def _pearson_from_sums(
    n: int, sum_a: float, sum_b: float, sum_aa: float, sum_bb: float, sum_ab: float,
) -> float:
    """Closed-form Pearson r from sufficient stats. NaN if degenerate."""
    if n < 2:
        return math.nan
    mean_a = sum_a / n
    mean_b = sum_b / n
    var_a = sum_aa / n - mean_a * mean_a
    var_b = sum_bb / n - mean_b * mean_b
    cov = sum_ab / n - mean_a * mean_b
    if var_a <= 0 or var_b <= 0:
        return math.nan
    return float(cov / math.sqrt(var_a * var_b))


def _sufficient_stats(a: np.ndarray, b: np.ndarray) -> dict[str, float] | None:
    mask = np.isfinite(a) & np.isfinite(b)
    n = int(mask.sum())
    if n == 0:
        return None
    aa = a[mask]
    bb = b[mask]
    return {
        "n": n,
        "sum_a": float(aa.sum()),
        "sum_b": float(bb.sum()),
        "sum_aa": float((aa * aa).sum()),
        "sum_bb": float((bb * bb).sum()),
        "sum_ab": float((aa * bb).sum()),
    }


def aggregate_per_vertical(df: pd.DataFrame, vertical_name: str) -> pd.DataFrame:
    """Long summary: scheme rows + (scheme, element) rows + pair rows + (pair, element) rows."""
    rows: list[dict[str, Any]] = []

    for scheme in SCHEMES:
        col = f"charge_{scheme}"
        if col not in df.columns:
            continue
        vals = df[col].to_numpy(dtype=float)
        finite = vals[np.isfinite(vals)]
        n = int(finite.size)
        rows.append({
            "vertical": vertical_name,
            "kind": "scheme",
            "scheme_a": scheme,
            "scheme_b": None,
            "element": None,
            "n": n,
            "mean": float(finite.mean()) if n else math.nan,
            "sum_a": float(finite.sum()) if n else 0.0,
            "sum_aa": float((finite * finite).sum()) if n else 0.0,
            "sum_b": math.nan, "sum_bb": math.nan, "sum_ab": math.nan,
            "sum_abs_residual": math.nan,
            "pearson_vertical": math.nan,
            "median_abs_residual": math.nan,
        })

        # Per-element scheme rows
        for elem, sub in df.groupby("element"):
            v = sub[col].to_numpy(dtype=float)
            v = v[np.isfinite(v)]
            ne = int(v.size)
            if ne == 0:
                continue
            rows.append({
                "vertical": vertical_name,
                "kind": "scheme_element",
                "scheme_a": scheme,
                "scheme_b": None,
                "element": elem,
                "n": ne,
                "mean": float(v.mean()),
                "sum_a": float(v.sum()),
                "sum_aa": float((v * v).sum()),
                "sum_b": math.nan, "sum_bb": math.nan, "sum_ab": math.nan,
                "sum_abs_residual": math.nan,
                "pearson_vertical": math.nan,
                "median_abs_residual": math.nan,
            })

    for a, b in combinations(SCHEMES, 2):
        ca = f"charge_{a}"
        cb = f"charge_{b}"
        if ca not in df.columns or cb not in df.columns:
            continue
        va = df[ca].to_numpy(dtype=float)
        vb = df[cb].to_numpy(dtype=float)
        stats = _sufficient_stats(va, vb)
        if stats is None:
            stats = {"n": 0, "sum_a": 0.0, "sum_b": 0.0,
                     "sum_aa": 0.0, "sum_bb": 0.0, "sum_ab": 0.0}
        mask = np.isfinite(va) & np.isfinite(vb)
        if mask.any():
            residuals = np.abs(va[mask] - vb[mask])
            mar = float(np.median(residuals))
            sar = float(residuals.sum())
        else:
            mar = math.nan
            sar = 0.0
        pearson = _pearson_from_sums(
            stats["n"], stats["sum_a"], stats["sum_b"],
            stats["sum_aa"], stats["sum_bb"], stats["sum_ab"],
        )
        rows.append({
            "vertical": vertical_name,
            "kind": "pair",
            "scheme_a": a,
            "scheme_b": b,
            "element": None,
            "n": stats["n"],
            "mean": math.nan,
            "sum_a": stats["sum_a"], "sum_b": stats["sum_b"],
            "sum_aa": stats["sum_aa"], "sum_bb": stats["sum_bb"], "sum_ab": stats["sum_ab"],
            "sum_abs_residual": sar,
            "pearson_vertical": pearson,
            "median_abs_residual": mar,
        })

        # Per-element pair rows
        for elem, sub in df.groupby("element"):
            va_e = sub[ca].to_numpy(dtype=float)
            vb_e = sub[cb].to_numpy(dtype=float)
            mask = np.isfinite(va_e) & np.isfinite(vb_e)
            if not mask.any():
                continue
            stats_e = _sufficient_stats(va_e, vb_e)
            residuals = np.abs(va_e[mask] - vb_e[mask])
            mar_e = float(np.median(residuals))
            sar_e = float(residuals.sum())
            pearson_e = _pearson_from_sums(
                stats_e["n"], stats_e["sum_a"], stats_e["sum_b"],
                stats_e["sum_aa"], stats_e["sum_bb"], stats_e["sum_ab"],
            )
            rows.append({
                "vertical": vertical_name,
                "kind": "pair_element",
                "scheme_a": a,
                "scheme_b": b,
                "element": elem,
                "n": stats_e["n"],
                "mean": math.nan,
                "sum_a": stats_e["sum_a"], "sum_b": stats_e["sum_b"],
                "sum_aa": stats_e["sum_aa"], "sum_bb": stats_e["sum_bb"],
                "sum_ab": stats_e["sum_ab"],
                "sum_abs_residual": sar_e,
                "pearson_vertical": pearson_e,
                "median_abs_residual": mar_e,
            })

    return pd.DataFrame(rows)


def run_vertical(
    root: Path | str,
    output_dir: Path | str,
    vertical_name: str | None = None,
    progress: bool = True,
) -> tuple[Path, Path]:
    """Stream charge.lmdb only; emit per-atom + per-vertical summary parquets."""
    root = Path(root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if vertical_name is None:
        vertical_name = root.name

    atoms_path = output_dir / f"{vertical_name}_charge_atoms.parquet"
    summary_path = output_dir / f"{vertical_name}_charge_summary.parquet"

    def _fn(k, charge_rec):
        return per_record_atoms(k, charge_rec, vertical=vertical_name)

    stream_to_parquet(
        root=root,
        lmdb_types=["charge"],
        per_record_fn=_fn,
        output_path=atoms_path,
        progress=progress,
    )

    df = pd.read_parquet(atoms_path)
    summary = aggregate_per_vertical(df, vertical_name)
    summary.to_parquet(summary_path, index=False)
    return atoms_path, summary_path
