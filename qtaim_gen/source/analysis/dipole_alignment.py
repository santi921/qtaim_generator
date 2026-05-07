"""Pairwise dipole agreement (Stream E2 / paper section 6.7).

Four independent Multiwfn dipoles plus the ORCA SCF dipole; five methods total,
no privileged reference. CM5 is excluded: the Multiwfn combined output prints
the Hirshfeld promolecular dipole in the CM5 section (confirmed from test
fixture; CM5 charges are valid but CM5 dipole is not). Single 2-LMDB join
(charge + orca). Wide per-record schema; sufficient-stats per-vertical summary
so corpus Pearson reconstructs in the notebook.

See docs/plans/2026-05-05-analysis-implementation-plan_E2_charge_dipole_comprehensive.md.
"""

from __future__ import annotations

import math
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from qtaim_gen.source.analysis.streaming_aggregator import stream_to_parquet

# Method names: <scheme>_<producer>. ORCA SCF is the 5th method, not a reference.
# Level-0 only (per validation.py: tier 0 = hirshfeld + adch + cm5 + becke).
# VDD/MBIS/CHELPG are tier-1; not present in the OMol4M level-0 LMDBs.
# CM5 excluded: Multiwfn prints the Hirshfeld promolecular dipole in the CM5
# section of the combined output file, making cm5_dipole == hirshfeld_dipole.
# CM5 charges are valid; CM5 dipoles are not.
METHODS: tuple[str, ...] = (
    "adch_multiwfn",
    "becke_multiwfn",
    "hirshfeld_multiwfn",
    "scf_orca",
)

_MULTIWFN_SCHEME_FOR_METHOD = {
    "adch_multiwfn": "adch",
    "becke_multiwfn": "becke",
    "hirshfeld_multiwfn": "hirshfeld",
}

_TRANSITION_METALS = frozenset({
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
})


def extract_method_dipoles(
    charge_record: dict | None,
    orca_record: dict | None,
) -> dict[str, tuple[float, list[float]]]:
    """Pull (mag, xyz) for each of the five methods.

    Missing fields produce ``(nan, [nan, nan, nan])`` rather than raising.
    """
    out: dict[str, tuple[float, list[float]]] = {}
    nan_xyz = [math.nan, math.nan, math.nan]

    for method, scheme in _MULTIWFN_SCHEME_FOR_METHOD.items():
        mag = math.nan
        xyz = nan_xyz
        if isinstance(charge_record, dict):
            scheme_payload = charge_record.get(scheme)
            if isinstance(scheme_payload, dict):
                dipole = scheme_payload.get("dipole")
                if isinstance(dipole, dict):
                    raw_mag = dipole.get("mag")
                    raw_xyz = dipole.get("xyz")
                    if isinstance(raw_mag, (int, float)):
                        mag = float(raw_mag)
                    if isinstance(raw_xyz, (list, tuple)) and len(raw_xyz) == 3:
                        try:
                            xyz = [float(c) for c in raw_xyz]
                        except (TypeError, ValueError):
                            xyz = nan_xyz
        out[method] = (mag, xyz)

    mag = math.nan
    xyz = nan_xyz
    if isinstance(orca_record, dict):
        raw_mag = orca_record.get("dipole_magnitude_au")
        raw_xyz = orca_record.get("dipole_au")
        if isinstance(raw_mag, (int, float)):
            mag = float(raw_mag)
        if isinstance(raw_xyz, (list, tuple)) and len(raw_xyz) == 3:
            try:
                xyz = [float(c) for c in raw_xyz]
            except (TypeError, ValueError):
                xyz = nan_xyz
    out["scf_orca"] = (mag, xyz)
    return out


def cos_sim(xyz_a: list[float], xyz_b: list[float]) -> float:
    """Cosine similarity. NaN if either norm < 1e-10 or any input is NaN."""
    a = np.asarray(xyz_a, dtype=float)
    b = np.asarray(xyz_b, dtype=float)
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        return math.nan
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-10 or nb < 1e-10:
        return math.nan
    return float(np.dot(a, b) / (na * nb))


def _element_set_from_charge(charge_record: dict | None) -> str:
    """Derive sorted unique-element string from charge atom_keys (e.g. '3_C')."""
    if not isinstance(charge_record, dict):
        return ""
    elements: set[str] = set()
    for scheme_payload in charge_record.values():
        if not isinstance(scheme_payload, dict):
            continue
        charges = scheme_payload.get("charge")
        if not isinstance(charges, dict):
            continue
        for atom_key in charges.keys():
            ak = str(atom_key)
            if "_" in ak:
                _, _, elem = ak.partition("_")
                if elem:
                    elements.add(elem)
        if elements:
            break
    return "_".join(sorted(elements))


def _n_atoms_from_charge(charge_record: dict | None) -> int:
    if not isinstance(charge_record, dict):
        return 0
    for scheme_payload in charge_record.values():
        if isinstance(scheme_payload, dict):
            charges = scheme_payload.get("charge")
            if isinstance(charges, dict) and charges:
                return len(charges)
    return 0


def per_record(
    key: str,
    charge_record: dict | None,
    orca_record: dict | None,
    vertical: str = "",
) -> dict[str, Any]:
    """Build the wide per-record row for the parquet."""
    dipoles = extract_method_dipoles(charge_record, orca_record)
    element_set = _element_set_from_charge(charge_record)
    has_tm = bool(element_set) and any(
        e in _TRANSITION_METALS for e in element_set.split("_")
    )

    row: dict[str, Any] = {
        "key": key,
        "vertical": vertical,
        "n_atoms": _n_atoms_from_charge(charge_record),
        "element_set": element_set,
        "has_tm": has_tm,
    }
    for method in METHODS:
        mag, xyz = dipoles[method]
        row[f"dipole_mag_{method}"] = mag
        row[f"dipole_x_{method}"] = xyz[0]
        row[f"dipole_y_{method}"] = xyz[1]
        row[f"dipole_z_{method}"] = xyz[2]
    return row


def _xyz_cols(method: str) -> list[str]:
    return [f"dipole_x_{method}", f"dipole_y_{method}", f"dipole_z_{method}"]


def _percentile(arr: np.ndarray, q: float) -> float:
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return math.nan
    return float(np.percentile(arr, q))


def aggregate_per_vertical(
    df: pd.DataFrame, vertical_name: str
) -> pd.DataFrame:
    """Long-form summary with method rows + method-pair sufficient stats."""
    rows: list[dict[str, Any]] = []

    for method in METHODS:
        mag_col = f"dipole_mag_{method}"
        if mag_col not in df.columns:
            continue
        mags = df[mag_col].to_numpy(dtype=float)
        finite = mags[np.isfinite(mags)]
        n = int(finite.size)
        rows.append({
            "vertical": vertical_name,
            "kind": "method",
            "method_a": method,
            "method_b": None,
            "n": n,
            "sum_mag": float(finite.sum()) if n else 0.0,
            "sum_mag_sq": float((finite * finite).sum()) if n else 0.0,
            "sum_a": math.nan,
            "sum_b": math.nan,
            "sum_aa": math.nan,
            "sum_bb": math.nan,
            "sum_ab": math.nan,
            "cos_sim_median": math.nan,
            "cos_sim_p10": math.nan,
            "pearson_mag_vertical": math.nan,
        })

    for a, b in combinations(METHODS, 2):
        mag_a_col = f"dipole_mag_{a}"
        mag_b_col = f"dipole_mag_{b}"
        if mag_a_col not in df.columns or mag_b_col not in df.columns:
            continue
        mag_a = df[mag_a_col].to_numpy(dtype=float)
        mag_b = df[mag_b_col].to_numpy(dtype=float)
        mask = np.isfinite(mag_a) & np.isfinite(mag_b)
        n = int(mask.sum())

        if n == 0:
            sum_a = sum_b = sum_aa = sum_bb = sum_ab = 0.0
            pearson = math.nan
        else:
            ma = mag_a[mask]
            mb = mag_b[mask]
            sum_a = float(ma.sum())
            sum_b = float(mb.sum())
            sum_aa = float((ma * ma).sum())
            sum_bb = float((mb * mb).sum())
            sum_ab = float((ma * mb).sum())
            pearson = _pearson_from_sums(n, sum_a, sum_b, sum_aa, sum_bb, sum_ab)

        a_xyz = df[_xyz_cols(a)].to_numpy(dtype=float)
        b_xyz = df[_xyz_cols(b)].to_numpy(dtype=float)
        cos_vals = np.array(
            [cos_sim(a_xyz[i].tolist(), b_xyz[i].tolist()) for i in range(len(df))],
            dtype=float,
        )
        rows.append({
            "vertical": vertical_name,
            "kind": "pair",
            "method_a": a,
            "method_b": b,
            "n": n,
            "sum_mag": math.nan,
            "sum_mag_sq": math.nan,
            "sum_a": sum_a,
            "sum_b": sum_b,
            "sum_aa": sum_aa,
            "sum_bb": sum_bb,
            "sum_ab": sum_ab,
            "cos_sim_median": _percentile(cos_vals, 50),
            "cos_sim_p10": _percentile(cos_vals, 10),
            "pearson_mag_vertical": pearson,
        })

    return pd.DataFrame(rows)


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


def run_vertical(
    root: Path | str,
    output_dir: Path | str,
    vertical_name: str | None = None,
    progress: bool = True,
) -> tuple[Path, Path]:
    """Stream charge + orca, write per-record + per-vertical summary parquets."""
    root = Path(root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if vertical_name is None:
        vertical_name = root.name

    records_path = output_dir / f"{vertical_name}_dipole_records.parquet"
    summary_path = output_dir / f"{vertical_name}_dipole_summary.parquet"

    def _fn(k, charge_rec, orca_rec):
        return per_record(k, charge_rec, orca_rec, vertical=vertical_name)

    stream_to_parquet(
        root=root,
        lmdb_types=["charge", "orca"],
        per_record_fn=_fn,
        output_path=records_path,
        progress=progress,
    )

    df = pd.read_parquet(records_path)
    summary = aggregate_per_vertical(df, vertical_name)
    summary.to_parquet(summary_path, index=False)
    return records_path, summary_path
