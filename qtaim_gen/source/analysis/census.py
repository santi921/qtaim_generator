"""Per-vertical census (Stream C).

Reads four of the eight per-vertical LMDBs (structure, charge, qtaim, bond),
aggregates corpus counts, and returns a single dict per vertical. The CLI
in ``qtaim_gen/source/scripts/analysis_census.py`` writes a parquet with one
row per vertical.

Schema (locked):
    vertical, n_structures, n_unique_formulas, n_atom_records,
    n_charge_scheme_records, n_bcps, n_bonds_total
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from qtaim_gen.source.analysis.streaming_aggregator import stream
from qtaim_gen.source.scripts.helpers.pull_holdout_records import NON_VERTICAL_DIRS

# Known bond schemes across the OMol4M corpus, mapped to short column suffixes.
# Coverage observed in the level-0 LMDBs as of 2026-05-05:
#   mayer_orca, loewdin_orca, fuzzy_bond  -> universal
#   ibsi_bond                             -> rmechdb (universal), noble_gas (~4%)
# Schemes not in this map still contribute to n_bonds_total but get no
# dedicated column. Add new entries here when the corpus grows.
_BOND_SCHEME_SHORT = {
    "mayer_orca": "mayer",
    "loewdin_orca": "loewdin",
    "fuzzy_bond": "fuzzy",
    "ibsi_bond": "ibsi",
}

CENSUS_FIELDS = (
    "vertical",
    "n_structures",
    "n_unique_formulas",
    "n_atom_records",
    "n_charge_scheme_records",
    "n_bcps",
    "n_bonds_total",
    *(f"n_bonds_{short}" for short in _BOND_SCHEME_SHORT.values()),
)


def _structure_fn(_key, rec):
    if not isinstance(rec, dict) or "molecule" not in rec:
        return {}
    mol = rec["molecule"]
    try:
        return {
            "formula": mol.composition.alphabetical_formula,
            "n_atoms": len(mol),
        }
    except AttributeError:
        return {}


def _charge_fn(_key, rec):
    if not isinstance(rec, dict):
        return {}
    return {"n_schemes": len(rec)}


def _qtaim_fn(_key, rec):
    if not isinstance(rec, dict):
        return {}
    n_bcps = sum(1 for k in rec if isinstance(k, str) and k.count("_") == 1)
    return {"n_bcps": n_bcps}


def _bond_fn(_key, rec):
    if not isinstance(rec, dict):
        return {}
    out = {"n_bonds": 0}
    for scheme, payload in rec.items():
        try:
            n = len(payload)
        except TypeError:
            continue
        out["n_bonds"] += n
        short = _BOND_SCHEME_SHORT.get(scheme)
        if short is not None:
            out[f"n_bonds_{short}"] = out.get(f"n_bonds_{short}", 0) + n
    return out


def _safe_sum(df, col) -> int:
    if col not in df.columns:
        return 0
    s = df[col].sum()
    if s != s:  # NaN guard: empty/all-NaN series .sum() returns 0 already, but defensive
        return 0
    return int(s)


def _safe_nunique(df, col) -> int:
    if col not in df.columns:
        return 0
    return int(df[col].nunique())


def _resolve_lmdb_dir(path: Path) -> Path | None:
    """Return ``path`` if it contains ``structure.lmdb``, else None.

    Level-0 calcs only. The level-2 / ``merged/`` fallback (e.g.
    ``geom_orca6/merged/``) is intentionally NOT handled here yet -- those
    LMDBs are still being finalized and will be added once stable.
    """
    if (path / "structure.lmdb").exists():
        return path
    return None


def census(root: Path | str, vertical_name: str | None = None) -> dict[str, Any]:
    """Return a one-row census dict for the vertical at ``root``.

    ``root`` must point at a directory containing ``structure.lmdb``.
    ``vertical_name`` overrides the display name for the ``vertical``
    column; if absent, defaults to ``root.name``.
    """
    root = Path(root)
    lmdb_dir = _resolve_lmdb_dir(root)
    if lmdb_dir is None:
        raise FileNotFoundError(f"no structure.lmdb directly under {root}")
    if vertical_name is None:
        vertical_name = root.name

    df_s = stream(lmdb_dir, ["structure"], _structure_fn, progress=False)
    df_c = stream(lmdb_dir, ["charge"], _charge_fn, progress=False)
    df_q = stream(lmdb_dir, ["qtaim"], _qtaim_fn, progress=False)
    df_b = stream(lmdb_dir, ["bond"], _bond_fn, progress=False)

    row = {
        "vertical": vertical_name,
        "n_structures": int(len(df_s)),
        "n_unique_formulas": _safe_nunique(df_s, "formula"),
        "n_atom_records": _safe_sum(df_s, "n_atoms"),
        "n_charge_scheme_records": _safe_sum(df_c, "n_schemes"),
        "n_bcps": _safe_sum(df_q, "n_bcps"),
        "n_bonds_total": _safe_sum(df_b, "n_bonds"),
    }
    for short in _BOND_SCHEME_SHORT.values():
        row[f"n_bonds_{short}"] = _safe_sum(df_b, f"n_bonds_{short}")
    return row


def discover_verticals(root: Path | str) -> list[tuple[str, Path]]:
    """Resolve a root into ``(display_name, lmdb_dir)`` pairs.

    Level-0 calcs only:

    - ``root/structure.lmdb`` exists -> single vertical, name = ``root.name``.
    - Otherwise -> corpus mode. One row per immediate subdir ``<v>`` for
      which ``root/<v>/structure.lmdb`` exists.

    Subdirs in ``NON_VERTICAL_DIRS`` (e.g. ``holdout_lmdbs``,
    ``filter_csv_for_holdouts``) are skipped. Verticals whose LMDBs live
    only under a ``merged/`` subdirectory (e.g. ``geom_orca6``, the
    level-2 / orca refinement) are also skipped for now and will be
    handled when those LMDBs are finalized. Result is sorted by name.
    """
    root = Path(root)
    direct = _resolve_lmdb_dir(root)
    if direct is not None:
        return [(root.name, direct)]
    out: list[tuple[str, Path]] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        if child.name in NON_VERTICAL_DIRS:
            continue
        lmdb_dir = _resolve_lmdb_dir(child)
        if lmdb_dir is not None:
            out.append((child.name, lmdb_dir))
    return out
