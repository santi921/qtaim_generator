"""
LMDB descriptor analysis utilities.

Provides iteration, extraction, and statistics functions for the 6 raw LMDB
types produced by json-to-lmdb: structure, charge, qtaim, bond, fuzzy, other.

Designed for import by notebooks and the analyze-lmdbs CLI.
Plotting functions are in the separate lmdb_plots module to allow headless use.
"""
from __future__ import annotations

import lmdb
import logging
import math
import pickle
from collections.abc import Iterator
from pathlib import Path
from typing import Any, TypedDict

import numpy as np

logger = logging.getLogger(__name__)

METADATA_KEYS: frozenset[str] = frozenset({"length", "scaled", "scaler_finalized"})

# Maps LMDB type name -> list of candidate filenames (canonical first, then merged_ variant)
LMDB_TYPE_NAMES: dict[str, list[str]] = {
    "structure": ["structure.lmdb", "merged_geom.lmdb"],
    "charge": ["charge.lmdb", "merged_charge.lmdb"],
    "qtaim": ["qtaim.lmdb", "merged_qtaim.lmdb"],
    "bond": ["bond.lmdb", "merged_bond.lmdb"],
    "fuzzy": ["fuzzy.lmdb", "merged_fuzzy.lmdb"],
    "other": ["other.lmdb", "merged_other.lmdb"],
}

__all__ = [
    "iter_lmdb",
    "get_lmdb_entry_count",
    "discover_lmdbs",
    "compute_stats",
    "flatten_entry",
    "flatten_bond_entry",
    "extract_structure_info",
    "split_qtaim_atom_bond",
    "StatsResult",
    "METADATA_KEYS",
    "LMDB_TYPE_NAMES",
]


class StatsResult(TypedDict):
    count: int
    nan_count: int
    inf_count: int
    mean: float | None
    std: float | None
    min: float | None
    max: float | None
    percentiles: dict[int, float]


def iter_lmdb(
    lmdb_path: str | Path,
    sample_size: int | None = None,
) -> Iterator[tuple[str, dict[str, Any]]]:
    """Iterate LMDB entries, yielding (key, unpickled_data) pairs.

    If sample_size is None, yields all entries.
    If sample_size is set, uses modular sampling (every Nth entry).
    Skips metadata keys and logs corrupt entries.
    """
    lmdb_path = str(lmdb_path)
    env = lmdb.open(
        lmdb_path,
        subdir=False,
        readonly=True,
        lock=False,
        readahead=True,
        meminit=False,
    )

    # Determine step size for modular sampling
    step = 1
    if sample_size is not None and sample_size > 0:
        total = get_lmdb_entry_count(lmdb_path)
        if total > sample_size:
            step = max(1, total // sample_size)

    entry_index = 0
    corrupt_count = 0

    try:
        with env.begin(write=False) as txn:
            cursor = txn.cursor()
            for key, value in cursor:
                key_str = key.decode("ascii")
                if key_str in METADATA_KEYS:
                    continue

                # Modular sampling: only yield every Nth entry
                if step > 1 and entry_index % step != 0:
                    entry_index += 1
                    continue

                try:
                    data = pickle.loads(value)
                except Exception as e:
                    corrupt_count += 1
                    logger.warning(f"Corrupt entry {key_str}: {e}")
                    entry_index += 1
                    continue

                yield key_str, data
                entry_index += 1
    finally:
        if corrupt_count > 0:
            logger.warning(f"Skipped {corrupt_count} corrupt entries in {lmdb_path}")
        env.close()


def get_lmdb_entry_count(lmdb_path: str | Path) -> int:
    """Read the 'length' metadata key from an LMDB file."""
    env = lmdb.open(
        str(lmdb_path),
        subdir=False,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
    )
    try:
        with env.begin(write=False) as txn:
            length_bytes = txn.get(b"length")
            if length_bytes is not None:
                return pickle.loads(length_bytes)
        return 0
    finally:
        env.close()


def discover_lmdbs(directory: str | Path) -> dict[str, Path]:
    """Auto-discover LMDB files by type from a directory.

    Checks canonical names first (e.g., structure.lmdb), then merged_
    prefix variants (e.g., merged_geom.lmdb). Returns the first match.
    """
    directory = Path(directory)
    found: dict[str, Path] = {}
    for type_name, candidates in LMDB_TYPE_NAMES.items():
        for candidate in candidates:
            path = directory / candidate
            if path.exists():
                found[type_name] = path
                break
    return found


def compute_stats(values: np.ndarray) -> StatsResult:
    """Compute statistics on a numpy array. NaN-aware.

    Replaces inf with nan before computing mean/std/percentiles.
    Returns None for mean/std/min/max when no valid values exist.
    """
    values = np.asarray(values, dtype=np.float64)

    nan_count = int(np.isnan(values).sum())
    inf_count = int(np.isinf(values).sum())
    count = len(values)

    # Replace inf with nan for stats computation
    clean = np.where(np.isinf(values), np.nan, values)
    valid_count = int(np.isfinite(clean).sum())

    if valid_count == 0:
        return StatsResult(
            count=count,
            nan_count=nan_count,
            inf_count=inf_count,
            mean=None,
            std=None,
            min=None,
            max=None,
            percentiles={},
        )

    return StatsResult(
        count=count,
        nan_count=nan_count,
        inf_count=inf_count,
        mean=float(np.nanmean(clean)),
        std=float(np.nanstd(clean)),
        min=float(np.nanmin(clean)),
        max=float(np.nanmax(clean)),
        percentiles={
            q: float(np.nanpercentile(clean, q))
            for q in (5, 25, 50, 75, 95)
        },
    )


def flatten_entry(entry: dict[str, Any], prefix: str = "") -> dict[str, float]:
    """Generic recursive flattener for nested dicts.

    Walks the dict tree, collecting numeric leaf values with dotted key paths.
    Skips non-numeric values (lists, strings, objects).
    Does NOT normalize bond keys — use flatten_bond_entry for bond LMDBs.
    """
    result: dict[str, float] = {}
    for key, value in entry.items():
        full_key = f"{prefix}{key}" if not prefix else f"{prefix}.{key}"
        if isinstance(value, (int, float)):
            if isinstance(value, bool):
                continue
            result[full_key] = float(value)
        elif isinstance(value, dict):
            result.update(flatten_entry(value, prefix=full_key))
    return result


def flatten_bond_entry(entry: dict[str, Any]) -> dict[str, float]:
    """Flatten a bond.lmdb entry with key normalization.

    Strips '_bond' suffix from top-level bond type keys
    (e.g., 'fuzzy_bond' -> 'fuzzy') before flattening.
    """
    normalized: dict[str, Any] = {}
    for key, value in entry.items():
        norm_key = key[:-5] if key.endswith("_bond") else key
        normalized[norm_key] = value
    return flatten_entry(normalized)


def extract_structure_info(entry: dict[str, Any]) -> dict[str, int | list[str]]:
    """Extract summary info from a structure.lmdb entry.

    Returns: {n_atoms, n_bonds, charge, spin, elements}
    Handles pymatgen Molecule objects.
    """
    molecule = entry.get("molecule")
    elements: list[str] = []
    n_atoms = 0

    if molecule is not None:
        try:
            n_atoms = len(molecule)
            elements = [
                str(site.species.elements).split(" ")[-1].split("]")[0]
                for site in molecule
            ]
        except Exception as e:
            logger.warning(f"Failed to extract structure info: {e}")

    bonds = entry.get("bonds", [])
    return {
        "n_atoms": n_atoms,
        "n_bonds": len(bonds),
        "charge": entry.get("charge", 0),
        "spin": entry.get("spin", 1),
        "elements": elements,
    }


def split_qtaim_atom_bond(
    entry: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Split a QTAIM entry into atom CPs vs bond CPs.

    Uses the underscore rule on TOP-LEVEL keys only:
    - Keys without underscore (e.g., "0", "1") -> atom (NCP)
    - Keys with underscore (e.g., "13_28") -> bond (BCP)

    Returns: (atom_entries, bond_entries)
    """
    atom_entries: dict[str, dict[str, Any]] = {}
    bond_entries: dict[str, dict[str, Any]] = {}

    for key, value in entry.items():
        if not isinstance(value, dict):
            continue
        if "_" in key:
            bond_entries[key] = value
        else:
            atom_entries[key] = value

    return atom_entries, bond_entries
