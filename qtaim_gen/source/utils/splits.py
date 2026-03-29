"""
Train/val/test splitting utilities for graph LMDB files.

Provides pure partition functions (testable without LMDB) and an LMDB I/O
wrapper that copies key-value pairs into separate split LMDB files.
"""

import hashlib
import os
import pickle
import random
from dataclasses import dataclass
from typing import Literal

import lmdb


SPLIT_NAMES = ("train", "val", "test")
METADATA_KEYS = {"length", "scaled", "split_name"}


@dataclass(frozen=True)
class SplitConfig:
    method: Literal["random", "composition"]
    ratios: tuple[float, float, float]
    seed: int = 42

    def __post_init__(self):
        if len(self.ratios) != 3:
            raise ValueError(f"split_ratios must have exactly 3 values, got {len(self.ratios)}")
        if any(r <= 0.0 for r in self.ratios):
            raise ValueError(f"All split_ratios must be > 0.0, got {self.ratios}")
        if abs(sum(self.ratios) - 1.0) > 1e-6:
            raise ValueError(f"split_ratios must sum to 1.0, got {sum(self.ratios)}")
        if self.method not in ("random", "composition"):
            raise ValueError(f"split_method must be 'random' or 'composition', got '{self.method}'")


def partition_keys_random(
    keys: list[str],
    config: SplitConfig,
) -> dict[str, list[str]]:
    """Partition keys into train/val/test by random shuffle with seed.

    Keys are sorted lexicographically first for deterministic results
    regardless of input order.
    """
    sorted_keys = sorted(keys)
    rng = random.Random(config.seed)
    rng.shuffle(sorted_keys)

    n = len(sorted_keys)
    train_end = int(n * config.ratios[0])
    val_end = train_end + int(n * config.ratios[1])

    return {
        "train": sorted_keys[:train_end],
        "val": sorted_keys[train_end:val_end],
        "test": sorted_keys[val_end:],
    }


def assign_formula_to_split(
    formula: str,
    ratios: tuple[float, float, float],
    seed: int,
    split_names: tuple[str, ...] = SPLIT_NAMES,
) -> str:
    """Hash a single formula to a split name deterministically.

    Uses SHA-256 of "{formula}_{seed}" mapped to [0, 1) via modular
    arithmetic, then assigned to a split via cumulative ratio thresholds.
    """
    hash_input = f"{formula}_{seed}"
    hash_val = int(hashlib.sha256(hash_input.encode()).hexdigest(), 16) % 10000 / 10000.0
    cumulative = 0.0
    for i, ratio in enumerate(ratios):
        cumulative += ratio
        if hash_val < cumulative:
            return split_names[i]
    return split_names[-1]


def partition_keys_by_composition(
    keys: list[str],
    formula_map: dict[str, str],
    config: SplitConfig,
) -> dict[str, list[str]]:
    """Partition keys by molecular formula — all molecules with the same
    formula land in the same split.

    Uses deterministic hash-based assignment: each unique formula is hashed
    with the seed to produce a float in [0, 1), which is mapped to a split
    via cumulative ratio thresholds.
    """
    # Group keys by formula
    formula_to_keys: dict[str, list[str]] = {}
    missing_formulas = []
    for key in keys:
        formula = formula_map.get(key)
        if formula is None:
            missing_formulas.append(key)
            continue
        formula_to_keys.setdefault(formula, []).append(key)

    if missing_formulas:
        print(f"Warning: {len(missing_formulas)} keys missing from formula_map, assigning to train")

    # Assign each formula to a split via hash
    result: dict[str, list[str]] = {name: [] for name in SPLIT_NAMES}
    for formula, formula_keys in formula_to_keys.items():
        split = assign_formula_to_split(formula, config.ratios, config.seed)
        result[split].extend(formula_keys)

    # Assign missing-formula keys to train
    result["train"].extend(missing_formulas)

    return result


def build_formula_map_from_structure_lmdb(
    structure_lmdb_path: str,
) -> dict[str, str]:
    """Read a structure LMDB and return {key: molecular_formula} mapping.

    Extracts formula from the pymatgen MoleculeGraph stored in each entry.
    """
    env = lmdb.open(
        structure_lmdb_path,
        subdir=False,
        readonly=True,
        lock=False,
        readahead=True,
        meminit=False,
    )

    formula_map = {}
    with env.begin(write=False) as txn:
        cursor = txn.cursor()
        for key_bytes, value_bytes in cursor:
            key_str = key_bytes.decode("ascii")
            if key_str in METADATA_KEYS or key_str in ("length", "scaled"):
                continue
            try:
                value = pickle.loads(value_bytes)
                mol_graph = value["molecule_graph"]
                formula = mol_graph.molecule.composition.formula.replace(" ", "")
                formula_map[key_str] = formula
            except Exception as e:
                print(f"Warning: could not extract formula for key '{key_str}': {e}")

    env.close()
    return formula_map


def partition_lmdb_into_splits(
    src_path: str,
    out_dir: str,
    base_name: str,
    config: SplitConfig,
    formula_map: dict[str, str] | None = None,
) -> dict[str, str]:
    """Split a graph LMDB into train/val/test LMDB files.

    Args:
        src_path: Path to the source (intermediate) graph LMDB file.
        out_dir: Directory to write split LMDB files.
        base_name: Base name for output files (e.g. "general_graphs").
        config: SplitConfig with method, ratios, and seed.
        formula_map: Required when config.method == "composition".

    Returns:
        Dict mapping split names to their LMDB file paths.
    """
    if config.method == "composition" and formula_map is None:
        raise ValueError("formula_map is required for composition-based splitting")

    # Read all non-metadata keys from source LMDB
    env_src = lmdb.open(
        src_path,
        subdir=False,
        readonly=True,
        lock=False,
        readahead=True,
        meminit=False,
    )

    skip_keys = {"length", "scaled", "split_name", "scaler_finalized"}
    all_keys = []
    with env_src.begin(write=False) as txn:
        cursor = txn.cursor()
        for key_bytes, _ in cursor:
            key_str = key_bytes.decode("ascii")
            if key_str not in skip_keys:
                all_keys.append(key_str)

    print(f"Total keys to split: {len(all_keys)}")

    # Partition keys
    if config.method == "random":
        partitions = partition_keys_random(all_keys, config)
    else:
        partitions = partition_keys_by_composition(all_keys, formula_map, config)

    # Print split sizes
    for name in SPLIT_NAMES:
        count = len(partitions[name])
        pct = count / len(all_keys) * 100 if all_keys else 0
        print(f"  {name}: {count} ({pct:.1f}%)")

    # Write each split LMDB
    split_paths = {}
    for split_name in SPLIT_NAMES:
        split_keys = set(partitions[split_name])
        if not split_keys:
            continue

        split_file = os.path.join(out_dir, f"{base_name}_{split_name}.lmdb")
        # Remove existing file to avoid stale data
        if os.path.exists(split_file):
            os.remove(split_file)
        lock_file = split_file + "-lock"
        if os.path.exists(lock_file):
            os.remove(lock_file)

        env_out = lmdb.open(
            split_file,
            map_size=int(1099511627776 * 2),
            subdir=False,
            meminit=False,
            map_async=True,
        )

        count = 0
        with env_src.begin(write=False) as txn_in:
            with env_out.begin(write=True) as txn_out:
                cursor = txn_in.cursor()
                for key_bytes, value_bytes in cursor:
                    key_str = key_bytes.decode("ascii")
                    if key_str in split_keys:
                        txn_out.put(key_bytes, value_bytes)
                        count += 1

        # Write metadata
        with env_out.begin(write=True) as txn:
            txn.put(b"length", pickle.dumps(count, protocol=-1))
            txn.put(b"scaled", pickle.dumps(False, protocol=-1))
            txn.put(b"split_name", pickle.dumps(split_name, protocol=-1))

        env_out.sync()
        env_out.close()
        split_paths[split_name] = split_file

    env_src.close()
    return split_paths
