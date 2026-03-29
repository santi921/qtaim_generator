"""
Reusable scaler fitting, application, and saving for graph LMDBs.

Extracted from generator_to_embed.scale_split_lmdbs() so both the
single-vertical (generator-to-embed --split) and multi-vertical
(multi-vertical-merge) pipelines can share the same logic.
"""

import os
import pickle
from typing import Set

import lmdb

from qtaim_embed.data.processing import HeteroGraphStandardScalerIterative
from qtaim_embed.data.lmdb import (
    load_graph_from_serialized,
    serialize_graph,
)

LMDB_MAP_SIZE: int = 1099511627776 * 2  # 2 TiB


def fit_scalers_on_lmdbs(
    train_lmdb_paths: list[str],
    skip_keys: set[str],
) -> tuple[HeteroGraphStandardScalerIterative, HeteroGraphStandardScalerIterative]:
    """Create fresh scalers and fit on train LMDBs only (streaming).

    Args:
        train_lmdb_paths: Paths to one or more train split LMDB files.
        skip_keys: Set of metadata key strings to skip (e.g. {"length", "scaled", "split_name"}).

    Returns:
        Tuple of (feature_scaler, label_scaler), both finalized.
    """
    feature_scaler = HeteroGraphStandardScalerIterative(
        features_tf=True, mean={}, std={}
    )
    label_scaler = HeteroGraphStandardScalerIterative(
        features_tf=False, mean={}, std={}
    )

    total_fit = 0
    for train_path in train_lmdb_paths:
        print(f"Fitting scalers on {train_path}...")
        env = lmdb.open(
            train_path,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=True,
            meminit=False,
        )
        with env.begin(write=False) as txn:
            cursor = txn.cursor()
            for key_bytes, value_bytes in cursor:
                key_str = key_bytes.decode("ascii")
                if key_str in skip_keys:
                    continue
                try:
                    graph = load_graph_from_serialized(pickle.loads(value_bytes))
                    feature_scaler.update([graph])
                    label_scaler.update([graph])
                    total_fit += 1
                except Exception as e:
                    print(f"Warning: failed to load graph for key '{key_str}' during scaler fitting: {e}")
        env.close()

    feature_scaler.finalize()
    label_scaler.finalize()
    print(f"Scalers fitted on {total_fit} train graphs")
    return feature_scaler, label_scaler


def apply_scalers_to_lmdb_inplace(
    lmdb_path: str,
    feature_scaler: HeteroGraphStandardScalerIterative,
    label_scaler: HeteroGraphStandardScalerIterative,
    skip_keys: set[str],
) -> int:
    """Apply scalers to all graphs in an LMDB in-place.

    Args:
        lmdb_path: Path to the graph LMDB file.
        feature_scaler: Finalized feature scaler.
        label_scaler: Finalized label scaler.
        skip_keys: Set of metadata key strings to skip.

    Returns:
        Count of scaled graphs.
    """
    db = lmdb.open(
        lmdb_path,
        map_size=LMDB_MAP_SIZE,
        subdir=False,
        meminit=False,
        map_async=True,
    )

    # Read all items first to avoid read-write cursor conflicts
    with db.begin(write=False) as txn:
        items = list(txn.cursor())

    scaled_count = 0
    for key_bytes, value_bytes in items:
        key_str = key_bytes.decode("ascii")
        if key_str in skip_keys:
            continue
        try:
            graph = load_graph_from_serialized(pickle.loads(value_bytes))
            scaled_graphs = feature_scaler([graph])
            label_scaler(scaled_graphs)
            with db.begin(write=True) as txn_w:
                txn_w.put(
                    key_bytes,
                    pickle.dumps(serialize_graph(scaled_graphs[0], ret=True), protocol=-1),
                )
            scaled_count += 1
        except Exception as e:
            print(f"Warning: failed to scale graph for key '{key_str}': {e}")

    # Mark as scaled
    with db.begin(write=True) as txn:
        txn.put(b"scaled", pickle.dumps(True, protocol=-1))

    db.sync()
    db.close()
    return scaled_count


def save_scalers(
    feature_scaler: HeteroGraphStandardScalerIterative,
    label_scaler: HeteroGraphStandardScalerIterative,
    output_dir: str,
) -> None:
    """Save scaler .pt files to the given directory."""
    os.makedirs(output_dir, exist_ok=True)
    feature_scaler.save_scaler(os.path.join(output_dir, "feature_scaler_iterative.pt"))
    label_scaler.save_scaler(os.path.join(output_dir, "label_scaler_iterative.pt"))
    print(f"Saved scaler files to {output_dir}")
