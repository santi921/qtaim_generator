"""
Reusable scaler fitting, application, and saving for graph LMDBs.

Extracted from generator_to_embed.scale_split_lmdbs() so both the
single-vertical (generator-to-embed --split) and multi-vertical
(multi-vertical-merge) pipelines can share the same logic.

Graphs are stored as ``pickle.dumps({"molecule_graph": <serialized bytes>})``
(see converter.py and qtaim_embed.data.lmdb). The serialized bytes must be
unwrapped from that dict before calling load_graph_from_serialized, and
re-wrapped on write so the trainer's LMDBMoleculeDataset can read them.
"""

import os
import pickle

import lmdb

from qtaim_embed.data.processing import HeteroGraphStandardScalerIterative
from qtaim_embed.data.lmdb import (
    load_graph_from_serialized,
    serialize_graph,
)

LMDB_MAP_SIZE: int = 1099511627776 * 2  # 2 TiB

# Non-graph metadata keys (bytes) written by the converter / split / qtaim_embed.
# These are passed through verbatim and never deserialized as graphs.
_METADATA_KEYS: frozenset = frozenset({
    b"length", b"length_chunk", b"scaled", b"scaler_finalized",
    b"processed_source_keys", b"feature_names", b"feature_size",
    b"target_dict", b"element_set", b"allowed_ring_size",
    b"allowed_charges", b"allowed_spins", b"extra_dataset_info",
    b"log_scale_features", b"split_name",
})


def _is_metadata(key_bytes: bytes, skip_keys: set) -> bool:
    """True if a key is non-graph metadata (skip during fit, copy during apply)."""
    if key_bytes in _METADATA_KEYS:
        return True
    return key_bytes.decode("ascii") in skip_keys


def _deserialize_graph(value_bytes: bytes):
    """Unwrap the ``{"molecule_graph": ...}`` payload and deserialize the graph."""
    raw = pickle.loads(value_bytes)
    serialized = raw["molecule_graph"] if isinstance(raw, dict) else raw
    return load_graph_from_serialized(serialized)


def fit_scalers_on_lmdbs(
    train_lmdb_paths: list[str],
    skip_keys: set[str],
) -> tuple[HeteroGraphStandardScalerIterative, HeteroGraphStandardScalerIterative]:
    """Create fresh scalers and fit on train LMDBs only (streaming).

    Args:
        train_lmdb_paths: Paths to one or more train split LMDB files.
        skip_keys: Extra metadata key strings to skip (in addition to the
            built-in metadata key set).

    Returns:
        Tuple of (feature_scaler, label_scaler), both finalized.

    Raises:
        RuntimeError: if any graph fails to load (a real format bug, not
            normal flow -- the previous bare-except masked exactly this).
    """
    feature_scaler = HeteroGraphStandardScalerIterative(
        features_tf=True, mean={}, std={}
    )
    label_scaler = HeteroGraphStandardScalerIterative(
        features_tf=False, mean={}, std={}
    )

    total_fit = 0
    failures: list[str] = []
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
            for key_bytes, value_bytes in txn.cursor():
                if _is_metadata(key_bytes, skip_keys):
                    continue
                try:
                    graph = _deserialize_graph(value_bytes)
                except Exception as e:  # noqa: BLE001 - reported and re-raised below
                    failures.append(f"{key_bytes!r}: {e}")
                    continue
                feature_scaler.update([graph])
                label_scaler.update([graph])
                total_fit += 1
        env.close()

    if failures:
        raise RuntimeError(
            f"{len(failures)} graph(s) failed to load during scaler fitting "
            f"(first: {failures[0]}). This indicates a serialization-format "
            f"mismatch, not a skippable error."
        )

    feature_scaler.finalize()
    label_scaler.finalize()
    print(f"Scalers fitted on {total_fit} train graphs")
    return feature_scaler, label_scaler


def apply_scalers_to_lmdb_inplace(
    lmdb_path: str,
    feature_scaler: HeteroGraphStandardScalerIterative,
    label_scaler: HeteroGraphStandardScalerIterative,
    skip_keys: set[str],
    batch_size: int = 2000,
) -> int:
    """Apply scalers to all graphs in an LMDB, replacing it atomically.

    Streams the source read-only and writes the scaled result to a temp LMDB
    in batches, then ``os.replace``s it over the original. The source is never
    mutated until the atomic replace, so a crash leaves the original intact
    (still ``scaled=False``) and a re-run reprocesses cleanly -- no
    double-scaling, no poisoned LMDB.

    Args:
        lmdb_path: Path to the graph LMDB file.
        feature_scaler: Finalized feature scaler.
        label_scaler: Finalized label scaler.
        skip_keys: Extra metadata key strings to copy through unscaled.
        batch_size: Graphs scaled+written per temp-LMDB transaction.

    Returns:
        Count of scaled graphs.
    """
    tmp_path = lmdb_path + ".scaling.tmp"
    # Clean any stale temp from a prior crash.
    for p in (tmp_path, tmp_path + "-lock"):
        if os.path.exists(p):
            os.remove(p)

    src = lmdb.open(
        lmdb_path, subdir=False, readonly=True, lock=False,
        readahead=True, meminit=False,
    )
    dst = lmdb.open(
        tmp_path, map_size=LMDB_MAP_SIZE, subdir=False,
        meminit=False, map_async=True,
    )

    scaled_count = 0
    meta: list[tuple[bytes, bytes]] = []
    buf: list[tuple[bytes, bytes]] = []

    def _flush():
        if not buf:
            return
        with dst.begin(write=True) as wtxn:
            for k, v in buf:
                wtxn.put(k, v)
        buf.clear()

    # src and dst are separate envs, so holding the source read cursor open
    # while writing dst in batches is safe (no same-env read/write conflict).
    with src.begin(write=False) as rtxn:
        for key_bytes, value_bytes in rtxn.cursor():
            if _is_metadata(key_bytes, skip_keys):
                meta.append((key_bytes, value_bytes))
                continue
            graph = _deserialize_graph(value_bytes)
            scaled = feature_scaler([graph])
            label_scaler(scaled)
            buf.append((
                key_bytes,
                pickle.dumps(
                    {"molecule_graph": serialize_graph(scaled[0], ret=True)},
                    protocol=-1,
                ),
            ))
            scaled_count += 1
            if len(buf) >= batch_size:
                _flush()
        _flush()

    # Copy metadata through, then mark scaled (override any copied scaled flag).
    with dst.begin(write=True) as wtxn:
        for k, v in meta:
            if k == b"scaled":
                continue
            wtxn.put(k, v)
        wtxn.put(b"scaled", pickle.dumps(True, protocol=-1))

    dst.sync()
    dst.close()
    src.close()

    os.replace(tmp_path, lmdb_path)
    stale_lock = tmp_path + "-lock"
    if os.path.exists(stale_lock):
        os.remove(stale_lock)
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
