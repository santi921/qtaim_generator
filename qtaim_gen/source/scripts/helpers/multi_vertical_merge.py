"""
Multi-vertical converter-merge pipeline.

Merges multiple dataset verticals into per-vertical train/val/test graph LMDBs
with composition-stratified splitting and train-only scaler fitting.

Each (vertical, split) is built as a *directory* of shard LMDBs
(`{output_dir}/{vertical}/{split}/shard_{i}.lmdb`). qtaim_embed's
LMDBMoleculeDataset consumes a directory of shards directly, so no merge step
is needed -- point the trainer's `src` at the split directory.

Usage:
    multi-vertical-merge --config pipeline.json
"""

import argparse
import multiprocessing
import os
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from glob import glob

import lmdb

from qtaim_gen.source.utils.multi_vertical import (
    MultiVerticalPipelineConfig,
    SplitPlan,
    load_pipeline_config,
    plan_phase,
)
from qtaim_gen.source.utils.splits import SPLIT_NAMES


def _lmdb_is_complete(lmdb_path: str) -> bool:
    """Check if an LMDB exists and has a valid length metadata key."""
    if not os.path.exists(lmdb_path):
        return False
    try:
        env = lmdb.open(
            lmdb_path, subdir=False, readonly=True, lock=False,
            readahead=False, meminit=False,
        )
        with env.begin(write=False) as txn:
            val = txn.get(b"length")
        env.close()
        return val is not None
    except Exception:
        return False


def _lmdb_is_scaled(lmdb_path: str) -> bool:
    """Check if an LMDB has scaled=True metadata."""
    if not os.path.exists(lmdb_path):
        return False
    try:
        env = lmdb.open(
            lmdb_path, subdir=False, readonly=True, lock=False,
            readahead=False, meminit=False,
        )
        with env.begin(write=False) as txn:
            val = txn.get(b"scaled")
        env.close()
        if val is None:
            return False
        return pickle.loads(val) is True
    except Exception:
        return False


def _shards_in(split_dir: str) -> list[str]:
    """Sorted shard LMDB paths in a split directory."""
    if not os.path.isdir(split_dir):
        return []
    return sorted(glob(os.path.join(split_dir, "shard_*.lmdb")))


# ---------- Phase 2: Build ----------

def _build_shard_job(
    vertical_name: str,
    split_name: str,
    shard_index: int,
    keys: list,
    converter_config: dict,
    split_dir: str,
    global_element_set: list,
) -> str:
    """Build one unscaled graph shard LMDB. Runs in a spawned subprocess.

    `keys` is this shard's disjoint key slice (parent pre-sliced), so the
    converter runs unsharded (total_shards=1) and filters via include_keys.
    Returns the output shard LMDB path.
    """
    from qtaim_gen.source.core.converter import GeneralConverter

    lmdb_name = f"shard_{shard_index}.lmdb"
    cfg = deepcopy(converter_config)
    cfg["lmdb_path"] = split_dir
    cfg["lmdb_name"] = lmdb_name
    cfg["total_shards"] = 1
    cfg["shard_index"] = 0
    cfg["auto_merge"] = False
    cfg["restart"] = False  # wipe any partial shard and rebuild cleanly
    cfg["skip_scaling"] = True
    cfg["save_scaler"] = False
    cfg["save_unfinalized_scaler"] = False
    cfg["include_keys"] = keys
    cfg["element_set"] = global_element_set

    print(f"[Build] {vertical_name}/{split_name}/shard_{shard_index}: "
          f"{len(keys)} keys -> {split_dir}")
    converter = GeneralConverter(cfg)
    converter.process()
    return os.path.join(split_dir, lmdb_name)


def build_phase(
    plan: SplitPlan,
    config: MultiVerticalPipelineConfig,
) -> dict[str, dict[str, str]]:
    """Phase 2: Build unscaled graph shard directories for each (vertical, split).

    Returns {vertical_name: {split_name: split_dir}}.
    """
    output_paths: dict[str, dict[str, str]] = {}

    # Collect shard jobs, skipping already-complete shards (resume).
    jobs = []  # (vertical, split, shard_index, slice_keys, split_dir)
    for vert in config.verticals:
        vert_dir = os.path.join(config.output_dir, vert.name)
        output_paths[vert.name] = {}
        for split_name in SPLIT_NAMES:
            split_dir = os.path.join(vert_dir, split_name)
            output_paths[vert.name][split_name] = split_dir

            keys = plan.assignment[vert.name][split_name]
            if not keys:
                print(f"[Build] Skipping {vert.name}/{split_name} (no keys assigned)")
                continue

            n_eff = min(config.n_shards_per_split, len(keys))
            os.makedirs(split_dir, exist_ok=True)
            for i in range(n_eff):
                shard_path = os.path.join(split_dir, f"shard_{i}.lmdb")
                if _lmdb_is_complete(shard_path):
                    print(f"[Build] Skipping {vert.name}/{split_name}/shard_{i} (complete)")
                    continue
                jobs.append((vert.name, split_name, i, keys[i::n_eff], split_dir))

    if not jobs:
        print("[Build] All shards already complete, skipping Phase 2")
        return output_paths

    print(f"[Build] Launching {len(jobs)} shard jobs "
          f"(max_workers={config.build_max_workers})...")
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=config.build_max_workers, mp_context=ctx
    ) as ex:
        futures = {}
        for vname, sname, i, slice_keys, split_dir in jobs:
            fut = ex.submit(
                _build_shard_job,
                vname, sname, i, slice_keys,
                plan.converter_configs[vname], split_dir, plan.global_element_set,
            )
            futures[fut] = (vname, sname, i)
        for fut in as_completed(futures):
            vname, sname, i = futures[fut]
            fut.result()  # re-raise any worker exception loudly
            print(f"[Build] Done {vname}/{sname}/shard_{i}")

    return output_paths


# ---------- Phase 3: Scale ----------

_SCALE_SKIP_KEYS = {"length", "scaled", "split_name", "scaler_finalized"}


def _scale_shard_job(
    shard_path: str,
    feature_scaler_path: str,
    label_scaler_path: str,
    skip_keys: set,
) -> int:
    """Load saved scalers and apply them to one shard. Runs in a subprocess."""
    from qtaim_embed.data.processing import HeteroGraphStandardScalerIterative
    from qtaim_gen.source.utils.scaling import apply_scalers_to_lmdb_inplace

    # finalized=True: __init__ restores fields from disk but then clobbers the
    # finalized flag with the constructor arg (default False); our saved scalers
    # are always finalized, and __call__ asserts finalized.
    feature_scaler = HeteroGraphStandardScalerIterative(
        features_tf=True, load=True, load_path=feature_scaler_path, finalized=True
    )
    label_scaler = HeteroGraphStandardScalerIterative(
        features_tf=False, load=True, load_path=label_scaler_path, finalized=True
    )
    return apply_scalers_to_lmdb_inplace(
        shard_path, feature_scaler, label_scaler, skip_keys
    )


def scale_phase(
    output_paths: dict[str, dict[str, str]],
    config: MultiVerticalPipelineConfig,
) -> None:
    """Phase 3: Fit scaler on all train shards, apply to every shard in parallel."""
    from qtaim_gen.source.utils.scaling import fit_scalers_on_lmdbs, save_scalers

    feat_scaler_path = os.path.join(config.output_dir, "feature_scaler_iterative.pt")
    label_scaler_path = os.path.join(config.output_dir, "label_scaler_iterative.pt")

    # Fit (or reuse saved scalers).
    if os.path.exists(feat_scaler_path) and os.path.exists(label_scaler_path):
        print("[Scale] Using existing scalers (delete .pt files to re-fit)")
    else:
        train_shards = []
        for split_paths in output_paths.values():
            train_shards.extend(_shards_in(split_paths.get("train", "")))
        if not train_shards:
            raise RuntimeError("No train shards found for scaler fitting")
        print(f"[Scale] Fitting scalers on {len(train_shards)} train shards...")
        feature_scaler, label_scaler = fit_scalers_on_lmdbs(
            train_shards, _SCALE_SKIP_KEYS
        )
        save_scalers(feature_scaler, label_scaler, config.output_dir)

    # Collect shards to apply, skipping already-scaled (resume).
    apply_jobs = []  # (vertical, split, shard_path)
    for vname, split_paths in output_paths.items():
        for split_name in SPLIT_NAMES:
            for shard_path in _shards_in(split_paths.get(split_name, "")):
                if _lmdb_is_scaled(shard_path):
                    print(f"[Scale] Skipping {vname}/{split_name}/"
                          f"{os.path.basename(shard_path)} (already scaled)")
                    continue
                apply_jobs.append((vname, split_name, shard_path))

    if not apply_jobs:
        print("[Scale] All shards already scaled")
        return

    print(f"[Scale] Applying scalers to {len(apply_jobs)} shards "
          f"(max_workers={config.build_max_workers})...")
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=config.build_max_workers, mp_context=ctx
    ) as ex:
        futures = {}
        for vname, sname, shard_path in apply_jobs:
            fut = ex.submit(
                _scale_shard_job,
                shard_path, feat_scaler_path, label_scaler_path, _SCALE_SKIP_KEYS,
            )
            futures[fut] = (vname, sname, shard_path)
        for fut in as_completed(futures):
            vname, sname, shard_path = futures[fut]
            count = fut.result()  # re-raise any worker exception loudly
            print(f"[Scale] Scaled {count} graphs in {vname}/{sname}/"
                  f"{os.path.basename(shard_path)}")


# ---------- Main ----------

def main():
    parser = argparse.ArgumentParser(
        description="Multi-vertical converter-merge pipeline",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the multi-vertical pipeline config JSON",
    )
    parser.add_argument(
        "--skip_scaling",
        action="store_true",
        help="Skip Phase 3 (scaler fitting and application)",
    )
    args = parser.parse_args()

    # Load and validate pipeline config
    print("=== Multi-Vertical Merge Pipeline ===\n")
    config = load_pipeline_config(args.config)
    print(f"Output: {config.output_dir}")
    print(f"Verticals: {[v.name for v in config.verticals]}")
    print(f"Split: {config.split_config.method} {config.split_config.ratios} "
          f"seed={config.split_config.seed}")
    print(f"Shards/split: {config.n_shards_per_split}  "
          f"build_max_workers: {config.build_max_workers}")
    print()

    # Phase 1: Plan
    print("--- Phase 1: Plan ---")
    plan = plan_phase(config)
    print()

    # Phase 2: Build
    print("--- Phase 2: Build ---")
    output_paths = build_phase(plan, config)
    print()

    # Phase 3: Scale
    if not args.skip_scaling:
        print("--- Phase 3: Scale ---")
        scale_phase(output_paths, config)
    else:
        print("--- Phase 3: Scale (skipped) ---")

    print("\n=== Pipeline complete ===")
    print(f"Output directory: {config.output_dir}")
    for vert in config.verticals:
        for s in SPLIT_NAMES:
            split_dir = os.path.join(config.output_dir, vert.name, s)
            shards = _shards_in(split_dir)
            status = f"{len(shards)} shard(s)" if shards else "MISSING"
            print(f"  {vert.name}/{s}/: {status}")


if __name__ == "__main__":
    main()
