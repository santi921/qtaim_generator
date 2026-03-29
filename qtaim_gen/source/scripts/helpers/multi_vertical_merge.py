"""
Multi-vertical converter-merge pipeline.

Merges multiple dataset verticals into per-vertical train/val/test graph LMDBs
with composition-stratified splitting and train-only scaler fitting.

Usage:
    multi-vertical-merge --config pipeline.json
"""

import argparse
import json
import os
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy

import lmdb

from qtaim_gen.source.utils.lmdbs import parse_config_gen_to_embed
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


# ---------- Phase 2: Build ----------

def _build_single_job(
    vertical_name: str,
    split_name: str,
    keys: list[str],
    converter_config: dict,
    output_dir: str,
    global_element_set: list[int],
) -> str:
    """Build unscaled graph LMDB for one (vertical, split) pair.

    Runs in a subprocess via ProcessPoolExecutor.
    Returns the output LMDB path.
    """
    from qtaim_gen.source.core.converter import GeneralConverter

    vert_dir = os.path.join(output_dir, vertical_name)
    os.makedirs(vert_dir, exist_ok=True)
    lmdb_name = f"{split_name}.lmdb"
    out_path = os.path.join(vert_dir, lmdb_name)

    # Inject overrides into a copy of the config
    cfg = deepcopy(converter_config)
    cfg["lmdb_path"] = vert_dir
    cfg["lmdb_name"] = lmdb_name
    cfg["skip_scaling"] = True
    cfg["save_scaler"] = False
    cfg["save_unfinalized_scaler"] = False
    cfg["include_keys"] = keys
    cfg["element_set"] = global_element_set

    print(f"[Build] {vertical_name}/{split_name}: {len(keys)} keys -> {out_path}")
    converter = GeneralConverter(cfg)
    converter.process()
    return out_path


def build_phase(
    plan: SplitPlan,
    config: MultiVerticalPipelineConfig,
) -> dict[str, dict[str, str]]:
    """Phase 2: Build unscaled graph LMDBs for each (vertical, split).

    Returns {vertical_name: {split_name: lmdb_path}}.
    """
    output_paths: dict[str, dict[str, str]] = {}

    # Collect all jobs, skipping already-complete ones
    jobs = []
    for vert in config.verticals:
        vert_paths: dict[str, str] = {}
        for split_name in SPLIT_NAMES:
            vert_dir = os.path.join(config.output_dir, vert.name)
            out_path = os.path.join(vert_dir, f"{split_name}.lmdb")
            vert_paths[split_name] = out_path

            if _lmdb_is_complete(out_path):
                print(f"[Build] Skipping {vert.name}/{split_name} (already complete)")
                continue

            keys = plan.assignment[vert.name][split_name]
            if not keys:
                print(f"[Build] Skipping {vert.name}/{split_name} (no keys assigned)")
                continue

            jobs.append((vert.name, split_name, keys))

        output_paths[vert.name] = vert_paths

    if not jobs:
        print("[Build] All graph LMDBs already complete, skipping Phase 2")
        return output_paths

    print(f"[Build] Launching {len(jobs)} converter jobs...")

    # Run jobs — use sequential execution to avoid issues with
    # converter's internal threading (each converter already uses
    # ThreadPoolExecutor internally for parallel graph construction)
    for vert_name, split_name, keys in jobs:
        _build_single_job(
            vertical_name=vert_name,
            split_name=split_name,
            keys=keys,
            converter_config=plan.converter_configs[vert_name],
            output_dir=config.output_dir,
            global_element_set=plan.global_element_set,
        )

    return output_paths


# ---------- Phase 3: Scale ----------

def scale_phase(
    output_paths: dict[str, dict[str, str]],
    config: MultiVerticalPipelineConfig,
) -> None:
    """Phase 3: Fit scaler on all train LMDBs, apply to all splits."""
    from qtaim_gen.source.utils.scaling import (
        apply_scalers_to_lmdb_inplace,
        fit_scalers_on_lmdbs,
        save_scalers,
    )

    skip_keys = {"length", "scaled", "split_name", "scaler_finalized"}

    # Check if scalers already exist
    feat_scaler_path = os.path.join(config.output_dir, "feature_scaler_iterative.pt")
    label_scaler_path = os.path.join(config.output_dir, "label_scaler_iterative.pt")

    if os.path.exists(feat_scaler_path) and os.path.exists(label_scaler_path):
        # Scalers exist — load them instead of re-fitting
        from qtaim_embed.data.processing import HeteroGraphStandardScalerIterative
        print("[Scale] Loading existing scalers (delete .pt files to re-fit)")
        feature_scaler = HeteroGraphStandardScalerIterative(
            features_tf=True, mean={}, std={}
        )
        label_scaler = HeteroGraphStandardScalerIterative(
            features_tf=False, mean={}, std={}
        )
        feature_scaler.load_scaler(feat_scaler_path)
        label_scaler.load_scaler(label_scaler_path)
    else:
        # Collect train LMDB paths
        train_paths = []
        for vert_name, split_paths in output_paths.items():
            train_path = split_paths.get("train")
            if train_path and os.path.exists(train_path):
                train_paths.append(train_path)

        if not train_paths:
            raise RuntimeError("No train LMDBs found for scaler fitting")

        print(f"[Scale] Fitting scalers on {len(train_paths)} train LMDBs...")
        feature_scaler, label_scaler = fit_scalers_on_lmdbs(train_paths, skip_keys)
        save_scalers(feature_scaler, label_scaler, config.output_dir)

    # Apply to all split LMDBs
    for vert_name, split_paths in output_paths.items():
        for split_name, lmdb_path in split_paths.items():
            if not os.path.exists(lmdb_path):
                continue
            if _lmdb_is_scaled(lmdb_path):
                print(f"[Scale] Skipping {vert_name}/{split_name} (already scaled)")
                continue
            print(f"[Scale] Applying scalers to {vert_name}/{split_name}...")
            count = apply_scalers_to_lmdb_inplace(
                lmdb_path, feature_scaler, label_scaler, skip_keys
            )
            print(f"  Scaled {count} graphs")


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
        vert_dir = os.path.join(config.output_dir, vert.name)
        for s in SPLIT_NAMES:
            p = os.path.join(vert_dir, f"{s}.lmdb")
            exists = "ok" if os.path.exists(p) else "MISSING"
            print(f"  {vert.name}/{s}.lmdb: {exists}")


if __name__ == "__main__":
    main()
