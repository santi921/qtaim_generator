"""
Manual run harness for converter configs (NOT a pytest suite).

Runs real converter jobs against the machine-local data paths inside each
config. Hermetic pytest coverage of these configs lives in
tests/test_converter_configs.py.

Usage:
    python run_configs.py --config base_unsharded
    python run_configs.py --config qtaim_unsharded
    python run_configs.py --config general_fuzzy_bonds
    python run_configs.py --test-sharding  # Test sharded workflow
"""

import os
import json
import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from qtaim_gen.source.core.converter import BaseConverter, QTAIMConverter, GeneralConverter


def load_config(config_name):
    """Load a config file."""
    config_dir = Path(__file__).parent
    config_path = config_dir / f"{config_name}.json"

    if not config_path.exists():
        raise ValueError(f"Config not found: {config_path}")

    with open(config_path) as f:
        config = json.load(f)

    return config, str(config_path)


def run_base_converter(config_name="base_unsharded"):
    """Test BaseConverter with a config."""
    print(f"\n{'='*60}")
    print(f"Testing BaseConverter with config: {config_name}")
    print(f"{'='*60}\n")

    config, config_path = load_config(config_name)

    print("Config:")
    print(json.dumps(config, indent=2))
    print()

    conv = BaseConverter(config, config_path=config_path)
    result = conv.process(return_info=True)

    print(f"\n✓ BaseConverter processing complete!")
    print(f"  Output: {conv.file}")
    print(f"  Graphs processed: {result.get('n_graphs', 'N/A')}")

    return conv


def run_qtaim_converter(config_name="qtaim_unsharded"):
    """Test QTAIMConverter with a config."""
    print(f"\n{'='*60}")
    print(f"Testing QTAIMConverter with config: {config_name}")
    print(f"{'='*60}\n")

    config, config_path = load_config(config_name)

    print("Config:")
    print(json.dumps(config, indent=2))
    print()

    conv = QTAIMConverter(config, config_path=config_path)
    result = conv.process(return_info=True)

    print(f"\n✓ QTAIMConverter processing complete!")
    print(f"  Output: {conv.file}")
    print(f"  Graphs processed: {result.get('n_graphs', 'N/A')}")

    return conv


def run_general_converter(config_name="general_fuzzy_bonds"):
    """Test GeneralConverter with a config."""
    print(f"\n{'='*60}")
    print(f"Testing GeneralConverter with config: {config_name}")
    print(f"{'='*60}\n")

    config, config_path = load_config(config_name)

    print("Config:")
    print(json.dumps(config, indent=2))
    print()

    conv = GeneralConverter(config, config_path=config_path)
    result = conv.process(return_info=True)

    print(f"\n✓ GeneralConverter processing complete!")
    print(f"  Output: {conv.file}")
    print(f"  Graphs processed: {result.get('n_graphs', 'N/A')}")

    return conv


def run_sharded_workflow():
    """Test the full sharded workflow: shard -> merge."""
    print(f"\n{'='*60}")
    print(f"Testing Sharded Workflow")
    print(f"{'='*60}\n")

    # Process both shards
    shard_dirs = []
    for i in range(2):
        config_name = f"base_sharded_shard{i}"
        print(f"\n--- Processing Shard {i} ---")
        config, config_path = load_config(config_name)
        conv = BaseConverter(config, config_path=config_path)
        conv.process(return_info=True)
        shard_dirs.append(config["lmdb_path"])
        print(f"✓ Shard {i} complete: {conv.file}")

    # Merge shards
    print(f"\n--- Merging Shards ---")
    output_dir = "/home/santiagovargas/dev/qtaim_generator/data/output_graphs/base_merged"

    merged_path = BaseConverter.merge_shards(
        shard_dirs=shard_dirs,
        output_dir=output_dir,
        output_name="base_graphs.lmdb",
        skip_scaling=False  # Test the scaling fix
    )

    print(f"\n✓ Merge complete!")
    print(f"  Merged LMDB: {merged_path}")
    print(f"  Feature scaler: {os.path.join(output_dir, 'feature_scaler_iterative.pt')}")
    print(f"  Label scaler: {os.path.join(output_dir, 'label_scaler_iterative.pt')}")

    # Verify merged LMDB
    import lmdb
    import pickle
    env = lmdb.open(merged_path, readonly=True, subdir=False, lock=False)
    with env.begin() as txn:
        stats = txn.stat()
        print(f"\n  Merged LMDB stats:")
        print(f"    Entries: {stats['entries']}")

        # Check scaled flag
        scaled_value = txn.get(b'scaled')
        if scaled_value:
            is_scaled = pickle.loads(scaled_value)
            print(f"    Scaled: {is_scaled}")
    env.close()

    return merged_path


def main():
    parser = argparse.ArgumentParser(description="Test converter configs")
    parser.add_argument(
        "--config",
        type=str,
        help="Config name to test (without .json extension)"
    )
    parser.add_argument(
        "--test-sharding",
        action="store_true",
        help="Test the full sharded workflow"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Test all configs"
    )

    args = parser.parse_args()

    try:
        if args.test_sharding:
            run_sharded_workflow()
        elif args.all:
            # Test all configs
            print("\n" + "="*60)
            print("TESTING ALL CONFIGS")
            print("="*60)

            run_base_converter("base_unsharded")
            run_qtaim_converter("qtaim_unsharded")
            run_general_converter("general_fuzzy_bonds")
            run_general_converter("general_qtaim_bonds")
            run_general_converter("general_with_global_dipoles")
            run_sharded_workflow()

            print("\n" + "="*60)
            print("✓ ALL TESTS PASSED!")
            print("="*60)
        elif args.config:
            # Detect converter type from config name
            if "qtaim" in args.config and "general" not in args.config:
                run_qtaim_converter(args.config)
            elif "general" in args.config:
                run_general_converter(args.config)
            else:
                run_base_converter(args.config)
        else:
            parser.print_help()
            print("\nAvailable configs:")
            config_dir = Path(__file__).parent
            for config_file in sorted(config_dir.glob("*.json")):
                print(f"  - {config_file.stem}")

    except Exception as e:
        print(f"\n✗ Test failed with error:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
