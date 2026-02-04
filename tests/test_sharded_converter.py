"""
Integration tests for sharded converter implementation.

These tests verify that:
1. Sharded processing produces same output as single-run processing
2. Merged scalers match single-run scalers
3. The full shard -> merge workflow works correctly
"""

import os
import shutil
import tempfile
import pytest
import torch
from pathlib import Path

from qtaim_gen.source.core.converter import BaseConverter


def _base_config(tmp_path, lmdb_location, lmdb_name="graphs_test.lmdb"):
    """Create base config for testing."""
    return {
        "chunk": -1,
        "filter_list": ["length"],
        "restart": False,
        "allowed_ring_size": [3, 4, 5, 6, 7, 8],
        "allowed_charges": None,
        "allowed_spins": None,
        "keys_target": {"atom": [], "bond": [], "global": ["n_atoms"]},
        "keys_data": {"atom": [], "bond": [], "global": ["n_atoms"]},
        "lmdb_path": str(tmp_path),
        "lmdb_name": lmdb_name,
        "lmdb_locations": {"geom_lmdb": lmdb_location},
        "n_workers": 1,
        "batch_size": 100,
    }


@pytest.mark.skipif(
    not os.path.exists(os.path.join(os.path.dirname(__file__), "test_files", "lmdb_tests")),
    reason="Test fixtures not available"
)
def test_sharded_vs_single_run(tmp_path):
    """
    Test that sharded processing produces same results as single-run.

    This is the critical integration test - verifies that partitioning,
    processing, and merging produces identical output to non-sharded run.
    """
    base_tests = os.path.dirname(__file__)
    merged_folder = os.path.join(
        base_tests, "test_files", "lmdb_tests", "generator_lmdbs_merged"
    )
    geom_lmdb = os.path.join(merged_folder, "merged_geom.lmdb")

    if not os.path.exists(geom_lmdb):
        pytest.skip("Test LMDB fixtures not available")

    # Run 1: Single converter (ground truth)
    single_dir = tmp_path / "single"
    single_dir.mkdir()
    config_single = _base_config(single_dir, geom_lmdb, "single.lmdb")

    conv_single = BaseConverter(
        config_single,
        config_path=os.path.join(str(single_dir), "config.json")
    )
    conv_single.process(return_info=True)

    # Get single-run scaler stats
    single_mean = {k: v.clone() for k, v in conv_single.feature_scaler_iterative._mean.items()}
    single_sizes = dict(conv_single.feature_scaler_iterative.dict_node_sizes)

    # Run 2: Sharded converters (2 shards)
    shard_dirs = []
    for shard_idx in range(2):
        shard_dir = tmp_path / f"shard_{shard_idx}"
        shard_dir.mkdir()

        config_shard = _base_config(shard_dir, geom_lmdb, "graphs.lmdb")
        config_shard["shard_index"] = shard_idx
        config_shard["total_shards"] = 2
        config_shard["skip_scaling"] = True
        config_shard["save_unfinalized_scaler"] = True

        conv_shard = BaseConverter(
            config_shard,
            config_path=os.path.join(str(shard_dir), "config.json")
        )
        conv_shard.process(return_info=True)
        shard_dirs.append(str(shard_dir))

    # Run 3: Merge shards
    merged_dir = tmp_path / "merged"
    merged_dir.mkdir()

    merged_path = BaseConverter.merge_shards(
        shard_dirs=shard_dirs,
        output_dir=str(merged_dir),
        output_name="merged.lmdb",
        skip_scaling=True  # Skip scaling for now to just test merging
    )

    # Verify merge completed
    assert os.path.exists(merged_path), "Merged LMDB should exist"
    assert os.path.exists(
        os.path.join(str(merged_dir), "feature_scaler_iterative.pt")
    ), "Merged feature scaler should exist"

    # Load merged scalers
    from qtaim_embed.data.processing import HeteroGraphStandardScalerIterative

    merged_feature_scaler = HeteroGraphStandardScalerIterative(
        features_tf=True,
        load=True,
        load_path=os.path.join(str(merged_dir), "feature_scaler_iterative.pt")
    )

    # Verify merged scaler stats match single-run stats
    for nt in single_mean.keys():
        assert torch.allclose(
            merged_feature_scaler._mean[nt],
            single_mean[nt],
            atol=1e-5
        ), f"Mean mismatch for {nt}"

        assert merged_feature_scaler.dict_node_sizes[nt] == single_sizes[nt], \
            f"Size mismatch for {nt}: {merged_feature_scaler.dict_node_sizes[nt]} vs {single_sizes[nt]}"


@pytest.mark.skipif(
    not os.path.exists(os.path.join(os.path.dirname(__file__), "test_files", "lmdb_tests")),
    reason="Test fixtures not available"
)
def test_sharding_partition_logic(tmp_path):
    """
    Test that key partitioning is deterministic and complete.

    Verifies that:
    1. All keys are assigned to exactly one shard
    2. Partitioning is deterministic (same result every time)
    3. Shards are roughly balanced
    """
    base_tests = os.path.dirname(__file__)
    merged_folder = os.path.join(
        base_tests, "test_files", "lmdb_tests", "generator_lmdbs_merged"
    )
    geom_lmdb = os.path.join(merged_folder, "merged_geom.lmdb")

    if not os.path.exists(geom_lmdb):
        pytest.skip("Test LMDB fixtures not available")

    # Create a base converter to get all keys
    test_dir = tmp_path / "test"
    test_dir.mkdir()
    config = _base_config(test_dir, geom_lmdb)
    conv = BaseConverter(config, config_path=os.path.join(str(test_dir), "config.json"))
    all_keys = conv.lmdb_dict["geom_lmdb"]["keys"]
    total_keys = len(all_keys)

    # Test partitioning with 3 shards by creating separate converters
    n_shards = 3
    partitioned_keys = []

    for shard_idx in range(n_shards):
        shard_dir = tmp_path / f"shard_{shard_idx}"
        shard_dir.mkdir()

        config_shard = _base_config(shard_dir, geom_lmdb)
        config_shard["shard_index"] = shard_idx
        config_shard["total_shards"] = n_shards

        conv_shard = BaseConverter(config_shard, config_path=os.path.join(str(shard_dir), "config.json"))
        shard_keys = conv_shard._partition_keys(all_keys)
        partitioned_keys.append(set(shard_keys))

    # Verify all keys assigned
    all_partitioned = set().union(*partitioned_keys)
    assert len(all_partitioned) == total_keys, "All keys should be assigned"

    # Verify no overlaps
    for i in range(n_shards):
        for j in range(i + 1, n_shards):
            overlap = partitioned_keys[i] & partitioned_keys[j]
            assert len(overlap) == 0, f"Shards {i} and {j} should not overlap"

    # Verify roughly balanced (within 20% for large datasets)
    # For very small datasets, just verify all shards have at least one key
    if total_keys >= 20:
        expected_per_shard = total_keys / n_shards
        for i, keys in enumerate(partitioned_keys):
            ratio = len(keys) / expected_per_shard
            assert 0.8 <= ratio <= 1.2, f"Shard {i} imbalanced: {len(keys)} keys (expected ~{expected_per_shard})"
    else:
        # Small datasets: just verify all shards got some keys
        for i, keys in enumerate(partitioned_keys):
            assert len(keys) > 0, f"Shard {i} should have at least one key"


def test_merge_shards_with_missing_scalers(tmp_path):
    """
    Test merge_shards handles missing scaler files gracefully.
    """
    # Create fake shard directories without scalers
    shard_dirs = []
    for i in range(2):
        shard_dir = tmp_path / f"shard_{i}"
        shard_dir.mkdir()

        # Create a dummy LMDB (empty)
        import lmdb
        env = lmdb.open(str(shard_dir / "graphs.lmdb"), subdir=False, map_size=1024*1024)
        env.close()

        shard_dirs.append(str(shard_dir))

    merged_dir = tmp_path / "merged"
    merged_dir.mkdir()

    # Should not crash, just warn about missing scalers
    merged_path = BaseConverter.merge_shards(
        shard_dirs=shard_dirs,
        output_dir=str(merged_dir),
        skip_scaling=True
    )

    assert os.path.exists(merged_path), "Merge should complete despite missing scalers"


@pytest.mark.skipif(
    not os.path.exists(os.path.join(os.path.dirname(__file__), "test_files", "lmdb_tests")),
    reason="Test fixtures not available"
)
def test_sharded_config_generation(tmp_path):
    """
    Test that sharded configs are generated correctly.
    """
    base_tests = os.path.dirname(__file__)
    merged_folder = os.path.join(
        base_tests, "test_files", "lmdb_tests", "generator_lmdbs_merged"
    )
    geom_lmdb = os.path.join(merged_folder, "merged_geom.lmdb")

    if not os.path.exists(geom_lmdb):
        pytest.skip("Test LMDB fixtures not available")

    shard_dir = tmp_path / "shard_0"
    shard_dir.mkdir()

    config = _base_config(shard_dir, geom_lmdb, "graphs.lmdb")
    config["shard_index"] = 0
    config["total_shards"] = 4
    config["skip_scaling"] = True
    config["save_unfinalized_scaler"] = True

    conv = BaseConverter(
        config,
        config_path=os.path.join(str(shard_dir), "config.json")
    )

    # Verify sharding parameters
    assert conv.shard_index == 0
    assert conv.total_shards == 4
    assert conv.skip_scaling == True
    assert conv.save_unfinalized_scaler == True

    # Verify output filename includes shard index
    assert "shard_0" in conv.file


def _helper_init_sharding(self, shard_index, total_shards):
    """Helper method for testing partitioning without full converter init."""
    self.shard_index = shard_index
    self.total_shards = total_shards


# Monkey patch for testing
BaseConverter._Converter__init_sharding = _helper_init_sharding


if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        print("Running test_sharded_vs_single_run...")
        try:
            test_sharded_vs_single_run(tmp_path / "test1")
            print("PASSED")
        except Exception as e:
            print(f"SKIPPED or FAILED: {e}")

        print("\nRunning test_sharding_partition_logic...")
        try:
            test_sharding_partition_logic(tmp_path / "test2")
            print("PASSED")
        except Exception as e:
            print(f"SKIPPED or FAILED: {e}")

        print("\nRunning test_merge_shards_with_missing_scalers...")
        try:
            test_merge_shards_with_missing_scalers(tmp_path / "test3")
            print("PASSED")
        except Exception as e:
            print(f"FAILED: {e}")

        print("\nRunning test_sharded_config_generation...")
        try:
            test_sharded_config_generation(tmp_path / "test4")
            print("PASSED")
        except Exception as e:
            print(f"SKIPPED or FAILED: {e}")

        print("\nAll sharded converter tests complete!")
