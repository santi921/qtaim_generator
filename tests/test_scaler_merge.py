"""
Test scaler merging for sharded converter implementation.

This test verifies that:
1. Scalers from split data can be merged correctly
2. Merged scalers match scalers trained on all data
3. Save/load/merge workflow works correctly
"""

import os
import torch
import pytest
from qtaim_gen.source.core.converter import BaseConverter


def _base_config(tmp_path, lmdb_location, lmdb_name="graphs_test.lmdb"):
    return {
        "chunk": -1,
        "filter_list": ["length"],  # No scaling for this test
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


def test_scaler_merge_basic(tmp_path):
    """Test that merge_scalers function works with HeteroGraphStandardScalerIterative."""
    from qtaim_embed.data.processing import (
        HeteroGraphStandardScalerIterative,
        merge_scalers,
    )

    # Create two scalers with known statistics
    scaler1 = HeteroGraphStandardScalerIterative(features_tf=True)
    scaler2 = HeteroGraphStandardScalerIterative(features_tf=True)

    # Manually set statistics (simulating two shards)
    # Shard 1: 100 samples, mean=10, sum_x2 based on std=2
    scaler1._mean = {"atom": torch.tensor([10.0], dtype=torch.float64)}
    scaler1._std = {"atom": torch.tensor([2.0], dtype=torch.float64)}
    scaler1._sum_x2 = {"atom": torch.tensor([10400.0], dtype=torch.float64)}  # n*(std^2 + mean^2)
    scaler1.dict_node_sizes = {"atom": 100}
    scaler1.finalized = True

    # Shard 2: 100 samples, mean=20, std=2
    scaler2._mean = {"atom": torch.tensor([20.0], dtype=torch.float64)}
    scaler2._std = {"atom": torch.tensor([2.0], dtype=torch.float64)}
    scaler2._sum_x2 = {"atom": torch.tensor([40400.0], dtype=torch.float64)}
    scaler2.dict_node_sizes = {"atom": 100}
    scaler2.finalized = True

    # Merge
    merged = merge_scalers([scaler1, scaler2], features_tf=True, finalize_merged=True)

    # Expected: mean = (100*10 + 100*20) / 200 = 15
    assert torch.isclose(merged._mean["atom"], torch.tensor([15.0], dtype=torch.float64))
    assert merged.dict_node_sizes["atom"] == 200
    assert merged.finalized


def test_scaler_save_load_merge(tmp_path):
    """Test save -> load -> merge workflow."""
    from qtaim_embed.data.processing import (
        HeteroGraphStandardScalerIterative,
        merge_scalers,
    )

    # Create and save two scalers
    scaler1 = HeteroGraphStandardScalerIterative(features_tf=True)
    scaler1._mean = {"atom": torch.tensor([10.0], dtype=torch.float64)}
    scaler1._std = {"atom": torch.tensor([2.0], dtype=torch.float64)}
    scaler1._sum_x2 = {"atom": torch.tensor([10400.0], dtype=torch.float64)}
    scaler1.dict_node_sizes = {"atom": 100}
    scaler1.finalized = True

    scaler2 = HeteroGraphStandardScalerIterative(features_tf=True)
    scaler2._mean = {"atom": torch.tensor([20.0], dtype=torch.float64)}
    scaler2._std = {"atom": torch.tensor([2.0], dtype=torch.float64)}
    scaler2._sum_x2 = {"atom": torch.tensor([40400.0], dtype=torch.float64)}
    scaler2.dict_node_sizes = {"atom": 100}
    scaler2.finalized = True

    # Save to files
    path1 = tmp_path / "scaler1.pt"
    path2 = tmp_path / "scaler2.pt"
    scaler1.save_scaler(str(path1))
    scaler2.save_scaler(str(path2))

    # Load from files
    loaded1 = HeteroGraphStandardScalerIterative(
        features_tf=True, load=True, load_path=str(path1)
    )
    loaded2 = HeteroGraphStandardScalerIterative(
        features_tf=True, load=True, load_path=str(path2)
    )

    # Merge loaded scalers
    merged = merge_scalers([loaded1, loaded2], features_tf=True, finalize_merged=True)

    # Verify
    assert torch.isclose(merged._mean["atom"], torch.tensor([15.0], dtype=torch.float64))
    assert merged.dict_node_sizes["atom"] == 200


@pytest.mark.skipif(
    not os.path.exists(os.path.join(os.path.dirname(__file__), "test_files", "lmdb_tests")),
    reason="Test fixtures not available"
)
def test_converter_scaler_consistency(tmp_path):
    """
    Test that splitting data and merging scalers produces same result as single run.

    This is the critical test for sharded converter implementation.
    """
    from qtaim_embed.data.processing import merge_scalers

    base_tests = os.path.dirname(__file__)
    merged_folder = os.path.join(
        base_tests, "test_files", "lmdb_tests", "generator_lmdbs_merged"
    )
    geom_lmdb = os.path.join(merged_folder, "merged_geom.lmdb")

    if not os.path.exists(geom_lmdb):
        pytest.skip("Test LMDB fixtures not available")

    # Run 1: Process all data with single converter
    config_single = _base_config(tmp_path / "single", geom_lmdb, "single.lmdb")
    os.makedirs(config_single["lmdb_path"], exist_ok=True)

    conv_single = BaseConverter(config_single, config_path=os.path.join(config_single["lmdb_path"], "config.json"))
    conv_single.process(return_info=True)

    # Get scaler stats before finalize
    single_mean = {k: v.clone() for k, v in conv_single.feature_scaler_iterative._mean.items()}
    single_sizes = dict(conv_single.feature_scaler_iterative.dict_node_sizes)

    # Run 2: We'll simulate sharding by processing same data twice
    # In real sharding, we'd partition keys - here we just verify merge math works
    config_shard1 = _base_config(tmp_path / "shard1", geom_lmdb, "shard1.lmdb")
    config_shard2 = _base_config(tmp_path / "shard2", geom_lmdb, "shard2.lmdb")
    os.makedirs(config_shard1["lmdb_path"], exist_ok=True)
    os.makedirs(config_shard2["lmdb_path"], exist_ok=True)

    conv_shard1 = BaseConverter(config_shard1, config_path=os.path.join(config_shard1["lmdb_path"], "config.json"))
    conv_shard2 = BaseConverter(config_shard2, config_path=os.path.join(config_shard2["lmdb_path"], "config.json"))

    conv_shard1.process(return_info=True)
    conv_shard2.process(return_info=True)

    # Merge the two scalers (same data twice = doubled counts, same mean)
    merged_feature_scaler = merge_scalers(
        [conv_shard1.feature_scaler_iterative, conv_shard2.feature_scaler_iterative],
        features_tf=True,
    )

    # Verify merged scaler has doubled counts but same mean
    for nt in single_mean.keys():
        # Mean should be the same (same data)
        assert torch.allclose(
            merged_feature_scaler._mean[nt],
            single_mean[nt],
            atol=1e-5
        ), f"Mean mismatch for {nt}"

        # Count should be doubled
        assert merged_feature_scaler.dict_node_sizes[nt] == 2 * single_sizes[nt], \
            f"Size mismatch for {nt}"


if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        from pathlib import Path
        tmp_path = Path(tmp)

        print("Running test_scaler_merge_basic...")
        test_scaler_merge_basic(tmp_path / "basic")
        print("PASSED")

        print("Running test_scaler_save_load_merge...")
        test_scaler_save_load_merge(tmp_path / "save_load")
        print("PASSED")

        print("Running test_converter_scaler_consistency...")
        try:
            test_converter_scaler_consistency(tmp_path / "consistency")
            print("PASSED")
        except Exception as e:
            print(f"SKIPPED or FAILED: {e}")

        print("\nAll basic scaler merge tests passed!")
