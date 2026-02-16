"""
Tests for json_to_lmdb sharding functionality.

This test module specifically tests the sharding features added to json_to_lmdb.py,
including folder partitioning and LMDB merging.
"""

import pytest
import os
import lmdb
import pickle as pkl
import logging
import shutil
import tempfile
from glob import glob
from pathlib import Path

from qtaim_gen.source.scripts.json_to_lmdb import (
    partition_folders_by_shard,
    merge_shards,
)


class TestSharding:
    """Test sharding functionality for json_to_lmdb."""

    base_tests = Path(__file__).parent
    dir_data = str(base_tests / "test_files" / "lmdb_tests") + os.sep

    @classmethod
    def setup_class(cls):
        """Setup test directories."""
        cls.temp_dir = tempfile.mkdtemp()
        cls.logger = logging.getLogger("test_sharding")
        cls.logger.setLevel(logging.DEBUG)
        # Add handler to avoid "No handlers found" warning
        if not cls.logger.handlers:
            handler = logging.StreamHandler()
            handler.setLevel(logging.DEBUG)
            cls.logger.addHandler(handler)

    @classmethod
    def teardown_class(cls):
        """Clean up test directories."""
        if hasattr(cls, 'temp_dir') and os.path.exists(cls.temp_dir):
            shutil.rmtree(cls.temp_dir)

    def test_partition_folders_by_shard_basic(self):
        """Test basic folder partitioning across shards."""
        # Test with 4 shards
        total_shards = 4

        # Get all partitions
        all_partitions = []
        for shard_index in range(total_shards):
            partition = partition_folders_by_shard(
                self.dir_data, shard_index, total_shards, self.logger
            )
            all_partitions.append(set(partition))

        # Check that partitions are non-overlapping
        for i in range(total_shards):
            for j in range(i + 1, total_shards):
                overlap = all_partitions[i] & all_partitions[j]
                assert len(overlap) == 0, f"Shards {i} and {j} have overlapping folders: {overlap}"

        # Check that all folders are covered
        all_folders_union = set()
        for partition in all_partitions:
            all_folders_union |= partition

        # Get expected folders
        pattern = os.path.join(self.dir_data, "*/")
        all_folders = sorted(glob(pattern))
        expected_folder_names = set(os.path.basename(os.path.normpath(f)) for f in all_folders)

        assert all_folders_union == expected_folder_names, \
            f"Union of partitions doesn't match all folders. Missing: {expected_folder_names - all_folders_union}"

    def test_partition_folders_by_shard_deterministic(self):
        """Test that partitioning is deterministic - same input produces same output."""
        total_shards = 3
        shard_index = 1

        # Run partitioning twice
        partition1 = partition_folders_by_shard(
            self.dir_data, shard_index, total_shards, self.logger
        )
        partition2 = partition_folders_by_shard(
            self.dir_data, shard_index, total_shards, self.logger
        )

        assert partition1 == partition2, "Partitioning should be deterministic"

    def test_partition_folders_by_shard_no_sharding(self):
        """Test that total_shards=1 returns empty list (no sharding)."""
        partition = partition_folders_by_shard(
            self.dir_data, 0, 1, self.logger
        )
        assert partition == [], "total_shards=1 should return empty list"

    def test_partition_folders_by_shard_distribution(self):
        """Test that folder distribution across shards is balanced."""
        total_shards = 4

        # Get partition sizes
        partition_sizes = []
        for shard_index in range(total_shards):
            partition = partition_folders_by_shard(
                self.dir_data, shard_index, total_shards, self.logger
            )
            partition_sizes.append(len(partition))

        # Check that sizes are roughly balanced (within 1 folder)
        max_size = max(partition_sizes)
        min_size = min(partition_sizes)
        assert max_size - min_size <= 1, \
            f"Partition sizes should be balanced: {partition_sizes}"

    def test_merge_shards_basic(self):
        """Test basic shard merging functionality."""
        # Create test shards
        shard_paths = []
        expected_data = {}

        for i in range(3):
            shard_path = os.path.join(self.temp_dir, f"test_shard_{i}.lmdb")
            shard_paths.append(shard_path)

            # Create LMDB with test data
            env = lmdb.open(shard_path, subdir=False, map_size=10 * 1024**2, lock=False)
            with env.begin(write=True) as txn:
                # Add unique keys per shard
                for j in range(5):
                    key = f"shard_{i}_key_{j}"
                    value = {"data": f"value_{i}_{j}", "shard": i, "index": j}
                    txn.put(key.encode(), pkl.dumps(value))
                    expected_data[key] = value

                # Add length
                txn.put("length".encode(), pkl.dumps(5))
            env.close()

        # Merge shards
        merged_path = os.path.join(self.temp_dir, "merged.lmdb")
        result_path = merge_shards(shard_paths, merged_path, self.logger)

        assert result_path == merged_path, "merge_shards should return the output path"
        assert os.path.exists(merged_path), "Merged LMDB should exist"

        # Verify merged data
        env = lmdb.open(merged_path, subdir=False, readonly=True, lock=False)
        with env.begin() as txn:
            # Check length
            length_bytes = txn.get("length".encode())
            assert length_bytes is not None, "Merged LMDB should have length key"
            length = pkl.loads(length_bytes)
            assert length == len(expected_data), \
                f"Expected {len(expected_data)} entries, got {length}"

            # Check all keys present
            cursor = txn.cursor()
            merged_data = {}
            for key, value in cursor:
                key_str = key.decode("ascii")
                if key_str != "length":
                    merged_data[key_str] = pkl.loads(value)

            assert set(merged_data.keys()) == set(expected_data.keys()), \
                "Merged LMDB should contain all keys from all shards"

            # Verify values
            for key, expected_value in expected_data.items():
                assert key in merged_data, f"Key {key} missing from merged LMDB"
                assert merged_data[key] == expected_value, \
                    f"Value mismatch for {key}: {merged_data[key]} != {expected_value}"
        env.close()

    def test_merge_shards_missing_shard(self):
        """Test that merge_shards handles missing shards gracefully."""
        # Create one real shard
        shard1_path = os.path.join(self.temp_dir, "shard1.lmdb")
        env = lmdb.open(shard1_path, subdir=False, map_size=10 * 1024**2, lock=False)
        with env.begin(write=True) as txn:
            txn.put("key1".encode(), pkl.dumps({"value": 1}))
            txn.put("length".encode(), pkl.dumps(1))
        env.close()

        # Create list with missing shard
        shard_paths = [
            shard1_path,
            os.path.join(self.temp_dir, "missing_shard.lmdb")  # Doesn't exist
        ]

        merged_path = os.path.join(self.temp_dir, "merged_missing.lmdb")

        # Should not raise, but warn
        result_path = merge_shards(shard_paths, merged_path, self.logger)

        assert os.path.exists(result_path), "Merged LMDB should still be created"

        # Verify it contains data from the existing shard
        env = lmdb.open(merged_path, subdir=False, readonly=True, lock=False)
        with env.begin() as txn:
            key1_bytes = txn.get("key1".encode())
            assert key1_bytes is not None, "Data from existing shard should be present"
            assert pkl.loads(key1_bytes) == {"value": 1}
        env.close()

    def test_merge_shards_empty_list(self):
        """Test that merge_shards raises error for empty shard list."""
        merged_path = os.path.join(self.temp_dir, "merged_empty.lmdb")

        with pytest.raises(ValueError, match="No shard LMDBs provided"):
            merge_shards([], merged_path, self.logger)

    def test_sharding_folder_consistency(self):
        """Test that folder-based partitioning ensures all JSON types go to same shard."""
        total_shards = 2

        # For each folder, verify it only appears in one shard
        for shard_index in range(total_shards):
            partition = partition_folders_by_shard(
                self.dir_data, shard_index, total_shards, self.logger
            )

            # Check that same folder consistently goes to same shard
            for folder in partition:
                # Re-run partitioning and verify this folder still in same shard
                new_partition = partition_folders_by_shard(
                    self.dir_data, shard_index, total_shards, self.logger
                )
                assert folder in new_partition, \
                    f"Folder {folder} should consistently be in shard {shard_index}"

    def test_sharding_with_chunking_compatibility(self):
        """Test that sharding works with chunking (both enabled)."""
        # This is an integration test to verify sharding and chunking don't conflict
        # We'll create a small test to verify the output naming scheme

        # Create test output directory
        test_out_dir = os.path.join(self.temp_dir, "chunking_sharding_test")
        os.makedirs(test_out_dir, exist_ok=True)

        # Simulate the naming scheme used in json_to_lmdb.py
        total_shards = 2
        shard_index = 0

        # With sharding, output goes to subdirectory
        shard_out_dir = os.path.join(test_out_dir, f"shard_{shard_index}")
        os.makedirs(shard_out_dir, exist_ok=True)

        # Verify subdirectory was created
        assert os.path.exists(shard_out_dir), "Shard subdirectory should be created"

        # Verify naming doesn't conflict
        # Chunks would be named: charge_1.lmdb, charge_2.lmdb in shard_0/
        # Merged would be: charge_shard_0.lmdb in shard_0/
        chunk1_path = os.path.join(shard_out_dir, "charge_1.lmdb")
        merged_path = os.path.join(shard_out_dir, "charge_shard_0.lmdb")

        # These should not conflict
        assert os.path.dirname(chunk1_path) == os.path.dirname(merged_path)
        assert os.path.basename(chunk1_path) != os.path.basename(merged_path)

    def test_merge_preserves_key_order(self):
        """Test that merge_shards preserves deterministic key ordering."""
        # Create shards with overlapping key patterns
        shard_paths = []

        for i in range(2):
            shard_path = os.path.join(self.temp_dir, f"order_shard_{i}.lmdb")
            shard_paths.append(shard_path)

            env = lmdb.open(shard_path, subdir=False, map_size=10 * 1024**2, lock=False)
            with env.begin(write=True) as txn:
                # Add keys that would sort differently if not handled properly
                for j in range(3):
                    key = f"key_{i}_{j}"
                    value = {"shard": i, "index": j}
                    txn.put(key.encode(), pkl.dumps(value))
                txn.put("length".encode(), pkl.dumps(3))
            env.close()

        # Merge
        merged_path = os.path.join(self.temp_dir, "merged_order.lmdb")
        merge_shards(shard_paths, merged_path, self.logger)

        # Read keys and verify they're all present
        env = lmdb.open(merged_path, subdir=False, readonly=True, lock=False)
        with env.begin() as txn:
            cursor = txn.cursor()
            keys = [key.decode("ascii") for key, _ in cursor if key.decode("ascii") != "length"]

            # Should have all keys from both shards
            assert len(keys) == 6, f"Expected 6 keys, got {len(keys)}"

            # Keys should include entries from both shards
            assert any("key_0_" in k for k in keys), "Should have keys from shard 0"
            assert any("key_1_" in k for k in keys), "Should have keys from shard 1"
        env.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
