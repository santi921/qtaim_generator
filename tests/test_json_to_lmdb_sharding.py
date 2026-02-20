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
        result_path, merged_count = merge_shards(shard_paths, merged_path, self.logger)

        assert result_path == merged_path, "merge_shards should return the output path"
        assert merged_count == len(expected_data), \
            f"merge_shards should return correct count, got {merged_count}"
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
        result_path, merged_count = merge_shards(shard_paths, merged_path, self.logger)

        assert os.path.exists(result_path), "Merged LMDB should still be created"
        assert merged_count == 1, "Should have merged 1 entry from existing shard"

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


class TestShardedVsNonSharded:
    """End-to-end test: sharded conversion + merge must produce
    the same LMDB contents as a single non-sharded conversion.

    This catches bugs like the glob-cleanup pattern that was
    deleting merged shard files (charge_shard_0.lmdb matched by
    charge_*.lmdb cleanup).
    """

    base_tests = Path(__file__).parent
    dir_data = str(base_tests / "test_files" / "lmdb_tests") + os.sep

    # Data types that have JSON files in the test fixtures
    JSON_TYPES = ["charge", "bond", "qtaim", "fuzzy_full", "other"]

    @classmethod
    def setup_class(cls):
        cls.temp_dir = tempfile.mkdtemp()
        cls.logger = logging.getLogger("test_sharded_vs_nonsharded")
        cls.logger.setLevel(logging.DEBUG)
        if not cls.logger.handlers:
            handler = logging.StreamHandler()
            handler.setLevel(logging.DEBUG)
            cls.logger.addHandler(handler)

        # Discover the 4 job folders (orca5, orca5_rks, orca5_uks, orca6_rks)
        all_folders = sorted(
            os.path.basename(os.path.normpath(f))
            for f in glob(os.path.join(cls.dir_data, "*/"))
            if os.path.basename(os.path.normpath(f)) not in
               ("generator_lmdbs", "generator_lmdbs_merged", "qtaim")
        )
        assert len(all_folders) == 4, f"Expected 4 job folders, got {all_folders}"
        cls.all_folders = all_folders

    @classmethod
    def teardown_class(cls):
        if hasattr(cls, "temp_dir") and os.path.exists(cls.temp_dir):
            shutil.rmtree(cls.temp_dir)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _nan_equal(a, b) -> bool:
        """Deep equality that treats NaN == NaN as True."""
        import math
        if isinstance(a, float) and isinstance(b, float):
            if math.isnan(a) and math.isnan(b):
                return True
            return a == b
        if isinstance(a, dict) and isinstance(b, dict):
            if a.keys() != b.keys():
                return False
            return all(
                TestShardedVsNonSharded._nan_equal(a[k], b[k]) for k in a
            )
        if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
            if len(a) != len(b):
                return False
            return all(
                TestShardedVsNonSharded._nan_equal(x, y) for x, y in zip(a, b)
            )
        return a == b

    @staticmethod
    def _read_lmdb(path: str) -> dict:
        """Read all (key -> deserialized value) pairs from an LMDB file,
        excluding the 'length' meta-key."""
        env = lmdb.open(path, subdir=False, readonly=True, lock=False)
        data = {}
        with env.begin() as txn:
            for key, value in txn.cursor():
                k = key.decode("ascii")
                if k != "length":
                    data[k] = pkl.loads(value)
        env.close()
        return data

    @staticmethod
    def _read_lmdb_length(path: str) -> int:
        """Read the 'length' meta-key from an LMDB."""
        env = lmdb.open(path, subdir=False, readonly=True, lock=False)
        with env.begin() as txn:
            raw = txn.get("length".encode())
            length = pkl.loads(raw) if raw else 0
        env.close()
        return length

    @staticmethod
    def _make_symlinked_root(dest_root: str, src_root: str, folders: list):
        """Create *dest_root* with symlinks to only the listed folders
        from *src_root*."""
        os.makedirs(dest_root, exist_ok=True)
        for folder in folders:
            src = os.path.join(src_root, folder)
            dst = os.path.join(dest_root, folder)
            if not os.path.exists(dst):
                os.symlink(src, dst)

    # ------------------------------------------------------------------
    # tests
    # ------------------------------------------------------------------
    def test_sharded_vs_nonsharded_json_types(self):
        """For every JSON data type, sharded (2-shard) + merge must match
        non-sharded conversion."""
        from qtaim_gen.source.utils.lmdbs import json_2_lmdbs

        total_shards = 2
        chunk_size = 2  # small chunk to exercise chunking + merging

        for data_type in self.JSON_TYPES:
            # --- non-sharded reference ---
            noshard_dir = os.path.join(self.temp_dir, f"noshard_{data_type}") + os.sep
            os.makedirs(noshard_dir, exist_ok=True)
            noshard_lmdb = f"{data_type}.lmdb"

            json_2_lmdbs(
                root_dir=self.dir_data,
                out_dir=noshard_dir,
                data_type=data_type,
                out_lmdb=noshard_lmdb,
                chunk_size=chunk_size,
                clean=True,
                merge=True,
            )
            noshard_path = os.path.join(noshard_dir, noshard_lmdb)
            assert os.path.exists(noshard_path), \
                f"Non-sharded LMDB not created for {data_type}"

            # --- sharded conversion (2 shards) ---
            # Partition folders: even-indexed → shard 0, odd-indexed → shard 1
            shard_assignments = [[], []]
            for i, folder in enumerate(self.all_folders):
                shard_assignments[i % total_shards].append(folder)

            shard_lmdb_paths = []
            for shard_idx in range(total_shards):
                # Create a temp root dir with symlinks to only this shard's folders
                shard_root = os.path.join(
                    self.temp_dir, f"shard_root_{data_type}_{shard_idx}"
                ) + os.sep
                self._make_symlinked_root(
                    shard_root, self.dir_data, shard_assignments[shard_idx]
                )

                shard_out = os.path.join(
                    self.temp_dir, f"shard_out_{data_type}", f"shard_{shard_idx}"
                ) + os.sep
                os.makedirs(shard_out, exist_ok=True)

                shard_lmdb_name = f"{data_type}_shard_{shard_idx}.lmdb"
                json_2_lmdbs(
                    root_dir=shard_root,
                    out_dir=shard_out,
                    data_type=data_type,
                    out_lmdb=shard_lmdb_name,
                    chunk_size=chunk_size,
                    clean=True,
                    merge=True,
                )

                shard_path = os.path.join(shard_out, shard_lmdb_name)
                assert os.path.exists(shard_path), \
                    f"Shard {shard_idx} LMDB not created for {data_type}"
                shard_lmdb_paths.append(shard_path)

            # --- merge shards ---
            merged_path = os.path.join(
                self.temp_dir, f"merged_{data_type}", f"{data_type}.lmdb"
            )
            os.makedirs(os.path.dirname(merged_path), exist_ok=True)
            merge_shards(shard_lmdb_paths, merged_path, self.logger)

            # --- compare ---
            ref_data = self._read_lmdb(noshard_path)
            merged_data = self._read_lmdb(merged_path)

            # Same key sets
            assert set(ref_data.keys()) == set(merged_data.keys()), (
                f"{data_type}: key mismatch.\n"
                f"  Non-sharded keys: {sorted(ref_data.keys())}\n"
                f"  Merged keys:      {sorted(merged_data.keys())}"
            )

            # Same values (NaN-aware comparison)
            for key in ref_data:
                assert self._nan_equal(ref_data[key], merged_data[key]), (
                    f"{data_type}: value mismatch for key '{key}'"
                )

            # Same length metadata
            ref_len = self._read_lmdb_length(noshard_path)
            merged_len = self._read_lmdb_length(merged_path)
            assert ref_len == merged_len, (
                f"{data_type}: length mismatch: "
                f"non-sharded={ref_len}, merged={merged_len}"
            )

    def test_sharded_vs_nonsharded_structure(self):
        """Structure (inp_files_2_lmdbs) sharded + merge must match
        non-sharded conversion."""
        from qtaim_gen.source.utils.lmdbs import inp_files_2_lmdbs

        total_shards = 2
        chunk_size = 2

        # --- non-sharded ---
        noshard_dir = os.path.join(self.temp_dir, "noshard_structure") + os.sep
        os.makedirs(noshard_dir, exist_ok=True)
        noshard_lmdb = "geom.lmdb"

        inp_files_2_lmdbs(
            root_dir=self.dir_data,
            out_dir=noshard_dir,
            out_lmdb=noshard_lmdb,
            chunk_size=chunk_size,
            clean=True,
            merge=True,
        )
        noshard_path = os.path.join(noshard_dir, noshard_lmdb)
        assert os.path.exists(noshard_path), "Non-sharded structure LMDB not created"

        # --- sharded ---
        shard_assignments = [[], []]
        for i, folder in enumerate(self.all_folders):
            shard_assignments[i % total_shards].append(folder)

        shard_lmdb_paths = []
        for shard_idx in range(total_shards):
            shard_root = os.path.join(
                self.temp_dir, f"shard_root_structure_{shard_idx}"
            ) + os.sep
            self._make_symlinked_root(
                shard_root, self.dir_data, shard_assignments[shard_idx]
            )

            shard_out = os.path.join(
                self.temp_dir, "shard_out_structure", f"shard_{shard_idx}"
            ) + os.sep
            os.makedirs(shard_out, exist_ok=True)

            shard_lmdb_name = f"geom_shard_{shard_idx}.lmdb"
            inp_files_2_lmdbs(
                root_dir=shard_root,
                out_dir=shard_out,
                out_lmdb=shard_lmdb_name,
                chunk_size=chunk_size,
                clean=True,
                merge=True,
            )

            shard_path = os.path.join(shard_out, shard_lmdb_name)
            assert os.path.exists(shard_path), \
                f"Shard {shard_idx} structure LMDB not created"
            shard_lmdb_paths.append(shard_path)

        # --- merge ---
        merged_path = os.path.join(
            self.temp_dir, "merged_structure", "geom.lmdb"
        )
        os.makedirs(os.path.dirname(merged_path), exist_ok=True)
        merge_shards(shard_lmdb_paths, merged_path, self.logger)

        # --- compare keys and length ---
        ref_data = self._read_lmdb(noshard_path)
        merged_data = self._read_lmdb(merged_path)

        assert set(ref_data.keys()) == set(merged_data.keys()), (
            f"Structure: key mismatch.\n"
            f"  Non-sharded keys: {sorted(ref_data.keys())}\n"
            f"  Merged keys:      {sorted(merged_data.keys())}"
        )

        ref_len = self._read_lmdb_length(noshard_path)
        merged_len = self._read_lmdb_length(merged_path)
        assert ref_len == merged_len, (
            f"Structure: length mismatch: "
            f"non-sharded={ref_len}, merged={merged_len}"
        )

    def test_shard_cleanup_preserves_merged_file(self):
        """Regression test: chunk cleanup glob must not delete the merged
        shard file.  e.g. charge_[0-9]*.lmdb must NOT match
        charge_shard_0.lmdb."""
        from qtaim_gen.source.utils.lmdbs import json_2_lmdbs

        data_type = "charge"
        shard_idx = 0
        shard_out = os.path.join(
            self.temp_dir, "cleanup_regression", f"shard_{shard_idx}"
        ) + os.sep
        os.makedirs(shard_out, exist_ok=True)

        shard_lmdb_name = f"{data_type}_shard_{shard_idx}.lmdb"

        # Write a dummy file that looks like the merged shard output.
        # If cleanup incorrectly deletes it, the assertion below will fail.
        dummy_shard_path = os.path.join(shard_out, shard_lmdb_name)
        env = lmdb.open(dummy_shard_path, subdir=False, map_size=1024**2, lock=False)
        with env.begin(write=True) as txn:
            txn.put(b"sentinel", pkl.dumps("should_survive"))
            txn.put(b"length", pkl.dumps(1))
        env.close()

        # Now run json_2_lmdbs which creates chunks and then cleans them up.
        # The merged shard file (charge_shard_0.lmdb) must survive cleanup.
        json_2_lmdbs(
            root_dir=self.dir_data,
            out_dir=shard_out,
            data_type=data_type,
            out_lmdb=f"{data_type}_final.lmdb",
            chunk_size=2,
            clean=True,
            merge=True,
        )

        # The dummy shard file must still exist
        assert os.path.exists(dummy_shard_path), (
            f"Cleanup incorrectly deleted {shard_lmdb_name}! "
            f"The [0-9]* glob pattern is not correctly excluding shard files."
        )

        # And the final merged file from this run must also exist
        assert os.path.exists(os.path.join(shard_out, f"{data_type}_final.lmdb")), \
            "Final merged LMDB was not created"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
