"""Tests for train/val/test splitting utilities."""

import os
import pickle
import tempfile

import lmdb
import pytest

from qtaim_gen.source.utils.splits import (
    SplitConfig,
    partition_keys_random,
    partition_keys_by_composition,
    build_formula_map_from_structure_lmdb,
    partition_lmdb_into_splits,
    SPLIT_NAMES,
)


# ---------------------------------------------------------------------------
# SplitConfig validation
# ---------------------------------------------------------------------------

class TestSplitConfig:
    def test_valid_config(self):
        cfg = SplitConfig(method="random", ratios=(0.8, 0.1, 0.1), seed=42)
        assert cfg.method == "random"
        assert cfg.ratios == (0.8, 0.1, 0.1)

    def test_invalid_method(self):
        with pytest.raises(ValueError, match="split_method"):
            SplitConfig(method="scaffold", ratios=(0.8, 0.1, 0.1))

    def test_ratios_wrong_length(self):
        with pytest.raises(ValueError, match="exactly 3"):
            SplitConfig(method="random", ratios=(0.5, 0.5))

    def test_ratios_zero(self):
        with pytest.raises(ValueError, match="> 0.0"):
            SplitConfig(method="random", ratios=(0.8, 0.2, 0.0))

    def test_ratios_dont_sum_to_one(self):
        with pytest.raises(ValueError, match="sum to 1.0"):
            SplitConfig(method="random", ratios=(0.5, 0.3, 0.1))

    def test_ratios_negative(self):
        with pytest.raises(ValueError, match="> 0.0"):
            SplitConfig(method="random", ratios=(0.9, 0.2, -0.1))


# ---------------------------------------------------------------------------
# partition_keys_random
# ---------------------------------------------------------------------------

class TestPartitionKeysRandom:
    def test_basic_split(self):
        keys = [f"mol_{i}" for i in range(100)]
        cfg = SplitConfig(method="random", ratios=(0.8, 0.1, 0.1), seed=42)
        result = partition_keys_random(keys, cfg)

        assert set(result.keys()) == {"train", "val", "test"}
        assert len(result["train"]) == 80
        assert len(result["val"]) == 10
        assert len(result["test"]) == 10

        # All keys accounted for
        all_assigned = result["train"] + result["val"] + result["test"]
        assert set(all_assigned) == set(keys)

    def test_deterministic_with_same_seed(self):
        keys = [f"mol_{i}" for i in range(50)]
        cfg = SplitConfig(method="random", ratios=(0.7, 0.15, 0.15), seed=123)
        r1 = partition_keys_random(keys, cfg)
        r2 = partition_keys_random(keys, cfg)
        assert r1 == r2

    def test_different_seed_different_result(self):
        keys = [f"mol_{i}" for i in range(50)]
        r1 = partition_keys_random(keys, SplitConfig("random", (0.8, 0.1, 0.1), seed=1))
        r2 = partition_keys_random(keys, SplitConfig("random", (0.8, 0.1, 0.1), seed=2))
        # Extremely unlikely to be the same
        assert r1["train"] != r2["train"]

    def test_deterministic_regardless_of_input_order(self):
        keys_a = [f"mol_{i}" for i in range(30)]
        keys_b = list(reversed(keys_a))
        cfg = SplitConfig(method="random", ratios=(0.8, 0.1, 0.1), seed=42)
        r1 = partition_keys_random(keys_a, cfg)
        r2 = partition_keys_random(keys_b, cfg)
        assert r1 == r2

    def test_small_dataset(self):
        keys = ["a", "b", "c"]
        cfg = SplitConfig(method="random", ratios=(0.6, 0.2, 0.2), seed=42)
        result = partition_keys_random(keys, cfg)
        all_assigned = result["train"] + result["val"] + result["test"]
        assert set(all_assigned) == set(keys)


# ---------------------------------------------------------------------------
# partition_keys_by_composition
# ---------------------------------------------------------------------------

class TestPartitionKeysByComposition:
    def test_same_formula_same_split(self):
        keys = [f"mol_{i}" for i in range(20)]
        formula_map = {f"mol_{i}": "C6H12O6" if i < 10 else "CH4" for i in range(20)}
        cfg = SplitConfig(method="composition", ratios=(0.8, 0.1, 0.1), seed=42)
        result = partition_keys_by_composition(keys, formula_map, cfg)

        # All C6H12O6 molecules should be in the same split
        c6_keys = {f"mol_{i}" for i in range(10)}
        ch4_keys = {f"mol_{i}" for i in range(10, 20)}

        for split_name in SPLIT_NAMES:
            split_set = set(result[split_name])
            # Each formula group must be entirely in one split (or not in this split)
            assert split_set & c6_keys in (set(), c6_keys)
            assert split_set & ch4_keys in (set(), ch4_keys)

    def test_all_keys_assigned(self):
        keys = [f"mol_{i}" for i in range(30)]
        formula_map = {f"mol_{i}": f"F{i % 5}" for i in range(30)}
        cfg = SplitConfig(method="composition", ratios=(0.7, 0.15, 0.15), seed=42)
        result = partition_keys_by_composition(keys, formula_map, cfg)

        all_assigned = set()
        for split_name in SPLIT_NAMES:
            all_assigned.update(result[split_name])
        assert all_assigned == set(keys)

    def test_deterministic(self):
        keys = [f"mol_{i}" for i in range(20)]
        formula_map = {f"mol_{i}": f"F{i % 3}" for i in range(20)}
        cfg = SplitConfig(method="composition", ratios=(0.8, 0.1, 0.1), seed=99)
        r1 = partition_keys_by_composition(keys, formula_map, cfg)
        r2 = partition_keys_by_composition(keys, formula_map, cfg)
        assert r1 == r2

    def test_single_formula(self):
        """All molecules with same formula end up in one split."""
        keys = [f"mol_{i}" for i in range(10)]
        formula_map = {k: "H2O" for k in keys}
        cfg = SplitConfig(method="composition", ratios=(0.8, 0.1, 0.1), seed=42)
        result = partition_keys_by_composition(keys, formula_map, cfg)

        # All in exactly one split
        non_empty = [name for name in SPLIT_NAMES if result[name]]
        assert len(non_empty) == 1
        assert len(result[non_empty[0]]) == 10

    def test_every_formula_unique(self):
        """When every molecule has a unique formula, degenerates to per-molecule assignment."""
        keys = [f"mol_{i}" for i in range(10)]
        formula_map = {f"mol_{i}": f"UNIQUE_{i}" for i in range(10)}
        cfg = SplitConfig(method="composition", ratios=(0.8, 0.1, 0.1), seed=42)
        result = partition_keys_by_composition(keys, formula_map, cfg)

        all_assigned = set()
        for name in SPLIT_NAMES:
            all_assigned.update(result[name])
        assert all_assigned == set(keys)

    def test_missing_formula_keys_go_to_train(self):
        keys = ["a", "b", "c"]
        formula_map = {"a": "H2O"}  # b and c missing
        cfg = SplitConfig(method="composition", ratios=(0.8, 0.1, 0.1), seed=42)
        result = partition_keys_by_composition(keys, formula_map, cfg)
        # b and c should be in train
        assert "b" in result["train"]
        assert "c" in result["train"]


# ---------------------------------------------------------------------------
# build_formula_map_from_structure_lmdb
# ---------------------------------------------------------------------------

class TestBuildFormulaMap:
    def test_with_real_fixture(self):
        fixture = "tests/test_files/lmdb_tests/generator_lmdbs_merged/merged_geom.lmdb"
        if not os.path.exists(fixture):
            pytest.skip("Test fixture not available")

        formula_map = build_formula_map_from_structure_lmdb(fixture)
        assert len(formula_map) == 4
        assert "orca5" in formula_map
        assert "orca5_rks" in formula_map
        # Each formula should be a non-empty string
        for key, formula in formula_map.items():
            assert isinstance(formula, str)
            assert len(formula) > 0


# ---------------------------------------------------------------------------
# partition_lmdb_into_splits (end-to-end LMDB I/O)
# ---------------------------------------------------------------------------

def _create_test_graph_lmdb(path: str, keys: list[str]) -> None:
    """Create a minimal LMDB with fake graph data for testing."""
    env = lmdb.open(
        path,
        map_size=int(1099511627776),
        subdir=False,
        meminit=False,
        map_async=True,
    )
    with env.begin(write=True) as txn:
        for key in keys:
            # Store a simple dict as a stand-in for serialized graph bytes
            txn.put(key.encode("ascii"), pickle.dumps({"fake_graph": key}, protocol=-1))
        txn.put(b"length", pickle.dumps(len(keys), protocol=-1))
        txn.put(b"scaled", pickle.dumps(False, protocol=-1))
    env.sync()
    env.close()


class TestPartitionLmdbIntoSplits:
    def test_random_split_creates_files(self, tmp_path):
        keys = [f"mol_{i}" for i in range(20)]
        src = str(tmp_path / "source.lmdb")
        _create_test_graph_lmdb(src, keys)

        cfg = SplitConfig(method="random", ratios=(0.8, 0.1, 0.1), seed=42)
        split_paths = partition_lmdb_into_splits(
            src_path=src,
            out_dir=str(tmp_path),
            base_name="graphs",
            config=cfg,
        )

        assert set(split_paths.keys()) == {"train", "val", "test"}
        for name, path in split_paths.items():
            assert os.path.exists(path)
            assert name in path

    def test_metadata_keys_correct(self, tmp_path):
        keys = [f"mol_{i}" for i in range(20)]
        src = str(tmp_path / "source.lmdb")
        _create_test_graph_lmdb(src, keys)

        cfg = SplitConfig(method="random", ratios=(0.8, 0.1, 0.1), seed=42)
        split_paths = partition_lmdb_into_splits(
            src_path=src, out_dir=str(tmp_path), base_name="g", config=cfg,
        )

        for name, path in split_paths.items():
            env = lmdb.open(path, subdir=False, readonly=True, lock=False)
            with env.begin() as txn:
                length = pickle.loads(txn.get(b"length"))
                scaled = pickle.loads(txn.get(b"scaled"))
                split_name = pickle.loads(txn.get(b"split_name"))
            env.close()
            assert isinstance(length, int)
            assert length > 0
            assert scaled is False
            assert split_name == name

    def test_all_keys_preserved(self, tmp_path):
        keys = [f"mol_{i}" for i in range(20)]
        src = str(tmp_path / "source.lmdb")
        _create_test_graph_lmdb(src, keys)

        cfg = SplitConfig(method="random", ratios=(0.8, 0.1, 0.1), seed=42)
        split_paths = partition_lmdb_into_splits(
            src_path=src, out_dir=str(tmp_path), base_name="g", config=cfg,
        )

        recovered_keys = set()
        metadata = {"length", "scaled", "split_name"}
        for path in split_paths.values():
            env = lmdb.open(path, subdir=False, readonly=True, lock=False)
            with env.begin() as txn:
                cursor = txn.cursor()
                for k, v in cursor:
                    ks = k.decode("ascii")
                    if ks not in metadata:
                        recovered_keys.add(ks)
                        # Verify data is intact
                        data = pickle.loads(v)
                        assert data == {"fake_graph": ks}
            env.close()

        assert recovered_keys == set(keys)

    def test_composition_split(self, tmp_path):
        keys = [f"mol_{i}" for i in range(20)]
        src = str(tmp_path / "source.lmdb")
        _create_test_graph_lmdb(src, keys)

        formula_map = {f"mol_{i}": "C6H12O6" if i < 10 else "CH4" for i in range(20)}
        cfg = SplitConfig(method="composition", ratios=(0.8, 0.1, 0.1), seed=42)
        split_paths = partition_lmdb_into_splits(
            src_path=src, out_dir=str(tmp_path), base_name="g",
            config=cfg, formula_map=formula_map,
        )

        # Verify no formula leaks
        metadata = {"length", "scaled", "split_name"}
        for path in split_paths.values():
            env = lmdb.open(path, subdir=False, readonly=True, lock=False)
            formulas_in_split = set()
            with env.begin() as txn:
                for k, _ in txn.cursor():
                    ks = k.decode("ascii")
                    if ks not in metadata and ks in formula_map:
                        formulas_in_split.add(formula_map[ks])
            env.close()
            # Each formula should appear in at most one split
            # (this is checked across splits below)

        # Verify each formula group is entirely in one split
        c6_keys = {f"mol_{i}" for i in range(10)}
        ch4_keys = {f"mol_{i}" for i in range(10, 20)}

        for group in (c6_keys, ch4_keys):
            splits_containing = []
            for name, path in split_paths.items():
                env = lmdb.open(path, subdir=False, readonly=True, lock=False)
                with env.begin() as txn:
                    found = any(txn.get(k.encode("ascii")) is not None for k in group)
                env.close()
                if found:
                    splits_containing.append(name)
            assert len(splits_containing) == 1

    def test_composition_requires_formula_map(self, tmp_path):
        keys = ["a", "b"]
        src = str(tmp_path / "source.lmdb")
        _create_test_graph_lmdb(src, keys)

        cfg = SplitConfig(method="composition", ratios=(0.8, 0.1, 0.1), seed=42)
        with pytest.raises(ValueError, match="formula_map is required"):
            partition_lmdb_into_splits(
                src_path=src, out_dir=str(tmp_path), base_name="g",
                config=cfg, formula_map=None,
            )


# ---------------------------------------------------------------------------
# Config parsing integration
# ---------------------------------------------------------------------------

class TestConfigParsingSplitDefaults:
    def test_split_defaults_added(self, tmp_path):
        import json
        config = {
            "lmdb_path": str(tmp_path),
            "lmdb_name": "test.lmdb",
            "lmdb_locations": {"geom_lmdb": "dummy"},
        }
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(config))

        from qtaim_gen.source.utils.lmdbs import parse_config_gen_to_embed
        result = parse_config_gen_to_embed(str(config_file))
        assert result["split_method"] == "random"
        assert result["split_ratios"] == [0.8, 0.1, 0.1]
        assert result["split_seed"] == 42

    def test_split_config_preserved(self, tmp_path):
        import json
        config = {
            "lmdb_path": str(tmp_path),
            "lmdb_name": "test.lmdb",
            "lmdb_locations": {"geom_lmdb": "dummy"},
            "split_method": "composition",
            "split_ratios": [0.7, 0.15, 0.15],
            "split_seed": 99,
        }
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(config))

        from qtaim_gen.source.utils.lmdbs import parse_config_gen_to_embed
        result = parse_config_gen_to_embed(str(config_file))
        assert result["split_method"] == "composition"
        assert result["split_ratios"] == [0.7, 0.15, 0.15]
        assert result["split_seed"] == 99
