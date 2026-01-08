"""Combined LMDB-related tests.

This file wraps existing focused test modules into a single test suite.
Each wrapper calls the corresponding test function from the original module
so the original tests remain untouched and the combined suite can be
run independently. Heavy LMDB module imports are guarded and skipped if
their environment assumptions are not met.
"""

import importlib.util
from pathlib import Path
import pytest


def _load_module_by_name(mod_name: str):
    """Load a test module by filename (robust to CWD and package layout)."""
    tests_dir = Path(__file__).parent
    mod_path = tests_dir / f"{mod_name}.py"
    spec = importlib.util.spec_from_file_location(mod_name, str(mod_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_restart_existing_keys_and_scaled_flag(tmp_path):
    mod = _load_module_by_name("test_restart_existing_keys")
    return mod.test_restart_existing_keys_and_scaled_flag(tmp_path)


def test_scale_single_file(tmp_path):
    mod = _load_module_by_name("test_scale_graphs")
    return mod.test_scale_single_file(tmp_path)


def test_scale_folder_input(tmp_path):
    mod = _load_module_by_name("test_scale_graphs")
    return mod.test_scale_folder_input(tmp_path)


def test_serialize_deserialize_roundtrip_synthetic():
    mod = _load_module_by_name("test_serialize_roundtrip")
    return mod.test_serialize_deserialize_roundtrip_synthetic()


def test_folder_indexing_maps_to_correct_env():
    mod = _load_module_by_name("test_lmdb_indexing")
    return mod.test_folder_indexing_maps_to_correct_env()


def test_key_normalization_and_scaling(tmp_path):
    mod = _load_module_by_name("test_key_normalization")
    return mod.test_key_normalization_and_scaling(tmp_path)


def test_lmdb_full_suite():
    # Run a selection of the heavier LMDB tests from test_lmdb.py.
    mod = _load_module_by_name("test_lmdb")

    # instantiate classes and call a few representative tests
    tm = mod.TestLMDB()
    # ensure class-level setup runs when invoking manually
    if hasattr(tm, "setup_class"):
        tm.setup_class()
    tm.test_write_read()
    tm.test_merge()

    tc = mod.TestConverters()
    if hasattr(tc, "setup_class"):
        tc.setup_class()
    tc.test_restarts()
    tc.test_scaling_counts()
    tc.test_process_counts()
    tc.test_feat_names()
    tc.test_scaling_ops_single()
    tc.test_scaling_ops_folder()

    assert True
