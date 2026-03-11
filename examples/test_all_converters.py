#!/usr/bin/env python3
"""
Comprehensive test of all converter types using real production data.

Demonstrates BaseConverter, QTAIMConverter, and GeneralConverter with
different bonding schemes, charge filters, and feature configurations.

Uses the rmechdb LMDBs in data/lmdb_rmechdb/ which contain 1001 molecules
with structure, charge, bond, qtaim, fuzzy, and other data.

Usage:
    python examples/test_all_converters.py
    python examples/test_all_converters.py --data-dir /path/to/other/lmdbs
"""

import argparse
import json
import os
import shutil
import tempfile
import lmdb
import pickle

from qtaim_gen.source.core.converter import (
    BaseConverter,
    QTAIMConverter,
    GeneralConverter,
)
from qtaim_embed.data.lmdb import load_graph_from_serialized


# Default path to production LMDBs (relative to repo root)
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_LMDB_DIR = os.path.join(REPO_ROOT, "data", "lmdb_rmechdb")

# Mapping from prod LMDB filenames to converter config keys
LMDB_FILE_MAP = {
    "geom_lmdb": "structure.lmdb",
    "charge_lmdb": "charge.lmdb",
    "bonds_lmdb": "bond.lmdb",
    "qtaim_lmdb": "qtaim.lmdb",
    "fuzzy_full_lmdb": "fuzzy.lmdb",
    "other_lmdb": "other.lmdb",
}


def lmdb_path(data_dir, config_key):
    """Resolve an LMDB path from a converter config key."""
    filename = LMDB_FILE_MAP.get(config_key, config_key)
    return os.path.join(data_dir, filename)


# ── Converter configs ──────────────────────────────────────────────────────


def base_config(output_dir, data_dir):
    """BaseConverter: structural info only (positions, elements, connectivity)."""
    return {
        "chunk": -1,
        "filter_list": ["length", "scaled"],
        "restart": False,
        "allowed_ring_size": [3, 4, 5, 6, 7, 8],
        "allowed_charges": None,
        "allowed_spins": None,
        "keys_target": {"atom": [], "bond": [], "global": ["n_atoms"]},
        "keys_data": {"atom": [], "bond": [], "global": ["n_atoms"]},
        "lmdb_path": os.path.join(output_dir, "base"),
        "lmdb_name": "graphs.lmdb",
        "lmdb_locations": {
            "geom_lmdb": lmdb_path(data_dir, "geom_lmdb"),
        },
        "n_workers": 1,
        "batch_size": 100,
    }


def qtaim_config(output_dir, data_dir):
    """QTAIMConverter: QTAIM bond paths + critical point properties."""
    return {
        "chunk": -1,
        "filter_list": ["length", "scaled"],
        "restart": False,
        "allowed_ring_size": [3, 4, 5, 6, 7, 8],
        "allowed_charges": None,
        "allowed_spins": None,
        "keys_target": {"atom": [], "bond": [], "global": ["n_atoms"]},
        "keys_data": {"atom": [], "bond": [], "global": ["n_atoms"]},
        "lmdb_path": os.path.join(output_dir, "qtaim"),
        "lmdb_name": "graphs.lmdb",
        "lmdb_locations": {
            "geom_lmdb": lmdb_path(data_dir, "geom_lmdb"),
            "qtaim_lmdb": lmdb_path(data_dir, "qtaim_lmdb"),
        },
        "n_workers": 1,
        "batch_size": 100,
    }


def general_fuzzy_bonds_config(output_dir, data_dir):
    """GeneralConverter: fuzzy bond orders + charge features."""
    return {
        "chunk": -1,
        "filter_list": ["length", "scaled"],
        "restart": False,
        "allowed_ring_size": [3, 4, 5, 6, 7, 8],
        "allowed_charges": None,
        "allowed_spins": None,
        "keys_target": {"atom": [], "bond": [], "global": ["n_atoms"]},
        "keys_data": {
            "atom": ["charge_hirshfeld", "charge_adch"],
            "bond": [],
            "global": ["n_atoms"],
        },
        "lmdb_path": os.path.join(output_dir, "general_fuzzy"),
        "lmdb_name": "graphs.lmdb",
        "lmdb_locations": {
            "geom_lmdb": lmdb_path(data_dir, "geom_lmdb"),
            "charge_lmdb": lmdb_path(data_dir, "charge_lmdb"),
            "bonds_lmdb": lmdb_path(data_dir, "bonds_lmdb"),
        },
        "bonding_scheme": "bonding",
        "bond_list_definition": "fuzzy",
        "bond_cutoff": 0.3,
        "bond_filter": ["fuzzy"],
        "charge_filter": ["hirshfeld", "adch"],
        "missing_data_strategy": "skip",
        "n_workers": 1,
        "batch_size": 100,
    }


def general_structural_bonds_config(output_dir, data_dir):
    """GeneralConverter: coordinate-based bonds with charge + fuzzy features."""
    return {
        "chunk": -1,
        "filter_list": ["length", "scaled"],
        "restart": False,
        "allowed_ring_size": [3, 4, 5, 6, 7, 8],
        "allowed_charges": None,
        "allowed_spins": None,
        "keys_target": {"atom": [], "bond": [], "global": ["n_atoms"]},
        "keys_data": {
            "atom": ["charge_hirshfeld", "charge_adch", "charge_cm5", "charge_becke"],
            "bond": [],
            "global": ["n_atoms"],
        },
        "lmdb_path": os.path.join(output_dir, "general_structural"),
        "lmdb_name": "graphs.lmdb",
        "lmdb_locations": {
            "geom_lmdb": lmdb_path(data_dir, "geom_lmdb"),
            "charge_lmdb": lmdb_path(data_dir, "charge_lmdb"),
            "bonds_lmdb": lmdb_path(data_dir, "bonds_lmdb"),
            "fuzzy_full_lmdb": lmdb_path(data_dir, "fuzzy_full_lmdb"),
        },
        "bonding_scheme": "structural",
        "bond_list_definition": "fuzzy",
        "bond_cutoff": 0.3,
        "bond_filter": ["fuzzy"],
        "charge_filter": ["hirshfeld", "adch", "cm5", "becke"],
        "fuzzy_filter": ["becke_fuzzy_density", "hirsh_fuzzy_density"],
        "missing_data_strategy": "skip",
        "n_workers": 1,
        "batch_size": 100,
    }


def general_all_features_config(output_dir, data_dir):
    """GeneralConverter: all available data sources (charge, bond, fuzzy, qtaim, other)."""
    return {
        "chunk": -1,
        "filter_list": ["length", "scaled"],
        "restart": False,
        "allowed_ring_size": [3, 4, 5, 6, 7, 8],
        "allowed_charges": None,
        "allowed_spins": None,
        "keys_target": {"atom": [], "bond": [], "global": ["n_atoms"]},
        "keys_data": {
            "atom": ["charge_hirshfeld", "charge_adch", "charge_cm5", "charge_becke"],
            "bond": [],
            "global": ["n_atoms"],
        },
        "lmdb_path": os.path.join(output_dir, "general_all"),
        "lmdb_name": "graphs.lmdb",
        "lmdb_locations": {
            "geom_lmdb": lmdb_path(data_dir, "geom_lmdb"),
            "charge_lmdb": lmdb_path(data_dir, "charge_lmdb"),
            "bonds_lmdb": lmdb_path(data_dir, "bonds_lmdb"),
            "fuzzy_full_lmdb": lmdb_path(data_dir, "fuzzy_full_lmdb"),
            "qtaim_lmdb": lmdb_path(data_dir, "qtaim_lmdb"),
            "other_lmdb": lmdb_path(data_dir, "other_lmdb"),
        },
        "bonding_scheme": "bonding",
        "bond_list_definition": "fuzzy",
        "bond_cutoff": 0.3,
        "bond_filter": ["fuzzy"],
        "charge_filter": ["hirshfeld", "adch", "cm5", "becke"],
        "fuzzy_filter": ["becke_fuzzy_density", "hirsh_fuzzy_density"],
        "missing_data_strategy": "skip",
        "n_workers": 1,
        "batch_size": 100,
    }


# ── Runner ─────────────────────────────────────────────────────────────────


def run_converter(converter_class, config, test_name):
    """Run a single converter and verify output."""
    print("\n" + "=" * 70)
    print(f"TEST: {test_name}")
    print("=" * 70)

    try:
        os.makedirs(config["lmdb_path"], exist_ok=True)
        config_path = os.path.join(config["lmdb_path"], "config.json")

        # Save config for reproducibility
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        print(f"Running {converter_class.__name__}...")
        converter = converter_class(config, config_path=config_path)
        converter.process()

        # Verify output
        output_lmdb = os.path.join(config["lmdb_path"], config["lmdb_name"])
        if not os.path.exists(output_lmdb):
            print("  FAILED: Output LMDB not created")
            return False

        env = lmdb.open(output_lmdb, readonly=True, subdir=False, lock=False)
        try:
            with env.begin() as txn:
                all_keys = [k for k, _ in txn.cursor()]
                meta_keys = {b"scaled", b"length", b"scaler_finalized"}
                graph_keys = [k for k in all_keys if k not in meta_keys]

                if not graph_keys:
                    print("  FAILED: No graphs in output")
                    return False

                # Inspect first graph
                value = txn.get(graph_keys[0])
                graph = load_graph_from_serialized(pickle.loads(value))

                print(f"  SUCCESS: {len(graph_keys)} graphs")
                print(f"  Sample: {graph_keys[0].decode('ascii')}")
                print(f"    Node types: {graph.node_types}")
                print(f"    Edge types: {graph.edge_types}")

                for ntype in graph.node_types:
                    if hasattr(graph[ntype], "feat"):
                        print(f"    {ntype} feat shape: {graph[ntype].feat.shape}")

                return True
        finally:
            env.close()

    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run all converter examples."""
    parser = argparse.ArgumentParser(
        description="Test all converter types against LMDB data."
    )
    parser.add_argument(
        "--data-dir",
        default=DEFAULT_LMDB_DIR,
        help="Path to directory containing source LMDBs "
        "(structure.lmdb, charge.lmdb, etc.). Default: data/lmdb_rmechdb/",
    )
    parser.add_argument(
        "--keep-output",
        action="store_true",
        help="Keep the output directory instead of cleaning up.",
    )
    args = parser.parse_args()

    data_dir = os.path.abspath(args.data_dir)

    # Verify data exists
    if not os.path.isdir(data_dir):
        print(f"LMDB data directory not found: {data_dir}")
        return 1

    required = ["structure.lmdb"]
    for f in required:
        if not os.path.exists(os.path.join(data_dir, f)):
            print(f"Missing required LMDB: {f} in {data_dir}")
            return 1

    # Check which optional LMDBs are available
    available = {}
    for config_key, filename in LMDB_FILE_MAP.items():
        path = os.path.join(data_dir, filename)
        available[config_key] = os.path.exists(path)
        status = "found" if available[config_key] else "missing"
        print(f"  {filename:20s} -> {config_key:20s} [{status}]")

    # Use a temp directory for outputs
    output_dir = tempfile.mkdtemp(prefix="converter_test_")
    print(f"\nOutput directory: {output_dir}")

    tests = [
        (BaseConverter, base_config(output_dir, data_dir), "Base Converter"),
        (QTAIMConverter, qtaim_config(output_dir, data_dir), "QTAIM Converter"),
        (
            GeneralConverter,
            general_fuzzy_bonds_config(output_dir, data_dir),
            "General - Fuzzy Bonds",
        ),
        (
            GeneralConverter,
            general_structural_bonds_config(output_dir, data_dir),
            "General - Structural Bonds",
        ),
        (
            GeneralConverter,
            general_all_features_config(output_dir, data_dir),
            "General - All Features",
        ),
    ]

    # Skip tests that need missing LMDBs
    runnable_tests = []
    for converter_class, config, test_name in tests:
        missing = [
            k for k in config["lmdb_locations"]
            if not available.get(k, False)
        ]
        if missing:
            print(f"\n  SKIP: {test_name} (missing: {', '.join(missing)})")
        else:
            runnable_tests.append((converter_class, config, test_name))

    results = {}
    for converter_class, config, test_name in runnable_tests:
        results[test_name] = run_converter(converter_class, config, test_name)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    total = len(runnable_tests)
    n_passed = sum(1 for v in results.values() if v)
    n_failed = total - n_passed

    for test_name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {status:4}  {test_name}")

    print(f"\nTotal: {n_passed}/{total} passed")

    # Cleanup
    if args.keep_output:
        print(f"Output kept at: {output_dir}")
    else:
        shutil.rmtree(output_dir, ignore_errors=True)
        print(f"Cleaned up {output_dir}")

    return 0 if n_passed == total else 1


if __name__ == "__main__":
    exit(main())
