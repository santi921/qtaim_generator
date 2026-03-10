#!/usr/bin/env python3
"""Comprehensive test of all converter configs"""

import json
import os
import lmdb
import pickle
from qtaim_gen.source.core.converter import BaseConverter, QTAIMConverter, GeneralConverter
from qtaim_embed.data.lmdb import load_graph_from_serialized

def test_converter(converter_class, config_file, test_name):
    """Test a single converter configuration"""
    print("\n" + "="*70)
    print(f"TEST: {test_name}")
    print("="*70)

    try:
        # Load config
        with open(config_file) as f:
            config = json.load(f)

        # Ensure output directory exists
        os.makedirs(config['lmdb_path'], exist_ok=True)
        config_path = os.path.join(config['lmdb_path'], 'config.json')

        # Create and run converter
        print(f"Running {converter_class.__name__}...")
        converter = converter_class(config, config_path=config_path)
        converter.process()

        # Verify output
        output_lmdb = os.path.join(config['lmdb_path'], config['lmdb_name'])

        if not os.path.exists(output_lmdb):
            print(f"✗ FAILED: Output LMDB not created")
            return False

        # Count graphs
        env = lmdb.open(output_lmdb, readonly=True, subdir=False, lock=False)
        with env.begin() as txn:
            cursor = txn.cursor()
            all_keys = [k for k, _ in cursor]
            graph_keys = [k for k in all_keys if k not in [b'scaled', b'length', b'scaler_finalized']]

            if not graph_keys:
                print(f"✗ FAILED: No graphs in output")
                env.close()
                return False

            # Check first graph structure
            value = txn.get(graph_keys[0])
            graph = load_graph_from_serialized(pickle.loads(value))

            print(f"\n✓ SUCCESS")
            print(f"  Graphs: {len(graph_keys)}/{len(all_keys)-1}")  # -1 for 'length' key
            print(f"  Sample: {graph_keys[0].decode('ascii')}")
            print(f"    Node types: {graph.node_types}")
            print(f"    Edge types: {graph.edge_types}")

            # Check for specific features based on converter type
            if 'atom' in graph.node_types and hasattr(graph['atom'], 'feat'):
                print(f"    Atom feat shape: {graph['atom'].feat.shape}")
            if 'global' in graph.node_types and hasattr(graph['global'], 'feat'):
                print(f"    Global feat shape: {graph['global'].feat.shape}")

                # Check for dipoles if this is the dipole config
                if 'dipole' in config_file:
                    print(f"    Global feat tensor present (check for dipoles in feature names)")

        env.close()
        return True

    except Exception as e:
        print(f"\n✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all converter tests"""
    # Clean previous outputs
    import shutil
    output_dir = '/home/santiagovargas/dev/qtaim_generator/data/output_graphs'
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    results = {}
    config_base = '/home/santiagovargas/dev/qtaim_generator/qtaim_gen/source/scripts/helpers/configs_converter'

    tests = [
        (BaseConverter, f'{config_base}/base_unsharded.json', 'Base Converter'),
        (QTAIMConverter, f'{config_base}/qtaim_unsharded.json', 'QTAIM Converter'),
        (GeneralConverter, f'{config_base}/general_fuzzy_bonds.json', 'General - Fuzzy Bonds'),
        (GeneralConverter, f'{config_base}/general_qtaim_bonds.json', 'General - QTAIM Bonds'),
        (GeneralConverter, f'{config_base}/general_ibsi_bonds.json', 'General - IBSI Bonds'),
        (GeneralConverter, f'{config_base}/general_with_global_dipoles.json', 'General - Global Dipoles'),
    ]

    for converter_class, config_file, test_name in tests:
        results[test_name] = test_converter(converter_class, config_file, test_name)

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status:8} {test_name}")

    print(f"\nTotal: {passed}/{total} passed")

    if passed == total:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    exit(main())
