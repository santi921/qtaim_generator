#!/usr/bin/env python3
"""Debug fuzzy bonds converter"""

import json
import os
import lmdb
import pickle
import traceback
from qtaim_gen.source.core.converter import GeneralConverter
from qtaim_embed.data.lmdb import load_graph_from_serialized

config_file = '/home/santiagovargas/dev/qtaim_generator/qtaim_gen/source/scripts/helpers/configs_converter/general_fuzzy_bonds.json'

# Load config
with open(config_file) as f:
    config = json.load(f)

print("Config loaded:")
print(f"  bond_filter: {config.get('bond_filter')}")
print(f"  bond_list_definition: {config.get('bond_list_definition')}")
print(f"  fuzzy_filter: {config.get('fuzzy_filter')}")
print(f"  charge_filter: {config.get('charge_filter')}")

# Clean previous output
import shutil
if os.path.exists(config['lmdb_path']):
    shutil.rmtree(config['lmdb_path'])

os.makedirs(config['lmdb_path'], exist_ok=True)
config_path = os.path.join(config['lmdb_path'], 'config.json')

# Create and run converter
print("\nCreating converter...")
try:
    converter = GeneralConverter(config, config_path=config_path)
    print("Converter created successfully")

    print("\nRunning process...")
    converter.process()

    print("\n✓ Process completed")

    # Check output
    output_lmdb = os.path.join(config['lmdb_path'], config['lmdb_name'])
    env = lmdb.open(output_lmdb, readonly=True, subdir=False, lock=False)
    with env.begin() as txn:
        cursor = txn.cursor()
        all_keys = [k for k, _ in cursor]
        graph_keys = [k for k in all_keys if k not in [b'scaled', b'length', b'scaler_finalized']]
        print(f"\nGraphs created: {len(graph_keys)}/{len(all_keys)}")

        if graph_keys:
            value = txn.get(graph_keys[0])
            graph = load_graph_from_serialized(pickle.loads(value))
            print(f"Sample graph: {graph_keys[0].decode('ascii')}")
            print(f"  Node types: {graph.node_types}, Edge types: {graph.edge_types}")
    env.close()

except Exception as e:
    print(f"\n✗ ERROR: {e}")
    traceback.print_exc()
