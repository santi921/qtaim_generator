#!/usr/bin/env python3
"""Debug script to inspect LMDB contents and identify problematic entries."""

import lmdb
import pickle
import sys
from collections import Counter

# Try to import DGL deserialization functions
try:
    from qtaim_embed.data.lmdb import load_dgl_graph_from_serialized
    HAS_DGL_DESER = True
except ImportError:
    HAS_DGL_DESER = False
    print("Warning: qtaim_embed not available, DGL graph deserialization disabled")

def inspect_lmdb(lmdb_path, deserialize_graphs=True):
    """Inspect LMDB and categorize entries by type.

    Args:
        lmdb_path: Path to LMDB file
        deserialize_graphs: If True and qtaim_embed available, deserialize DGL graphs from bytes
    """
    print(f"\n{'='*60}")
    print(f"Inspecting: {lmdb_path}")
    print(f"{'='*60}\n")

    env = lmdb.open(lmdb_path, readonly=True, lock=False, subdir=False)

    type_counter = Counter()
    metadata_keys = []
    graph_keys = []
    error_keys = []

    # Track deserialization stats
    bytes_objects = 0
    successful_deser = 0
    failed_deser = 0

    with env.begin() as txn:
        cursor = txn.cursor()
        total_entries = 0

        for key, value in cursor:
            total_entries += 1

            # Decode key
            try:
                key_str = key.decode('utf-8')
            except:
                key_str = str(key)

            # Try to unpickle value
            try:
                obj = pickle.loads(value)
                obj_type = type(obj).__name__
                type_counter[obj_type] += 1

                # Categorize
                if key_str in ['scaled', 'scaler_finalized']:
                    metadata_keys.append((key_str, obj_type, obj))
                elif obj_type == 'DGLGraph' or (hasattr(obj, 'ndata') and hasattr(obj, 'edata')):
                    graph_keys.append((key_str, obj_type))
                elif obj_type in ['int', 'str', 'bool', 'float']:
                    error_keys.append((key_str, obj_type, obj))
                elif obj_type == 'bytes':
                    # Bytes object - might be serialized DGL graph
                    bytes_objects += 1

                    # Try to deserialize as DGL graph
                    if deserialize_graphs and HAS_DGL_DESER:
                        try:
                            graph = load_dgl_graph_from_serialized(obj)
                            # Successfully deserialized to DGL graph

                            # Analyze feature/label sizes by node type
                            feature_info = []
                            for ntype in graph.ntypes:
                                ndata = graph.nodes[ntype].data
                                if ndata:
                                    for feat_name, feat_tensor in ndata.items():
                                        feature_info.append(f"{ntype}.{feat_name}: {tuple(feat_tensor.shape)}")

                            # Create summary
                            basic_info = f"{graph.num_nodes()} nodes, {graph.num_edges()} edges"
                            if feature_info:
                                feat_summary = "; Features: " + ", ".join(feature_info[:3])
                                if len(feature_info) > 3:
                                    feat_summary += f" + {len(feature_info)-3} more"
                                basic_info += feat_summary

                            graph_keys.append((key_str, 'DGLGraph(deserialized)', basic_info, graph))
                            successful_deser += 1
                            type_counter['DGLGraph(deserialized)'] += 1
                        except Exception as deser_e:
                            # Failed to deserialize - probably not a DGL graph
                            failed_deser += 1
                            error_keys.append((key_str, 'BYTES(failed_deser)',
                                             f"{len(obj)} bytes, error: {str(deser_e)[:50]}"))
                    else:
                        # Deserialization disabled or not available
                        error_keys.append((key_str, 'RAW_BYTES',
                                         f"{len(obj)} bytes, starts with {obj[:20]}"))
                else:
                    # Could be scaler or other object
                    metadata_keys.append((key_str, obj_type, str(obj)[:100]))

            except Exception as e:
                type_counter['unpickle_error'] += 1
                error_keys.append((key_str, 'ERROR', str(e)[:100]))

    env.close()

    # Print summary
    print(f"Total entries: {total_entries}")
    print(f"\nType distribution:")
    for obj_type, count in type_counter.most_common():
        print(f"  {obj_type}: {count}")

    # Print metadata
    if metadata_keys:
        print(f"\nMetadata keys ({len(metadata_keys)}):")
        for key, obj_type, value in metadata_keys[:10]:
            print(f"  {key}: {obj_type} = {value}")
        if len(metadata_keys) > 10:
            print(f"  ... and {len(metadata_keys) - 10} more")

    # Print deserialization stats if applicable
    if bytes_objects > 0:
        print(f"\nDGL Deserialization:")
        print(f"  Bytes objects found: {bytes_objects}")
        if HAS_DGL_DESER and deserialize_graphs:
            print(f"  Successfully deserialized: {successful_deser}")
            print(f"  Failed to deserialize: {failed_deser}")
        else:
            print(f"  Deserialization skipped (qtaim_embed not available or disabled)")

    # Print graphs
    if graph_keys:
        print(f"\nGraph keys ({len(graph_keys)}):")
        for item in graph_keys[:5]:
            if len(item) == 4:
                key, obj_type, info, graph = item
                print(f"  {key}: {obj_type} ({info})")
            elif len(item) == 3:
                key, obj_type, info = item
                print(f"  {key}: {obj_type} ({info})")
            else:
                key, obj_type = item
                print(f"  {key}: {obj_type}")
        if len(graph_keys) > 5:
            print(f"  ... and {len(graph_keys) - 5} more")

        # Print detailed feature breakdown for first graph
        if graph_keys and len(graph_keys[0]) == 4:
            first_graph_item = graph_keys[0]
            key, obj_type, info, graph = first_graph_item
            print(f"\n  Detailed Feature/Label Breakdown for '{key}':")
            for ntype in graph.ntypes:
                ndata = graph.nodes[ntype].data
                if ndata:
                    print(f"    {ntype} node features:")
                    for feat_name, feat_tensor in ndata.items():
                        print(f"      {feat_name}: shape={tuple(feat_tensor.shape)}, dtype={feat_tensor.dtype}")
                else:
                    print(f"    {ntype} node features: (none)")

            # Check edge features
            if graph.etypes:
                for etype in graph.canonical_etypes[:3]:  # Show first 3 edge types
                    edata = graph.edges[etype].data
                    if edata:
                        print(f"    {etype} edge features:")
                        for feat_name, feat_tensor in edata.items():
                            print(f"      {feat_name}: shape={tuple(feat_tensor.shape)}, dtype={feat_tensor.dtype}")

    # Print errors
    if error_keys:
        print(f"\n  PROBLEM KEYS ({len(error_keys)}):")
        for key, obj_type, value in error_keys[:20]:
            print(f"  {key}: {obj_type} = {value}")
        if len(error_keys) > 20:
            print(f"  ... and {len(error_keys) - 20} more")

    return {
        'total': total_entries,
        'types': dict(type_counter),
        'graphs': len(graph_keys),
        'errors': len(error_keys),
        'metadata': len(metadata_keys),
        'bytes_objects': bytes_objects,
        'successful_deser': successful_deser,
        'failed_deser': failed_deser,
    }

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Inspect LMDB contents and identify problematic entries")
    parser.add_argument("lmdb_paths", nargs="+", help="Path(s) to LMDB file(s)")
    parser.add_argument("--no-deserialize", action="store_true",
                       help="Skip DGL graph deserialization (faster but less detailed)")

    args = parser.parse_args()

    if not args.lmdb_paths:
        print("Usage: python debug_lmdb_contents.py <lmdb_path> [lmdb_path2 ...]")
        print("\nExample:")
        print("  python debug_lmdb_contents.py output/shard_0/base_graphs.lmdb")
        print("  python debug_lmdb_contents.py output/shard_*/base_graphs.lmdb")
        print("  python debug_lmdb_contents.py --no-deserialize output/merged/merged.lmdb")
        sys.exit(1)

    results = {}
    for lmdb_path in args.lmdb_paths:
        results[lmdb_path] = inspect_lmdb(lmdb_path, deserialize_graphs=not args.no_deserialize)

    # Summary comparison
    if len(results) > 1:
        print(f"\n{'='*60}")
        print("SUMMARY COMPARISON")
        print(f"{'='*60}\n")
        for path, stats in results.items():
            print(f"{path}:")
            print(f"  Total: {stats['total']}, Graphs: {stats['graphs']}, Errors: {stats['errors']}")
