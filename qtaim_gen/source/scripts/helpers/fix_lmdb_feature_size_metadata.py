#!/usr/bin/env python3
"""Patch inflated feature_size/feature_names metadata in graph LMDBs.

Converters built after commit 9107606 wrote feature_size/feature_names from
the grapher's PRE-split schema -- it includes the target columns that
split_graph_labels moves into `labels`. The stored graphs' `.feat` width is
the POST-split width, so the metadata is too wide by the target count and a
model sized from it builds an embedding wider than the graphs it is fed:

    RuntimeError: mat1 and mat2 shapes cannot be multiplied

The graph tensors are correct; only the metadata is wrong. This rewrites
feature_size and feature_names to the post-split schema, recoverable from each
LMDB's own metadata: the target columns are the trailing entries of
feature_names[nt] (the featurizer appends target keys last), and target_dict
gives their count. A guard asserts the trailing names actually match the
targets before trimming, so it cannot mistrim a schema it does not understand.

Usage:
    python fix_lmdb_feature_size_metadata.py LMDB [LMDB ...]
    python fix_lmdb_feature_size_metadata.py --dry-run LMDB
"""
import argparse
import os
import pickle

import lmdb


def _corrected_schema(feature_names: dict, target_dict: dict):
    """Return (new_feature_names, new_feature_size, changed: bool).

    Drops the trailing target columns from each node type's feature_names.
    Raises ValueError if the trailing columns do not match the declared
    targets (refuses to trim a schema it cannot verify).
    """
    new_names = {}
    new_size = {}
    changed = False
    for ntype, names in feature_names.items():
        targets = [t for t in target_dict.get(ntype, []) if t]
        n_targets = len(targets)
        if n_targets == 0:
            new_names[ntype] = list(names)
            new_size[ntype] = len(names)
            continue
        if len(names) < n_targets:
            raise ValueError(
                f"{ntype}: feature_names ({len(names)}) shorter than targets "
                f"({n_targets}); cannot trim"
            )
        trailing = names[-n_targets:]
        if set(trailing) != set(targets):
            raise ValueError(
                f"{ntype}: trailing feature_names {trailing} do not match targets "
                f"{targets}; refusing to trim (schema not understood)"
            )
        kept = list(names[:-n_targets])
        new_names[ntype] = kept
        new_size[ntype] = len(kept)
        changed = True
    return new_names, new_size, changed


def fix_one(path: str, dry_run: bool = False) -> None:
    if not os.path.exists(path):
        print(f"SKIP (missing): {path}")
        return

    map_size = os.path.getsize(path) + 100 * 1024 * 1024
    env = lmdb.open(path, subdir=False, readonly=False, lock=True, map_size=map_size)
    with env.begin() as txn:
        fn_raw = txn.get(b"feature_names")
        td_raw = txn.get(b"target_dict")
        fs_raw = txn.get(b"feature_size")

    if fn_raw is None or td_raw is None:
        print(f"SKIP (no feature_names/target_dict): {path}")
        env.close()
        return

    feature_names = pickle.loads(fn_raw)
    target_dict = pickle.loads(td_raw)
    old_size = pickle.loads(fs_raw) if fs_raw is not None else None

    new_names, new_size, changed = _corrected_schema(feature_names, target_dict)

    if not changed and old_size == new_size:
        print(f"OK (already post-split): {path}  feature_size={new_size}")
        env.close()
        return

    print(f"{'DRY-RUN ' if dry_run else 'PATCH '}{path}")
    print(f"  feature_size: {old_size} -> {new_size}")

    if not dry_run:
        with env.begin(write=True) as txn:
            txn.put(b"feature_names", pickle.dumps(new_names, protocol=-1))
            txn.put(b"feature_size", pickle.dumps(new_size, protocol=-1))
        env.sync()
    env.close()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("lmdbs", nargs="+", help="LMDB file(s) to patch (subdir=False)")
    p.add_argument("--dry-run", action="store_true", help="report changes without writing")
    args = p.parse_args(argv)
    for path in args.lmdbs:
        fix_one(path, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
