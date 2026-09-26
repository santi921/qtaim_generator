#!/usr/bin/env python3
"""Find canonical train/val/test keys absent from a merged split tree and look
each one up in the per-vertical source LMDBs (and optionally the job folders).

split_descriptor_lmdbs.py takes its key list from each vertical's source
structure.lmdb and copies every other descriptor only for those keys, so a key
missing from the split tree's structure.lmdb is missing from every descriptor
there. This tells whether such keys exist upstream, and with which descriptors:

  missing   canonical keys (destination train/val/test) not in any
            <split_root>/{train,val,test}/structure.lmdb, or the keys in --keys_file
  source    per descriptor, present in <source_root>/<vertical>/<descriptor>.lmdb
            (or .../<vertical>/merged/<descriptor>.lmdb) under the unprefixed key;
            merge_split_descriptors.py adds the `<vertical>__` prefix, sources lack it
  job       with --job_root: whether <job_root>/<rel_path> exists (rel_path =
            key with `__` -> `/`) and which result JSONs it holds

Standalone: needs `lmdb`; `pyarrow` only for a parquet mapping.

Usage:
    python lookup_missing_split_keys.py --mapping split_destinations.tsv.gz \\
        --split_root .../OMol-Descriptors-4M --source_root /p/lustre5/vargas58/converters/converters_final \\
        [--job_root /p/lustre5/.../OMol4M] --out missing_keys.tsv
"""
from __future__ import annotations

import argparse
import gzip
import os
import sys
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Set

import lmdb

DESCRIPTORS = ("structure", "charge", "bond", "qtaim", "fuzzy", "other", "orca", "timings")
SPLITS = ("train", "val", "test")
JOB_JSONS = ("charge.json", "bond.json", "qtaim.json", "fuzzy.json", "other.json", "orca.json", "timings.json")


def load_canonical(path: str) -> Dict[str, str]:
    """key -> destination, train/val/test only."""
    out: Dict[str, str] = {}
    if path.endswith(".parquet"):
        import pyarrow.parquet as pq
        t = pq.read_table(path, columns=["key", "destination"])
        pairs = zip(t.column("key").to_pylist(), t.column("destination").to_pylist())
    else:
        opener = gzip.open if path.endswith(".gz") else open
        with opener(path, "rt") as f:
            pairs = [ln.rstrip("\n").split("\t")[:2] for ln in f]
    for k, d in pairs:
        if d in SPLITS:
            out[k] = d
    return out


def _open(path: str) -> lmdb.Environment:
    return lmdb.open(path, subdir=os.path.isdir(path), readonly=True, lock=False, readahead=False, meminit=False)


def read_keys(path: str) -> Set[str]:
    env = _open(path)
    try:
        with env.begin() as txn:
            return {k.decode("utf-8", "replace") for k in txn.cursor().iternext(keys=True, values=False)}
    finally:
        env.close()


def find_source_lmdb(source_root: str, vertical: str, descriptor: str) -> Optional[str]:
    for p in (os.path.join(source_root, vertical, f"{descriptor}.lmdb"),
              os.path.join(source_root, vertical, "merged", f"{descriptor}.lmdb")):
        if os.path.exists(p):
            return p
    return None


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mapping", required=True)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--split_root", help="Merged split tree holding {train,val,test}/structure.lmdb")
    src.add_argument("--keys_file", help="Explicit prefixed keys, one per line")
    p.add_argument("--source_root", required=True, help="Per-vertical source LMDB root (split_descriptor_lmdbs --lmdb_root)")
    p.add_argument("--job_root", default=None, help="Job-folder root; key `__` -> `/` gives the folder")
    p.add_argument("--out", default=None, help="Per-key TSV")
    args = p.parse_args(argv)
    for flag, path in (("--split_root", args.split_root), ("--source_root", args.source_root),
                       ("--job_root", args.job_root)):
        if path and not os.path.isdir(path):
            p.error(f"{flag} {path!r} is not a directory")

    canon = load_canonical(args.mapping)
    if args.split_root:
        present: Set[str] = set()
        for s in SPLITS:
            sp = os.path.join(args.split_root, s, "structure.lmdb")
            if not os.path.exists(sp):
                p.error(f"missing {sp}")
            present |= read_keys(sp)
        missing = sorted(set(canon) - present)
    else:
        with open(args.keys_file) as f:
            missing = sorted({ln.strip() for ln in f if ln.strip()})
    print(f"{len(missing)} canonical keys to look up", file=sys.stderr)

    by_vertical: Dict[str, List[str]] = defaultdict(list)
    for k in missing:
        by_vertical[k.split("__", 1)[0]].append(k)

    rows = []
    patterns: Counter = Counter()
    per_vertical: Dict[str, Counter] = defaultdict(Counter)
    for vertical, keys in sorted(by_vertical.items()):
        found: Dict[str, Optional[Set[str]]] = {}
        for d in DESCRIPTORS:
            path = find_source_lmdb(args.source_root, vertical, d)
            found[d] = None if path is None else read_keys(path)
        for k in keys:
            local = k.split("__", 1)[1] if "__" in k else k
            flags = {d: ("no_lmdb" if found[d] is None else ("yes" if local in found[d] else "no"))
                     for d in DESCRIPTORS}
            row = {"key": k, "vertical": vertical, "canonical": canon.get(k, "?"), **flags}
            if args.job_root:
                folder = os.path.join(args.job_root, k.replace("__", "/"))
                row["job_folder"] = "yes" if os.path.isdir(folder) else "no"
                row["job_jsons"] = ",".join(j for j in JOB_JSONS if os.path.isfile(os.path.join(folder, j))) \
                    if row["job_folder"] == "yes" else ""
            have = [d for d in DESCRIPTORS if flags[d] == "yes"]
            pattern = "none" if not have else ("all" if len(have) == len(DESCRIPTORS) else "+".join(have))
            patterns[pattern] += 1
            per_vertical[vertical][pattern] += 1
            rows.append(row)

    if args.out and rows:
        cols = list(rows[0])
        with open(args.out, "w") as f:
            f.write("\t".join(cols) + "\n")
            for r in rows:
                f.write("\t".join(str(r[c]) for c in cols) + "\n")

    print(f"\n{len(missing)} missing keys; source descriptor presence patterns:")
    for pat, n in patterns.most_common():
        print(f"  {n:6d}  {pat}")
    print("\nper vertical:")
    for v, c in sorted(per_vertical.items(), key=lambda kv: -sum(kv[1].values())):
        print(f"  {v:28s} {sum(c.values()):6d}  " + "  ".join(f"{pat}={n}" for pat, n in c.most_common()))
    if args.job_root:
        jf = Counter(r["job_folder"] for r in rows)
        print(f"\njob folders: exist {jf.get('yes', 0)}, absent {jf.get('no', 0)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
