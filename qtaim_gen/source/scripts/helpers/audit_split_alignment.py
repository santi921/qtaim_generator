#!/usr/bin/env python3
"""Three-way split alignment audit: canonical split vs descriptor LMDBs vs graph LMDBs.

For every set (train / val / test / H1 / H3 / ...) this compares

  canonical   the per-key destination in the split oracle
              (data/omol_manifest/energy_mapping.parquet, or the
              split_destinations.tsv.gz export). Dual-suite holdout keys carry
              a comma-joined destination ("H7,H8") and belong to both suites.
  descriptor  the keys of every descriptor LMDB under --descriptor_root
  graph       the mol_name of every graph under --graph_root (read by
              deserializing each graph; integer LMDB keys carry no identity)

and reports, per set: counts from each source, records sitting in the wrong
set (with where they canonically belong), canonical keys with no graph
(split into "descriptor present" = build skip / NaN drop, and "descriptor
absent"), graphs with no descriptor record in the same set, keys absent from
the mapping, and duplicate mol_names within and across graph sets.

Set discovery walks each root. A directory is a descriptor set when it holds
structure.lmdb; a graph set when it holds merged/merged.lmdb, or otherwise
*.lmdb files (shards, graphs.lmdb). The set label is the directory's last path
component, so NERSC `train/merged/merged.lmdb` and `holdouts/H1/merged/merged.lmdb`,
the local `holdouts/H6/graphs.lmdb`, and per-vertical `splits/<v>/train/shard_*.lmdb`
all resolve (directories sharing a label are pooled).

Exit code 1 when any record is misplaced, duplicated, or unknown to the
mapping; missing graphs alone are reported but do not fail the audit.

Usage (NERSC):
    python -m qtaim_gen.source.scripts.helpers.audit_split_alignment \\
        --mapping split_destinations.tsv.gz \\
        --descriptor_root /pscratch/sd/s/santiago/qtaim_embed_experiments/data/generator_lmdbs/OMol-Descriptors-4M \\
        --graph_root /pscratch/sd/s/santiago/qtaim_embed_experiments/data/node/OMol4M_charge_full \\
        --workers 32 --report alignment.json --moves_tsv graph_moves.tsv
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import pickle
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, Iterable, List, Optional, Set, Tuple

import lmdb

DESCRIPTORS = ("structure", "charge", "bond", "qtaim", "fuzzy", "other", "orca", "timings")
META_KEYS = {b"length", b"scaled", b"scaler_finalized", b"element_set", b"feature_names",
             b"feature_size", b"target_dict", b"processed_source_keys", b"allowed_charges",
             b"allowed_ring_size", b"allowed_spins"}
SAMPLES = 20
CHUNK = 20000


# ------------------------------------------------------------------ mapping --

def load_mapping(path: str) -> Dict[str, Tuple[str, ...]]:
    """key -> tuple of canonical sets (one entry, or several for dual-suite holdouts)."""
    out: Dict[str, Tuple[str, ...]] = {}
    if path.endswith(".parquet"):
        import pyarrow.parquet as pq
        table = pq.read_table(path, columns=["key", "destination"])
        for k, d in zip(table.column("key").to_pylist(), table.column("destination").to_pylist()):
            out[k] = tuple(d.split(","))
        return out
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2 or parts[0] == "key":
                continue
            out[parts[0]] = tuple(parts[1].split(","))
    return out


# ---------------------------------------------------------------- discovery --

def discover_descriptor_sets(root: str) -> Dict[str, List[str]]:
    """label -> [set directories holding structure.lmdb]."""
    out: Dict[str, List[str]] = defaultdict(list)
    for d, _dirs, files in os.walk(root):
        if "structure.lmdb" in files or os.path.isdir(os.path.join(d, "structure.lmdb")):
            out[os.path.basename(d.rstrip(os.sep))].append(d)
    return {k: sorted(v) for k, v in out.items()}


def discover_graph_sets(root: str) -> Dict[str, List[str]]:
    """label -> [graph LMDB files]. merged/merged.lmdb wins over sibling shards."""
    out: Dict[str, List[str]] = defaultdict(list)
    for d, dirs, files in os.walk(root):
        if os.path.basename(d) == "merged":
            continue
        merged = os.path.join(d, "merged", "merged.lmdb")
        if os.path.isfile(merged):
            out[os.path.basename(d.rstrip(os.sep))].append(merged)
            dirs[:] = []
            continue
        lmdbs = sorted(os.path.join(d, f) for f in files if f.endswith(".lmdb") and not f.endswith(".tmp"))
        if lmdbs:
            out[os.path.basename(d.rstrip(os.sep))].extend(lmdbs)
    return {k: sorted(v) for k, v in out.items()}


# ------------------------------------------------------------------ reading --

def _open(path: str) -> lmdb.Environment:
    return lmdb.open(path, subdir=os.path.isdir(path), readonly=True, lock=False,
                     readahead=False, meminit=False, max_readers=4096)


def read_keys(path: str, prefix: str = "") -> Set[str]:
    env = _open(path)
    try:
        with env.begin() as txn:
            return {prefix + k.decode("utf-8", errors="replace")
                    for k in txn.cursor().iternext(keys=True, values=False) if k not in META_KEYS}
    finally:
        env.close()


def _graph_record_keys(path: str) -> List[bytes]:
    env = _open(path)
    try:
        with env.begin() as txn:
            return [k for k in txn.cursor().iternext(keys=True, values=False) if k not in META_KEYS]
    finally:
        env.close()


def _read_mol_names(args: Tuple[str, List[bytes]]) -> Tuple[str, List[Tuple[str, Optional[str], str]]]:
    """Worker: (path, record keys) -> (path, [(record key, mol_name or None, error)])."""
    from qtaim_embed.data.lmdb import load_graph_from_serialized

    path, keys = args
    out: List[Tuple[str, Optional[str], str]] = []
    env = _open(path)
    try:
        with env.begin() as txn:
            for k in keys:
                raw = txn.get(k)
                try:
                    obj = pickle.loads(raw)
                    graph = load_graph_from_serialized(obj["molecule_graph"] if isinstance(obj, dict) else obj)
                    out.append((k.decode(), str(graph.mol_name), ""))
                except Exception as e:
                    out.append((k.decode(), None, f"{type(e).__name__}: {str(e)[:80]}"))
    finally:
        env.close()
    return path, out


def read_graph_names(files: Iterable[str], workers: int) -> Dict[str, List[Tuple[str, Optional[str], str]]]:
    tasks = []
    for path in files:
        keys = _graph_record_keys(path)
        tasks.extend((path, keys[i:i + CHUNK]) for i in range(0, len(keys), CHUNK))
    result: Dict[str, List[Tuple[str, Optional[str], str]]] = defaultdict(list)
    if workers <= 1:
        for t in tasks:
            path, rows = _read_mol_names(t)
            result[path].extend(rows)
        return result
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, (path, rows) in enumerate(ex.map(_read_mol_names, tasks), 1):
            result[path].extend(rows)
            if i % 20 == 0 or i == len(tasks):
                print(f"  graph chunks {i}/{len(tasks)}", file=sys.stderr)
    return result


def _vertical(key: str) -> str:
    return key.split("__", 1)[0] if "__" in key else "<no_prefix>"


def _parent_prefix(set_dir: str) -> str:
    """Per-vertical trees (<vertical>/<split>/...) store keys without the
    `<vertical>__` prefix the canonical mapping uses; rebuild it from the path."""
    return os.path.basename(os.path.dirname(os.path.abspath(set_dir))) + "__"


def _graph_set_dir(lmdb_path: str) -> str:
    d = os.path.dirname(lmdb_path)
    return os.path.dirname(d) if os.path.basename(d) == "merged" else d


# ---------------------------------------------------------------- analysis --

def compare_to_canonical(label: str, keys: Set[str], mapping: Dict[str, Tuple[str, ...]]) -> Dict:
    """Where do the keys found in set `label` canonically belong?"""
    dest: Counter = Counter()
    misplaced_by_vertical: Counter = Counter()
    unknown: List[str] = []
    misplaced_samples: List[str] = []
    for k in keys:
        d = mapping.get(k)
        if d is None:
            unknown.append(k)
            continue
        if label in d:
            dest["in_place"] += 1
            continue
        dest[",".join(d)] += 1
        misplaced_by_vertical[_vertical(k)] += 1
        if len(misplaced_samples) < SAMPLES:
            misplaced_samples.append(f"{k} -> {','.join(d)}")
    n_misplaced = sum(v for k, v in dest.items() if k != "in_place")
    return {
        "n": len(keys), "in_place": dest.get("in_place", 0), "misplaced": n_misplaced,
        "misplaced_to": {k: v for k, v in dest.items() if k != "in_place"},
        "misplaced_by_vertical": dict(misplaced_by_vertical.most_common()),
        "not_in_mapping": len(unknown), "not_in_mapping_samples": sorted(unknown)[:SAMPLES],
        "misplaced_samples": misplaced_samples,
    }


def audit(mapping: Dict[str, Tuple[str, ...]], descriptor_root: Optional[str], graph_root: Optional[str],
          sets: Optional[List[str]], workers: int, data_types: List[str],
          prefix_vertical_from_parent: bool = False) -> Tuple[Dict, List[Tuple]]:
    t0 = time.time()
    desc_sets = discover_descriptor_sets(descriptor_root) if descriptor_root else {}
    graph_sets = discover_graph_sets(graph_root) if graph_root else {}
    canonical_by_label: Dict[str, Set[str]] = defaultdict(set)
    for k, d in mapping.items():
        for s in d:
            canonical_by_label[s].add(k)

    found = set(desc_sets) | set(graph_sets)
    labels = sorted(found & set(canonical_by_label))
    if sets:
        labels = [s for s in labels if s in sets]
    ignored = {"descriptor": {k: v for k, v in desc_sets.items() if k not in canonical_by_label},
               "graph": {k: v for k, v in graph_sets.items() if k not in canonical_by_label}}

    report: Dict = {"descriptor_root": descriptor_root, "graph_root": graph_root,
                    "n_mapping_keys": len(mapping), "ignored_dirs": ignored, "sets": {}}
    desc_keys: Dict[str, Set[str]] = {}
    graph_keys: Dict[str, Set[str]] = {}
    moves: List[Tuple] = []

    for label in labels:
        entry: Dict = {"canonical_n": len(canonical_by_label.get(label, ()))}
        if label in desc_sets:
            per_type: Dict[str, Dict] = {}
            struct: Set[str] = set()
            for d in desc_sets[label]:
                pre = _parent_prefix(d) if prefix_vertical_from_parent else ""
                struct |= read_keys(os.path.join(d, "structure.lmdb"), pre)
            desc_keys[label] = struct
            for dt in data_types:
                if dt == "structure":
                    continue
                ks: Set[str] = set()
                found = False
                for d in desc_sets[label]:
                    p = os.path.join(d, f"{dt}.lmdb")
                    if os.path.exists(p):
                        found = True
                        ks |= read_keys(p, _parent_prefix(d) if prefix_vertical_from_parent else "")
                if not found:
                    per_type[dt] = {"present": False}
                    continue
                per_type[dt] = {"present": True, "n": len(ks),
                                "missing_vs_structure": len(struct - ks),
                                "extra_vs_structure": len(ks - struct),
                                "extra_samples": sorted(ks - struct)[:SAMPLES]}
            entry["descriptor"] = {"dirs": desc_sets[label], "structure_n": len(struct),
                                   "vs_canonical": compare_to_canonical(label, struct, mapping),
                                   "types_vs_structure": per_type}
            print(f"[descriptor] {label}: {len(struct)} structure keys", file=sys.stderr)

        if label in graph_sets:
            files = graph_sets[label]
            print(f"[graph] {label}: reading mol_name from {len(files)} LMDB(s)", file=sys.stderr)
            names = read_graph_names(files, workers)
            counts: Counter = Counter()
            unreadable: List[str] = []
            gk: Set[str] = set()
            where: Dict[str, Tuple[str, str]] = {}
            for path, rows in names.items():
                pre = _parent_prefix(_graph_set_dir(path)) if prefix_vertical_from_parent else ""
                for rec_key, mol, err in rows:
                    if mol is not None:
                        mol = pre + mol
                    if mol is None:
                        counts["unreadable"] += 1
                        if len(unreadable) < SAMPLES:
                            unreadable.append(f"{path}:{rec_key}: {err}")
                        continue
                    if mol in gk:
                        counts["duplicate_within_set"] += 1
                    gk.add(mol)
                    where[mol] = (path, rec_key)
            graph_keys[label] = gk
            vs = compare_to_canonical(label, gk, mapping)
            for mol, (path, rec_key) in where.items():
                d = mapping.get(mol)
                if d is None or label not in d:
                    moves.append((mol, label, path, rec_key, ",".join(d) if d else "NOT_IN_MAPPING"))
            g: Dict = {"files": files, "n_records": sum(len(r) for r in names.values()),
                       "n_unique_mol_names": len(gk), "unreadable": counts["unreadable"],
                       "unreadable_samples": unreadable,
                       "duplicate_within_set": counts["duplicate_within_set"], "vs_canonical": vs}
            if label in desc_keys:
                struct = desc_keys[label]
                g["not_in_same_set_descriptor"] = len(gk - struct)
                g["not_in_same_set_descriptor_samples"] = sorted(gk - struct)[:SAMPLES]
            entry["graph"] = g
            print(f"[graph] {label}: {len(gk)} graphs, {vs['misplaced']} misplaced", file=sys.stderr)

        report["sets"][label] = entry

    # canonical keys with no graph in their set: elsewhere in graphs, or absent everywhere
    all_graph = set().union(*graph_keys.values()) if graph_keys else set()
    all_desc = set().union(*desc_keys.values()) if desc_keys else set()
    for label in labels:
        entry = report["sets"][label]
        canon = canonical_by_label.get(label, set())
        if label in graph_keys:
            missing = canon - graph_keys[label]
            elsewhere = missing & all_graph
            absent = missing - all_graph
            in_desc = absent & desc_keys.get(label, set())
            entry["graph"]["canonical_missing"] = {
                "total": len(missing), "in_other_graph_set": len(elsewhere),
                "absent_from_all_graphs": len(absent),
                "absent_but_descriptor_present": len(in_desc),
                "absent_and_no_descriptor": len(absent) - len(in_desc),
                "absent_samples": sorted(absent)[:SAMPLES],
                "absent_by_vertical": dict(Counter(_vertical(k) for k in absent).most_common()),
            }
        if label in desc_keys:
            missing = canon - desc_keys[label]
            entry["descriptor"]["canonical_missing"] = {
                "total": len(missing), "in_other_descriptor_set": len(missing & all_desc),
                "absent_from_all_descriptor_sets": len(missing - all_desc),
                "absent_samples": sorted(missing - all_desc)[:SAMPLES],
            }

    # the same key in two graph sets is legitimate only for dual-suite holdout keys
    cross: Counter = Counter()
    cross_samples: List[str] = []
    for i, a in enumerate(labels):
        for b in labels[i + 1:]:
            if a not in graph_keys or b not in graph_keys:
                continue
            for k in graph_keys[a] & graph_keys[b]:
                d = mapping.get(k, ())
                if a in d and b in d:
                    continue
                cross[f"{a}&{b}"] += 1
                if len(cross_samples) < SAMPLES:
                    cross_samples.append(f"{k} in {a},{b}")
    report["graph_cross_set_duplicates"] = dict(cross)
    report["graph_cross_set_duplicate_samples"] = cross_samples
    report["elapsed_sec"] = round(time.time() - t0, 1)
    report["problems"] = problems(report)
    return report, moves


def problems(report: Dict) -> List[str]:
    out: List[str] = []
    for label, e in report["sets"].items():
        for side in ("descriptor", "graph"):
            s = e.get(side)
            if not s:
                continue
            vs = s["vs_canonical"]
            if vs["misplaced"]:
                out.append(f"{label}/{side}: {vs['misplaced']} records belong elsewhere {vs['misplaced_to']}")
            if vs["not_in_mapping"]:
                out.append(f"{label}/{side}: {vs['not_in_mapping']} keys not in the canonical mapping")
        g = e.get("graph")
        if g:
            if g["duplicate_within_set"]:
                out.append(f"{label}/graph: {g['duplicate_within_set']} duplicate mol_names")
            if g["unreadable"]:
                out.append(f"{label}/graph: {g['unreadable']} unreadable graph records")
    for pair, n in report.get("graph_cross_set_duplicates", {}).items():
        out.append(f"graph key in two sets {pair}: {n}")
    return out


def render(report: Dict) -> str:
    L = [f"# Split alignment: {report['graph_root'] or '-'} vs {report['descriptor_root'] or '-'}",
         f"mapping keys {report['n_mapping_keys']}, {report['elapsed_sec']} s", ""]
    L.append("| set | canonical | descriptor | desc misplaced | graphs | graph misplaced | canon w/o graph (other set / desc only / neither) | graph w/o same-set desc |")
    L.append("|---|---|---|---|---|---|---|---|")
    for label, e in report["sets"].items():
        d, g = e.get("descriptor"), e.get("graph")
        row = [label, str(e["canonical_n"])]
        row += [str(d["structure_n"]), str(d["vs_canonical"]["misplaced"])] if d else ["-", "-"]
        if g:
            cm = g["canonical_missing"]
            row += [str(g["n_unique_mol_names"]), str(g["vs_canonical"]["misplaced"]),
                    f"{cm['in_other_graph_set']} / {cm['absent_but_descriptor_present']} / {cm['absent_and_no_descriptor']}",
                    str(g.get("not_in_same_set_descriptor", "-"))]
        else:
            row += ["-"] * 4
        L.append("| " + " | ".join(row) + " |")
    L.append("")
    for side, dirs in report["ignored_dirs"].items():
        if dirs:
            L.append(f"ignored {side} dirs (label not a canonical set): {sorted(dirs)}")
    probs = report["problems"]
    L.append("ALIGNED" if not probs else f"NOT ALIGNED: {len(probs)} finding(s)")
    L.extend(f"- {p}" for p in probs)
    return "\n".join(L)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mapping", required=True, help="energy_mapping.parquet or key<TAB>destination .tsv[.gz]")
    p.add_argument("--descriptor_root", default=None)
    p.add_argument("--graph_root", default=None)
    p.add_argument("--sets", nargs="+", default=None, help="Only these set labels (train val test H1 ...)")
    p.add_argument("--data_types", nargs="+", default=list(DESCRIPTORS))
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--prefix_vertical_from_parent", action="store_true",
                   help="Per-vertical trees (<vertical>/<split>/...): prepend '<vertical>__' to every key "
                        "before matching the canonical mapping")
    p.add_argument("--report", default=None, help="Write the full JSON report here")
    p.add_argument("--moves_tsv", default=None,
                   help="Write misplaced graphs: mol_name, current set, lmdb path, record key, canonical set(s)")
    args = p.parse_args(argv)
    if not args.descriptor_root and not args.graph_root:
        p.error("give --descriptor_root and/or --graph_root")

    print(f"loading mapping {args.mapping}", file=sys.stderr)
    mapping = load_mapping(args.mapping)
    report, moves = audit(mapping, args.descriptor_root, args.graph_root, args.sets,
                          args.workers, args.data_types, args.prefix_vertical_from_parent)
    print(render(report))
    if args.report:
        with open(args.report, "w") as f:
            json.dump(report, f, indent=1)
    if args.moves_tsv:
        with open(args.moves_tsv, "w") as f:
            f.write("mol_name\tcurrent_set\tlmdb\trecord_key\tcanonical\n")
            for row in sorted(moves):
                f.write("\t".join(row) + "\n")
    return 1 if report["problems"] else 0


if __name__ == "__main__":
    sys.exit(main())
