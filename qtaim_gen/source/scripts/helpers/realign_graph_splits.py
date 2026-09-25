#!/usr/bin/env python3
"""Move graphs between merged split LMDBs so every graph sits in its canonical split.

Consumes the ``--moves_tsv`` written by audit_split_alignment.py (columns
mol_name, current_set, lmdb, record_key, canonical) and never edits the source
tree: it writes a new tree under --out_root. Three phases, run in order:

  move          For each of train/val/test, stream the source merged.lmdb,
                drop graphs moving out, append graphs moving in (copied as raw
                bytes from the other split, mol_name verified), renumber keys
                0..N-1, and copy metadata with ``length`` (and
                ``processed_source_keys`` when present) updated. Copies the
                train scaler files and links (or copies) the holdout LMDBs
                unchanged. Writes realign_report.json.
  scaler_check  Stream the NEW train split, undo the old train scaler, and fit
                fresh feature and label scalers on it. Reports the per-column
                shift in mean and std (in old-std units), columns with a
                near-zero old std (pre-guard-fix scalers), and stored values
                with |x| > 1e3 (sentinel rows). Saves the refit scalers to
                <out_root>/train/merged/refit/. Changes no graph.
  rescale       Optional. Re-scale every LMDB in the new tree (train, val,
                test, holdouts) as new(old.inverse(x)) via
                apply_scalers_to_lmdb_inplace (atomic temp + os.replace), then
                install the refit scalers as the train scalers (old ones kept
                in <out_root>/train/merged/pre_refit/). A per-LMDB
                ``.rescaled`` marker makes re-runs skip finished LMDBs, so a
                crash never double-scales.

Usage (NERSC):
    python -m qtaim_gen.source.scripts.helpers.realign_graph_splits --phase move \\
        --graph_root .../OMol4M_charge_full --out_root .../OMol4M_charge_canonical \\
        --moves_tsv graph_moves.tsv
    python -m qtaim_gen.source.scripts.helpers.realign_graph_splits --phase scaler_check \\
        --out_root .../OMol4M_charge_canonical
    python -m qtaim_gen.source.scripts.helpers.realign_graph_splits --phase rescale \\
        --out_root .../OMol4M_charge_canonical      # only if the check says so
    python -m qtaim_gen.source.scripts.helpers.audit_split_alignment \\
        --mapping split_destinations.tsv.gz --graph_root .../OMol4M_charge_canonical ...
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import shutil
import sys
import time
from collections import defaultdict
from typing import Dict, List, Tuple

import lmdb
from tqdm import tqdm

from qtaim_embed.data.lmdb import load_graph_from_serialized
from qtaim_embed.data.processing import HeteroGraphStandardScalerIterative

from qtaim_gen.source.utils.scaling import (
    LMDB_MAP_SIZE,
    _METADATA_KEYS,
    apply_scalers_to_lmdb_inplace,
    save_scalers,
)

SPLITS = ("train", "val", "test")
FEAT_SCALER = "feature_scaler_iterative.pt"
LABEL_SCALER = "label_scaler_iterative.pt"
BATCH = 5000
SKIP_KEYS = {"length", "scaled", "split_name", "scaler_finalized"}


def _merged(root: str, rel: str) -> str:
    return os.path.join(root, rel, "merged", "merged.lmdb")


def _open_ro(path: str) -> lmdb.Environment:
    return lmdb.open(path, subdir=False, readonly=True, lock=False, readahead=True, meminit=False)


def _mol_name(value: bytes) -> str:
    obj = pickle.loads(value)
    return str(load_graph_from_serialized(obj["molecule_graph"] if isinstance(obj, dict) else obj).mol_name)


def _holdout_rels(root: str) -> List[str]:
    hd = os.path.join(root, "holdouts")
    if not os.path.isdir(hd):
        return []
    return sorted(os.path.relpath(os.path.dirname(d), root)
                  for d, _dirs, files in os.walk(hd) if "merged.lmdb" in files)


def _load_scaler(path: str, features: bool) -> HeteroGraphStandardScalerIterative:
    # finalized=True: the constructor otherwise resets the flag loaded from disk
    return HeteroGraphStandardScalerIterative(features_tf=features, load=True, load_path=path, finalized=True)


# -------------------------------------------------------------------- move --

def load_moves(path: str) -> List[Dict[str, str]]:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    bad = [r for r in rows if r["canonical"] not in SPLITS or r["current_set"] not in SPLITS]
    if bad:
        raise SystemExit(f"{len(bad)} move rows are not train/val/test -> train/val/test "
                         f"(first: {bad[0]}); holdout or unmapped keys need a human decision")
    seen = set()
    for r in rows:
        if r["mol_name"] in seen:
            raise SystemExit(f"mol_name listed twice in {path}: {r['mol_name']}")
        seen.add(r["mol_name"])
    return rows


def phase_move(graph_root: str, out_root: str, moves_tsv: str, holdouts: str, overwrite: bool) -> Dict:
    rows = load_moves(moves_tsv)
    for s in SPLITS:
        if not os.path.isfile(_merged(graph_root, s)):
            raise FileNotFoundError(_merged(graph_root, s))
        dst = _merged(out_root, s)
        if os.path.exists(dst) and not overwrite:
            raise SystemExit(f"{dst} exists; pass --overwrite to rebuild it")

    leaving: Dict[str, Dict[bytes, str]] = defaultdict(dict)
    arriving: Dict[str, List[Tuple[str, bytes, str]]] = defaultdict(list)
    for r in rows:
        if os.path.basename(os.path.dirname(os.path.dirname(r["lmdb"]))) != r["current_set"]:
            raise SystemExit(f"moves row lmdb path does not match current_set: {r}")
        k = r["record_key"].encode()
        leaving[r["current_set"]][k] = r["mol_name"]
        arriving[r["canonical"]].append((r["current_set"], k, r["mol_name"]))

    src_envs = {s: _open_ro(_merged(graph_root, s)) for s in SPLITS}
    report: Dict = {"graph_root": graph_root, "out_root": out_root, "moves_tsv": moves_tsv,
                    "n_moves": len(rows), "splits": {}}
    try:
        for s in SPLITS:
            dst_path = _merged(out_root, s)
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            for p in (dst_path, dst_path + "-lock"):
                if os.path.exists(p):
                    os.remove(p)
            dst = lmdb.open(dst_path, map_size=LMDB_MAP_SIZE, subdir=False, meminit=False, map_async=True)
            meta: Dict[bytes, bytes] = {}
            buf: List[Tuple[bytes, bytes]] = []
            idx = kept = dropped = 0

            def flush():
                if buf:
                    with dst.begin(write=True) as wtxn:
                        for kk, vv in buf:
                            wtxn.put(kk, vv)
                    buf.clear()

            out_keys = leaving[s]
            with src_envs[s].begin() as rtxn:
                total = rtxn.stat()["entries"]
                for k, v in tqdm(rtxn.cursor(), total=total, desc=f"{s} keep", unit="graph"):
                    if k in _METADATA_KEYS or k.decode("ascii", "replace") in SKIP_KEYS:
                        meta[k] = v
                        continue
                    if k in out_keys:
                        name = _mol_name(v)
                        if name != out_keys[k]:
                            raise RuntimeError(f"{s}:{k!r} holds {name!r}, moves file says {out_keys[k]!r}")
                        dropped += 1
                        continue
                    buf.append((str(idx).encode(), v))
                    idx += 1
                    kept += 1
                    if len(buf) >= BATCH:
                        flush()
            flush()
            if dropped != len(out_keys):
                raise RuntimeError(f"{s}: moves file lists {len(out_keys)} records leaving, found {dropped}")

            added = 0
            for src_set, k, name in tqdm(arriving[s], desc=f"{s} add", unit="graph"):
                with src_envs[src_set].begin() as rtxn:
                    v = rtxn.get(k)
                if v is None:
                    raise RuntimeError(f"{src_set}:{k!r} ({name}) not found in {_merged(graph_root, src_set)}")
                got = _mol_name(v)
                if got != name:
                    raise RuntimeError(f"{src_set}:{k!r} holds {got!r}, moves file says {name!r}")
                buf.append((str(idx).encode(), v))
                idx += 1
                added += 1
                if len(buf) >= BATCH:
                    flush()
            flush()

            old_len = pickle.loads(meta[b"length"]) if b"length" in meta else None
            with dst.begin(write=True) as wtxn:
                for mk, mv in meta.items():
                    if mk == b"length":
                        continue
                    if mk == b"processed_source_keys":
                        psk = set(pickle.loads(mv))
                        psk -= set(out_keys.values())
                        psk |= {n for _, _, n in arriving[s]}
                        mv = pickle.dumps(psk, protocol=-1)
                    wtxn.put(mk, mv)
                wtxn.put(b"length", pickle.dumps(idx, protocol=-1))
            dst.sync()
            dst.close()
            stale = dst_path + "-lock"
            if os.path.exists(stale):
                os.remove(stale)
            report["splits"][s] = {"old_length": old_len, "kept": kept, "moved_out": dropped,
                                   "moved_in": added, "new_length": idx}
            print(f"[move] {s}: kept {kept}, out {dropped}, in {added} -> {idx} graphs", file=sys.stderr)
    finally:
        for env in src_envs.values():
            env.close()

    src_train = os.path.dirname(_merged(graph_root, "train"))
    dst_train = os.path.dirname(_merged(out_root, "train"))
    for name in (FEAT_SCALER, LABEL_SCALER):
        shutil.copy2(os.path.join(src_train, name), os.path.join(dst_train, name))

    report["holdouts"] = {}
    if holdouts != "skip":
        for rel in _holdout_rels(graph_root):
            src, dst = _merged(graph_root, rel), _merged(out_root, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            if os.path.exists(dst):
                os.remove(dst)
            mode = holdouts
            if holdouts == "link":
                try:
                    os.link(src, dst)
                except OSError:
                    mode = "copy"
            if mode == "copy":
                shutil.copy2(src, dst)
            report["holdouts"][rel] = mode
    with open(os.path.join(out_root, "realign_report.json"), "w") as f:
        json.dump(report, f, indent=1)
    return report


# ------------------------------------------------------------ scaler_check --

NEAR_CONSTANT_STD = 1e-5


def _col_stats(old: HeteroGraphStandardScalerIterative, new: HeteroGraphStandardScalerIterative) -> Dict:
    """Per node type: shift of the refit relative to the old scaler, in old-std units.

    Columns where either std is below NEAR_CONSTANT_STD (constant or near-constant
    columns, or legacy 1e-6 guard values) are listed separately with raw values:
    their ratios are dominated by the guard, not by a change in the data.
    """
    out = {}
    for nt in old.mean:
        om, os_ = old.mean[nt].double(), old.std[nt].double()
        nm, ns = new.mean[nt].double(), new.std[nt].double()
        cols = range(len(om))
        near = [i for i in cols if os_[i] < NEAR_CONSTANT_STD or ns[i] < NEAR_CONSTANT_STD]
        live = [i for i in cols if i not in set(near)]
        dmean = {i: float((nm[i] - om[i]).abs() / os_[i]) for i in live}
        dstd = {i: float((ns[i] / os_[i] - 1.0).abs()) for i in live}
        out[nt] = {
            "n_cols": len(om),
            "max_mean_shift_in_old_std": max(dmean.values(), default=0.0),
            "max_rel_std_change": max(dstd.values(), default=0.0),
            "worst_cols": sorted(live, key=lambda i: -max(dmean[i], dstd[i]))[:5],
            "near_constant_cols": {i: {"old_mean": float(om[i]), "old_std": float(os_[i]),
                                       "new_mean": float(nm[i]), "new_std": float(ns[i])} for i in near},
        }
    return out


def phase_scaler_check(out_root: str, n_roundtrip: int) -> Dict:
    train = _merged(out_root, "train")
    merged_dir = os.path.dirname(train)
    old_f = _load_scaler(os.path.join(merged_dir, FEAT_SCALER), True)
    old_l = _load_scaler(os.path.join(merged_dir, LABEL_SCALER), False)
    new_f = HeteroGraphStandardScalerIterative(features_tf=True, mean={}, std={})
    new_l = HeteroGraphStandardScalerIterative(features_tf=False, mean={}, std={})

    extreme: Dict[str, int] = defaultdict(int)
    roundtrip_max = 0.0
    n = 0
    env = _open_ro(train)
    with env.begin() as txn:
        total = txn.stat()["entries"]
        for k, v in tqdm(txn.cursor(), total=total, desc="scaler_check", unit="graph"):
            if k in _METADATA_KEYS or k.decode("ascii", "replace") in SKIP_KEYS:
                continue
            obj = pickle.loads(v)
            g = load_graph_from_serialized(obj["molecule_graph"] if isinstance(obj, dict) else obj)
            for nt in g.node_types:
                x = getattr(g[nt], "feat", None)
                if x is not None and x.numel():
                    extreme[nt] += int((x.abs() > 1e3).sum())
            stored = {nt: g[nt].feat.clone() for nt in g.node_types if getattr(g[nt], "feat", None) is not None} \
                if n < n_roundtrip else None
            old_f.inverse([g])
            old_l.inverse([g])
            new_f.update([g])
            new_l.update([g])
            if stored is not None:
                back = old_f([g.clone()])[0]
                for nt, x in stored.items():
                    if x.numel():
                        roundtrip_max = max(roundtrip_max, float((back[nt].feat - x).abs().max()))
            n += 1
    env.close()
    new_f.finalize()
    new_l.finalize()
    refit_dir = os.path.join(merged_dir, "refit")
    save_scalers(new_f, new_l, refit_dir)

    report = {"train_graphs": n, "refit_dir": refit_dir,
              "stored_feat_values_abs_gt_1e3": dict(extreme),
              "inverse_roundtrip_max_abs_err": roundtrip_max,
              "features": _col_stats(old_f, new_f), "labels": _col_stats(old_l, new_l)}
    with open(os.path.join(out_root, "scaler_check.json"), "w") as f:
        json.dump(report, f, indent=1)
    return report


# ----------------------------------------------------------------- rescale --

class _Rescale:
    """Callable scaler for apply_scalers_to_lmdb_inplace: new(old.inverse(g))."""

    def __init__(self, old, new):
        self.old, self.new = old, new

    def __call__(self, graphs):
        key = "feat" if self.old.features_tf else "labels"
        dtypes = [{nt: getattr(g[nt], key).dtype for nt in g.node_types
                   if getattr(g[nt], key, None) is not None} for g in graphs]
        out = self.new(self.old.inverse(graphs))
        for g, dt in zip(out, dtypes):
            for nt, d in dt.items():
                setattr(g[nt], key, getattr(g[nt], key).to(d))
        return out


def phase_rescale(out_root: str) -> Dict:
    merged_dir = os.path.dirname(_merged(out_root, "train"))
    refit_dir = os.path.join(merged_dir, "refit")
    pre_dir = os.path.join(merged_dir, "pre_refit")
    if os.path.isdir(pre_dir):
        old_dir = pre_dir
    else:
        old_dir = merged_dir
    old_f = _load_scaler(os.path.join(old_dir, FEAT_SCALER), True)
    old_l = _load_scaler(os.path.join(old_dir, LABEL_SCALER), False)
    new_f = _load_scaler(os.path.join(refit_dir, FEAT_SCALER), True)
    new_l = _load_scaler(os.path.join(refit_dir, LABEL_SCALER), False)
    feat, label = _Rescale(old_f, new_f), _Rescale(old_l, new_l)

    done: Dict[str, int] = {}
    for rel in list(SPLITS) + _holdout_rels(out_root):
        path = _merged(out_root, rel)
        marker = path + ".rescaled"
        if os.path.exists(marker):
            print(f"[rescale] {rel}: already rescaled, skipping", file=sys.stderr)
            continue
        done[rel] = apply_scalers_to_lmdb_inplace(path, feat, label, SKIP_KEYS)
        with open(marker, "w") as f:
            json.dump({"rescaled": done[rel], "old_scaler_dir": old_dir, "new_scaler_dir": refit_dir,
                       "time": time.strftime("%Y-%m-%d %H:%M:%S")}, f)

    if not os.path.isdir(pre_dir):
        os.makedirs(pre_dir)
        for name in (FEAT_SCALER, LABEL_SCALER):
            os.replace(os.path.join(merged_dir, name), os.path.join(pre_dir, name))
            shutil.copy2(os.path.join(refit_dir, name), os.path.join(merged_dir, name))
    return done


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--phase", required=True, choices=["move", "scaler_check", "rescale"])
    p.add_argument("--out_root", required=True, help="New tree; never the source tree")
    p.add_argument("--graph_root", help="Source tree (move phase)")
    p.add_argument("--moves_tsv", help="audit_split_alignment.py --moves_tsv output (move phase)")
    p.add_argument("--holdouts", choices=["link", "copy", "skip"], default="link",
                   help="How holdout LMDBs enter the new tree (hardlink falls back to copy)")
    p.add_argument("--overwrite", action="store_true", help="Rebuild split LMDBs that already exist in out_root")
    p.add_argument("--n_roundtrip", type=int, default=1000,
                   help="scaler_check: graphs on which old(old.inverse(x)) == x is verified")
    args = p.parse_args(argv)

    if args.phase == "move":
        if not (args.graph_root and args.moves_tsv):
            p.error("--phase move needs --graph_root and --moves_tsv")
        if os.path.abspath(args.graph_root) == os.path.abspath(args.out_root):
            p.error("--out_root must differ from --graph_root")
        rep = phase_move(args.graph_root, args.out_root, args.moves_tsv, args.holdouts, args.overwrite)
    elif args.phase == "scaler_check":
        rep = phase_scaler_check(args.out_root, args.n_roundtrip)
    else:
        rep = phase_rescale(args.out_root)
    print(json.dumps(rep, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
