"""
Rewrite the global block of existing, already-scaled graph LMDBs as
[num atoms, num bonds, molecule weight, charge one-hot, spin one-hot].

Every other global column (e.g. the charge_<scheme>_dipole_mag inputs that leak
charge labels) is dropped. The result matches a fresh converter build with
charge_dipole_global=false and the same allowed_charges/allowed_spins, scaled with
a train-fit scaler: the three base columns keep their stored (train-scaled) values,
and each one-hot column is scaled with its train mean p and std sqrt(p (1 - p)),
the population statistics HeteroGraphStandardScalerIterative computes (a column
never set in train gets std 1.0 via guard_std).

Net charge and spin come from each graph's structure.lmdb record, joined on
mol_name like add_pos_z_to_graphs. A key miss, or a charge/spin outside the
allowed lists, aborts before the atomic replace.

Two steps:
  fit_onehot_stats(train graph LMDB)  -> counts, one-hot mean/std
  patch_lmdb(each graph LMDB, ...)    -> rewrite global feats + metadata in place
build_feature_scaler() writes the matching feature scaler (atom/bond untouched).
"""

import os
import pickle

import lmdb
import numpy as np
import torch
from tqdm import tqdm

from qtaim_embed.data.lmdb import load_graph_from_serialized, serialize_graph
from qtaim_embed.utils.scalers import guard_std
from qtaim_gen.source.scripts.helpers.add_pos_z_to_graphs import StructureLookup, _open_readonly
from qtaim_gen.source.utils.scaling import LMDB_MAP_SIZE, _is_metadata

BASE_GLOBAL = ["num atoms", "num bonds", "molecule weight"]


def new_global_names(allowed_charges, allowed_spins):
    """Global feature names exactly as qtaim_embed's GlobalFeaturizerGraph emits them."""
    return BASE_GLOBAL + ["charge one hot"] * len(allowed_charges) + ["spin one hot"] * len(allowed_spins)


def charge_spin(lookup: StructureLookup, mol_name: str):
    raw = lookup._get_raw(mol_name)
    if raw is None:
        raise RuntimeError(f"mol_name {mol_name!r} not found in any structure LMDB")
    rec = pickle.loads(raw)
    return int(rec["charge"]), int(rec["spin"])


def onehot(charge, spin, allowed_charges, allowed_spins, mol_name=""):
    if charge not in allowed_charges or spin not in allowed_spins:
        raise RuntimeError(f"{mol_name!r}: charge {charge} / spin {spin} outside the allowed lists")
    v = np.zeros(len(allowed_charges) + len(allowed_spins))
    v[allowed_charges.index(charge)] = 1.0
    v[len(allowed_charges) + allowed_spins.index(spin)] = 1.0
    return v


def _iter_graphs(path, desc):
    env = _open_readonly(path)
    with env.begin(write=False) as txn:
        total = txn.stat()["entries"]
        for k, v in tqdm(txn.cursor(), total=total, desc=desc, unit="graph"):
            yield k, v
    env.close()


def fit_onehot_stats(train_lmdb, lookup, allowed_charges, allowed_spins):
    """Count charge/spin over the train graphs; return n, counts, one-hot mean and std."""
    counts = np.zeros(len(allowed_charges) + len(allowed_spins))
    n = 0
    for k, v in _iter_graphs(train_lmdb, "fit " + os.path.basename(train_lmdb)):
        if _is_metadata(k, set()):
            continue
        obj = pickle.loads(v)
        g = load_graph_from_serialized(obj["molecule_graph"] if isinstance(obj, dict) else obj)
        q, s = charge_spin(lookup, g.mol_name)
        counts += onehot(q, s, allowed_charges, allowed_spins, g.mol_name)
        n += 1
    mean = counts / n
    std = guard_std(torch.tensor(np.sqrt(np.clip(mean - mean ** 2, 0.0, None)), dtype=torch.float64),
                    torch.tensor(mean, dtype=torch.float64)).numpy()
    return {"n_graphs": n, "counts": counts.tolist(), "mean": mean.tolist(), "std": std.tolist(),
            "allowed_charges": list(allowed_charges), "allowed_spins": list(allowed_spins)}


def build_feature_scaler(old_path, out_path, stats):
    """Old feature scaler with global = old base columns + the one-hot statistics."""
    d = torch.load(old_path, weights_only=False)
    n_old = int(d["dict_node_sizes"]["global"])
    if n_old != stats["n_graphs"]:
        raise RuntimeError(f"old scaler saw {n_old} global nodes, one-hot stats saw {stats['n_graphs']} train graphs")
    m = torch.tensor(stats["mean"], dtype=torch.float64)
    d["mean"]["global"] = torch.cat([d["mean"]["global"][:3].to(torch.float64), m])
    d["std"]["global"] = torch.cat([d["std"]["global"][:3].to(torch.float64),
                                    torch.tensor(stats["std"], dtype=torch.float64)])
    if d.get("sum_x2") and "global" in d["sum_x2"]:
        d["sum_x2"]["global"] = torch.cat([d["sum_x2"]["global"][:3].to(torch.float64),
                                           torch.tensor(stats["counts"], dtype=torch.float64)])
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    torch.save(d, out_path)
    return d


def patch_lmdb(graph_lmdb, lookup, stats, batch_size=2000, dry_run=False):
    """Rewrite each graph's global feat and the LMDB metadata in place. Returns counts."""
    ac, asp = stats["allowed_charges"], stats["allowed_spins"]
    names_new = new_global_names(ac, asp)
    mean = np.asarray(stats["mean"])
    std = np.asarray(stats["std"])

    src = _open_readonly(graph_lmdb)
    with src.begin(write=False) as t:
        meta_names = pickle.loads(t.get(b"feature_names"))
    src.close()
    old = list(meta_names["global"])
    if old == names_new:
        return {"patched": 0, "already_patched": True}
    if old[:3] != BASE_GLOBAL:
        raise RuntimeError(f"{graph_lmdb}: global names start {old[:3]}, expected {BASE_GLOBAL}")
    meta_names = {**meta_names, "global": names_new}
    meta_new = {
        b"feature_names": meta_names,
        b"feature_size": {nt: len(v) for nt, v in meta_names.items()},
        b"allowed_charges": list(ac),
        b"allowed_spins": list(asp),
    }

    tmp = graph_lmdb + ".globalcs.tmp"
    for p in (tmp, tmp + "-lock"):
        if os.path.exists(p):
            os.remove(p)
    dst = None if dry_run else lmdb.open(tmp, map_size=LMDB_MAP_SIZE, subdir=False, meminit=False, map_async=True)
    buf, patched = [], 0

    def flush():
        if buf and not dry_run:
            with dst.begin(write=True) as w:
                for kk, vv in buf:
                    w.put(kk, vv)
        buf.clear()

    try:
        for k, v in _iter_graphs(graph_lmdb, os.path.basename(graph_lmdb)):
            if _is_metadata(k, set()):
                buf.append((k, pickle.dumps(meta_new[k], protocol=-1) if k in meta_new else v))
                continue
            obj = pickle.loads(v)
            is_dict = isinstance(obj, dict)
            g = load_graph_from_serialized(obj["molecule_graph"] if is_dict else obj)
            feat = g["global"].feat
            if feat.shape[1] != len(old):
                raise RuntimeError(f"{g.mol_name!r}: global width {feat.shape[1]} != metadata {len(old)}")
            q, s = charge_spin(lookup, g.mol_name)
            oh = (onehot(q, s, ac, asp, g.mol_name) - mean) / std
            g["global"].feat = torch.cat([feat[:, :3], torch.tensor(oh, dtype=feat.dtype).unsqueeze(0)], dim=1)
            b = serialize_graph(g, ret=True)
            buf.append((k, pickle.dumps({"molecule_graph": b} if is_dict else b, protocol=-1)))
            patched += 1
            if len(buf) >= batch_size:
                flush()
        flush()
    except Exception:
        if dst is not None:
            dst.close()
            for p in (tmp, tmp + "-lock"):
                if os.path.exists(p):
                    os.remove(p)
        raise

    if not dry_run:
        dst.sync()
        dst.close()
        os.replace(tmp, graph_lmdb)
        if os.path.exists(tmp + "-lock"):
            os.remove(tmp + "-lock")
    return {"patched": patched, "already_patched": False}
