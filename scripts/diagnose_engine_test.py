"""Two follow-up checks on the LRC charge-engine test.

fuzzy_bond: per job, multiplicity and the median stored/engine bond-order ratio.
A ratio near 0.5 is the alpha-only parser bug (#28); the recheck only looks at
mult > 1, so this shows whether unrestricted singlets carry it too.

qtaim: whether two QTAIM runs of the same job agree once cp_num (Multiwfn's CP
index, set by search order) is ignored: CP key sets, then the largest abs diff
of every numeric property.

    python scripts/diagnose_engine_test.py --job_file jobs.txt --root_omol_inputs SRC/ \
        --orig_root RES/ --new_root WORK/merge/ --check fuzzy_bond
    python scripts/diagnose_engine_test.py ... --orig_root WORK/ref/ --new_root WORK/scratch/ --check qtaim
"""

import argparse
import json
import os
from collections import Counter

import numpy as np

from qtaim_gen.source.core.parse_qtaim import dft_inp_to_dict


def load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def mult_of(*folders):
    for folder in folders:
        inp = os.path.join(folder, "orca.inp")
        if os.path.isfile(inp):
            try:
                return int(dft_inp_to_dict(inp, parse_charge_spin=True)["spin"])
            except Exception:
                pass
    return None


def numeric_leaves(d, prefix=""):
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(numeric_leaves(v, f"{prefix}/{k}" if prefix else k))
    elif isinstance(d, (list, tuple)):
        for i, v in enumerate(d):
            out.update(numeric_leaves(v, f"{prefix}/{i}"))
    elif isinstance(d, (int, float)) and not isinstance(d, bool):
        out[prefix] = float(d)
    return out


def check_fuzzy_bond(pairs):
    rows = []
    for rel, orig, new in pairs:
        o = (load(os.path.join(orig, "generator", "bond.json")) or {}).get("fuzzy_bond")
        n = (load(os.path.join(new, "generator", "bond.json")) or {}).get("fuzzy_bond")
        if not o or not n:
            continue
        keys = set(o) & set(n)
        ratio = float(np.median([o[k] / n[k] for k in keys if n[k]])) if keys else float("nan")
        maxd = max(abs(o[k] - n[k]) for k in keys) if keys else float("nan")
        rows.append((rel, mult_of(new, orig), ratio, maxd, len(set(o) ^ set(n))))

    def cls(r):
        if r[3] < 1e-5 and r[4] == 0:
            return "matches"
        if 0.4 < r[2] < 0.6:
            return "alpha-only (ratio ~0.5)"
        return "other mismatch"

    tally = Counter((r[1], cls(r)) for r in rows)
    print(f"fuzzy_bond, {len(rows)} jobs: (multiplicity, stored vs engine) -> jobs")
    for (m, c), n in sorted(tally.items(), key=lambda kv: (str(kv[0][0]), kv[0][1])):
        print(f"  mult {m} | {c} | {n}")
    others = [r for r in rows if cls(r) == "other mismatch"]
    for r in others[:10]:
        print(f"  other: {r[0]} mult {r[1]} ratio {r[2]:.3f} max_diff {r[3]:.3e} pairs_one_side {r[4]}")


def check_qtaim(pairs):
    n_jobs = n_same_keys = 0
    field_max = {}
    worst = []
    for rel, orig, new in pairs:
        o = load(os.path.join(orig, "generator", "qtaim.json"))
        n = load(os.path.join(new, "generator", "qtaim.json"))
        if not o or not n:
            continue
        n_jobs += 1
        if set(o) != set(n):
            worst.append((rel, f"CP keys differ: only orig {sorted(set(o) - set(n))[:4]}, only new {sorted(set(n) - set(o))[:4]}"))
            continue
        n_same_keys += 1
        job_max = 0.0
        for cp in o:
            lo = numeric_leaves({k: v for k, v in o[cp].items() if k != "cp_num"})
            ln = numeric_leaves({k: v for k, v in n[cp].items() if k != "cp_num"})
            for k in set(lo) & set(ln):
                d = abs(lo[k] - ln[k])
                field = k.split("/")[0]
                field_max[field] = max(field_max.get(field, 0.0), d)
                job_max = max(job_max, d)
        if job_max > 1e-6:
            worst.append((rel, f"max property diff {job_max:.3e}"))
    print(f"qtaim, {n_jobs} jobs: identical CP key sets in {n_same_keys}")
    print("largest abs diff per property (cp_num ignored):")
    for f, v in sorted(field_max.items(), key=lambda kv: -kv[1])[:12]:
        print(f"  {f}: {v:.3e}")
    for r in worst[:10]:
        print(f"  {r[0]}: {r[1]}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--job_file", required=True)
    p.add_argument("--root_omol_inputs", required=True)
    p.add_argument("--orig_root", required=True)
    p.add_argument("--new_root", required=True)
    p.add_argument("--check", choices=("fuzzy_bond", "qtaim"), required=True)
    args = p.parse_args()

    with open(args.job_file) as f:
        jobs = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
    pairs = []
    for job in jobs:
        rel = os.path.relpath(job, args.root_omol_inputs)
        pairs.append((rel, os.path.join(args.orig_root, rel), os.path.join(args.new_root, rel)))
    (check_fuzzy_bond if args.check == "fuzzy_bond" else check_qtaim)(pairs)


if __name__ == "__main__":
    main()
