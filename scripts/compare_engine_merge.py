"""Compare charge-engine result trees against the original Multiwfn results.

For each job (relative path shared by both trees), reads generator/*.json from
the original Multiwfn tree (--orig_root) and from a tree produced by
full-runner-engine (--new_root; either a --restart merge on a copy, or a run
from scratch) and reports:

  - per engine routine, per-atom / per-pair abs diffs (acceptance: median <= 0.005
    and max <= 0.03, docs/plans/2026-10-04-feat-one-pass-charge-engine-plan.md)
  - whether every other key of charge/bond/fuzzy_full.json, and qtaim.json and
    other.json, is unchanged (merge mode must not touch them)
  - wall time of the 9 Multiwfn routines (original timings.json) vs the
    charge_engine timing (new timings.json) on the same job

    python scripts/compare_engine_merge.py --job_file jobs.txt \
        --root_omol_inputs SRC/ --orig_root RES/ --new_root WORK/ --report out.json
"""

import argparse
import json
import os

import numpy as np

from qtaim_gen.source.data.multiwfn import ENGINE_ROUTINES

MEDIAN_TOL = 0.005
MAX_TOL = 0.03
CHARGE = ("hirshfeld", "adch", "cm5", "becke")
FUZZY = ("becke_fuzzy_density", "hirsh_fuzzy_density", "becke_fuzzy_spin", "hirsh_fuzzy_spin")
FILE_OF = {**{s: "charge.json" for s in CHARGE}, **{s: "fuzzy_full.json" for s in FUZZY},
           "fuzzy_bond": "bond.json"}


def load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def values(step, d):
    if d is None:
        return None
    if step in CHARGE:
        return d.get("charge")
    if step in FUZZY:
        return {k: v for k, v in d.items() if k not in ("sum", "abs_sum")}
    return d


def compare_job(orig, new):
    rec = {"diffs": {}, "missing": [], "pair_set_changes": 0, "untouched_changed": []}
    files = {}
    for name in ("charge.json", "bond.json", "fuzzy_full.json", "qtaim.json", "other.json", "timings.json"):
        files[name] = (load(os.path.join(orig, "generator", name)), load(os.path.join(new, "generator", name)))

    for step, fname in FILE_OF.items():
        o_all, n_all = files[fname]
        o = values(step, (o_all or {}).get(step))
        n = values(step, (n_all or {}).get(step))
        if o is None:
            continue  # e.g. spin routines on closed shell
        if n is None:
            rec["missing"].append(step)
            continue
        keys = set(o) & set(n)
        if step == "fuzzy_bond":
            rec["pair_set_changes"] += len(set(o) ^ set(n))
        rec["diffs"][step] = [abs(n[k] - o[k]) for k in keys if isinstance(o[k], (int, float))]

    for fname in ("charge.json", "bond.json", "fuzzy_full.json"):
        o_all, n_all = files[fname]
        for k in set(o_all or {}) - ENGINE_ROUTINES:
            if (n_all or {}).get(k) != o_all[k]:
                rec["untouched_changed"].append(f"{fname}:{k}")
    for fname in ("qtaim.json", "other.json"):
        o_all, n_all = files[fname]
        if o_all != n_all:
            rec["untouched_changed"].append(fname)

    o_t, n_t = files["timings.json"]
    o_t, n_t = o_t or {}, n_t or {}
    mwfn = [o_t.get(r) for r in ENGINE_ROUTINES if isinstance(o_t.get(r), (int, float)) and o_t.get(r) > 0]
    rec["mwfn_engine_routines_s"] = sum(mwfn) if mwfn else None
    rec["charge_engine_s"] = n_t.get("charge_engine")
    return rec


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--job_file", required=True, help="input job folders, one per line (as given to the runner)")
    p.add_argument("--root_omol_inputs", required=True)
    p.add_argument("--orig_root", required=True, help="original Multiwfn results root")
    p.add_argument("--new_root", required=True, help="full-runner-engine results root")
    p.add_argument("--report", help="per-job JSON report")
    args = p.parse_args()

    with open(args.job_file) as f:
        jobs = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]

    per_job = {}
    no_engine = []
    for job in jobs:
        rel = os.path.relpath(job, args.root_omol_inputs)
        orig, new = os.path.join(args.orig_root, rel), os.path.join(args.new_root, rel)
        rec = compare_job(orig, new)
        if not rec["charge_engine_s"]:
            no_engine.append(rel)
        per_job[rel] = rec

    print(f"jobs: {len(jobs)}  without a charge_engine timing (not processed or failed): {len(no_engine)}")
    print("\nscheme | jobs | values | median_abs_diff | max_abs_diff | pass (median<=0.005, max<=0.03)")
    for step in CHARGE + FUZZY + ("fuzzy_bond",):
        d = [np.array(r["diffs"][step]) for r in per_job.values() if r["diffs"].get(step)]
        if not d:
            continue
        a = np.concatenate(d)
        ok = np.median(a) <= MEDIAN_TOL and a.max() <= MAX_TOL
        print(f"{step} | {len(d)} | {len(a)} | {np.median(a):.2e} | {a.max():.2e} | {ok}")

    missing = sum(bool(r["missing"]) for r in per_job.values())
    pairs = sum(r["pair_set_changes"] for r in per_job.values())
    changed = [(k, r["untouched_changed"]) for k, r in per_job.items() if r["untouched_changed"]]
    print(f"\njobs missing an engine routine: {missing}")
    print(f"fuzzy_bond pairs present on one side only: {pairs}")
    print(f"jobs where non-engine data changed: {len(changed)}")
    for k, v in changed[:10]:
        print(f"  {k}: {v[:5]}")

    sp = [(r["mwfn_engine_routines_s"], r["charge_engine_s"]) for r in per_job.values()
          if r["mwfn_engine_routines_s"] and r["charge_engine_s"]]
    if sp:
        m, e = np.array(sp).T
        ratio = m / e
        print(f"\ntiming over {len(sp)} jobs: Multiwfn 9 routines {m.sum():.0f} s total "
              f"(median {np.median(m):.1f} s/job), engine {e.sum():.0f} s total "
              f"(median {np.median(e):.1f} s/job); speedup median {np.median(ratio):.1f}x, "
              f"p10 {np.percentile(ratio, 10):.1f}x, p90 {np.percentile(ratio, 90):.1f}x, "
              f"aggregate {m.sum() / e.sum():.1f}x")

    if args.report:
        with open(args.report, "w") as f:
            json.dump({"no_engine": no_engine, "jobs": {k: {kk: vv for kk, vv in v.items() if kk != "diffs"}
                                                        for k, v in per_job.items()}}, f, indent=1)


if __name__ == "__main__":
    main()
