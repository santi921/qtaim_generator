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
  - which original records carry the known Multiwfn fuzzy bugs (tracker #28:
    all-zero hirsh_fuzzy_density, open-shell spin sums != mult - 1,
    hirsh_fuzzy_spin holding the density, alpha-only fuzzy_bond at any
    multiplicity, and doubled fuzzy_bond from an all-alpha .wfn of an
    unrestricted singlet);
    those diffs are reported apart from the clean-original comparison, together
    with the engine's own physical checks (spin sums, density sums)

Only jobs completed on both sides are compared (new: a charge_engine timing;
original: a generator/timings.json).

    python scripts/compare_engine_merge.py --job_file jobs.txt \
        --root_omol_inputs SRC/ --orig_root RES/ --new_root WORK/ --report out.json
"""

import argparse
import json
import os

import numpy as np

from qtaim_gen.source.core.parse_qtaim import dft_inp_to_dict

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


# which original-record bug signature invalidates which scheme's comparison
AFFECTED_BY = {
    "hirsh_fuzzy_density": {"hirsh_density_zero"},
    "becke_fuzzy_spin": {"spin_sum_wrong"},
    "hirsh_fuzzy_spin": {"spin_sum_wrong", "hirsh_spin_is_density"},
    "fuzzy_bond": {"spin_sum_wrong", "fuzzy_bond_alpha_only", "fuzzy_bond_doubled"},
}


def multiplicity(*folders):
    for folder in folders:
        inp = os.path.join(folder, "orca.inp")
        if os.path.isfile(inp):
            try:
                return int(dft_inp_to_dict(inp, parse_charge_spin=True)["spin"])
            except Exception:
                pass
    return None


def fuzzy_sum(fz, step):
    v = (fz or {}).get(step) or {}
    return v.get("sum")


def compare_job(orig, new):
    rec = {"diffs": {}, "missing": [], "pair_set_changes": 0, "untouched_changed": [],
           "orig_flags": [], "engine_checks": {}}
    files = {}
    for name in ("charge.json", "bond.json", "fuzzy_full.json", "qtaim.json", "other.json", "timings.json"):
        files[name] = (load(os.path.join(orig, "generator", name)), load(os.path.join(new, "generator", name)))
    o_t, n_t = files["timings.json"]
    rec["charge_engine_s"] = (n_t or {}).get("charge_engine")
    rec["complete"] = bool(o_t) and bool(rec["charge_engine_s"])
    if not rec["complete"]:
        return rec

    mult = multiplicity(new, orig)
    rec["mult"] = mult
    open_shell = mult is not None and mult != 1
    o_fz, n_fz = files["fuzzy_full.json"]

    # bug signatures in the original record
    o_h = values("hirsh_fuzzy_density", (o_fz or {}).get("hirsh_fuzzy_density"))
    if o_h is not None and sum(abs(v) for v in o_h.values()) < 1e-6:
        rec["orig_flags"].append("hirsh_density_zero")
    if open_shell:
        o_bs, o_hs = fuzzy_sum(o_fz, "becke_fuzzy_spin"), fuzzy_sum(o_fz, "hirsh_fuzzy_spin")
        if o_bs is not None and abs(o_bs - (mult - 1)) > 0.01:
            rec["orig_flags"].append("spin_sum_wrong")
        if o_hs is not None and o_hs > (mult - 1) + 0.5:
            rec["orig_flags"].append("hirsh_spin_is_density")
    # unrestricted singlets carry both fuzzy_bond bugs too (the #28 recheck gates on mult > 1)
    o_b, n_b = (files["bond.json"][0] or {}).get("fuzzy_bond"), (files["bond.json"][1] or {}).get("fuzzy_bond")
    if o_b and n_b:
        ratios = [o_b[k] / n_b[k] for k in set(o_b) & set(n_b) if n_b[k]]
        if ratios:
            r = float(np.median(ratios))
            if 0.35 < r < 0.65:
                rec["orig_flags"].append("fuzzy_bond_alpha_only")
            elif 1.9 < r < 2.1:
                rec["orig_flags"].append("fuzzy_bond_doubled")

    # engine's own physics: spin sums = mult - 1, both partitions hold the same electrons
    n_bd, n_hd = fuzzy_sum(n_fz, "becke_fuzzy_density"), fuzzy_sum(n_fz, "hirsh_fuzzy_density")
    if n_bd is not None and n_hd is not None:
        rec["engine_checks"]["density_sum_becke_minus_hirsh"] = abs(n_bd - n_hd)
    if open_shell:
        for step in ("becke_fuzzy_spin", "hirsh_fuzzy_spin"):
            v = fuzzy_sum(n_fz, step)
            if v is not None:
                rec["engine_checks"][f"{step}_minus_mult1"] = abs(v - (mult - 1))

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

    mwfn = [o_t.get(r) for r in ENGINE_ROUTINES if isinstance(o_t.get(r), (int, float)) and o_t.get(r) > 0]
    rec["mwfn_engine_routines_s"] = sum(mwfn) if mwfn else None
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
    for job in jobs:
        rel = os.path.relpath(job, args.root_omol_inputs)
        per_job[rel] = compare_job(os.path.join(args.orig_root, rel), os.path.join(args.new_root, rel))
    done = {k: r for k, r in per_job.items() if r["complete"]}
    print(f"jobs: {len(jobs)}  completed on both sides (compared): {len(done)}")

    flag_counts = {}
    for r in done.values():
        for fl in r["orig_flags"]:
            flag_counts[fl] = flag_counts.get(fl, 0) + 1
    print(f"original records with a #28 bug signature: {flag_counts or 'none'}")

    print("\nscheme | clean jobs | values | median_abs_diff | max_abs_diff | pass (median<=0.005, max<=0.03)"
          " | #28-affected jobs | their max_abs_diff")
    for step in CHARGE + FUZZY + ("fuzzy_bond",):
        clean, bad = [], []
        for r in done.values():
            if not r["diffs"].get(step):
                continue
            (bad if AFFECTED_BY.get(step, set()) & set(r["orig_flags"]) else clean).append(np.array(r["diffs"][step]))
        if not clean and not bad:
            continue
        if clean:
            a = np.concatenate(clean)
            ok = np.median(a) <= MEDIAN_TOL and a.max() <= MAX_TOL
            row = f"{step} | {len(clean)} | {len(a)} | {np.median(a):.2e} | {a.max():.2e} | {ok}"
        else:
            row = f"{step} | 0 | 0 | - | - | -"
        bad_max = f"{np.concatenate(bad).max():.2e}" if bad else "-"
        print(f"{row} | {len(bad)} | {bad_max}")

    checks = {}
    for r in done.values():
        for k, v in r["engine_checks"].items():
            checks[k] = max(checks.get(k, 0.0), v)
    print("\nengine physical checks (max over jobs): "
          + (", ".join(f"{k} {v:.2e}" for k, v in sorted(checks.items())) or "none"))

    missing = sum(bool(r["missing"]) for r in done.values())
    pairs = sum(r["pair_set_changes"] for r in done.values() if not AFFECTED_BY["fuzzy_bond"] & set(r["orig_flags"]))
    changed = [(k, r["untouched_changed"]) for k, r in done.items() if r["untouched_changed"]]
    print(f"jobs missing an engine routine: {missing}")
    print(f"fuzzy_bond pairs on one side only (clean originals): {pairs}")
    print(f"jobs where non-engine data changed: {len(changed)}")
    for k, v in changed[:10]:
        print(f"  {k}: {v[:5]}")

    sp = [(r["mwfn_engine_routines_s"], r["charge_engine_s"]) for r in done.values()
          if r.get("mwfn_engine_routines_s") and r["charge_engine_s"]]
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
            json.dump({k: {kk: vv for kk, vv in v.items() if kk != "diffs"} for k, v in per_job.items()}, f, indent=1)


if __name__ == "__main__":
    main()
