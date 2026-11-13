"""Compare charge-engine result trees against the original Multiwfn results.

For each job (relative path shared by both trees), reads generator/*.json from
the original Multiwfn tree (--orig_root) and from a tree produced by
full-runner-engine (--new_root; either a --restart merge on a copy, or a run
from scratch) and reports:

  - per engine routine, per-atom / per-pair abs diffs (acceptance: median <= 0.005
    and max <= 0.03, docs/plans/2026-10-04-feat-one-pass-charge-engine-plan.md)
  - whether every other key of charge/bond/fuzzy_full.json, and qtaim.json and
    other.json, is unchanged (merge mode must not touch them): numbers within
    atol 1e-9 + rtol 1e-10 (a printed last digit can flip), NaN equal to NaN; a
    changed key is reported with its largest difference
  - for the surface engine's fields of other.json: the eight ALIE_* values of
    other_alie (volume and areas relative, the rest absolute), the planarity values
    of other_geometry, and the ESP_* values of other_esp (reported without a pass:
    the exact ESP differs from Multiwfn's libreta by design); other.json must be
    unchanged otherwise
  - wall time of the Multiwfn routines the engines cover (original timings.json)
    vs the charge_engine + surface_engine timings (new timings.json) on the same job
  - which original records carry the known Multiwfn fuzzy bugs (tracker #28:
    all-zero hirsh_fuzzy_density, open-shell spin sums != mult - 1,
    hirsh_fuzzy_spin holding the density, alpha-only fuzzy_bond at any
    multiplicity, and doubled fuzzy_bond from an all-alpha .wfn of an
    unrestricted singlet, hirsh_fuzzy_density rebuilt as Z - q_hirshfeld by
    the #28 repair (commit 2105913; normalized charges, so not the integral),
    and an empty stored fuzzy_bond);
    those diffs are reported apart from the clean-original comparison, together
    with the engine's own physical checks (spin sums, density sums)

Only jobs completed on both sides are compared (new: a charge_engine timing;
original: a generator/timings.json).

    python scripts/compare_engine_merge.py --job_file jobs.txt \
        --root_omol_inputs SRC/ --orig_root RES/ --new_root WORK/ --report out.json
"""

import argparse
import json
import math
import os

import numpy as np
from rdkit import Chem

from qtaim_gen.source.core.parse_qtaim import dft_inp_to_dict

from qtaim_gen.source.data.multiwfn import SURFACE_ENGINE_ROUTINES
from qtaim_gen.source.utils import provenance

MEDIAN_TOL = 0.005
MAX_TOL = 0.03
CHARGE = ("hirshfeld", "adch", "cm5", "becke", "vdd", "mbis")
FUZZY = ("becke_fuzzy_density", "hirsh_fuzzy_density", "becke_fuzzy_spin", "hirsh_fuzzy_spin",
         "mbis_fuzzy_density", "mbis_fuzzy_spin", "elf_fuzzy", "grad_norm_rho_fuzzy", "laplacian_rho_fuzzy")
# other_alie (surface engine): relative tolerance for volume/areas, absolute otherwise
ALIE_TOL = {"ALIE_Volume": ("rel", 1e-4), "ALIE_Overall_surface_area": ("rel", 1e-4),
            "ALIE_Positive_surface_area": ("rel", 1e-4), "ALIE_Negative_surface_area": ("abs", 1e-3),
            "ALIE_Minimal_value": ("abs", 1e-3), "ALIE_Maximal_value": ("abs", 1e-3),
            "ALIE_Overall_skewness": ("abs", 1e-3), "ALIE_Surface_Density": ("abs", 1e-3)}
# other_geometry (surface engine): planarity, Angstrom
GEOMETRY_TOL = {k: 1e-5 for k in ("mpp_full", "mpp_heavy", "sdp_full", "sdp_heavy")}
# other.json fields each surface-engine routine writes
OTHER_FIELDS = {"other_alie": lambda k: k.startswith("ALIE_"), "other_geometry": lambda k: k in GEOMETRY_TOL,
                "other_esp": lambda k: k.startswith("ESP_")}
FILE_OF = {**{s: "charge.json" for s in CHARGE}, **{s: "fuzzy_full.json" for s in FUZZY},
           "fuzzy_bond": "bond.json"}
PERIODIC_TABLE = Chem.GetPeriodicTable()


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
    "hirsh_fuzzy_density": {"hirsh_density_zero", "hirsh_density_rebuilt"},
    "becke_fuzzy_spin": {"spin_sum_wrong"},
    "hirsh_fuzzy_spin": {"spin_sum_wrong", "hirsh_spin_is_density"},
    "mbis_fuzzy_spin": {"spin_sum_wrong"},
    "fuzzy_bond": {"spin_sum_wrong", "fuzzy_bond_alpha_only", "fuzzy_bond_doubled", "fuzzy_bond_empty"},
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


def _leaves(d, prefix=""):
    if isinstance(d, dict):
        for k, v in d.items():
            yield from _leaves(v, f"{prefix}/{k}")
    elif isinstance(d, (list, tuple)):
        for i, v in enumerate(d):
            yield from _leaves(v, f"{prefix}/{i}")
    else:
        yield prefix, d


def is_num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def changed_fields(o, n, rtol=1e-10, atol=1e-9):
    """Leaves of o and n that differ: numbers beyond atol + rtol * max(|a|, |b|), NaN
    equal to NaN, anything else by ==. Returns (path, abs diff or None) pairs."""
    lo, ln = dict(_leaves(o)), dict(_leaves(n))
    out = []
    for k in set(lo) | set(ln):
        if k not in lo or k not in ln:
            out.append((k, None))
            continue
        a, b = lo[k], ln[k]
        if is_num(a) and is_num(b):
            if math.isnan(a) or math.isnan(b):
                if not (math.isnan(a) and math.isnan(b)):
                    out.append((k, None))
            elif abs(a - b) > atol + rtol * max(abs(a), abs(b)):
                out.append((k, abs(a - b)))
        elif a != b:
            out.append((k, None))
    return out


def largest(fields):
    """The (path, diff) of fields with the largest diff; a non-numeric change counts as largest."""
    return max(fields, key=lambda f: math.inf if f[1] is None else f[1])


def qtaim_same(o, n, rtol=1e-10, atol=1e-9):
    """Same CP set and every field equal, numbers within rtol/atol: two Multiwfn
    runs of one job differ at ~1e-15 (summation order), and a printed value can
    flip its last digit (esp_total by 1e-10). cp_num is ignored, as Multiwfn's
    CP index follows search order."""
    if not isinstance(o, dict) or not isinstance(n, dict):
        return o == n
    # "_" keys (_meta) are record provenance, not critical points
    o = {k: v for k, v in o.items() if not k.startswith("_")}
    n = {k: v for k, v in n.items() if not k.startswith("_")}
    if set(o) != set(n):
        return False
    for cp in o:
        lo = {k: v for k, v in _leaves(o[cp]) if not k.endswith("/cp_num")}
        ln = {k: v for k, v in _leaves(n[cp]) if not k.endswith("/cp_num")}
        if set(lo) != set(ln):
            return False
        for k, a in lo.items():
            b = ln[k]
            if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
                if not abs(a - b) <= atol + rtol * max(abs(a), abs(b)):
                    return False
            elif a != b:
                return False
    return True


def compare_job(orig, new):
    rec = {"diffs": {}, "missing": [], "pair_set_changes": 0, "untouched_changed": [],
           "untouched_largest": {}, "orig_flags": [], "engine_checks": {}}
    files = {}
    for name in ("charge.json", "bond.json", "fuzzy_full.json", "qtaim.json", "other.json", "timings.json"):
        files[name] = (load(os.path.join(orig, "generator", name)), load(os.path.join(new, "generator", name)))
    o_t, n_t = files["timings.json"]
    rec["charge_engine_s"] = (n_t or {}).get("charge_engine")
    # the charge- and surface-engine routines of the new tree (its provenance.json): the rest
    # must be unchanged, and the Multiwfn baseline time covers exactly these (QTAIM is not in
    # charge_engine_s, so its Multiwfn time is not in the baseline either)
    prov_n = provenance.load(new)
    engine_routines = provenance.engine_produced(prov_n) - {"qtaim"}
    rec["engine_versions"] = ",".join(map(str, provenance.engine_versions(prov_n))) or "none"
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
    elif o_h:
        q = (((files["charge.json"][0] or {}).get("hirshfeld") or {}).get("charge")) or {}
        if set(o_h) <= set(q) and all(
                abs(o_h[a] - (PERIODIC_TABLE.GetAtomicNumber(a.split("_", 1)[1]) - q[a])) < 1e-6 for a in o_h):
            rec["orig_flags"].append("hirsh_density_rebuilt")
    if open_shell:
        o_bs, o_hs = fuzzy_sum(o_fz, "becke_fuzzy_spin"), fuzzy_sum(o_fz, "hirsh_fuzzy_spin")
        if o_bs is not None and abs(o_bs - (mult - 1)) > 0.01:
            rec["orig_flags"].append("spin_sum_wrong")
        if o_hs is not None and o_hs > (mult - 1) + 0.5:
            rec["orig_flags"].append("hirsh_spin_is_density")
    # unrestricted singlets carry both fuzzy_bond bugs too (the #28 recheck gates on mult > 1)
    o_b, n_b = (files["bond.json"][0] or {}).get("fuzzy_bond"), (files["bond.json"][1] or {}).get("fuzzy_bond")
    if o_b == {} and n_b:
        rec["orig_flags"].append("fuzzy_bond_empty")
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
        for step in ("becke_fuzzy_spin", "hirsh_fuzzy_spin", "mbis_fuzzy_spin"):
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

    def untouched(name, o, n):
        fields = changed_fields(o, n)
        if fields:
            rec["untouched_changed"].append(name)
            rec["untouched_largest"][name] = largest(fields)

    for fname in ("charge.json", "bond.json", "fuzzy_full.json"):
        o_all, n_all = files[fname]
        for k in set(o_all or {}) - engine_routines:
            untouched(f"{fname}:{k}", o_all[k], (n_all or {}).get(k))
    o_q, n_q = files["qtaim.json"]
    if not qtaim_same(o_q, n_q):
        rec["untouched_changed"].append("qtaim.json")
    o_o, n_o = files["other.json"]
    o_o, n_o = o_o or {}, n_o or {}
    # fields the surface engine wrote, for the routines it computed in this folder
    surface_routines = engine_routines & SURFACE_ENGINE_ROUTINES

    def engine_field(k):
        return any(OTHER_FIELDS[r](k) for r in surface_routines)

    if surface_routines:
        rec["alie_diffs"], rec["geometry_diffs"], rec["esp_diffs"] = {}, {}, {}
        for k in set(o_o) & set(n_o):
            if not (engine_field(k) and is_num(o_o[k]) and is_num(n_o[k])):
                continue
            if math.isnan(o_o[k]) and math.isnan(n_o[k]):
                continue
            # NaN on one side only (e.g. no negative facets on one side) counts as infinite
            d = math.inf if math.isnan(o_o[k]) or math.isnan(n_o[k]) else abs(n_o[k] - o_o[k])
            if k in ALIE_TOL:
                rec["alie_diffs"][k] = d / abs(o_o[k]) if ALIE_TOL[k][0] == "rel" and o_o[k] else d
            elif k in GEOMETRY_TOL:
                rec["geometry_diffs"][k] = d
            elif k.startswith("ESP_"):
                rec["esp_diffs"][k] = d
    untouched("other.json", {k: v for k, v in o_o.items() if not engine_field(k)},
              {k: v for k, v in n_o.items() if not engine_field(k)})

    mwfn = [o_t.get(r) for r in engine_routines if isinstance(o_t.get(r), (int, float)) and o_t.get(r) > 0]
    rec["mwfn_engine_routines_s"] = sum(mwfn) if mwfn else None
    rec["charge_engine_s"] += n_t.get("surface_engine") or 0.0
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

    alie = [r["alie_diffs"] for r in done.values() if r.get("alie_diffs")]
    if alie:
        print("\nother_alie (surface engine) | jobs | max diff | tolerance | pass")
        for k, (kind, tol) in ALIE_TOL.items():
            vals = [a[k] for a in alie if k in a]
            if vals:
                print(f"{k} | {len(vals)} | {max(vals):.2e} {kind} | {tol:g} | {max(vals) <= tol}")

    geometry = [r["geometry_diffs"] for r in done.values() if r.get("geometry_diffs")]
    if geometry:
        print("\nother_geometry (surface engine), Angstrom | jobs | max diff | tolerance | pass")
        for k, tol in GEOMETRY_TOL.items():
            vals = [g[k] for g in geometry if k in g]
            if vals:
                print(f"{k} | {len(vals)} | {max(vals):.2e} | {tol:g} | {max(vals) <= tol}")

    esp = {k: r["esp_diffs"] for k, r in done.items() if r.get("esp_diffs")}
    if esp:
        print("\nother_esp (surface engine, exact ESP vs the reference; libreta differs by design) "
              "| jobs | median abs diff | max abs diff | job of the max")
        for k in sorted({k for e in esp.values() for k in e}):
            vals = [(e[k], job) for job, e in esp.items() if k in e]
            d = np.array([v for v, _ in vals])
            print(f"{k} | {len(d)} | {np.median(d):.2e} | {d.max():.2e} | {max(vals)[1]}")

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
    by_entry = {}
    for job, r in done.items():
        for name, (path, d) in r["untouched_largest"].items():
            by_entry.setdefault(name, []).append((math.inf if d is None else d, path, job))
    if by_entry:
        print("changed entry | jobs | largest diff (inf: not numeric, or present on one side) | field | job")
        for name, rows in sorted(by_entry.items()):
            d, path, job = max(rows)
            print(f"  {name} | {len(rows)} | {d:.2e} | {path} | {job}")
    for k, v in changed[:10]:
        print(f"  {k}: {v[:5]}")

    sp = [(r["mwfn_engine_routines_s"], r["charge_engine_s"]) for r in done.values()
          if r.get("mwfn_engine_routines_s") and r["charge_engine_s"]]
    if sp:
        m, e = np.array(sp).T
        ratio = m / e
        print(f"\ntiming over {len(sp)} jobs: Multiwfn engine-covered routines {m.sum():.0f} s total "
              f"(median {np.median(m):.1f} s/job), engine {e.sum():.0f} s total "
              f"(median {np.median(e):.1f} s/job); speedup median {np.median(ratio):.1f}x, "
              f"p10 {np.percentile(ratio, 10):.1f}x, p90 {np.percentile(ratio, 90):.1f}x, "
              f"aggregate {m.sum() / e.sum():.1f}x")

    if args.report:
        with open(args.report, "w") as f:
            json.dump({k: {kk: vv for kk, vv in v.items() if kk != "diffs"} for k, v in per_job.items()}, f, indent=1)


if __name__ == "__main__":
    main()
