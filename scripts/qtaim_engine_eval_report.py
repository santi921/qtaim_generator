"""Summarize qtaim_verify_stored.py --engine records (JSONL files) per engine mode:
CP sets and bond pairs against the stored Multiwfn results, Poincare-Hopf,
CPprop integrity, qtaim.json differences and time. Lists the jobs behind every
failure count, so they can be pulled for a closer look.

    python scripts/qtaim_engine_eval_report.py $WORK/qengine_*.jsonl [--show 10] \
        [--failing_list rerun.txt --root_omol_inputs SRC/]

--failing_list writes the input folders (SRC/<job>) of every job that, in engine
mode, breaks Poincare-Hopf, misses a Multiwfn CP, loses or changes a Multiwfn
bond pair or fails with an error, for a rerun.
"""

import argparse
import collections
import json
import os

import numpy as np

# largest field difference treated as print precision (relative unless noted in compare_qtaim_json)
FIELD_OK = {"esp_e": 1e-8, "esp_total": None, "grad_norm": 1e-10, "ellip_e_dens": 2e-6, "eta": 2e-6}
DEFAULT_OK = 1e-9


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("files", nargs="+")
    p.add_argument("--show", type=int, default=10, help="jobs listed per failure class")
    p.add_argument("--failing_list", help="write the input folders of engine-mode failures here")
    p.add_argument("--root_omol_inputs", help="input root the records' job paths are relative to")
    args = p.parse_args()
    recs = {}
    for fn in args.files:
        for ln in open(fn):
            if ln.strip():
                r = json.loads(ln)
                recs[r["job"]] = r  # a requeued task may repeat a job; keep the last
    status = collections.Counter(r["status"].split(":")[0] for r in recs.values())
    print(f"{len(recs)} jobs: " + ", ".join(f"{k} {v}" for k, v in status.most_common()))
    ok = [r for r in recs.values() if r["status"] == "ok" and "engine" in r]
    print(f"with engine records: {len(ok)}; stored qtaim.json present: {sum(r['engine'].get('stored_json', False) for r in ok)}; "
          f"orca.inp found: {sum(r['engine'].get('inp', False) for r in ok)}")
    stored_ph = sum(r["stored"]["poincare_hopf"] != 1 for r in ok)
    print(f"stored Multiwfn sets breaking Poincare-Hopf: {stored_ph}")

    rerun = set()
    for mode in ("multiwfn", "engine"):
        rows = [(r["job"], r["engine"][mode]) for r in ok if mode in r["engine"]]
        errors = [(j, x["error"]) for j, x in rows if "error" in x]
        rows = [(j, x) for j, x in rows if "error" not in x]
        print(f"\n== mode {mode}: {len(rows)} jobs, {len(errors)} errors")
        for j, e in errors[: args.show]:
            print(f"   error {j}: {e[:150]}")
        fails = collections.defaultdict(list)
        tot = collections.Counter()
        worst = {}
        for j, x in rows:
            miss, missw, extra = sum(x["missing"].values()), sum(x["missing_weak"].values()), sum(x["extra"].values())
            tot.update(missing=miss, missing_weak=missw, extra=extra, **{f"pairs_{k}": v for k, v in x["bcp_pairs"].items()})
            for lab in x["missing"]:
                tot[f"missing_{lab}"] += x["missing"][lab]
                tot[f"extra_{lab}"] += x["extra"][lab]
            if miss:
                fails["missing a Multiwfn CP"].append(j)
            if extra:
                fails["extra CPs"].append(j)
            if x["bcp_pairs"]["differ"] or x["bcp_pairs"]["multiwfn_only"]:
                fails["a Multiwfn bond pair lost or changed"].append(j)
            if x["bcp_pairs"]["engine_only"]:
                fails["BCPs paired that Multiwfn leaves unpaired"].append(j)
            if x["poincare_hopf"] != 1:
                fails["Poincare-Hopf != 1 (kept CPs; full set for records before f708b32+)"].append(j)
            if x.get("poincare_hopf_full_set", x["poincare_hopf"]) != 1:
                fails["Poincare-Hopf != 1 (full set)"].append(j)
            if x.get("reseed_rounds"):
                fails["reseeded"].append(j)
            if x.get("near_pairs"):
                fails["near-degenerate CP pairs flagged"].append(j)
            if mode == "engine" and (x["poincare_hopf"] != 1 or miss or x["bcp_pairs"]["differ"]
                                     or x["bcp_pairs"]["multiwfn_only"]):
                rerun.add(j)
            if x["integrity"]:
                fails["CPprop integrity problems"].append(j)
            if x["max_pos_diff"] > 1e-9:
                fails["matched CP moved > 1e-9 Bohr"].append(j)
            for cmp in ("vs_stored_cpprop", "vs_stored_json"):
                c = x.get(cmp)
                if not c:
                    continue
                if c["only_ref"] or c["only_engine"]:
                    fails[f"{cmp}: key sets differ"].append(j)
                for f, w in c["worst"].items():
                    if w["diff"] > worst.get((cmp, f), (-1.0,))[0]:
                        worst[(cmp, f)] = (w["diff"], j, w["key"])
        print("   totals: " + ", ".join(f"{k} {v}" for k, v in sorted(tot.items()) if v))
        for name, jobs in fails.items():
            print(f"   {name}: {len(jobs)} jobs, e.g. {jobs[: args.show]}")
        for cmp in ("vs_stored_cpprop", "vs_stored_json"):
            items = sorted(((f, w) for (c, f), w in worst.items() if c == cmp), key=lambda kv: -kv[1][0])
            over = [(f, w) for f, w in items if FIELD_OK.get(f, DEFAULT_OK) is not None and w[0] > FIELD_OK.get(f, DEFAULT_OK)]
            print(f"   {cmp}: largest field differences " + ", ".join(f"{f} {w[0]:.1e}" for f, w in items[:6]))
            for f, (d, j, k) in over[: args.show]:
                print(f"      over print precision: {f} {d:.1e} at {j} key {k}")
        if mode == "engine":
            rerun.update(j for j, _ in errors)
        secs = np.array([x["s"] for _, x in rows])
        if len(secs):
            print(f"   time per job: median {np.median(secs):.1f} s, p90 {np.percentile(secs, 90):.1f} s, max {secs.max():.1f} s, total {secs.sum() / 3600:.2f} h")
    if args.failing_list:
        root = args.root_omol_inputs or ""
        with open(args.failing_list, "w") as f:
            f.writelines(os.path.join(root, j) + "\n" for j in sorted(rerun))
        print(f"\n{len(rerun)} engine-mode failures written to {args.failing_list}")


if __name__ == "__main__":
    main()
