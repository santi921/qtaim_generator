"""Per-routine time: charge engine (provenance timings_s) in NEW vs the reference tree's
timings.json steps and, if present, its own charge-engine timings_s.

    python scripts/ibsi_lbo_reference/timing_breakdown.py NEW_TREE REF_TREE
"""
import glob
import json
import os
import sys

import numpy as np

new_root, ref_root = sys.argv[1], sys.argv[2]


def ld(p):
    try:
        return json.load(open(p))
    except (OSError, json.JSONDecodeError):
        return None


rows = []
for p in sorted(glob.glob(os.path.join(new_root, "**", "generator", "provenance.json"), recursive=True)):
    rel = os.path.relpath(os.path.dirname(p), new_root)
    ref_t = ld(os.path.join(ref_root, rel, "timings.json"))
    if ref_t is None:
        continue
    prov = ld(p) or {}
    ref_prov = ld(os.path.join(ref_root, rel, "provenance.json")) or {}
    ce = ((prov.get("engines") or {}).get("charge_engine") or {}).get("timings_s") or {}
    rce = ((ref_prov.get("engines") or {}).get("charge_engine") or {}).get("timings_s") or {}
    rows.append((ce, ld(os.path.join(os.path.dirname(p), "timings.json")) or {}, rce, ref_t))

print(f"jobs {len(rows)}  new={new_root}  ref={ref_root}")


def table(title, dicts):
    keys = sorted({k for d in dicts for k, v in d.items() if isinstance(v, (int, float))})
    print(f"\n{title} (s) | n | median | p90 | total")
    for k in keys:
        v = np.array([d[k] for d in dicts if isinstance(d.get(k), (int, float))])
        print(f"{k} | {len(v)} | {np.median(v):.1f} | {np.percentile(v, 90):.1f} | {v.sum():.0f}")


table("new charge_engine timings_s", [r[0] for r in rows])
table("new timings.json", [r[1] for r in rows])
if any(r[2] for r in rows):
    table("ref charge_engine timings_s", [r[2] for r in rows])
table("ref timings.json", [r[3] for r in rows])

pairs = {"ibsi": ["ibsi_bond"], "chelpg": ["chelpg"], "laplacian_bond": ["laplacian_bond"]}
print("\nreplaced Multiwfn step vs engine routine, per job | n | engine median | Multiwfn median | "
      "engine total | Multiwfn total | ratio of totals")
for ek, rks in pairs.items():
    for rk in rks:
        e = np.array([r[0][ek] for r in rows if ek in r[0] and rk in r[3]])
        m = np.array([r[3][rk] for r in rows if ek in r[0] and rk in r[3]])
        if len(e):
            print(f"{ek} vs {rk} | {len(e)} | {np.median(e):.1f} | {np.median(m):.1f} | {e.sum():.0f} | "
                  f"{m.sum():.0f} | {m.sum() / e.sum():.1f}x")
