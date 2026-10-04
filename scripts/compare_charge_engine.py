"""Compare charge_engine_worker output against Multiwfn per-step outputs.

For each <name>.json in --engine_dir, parses the matching Multiwfn outputs in
<bench_dir>/<name>/sep/<step>.out (as written by scripts/bench_charge_engines.py)
with the production parsers and reports, per scheme, the per-atom absolute
differences. Acceptance (docs/plans/2026-10-04-feat-one-pass-charge-engine-plan.md):
median <= 0.005 and max <= 0.03 per scheme.

    python scripts/compare_charge_engine.py --bench_dir BENCH --engine_dir ENGINE
"""

import argparse
import glob
import json
import os

import numpy as np

from qtaim_gen.source.core.omol import _parse_routine_out

MEDIAN_TOL = 0.005
MAX_TOL = 0.03
FUZZY = ("becke_fuzzy_density", "hirsh_fuzzy_density", "becke_fuzzy_spin", "hirsh_fuzzy_spin")
CHARGE = ("hirshfeld", "adch", "cm5", "becke")
BOND = ("fuzzy_bond",)


def per_atom(step, d):
    """Per-atom values keyed by atom label, plus scalar extras for reporting."""
    if step in BOND:
        return d, {"n_bonds": len(d)}
    if step in FUZZY:
        vals = d[step]
        return {k: v for k, v in vals.items() if k not in ("sum", "abs_sum")}, {"sum": vals.get("sum")}
    return d["charge"], {"dipole_mag": d.get("dipole", {}).get("mag")}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bench_dir", required=True)
    p.add_argument("--engine_dir", required=True)
    p.add_argument("--csv", help="optional per-atom diff CSV")
    args = p.parse_args()

    rows = []
    diffs = {}
    for path in sorted(glob.glob(os.path.join(args.engine_dir, "*.json"))):
        name = os.path.basename(path)[: -len(".json")]
        with open(path) as f:
            eng = json.load(f)
        sep = os.path.join(args.bench_dir, name, "sep")
        for step in CHARGE + FUZZY + BOND:
            out = os.path.join(sep, f"{step}.out")
            if step not in eng or not os.path.isfile(out):
                continue
            ref = _parse_routine_out(step, out, set(FUZZY))
            ref_vals, ref_extra = per_atom(step, ref)
            eng_vals, eng_extra = per_atom(step, eng[step])
            keys = sorted(set(ref_vals) & set(eng_vals))
            if set(ref_vals) != set(eng_vals):
                print(f"WARN {name} {step}: label mismatch {sorted(set(ref_vals) ^ set(eng_vals))[:4]}")
            d = np.array([abs(eng_vals[k] - ref_vals[k]) for k in keys])
            diffs.setdefault(step, []).append(d)
            extra_key = next(iter(ref_extra))
            ex = (
                abs(eng_extra[extra_key] - ref_extra[extra_key])
                if ref_extra[extra_key] is not None and eng_extra[extra_key] is not None
                else float("nan")
            )
            worst = keys[int(d.argmax())]
            rows.append((name[:40], step, len(keys), np.median(d), d.max(), worst, extra_key, ex))
            if args.csv:
                with open(args.csv, "a") as f:
                    for k in keys:
                        f.write(f"{name},{step},{k},{ref_vals[k]},{eng_vals[k]}\n")

    print("job | scheme | atoms | median_abs_diff | max_abs_diff | worst_atom | extra | extra_abs_diff")
    for r in rows:
        print(f"{r[0]} | {r[1]} | {r[2]} | {r[3]:.5f} | {r[4]:.5f} | {r[5]} | {r[6]} | {r[7]:.5f}")

    print("\nscheme | jobs | atoms | median_abs_diff | max_abs_diff | pass (median<=0.005, max<=0.03)")
    for step in CHARGE + FUZZY + BOND:
        if step not in diffs:
            continue
        d = np.concatenate(diffs[step])
        ok = np.median(d) <= MEDIAN_TOL and d.max() <= MAX_TOL
        print(f"{step} | {len(diffs[step])} | {len(d)} | {np.median(d):.5f} | {d.max():.5f} | {ok}")


if __name__ == "__main__":
    main()
