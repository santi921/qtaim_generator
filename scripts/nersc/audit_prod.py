"""Audit a NERSC actinide result tree: physical checks on each folder and, with --ref,
agreement with a Multiwfn tree of the same jobs. Reads the trees only.

Per folder: validation; charges summing to the net charge (hirshfeld, adch, cm5, becke);
fuzzy spin summing to mult - 1; fuzzy density and the charge engine's grid summing to
the all-electron count; Poincare-Hopf (kept and full set); an NCP on every actinide;
the nucleus-frame and precision-limited counts; far-from-origin heavy nuclei; routines
recorded as permanently skipped. With --ref, for folders validated in both trees: the
largest per-atom difference per charge and fuzzy scheme, fuzzy_bond and the ALIE values,
and whether the QTAIM atom and bond sets are the same.

    python scripts/nersc/audit_prod.py WORK/prod_l1 [--ref WORK/mwfn_l0] [--tsv audit.tsv]
"""
import argparse
import collections
import glob
import json
import os
import re

import numpy as np

from qtaim_gen.source.utils.validation import all_electron_count, get_charge_spin_n_atoms_from_folder

ACT = {"Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr"}
CHARGES = ("hirshfeld", "adch", "cm5", "becke")
FUZZY_DENSITY = ("hirsh_fuzzy_density", "becke_fuzzy_density")
FUZZY_SPIN = ("hirsh_fuzzy_spin", "becke_fuzzy_spin")
NELEC = re.compile(r"nelec grid ([0-9.]+) of")
# acceptance for engine vs Multiwfn (docs/plans/2026-10-04-feat-one-pass-charge-engine-plan.md)
MAX_DIFF_E = 0.03
SUM_TOL_E = 0.02
NELEC_TOL_REL = 1e-3


def load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError, TypeError):
        return None


def gen(folder, name):
    for p in (os.path.join(folder, "generator", name), os.path.join(folder, name)):
        if os.path.exists(p):
            return p
    return None


def validated(folder):
    p = gen(folder, "gbw_analysis.log")
    if p is None:
        return None, ""
    log = open(p, errors="replace").read()
    status = re.findall(r"Validation status: (\w+)", log)
    return (bool(status) and status[-1] == "True"), log


def per_atom(d):
    """Per-atom ("1_Pu") or per-pair ("1_Pu_to_2_Cl") values; drops the "sum"/"abs_sum" totals."""
    return {k: v for k, v in (d or {}).items() if isinstance(v, (int, float)) and re.match(r"\d+_", k)}


def audit(folder):
    ok, log = validated(folder)
    rec = {"validated": ok, "flags": []}
    dft = get_charge_spin_n_atoms_from_folder(folder) or {}
    mol = dft.get("mol") or {}
    elements = [a["element"] for a in mol.values()]
    rec["actinides"] = sorted({e for e in elements if e in ACT})
    rec["actinide_atoms"] = [str(i) for i, a in mol.items() if a["element"] in ACT]
    rec["n_atoms"] = len(mol)
    if not ok:
        return rec
    charge = int(dft.get("charge", 0))
    mult = int(dft.get("spin", 1))
    n_elec = all_electron_count(dft)

    ch = load(gen(folder, "charge.json")) or {}
    for s in CHARGES:
        q = per_atom((ch.get(s) or {}).get("charge"))
        if q:
            dev = abs(sum(q.values()) - charge)
            rec[f"sum_{s}"] = dev
            if dev > SUM_TOL_E:
                rec["flags"].append(f"{s}_sum")
    fz = load(gen(folder, "fuzzy_full.json")) or {}
    for s in FUZZY_SPIN:
        v = per_atom(fz.get(s))
        if v:
            dev = abs(sum(v.values()) - (mult - 1))
            rec[f"sum_{s}"] = dev
            if dev > SUM_TOL_E:
                rec["flags"].append(f"{s}_sum")
    for s in FUZZY_DENSITY:
        v = per_atom(fz.get(s))
        if v and n_elec:
            dev = abs(sum(v.values()) - n_elec) / n_elec
            rec[f"rel_{s}"] = dev
            if dev > NELEC_TOL_REL:
                rec["flags"].append(f"{s}_sum")
    m = NELEC.findall(log)
    if m and n_elec:
        rec["rel_nelec_grid"] = abs(float(m[-1]) - n_elec) / n_elec
        if rec["rel_nelec_grid"] > NELEC_TOL_REL:
            rec["flags"].append("nelec_grid")

    qt = load(gen(folder, "qtaim.json")) or {}
    meta = qt.get("_meta") or {}
    eng = meta.get("qtaim_engine") or {}
    rec["ph"] = meta.get("poincare_hopf")
    rec["ph_full"] = eng.get("poincare_hopf_full_set")
    rec["nucleus_frame"] = eng.get("nucleus_frame")
    rec["precision_limited"] = eng.get("precision_limited")
    rec["far_heavy"] = len(meta.get("far_heavy_nuclei") or [])
    if rec["ph"] != 1 and rec["ph_full"] != 1:
        rec["flags"].append("poincare_hopf")
    missing_act = [f"{i}_{a['element']}" for i, a in mol.items() if a["element"] in ACT and str(i) not in qt]
    rec["actinide_ncp_missing"] = missing_act
    if missing_act:
        rec["flags"].append("actinide_ncp")
    rec["ncp_missing"] = sum(str(i) not in qt for i in mol)

    skipped = (load(gen(folder, "timings.json")) or {}).get("_skipped_routines") or {}
    rec["skipped"] = sorted(f"{r}:{v.get('reason')}" for r, v in skipped.items())
    return rec


def compare(new_folder, ref_folder):
    """Largest per-atom (per-pair) difference of each scheme present in both trees,
    relative differences of the ALIE values, and QTAIM key-set agreement."""
    out = {}
    for name, schemes, sub in (("charge.json", CHARGES, "charge"), ("fuzzy_full.json", FUZZY_DENSITY + FUZZY_SPIN, None),
                               ("bond.json", ("fuzzy_bond",), None)):
        a, b = load(gen(new_folder, name)) or {}, load(gen(ref_folder, name)) or {}
        for s in schemes:
            x = per_atom((a.get(s) or {}).get(sub) if sub else a.get(s))
            y = per_atom((b.get(s) or {}).get(sub) if sub else b.get(s))
            keys = set(x) & set(y)
            if keys:
                out[f"d_{s}"] = max(abs(x[k] - y[k]) for k in keys)
    a, b = load(gen(new_folder, "other.json")) or {}, load(gen(ref_folder, "other.json")) or {}
    alie = [abs(a[k] - b[k]) / max(abs(b[k]), 1e-12) for k in a if k.startswith("ALIE_") and isinstance(b.get(k), (int, float))
            and isinstance(a[k], (int, float))]
    if alie:
        out["rel_alie"] = max(alie)
    qa, qb = load(gen(new_folder, "qtaim.json")) or {}, load(gen(ref_folder, "qtaim.json")) or {}
    ka = {k for k in qa if not k.startswith("_")}
    kb = {k for k in qb if not k.startswith("_")}
    out["qtaim_only_new"] = sorted(ka - kb)
    out["qtaim_only_ref"] = sorted(kb - ka)
    out["ph_ref"] = (qb.get("_meta") or {}).get("poincare_hopf")
    return out


def pct(x, q):
    return float(np.percentile(x, q)) if len(x) else float("nan")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("tree")
    p.add_argument("--ref", help="Multiwfn tree of the same jobs (same relative paths)")
    p.add_argument("--tsv", help="write one row per folder here")
    args = p.parse_args()
    logs = glob.glob(os.path.join(args.tree, "**", "gbw_analysis.log"), recursive=True)
    folders = sorted({os.path.dirname(l).removesuffix(os.sep + "generator") for l in logs})
    rows = []
    for folder in folders:
        rel = os.path.relpath(folder, args.tree)
        parts = rel.split(os.sep)
        rec = {"job": rel, "group": "/".join(parts[1:3]) if len(parts) > 3 else os.path.dirname(rel)}
        rec.update(audit(folder))
        if args.ref and rec["validated"]:
            ref = os.path.join(args.ref, rel)
            ok_ref, _ = validated(ref)
            rec["ref_validated"] = ok_ref
            if ok_ref:
                rec.update(compare(folder, ref))
        rows.append(rec)

    done = [r for r in rows if r["validated"]]
    print(f"folders {len(rows)}, validated {len(done)}")
    print("\n| group | validated | flagged | flags |")
    print("| --- | --- | --- | --- |")
    by = collections.defaultdict(list)
    for r in rows:
        by[r["group"]].append(r)
    for g in sorted(by):
        rs = by[g]
        fl = collections.Counter(f for r in rs for f in r["flags"])
        print(f"| {g} | {sum(bool(r['validated']) for r in rs)}/{len(rs)} | {sum(bool(r['flags']) for r in rs)} | "
              f"{', '.join(f'{k} {v}' for k, v in fl.most_common()) or '-'} |")

    act = [r for r in done if r["actinides"]]
    print(f"\nactinide folders validated: {len(act)}, by element: "
          f"{dict(collections.Counter(a for r in act for a in r['actinides']))}")
    print(f"  actinide NCP missing: {sum(bool(r['actinide_ncp_missing']) for r in act)}; "
          f"far heavy nuclei flagged: {sum(r['far_heavy'] > 0 for r in act)}; "
          f"nucleus-frame used: {sum((r['nucleus_frame'] or 0) > 0 for r in act)}; "
          f"precision-limited rescue: {sum((r['precision_limited'] or 0) > 0 for r in act)}")
    print(f"  Poincare-Hopf kept 1: {sum(r['ph'] == 1 for r in done)}/{len(done)}, full set 1: "
          f"{sum(r['ph_full'] == 1 for r in done)}/{len(done)}, neither: {sum('poincare_hopf' in r['flags'] for r in done)}")
    for key in ("rel_nelec_grid",) + tuple(f"rel_{s}" for s in FUZZY_DENSITY):
        x = np.array([r[key] for r in done if key in r])
        if len(x):
            print(f"  {key}: median {pct(x, 50):.1e}, p99 {pct(x, 99):.1e}, max {x.max():.1e} (n {len(x)})")
    sk = collections.Counter((a, s) for r in done for a in (r["actinides"] or ["-"]) for s in r["skipped"])
    if sk:
        print("  skipped routines (element, routine:reason): " + "; ".join(f"{a} {s} x{n}" for (a, s), n in sk.most_common(12)))

    if args.ref:
        both = [r for r in done if r.get("ref_validated")]
        print(f"\nvs {args.ref}: {len(both)} folders validated in both (max per-atom |diff|, e; acceptance max {MAX_DIFF_E})")
        print("| scheme | n | median | p99 | max | over max |")
        print("| --- | --- | --- | --- | --- | --- |")
        for key in [f"d_{s}" for s in CHARGES + FUZZY_DENSITY + FUZZY_SPIN + ("fuzzy_bond",)] + ["rel_alie"]:
            x = np.array([r[key] for r in both if key in r])
            if len(x):
                over = int((x > MAX_DIFF_E).sum()) if key != "rel_alie" else "-"
                print(f"| {key[2:] if key.startswith('d_') else key} | {len(x)} | {pct(x, 50):.1e} | {pct(x, 99):.1e} | {x.max():.1e} | {over} |")
        same = sum(not r["qtaim_only_new"] and not r["qtaim_only_ref"] for r in both)
        print(f"QTAIM atom+bond key sets identical: {same}/{len(both)}; engine-only keys in {sum(bool(r['qtaim_only_new']) for r in both)}, "
              f"Multiwfn-only keys in {sum(bool(r['qtaim_only_ref']) for r in both)}")
        gained = [r for r in both if set(r["actinide_atoms"]) & set(r["qtaim_only_new"])]
        lost = [r for r in both if set(r["actinide_atoms"]) & set(r["qtaim_only_ref"])]
        print(f"actinide NCP only in the engine output (Multiwfn lost it): {len(gained)} folders; "
              f"only in Multiwfn's: {len(lost)}")
        print(f"Multiwfn Poincare-Hopf != 1 in {sum(r.get('ph_ref') != 1 for r in both)}/{len(both)}")

    if args.tsv:
        cols = sorted({k for r in rows for k in r} - {"job"})
        with open(args.tsv, "w") as f:
            f.write("job\t" + "\t".join(cols) + "\n")
            for r in rows:
                f.write(r["job"] + "\t" + "\t".join(
                    (",".join(map(str, r[c])) if isinstance(r.get(c), list) else "" if r.get(c) is None else
                     f"{r[c]:.3g}" if isinstance(r.get(c), float) else str(r[c])) for c in cols) + "\n")
        print(f"\nper-folder rows -> {args.tsv}")


if __name__ == "__main__":
    main()
