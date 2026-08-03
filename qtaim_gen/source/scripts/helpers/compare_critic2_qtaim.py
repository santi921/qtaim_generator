"""Compare Critic2 bond critical points against Multiwfn's shipped qtaim.lmdb.

Two comparisons, in order of importance:

1. **BCP-set agreement** - which atom pairs each code finds. With
   `bonding_scheme="qtaim"` the BCP set *is* the graph edge set, so this
   validates graph topology, not just feature values. Reported as Jaccard plus
   precision/recall against Multiwfn, with the density of every one-code-only
   BCP so that a disputed weak interaction is distinguishable from a disputed
   covalent bond.
2. **Per-property deltas** on the shared BCPs, per slice (all / all-electron /
   ECP jobs), plus the BCP position offset.

Example:
    python -m qtaim_gen.source.scripts.helpers.compare_critic2_qtaim \
        --wfx_pull_root data/cross_validation_wfns/wfx_pull \
        --lmdb_root data/OMol4M_lmdbs \
        --out_csv data/cross_validation_wfns/critic2_comparison.csv
"""

import argparse
import csv
import glob
import json
import os
import pickle
import sys
from typing import List, Optional

import numpy as np

from qtaim_gen.source.core.critic2 import CONVENTION_DIVERGENT

# Fields verified to share a definition and sign convention between the codes.
# grad_norm is excluded: it is ~0 at every CP by construction, so a relative
# comparison is meaningless.
COMPARE_FIELDS = (
    "density_all",
    "lap_e_density",
    "ellip_e_dens",
    "eta",
    "det_hessian",
    "eig_hess",
    "Lagrangian_K",
    "Hamiltonian_K",
    "energy_density",
)


def load_lmdb_entry(lmdb_path: str, key: str) -> Optional[dict]:
    import lmdb

    env = lmdb.open(lmdb_path, readonly=True, lock=False, subdir=False)
    try:
        with env.begin() as txn:
            raw = txn.get(key.encode())
            return pickle.loads(raw) if raw is not None else None
    finally:
        env.close()


def bcp_keys(record: dict) -> set:
    """0-based atom-pair keys, normalized to min_max ordering."""
    keys = set()
    for k in record:
        if k == "_meta" or "_" not in k:
            continue
        try:
            i, j = (int(x) for x in k.split("_"))
        except ValueError:
            continue
        keys.add(f"{min(i, j)}_{max(i, j)}")
    return keys


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wfx_pull_root", required=True)
    parser.add_argument("--lmdb_root", required=True)
    parser.add_argument("--out_csv", required=True)
    parser.add_argument(
        "--out_sets_csv",
        default=None,
        help="optional per-job BCP-set summary CSV (default: alongside out_csv)",
    )
    args = parser.parse_args(argv)

    rows, set_rows, only_rows, missing = [], [], [], []
    for c2_path in sorted(
        glob.glob(os.path.join(args.wfx_pull_root, "*", "*", "critic2.json"))
    ):
        job_dir = os.path.dirname(c2_path)
        key = os.path.basename(job_dir)
        vertical = os.path.basename(os.path.dirname(job_dir))
        with open(c2_path) as f:
            c2 = json.load(f)

        lmdb_path = os.path.join(args.lmdb_root, vertical, "qtaim.lmdb")
        ref = load_lmdb_entry(lmdb_path, key) if os.path.isfile(lmdb_path) else None
        if ref is None:
            missing.append((vertical, key))
            continue

        meta = c2.get("_meta", {})
        has_ecp = bool(meta.get("nna_remapped"))
        c2_keys, ref_keys = bcp_keys(c2), bcp_keys(ref)
        shared = c2_keys & ref_keys
        union = c2_keys | ref_keys
        set_rows.append(
            {
                "vertical": vertical,
                "key": key,
                "n_atoms": meta.get("n_atoms"),
                "n_bcp_critic2": len(c2_keys),
                "n_bcp_multiwfn": len(ref_keys),
                "n_shared": len(shared),
                "jaccard": round(len(shared) / len(union), 6) if union else 1.0,
                "recall_vs_multiwfn": round(len(shared) / len(ref_keys), 6)
                if ref_keys
                else float("nan"),
                "precision_vs_multiwfn": round(len(shared) / len(c2_keys), 6)
                if c2_keys
                else float("nan"),
                "poincare_hopf_sum": meta.get("poincare_hopf_sum"),
                "poincare_hopf_ok": int(bool(meta.get("poincare_hopf_ok"))),
                "n_nna_remapped": len(meta.get("nna_remapped", [])),
                "ecp_job": int(has_ecp),
            }
        )

        for k in sorted(c2_keys ^ ref_keys):
            src = "critic2_only" if k in c2_keys else "multiwfn_only"
            rec = c2 if k in c2_keys else ref
            entry = rec.get(k) or rec.get("_".join(reversed(k.split("_"))), {})
            only_rows.append(
                {
                    "vertical": vertical,
                    "key": key,
                    "pair": k,
                    "found_by": src,
                    "density_all": entry.get("density_all"),
                    "ecp_job": int(has_ecp),
                }
            )

        for k in sorted(shared):
            r_entry = ref.get(k) or ref.get("_".join(reversed(k.split("_"))))
            c_entry = c2[k]
            if not r_entry:
                continue
            base = {"vertical": vertical, "key": key, "pair": k, "ecp_job": int(has_ecp)}
            for field in COMPARE_FIELDS + CONVENTION_DIVERGENT:
                a, b = c_entry.get(field), r_entry.get(field)
                if a is None or b is None:
                    continue
                rows.append(
                    {
                        **base,
                        "field": field,
                        "convention_divergent": int(field in CONVENTION_DIVERGENT),
                        "multiwfn": b,
                        "critic2": a,
                        "abs_diff": abs(a - b),
                    }
                )
            if c_entry.get("pos_ang") and r_entry.get("pos_ang"):
                offset = float(
                    np.linalg.norm(
                        np.array(c_entry["pos_ang"]) - np.array(r_entry["pos_ang"])
                    )
                )
                rows.append(
                    {
                        **base,
                        "field": "position_offset_ang",
                        "convention_divergent": 0,
                        "multiwfn": 0.0,
                        "critic2": offset,
                        "abs_diff": offset,
                    }
                )

    if not rows:
        print("no comparable data found", file=sys.stderr)
        return 1

    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    sets_csv = args.out_sets_csv or args.out_csv.replace(".csv", "_bcp_sets.csv")
    with open(sets_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(set_rows[0].keys()))
        w.writeheader()
        w.writerows(set_rows)
    if only_rows:
        only_csv = args.out_csv.replace(".csv", "_disputed_bcps.csv")
        with open(only_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(only_rows[0].keys()))
            w.writeheader()
            w.writerows(only_rows)
    print(f"wrote {len(rows)} property comparisons -> {args.out_csv}")
    print(f"wrote {len(set_rows)} per-job BCP-set rows -> {sets_csv}")
    if missing:
        print(f"jobs missing from qtaim.lmdb: {len(missing)}")
        for m in missing:
            print("  MISSING", *m)

    # ---- BCP-set agreement (the headline) ----
    print("\n=== BCP-set agreement ===")
    for name, sub in (
        ("all", set_rows),
        ("all-electron", [r for r in set_rows if not r["ecp_job"]]),
        ("ecp-jobs", [r for r in set_rows if r["ecp_job"]]),
    ):
        if not sub:
            continue
        exact = sum(1 for r in sub if r["n_shared"] == r["n_bcp_critic2"] == r["n_bcp_multiwfn"])
        print(
            f"{name:<14} jobs={len(sub):>3} exact-match={exact:>3} "
            f"({100 * exact / len(sub):.0f}%) "
            f"median-jaccard={np.median([r['jaccard'] for r in sub]):.4f} "
            f"BCPs c2={sum(r['n_bcp_critic2'] for r in sub)} "
            f"mwfn={sum(r['n_bcp_multiwfn'] for r in sub)} "
            f"PH-ok={sum(r['poincare_hopf_ok'] for r in sub)}"
        )
    if only_rows:
        print(f"\ndisputed BCPs: {len(only_rows)}")
        for src in ("critic2_only", "multiwfn_only"):
            sub = [r for r in only_rows if r["found_by"] == src]
            if not sub:
                continue
            dens = [r["density_all"] for r in sub if r["density_all"] is not None]
            print(
                f"  {src:<14} n={len(sub):>3}"
                + (
                    f" density median={np.median(dens):.5f} max={max(dens):.5f}"
                    if dens
                    else ""
                )
            )
        worst = sorted(
            (r for r in only_rows if r["density_all"] is not None),
            key=lambda r: -r["density_all"],
        )[:5]
        if worst:
            print("  highest-density disputed BCPs (these matter most):")
            for r in worst:
                print(
                    f"    {r['found_by']:<13} {r['vertical']}/{r['key']} "
                    f"pair={r['pair']} rho={r['density_all']:.5f}"
                )

    # ---- per-property deltas ----
    print("\n=== Per-property agreement on shared BCPs ===")
    print(f"{'field':<20} {'slice':<14} {'n':>5} {'median':>10} {'mean':>10} {'max':>10} {'pearson':>8}")
    for field in COMPARE_FIELDS + ("position_offset_ang",) + CONVENTION_DIVERGENT:
        base = [r for r in rows if r["field"] == field]
        if not base:
            continue
        for name, sub in (
            ("all", base),
            ("all-electron", [r for r in base if not r["ecp_job"]]),
            ("ecp-jobs", [r for r in base if r["ecp_job"]]),
        ):
            if not sub:
                continue
            d = np.array([r["abs_diff"] for r in sub])
            if field == "position_offset_ang":
                pearson = float("nan")
            else:
                a = np.array([r["multiwfn"] for r in sub])
                b = np.array([r["critic2"] for r in sub])
                pearson = np.corrcoef(a, b)[0, 1] if len(sub) > 2 else float("nan")
            tag = field + (" *" if field in CONVENTION_DIVERGENT else "")
            print(
                f"{tag:<20} {name:<14} {len(sub):>5} {np.median(d):>10.6f} "
                f"{d.mean():>10.6f} {d.max():>10.6f} {pearson:>8.4f}"
            )
    if CONVENTION_DIVERGENT:
        print(
            "\n* definition differs between codes (different uniform-electron-gas "
            "reference); disagreement expected and is not an implementation error"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
