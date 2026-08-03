"""Compare HORTON charges (horton.json) against Multiwfn references (charge.lmdb).

Cross-package validation analysis: joins per-job horton.json outputs from a
wfx_pull-style tree against the shipped Multiwfn charge schemes in per-vertical
charge.lmdb files, atom by atom. Emits a long-format CSV and prints a summary
table (median/mean/max absolute difference and Pearson r per scheme pair).

Example:
    python -m qtaim_gen.source.scripts.helpers.compare_horton_charges \
        --wfx_pull_root data/cross_validation_wfns/wfx_pull \
        --lmdb_root data/OMol4M_lmdbs \
        --out_csv data/cross_validation_wfns/horton_comparison.csv
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

# (horton scheme, reference scheme in charge.lmdb)
# note: the shipped "becke" scheme is Multiwfn's ADC-corrected Becke charge
# (menu 7 option 10), a different quantity than HORTON's raw Becke population.
# The raw-vs-raw comparison uses becke_mwfn_raw.json sidecars (local Multiwfn
# reruns on the same wfx, see mwfn_raw_becke batch) when present.
SCHEME_PAIRS = [
    ("becke_horton", "becke_adc_shipped"),
    ("becke_horton", "becke_mwfn_raw"),
    ("becke_csd_horton", "becke_mwfn_raw"),  # matched radii + clip convention
    ("hirshfeld_horton", "hirshfeld"),
    ("is_horton", "hirshfeld"),  # stockholder-family reference, not same-scheme
]


def load_lmdb_entry(lmdb_path: str, key: str) -> Optional[dict]:
    import lmdb

    env = lmdb.open(lmdb_path, readonly=True, lock=False, subdir=False)
    try:
        with env.begin() as txn:
            raw = txn.get(key.encode())
            return pickle.loads(raw) if raw is not None else None
    finally:
        env.close()


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wfx_pull_root", required=True)
    parser.add_argument("--lmdb_root", required=True)
    parser.add_argument("--out_csv", required=True)
    args = parser.parse_args(argv)

    rows = []
    missing = []
    for horton_path in sorted(glob.glob(os.path.join(args.wfx_pull_root, "*", "*", "horton.json"))):
        job_dir = os.path.dirname(horton_path)
        key = os.path.basename(job_dir)
        vertical = os.path.basename(os.path.dirname(job_dir))
        with open(horton_path) as f:
            horton = json.load(f)
        csd_path = os.path.join(job_dir, "horton_csd.json")
        if os.path.isfile(csd_path):
            with open(csd_path) as f:
                horton.update(
                    {k: v for k, v in json.load(f).items() if k.endswith("_horton")}
                )

        lmdb_path = os.path.join(args.lmdb_root, vertical, "charge.lmdb")
        ref = load_lmdb_entry(lmdb_path, key) if os.path.isfile(lmdb_path) else None
        if ref is None:
            missing.append((vertical, key))
            continue

        references = {
            "becke_adc_shipped": ref.get("becke", {}),
            "hirshfeld": ref.get("hirshfeld", {}),
        }
        sidecar_path = os.path.join(job_dir, "becke_mwfn_raw.json")
        if os.path.isfile(sidecar_path):
            with open(sidecar_path) as f:
                references["becke_mwfn_raw"] = json.load(f).get("becke_raw", {})

        ecp = horton.get("_meta", {}).get("has_ecp", False)
        for h_scheme, r_scheme in SCHEME_PAIRS:
            h_charges = horton.get(h_scheme, {}).get("charge")
            r_charges = references.get(r_scheme, {}).get("charge")
            if not h_charges or not r_charges:
                continue
            for atom_key, q_h in h_charges.items():
                q_r = r_charges.get(atom_key)
                if q_r is None:
                    continue
                rows.append(
                    {
                        "vertical": vertical,
                        "key": key,
                        "pair": f"{h_scheme}_vs_{r_scheme}",
                        "atom": atom_key,
                        "element": atom_key.split("_", 1)[1],
                        "ecp_job": int(ecp),
                        "q_ref": q_r,
                        "q_horton": q_h,
                        "abs_diff": abs(q_h - q_r),
                    }
                )

    if not rows:
        print("no comparable data found", file=sys.stderr)
        return 1

    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} atom comparisons to {args.out_csv}")
    if missing:
        print(f"jobs missing from charge.lmdb: {len(missing)}")
        for m in missing:
            print("  MISSING", *m)

    # summary table
    print(f"\n{'pair':<32} {'slice':<14} {'atoms':>6} {'median':>8} {'mean':>8} {'max':>8} {'pearson':>8}")
    for pair in sorted({r["pair"] for r in rows}):
        subsets = [("all", [r for r in rows if r["pair"] == pair])]
        subsets.append(("all-electron", [r for r in subsets[0][1] if not r["ecp_job"]]))
        subsets.append(("ecp-jobs", [r for r in subsets[0][1] if r["ecp_job"]]))
        for name, sub in subsets:
            if not sub:
                continue
            d = np.array([r["abs_diff"] for r in sub])
            qr = np.array([r["q_ref"] for r in sub])
            qh = np.array([r["q_horton"] for r in sub])
            pearson = np.corrcoef(qr, qh)[0, 1] if len(sub) > 2 else float("nan")
            print(
                f"{pair:<32} {name:<14} {len(sub):>6} {np.median(d):>8.4f} "
                f"{d.mean():>8.4f} {d.max():>8.4f} {pearson:>8.4f}"
            )

    # worst cases per same-scheme pair
    for pair in ("becke_csd_horton_vs_becke_mwfn_raw", "hirshfeld_horton_vs_hirshfeld"):
        sub = sorted(
            (r for r in rows if r["pair"] == pair),
            key=lambda r: r["abs_diff"],
            reverse=True,
        )[:5]
        if sub:
            print(f"\nworst {pair}:")
            for r in sub:
                print(
                    f"  {r['vertical']}/{r['key']} {r['atom']}: "
                    f"ref={r['q_ref']:.4f} horton={r['q_horton']:.4f} "
                    f"diff={r['abs_diff']:.4f}"
                )
    return 0


if __name__ == "__main__":
    sys.exit(main())
