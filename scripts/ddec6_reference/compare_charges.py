"""Run the DDEC6 port on a list of wfx files and compare its charges with Chargemol
references written by run_chargemol.py (one <parent folder>.json per wfx in ref_dir).

    python compare_charges.py --list wfx_list.txt --ref_dir data/ddec6_ref/wfx_pull --out results.tsv

Appends one TSV row per job (skips jobs already in --out): natoms, seconds, iterations
(port, Chargemol), max |dq| in e, and the error if the port failed.
"""

import argparse
import json
import os
import time
import traceback

from qtaim_gen.source.core import ddec6


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--list", required=True)
    p.add_argument("--ref_dir", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    done = set()
    if os.path.exists(args.out):
        with open(args.out) as f:
            done = {line.split("\t")[0] for line in f if line.strip()}
    else:
        with open(args.out, "w") as f:
            f.write("job\tnatoms\tseconds\titer_port\titer_chargemol\tmax_abs_dq\terror\n")
    for wfx in [line.strip() for line in open(args.list) if line.strip()]:
        job = os.path.basename(os.path.dirname(wfx))
        ref_path = os.path.join(args.ref_dir, job + ".json")
        if job in done or not os.path.exists(ref_path):
            continue
        with open(ref_path) as f:
            ref = json.load(f)
        q_ref = list(ref["ddec6"]["charge"].values())
        t0 = time.time()
        try:
            st = ddec6.run(wfx)
            dq = max(abs(a - b) for a, b in zip(st["net_atomic_charge"], q_ref))
            row = [job, len(q_ref), f"{time.time() - t0:.1f}", st["n_iterations"], ref["_meta"]["n_iterations"],
                   f"{dq:.3e}", ""]
        except Exception as e:  # record and continue the batch
            row = [job, len(q_ref), f"{time.time() - t0:.1f}", "", ref["_meta"]["n_iterations"], "",
                   f"{type(e).__name__}: {e}".replace("\t", " ").replace("\n", " ")[:300]]
            traceback.print_exc()
        with open(args.out, "a") as f:
            f.write("\t".join(str(x) for x in row) + "\n")
        print("\t".join(str(x) for x in row), flush=True)


if __name__ == "__main__":
    main()
