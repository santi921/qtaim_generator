"""QTAIM reference harness (qtaim engine plan, phase P0).

For each wfx, runs the local Multiwfn topology module in four cumulative
stages and times each one:

  search       production CP seeds (nuclei, pairs, triads, quads), no output
  paths        + bond paths
  props_noesp  + CPprop.txt without ESP (option 7, -1)
  full         the production input (qtaim_data(): + CPprop.txt with ESP)
  exhaustive   (with --exhaustive) qtaim_data(exhaustive=True)

The full run's CPprop.txt is kept as reference (CPprop_full.txt, all CP types)
and summarized: CP counts by type, Poincare-Hopf sum, NCPs without a nucleus,
BCPs without a connected pair. One JSON record per wfx is appended to
OUT_DIR/results.jsonl; a summary table of per-stage costs prints at the end.

    python scripts/qtaim_reference.py --wfx_list wfx.txt --out_dir DIR --nthreads 4
"""

import argparse
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_charge_engines import run_multiwfn, wfx_header  # noqa: E402

from qtaim_gen.source.core.omol import write_settings_file  # noqa: E402
from qtaim_gen.source.core.parse_qtaim import load_cpprop_full, poincare_hopf  # noqa: E402
from qtaim_gen.source.data.multiwfn import qtaim_data  # noqa: E402

SEEDS = "2\n2\n3\n4\n5\n"
STAGES = {
    "search": SEEDS + "-10\nq\n",
    "paths": SEEDS + "8\n-10\nq\n",
    "props_noesp": SEEDS + "8\n7\n-1\n-10\nq\n",
    "full": qtaim_data(exhaustive=False),
}


def summarize_cps(path):
    cps = load_cpprop_full(path)
    counts = {lab: sum(1 for c in cps if c["label"] == lab) for lab in ("NCP", "BCP", "RCP", "CCP")}
    return {
        "counts": counts,
        "poincare_hopf": poincare_hopf(cps),
        "ncp_no_nucleus": sum(1 for c in cps if c["label"] == "NCP" and c["nucleus"] is None),
        "bcp_unpaired": sum(1 for c in cps if c["label"] == "BCP" and c["connected"] is None),
    }


def run_one(wfx_src, args, results_path):
    hdr = wfx_header(wfx_src)
    name = os.path.basename(os.path.dirname(os.path.abspath(wfx_src)))
    workdir = os.path.join(args.out_dir, name)
    os.makedirs(workdir, exist_ok=True)
    wfx = os.path.join(workdir, "orca.wfx")
    if not os.path.exists(wfx):
        os.symlink(os.path.abspath(wfx_src), wfx)
    write_settings_file(workdir, n_threads=args.nthreads)
    env = dict(os.environ, OMP_STACKSIZE="1G")

    stages = dict(STAGES)
    if args.exhaustive:
        stages["exhaustive"] = qtaim_data(exhaustive=True)
    rec = {"wfx": wfx_src, "name": name, "nthreads": args.nthreads, **hdr, "stages": {}}
    for stage, text in stages.items():
        cpprop = os.path.join(workdir, "CPprop.txt")
        if os.path.exists(cpprop):
            os.remove(cpprop)
        wall, rss, rc = run_multiwfn(
            args.multiwfn, workdir, wfx, text, os.path.join(workdir, f"{stage}.out"), args.timeout, env
        )
        rec["stages"][stage] = {"wall_s": round(wall, 2), "rc": rc}
        if stage in ("full", "exhaustive") and os.path.exists(cpprop):
            kept = os.path.join(workdir, f"CPprop_{stage}.txt")
            shutil.move(cpprop, kept)
            rec[stage] = summarize_cps(kept)
        if rc == "timeout":
            break
    with open(results_path, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec), flush=True)


def summary(results_path):
    rows = [json.loads(line) for line in open(results_path)]
    print("\natoms | prims | mult | edf | search s | paths s | props (no ESP) s | ESP s | full s | ESP share | NCP/BCP/RCP/CCP | PH | unpaired BCP")
    for r in sorted(rows, key=lambda r: r["natoms"]):
        st = {k: v["wall_s"] for k, v in r["stages"].items()}
        if "full" not in st:
            print(f"{r['natoms']} | {r['nprim']} | {r['mult']} | {r['edf']} | incomplete: {st}")
            continue
        search = st["search"]
        paths = st["paths"] - st["search"]
        props = st["props_noesp"] - st["paths"]
        esp = st["full"] - st["props_noesp"]
        f = r.get("full", {})
        c = f.get("counts", {})
        print(" | ".join(str(x) for x in (
            r["natoms"], r["nprim"], r["mult"], r["edf"], search, round(paths, 2), round(props, 2), round(esp, 2),
            st["full"], f"{esp / st['full']:.0%}" if st["full"] else "-",
            "/".join(str(c.get(k, 0)) for k in ("NCP", "BCP", "RCP", "CCP")), f.get("poincare_hopf"), f.get("bcp_unpaired"),
        )))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--wfx_list", required=True, help="file with one wfx path per line")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--multiwfn", default=os.path.expanduser("~/dev/Multiwfn_3_8/Multiwfn_noGUI"))
    p.add_argument("--nthreads", type=int, default=4)
    p.add_argument("--exhaustive", action="store_true", help="also time the exhaustive production variant")
    p.add_argument("--timeout", type=int, default=6 * 3600, help="per Multiwfn run, s")
    p.add_argument("--summary_only", action="store_true")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    results_path = os.path.join(args.out_dir, "results.jsonl")
    if not args.summary_only:
        with open(args.wfx_list) as f:
            for line in f:
                if line.strip():
                    run_one(line.strip(), args, results_path)
    summary(results_path)


if __name__ == "__main__":
    main()
