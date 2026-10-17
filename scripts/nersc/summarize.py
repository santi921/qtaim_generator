"""Summarize a NERSC actinide test tree: validation status, failed Multiwfn steps,
QTAIM Poincare-Hopf and far-from-origin flags, the charge-engine stamp, and
timings, overall and for folders with an actinide.

    python scripts/nersc/summarize.py WORK/TREE [--list sample.txt --src SRC] [--failed failed.txt]
"""
import argparse
import collections
import glob
import json
import os
import re

ACT = {"Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr"}
FAILED = re.compile(r"Failed (\w+) after [0-9.]+ seconds \(exit status (\d+)")


def load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def first(folder, name):
    for p in (os.path.join(folder, "generator", name), os.path.join(folder, name)):
        if os.path.exists(p):
            return p
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("tree")
    p.add_argument("--list", help="input list, to count folders that never produced output")
    p.add_argument("--src", help="input root the list is under")
    p.add_argument("--failed", help="write folders that did not validate here")
    args = p.parse_args()
    logs = glob.glob(os.path.join(args.tree, "**", "gbw_analysis.log"), recursive=True)
    folders = sorted({os.path.dirname(l).removesuffix(os.sep + "generator") for l in logs})
    stats = collections.Counter()
    steps = collections.Counter()
    failed_folders, times = [], collections.defaultdict(list)
    for folder in folders:
        log = open(first(folder, "gbw_analysis.log"), errors="replace").read()
        status = re.findall(r"Validation status: (\w+)", log)
        ok = bool(status) and status[-1] == "True"
        qt = load(first(folder, "qtaim.json") or "") or {}
        elements = {k.split("_", 1)[1] for k in ((load(first(folder, "charge.json") or "") or {}).get("hirshfeld", {}).get("charge", {}))}
        act = bool(elements & ACT)
        tag = "actinide" if act else "other"
        stats[f"{tag} folders"] += 1
        stats[f"{tag} validated"] += ok
        if not ok:
            failed_folders.append(folder)
        for step, code in FAILED.findall(log):
            steps[(tag, step, code)] += 1
        meta = qt.get("_meta", {})
        if meta:
            stats[f"{tag} PH==1"] += meta.get("poincare_hopf") == 1
            stats[f"{tag} far_heavy_nuclei flagged"] += bool(meta.get("far_heavy_nuclei"))
        t = load(first(folder, "timings.json") or "") or {}
        stats[f"{tag} charge_engine stamp"] += bool(t.get("charge_engine"))
        tot = sum(v for v in t.values() if isinstance(v, (int, float)) and v > 0)
        if tot:
            times[tag].append(tot)
        stats[f"{tag} chelpg radii file"] += os.path.exists(os.path.join(folder, "chelpg_radii.txt")) or \
            os.path.exists(os.path.join(folder, "generator", "chelpg_radii.txt"))
    for k in sorted(stats):
        print(f"{k}: {stats[k]}")
    for tag, v in sorted(times.items()):
        v = sorted(v)
        print(f"{tag} timings.json total: median {v[len(v) // 2]:.0f} s, max {v[-1]:.0f} s over {len(v)} folders")
    print("failed steps (group, step, exit status): count")
    for k, n in sorted(steps.items()):
        print(f"  {k}: {n}")
    if args.list:
        wanted = [l.strip() for l in open(args.list) if l.strip()]
        have = {os.path.relpath(f, args.tree) for f in folders}
        missing = [w for w in wanted if os.path.relpath(w, args.src or "/") not in have]
        print(f"listed {len(wanted)}, with output {len(wanted) - len(missing)}, no output {len(missing)}")
    if args.failed:
        with open(args.failed, "w") as f:
            f.write("\n".join(failed_folders) + "\n")
        print(f"{len(failed_folders)} folders that did not validate -> {args.failed}")


if __name__ == "__main__":
    main()
