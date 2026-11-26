"""Compare two full-runner result trees of the same job list (e.g. trunk and a branch): per output
file and top-level key, the largest absolute and relative difference over every job folder both
trees completed, and the files or keys present in only one tree.

usage: compare_trees.py TREE_A TREE_B [--top N]

Timing fields (timings.json, timings_s, wall_s, seconds) are left out. Relative differences are
taken against max(|a|, 1e-12), so values near zero do not dominate. A row per file/key, sorted by
the absolute difference; values that are NaN in both trees count as equal.
"""

import argparse
import json
import math
import os

SKIP_FILES = {"timings.json", "throughput.json"}
SKIP_KEYS = {"timings_s", "wall_s", "timings", "seconds", "elapsed", "elapsed_s"}


def job_files(tree):
    """{folder relative to tree: {file name: path}} for every folder holding gbw_analysis.log."""
    out = {}
    for dp, _, files in os.walk(tree):
        if "gbw_analysis.log" not in files:
            continue
        found = {}
        for sub in (dp, os.path.join(dp, "generator")):
            if os.path.isdir(sub):
                for f in os.listdir(sub):
                    if f.endswith(".json") and f not in SKIP_FILES:
                        found.setdefault(f, os.path.join(sub, f))
        out[os.path.relpath(dp, tree)] = found
    return out


def diff(a, b, acc, path):
    """Largest |a - b| and relative difference into acc[path[:1]]; structure mismatches as None."""
    if isinstance(a, dict) and isinstance(b, dict):
        for k in a.keys() | b.keys():
            if k in SKIP_KEYS:
                continue
            if k not in a or k not in b:
                acc.setdefault(("only one tree",) + path + (k,), [0, 0.0, 0.0])[0] += 1
                continue
            diff(a[k], b[k], acc, path + (k,) if len(path) == 1 else path)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            acc.setdefault(("length differs",) + path, [0, 0.0, 0.0])[0] += 1
            return
        for x, y in zip(a, b):
            diff(x, y, acc, path)
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        a, b = float(a), float(b)
        if math.isnan(a) and math.isnan(b):
            return
        d = abs(a - b) if not (math.isnan(a) or math.isnan(b)) else math.inf
        row = acc.setdefault(path, [0, 0.0, 0.0])
        row[0] += 1
        row[1] = max(row[1], d)
        row[2] = max(row[2], d / max(abs(a), 1e-12))
    elif a != b:
        acc.setdefault(("value differs",) + path, [0, 0.0, 0.0])[0] += 1


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("tree_a")
    p.add_argument("tree_b")
    p.add_argument("--top", type=int, default=40)
    args = p.parse_args()
    ja, jb = job_files(args.tree_a), job_files(args.tree_b)
    common = sorted(ja.keys() & jb.keys())
    acc = {}
    for job in common:
        fa, fb = ja[job], jb[job]
        for f in fa.keys() ^ fb.keys():
            acc.setdefault(("file in one tree only", f), [0, 0.0, 0.0])[0] += 1
        for f in fa.keys() & fb.keys():
            try:
                with open(fa[f]) as x, open(fb[f]) as y:
                    a, b = json.load(x), json.load(y)
            except (OSError, ValueError):
                acc.setdefault(("unreadable", f), [0, 0.0, 0.0])[0] += 1
                continue
            diff(a, b, acc, (f,))
    print(f"jobs: {len(ja)} in A, {len(jb)} in B, {len(common)} compared")
    rows = sorted(acc.items(), key=lambda kv: -kv[1][1])
    print("file / key | values | max abs diff | max rel diff")
    print("--- | --- | --- | ---")
    for key, (n, d, r) in rows[: args.top]:
        print(f"{' / '.join(key)} | {n} | {d:.3e} | {r:.3e}")


if __name__ == "__main__":
    main()
