"""Throughput of full-runner-parsl-alcf runs, one result tree per jobs-per-node x threads mix.

usage: summarize_throughput.py TREE [TREE ...]   (each TREE holds a throughput.json from
throughput_*.slurm, the job folders, and rss.log from the memory sampler)

Per tree: folders completed, wall time of the run, jobs per hour over the run and over its
steady state (the 10th to 90th percentile of completion times, which leaves out the start-up
and the tail of the last long jobs), the median and p90 seconds per job, core-seconds per job
(seconds x threads per job), and peak memory: the largest worker RSS, with the processes it
started (orca_2mkl, Multiwfn) counted to it, and the largest sum over the node, from rss.log.
Only completions logged during this run count (gbw_analysis.log is appended to by every run).
"""

import json
import os
import re
import sys
from datetime import datetime

import numpy as np

DONE = re.compile(r"^(\S+ \S+) - INFO - Completed folder (.+) in ([0-9.]+) s$")


def rss_peaks(path):
    """Peak worker and node RSS from rss.log: lines "epoch pid ppid kB command" (or the older
    "epoch pid kB" of python processes only). Each process counts to its nearest python
    ancestor, itself if it is one, so a worker includes the programs it runs."""
    snaps = {}
    with open(path) as f:
        for line in f:
            p = line.split()
            if len(p) == 3:
                snaps.setdefault(p[0], []).append((p[1], None, int(p[2]), "python"))
            elif len(p) >= 5:
                snaps.setdefault(p[0], []).append((p[1], p[2], int(p[3]), p[4]))
    worker, node = {}, 0
    for procs in snaps.values():
        parent = {pid: ppid for pid, ppid, _, _ in procs}
        is_py = {pid for pid, _, _, comm in procs if "python" in comm}
        totals = {}
        for pid, _, kb, _ in procs:
            owner, seen = pid, set()
            while owner is not None and owner not in is_py and owner not in seen:
                seen.add(owner)
                owner = parent.get(owner)
            if owner in is_py:
                totals[owner] = totals.get(owner, 0) + kb
        for pid, kb in totals.items():
            worker[pid] = max(worker.get(pid, 0), kb)
        node = max(node, sum(kb for _, _, kb, _ in procs))
    if not worker:
        return {}
    return {"max_worker_rss_gb": round(max(worker.values()) / 2**20, 2), "max_node_rss_gb": round(node / 2**20, 1)}


def tree_stats(tree):
    with open(os.path.join(tree, "throughput.json")) as f:
        run = json.load(f)
    ends, elapsed = [], []
    for dp, _, files in os.walk(tree):
        if "gbw_analysis.log" not in files:
            continue
        with open(os.path.join(dp, "gbw_analysis.log"), errors="replace") as f:
            for line in f:
                m = DONE.match(line.strip())
                if m:
                    t = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S,%f").timestamp()
                    if run["start"] - 1 <= t <= run["end"] + 1:
                        ends.append(t)
                        elapsed.append(float(m.group(3)))
    out = {"tree": os.path.basename(os.path.normpath(tree)), "workers": run["n_par"], "threads": run["threads"],
           "listed": run["listed"], "done": len(ends), "wall_s": round(run["end"] - run["start"], 1)}
    if ends:
        ends = np.sort(np.array(ends))
        el = np.array(elapsed)
        lo, hi = np.percentile(ends, [10, 90])
        steady = ((ends >= lo) & (ends <= hi)).sum()
        out.update({
            "jobs_per_hour": round(3600 * len(ends) / max(run["end"] - run["start"], 1e-9), 1),
            "jobs_per_hour_steady": round(3600 * steady / max(hi - lo, 1e-9), 1) if hi > lo else None,
            "median_s": round(float(np.median(el)), 1), "p90_s": round(float(np.percentile(el, 90)), 1),
            "core_s_per_job_median": round(float(np.median(el)) * run["threads"], 1),
            "core_s_per_job_mean": round(float(el.mean()) * run["threads"], 1),
        })
    rss = os.path.join(tree, "rss.log")
    if os.path.exists(rss):
        out.update(rss_peaks(rss))
    return out


if __name__ == "__main__":
    rows = [tree_stats(t) for t in sys.argv[1:]]
    keys = list(dict.fromkeys(k for r in rows for k in r))
    print(" | ".join(keys))
    print(" | ".join("---" for _ in keys))
    for r in rows:
        print(" | ".join(str(r.get(k, "")) for k in keys))
