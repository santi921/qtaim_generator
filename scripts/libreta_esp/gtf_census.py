"""GTF (primitive) counts per vertical, to size a libreta memory gate. Reads only.

Exact count from an existing qtaim.out header ("Atoms: N, GTFs: P", plain or in
out_files.zip); otherwise estimated as 1.61 x "Number of basis functions" from
orca.out (local calibration, 95 jobs, ratio 1.43-2.18).

    python3 gtf_census.py RESULTS_ROOT [--verticals a b] [--n 200 | --all]
    python3 gtf_census.py --list jobs.txt          # one job folder per line
"""
import argparse
import collections
import os
import random
import re
import zipfile

GTF_PER_BASIS = 1.61
PEAK_GB_PER_GTF2 = 3.9e-7  # libreta init peak, local fit (7.4k GTFs -> 20-22 GB)
GATES = (3000, 3500, 4000, 5000)

ap = argparse.ArgumentParser()
ap.add_argument("root", nargs="?")
ap.add_argument("--verticals", nargs="*")
ap.add_argument("--n", type=int, default=200)
ap.add_argument("--all", action="store_true")
ap.add_argument("--list")
args = ap.parse_args()
random.seed(0)


def read_text(job, name, nbytes):
    for z in (os.path.join(job, "generator", "out_files.zip"), os.path.join(job, "out_files.zip")):
        try:
            with zipfile.ZipFile(z) as f:
                names = {os.path.basename(n): n for n in f.namelist()}
                if name in names:
                    with f.open(names[name]) as h:
                        return h.read(nbytes).decode("latin1")
        except Exception:
            pass
    for p in (os.path.join(job, "generator", name), os.path.join(job, name)):
        if os.path.isfile(p):
            with open(p, "rb") as h:
                return h.read(nbytes).decode("latin1")
    return None


def gtfs(job):
    t = read_text(job, "qtaim.out", 20000)
    m = t and re.search(r"Atoms:\s*(\d+),\s*GTFs:\s*(\d+)", t)
    if m:
        return int(m.group(2)), True
    t = read_text(job, "orca.out", 400000)
    m = t and re.search(r"Number of basis functions\s+\.+\s+(\d+)", t)
    if m:
        return int(round(int(m.group(1)) * GTF_PER_BASIS)), False
    return None, False


def pct(xs, q):
    return xs[min(len(xs) - 1, int(q * len(xs)))]


groups = collections.OrderedDict()
if args.list:
    with open(args.list) as f:
        jobs = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    groups["list"] = jobs if args.all or len(jobs) <= args.n else random.sample(jobs, args.n)
else:
    verticals = args.verticals or sorted(d for d in os.listdir(args.root) if os.path.isdir(os.path.join(args.root, d)))
    for v in verticals:
        jobs = [e.path for e in os.scandir(os.path.join(args.root, v)) if e.is_dir()]
        groups[v] = jobs if args.all else random.sample(jobs, min(args.n, len(jobs)))

hdr = "vertical | sampled | counted | exact | median GTF | p90 GTF | p99 GTF | max GTF | est peak at max (GB) | " + " | ".join("<=%d" % g for g in GATES)
print(hdr)
allv = []
for v, jobs in groups.items():
    vals, exact = [], 0
    for job in jobs:
        g, ex = gtfs(job)
        if g is not None:
            vals.append(g)
            exact += ex
    allv += vals
    if not vals:
        print("%s | %d | 0 | 0 | | | | | | " % (v, len(jobs)) + " | ".join("" for _ in GATES))
        continue
    vals.sort()
    frac = " | ".join("%.1f%%" % (100.0 * sum(x <= g for x in vals) / len(vals)) for g in GATES)
    print("%s | %d | %d | %d | %d | %d | %d | %d | %.1f | %s" % (
        v, len(jobs), len(vals), exact, pct(vals, 0.5), pct(vals, 0.9), pct(vals, 0.99), vals[-1],
        PEAK_GB_PER_GTF2 * vals[-1] ** 2, frac))
if allv:
    allv.sort()
    frac = " | ".join("%.1f%%" % (100.0 * sum(x <= g for x in allv) / len(allv)) for g in GATES)
    print("TOTAL | | %d | | %d | %d | %d | %d | %.1f | %s" % (
        len(allv), pct(allv, 0.5), pct(allv, 0.9), pct(allv, 0.99), allv[-1], PEAK_GB_PER_GTF2 * allv[-1] ** 2, frac))
