"""Rerun check for a DDEC6 port change that should not move any output: the ddec6 and
fuzzy sections of each job's ddec6_port.json against an earlier campaign.py run on the
same orca.wfx, and per-job port timings.

    python compare_perf_run.py OLD_WORK NEW_WORK "NEW_JSONL_GLOB" "PREV_PORT_JSONL_GLOB" "CHARGEMOL_JSONL_GLOB"

OLD_WORK / NEW_WORK: the --work trees of the two runs. PREV_PORT: the earlier run's
results (port_s, density_grids stage time). CHARGEMOL: a parity run's results
(chargemol_s). Prints one row per job (identical, largest difference over the values
both files hold, timings), keys found in one file only, and the totals. Keys in one
file only (e.g. fields a later branch added) do not count as a numeric difference.
"""
import glob
import json
import os
import sys
import time

old_work, new_work, new_glob, prev_glob, cm_glob = sys.argv[1:6]


def rows(pattern):
    out = {}
    for f in sorted(glob.glob(pattern)):
        for line in open(f):
            r = json.loads(line)
            out[r["job"]] = r
    return out


def maxdiff(a, b, path, bad):
    """Largest absolute difference over numeric leaves present in both; structural
    mismatches (keys in one only, list lengths, differing non-numbers) go to bad."""
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) ^ set(b)):
            bad.append(f"{path}/{k} only in {'old' if k in a else 'new'}")
        return max([maxdiff(a[k], b[k], f"{path}/{k}", bad) for k in a if k in b] or [0.0])
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            bad.append(f"{path} length {len(a)} vs {len(b)}")
        return max([maxdiff(x, y, f"{path}[{i}]", bad) for i, (x, y) in enumerate(zip(a, b))] or [0.0])
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(a - b)
    if a != b:
        bad.append(f"{path}: {str(a)[:40]} vs {str(b)[:40]}")
    return 0.0


new, prev, cm = rows(new_glob), rows(prev_glob), rows(cm_glob)
n_same = n_cmp = 0
print("job|identical|max_abs_diff|port_prev_s|port_new_s|speedup|dg_prev_s|dg_new_s|chargemol_s|new_over_chargemol")
ratios = []
tot = [0.0, 0.0, 0.0]
structure = {}
old_meta = None
for job, r in sorted(new.items()):
    if "skipped" in r or "error" in r:
        print(f"{job}|{r.get('skipped') or r.get('error')[:60]}||||||||")
        continue
    a = json.load(open(os.path.join(old_work, job, "ddec6_port.json")))
    b = json.load(open(os.path.join(new_work, job, "ddec6_port.json")))
    same = all(a[k] == b[k] for k in ("ddec6", "fuzzy"))
    bad = []
    d = max(maxdiff(a[k], b[k], k, bad) for k in ("ddec6", "fuzzy"))
    for pat in sorted({p.split("/")[0] + "/.../" + p.split("/")[-1] if p.count("/") > 1 else p for p in bad})[:6]:
        structure[pat] = structure.get(pat, 0) + 1
    if old_meta is None:
        old_meta = (job, a["_meta"].keys(), b["_meta"].keys(), os.path.getmtime(os.path.join(old_work, job, "ddec6_port.json")))
    n_cmp += 1
    n_same += same
    p = prev.get(job, {})
    c = cm.get(job, {})
    ps, ns, cs = p.get("port_s"), r.get("port_s"), c.get("chargemol_s")
    dgp = (p.get("port_stage_s") or {}).get("density_grids")
    dgn = (r.get("port_stage_s") or {}).get("density_grids")
    rat = ns / cs if ns and cs else None
    if rat:
        ratios.append(rat)
    if ps and ns and cs:
        tot[0] += ps
        tot[1] += ns
        tot[2] += cs
    print(f"{job}|{same}|{d:.3g}|{ps}|{ns}|{(ps / ns if ps and ns else float('nan')):.2f}|{dgp}|{dgn}|{cs}|"
          f"{(rat if rat else float('nan')):.2f}")
ratios.sort()
med = ratios[len(ratios) // 2] if len(ratios) % 2 else sum(ratios[len(ratios) // 2 - 1:len(ratios) // 2 + 1]) / 2
if old_meta:
    print(f"first job {old_meta[0]}: old _meta keys {sorted(old_meta[1])}, new {sorted(old_meta[2])}, "
          f"old file written {time.ctime(old_meta[3])}")
for pat, n in sorted(structure.items(), key=lambda x: -x[1]):
    print(f"structure mismatch in {n} jobs: {pat}")
print(f"identical {n_same}/{n_cmp}; new/Chargemol median {med:.2f}, total {tot[1] / tot[2]:.2f} "
      f"(previous port total {tot[0] / tot[2]:.2f}); port speedup total {tot[0] / tot[1]:.2f}x")
