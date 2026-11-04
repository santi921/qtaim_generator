"""Summarize DDEC6 campaign results (campaign.py JSONL files), grouped by the first
path component of each job (the vertical) or, with --by basis, by the job name's last
underscore field.

    python summarize_campaign.py results_*.jsonl [--by basis]
"""

import argparse
import json
import statistics as st
from collections import defaultdict


def main():
    p = argparse.ArgumentParser()
    p.add_argument("files", nargs="+")
    p.add_argument("--by", choices=("vertical", "basis"), default="vertical")
    args = p.parse_args()
    rows = {}
    for path in args.files:
        with open(path) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    rows[r["job"]] = r
    groups = defaultdict(list)
    for r in rows.values():
        key = r["job"].split("/")[0] if args.by == "vertical" else r["job"].rstrip("/").split("_")[-1]
        groups[key].append(r)
    print("group\tn\tok\tskipped\tchargemol_stop\terrors\titer_mismatch\tmax_dq\tmax_ddipole\tmax_dspin"
          "\tmedian_atoms\tmedian_port_s\tmedian_chargemol_s\tmedian_port/chargemol")
    for key in sorted(groups):
        g = groups[key]
        ok = [r for r in g if "max_dq" in r]
        mx = lambda k: f"{max(r[k] for r in ok if k in r):.2e}" if any(k in r for r in ok) else ""
        med = lambda k: f"{st.median(r[k] for r in g if k in r):.1f}" if any(k in r for r in g) else ""
        ratio = [r["port_s"] / r["chargemol_s"] for r in ok if r.get("chargemol_s")]
        print("\t".join(str(x) for x in (
            key, len(g), len(ok), sum("skipped" in r for r in g), sum("port_chargemol_stop" in r for r in g),
            sum("error" in r for r in g), sum(r.get("iter_port") != r.get("iter_chargemol") for r in ok),
            mx("max_dq"), mx("max_ddipole"), mx("max_dspin"), med("natoms"), med("port_s"), med("chargemol_s"),
            f"{st.median(ratio):.2f}" if ratio else "")))
    for r in rows.values():
        if "error" in r:
            print("ERROR", r["job"], r["error"])


if __name__ == "__main__":
    main()
