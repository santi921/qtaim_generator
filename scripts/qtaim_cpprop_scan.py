"""Scan archived CPprop.txt files for damage (validation.cpprop_integrity).

For every results job folder, reads generator/out_files.zip (CPprop.txt and
qtaim.out) and generator/qtaim.json, and reports files with NUL bytes, gaps or
repeats in the CP numbering, incomplete blocks, Hessian signs contradicting the
CP type, or a (3,-1) count different from qtaim.out's. For each damaged file it
also says whether qtaim.json still matches qtaim.out (then only the archive is
bad) or not (then the parsed record is suspect too). Reads only; writes the TSV.

    python scripts/qtaim_cpprop_scan.py --root_omol_results RES/ [--verticals a b] \\
        --out cpprop_scan.tsv [--workers 8]
    python scripts/qtaim_cpprop_scan.py --job_file results_folders.txt --out cpprop_scan.tsv
"""

import argparse
import collections
import json
import os
import zipfile
from concurrent.futures import ProcessPoolExecutor

from qtaim_gen.source.utils.validation import QTAIM_COUNT_PATTERN, cpprop_integrity


def scan(folder):
    gen = os.path.join(folder, "generator")
    row = {"folder": folder, "status": "ok", "problems": "", "reported_bcp": "", "json_ncp": "", "json_bcp": ""}
    try:
        with zipfile.ZipFile(os.path.join(gen, "out_files.zip")) as z:
            names = {os.path.basename(n): n for n in z.namelist()}
            if "CPprop.txt" not in names:
                row["status"] = "no_cpprop"
                return row
            reported = None
            if "qtaim.out" in names:
                found = QTAIM_COUNT_PATTERN.findall(z.read(names["qtaim.out"]).decode("latin1"))
                reported = int(found[-1]) if found else None
            problems = cpprop_integrity(z.read(names["CPprop.txt"]), reported_bcp=reported)
    except FileNotFoundError:
        row["status"] = "no_zip"
        return row
    except (zipfile.BadZipFile, OSError) as e:
        row["status"] = f"bad_zip: {e}"
        return row
    row["reported_bcp"] = "" if reported is None else reported
    if problems:
        row["status"] = "corrupt"
        row["problems"] = "; ".join(problems)
    try:
        with open(os.path.join(gen, "qtaim.json")) as f:
            d = json.load(f)
        # validate_qtaim_dict's convention: bond CP keys contain "_"
        row["json_ncp"] = sum("_" not in k for k in d)
        row["json_bcp"] = sum("_" in k for k in d)
    except (OSError, ValueError):
        pass
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root_omol_results", help="RES/<vertical>/<job>/generator/out_files.zip")
    p.add_argument("--verticals", nargs="*", help="default: every directory under the root")
    p.add_argument("--job_file", help="results job folders, one per line (instead of the root)")
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()

    if args.job_file:
        with open(args.job_file) as f:
            folders = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
    else:
        root = args.root_omol_results
        verticals = args.verticals or sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
        folders = [os.path.join(root, v, j) for v in verticals for j in sorted(os.listdir(os.path.join(root, v)))]

    counts = collections.Counter()
    by_vertical = collections.defaultdict(collections.Counter)
    cols = ("folder", "status", "problems", "reported_bcp", "json_ncp", "json_bcp")
    with open(args.out, "w") as out, ProcessPoolExecutor(args.workers) as pool:
        out.write("\t".join(cols) + "\n")
        for row in pool.map(scan, folders, chunksize=64):
            status = row["status"].split(":")[0]
            counts[status] += 1
            by_vertical[os.path.basename(os.path.dirname(row["folder"].rstrip("/")))][status] += 1
            if status != "ok":
                out.write("\t".join(str(row[c]) for c in cols) + "\n")
    print(f"{len(folders)} folders: " + ", ".join(f"{k} {v}" for k, v in counts.most_common()))
    for v, c in sorted(by_vertical.items()):
        if c["corrupt"]:
            print(f"  {v}: corrupt {c['corrupt']} of {sum(c.values())}")
    print(f"non-ok rows -> {args.out}")


if __name__ == "__main__":
    main()
