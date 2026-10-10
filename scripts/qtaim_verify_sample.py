"""Stratified sample for tier 2 QTAIM verification (qtaim_verification_campaign.md).

Scans random job folders of one vertical (fixed seed) and keeps those whose
results folder has generator/out_files.zip with a CPprop.txt and whose input
folder has orca.gbw(.zstd0). Each candidate is binned by size (nuclear CPs in
the stored CPprop.txt) and multiplicity (the last token of the folder name), and
the sample is drawn round-robin across bins so rare bins are not crowded out.
Writes the input folder paths, one per line, plus a .tsv with the bins and the
Multiwfn build from the qtaim.out banner.

    python scripts/qtaim_verify_sample.py --root_omol_inputs SRC/ --root_omol_results RES/ \
        --vertical tm_react --n 100 --out sample_tm_react.txt
"""

import argparse
import os
import random
import re
import zipfile
import zlib

SIZE_BINS = (20, 50, 100, 200)


def size_bin(n):
    for i, edge in enumerate(SIZE_BINS):
        if n <= edge:
            return i
    return len(SIZE_BINS)


def inspect(results_folder, input_folder):
    if not any(os.path.isfile(os.path.join(input_folder, f)) for f in ("orca.gbw.zstd0", "orca.gbw")):
        return None
    zpath = os.path.join(results_folder, "generator", "out_files.zip")
    if not os.path.isfile(zpath):
        return None
    try:
        with zipfile.ZipFile(zpath) as z:
            names = {os.path.basename(n): n for n in z.namelist()}
            if "CPprop.txt" not in names:
                return None
            ncp = z.read(names["CPprop.txt"]).count(b"Type (3,-3)")
            build = None
            if "qtaim.out" in names:
                m = re.search(rb"update date:\s+(\S+)", z.read(names["qtaim.out"])[:4000])
                build = m.group(1).decode() if m else None
    except (zipfile.BadZipFile, OSError, zlib.error, EOFError):
        return None
    try:
        mult = int(os.path.basename(results_folder.rstrip("/")).split("_")[-1])
    except ValueError:
        mult = 0
    return {"ncp": ncp, "mult": mult, "build": build}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root_omol_inputs", required=True)
    p.add_argument("--root_omol_results", required=True)
    p.add_argument("--vertical", required=True)
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--max_scan", type=int, default=5000, help="random folders to inspect at most")
    p.add_argument("--seed", type=int, default=20261009)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    vdir = os.path.join(args.root_omol_results, args.vertical)
    folders = sorted(os.listdir(vdir))
    random.Random(args.seed).shuffle(folders)
    bins = {}
    scanned = 0
    for name in folders[: args.max_scan]:
        scanned += 1
        info = inspect(os.path.join(vdir, name), os.path.join(args.root_omol_inputs, args.vertical, name))
        if info is None:
            continue
        key = (size_bin(info["ncp"]), min(info["mult"], 3))
        bins.setdefault(key, []).append((name, info))
    picked = []
    queues = [bins[k] for k in sorted(bins)]
    while len(picked) < args.n and any(queues):
        for q in queues:
            if q and len(picked) < args.n:
                picked.append(q.pop(0))
    with open(args.out, "w") as f:
        for name, _ in picked:
            f.write(os.path.join(args.root_omol_inputs, args.vertical, name) + "\n")
    with open(os.path.splitext(args.out)[0] + ".tsv", "w") as f:
        f.write("job\tnuclear_cps\tmult\tmultiwfn_build\n")
        for name, info in picked:
            f.write(f"{args.vertical}/{name}\t{info['ncp']}\t{info['mult']}\t{info['build']}\n")
    usable = sum(len(v) for v in bins.values()) + len(picked)
    print(f"{args.vertical}: scanned {scanned}, usable {usable}, picked {len(picked)} across {len(bins)} bins -> {args.out}")


if __name__ == "__main__":
    main()
