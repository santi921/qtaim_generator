"""Stratified sample for tier 2 QTAIM verification (qtaim_verification_campaign.md).

Scans random job folders of one vertical (fixed seed) and keeps those whose
results folder has generator/out_files.zip with a CPprop.txt and whose input
folder has orca.gbw(.zstd0). Each candidate is binned by size (nuclear CPs in
the stored CPprop.txt), multiplicity (the last token of the folder name) and
element class (the heaviest present among the nuclear CPs' elements; classes
2-4 are exactly the jobs with def2 ECPs, so with EDF core densities), and the
sample is drawn round-robin across bins so rare bins are not crowded out.
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
ELEMENTS = (
    "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr "
    "Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb "
    "Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr"
).split()
ELEMENT_CLASSES = ("light main group", "3d", "heavy main group (ECP)", "4d/5d (ECP)", "lanthanide/actinide (ECP)")
NUCLEUS = re.compile(rb"Corresponding nucleus:\s+\d+\(\s*([A-Za-z]+)\s*\)")


def size_bin(n):
    for i, edge in enumerate(SIZE_BINS):
        if n <= edge:
            return i
    return len(SIZE_BINS)


def element_class(z):
    """Index into ELEMENT_CLASSES; def2 puts an ECP on every element from Rb (Z = 37)."""
    if 57 <= z <= 71 or 89 <= z <= 103:
        return 4
    if 39 <= z <= 48 or 72 <= z <= 80:
        return 3
    if z >= 37:
        return 2
    if 21 <= z <= 30:
        return 1
    return 0


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
            cpprop = z.read(names["CPprop.txt"])
            ncp = cpprop.count(b"Type (3,-3)")
            zs = {ELEMENTS.index(m.decode().capitalize()) + 1 for m in NUCLEUS.findall(cpprop)
                  if m.decode().capitalize() in ELEMENTS}
            eclass = max((element_class(z) for z in zs), default=0)
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
    return {"ncp": ncp, "mult": mult, "build": build, "eclass": eclass}


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
        key = (size_bin(info["ncp"]), min(info["mult"], 3), info["eclass"])
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
        f.write("job\tnuclear_cps\tmult\telement_class\tmultiwfn_build\n")
        for name, info in picked:
            f.write(f"{args.vertical}/{name}\t{info['ncp']}\t{info['mult']}\t"
                    f"{ELEMENT_CLASSES[info['eclass']]}\t{info['build']}\n")
    usable = sum(len(v) for v in bins.values()) + len(picked)
    print(f"{args.vertical}: scanned {scanned}, usable {usable}, picked {len(picked)} across {len(bins)} bins -> {args.out}")


if __name__ == "__main__":
    main()
