"""Sample job folders across verticals for the NERSC actinide test.

A job folder is a directory holding orca.inp, orca.tar.zst or a *.gbw. Each root
(one vertical) is walked to --max_depth, and --per_vertical folders are drawn from
it at random (fixed --seed), with --actinide_first putting folders whose orca.inp
contains an actinide first. One absolute path per line goes to --out.

    python scripts/nersc/sample_jobs.py --roots /pscratch/.../vertA /pscratch/.../vertB \
        --per_vertical 50 --out sample.txt [--actinide_first]
"""
import argparse
import os
import random

ACTINIDES = {"Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr"}


def job_folders(root, max_depth):
    base = root.rstrip(os.sep).count(os.sep)
    for dirpath, dirnames, filenames in os.walk(root):
        if dirpath.count(os.sep) - base >= max_depth:
            dirnames[:] = []
        if "orca.inp" in filenames or "orca.tar.zst" in filenames or any(f.endswith(".gbw") for f in filenames):
            dirnames[:] = []
            yield dirpath


def has_actinide(folder):
    try:
        with open(os.path.join(folder, "orca.inp")) as f:
            lines = f.read().splitlines()
    except OSError:
        return False
    inside = False
    for line in lines:
        if "* xyz" in line or "*xyz" in line:
            inside = True
            continue
        if inside and line.strip().startswith("*"):
            break
        if inside and line.split() and "".join(c for c in line.split()[0] if c.isalpha()).capitalize() in ACTINIDES:
            return True
    return False


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--roots", nargs="+", required=True, help="one directory per vertical")
    p.add_argument("--per_vertical", type=int, default=50)
    p.add_argument("--max_depth", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--actinide_first", action="store_true")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    rng = random.Random(args.seed)
    picked = []
    for root in args.roots:
        folders = sorted(job_folders(os.path.abspath(root), args.max_depth))
        rng.shuffle(folders)
        if args.actinide_first:
            folders.sort(key=lambda f: not has_actinide(f))
        take = folders[: args.per_vertical]
        print(f"{root}: {len(folders)} job folders, took {len(take)}"
              + (f" ({sum(has_actinide(f) for f in take)} with an actinide)" if args.actinide_first else ""))
        picked += take
    with open(args.out, "w") as f:
        f.write("\n".join(picked) + "\n")
    print(f"{len(picked)} folders -> {args.out}")


if __name__ == "__main__":
    main()
