"""Select and stage Santiago's benchmark jobs (santi_benchmarks/raw) for the level-0
descriptor run. Reads the source tree and the status CSV only; writes only into the
results root and the output directory.

Selection (multi_spin/<lot>/<category>/<molecule>/spin_<n>): the rows of
status_by_fluxid.csv with done == True. That CSV, not multi_spin_jobs.sqlite3, is the
status source: the database mixes stale LLNL rows with new ones, takes the newest
output by mtime (the archive copy stamped every file within seconds), and marks Sella
jobs done from ORCA termination alone. Its columns used here: rel (job dir under
raw/multi_spin/multi_spin), kind ("sella" or "orca"), done, last_flux (the
last-submitted flux-<jobid>.out). Sella's done is fmax < 0.05 eV/Angstrom, looser than
ORCA NormalOpt; kind is kept in the staged TSV.

Every job needs a non-empty orca.gbw (never an orca_atom<Z>.gbw), and its output must
end with "ORCA TERMINATED NORMALLY". The output parsed is orca.out for Sella jobs and
last_flux for ORCA-opt jobs. The pipeline parses only orca.out, so last_flux is copied
to <results>/<rel>/orca.out, rel being the path under --src_root as the runner computes
it; the runner copies source files only when the results copy is missing.

--wave2 NAME adds wave_2_omol_opt_filtered jobs, which the CSV does not cover (none by
default). Their output is the last-submitted flux-*.out by job ID, and it must say
"THE OPTIMIZATION HAS CONVERGED".

    python scripts/nersc/stage_benchmarks.py --raw SANTI/raw --status SANTI/status_by_fluxid.csv \\
        --results WORK/prod_l1_santi --out_dir WORK [--dry_run]

Writes <out_dir>/santi_jobs.txt (one source folder per line, for the runner),
santi_staged.tsv (job, lot, kind, output) and santi_skipped.tsv (job, reason).
"""
import argparse
import collections
import csv
import glob
import os
import shutil

TERMINATED = b"ORCA TERMINATED NORMALLY"
OPT_CONVERGED = b"THE OPTIMIZATION HAS CONVERGED"


def tail_has(path, needle, nbytes=1 << 16):
    with open(path, "rb") as f:
        f.seek(max(0, os.path.getsize(path) - nbytes))
        return needle in f.read()


def file_has(path, needle, chunk=1 << 24):
    overlap = len(needle) - 1
    prev = b""
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                return False
            if needle in prev + block:
                return True
            prev = block[-overlap:]


def last_submitted_flux(folder):
    """flux-<jobid>.out of the last-submitted job. Flux F58 job IDs sort by submission
    time when compared by length, then character; mtimes are not reliable here."""
    outs = glob.glob(os.path.join(folder, "flux-*.out"))
    jobid = lambda p: os.path.basename(p)[len("flux-"):-len(".out")]
    return max(outs, key=lambda p: (len(jobid(p)), jobid(p))) if outs else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw", required=True, help="santi_benchmarks/raw")
    p.add_argument("--status", required=True, help="status_by_fluxid.csv")
    p.add_argument("--results", required=True, help="results root the runner writes to (TREE)")
    p.add_argument("--src_root", help="root the runner strips from job paths (its SRC); default --raw")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--wave2", nargs="*", default=[], help="wave_2_omol_opt_filtered job names to add")
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()
    raw = os.path.abspath(args.raw)
    src_root = os.path.abspath(args.src_root or args.raw)
    ms = os.path.join(raw, "multi_spin", "multi_spin")

    jobs = []  # (folder, lot, kind, output file name or None, done)
    with open(args.status, newline="") as f:
        for row in csv.DictReader(f):
            done = str(row["done"]).strip().lower() in ("true", "1")
            kind = row["kind"].strip()
            out = (os.path.basename(row["last_flux"].strip()) or None) if kind == "orca" else "orca.out"
            jobs.append((os.path.join(ms, row["rel"].strip()), row["rel"].split("/")[0], kind, out, done))
    for name in args.wave2:
        hits = glob.glob(os.path.join(raw, "wave_2_omol_opt_filtered", "wave_2_omol_opt_filtered", "*", "*", name, "orca.inp"))
        if not hits:
            print(f"wave_2 job not found: {name}")
        for h in hits:
            last = last_submitted_flux(os.path.dirname(h))
            jobs.append((os.path.dirname(h), "omol", "orca", last and os.path.basename(last), None))

    staged, skipped = [], []
    for folder, lot, kind, out_name, done in jobs:
        if done is False:
            skipped.append((folder, f"not done ({kind})"))
            continue
        gbw = os.path.join(folder, "orca.gbw")
        if not (os.path.isfile(gbw) and os.path.getsize(gbw) > 0):
            skipped.append((folder, "no orca.gbw"))
            continue
        out = os.path.join(folder, out_name) if out_name else None
        if out is None or not os.path.isfile(out):
            skipped.append((folder, f"output {out_name} missing"))
            continue
        if not tail_has(out, TERMINATED):
            skipped.append((folder, f"{kind} output did not terminate normally"))
            continue
        if done is None and not file_has(out, OPT_CONVERGED):
            skipped.append((folder, "optimization not converged"))
            continue
        if kind == "orca":
            dst = os.path.join(os.path.abspath(args.results), os.path.relpath(folder, src_root), "orca.out")
            if not args.dry_run and not (os.path.isfile(dst) and os.path.getsize(dst) == os.path.getsize(out)):
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy2(out, dst)
        staged.append((folder, lot, kind, os.path.basename(out)))

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "santi_jobs.txt"), "w") as f:
        f.writelines(s[0] + "\n" for s in staged)
    with open(os.path.join(args.out_dir, "santi_staged.tsv"), "w") as f:
        f.write("job\tlot\tkind\toutput\n")
        f.writelines("\t".join(s) + "\n" for s in staged)
    with open(os.path.join(args.out_dir, "santi_skipped.tsv"), "w") as f:
        f.write("job\treason\n")
        f.writelines(f"{j}\t{r}\n" for j, r in skipped)

    print(f"found {len(jobs)}, staged {len(staged)}, skipped {len(skipped)}{' (dry run: nothing copied)' if args.dry_run else ''}")
    print("staged by lot/kind:", dict(sorted(collections.Counter((s[1], s[2]) for s in staged).items())))
    print("skipped by reason:", dict(collections.Counter(r for _, r in skipped)))
    print(f"lists -> {args.out_dir}/santi_jobs.txt, santi_staged.tsv, santi_skipped.tsv")


if __name__ == "__main__":
    main()
