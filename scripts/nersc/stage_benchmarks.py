"""Select and stage Santiago's benchmark jobs (santi_benchmarks/raw) for the level-0
descriptor run. Reads the source tree and the multi_spin job database only; writes
only into the results root and the output directory.

Selection:
- multi_spin/<lot>/<category>/<molecule>/spin_<n>: the job's latest row in
  multi_spin_jobs.sqlite3 must have status 1 (check_multi_spin.py: completed). This
  also drops the two impossible-multiplicity jobs, which have status -1.
- wave_2_omol_opt_filtered: only the jobs named by --wave2 (79 of its 80 repeat a
  multi_spin/omol job). They are not in the database, so the newest output must say
  "THE OPTIMIZATION HAS CONVERGED" (ORCA opt) and "ORCA TERMINATED NORMALLY".
- Every job needs a non-empty orca.gbw (never an orca_atom<Z>.gbw).

Output file: Sella jobs (EnGrad in orca.inp) write orca.out; ORCA-opt jobs write
flux-*.out, one per restart, and the newest by mtime is the run that finished. The
pipeline parses only orca.out, so for ORCA-opt jobs the newest flux-*.out is copied to
<results>/<rel>/orca.out, where rel is the path under --src_root as the runner
computes it. The runner copies source files only when the results copy is missing,
so the staged file is the one parsed. Every chosen output must have terminated
normally.

    python scripts/nersc/stage_benchmarks.py --raw SANTI/raw --db SANTI/raw/multi_spin/multi_spin/multi_spin_jobs.sqlite3 \\
        --results WORK/prod_l1_santi --out_dir WORK [--dry_run]

Writes <out_dir>/santi_jobs.txt (one source folder per line, for the runner),
santi_staged.tsv (job, lot, run type, output used) and santi_skipped.tsv (job, reason).
"""
import argparse
import collections
import glob
import os
import re
import shutil
import sqlite3

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


def is_orca_opt(inp):
    """An ORCA geometry optimization: a "!" keyword ending in "opt" (Opt, TightOpt, ...)."""
    with open(inp, errors="replace") as f:
        return any(tok.lower().endswith("opt") for line in f if line.lstrip().startswith("!")
                   for tok in line.lstrip()[1:].split())


def newest_flux(folder):
    outs = glob.glob(os.path.join(folder, "flux-*.out"))
    return max(outs, key=lambda p: (os.path.getmtime(p), p)) if outs else None


def latest_db_rows(db):
    """(lot, category, name, spin) -> (status, note, run_type) of its most recent check."""
    rows = {}
    # read-only: a plain connect on a wrong path silently creates an empty database
    with sqlite3.connect(f"file:{os.path.abspath(db)}?mode=ro", uri=True) as c:
        tables = {r[0] for r in c.execute("select name from sqlite_master where type='table'")}
        if "jobs" not in tables:
            raise SystemExit(f"{db}: no 'jobs' table (tables: {sorted(tables)}); is this check_multi_spin.py's database?")
        for lot, cat, name, spin, status, note, run_type, checked in c.execute(
                "select lot, category, name, spin, status, note, run_type, checked_at from jobs"):
            key = (lot, cat, name, spin)
            if key not in rows or (checked or 0) > rows[key][3]:
                rows[key] = (status, note, run_type, checked or 0)
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw", required=True, help="santi_benchmarks/raw")
    p.add_argument("--db", required=True, help="multi_spin_jobs.sqlite3")
    p.add_argument("--results", required=True, help="results root the runner writes to (TREE)")
    p.add_argument("--src_root", help="root the runner strips from job paths (its SRC); default --raw")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--wave2", nargs="*", default=["CfK3C22H36OCl4-ZORA_PBE0"])
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()
    raw = os.path.abspath(args.raw)
    src_root = os.path.abspath(args.src_root or args.raw)
    db = latest_db_rows(args.db)

    jobs = []  # (folder, source, key)
    for inp in sorted(glob.glob(os.path.join(raw, "multi_spin", "multi_spin", "*", "*", "*", "spin_*", "orca.inp"))):
        folder = os.path.dirname(inp)
        lot, cat, name, spin = folder.split(os.sep)[-4:]
        jobs.append((folder, "multi_spin", (lot, cat, name, spin)))
    for name in args.wave2:
        hits = glob.glob(os.path.join(raw, "wave_2_omol_opt_filtered", "wave_2_omol_opt_filtered", "*", "*", name, "orca.inp"))
        if not hits:
            print(f"wave_2 job not found: {name}")
        jobs += [(os.path.dirname(h), "wave_2", None) for h in hits]

    staged, skipped = [], []
    for folder, source, key in jobs:
        gbw = os.path.join(folder, "orca.gbw")
        if not (os.path.isfile(gbw) and os.path.getsize(gbw) > 0):
            skipped.append((folder, "no orca.gbw"))
            continue
        if source == "multi_spin":
            row = db.get(key)
            if row is None:
                skipped.append((folder, "not in job database"))
                continue
            if row[0] != 1:
                skipped.append((folder, f"database status {row[0]} ({row[1]})"))
                continue
        opt = is_orca_opt(os.path.join(folder, "orca.inp"))
        out = newest_flux(folder) if opt else os.path.join(folder, "orca.out")
        if out is None or not os.path.isfile(out):
            skipped.append((folder, "no output file"))
            continue
        if not tail_has(out, TERMINATED):
            skipped.append((folder, f"{os.path.basename(out)} did not terminate normally"))
            continue
        if source == "wave_2" and opt and not file_has(out, OPT_CONVERGED):
            skipped.append((folder, f"{os.path.basename(out)}: optimization not converged"))
            continue
        if opt:
            dst = os.path.join(os.path.abspath(args.results), os.path.relpath(folder, src_root), "orca.out")
            if not args.dry_run and not (os.path.isfile(dst) and os.path.getsize(dst) == os.path.getsize(out)):
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy2(out, dst)
        lot = key[0] if key else "omol"
        staged.append((folder, lot, "orca_opt" if opt else "sella", os.path.basename(out)))

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "santi_jobs.txt"), "w") as f:
        f.writelines(s[0] + "\n" for s in staged)
    with open(os.path.join(args.out_dir, "santi_staged.tsv"), "w") as f:
        f.write("job\tlot\trun_type\toutput\n")
        f.writelines("\t".join(s) + "\n" for s in staged)
    with open(os.path.join(args.out_dir, "santi_skipped.tsv"), "w") as f:
        f.write("job\treason\n")
        f.writelines(f"{j}\t{r}\n" for j, r in skipped)

    print(f"found {len(jobs)}, staged {len(staged)}, skipped {len(skipped)}{' (dry run: nothing copied)' if args.dry_run else ''}")
    print("staged by lot/run type:", dict(collections.Counter((s[1], s[2]) for s in staged)))
    print("skipped by reason:", dict(collections.Counter(re.sub(r"flux-\S+", "flux-*.out", r) for _, r in skipped)))
    print(f"lists -> {args.out_dir}/santi_jobs.txt, santi_staged.tsv, santi_skipped.tsv")


if __name__ == "__main__":
    main()
