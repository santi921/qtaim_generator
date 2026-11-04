"""DDEC6 validation campaign: the port (core/ddec6.py) against Chargemol on raw ORCA jobs.

For each input job folder (orca.gbw[.zstd0], orca.inp / orca.out loose or inside
orca.tar.zst) in --job_file:
  1. rebuild orca.wfx in WORK/<relpath>/ as the runner does: orca_2mkl -molden, then
     omol._native_convert in "fixed" mode (no Multiwfn);
  2. run the port (ddec6.analyze) and Chargemol (run_chargemol.run_one) on it, one
     after the other on the same cores, each timed;
  3. append one JSON line to --out: atoms, primitives, wall times, iteration counts and
     the largest port - Chargemol difference per output.
Jobs above ddec6.MAX_ATOMS atoms are recorded and skipped. Jobs already in --out are
skipped, so a requeued Slurm task resumes its slice. --workers jobs run at once, each
with --threads threads (numba for the port, OpenMP for Chargemol).

    python campaign.py --job_file slice.txt --root_omol_inputs SRC/ --work WORK/ddec6 \\
        --out results.jsonl --orca_2mkl_cmd /path/orca_2mkl --chargemol_bin /path/chargemol_omp_precise \\
        --atomic_densities /path/atomic_densities/ --workers 8 --threads 16
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import tarfile
import time
import traceback
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def _from_tar(input_folder, workdir, names):
    """Members of input_folder/orca.tar.zst whose basename is in names, streamed through
    unzstd into workdir (LRC's tar has no --zstd)."""
    tar_path = os.path.join(input_folder, "orca.tar.zst")
    if not os.path.isfile(tar_path):
        return
    proc = subprocess.Popen(["unzstd", "-q", "-c", tar_path], stdout=subprocess.PIPE)
    try:
        with tarfile.open(fileobj=proc.stdout, mode="r|") as tar:
            for member in tar:
                base = os.path.basename(member.name)
                if member.isfile() and base in names and not os.path.exists(os.path.join(workdir, base)):
                    with open(os.path.join(workdir, base), "wb") as f:
                        shutil.copyfileobj(tar.extractfile(member), f)
    finally:
        proc.stdout.close()
        proc.wait()


def prepare_wfx(input_folder, workdir, orca_2mkl_cmd, logger):
    """orca.wfx in workdir from the job's gbw, as the runner writes it; returns its path."""
    from qtaim_gen.source.core.omol import _native_convert

    wfx = os.path.join(workdir, "orca.wfx")
    if os.path.isfile(wfx) and os.path.getsize(wfx) > 0:
        return wfx
    os.makedirs(workdir, exist_ok=True)
    for name in ("orca.inp", "orca.out"):
        src = os.path.join(input_folder, name)
        if os.path.isfile(src):
            shutil.copy(src, workdir)
    _from_tar(input_folder, workdir, {"orca.inp", "orca.out"})
    gbw = os.path.join(workdir, "orca.gbw")
    if os.path.isfile(os.path.join(input_folder, "orca.gbw.zstd0")):
        subprocess.run(["unzstd", "-q", "-f", "-o", gbw, os.path.join(input_folder, "orca.gbw.zstd0")], check=True)
    elif os.path.isfile(os.path.join(input_folder, "orca.gbw")):
        shutil.copy(os.path.join(input_folder, "orca.gbw"), gbw)
    else:
        raise FileNotFoundError(f"no orca.gbw[.zstd0] in {input_folder}")
    subprocess.run([orca_2mkl_cmd, "orca", "-molden"], cwd=workdir, check=True, capture_output=True, timeout=3600)
    err = _native_convert(workdir, "fixed", logger)
    if err:
        raise RuntimeError(f"native convert: {err}")
    os.remove(gbw)
    return wfx


def _max_diff(a, b):
    out = 0.0
    for k, va in a.items():
        vb = b.get(k)
        if vb is None:
            return float("inf")
        if isinstance(va, list):
            out = max([out] + [abs(x - y) for x, y in zip(va, vb)])
        else:
            out = max(out, abs(va - vb))
    return out


def run_job(job, args):
    import numba

    from qtaim_gen.source.core import ddec6
    from run_chargemol import run_one

    numba.set_num_threads(max(1, min(args.threads, numba.config.NUMBA_NUM_THREADS)))
    logger = logging.getLogger(f"ddec6_campaign.{os.getpid()}")
    rel = os.path.relpath(job, args.root_omol_inputs)
    workdir = os.path.join(args.work, rel)
    row = {"job": rel}
    try:
        t0 = time.time()
        wfx = prepare_wfx(job, workdir, args.orca_2mkl_cmd, logger)
        row["prep_s"] = round(time.time() - t0, 2)
        from qtaim_gen.source.core.charge_engine import read_wfx

        w = read_wfx(wfx)
        row["natoms"] = int((w["zeff"] > 0.5).sum())
        row["nprim"] = int(len(w["prim_exp"]))
        if row["natoms"] > ddec6.MAX_ATOMS:
            row["skipped"] = f"natoms_above_{ddec6.MAX_ATOMS}"
            return row
        t0 = time.time()
        try:
            port = ddec6.analyze(wfx)
        except ddec6.ChargemolStop as e:
            port = None
            row["port_chargemol_stop"] = str(e)
        row["port_s"] = round(time.time() - t0, 2)
        if port is not None:
            with open(os.path.join(workdir, "ddec6_port.json"), "w") as f:
                json.dump(port, f)
        if args.skip_chargemol:
            return row
        _, status, sec = run_one(wfx, binary=args.chargemol_bin, densities=args.atomic_densities,
                                 out_name="ddec6_chargemol.json", threads=args.threads)
        row["chargemol_s"] = round(sec, 2)
        row["chargemol_status"] = status
        if status != "ok" or port is None:
            return row
        with open(os.path.join(workdir, "ddec6_chargemol.json")) as f:
            ref = json.load(f)
        r, p = ref["ddec6"], port["ddec6"]
        row["iter_port"] = port["_meta"]["n_iterations"]
        row["iter_chargemol"] = ref["_meta"].get("n_iterations")
        row["max_dq"] = _max_diff(p["charge"], r["charge"])
        row["max_ddipole"] = _max_diff(p["atomic_dipole"], r["atomic_dipole"])
        row["max_dquadrupole"] = _max_diff(p["atomic_quadrupole"], r["atomic_quadrupole"])
        if "spin" in p and "spin" in r:
            row["max_dspin"] = _max_diff(p["spin"], r["spin"])
            row["spin_sum"] = sum(p["spin"].values())
        row["charge_sum"] = sum(p["charge"].values())
    except Exception as e:  # record the failure and keep the batch going
        row["error"] = f"{type(e).__name__}: {e}"
        row["traceback"] = traceback.format_exc()[-2000:]
    return row


def _star(item):
    return run_job(*item)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--job_file", required=True, help="input job folders, one per line (absolute or under --root_omol_inputs)")
    p.add_argument("--root_omol_inputs", required=True)
    p.add_argument("--work", required=True, help="where the per-job orca.wfx and results go")
    p.add_argument("--out", required=True, help="results JSONL (appended)")
    p.add_argument("--orca_2mkl_cmd", required=True)
    p.add_argument("--chargemol_bin")
    p.add_argument("--atomic_densities")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--threads", type=int, default=16)
    p.add_argument("--skip_chargemol", action="store_true", help="port timings only")
    args = p.parse_args(argv)
    if not args.skip_chargemol and not (args.chargemol_bin and args.atomic_densities):
        p.error("--chargemol_bin and --atomic_densities are required unless --skip_chargemol")
    args.atomic_densities = os.path.join(args.atomic_densities or "", "")
    done = set()
    if os.path.exists(args.out):
        with open(args.out) as f:
            done = {json.loads(line)["job"] for line in f if line.strip()}
    with open(args.job_file) as f:
        jobs = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
    jobs = [j if os.path.isabs(j) else os.path.join(args.root_omol_inputs, j) for j in jobs]
    jobs = [j for j in jobs if os.path.relpath(j, args.root_omol_inputs) not in done]
    print(f"{len(jobs)} jobs to run ({len(done)} already in {args.out})", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for row in pool.map(_star, [(j, args) for j in jobs]):
            with open(args.out, "a") as f:
                f.write(json.dumps(row) + "\n")
            print(json.dumps({k: v for k, v in row.items() if k != "traceback"}), flush=True)


if __name__ == "__main__":
    sys.exit(main())
