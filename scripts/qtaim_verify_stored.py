"""Tier 2 of docs/engine_roadmap/qtaim_verification_campaign.md: compare the QTAIM
engine against the CPprop.txt stored by a production run, without rerunning
Multiwfn.

Per job:
  1. read CPprop.txt and the qtaim.out banner (Multiwfn build) from
     <results>/generator/out_files.zip, plus both files' zip timestamps;
  2. regenerate orca.wfx in a temporary directory from the input folder's
     orca.gbw(.zstd0) with the production convert step (settings.ini,
     create_jobs, convert.in = orca_2mkl, props_convert.mfwn = Multiwfn);
  3. evaluate qtaim_engine.point_properties at the stored CP positions and
     compare (scripts/qtaim_compare_props.compare_cps);
  4. with --control, also run production QTAIM on the regenerated wfx and
     compare that fresh CPprop.txt with the stored one (conversion drift and
     CP-set reproducibility);
  5. append one JSON record to --out and delete the temporary directory.

Nothing in the job folders is modified. Jobs already recorded in --out are
skipped, so a requeued Slurm task resumes its slice. Each Multiwfn or orca_2mkl
call is limited to --timeout seconds; on SIGTERM (preemption, walltime) the
running job is abandoned unrecorded and its temporary directory removed.

    python scripts/qtaim_verify_stored.py --job_file jobs.txt \
        --root_omol_inputs SRC/ --root_omol_results RES/ --out verify.jsonl \
        --orca_2mkl_cmd /path/orca_2mkl --multiwfn_cmd /path/Multiwfn --n_threads 4 [--control]
"""

import argparse
import json
import os
import re
import resource
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import zipfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qtaim_compare_props import compare_cps  # noqa: E402

from qtaim_gen.source.core.charge_engine import prepare_basis, read_wfx  # noqa: E402
from qtaim_gen.source.core.omol import create_jobs, write_settings_file  # noqa: E402
from qtaim_gen.source.core.parse_qtaim import load_cpprop_full, poincare_hopf  # noqa: E402
from qtaim_gen.source.utils.validation import QTAIM_COUNT_PATTERN, cpprop_integrity  # noqa: E402


# as the production Slurm scripts: Multiwfn crashes in the molden -> wfx step otherwise
MULTIWFN_ENV = dict(os.environ, OMP_STACKSIZE="4G", KMP_STACKSIZE="200M")


def _run(script, workdir, timeout):
    # the scripts tee Multiwfn's output into <step>.out; only stderr is kept for the record
    proc = subprocess.Popen(["bash", script], cwd=workdir, stdout=subprocess.DEVNULL,
                            stderr=subprocess.PIPE, env=MULTIWFN_ENV, start_new_session=True)
    try:
        _, err = proc.communicate(timeout=timeout)
    except BaseException:
        # timeout or SIGTERM: kill the whole group, since Multiwfn and tee outlive bash
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()
        raise
    if proc.returncode:
        raise subprocess.CalledProcessError(proc.returncode, ["bash", script], stderr=err)


def stored_reference(results_folder, workdir):
    """Extract CPprop.txt from the job's out_files.zip into workdir; returns
    (path or None, info dict with the Multiwfn build and zip timestamps)."""
    zpath = os.path.join(results_folder, "generator", "out_files.zip")
    info = {"zip": os.path.isfile(zpath)}
    if not info["zip"]:
        return None, info
    with zipfile.ZipFile(zpath) as z:
        names = {os.path.basename(n): n for n in z.namelist()}
        if "CPprop.txt" not in names:
            return None, info
        info["cpprop_time"] = "%04d-%02d-%02d %02d:%02d:%02d" % z.getinfo(names["CPprop.txt"]).date_time
        reported = None
        if "qtaim.out" in names:
            info["qtaim_out_time"] = "%04d-%02d-%02d %02d:%02d:%02d" % z.getinfo(names["qtaim.out"]).date_time
            text = z.read(names["qtaim.out"]).decode("latin1")
            m = re.search(r"Version\s+(\S+),\s+update date:\s+(\S+)", text[:4000])
            info["multiwfn_build"] = f"{m.group(1)} {m.group(2)}" if m else None
            found = QTAIM_COUNT_PATTERN.findall(text)
            reported = int(found[-1]) if found else None
        data = z.read(names["CPprop.txt"])
        info["cpprop_problems"] = cpprop_integrity(data, reported_bcp=reported)
        path = os.path.join(workdir, "CPprop_stored.txt")
        with open(path, "wb") as f:
            f.write(data)
    return path, info


def regenerate_wfx(input_folder, workdir, args):
    """Production convert step in workdir; returns the orca.wfx path or None."""
    gbw = os.path.join(workdir, "orca.gbw")
    if os.path.isfile(os.path.join(input_folder, "orca.gbw.zstd0")):
        subprocess.run(["unzstd", "-q", "-f", "-o", gbw, os.path.join(input_folder, "orca.gbw.zstd0")],
                       check=True, timeout=args.timeout)
    elif os.path.isfile(os.path.join(input_folder, "orca.gbw")):
        shutil.copy(os.path.join(input_folder, "orca.gbw"), gbw)
    else:
        return None
    write_settings_file(workdir, n_threads=args.n_threads)
    # debug=True writes the convert and QTAIM steps only (no step that reads orca.inp)
    create_jobs(folder=workdir, multiwfn_cmd=args.multiwfn_cmd, orca_2mkl_cmd=args.orca_2mkl_cmd,
                separate=True, full_set=0, wfx=True, debug=True)
    _run("convert.in", workdir, args.timeout)
    _run("props_convert.mfwn", workdir, args.timeout)
    wfx = os.path.join(workdir, "orca.wfx")
    return wfx if os.path.isfile(wfx) else None


def summarize(cps):
    counts = {lab: sum(c["label"] == lab for c in cps) for lab in ("NCP", "BCP", "RCP", "CCP")}
    return {
        "counts": counts,
        "poincare_hopf": poincare_hopf(cps),
        "ncp_no_nucleus": sum(c["label"] == "NCP" and c["nucleus"] is None for c in cps),
        "bcp_unpaired": sum(c["label"] == "BCP" and c["connected"] is None for c in cps),
    }


def control_run(workdir, stored_cps, timeout):
    """Fresh production QTAIM on the regenerated wfx, compared with the stored CPs."""
    _run("props_qtaim.mfwn", workdir, timeout)
    fresh = load_cpprop_full(os.path.join(workdir, "CPprop.txt"))
    out = {"fresh": summarize(fresh)}
    if len(fresh) != len(stored_cps) or any(a["type"] != b["type"] for a, b in zip(fresh, stored_cps)):
        out["same_cp_set"] = False
        return out
    pos = max(float(np.max(np.abs(np.array(a["pos_bohr"]) - b["pos_bohr"]))) for a, b in zip(fresh, stored_cps))
    # largest relative change per printed property, with the CP it occurs at; the
    # 1e-10 floor keeps converged-gradient noise (~1e-16, run to run with
    # threads) from reading as a large relative change
    worst = {}
    for a, b in zip(fresh, stored_cps):
        for k, v in b["props"].items():
            w = a["props"].get(k)
            if w is not None and w != v:
                r = abs(w - v) / max(abs(v), 1e-10)
                if r > worst.get(k, (0.0,))[0]:
                    worst[k] = (r, b["index"], v, w)
    top = sorted(worst.items(), key=lambda kv: -kv[1][0])[:5]
    out.update(same_cp_set=True, max_pos_diff_bohr=pos,
               max_prop_rel_diff={k: {"rel": r, "cp": i, "stored": v, "fresh": w} for k, (r, i, v, w) in top})
    return out


def verify(job, args):
    rel = os.path.relpath(job, args.root_omol_inputs)
    rec = {"job": rel}
    workdir = tempfile.mkdtemp(prefix="qtaim_verify_", dir=args.tmp_dir)
    try:
        cpprop, info = stored_reference(os.path.join(args.root_omol_results, rel), workdir)
        rec.update(info)
        if cpprop is None:
            rec["status"] = "no_stored_cpprop"
            return rec
        if info["cpprop_problems"]:
            rec["status"] = "corrupt_reference"
            return rec
        t0 = time.perf_counter()
        wfx_path = regenerate_wfx(job, workdir, args)
        rec["convert_s"] = round(time.perf_counter() - t0, 2)
        if wfx_path is None:
            rec["status"] = "no_wfx"
            return rec
        cps = load_cpprop_full(cpprop)
        rec["stored"] = summarize(cps)
        wfx = read_wfx(wfx_path)
        t0 = time.perf_counter()
        res = compare_cps(cps, wfx, prepare_basis(wfx))
        rec["engine_s"] = round(time.perf_counter() - t0, 2)
        rec["diffs"] = {lab: {"n": n, **r} for lab, (n, r) in res.items()}
        if args.control:
            t0 = time.perf_counter()
            rec["control"] = control_run(workdir, cps, args.timeout)
            rec["control"]["multiwfn_s"] = round(time.perf_counter() - t0, 2)
        rec["status"] = "ok"
        return rec
    except subprocess.TimeoutExpired as e:
        rec["status"] = f"timeout: {e.cmd} after {e.timeout:.0f} s"
        return rec
    except subprocess.CalledProcessError as e:
        rec["status"] = f"error: {e.cmd} exited {e.returncode}"
        rec["stderr_tail"] = (e.stderr or b"")[-2000:].decode("latin1")
        return rec
    except Exception as e:
        # one bad job must not end the slice
        rec["status"] = f"error: {type(e).__name__}: {e}"
        return rec
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--job_file", required=True, help="input job folders (with orca.gbw[.zstd0]), one per line")
    p.add_argument("--root_omol_inputs", required=True)
    p.add_argument("--root_omol_results", required=True)
    p.add_argument("--out", required=True, help="JSONL, one record per job (appended)")
    p.add_argument("--orca_2mkl_cmd", required=True)
    p.add_argument("--multiwfn_cmd", required=True)
    p.add_argument("--n_threads", type=int, default=4)
    p.add_argument("--control", action="store_true", help="also run fresh production QTAIM on the regenerated wfx")
    p.add_argument("--tmp_dir", default=None, help="where temporary job copies go (default: system tmp)")
    p.add_argument("--timeout", type=float, default=3600, help="seconds per Multiwfn/orca_2mkl call")
    args = p.parse_args()
    # create_jobs writes its scripts relative to the home directory unless given an absolute path
    args.tmp_dir = os.path.abspath(args.tmp_dir) if args.tmp_dir else None
    if args.tmp_dir:
        os.makedirs(args.tmp_dir, exist_ok=True)

    import numba

    numba.set_num_threads(args.n_threads)
    # Multiwfn needs a large stack; raise the soft limit to the hard one (inherited by children)
    _, hard = resource.getrlimit(resource.RLIMIT_STACK)
    resource.setrlimit(resource.RLIMIT_STACK, (hard, hard))
    # preemption or walltime: raise SystemExit so the running subprocess is killed and
    # its temporary directory removed; that job is left unrecorded and reruns on resume
    signal.signal(signal.SIGTERM, lambda *a: sys.exit(143))
    with open(args.job_file) as f:
        jobs = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
    done = set()
    if os.path.isfile(args.out):
        with open(args.out) as f:
            done = {json.loads(ln)["job"] for ln in f if ln.strip()}
    for job in jobs:
        if os.path.relpath(job, args.root_omol_inputs) in done:
            continue
        rec = verify(job, args)
        with open(args.out, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps({k: rec.get(k) for k in ("job", "status", "multiwfn_build")}), flush=True)


if __name__ == "__main__":
    main()
