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
  5. with --engine, run the engine's whole QTAIM step (cp_search.analyze) twice,
     in its Multiwfn-compatible modes and with its defaults, and compare each
     with the stored CP set and bond pairs, and its CPprop.txt (through
     parse_qtaim) with the stored CPprop.txt through the same parser and with
     the stored qtaim.json;
  6. with --adjudicate, start a Multiwfn search (topology option 1-4) and, with
     --critic2_cmd, a critic2 search at every engine CP the stored set lacks and
     record whether each converges to a CP of the same type there; and check
     whether position and print rounding explain the Laplacian and esp_total
     differences at the stored positions (qtaim_compare_props.rounding_check);
  7. append one JSON record to --out and delete the temporary directory.

Nothing in the job folders is modified. Jobs already recorded in --out are
skipped, so a requeued Slurm task resumes its slice. Each Multiwfn or orca_2mkl
call is limited to --timeout seconds; on SIGTERM (preemption, walltime) the
running job is abandoned unrecorded and its temporary directory removed.

    python scripts/qtaim_verify_stored.py --job_file jobs.txt \
        --root_omol_inputs SRC/ --root_omol_results RES/ --out verify.jsonl \
        --orca_2mkl_cmd /path/orca_2mkl --multiwfn_cmd /path/Multiwfn --n_threads 4 [--control]
"""

import argparse
import contextlib
import glob
import io
import json
import os
import re
import resource
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import time
import zipfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qtaim_compare_props import (  # noqa: E402
    LABEL_TYPE, SAME_CP, compare_cp_sets, compare_cps, compare_qtaim_json, rounding_check,
)

from qtaim_gen.source.core.charge_engine import prepare_basis, read_wfx  # noqa: E402
from qtaim_gen.source.core.cp_search import analyze, write_cpprop  # noqa: E402
from qtaim_gen.source.core.omol import create_jobs, write_settings_file  # noqa: E402
from qtaim_gen.source.core.parse_multiwfn import parse_qtaim  # noqa: E402
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


ADJ_TOL = 1e-4  # Bohr: a search started at an engine CP confirms it when it converges this close
ECP_CORE = 1.0  # Bohr: critic2 ignores the EDF core density, so CPs this close to an ECP nucleus are not judged by it
SIGNATURE_TYPE = {-3: 0, -1: 1, 1: 2, 3: 3}


def engine_extras(cps, res):
    """Indices (into res) of engine CPs with no stored CP of the same type within SAME_CP."""
    out = []
    for lab, t in LABEL_TYPE.items():
        idx = np.flatnonzero(res["type"] == t)
        rp = np.array([c["pos_bohr"] for c in cps if c["label"] == lab]).reshape(-1, 3)
        for i in idx:
            if not len(rp) or np.linalg.norm(rp - res["pos"][i], axis=1).min() >= SAME_CP:
                out.append(int(i))
    return out


def multiwfn_from_points(workdir, points, multiwfn_cmd, timeout):
    """Multiwfn's CP search from the given points (topology option 1, then 4: a file
    of X Y Z in Bohr) on workdir's orca.wfx; returns [(type, pos)] of the CPs found."""
    sub = os.path.join(workdir, "adj_multiwfn")
    os.makedirs(sub, exist_ok=True)
    for f in ("orca.wfx", "settings.ini"):
        if not os.path.exists(os.path.join(sub, f)):
            os.symlink(os.path.join(workdir, f), os.path.join(sub, f))
    np.savetxt(os.path.join(sub, "points.txt"), points, fmt="%.12f")
    with open(os.path.join(sub, "adj.stdin"), "w") as f:
        f.write("2\n1\n4\npoints.txt\n0\n7\n0\n-10\nq\n")
    with open(os.path.join(sub, "adj.mfwn"), "w") as f:
        f.write(f"{multiwfn_cmd} orca.wfx < adj.stdin > adj.out\n")
    _run("adj.mfwn", sub, timeout)
    path = os.path.join(sub, "CPprop.txt")
    if not os.path.isfile(path):
        return []
    return [(LABEL_TYPE[c["label"]], np.array(c["pos_bohr"])) for c in load_cpprop_full(path)]


def critic2_from_points(workdir, points, critic2_cmd, timeout):
    """critic2's CP search seeded at the given points (Bohr, the wfx frame) on
    workdir's orca.wfx; returns [(type, pos)] in the wfx frame."""
    sub = os.path.join(workdir, "adj_critic2")
    os.makedirs(sub, exist_ok=True)
    if not os.path.exists(os.path.join(sub, "orca.wfx")):
        os.symlink(os.path.join(workdir, "orca.wfx"), os.path.join(sub, "orca.wfx"))
    seeds = " ".join("seed point x0 %.12f %.12f %.12f" % tuple(p) for p in points)
    with open(os.path.join(sub, "adj.cri"), "w") as f:
        f.write(f"molecule orca.wfx\nload orca.wfx\nunits bohr\nauto {seeds}\ncpreport adj.json\n")
    subprocess.run([critic2_cmd, "adj.cri", "adj.cro"], cwd=sub, check=True, timeout=timeout,
                   stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    with open(os.path.join(sub, "adj.json")) as f:
        data = json.load(f)
    cv = np.array(data["structure"]["molecule_centering_vector"])
    return [(SIGNATURE_TYPE[c["signature"]], np.array(c["cartesian_coordinates"]) + cv)
            for c in data["critical_points"]["nonequivalent_cps"] if c["signature"] in SIGNATURE_TYPE]


def _confirms(found, t, pos):
    return any(ft == t and np.linalg.norm(fp - pos) < ADJ_TOL for ft, fp in found)


def adjudicate(cps, res, wfx, workdir, args):
    """Engine-only CPs (no stored Multiwfn CP of the same type within SAME_CP): does a
    search started at each, by Multiwfn and (with --critic2_cmd) critic2, converge to
    a CP of the same type within ADJ_TOL? critic2 is not asked about CPs within
    ECP_CORE of an ECP nucleus (no EDF there). One record per CP; an *_error key when
    a program failed, and the other one's verdict is kept."""
    extras = engine_extras(cps, res)
    if not extras:
        return {"extras": []}
    pts = res["pos"][extras]
    ecp = np.flatnonzero(wfx["zeff"] != wfx["atnums"])
    labels = {t: lab for lab, t in LABEL_TYPE.items()}
    recs = []
    for i in extras:
        d = np.linalg.norm(wfx["coords"] - res["pos"][i], axis=1)
        recs.append({"cp": i + 1, "type": labels[int(res["type"][i])], "rho": float(res["props"]["rho"][i]),
                     "nearest_nucleus_bohr": float(d.min()),
                     "near_ecp_core": bool(len(ecp) and d[ecp].min() < ECP_CORE),
                     "nucleus": int(res["ncp_atom"][i]) + 1 if res["type"][i] == 0 and res["ncp_atom"][i] >= 0 else None,
                     "nna": bool(res["type"][i] == 0 and res["ncp_atom"][i] < 0)})
    out = {"extras": recs}
    try:
        found = multiwfn_from_points(workdir, pts, args.multiwfn_cmd, args.timeout)
        for r, i in zip(recs, extras):
            r["multiwfn"] = _confirms(found, int(res["type"][i]), res["pos"][i])
    except Exception as e:
        out["multiwfn_error"] = f"{type(e).__name__}: {e}"
    if args.critic2_cmd:
        try:
            found = critic2_from_points(workdir, pts, args.critic2_cmd, args.timeout)
            for r, i in zip(recs, extras):
                r["critic2"] = None if r["near_ecp_core"] else _confirms(found, int(res["type"][i]), res["pos"][i])
        except Exception as e:
            out["critic2_error"] = f"{type(e).__name__}: {e}"
    return out


ENGINE_MODES = {
    "multiwfn": {"seeds": "multiwfn", "paths": "multiwfn", "boys_mode": "multiwfn"},
    "engine": {"seeds": "superset", "paths": "engine", "boys_mode": "exact"},
}


def _quiet_parse(cpprop, inp):
    with contextlib.redirect_stdout(io.StringIO()):
        return parse_qtaim(cpprop, inp, orca_tf=inp.endswith(".inp"))


def _orca_inp_from_tar(input_folder, workdir):
    """orca.inp from the input folder's orca.tar.zst, streamed through unzstd
    into workdir (LRC's tar has no --zstd); None if there is none."""
    tar_path = os.path.join(input_folder, "orca.tar.zst")
    if not os.path.isfile(tar_path):
        return None
    out = os.path.join(workdir, "orca.inp")
    proc = subprocess.Popen(["unzstd", "-q", "-c", tar_path], stdout=subprocess.PIPE)
    try:
        with tarfile.open(fileobj=proc.stdout, mode="r|") as tar:
            for member in tar:
                if os.path.basename(member.name) == "orca.inp" and member.isfile():
                    with open(out, "wb") as f:
                        shutil.copyfileobj(tar.extractfile(member), f)
                    break
    finally:
        proc.stdout.close()
        proc.kill()
        proc.wait()
    return out if os.path.isfile(out) else None


def engine_check(cps, stored_cpprop, wfx, basis, workdir, input_folder, results_folder, args=None):
    """The engine's whole QTAIM step in each ENGINE_MODES entry against the
    stored Multiwfn results (see the module docstring, step 5). A failure in
    one part is recorded under an *_error key and the rest still runs, so the
    job keeps its record and reaches the report's failing list."""
    out = {}
    inp = None
    try:
        inps = sorted(glob.glob(os.path.join(input_folder, "*.inp"))) or sorted(glob.glob(os.path.join(results_folder, "*.inp")))
        inp = inps[0] if inps else _orca_inp_from_tar(input_folder, workdir)
    except Exception as e:
        out["inp_error"] = f"{type(e).__name__}: {e}"
    stored_json = None
    try:
        with open(os.path.join(results_folder, "generator", "qtaim.json")) as f:
            stored_json = json.load(f)
    except (OSError, ValueError):
        pass
    ref_parsed = None
    if inp:
        try:
            ref_parsed = _quiet_parse(stored_cpprop, inp)
        except Exception as e:
            out["ref_parse_error"] = f"{type(e).__name__}: {e}"
    out.update(inp=inp is not None, stored_json=stored_json is not None)
    for name, cfg in ENGINE_MODES.items():
        r = {}
        try:
            t0 = time.perf_counter()
            res = analyze(wfx, basis, **cfg)
            r.update(s=round(time.perf_counter() - t0, 2), **compare_cp_sets(cps, res),
                     poincare_hopf=res["poincare_hopf"], poincare_hopf_full_set=res["poincare_hopf_full_set"],
                     weak=len(res["weak"]), dropped=res["dropped"], near_pairs=len(res["near_pairs"]),
                     reseed_rounds=[st["stage"] for st in res["stats"] if st["stage"].startswith("sphere")])
            if name == "engine" and args is not None and args.adjudicate:
                r["adjudicate"] = adjudicate(cps, res, wfx, workdir, args)
            path = os.path.join(workdir, f"CPprop_{name}.txt")
            write_cpprop(path, res, wfx)
            with open(path, "rb") as f:
                r["integrity"] = cpprop_integrity(f.read())
            if inp:
                eng = _quiet_parse(path, inp)
                if ref_parsed is not None:
                    r["vs_stored_cpprop"] = compare_qtaim_json(ref_parsed, eng)
                if stored_json is not None:
                    r["vs_stored_json"] = compare_qtaim_json(stored_json, eng)
        except Exception as e:
            # keep what was computed before the failure
            r["error"] = f"{type(e).__name__}: {e}"
        out[name] = r
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
        basis = prepare_basis(wfx)
        res = compare_cps(cps, wfx, basis)
        rec["engine_s"] = round(time.perf_counter() - t0, 2)
        rec["diffs"] = {lab: {"n": n, **r} for lab, (n, r) in res.items()}
        if args.engine:
            rec["engine"] = engine_check(cps, cpprop, wfx, basis, workdir, job,
                                         os.path.join(args.root_omol_results, rel), args)
        if args.adjudicate:
            rec["rounding"] = rounding_check(cps, wfx, basis)
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
    p.add_argument("--engine", action="store_true", help="also run the engine's whole QTAIM step (step 5)")
    p.add_argument("--adjudicate", action="store_true",
                   help="implies --engine: check each engine-only CP with a Multiwfn (and critic2) search started "
                        "there, and whether position and print rounding explain the Laplacian and esp_total "
                        "differences at the stored positions (step 6)")
    p.add_argument("--critic2_cmd", default=None, help="critic2 executable for --adjudicate (optional)")
    p.add_argument("--tmp_dir", default=None, help="where temporary job copies go (default: system tmp)")
    p.add_argument("--timeout", type=float, default=3600, help="seconds per Multiwfn/orca_2mkl call")
    args = p.parse_args()
    args.engine = args.engine or args.adjudicate
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
