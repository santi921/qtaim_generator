"""Benchmark the charge/fuzzy steps: Multiwfn (per-step vs one session) vs HORTON.

Driver mode (generator env) times, per wfx:
  mwfn_load  Multiwfn load-and-quit (startup + wfx read floor)
  mwfn_sep   production layout: one Multiwfn process per step
  mwfn_comb  all steps chained in one Multiwfn session; output is split at the
             main-menu banner and every step is checked against mwfn_sep with
             the production parsers
  horton     this script re-invoked under the horton env (--horton_stages),
             timing load / grid / density / each partition separately

    python scripts/bench_charge_engines.py --wfx_list wfx.txt --out_dir DIR \
        --multiwfn ~/dev/Multiwfn_3_8/Multiwfn_noGUI \
        --horton_python ~/miniconda3/envs/horton/bin/python --nthreads 4

Results append to DIR/results.jsonl (one record per wfx x mode) and a summary
table prints at the end. Steps are the production full_set=0 set: the four
charge schemes, the fuzzy density (+ spin when mult != 1) integrals, and
fuzzy_bond.
"""

import argparse
import json
import os
import resource
import shutil
import subprocess
import sys
import time

MENU_MARKER = "************ Main function menu ************"
EDF_TAG = "<Additional Electron Density Function (EDF)>"
EDF_END = "</Additional Electron Density Function (EDF)>"


def wfx_header(path):
    want = {
        "<Number of Nuclei>": "natoms",
        "<Number of Primitives>": "nprim",
        "<Electronic Spin Multiplicity>": "mult",
    }
    out, key = {"edf": False}, None
    with open(path) as f:
        for line in f:
            s = line.strip()
            if key:
                out[key] = int(s)
                key = None
                continue
            if s in want:
                key = want[s]
            elif s == EDF_TAG:
                out["edf"] = True
            elif s.startswith("<Molecular Orbital Primitive Coefficients>"):
                break
    return out


# ---------------------------------------------------------------- horton mode


def horton_stages(args):
    """Runs inside the horton env. Must not import qtaim_gen."""
    import tempfile

    import numpy as np

    sys.path.insert(
        0,
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..", "qtaim_gen", "source", "scripts", "helpers",
        ),
    )
    import logging
    import warnings

    logging.disable(logging.CRITICAL)
    warnings.filterwarnings("ignore")

    t = {}
    t0 = time.perf_counter()
    import horton_worker as hw
    from gbasis.evals.density import evaluate_density
    from gbasis.wrappers import from_iodata
    from iodata import load_one

    t["import"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    with open(args.wfx[0]) as f:
        text = f.read()
    i = text.find(EDF_TAG)
    if i >= 0:
        j = text.index(EDF_END) + len(EDF_END)
        while j < len(text) and text[j] in "\r\n":
            j += 1
        text = text[:i] + text[j:]
    with tempfile.NamedTemporaryFile("w", suffix=".wfx", delete=False) as tmp:
        tmp.write(text)
    try:
        mol = load_one(tmp.name)
    finally:
        os.unlink(tmp.name)
    t["load"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    basis = from_iodata(mol)
    t["basis"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    molgrid, grid_used = hw.build_molgrid(mol.atnums, mol.atcoords, args.grid)
    t["grid"] = time.perf_counter() - t0

    nbasis = mol.mo.coeffs.shape[0]
    chunk = max(2000, int(args.chunk_gb * 1e9 / (8 * nbasis)))

    def density(dm):
        out = np.empty(molgrid.size)
        for s in range(0, molgrid.size, chunk):
            out[s : s + chunk] = evaluate_density(dm, basis, molgrid.points[s : s + chunk])
        return out

    open_shell = mol.mo.kind == "unrestricted"
    t0 = time.perf_counter()
    if open_shell:
        na = mol.mo.norba
        ca, cb = mol.mo.coeffs[:, :na], mol.mo.coeffs[:, na:]
        rho_a = density((ca * mol.mo.occsa) @ ca.T)
        t["rho"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        rho_b = density((cb * mol.mo.occsb) @ cb.T)
        t["spin_extra"] = time.perf_counter() - t0
        rho, spin = rho_a + rho_b, rho_a - rho_b
    else:
        rho = density((mol.mo.coeffs * mol.mo.occs) @ mol.mo.coeffs.T)
        t["rho"] = time.perf_counter() - t0
        spin = None

    nelec_grid = float(molgrid.integrate(rho))
    has_ecp = bool((mol.atcorenums != mol.atnums).any())

    skipped = []
    for scheme in ("becke_csd", "hirshfeld"):
        if scheme == "hirshfeld" and has_ecp:
            skipped.append(scheme)
            continue
        t0 = time.perf_counter()
        if scheme == "hirshfeld":
            # proatom build/cache cost is separated from the partition itself
            hw.build_proatomdb(mol.atnums)
            t["hirshfeld_proatoms"] = time.perf_counter() - t0
            t0 = time.perf_counter()
        part = hw.build_part(scheme, mol, molgrid, rho)
        if spin is not None:
            part._spindens = spin
        part.do_charges()
        t[f"{scheme}_charges"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        part.do_moments()
        t[f"{scheme}_moments"] = time.perf_counter() - t0
        if spin is not None:
            t0 = time.perf_counter()
            try:
                part.do_spin_charges()
                t[f"{scheme}_spin_charges"] = time.perf_counter() - t0
            except Exception as e:
                skipped.append(f"{scheme}_spin: {type(e).__name__}: {e}"[:160])

    res = {
        "stages": {k: round(v, 3) for k, v in t.items()},
        "grid": grid_used,
        "npoints": int(molgrid.size),
        "nbasis": int(nbasis),
        "chunk_points": chunk,
        "nelec_grid": round(nelec_grid, 4),
        "nelec_expected": float(mol.nelec),
        "open_shell": open_shell,
        "has_ecp": has_ecp,
        "skipped": skipped,
        "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024,
    }
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)
    return 0


# ---------------------------------------------------------------- driver mode


def production_steps(mult):
    from qtaim_gen.source.data.multiwfn import (
        bond_order_dict,
        charge_data_dict,
        fuzzy_data,
    )

    steps = dict(charge_data_dict(full_set=0))
    steps.update(fuzzy_data(spin=mult != 1, full_set=0))
    steps["fuzzy_bond"] = bond_order_dict(full_set=0)["fuzzy_bond"]
    return steps


def _unlimited_stack():
    resource.setrlimit(
        resource.RLIMIT_STACK, (resource.RLIM_INFINITY, resource.RLIM_INFINITY)
    )


def run_multiwfn(multiwfn, workdir, wfx, script_text, out_path, timeout, env):
    """Run one Multiwfn session; returns (wall_s, peak_rss_mb, returncode)."""
    with open(out_path, "w") as fout:
        before = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
        t0 = time.perf_counter()
        try:
            # bare name, cwd=workdir: Multiwfn rejects input paths over 200 chars
            p = subprocess.run(
                [multiwfn, os.path.basename(wfx)],
                input=script_text,
                text=True,
                stdout=fout,
                stderr=subprocess.STDOUT,
                cwd=workdir,
                env=env,
                timeout=timeout,
                preexec_fn=_unlimited_stack,
            )
            rc = p.returncode
        except subprocess.TimeoutExpired:
            rc = "timeout"
        wall = time.perf_counter() - t0
    # RUSAGE_CHILDREN maxrss is a running max over all children, so it only
    # reports a new peak; good enough for the large-system rows
    after = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    return wall, (after // 1024 if after > before else None), rc


def flat_numbers(d, prefix=""):
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(flat_numbers(v, f"{prefix}/{k}"))
    elif isinstance(d, (list, tuple)):
        for i, v in enumerate(d):
            out.update(flat_numbers(v, f"{prefix}/{i}"))
    elif isinstance(d, (int, float)) and not isinstance(d, bool):
        out[prefix] = float(d)
    return out


def compare_parsed(a, b):
    fa, fb = flat_numbers(a), flat_numbers(b)
    if set(fa) != set(fb):
        return {"match": False, "reason": f"key sets differ ({len(fa)} vs {len(fb)})"}
    diffs = [abs(fa[k] - fb[k]) for k in fa if fa[k] == fa[k] and fb[k] == fb[k]]
    mx = max(diffs) if diffs else 0.0
    return {"match": mx < 1e-6, "max_abs_diff": mx, "n": len(fa)}


def bench_one(wfx_src, args, results_path):
    from qtaim_gen.source.core.omol import _parse_routine_out, write_settings_file

    hdr = wfx_header(wfx_src)
    name = os.path.basename(os.path.dirname(os.path.abspath(wfx_src)))
    workdir = os.path.join(args.out_dir, name)
    os.makedirs(workdir, exist_ok=True)
    wfx = os.path.join(workdir, "orca.wfx")
    if not os.path.exists(wfx):
        os.symlink(os.path.abspath(wfx_src), wfx)
    write_settings_file(workdir, n_threads=args.nthreads)

    env = dict(os.environ, OMP_STACKSIZE="1G")
    steps = production_steps(hdr["mult"])
    fuzzy_routines = {k for k in steps if "fuzzy" in k and k != "fuzzy_bond"}
    base = {"wfx": wfx_src, "name": name, "nthreads": args.nthreads, **hdr}

    def emit(rec):
        rec = {**base, **rec}
        with open(results_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)

    modes = args.modes.split(",")
    sep_parsed = {}
    # parse_fuzzy_real_space keys its result by the file stem, so sep and comb
    # outputs share names and live in separate subdirs
    for sub in ("sep", "comb"):
        os.makedirs(os.path.join(workdir, sub), exist_ok=True)

    if "mwfn_load" in modes:
        wall, rss, rc = run_multiwfn(
            args.multiwfn, workdir, wfx, "q\n",
            os.path.join(workdir, "load.out"), args.timeout, env,
        )
        emit({"mode": "mwfn_load", "wall_s": round(wall, 2), "rc": rc})

    if "mwfn_sep" in modes:
        per_step = {}
        for step, text in steps.items():
            out = os.path.join(workdir, "sep", f"{step}.out")
            wall, rss, rc = run_multiwfn(
                args.multiwfn, workdir, wfx, text, out, args.timeout, env
            )
            per_step[step] = {"wall_s": round(wall, 2), "rc": rc, "peak_rss_mb": rss}
            try:
                sep_parsed[step] = _parse_routine_out(step, out, fuzzy_routines)
            except Exception as e:
                per_step[step]["parse_error"] = f"{type(e).__name__}: {e}"[:160]
        emit({
            "mode": "mwfn_sep",
            "wall_s": round(sum(v["wall_s"] for v in per_step.values()), 2),
            "steps": per_step,
        })

    if "mwfn_comb" in modes:
        # every production string ends "...\nq\n" after returning to the main
        # menu, so dropping the trailing q chains them in one session
        # the fuzzy-menu partition choice (-1 -> 3 = Hirshfeld) persists within a
        # session, so every Becke-partition step must run before the hirsh_* ones
        names = sorted(steps, key=lambda s: s.startswith("hirsh_fuzzy"))
        script = "".join(steps[s][: -len("q\n")] for s in names) + "q\n"
        out = os.path.join(workdir, "comb.out")
        wall, rss, rc = run_multiwfn(args.multiwfn, workdir, wfx, script, out, args.timeout, env)
        rec = {"mode": "mwfn_comb", "wall_s": round(wall, 2), "rc": rc, "peak_rss_mb": rss}
        with open(out) as f:
            segments = f.read().split(MENU_MARKER)
        # segment 0 = load banner, 1..n = one step each, n+1 = final menu
        if len(segments) != len(names) + 2:
            rec["split_error"] = f"{len(segments) - 2} segments for {len(names)} steps"
        else:
            checks = {}
            for k, step in enumerate(names, start=1):
                seg_path = os.path.join(workdir, "comb", f"{step}.out")
                with open(seg_path, "w") as f:
                    f.write(segments[k])
                try:
                    parsed = _parse_routine_out(step, seg_path, fuzzy_routines)
                except Exception as e:
                    checks[step] = {"match": False, "reason": f"parse: {type(e).__name__}"}
                    continue
                if step in sep_parsed:
                    checks[step] = compare_parsed(parsed, sep_parsed[step])
            rec["vs_sep"] = checks
        emit(rec)

    if "horton" in modes:
        out = os.path.join(workdir, "horton_stages.json")
        thread_env = {
            k: str(args.nthreads)
            for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
        }
        cmd = [
            "prlimit", f"--as={int(args.horton_mem_gb * 2**30)}",
            args.horton_python, os.path.abspath(__file__), "--horton_stages",
            "--wfx", os.path.abspath(wfx_src), "--out", out, "--grid", args.grid,
            "--chunk_gb", str(args.chunk_gb),
        ]
        t0 = time.perf_counter()
        try:
            p = subprocess.run(
                cmd, env=dict(os.environ, **thread_env), capture_output=True,
                text=True, timeout=args.timeout,
            )
            rc, err = p.returncode, p.stderr[-400:]
        except subprocess.TimeoutExpired:
            rc, err = "timeout", ""
        rec = {"mode": "horton", "wall_s": round(time.perf_counter() - t0, 2), "rc": rc}
        if rc == 0:
            with open(out) as f:
                rec.update(json.load(f))
        else:
            rec["stderr_tail"] = err
        emit(rec)


def summarize(results_path):
    rows = {}
    with open(results_path) as f:
        for line in f:
            r = json.loads(line)
            rows.setdefault((r["natoms"], r["name"]), {})[r["mode"]] = r
    print("\natoms | prims | mult | ecp | load_s | sep_s | comb_s | comb_ok | horton_s | horton_rho_s")
    for (nat, name), m in sorted(rows.items()):
        any_r = next(iter(m.values()))
        comb = m.get("mwfn_comb", {})
        checks = comb.get("vs_sep") or {}
        ok = all(c.get("match") for c in checks.values()) if checks else comb.get("split_error", "-")
        h = m.get("horton", {})
        print(" | ".join(str(x) for x in (
            nat, any_r["nprim"], any_r["mult"], any_r["edf"],
            m.get("mwfn_load", {}).get("wall_s", "-"),
            m.get("mwfn_sep", {}).get("wall_s", "-"),
            comb.get("wall_s", "-"), ok,
            h.get("wall_s", "-") if h.get("rc") == 0 else h.get("rc", "-"),
            h.get("stages", {}).get("rho", "-"),
        )))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--horton_stages", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--wfx", action="append", default=[])
    p.add_argument("--wfx_list", help="file with one wfx path per line")
    p.add_argument("--out", help=argparse.SUPPRESS)
    p.add_argument("--out_dir")
    p.add_argument("--multiwfn", default=os.path.expanduser("~/dev/Multiwfn_3_8/Multiwfn_noGUI"))
    p.add_argument("--horton_python", default=os.path.expanduser("~/miniconda3/envs/horton/bin/python"))
    p.add_argument("--nthreads", type=int, default=4)
    p.add_argument("--modes", default="mwfn_load,mwfn_sep,mwfn_comb,horton")
    p.add_argument("--grid", default="fine", help="HORTON MolGrid preset")
    p.add_argument("--chunk_gb", type=float, default=2.0, help="HORTON density-eval chunk size")
    p.add_argument("--horton_mem_gb", type=float, default=40.0, help="address-space cap for HORTON")
    p.add_argument("--timeout", type=int, default=4 * 3600, help="per Multiwfn/HORTON process, s")
    p.add_argument("--summary_only", action="store_true")
    args = p.parse_args()

    if args.horton_stages:
        return horton_stages(args)

    os.makedirs(args.out_dir, exist_ok=True)
    results_path = os.path.join(args.out_dir, "results.jsonl")
    if not args.summary_only:
        wfxs = list(args.wfx)
        if args.wfx_list:
            with open(args.wfx_list) as f:
                wfxs += [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
        if not shutil.which(args.multiwfn) and not os.path.isfile(args.multiwfn):
            sys.exit(f"Multiwfn not found: {args.multiwfn}")
        for w in wfxs:
            bench_one(w, args, results_path)
    summarize(results_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
