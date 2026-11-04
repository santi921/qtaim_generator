"""Run Chargemol 09_26_2017 (DDEC6) on wfx files and parse its outputs to JSON.

Validation reference for the native DDEC6 port; the pipeline never calls Chargemol.
Build the binaries with build_chargemol.sh first.

    python run_chargemol.py run orca.wfx [more.wfx ...] [--list folders.txt] [--jobs 8]
    python run_chargemol.py golden --binary ~/dev/chargemol_src/bin/chargemol_serial

run writes <out_name> (default ddec6_ref.json) next to each wfx:
    {"ddec6": {"charge", "spin", "atomic_dipole", "atomic_quadrupole",
               "quadrupole_eigenvalues", "r2", "r3", "r4", "tail_fit"},
     "ddec6_bond": {"1_O_to_2_O": bo}, "ddec6_sbo": {"1_O": sbo},
     "overlap_population": {"1_O_to_2_O": p}, "_meta": {...}}
Atom keys are "{1-based index}_{element}"; quadrupole components are Chargemol's
(Qxy, Qxz, Qyz, Q(x^2-y^2), Q(3z^2-R^2)); dipoles and quadrupoles in atomic units.

golden runs Chargemol's bundled Gaussian wfx examples and compares every parsed value
with the outputs shipped next to them (shipped files are f13.6 / f10.4 text).
"""

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor

CHARGEMOL = os.path.expanduser("~/dev/chargemol_src")
DEFAULT_BINARY = os.path.join(CHARGEMOL, "bin", "chargemol_serial_precise")
DEFAULT_DENSITIES = os.path.join(CHARGEMOL, "chargemol_09_26_2017", "atomic_densities") + "/"

PREFIX = "DDEC6_even_tempered_"
ATOM_ROW = re.compile(r"^\s*(\d+)\s+([A-Za-z]{1,3})\s+(.*)$")
BO_ROW = re.compile(r"atom number\s+(\d+)\s+\(\s*([A-Za-z]+)\s*\)\s+with bond order =\s*(\S+)\s+"
                    r"The average spin polarization of this bonding =\s*(\S+)")


def _key(idx, sym):
    return f"{int(idx)}_{sym.capitalize()}"


def _floats(text):
    return [float(t) for t in text.split()]


def _rows_after(lines, header_start, natoms):
    """natoms rows "idx sym v1 v2 ..." following the line that starts with header_start."""
    for i, line in enumerate(lines):
        if line.strip().startswith(header_start):
            out = {}
            for row in lines[i + 1:i + 1 + natoms]:
                m = ATOM_ROW.match(row)
                out[_key(m.group(1), m.group(2))] = _floats(m.group(3))
            return out
    return None


def _bond_key(i, ei, j, ej):
    (i, ei), (j, ej) = sorted([(int(i), ei.capitalize()), (int(j), ej.capitalize())])
    return f"{i}_{ei}_to_{j}_{ej}"


def parse_outputs(workdir):
    """Parse the DDEC6 output files Chargemol leaves in workdir."""
    res, ddec6 = {}, {}
    meta = {}

    with open(os.path.join(workdir, PREFIX + "net_atomic_charges.xyz")) as f:
        lines = f.read().splitlines()
    natoms = int(lines[0].split()[0])
    for line in lines:
        if "Number of iterations to convergence" in line:
            meta["n_iterations"] = int(line.split("=")[1])
        elif "Error in the integrated total number of electrons before renormalization" in line:
            meta["electron_error_before_renorm"] = float(line.split("=")[1])
    rows = _rows_after(lines, "atom number, atomic symbol, x, y, z, net_charge", natoms)
    ddec6["charge"] = {k: v[3] for k, v in rows.items()}
    ddec6["atomic_dipole"] = {k: v[4:7] for k, v in rows.items()}
    ddec6["atomic_quadrupole"] = {k: v[8:13] for k, v in rows.items()}
    ddec6["quadrupole_eigenvalues"] = {k: v[13:16] for k, v in rows.items()}
    rows = _rows_after(lines, "atom number,atomic symbol, x, y, z, a, b, Rsquared", natoms)
    ddec6["tail_fit"] = {k: v[3:6] for k, v in rows.items()}

    spin_file = os.path.join(workdir, PREFIX + "atomic_spin_moments.xyz")
    if os.path.exists(spin_file):
        with open(spin_file) as f:
            spin = {}
            for line in f:
                m = ATOM_ROW.match(line)
                if m and len(m.group(3).split()) == 4:
                    spin[_key(m.group(1), m.group(2))] = _floats(m.group(3))[3]
        if spin:
            ddec6["spin"] = spin

    for tag, name in (("r2", "Rsquared"), ("r3", "Rcubed"), ("r4", "Rfourth")):
        path = os.path.join(workdir, f"DDEC_atomic_{name}_moments.xyz")
        if os.path.exists(path):
            with open(path) as f:
                rows = _rows_after(f.read().splitlines(), "Same information as above printed with atom number", natoms)
            ddec6[tag] = {k: v[3] for k, v in rows.items()}
    res["ddec6"] = ddec6

    bo_file = os.path.join(workdir, PREFIX + "bond_orders.xyz")
    if os.path.exists(bo_file):
        bonds, sbo, current = {}, {}, None
        with open(bo_file) as f:
            for line in f:
                m = re.search(r"Printing BOs for ATOM #\s*(\d+)\s*\(\s*([A-Za-z]+)\s*\)", line)
                if m:
                    current = (m.group(1), m.group(2))
                    continue
                m = BO_ROW.search(line)
                if m and current:
                    bonds[_bond_key(*current, m.group(1), m.group(2))] = float(m.group(3))
                elif current and "The sum of bond orders for this atom is SBO" in line:
                    sbo[_key(*current)] = float(line.split("=")[1])
        res["ddec6_bond"], res["ddec6_sbo"] = bonds, sbo

    ov_file = os.path.join(workdir, "overlap_populations.xyz")
    if os.path.exists(ov_file):
        syms = {int(k.split("_")[0]): k.split("_")[1] for k in ddec6["charge"]}
        ov = {}
        with open(ov_file) as f:
            for line in f:
                t = line.split()
                if len(t) == 6 and t[0].isdigit():
                    i, j = int(t[0]), int(t[1])
                    ov[_bond_key(i, syms[i], j, syms[j])] = float(t[5])
        res["overlap_population"] = ov

    res["_meta"] = meta
    return res


def _stage_wfx(src, dst):
    """Link the wfx into the scratch directory. Multiwfn writes the EDF primitive types
    of its built-in core densities as 0 (fileIO.f90:4058, outwfx 7936) and the native
    writer copies that; Chargemol accepts only wfx type codes (s = 1) and stops. Those
    EDF functions are s Gaussians, so a copy with the types set to 1 is staged instead.
    Returns True when the copy was rewritten."""
    with open(src) as f:
        text = f.read()
    start = text.find("<EDF Primitive Types>")
    if start < 0:
        os.symlink(src, dst)
        return False
    end = text.index("</EDF Primitive Types>", start)
    block = text[start:end].split("\n")
    block = [block[0]] + [re.sub(r"\b0\b", "1", line) for line in block[1:]]
    with open(dst, "w") as f:
        f.write(text[:start] + "\n".join(block) + text[end:])
    return True


def _out_path(wfx, out_name, out_dir):
    if out_dir:
        return os.path.join(out_dir, os.path.basename(os.path.dirname(os.path.abspath(wfx))) + ".json")
    return os.path.join(os.path.dirname(os.path.abspath(wfx)), out_name)


def run_one(wfx, binary=DEFAULT_BINARY, densities=DEFAULT_DENSITIES, out_name="ddec6_ref.json",
            out_dir=None, keep=False, timeout=None, threads=1):
    """Run Chargemol on one wfx in a scratch directory; return (wfx, status, seconds)."""
    wfx = os.path.abspath(wfx)
    out_path = _out_path(wfx, out_name, out_dir)
    work = tempfile.mkdtemp(prefix="chargemol_")
    try:
        edf_rewritten = _stage_wfx(wfx, os.path.join(work, "input.wfx"))
        with open(os.path.join(work, "job_control.txt"), "w") as f:
            f.write("<atomic densities directory complete path>\n" + densities + "\n"
                    "</atomic densities directory complete path>\n\n"
                    "<input filename>\ninput.wfx\n</input filename>\n\n"
                    "<charge type>\nDDEC6\n</charge type>\n\n"
                    "<compute BOs>\n.true.\n</compute BOs>\n")
        env = dict(os.environ, OMP_NUM_THREADS=str(threads))
        t0 = time.time()
        with open(os.path.join(work, "Chargemol.out"), "w") as log:
            proc = subprocess.run([binary], cwd=work, stdout=log, stderr=subprocess.STDOUT, env=env, timeout=timeout)
        seconds = time.time() - t0
        try:
            res = parse_outputs(work)
        except (OSError, ValueError, AttributeError, TypeError, IndexError) as e:
            tail = ""
            out_file = os.path.join(work, "input.output")
            if os.path.exists(out_file):
                with open(out_file) as f:
                    tail = "".join(f.readlines()[-5:])
            return wfx, f"failed (rc {proc.returncode}): {type(e).__name__} {e}; {tail.strip()}", seconds
        res["_meta"].update(binary=os.path.basename(binary), seconds=round(seconds, 3),
                            returncode=proc.returncode, threads=threads, wfx=wfx,
                            edf_types_rewritten=edf_rewritten)
        with open(out_path, "w") as f:
            json.dump(res, f, indent=1)
        return wfx, "ok", seconds
    finally:
        if keep:
            print(f"kept {work}", file=sys.stderr)
        else:
            shutil.rmtree(work, ignore_errors=True)


def _wfx_list(args):
    paths = list(args.wfx)
    if args.list:
        with open(args.list) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    paths.append(line if line.endswith(".wfx") else os.path.join(line, "orca.wfx"))
    return paths


def cmd_run(args):
    paths = _wfx_list(args)
    if args.skip_existing:
        paths = [p for p in paths if not os.path.exists(_out_path(p, args.out_name, args.out_dir))]
    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
    kw = dict(binary=args.binary, densities=args.atomic_densities, out_name=args.out_name,
              out_dir=args.out_dir, keep=args.keep, timeout=args.timeout, threads=args.threads)
    n_fail = 0
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        for wfx, status, sec in pool.map(_run_star, [(p, kw) for p in paths]):
            n_fail += status != "ok"
            print(f"{status:>6s} {sec:9.1f} s  {wfx}", flush=True)
    print(f"{len(paths) - n_fail}/{len(paths)} ok")
    return 1 if n_fail else 0


def _run_star(item):
    path, kw = item
    return run_one(path, **kw)


def _compare(a, b, path, worst):
    """Walk two parsed results; record the worst absolute difference per field
    (ddec6/<field> or the top-level key); a key on one side only counts as inf."""
    if isinstance(a, dict):
        for k in set(a) | set(b):
            if k == "_meta":
                continue
            sub = f"{path}/{k}" if path else k
            if k not in a or k not in b:
                worst[sub] = math.inf
            else:
                _compare(a[k], b[k], sub, worst)
    elif isinstance(a, list):
        for x, y in zip(a, b):
            _compare(x, y, path, worst)
    else:
        parts = path.split("/")
        field = "/".join(parts[:2]) if parts[0] == "ddec6" else parts[0]
        worst[field] = max(worst.get(field, 0.0), abs(a - b))


def cmd_golden(args):
    root = os.path.join(CHARGEMOL, "chargemol_09_26_2017", "examples_to_run")
    examples = sorted(d for d in os.listdir(root) if d.startswith("GAUSSIAN_WFX") and
                      os.path.isdir(os.path.join(root, d, "DDEC6")))
    ok = True
    for ex in examples:
        wfx = [f for f in os.listdir(os.path.join(root, ex)) if f.endswith(".wfx")][0]
        tmp = tempfile.mkdtemp(prefix="golden_")
        shutil.copy(os.path.join(root, ex, wfx), os.path.join(tmp, "orca.wfx"))
        _, status, sec = run_one(os.path.join(tmp, "orca.wfx"), binary=args.binary,
                                 densities=args.atomic_densities, threads=args.threads)
        if status != "ok":
            print(f"{ex}: {status}")
            ok = False
            continue
        with open(os.path.join(tmp, "ddec6_ref.json")) as f:
            mine = json.load(f)
        shipped = parse_outputs(os.path.join(root, ex, "DDEC6"))
        worst = {}
        _compare(shipped, mine, "", worst)
        # shipped text: bond orders f10.4, overlap populations f.10, the rest f13.6 / f11.6;
        # a matching run differs by at most half a unit in the last printed place
        limits = {"ddec6_bond": 5.0001e-5, "overlap_population": 5.0001e-11}
        bad = {k: v for k, v in worst.items() if v > limits.get(k, 5.0001e-7)}
        ok &= not bad and shipped["_meta"].get("n_iterations") == mine["_meta"].get("n_iterations")
        print(f"{ex}: {sec:.1f} s, iterations {mine['_meta'].get('n_iterations')} "
              f"(shipped {shipped['_meta'].get('n_iterations')}), "
              f"worst {max(worst.values()):.2e}" + (f", OVER: {bad}" if bad else ""))
        shutil.rmtree(tmp, ignore_errors=True)
    print("golden: PASS" if ok else "golden: FAIL")
    return 0 if ok else 1


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("run", "golden"):
        s = sub.add_parser(name)
        s.add_argument("--binary", default=DEFAULT_BINARY)
        s.add_argument("--atomic_densities", default=DEFAULT_DENSITIES)
        s.add_argument("--threads", type=int, default=1, help="OMP_NUM_THREADS for the _omp builds")
    r = sub.choices["run"]
    r.add_argument("wfx", nargs="*")
    r.add_argument("--list", help="file of job folders (orca.wfx inside) or wfx paths, one per line")
    r.add_argument("--jobs", type=int, default=1, help="Chargemol processes in parallel")
    r.add_argument("--out_name", default="ddec6_ref.json")
    r.add_argument("--out_dir", help="write <parent folder name>.json here instead of next to the wfx")
    r.add_argument("--skip_existing", action="store_true")
    r.add_argument("--keep", action="store_true", help="keep the scratch directories")
    r.add_argument("--timeout", type=float)
    args = p.parse_args(argv)
    return cmd_run(args) if args.cmd == "run" else cmd_golden(args)


if __name__ == "__main__":
    sys.exit(main())
