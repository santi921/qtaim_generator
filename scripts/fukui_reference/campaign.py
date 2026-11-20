"""Condensed Fukui functions: charge engine vs Multiwfn on the same molden.

Per job: the molden (copied, or written from the gbw by orca_2mkl) and the orca.wfx the native
converter writes from it; Multiwfn reads the molden (it needs the unoccupied orbitals):
  - menu 8 option 8, Hirshfeld composition of every orbital (-4, orbcomp.txt), on the engine's
    Fukui grid (iautointgrid=0, 75 x 434, radcut=0: every radial point, as CDFT option 6);
  - closed shells only (Multiwfn refuses others): CDFT option 6, condensed orbital-weighted f+ and
    f-, on the engine's grid and on the production one (iautointgrid=1: 30 x 302).
Then the charge engine at full_set 0 with the molden. Compared:
  - comp_max: every window orbital's normalized composition against orbcomp.txt, the engine's
    composition (molden orbitals, full radial grid) taken with Multiwfn's 1e-9 weight cut (the
    -4 export skips smaller weights); bar 5e-6 (f11.3 %);
  - ow_max (closed shells): the engine's fukui_plus_ow / fukui_minus_ow against CDFT option 6;
    bar 5e-6 (f15.5); ow_grid_max: the same against the production grid (the divergence);
  - engine_path_max: run()'s values against fukui_values on the compositions of every window
    orbital (no truncation, molden orbitals only): the cost of FUKUI_DROP and of taking the
    occupied orbitals from the wfx;
  - fmo_max / ow_comp_max: the engine's FMO and OW values against fukui_values on Multiwfn's
    compositions (rescaled to the engine's per-orbital totals, which Multiwfn does not print),
    which covers open shells.

    python campaign.py one JOB_DIR OUT_ROOT --src SRC --multiwfn PATH [--orca_2mkl PATH] [--nthreads 8]
    python campaign.py summarize OUT_ROOT

`one` writes OUT_ROOT/<JOB_DIR relative to SRC>/result.json; a job with a result.json is skipped.
"""

import argparse
import glob
import json
import os
import re
import resource
import shutil
import subprocess
import sys
import time

STDIN_COMP = "8\n8\n1\n-4\n0\n-10\nq\n"
STDIN_OW = "22\n6\n0\nq\n"
OW_ROW = re.compile(r"^\s*(\d+)\((\w+)\s*\)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s*$", re.M)
VERSION_LINE = re.compile(r"update date:\s*(\S+)")


def unlimited_stack():
    resource.setrlimit(resource.RLIMIT_STACK, (resource.RLIM_INFINITY, resource.RLIM_INFINITY))


def parse_orbcomp(path):
    """orbcomp.txt -> {0-based orbital: [fraction per atom]} (Multiwfn's normalized %)."""
    out, cur = {}, None
    for line in open(path):
        m = re.match(r"\s*Orbital\s+(\d+)", line)
        if m:
            cur = int(m[1]) - 1
            out[cur] = []
        elif line.strip():
            out[cur].append(float(line.split()[1]) / 100)
    return out


def parse_ow(path):
    """CDFT option 6 -> [[f+, f-] per atom]."""
    return [[float(m[3]), float(m[4])] for m in OW_ROW.finditer(open(path, errors="replace").read())]


def prepare(job, wd, orca_2mkl):
    """orca.molden.input and orca.wfx (native converter, fixed) in wd."""
    from qtaim_gen.source.core.wavefunction import read_orca_molden, write_wfx

    molden = os.path.join(wd, "orca.molden.input")
    if not os.path.exists(molden):
        found = sorted(glob.glob(os.path.join(job, "*.molden.input")))
        if found:
            shutil.copy(found[0], molden)
        else:
            gbws = [g for g in sorted(glob.glob(os.path.join(job, "*.gbw*"))) if not re.search(r"_atom\d", g)]
            if not gbws or not orca_2mkl:
                raise FileNotFoundError(f"no molden, or gbw without --orca_2mkl, in {job}")
            g = gbws[0]
            if g.endswith(".zstd0"):
                subprocess.run(["unzstd", "-q", "-f", "-o", os.path.join(wd, "orca.gbw"), g], check=True)
            else:
                shutil.copy(g, os.path.join(wd, "orca.gbw"))
            subprocess.run([orca_2mkl, "orca", "-molden"], cwd=wd, check=True, capture_output=True)
            os.remove(os.path.join(wd, "orca.gbw"))
    wfx = os.path.join(wd, "orca.wfx")
    if not os.path.exists(wfx):
        write_wfx(read_orca_molden(molden), wfx)
    return molden, wfx


def multiwfn(args, wd, name, stdin, auto_grid):
    """Run Multiwfn on the molden with settings.ini at the engine grid (auto_grid=False) or
    with iautointgrid=1 (production); returns the stdout path and wall time."""
    from qtaim_gen.source.core.omol import write_settings_file

    out = os.path.join(wd, f"{name}.out")
    if os.path.exists(out):
        return out, None
    write_settings_file(wd, n_threads=args.nthreads)
    path = os.path.join(wd, "settings.ini")
    text = open(path).read()
    if not auto_grid:
        text, n = re.subn(r"iautointgrid= 1", "iautointgrid= 0", text)
        text, m = re.subn(r"radcut= 10\b", "radcut= 0", text)
        assert n == 1 and m == 1, "iautointgrid or radcut line not found in settings.ini"
    open(path, "w").write(text)
    t0 = time.perf_counter()
    with open(out, "w") as f:
        subprocess.run([args.multiwfn, "orca.molden.input"], input=stdin, text=True, stdout=f,
                       stderr=subprocess.STDOUT, cwd=wd, preexec_fn=unlimited_stack,
                       env=dict(os.environ, OMP_STACKSIZE="1G"))
    return out, time.perf_counter() - t0


def engine_compositions(molden, wfx, cut=None):
    """Per-orbital Hirshfeld compositions of every window orbital from the molden on the full
    radial grid (the integrals run() takes, without its truncation), optionally with
    Multiwfn's weight cut; returns (comp, window wfx dict, window indices in file order)."""
    import numpy as np

    from qtaim_gen.source.core import charge_engine as ce
    from qtaim_gen.source.core.wavefunction import frontier_window, molden_mo_headers, read_orca_molden

    w = ce.read_wfx(wfx)
    fw = read_orca_molden(molden, orbital_window=ce.FUKUI_WINDOW)
    idx = sorted(frontier_window(molden_mo_headers(molden), ce.FUKUI_WINDOW))
    fb = ce.prepare_basis(fw, keep=np.ones(len(fw["occ"]), dtype=bool))
    pro, _ = ce.mwfn_proatom_tables(w["atnums"])
    gp, gw = ce.multiwfn_atom_grid(75, 434, np.inf)
    comp = np.zeros((len(w["atnums"]), len(idx)))
    for b in range(len(w["atnums"])):
        pts = gp + w["coords"][b]
        h, _, _ = ce._hirshfeld_mwfn(pts, w["coords"], *pro)
        hw = h[:, b] * gw
        if cut is not None:
            hw = np.where(hw < cut, 0.0, hw)
        comp[b] = hw @ ce.orbital_values(pts, w["coords"], fb) ** 2
    return comp, fw, idx


def fukui_from_multiwfn(R, comp_cut, fw):
    """fmo_max / ow_comp_max: fukui_values on Multiwfn's normalized compositions R, rescaled to
    the engine's per-orbital totals, against fukui_values on the engine's compositions; both
    with Multiwfn's 1e-9 weight cut, which removes up to 7.6e-4 of a lanthanide orbital's
    composition (so an uncut engine total would not match R). run() itself takes no cut;
    engine_path_max covers it."""
    import numpy as np

    from qtaim_gen.source.core import charge_engine as ce

    args = (fw["mo_energy"], fw["occ"], fw["spin_types"])
    mw, _ = ce.fukui_values(R * comp_cut.sum(0), *args)
    eng, _ = ce.fukui_values(comp_cut, *args)
    return {"fmo_max": max(float(np.abs(mw[k] - eng[k]).max()) for k in ("plus_fmo", "minus_fmo")),
            "ow_comp_max": max(float(np.abs(mw[k] - eng[k]).max()) for k in ("plus_ow", "minus_ow"))}


def one(args):
    import numpy as np

    from qtaim_gen.source.core import charge_engine as ce

    job = os.path.abspath(args.job)
    rel = os.path.relpath(job, os.path.abspath(args.src))
    wd = os.path.join(args.out_root, rel)
    if os.path.exists(os.path.join(wd, "result.json")):
        print("done already:", rel)
        return
    os.makedirs(wd, exist_ok=True)
    molden, wfx = prepare(job, wd, args.orca_2mkl)
    w = ce.read_wfx(wfx)
    closed = w["mult"] == 1 and all(s == "Alpha and Beta" for s in w["spin_types"])
    rec = {"job": rel, "natoms": len(w["atnums"]), "closed_shell": closed, "mult": int(w["mult"])}

    out, rec["mw_comp_s"] = multiwfn(args, wd, "comp", STDIN_COMP, auto_grid=False)
    shutil.move(os.path.join(wd, "orbcomp.txt"), os.path.join(wd, "orbcomp_engine_grid.txt"))
    build = VERSION_LINE.search(open(out, errors="replace").read())
    rec["mw_build"] = build[1] if build else None
    if closed:
        _, rec["mw_ow_s"] = multiwfn(args, wd, "ow", STDIN_OW, auto_grid=False)
        multiwfn(args, wd, "ow_prod_grid", STDIN_OW, auto_grid=True)

    t0 = time.perf_counter()
    res = ce.run(wfx, molden_path=molden)
    rec["engine_s"] = time.perf_counter() - t0
    meta = res["_meta"]
    rec.update({"fukui_s": meta["timings_s"].get("fukui"), "fukui_meta": meta["fukui"],
                "skipped": [s["schemes"] for s in meta["skipped"]]})
    if meta["fukui"] is None:
        with open(os.path.join(wd, "result.json"), "w") as f:
            json.dump(rec, f, indent=1)
        print(rel, "no Fukui functions:", rec["skipped"], flush=True)
        return
    labels = ce.atom_labels(w["atnums"])
    eng = {k: np.array([res[k][k][lab] for lab in labels]) for k in res if k.startswith("fukui_")}

    ref = parse_orbcomp(os.path.join(wd, "orbcomp_engine_grid.txt"))
    comp_cut, fw, idx = engine_compositions(molden, wfx, cut=1e-9)
    R = np.array([ref[i] for i in idx]).T
    rec["comp_max"] = float(np.abs(comp_cut / comp_cut.sum(0) - R).max())
    rec["window_orbitals"] = len(idx)

    comp, _, _ = engine_compositions(molden, wfx)
    same, _ = ce.fukui_values(comp, fw["mo_energy"], fw["occ"], fw["spin_types"])
    rec["engine_path_max"] = max(float(np.abs(same[k[6:]] - v).max()) for k, v in eng.items())
    rec.update(fukui_from_multiwfn(R, comp_cut, fw))
    rec["sums"] = {k: float(v.sum()) for k, v in eng.items()}

    if closed:
        ow = np.array(parse_ow(os.path.join(wd, "ow.out")))
        ow_prod = np.array(parse_ow(os.path.join(wd, "ow_prod_grid.out")))
        eng_ow = np.stack([eng["fukui_plus_ow"], eng["fukui_minus_ow"]], axis=1)
        rec["ow_max"] = float(np.abs(eng_ow - ow).max())
        rec["ow_grid_max"] = float(np.abs(eng_ow - ow_prod).max())
    with open(os.path.join(wd, "result.json"), "w") as f:
        json.dump(rec, f, indent=1)
    print(rel, {k: rec.get(k) for k in ("comp_max", "ow_max", "fmo_max", "ow_comp_max", "engine_path_max",
                                        "ow_grid_max")}, flush=True)


def summarize(args):
    import numpy as np

    recs = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(args.out_root, "**", "result.json"),
                                                         recursive=True))]
    done = [r for r in recs if r.get("fukui_meta")]
    print(f"jobs {len(recs)} | with Fukui {len(done)} | closed shell {sum(r['closed_shell'] for r in done)} | "
          f"Multiwfn builds {sorted({str(r.get('mw_build')) for r in recs})}")
    for r in recs:
        if not r.get("fukui_meta"):
            print("  no Fukui:", r["job"], r["skipped"])
    for key, bar in (("comp_max", 5e-6), ("ow_max", 5e-6), ("fmo_max", None), ("ow_comp_max", None),
                     ("engine_path_max", 1e-9), ("ow_grid_max", None)):
        v = np.array([r[key] for r in done if r.get(key) is not None])
        if len(v):
            worst = max((r for r in done if r.get(key) is not None), key=lambda r: r[key])
            above = f" | above {bar:g}: {(v > bar + 1e-12).sum()}" if bar is not None else ""
            print(f"{key}: jobs {len(v)} | max {v.max():.2e} median {np.median(v):.2e}{above} | worst {worst['job']}")
    sums = np.array([[r["sums"][k] for k in sorted(r["sums"])] for r in done])
    if len(sums):
        print("sums (min, max) per key:", {k: (round(float(sums[:, i].min()), 6), round(float(sums[:, i].max()), 6))
                                            for i, k in enumerate(sorted(done[0]["sums"]))})
    for key in ("fukui_s", "engine_s", "mw_comp_s", "mw_ow_s"):
        v = [r[key] for r in recs if r.get(key) is not None]
        if v:
            print(f"{key}: median {np.median(v):.2f} s, max {max(v):.1f} s, total {sum(v):.0f} s")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    o = sub.add_parser("one")
    o.add_argument("job")
    o.add_argument("out_root")
    o.add_argument("--src", required=True)
    o.add_argument("--multiwfn", required=True)
    o.add_argument("--orca_2mkl")
    o.add_argument("--nthreads", type=int, default=8)
    s = sub.add_parser("summarize")
    s.add_argument("out_root")
    args = p.parse_args()
    one(args) if args.cmd == "one" else summarize(args)


if __name__ == "__main__":
    sys.exit(main())
