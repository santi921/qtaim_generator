"""Tier 1 of docs/engine_roadmap/qtaim_verification_campaign.md: full-precision
check of the engine's density kernels against an independent implementation
(HORTON's iodata + gbasis, unscreened), with no printed text in between.

For each wfx, points are drawn
  - at Multiwfn's CP positions (if a CPprop.txt is given or found next to it),
  - around every atom at log-spaced radii 0.01-6 Bohr (reaching the cusps),
  - uniformly in the molecular box,
and the engine's MO-only quantities (point_properties without the EDF) are
compared with gbasis: alpha and beta densities, spin, gradient, Hessian,
Laplacian, G = 1/2 sum n |grad phi|^2 and K, the latter checked through the
identity K = G - lap(rho)/4. ELF, LOL and ALIE are rebuilt here from HORTON's
own ingredients (per-spin densities, gradients, G, and MO values with |e_i|),
with Multiwfn's definitions written out again rather than imported, and compared
where rho > 1e-6 (below that the ELF ratio is roundoff). The EDF part is checked
separately against its closed form.

Runs in the generator env and calls the horton env for the reference values:

    python scripts/qtaim_horton_check.py --wfx_list wfx.txt --out_dir DIR \
        [--horton_python ~/miniconda3/envs/horton/bin/python]
"""

import argparse
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.abspath(__file__)


# ---------------------------------------------------------------- horton side (horton env)


def horton_eval(wfx_path, points_path, out_path):
    from gbasis.evals.density import (
        evaluate_density,
        evaluate_density_gradient,
        evaluate_density_hessian,
        evaluate_posdef_kinetic_energy_density,
    )
    from gbasis.evals.eval import evaluate_basis
    from gbasis.wrappers import from_iodata
    from iodata import load_one

    # iodata 1.0 cannot parse the EDF section Multiwfn writes; the comparison is MO-only anyway
    with open(wfx_path) as f:
        text = f.read()
    tag, end = "<Additional Electron Density Function (EDF)>", "</Additional Electron Density Function (EDF)>"
    if tag in text:
        # drop the trailing newline too: this iodata parser rejects blank lines
        text = text[: text.index(tag)] + text[text.index(end) + len(end):].lstrip("\n")
        wfx_path = out_path + ".noedf.wfx"
        with open(wfx_path, "w") as f:
            f.write(text)
    mol = load_one(wfx_path)
    basis = from_iodata(mol)
    pts = np.load(points_path)
    res = {}
    # iodata's occsa/occsb split restricted closed-shell (2 -> 1 + 1), ROKS (1 -> 1 + 0)
    # and natural orbitals (n -> n/2 + n/2); halving every restricted occupation
    # put ROKS singly occupied orbitals half in beta
    mo = mol.mo
    spins = {"a": (mo.coeffsa, mo.occsa, mo.energiesa), "b": (mo.coeffsb, mo.occsb, mo.energiesb)}
    kw = {"screen_basis": False}
    # gbasis's Hessian holds ~8 float64 arrays of shape (3, 3, nbasis, npoints); chunk to ~4 GB
    step = max(1, int(7e6 // len(mol.mo.coeffs)))
    for s, (c, o, e) in spins.items():
        dm = (c * o) @ c.T
        for key, fn in (("rho", evaluate_density), ("grad", evaluate_density_gradient),
                        ("hess", evaluate_density_hessian), ("G", evaluate_posdef_kinetic_energy_density)):
            res[f"{key}_{s}"] = np.concatenate([fn(dm, basis, pts[i:i + step], **kw) for i in range(0, len(pts), step)])
        # ALIE numerator sum_i n_i |e_i| phi_i^2, from HORTON's own MO values
        occ = o > 0
        w = o[occ] * np.abs(e[occ])
        res[f"alie_num_{s}"] = np.concatenate([
            ((c[:, occ].T @ evaluate_basis(basis, pts[i:i + step], **kw)) ** 2).T @ w
            for i in range(0, len(pts), step)])
    np.savez(out_path, **res)


# ---------------------------------------------------------------- engine side (generator env)


def make_points(wfx, cpprop, rng):
    coords = wfx["coords"]
    parts = []
    if cpprop and os.path.isfile(cpprop):
        from qtaim_gen.source.core.parse_qtaim import load_cpprop_full

        parts.append(np.array([c["pos_bohr"] for c in load_cpprop_full(cpprop)]))
    radii = np.logspace(-2, np.log10(6.0), 40)
    for a in coords:
        dirs = rng.normal(size=(len(radii), 3))
        dirs /= np.linalg.norm(dirs, axis=1)[:, None]
        parts.append(a + radii[:, None] * dirs)
    lo, hi = coords.min(axis=0) - 3, coords.max(axis=0) + 3
    parts.append(lo + (hi - lo) * rng.random((500, 3)))
    return np.concatenate(parts)


def edf_check(wfx, rng):
    """Engine EDF gradient/Hessian against central differences of its value."""
    from qtaim_gen.source.core.qtaim_engine import _edf_derivatives

    if not len(wfx["edf_center"]):
        return None
    c = wfx["coords"][wfx["edf_center"][0]]
    pts = c + rng.normal(scale=0.3, size=(200, 3))
    r, g, h = _edf_derivatives(pts, wfx["coords"], wfx)
    e = np.eye(3) * 1e-5
    g_fd = np.stack([(_edf_derivatives(pts + e[k], wfx["coords"], wfx)[0] - _edf_derivatives(pts - e[k], wfx["coords"], wfx)[0]) / 2e-5 for k in range(3)], axis=1)
    h_fd = np.stack([(_edf_derivatives(pts + e[k], wfx["coords"], wfx)[1] - _edf_derivatives(pts - e[k], wfx["coords"], wfx)[1]) / 2e-5 for k in range(3)], axis=2)
    return {"grad_rel": float(np.abs(g - g_fd).max() / np.abs(g).max()), "hess_rel": float(np.abs(h - h_fd).max() / np.abs(h).max())}


FC, FC_POL, ELF_ADDMINIMAL = 2.871234000, 4.557799872, 1e-5  # Multiwfn's constants


def elf_lol_reference(rho_a, rho_b, grad_a, grad_b, tau, spin_polarized):
    """Becke-Edgecombe ELF and the LOL of Schmider and Becke, as Multiwfn defines
    them (ELF_addminimal = 1e-5; restricted or spin-polarized by its wfntype rule),
    from per-spin densities, gradients and the positive-definite tau = G."""
    with np.errstate(divide="ignore", invalid="ignore"):
        if spin_polarized:
            dh = FC_POL * (rho_a ** (5 / 3) + rho_b ** (5 / 3))
            weiz = sum(np.where(r > 0, (g ** 2).sum(axis=1) / (8 * r), 0.0)
                       for r, g in ((rho_a, grad_a), (rho_b, grad_b)))
        else:
            rho, grad = rho_a + rho_b, grad_a + grad_b
            dh = FC * rho ** (5 / 3)
            weiz = np.where(rho > 0, (grad ** 2).sum(axis=1) / (8 * rho), 0.0)
        elf = 1 / (1 + ((tau - weiz + ELF_ADDMINIMAL) / dh) ** 2)
        t = np.where(tau > 0, dh / tau, 0.0)
        lol = t / (1 + t)
    return elf, lol


def rel_err(e, h, floor):
    """max over points of |e - h| / max(|h|, floor), with vector and matrix
    quantities measured by their norm at each point (so near-zero components
    of a gradient or Hessian do not read as large relative errors)."""
    e, h = np.asarray(e), np.asarray(h)
    d = (e - h).reshape(len(h), -1)
    ref = h.reshape(len(h), -1)
    dn = np.linalg.norm(d, axis=1)
    return float(np.max(dn / np.maximum(np.linalg.norm(ref, axis=1), floor))), float(np.max(dn))


def check_one(wfx_path, args):
    from qtaim_gen.source.core.charge_engine import prepare_basis, read_wfx
    from qtaim_gen.source.core.qtaim_engine import point_properties, wavefunction_type

    name = os.path.basename(os.path.dirname(os.path.abspath(wfx_path)))
    work = os.path.join(args.out_dir, name)
    os.makedirs(work, exist_ok=True)
    rng = np.random.default_rng(0)
    wfx = read_wfx(wfx_path)
    cpprop = args.cpprop_name and os.path.join(os.path.dirname(wfx_path), args.cpprop_name)
    pts = make_points(wfx, cpprop, rng)
    np.save(os.path.join(work, "points.npy"), pts)
    subprocess.run([args.horton_python, HERE, "--horton_eval", wfx_path, os.path.join(work, "points.npy"),
                    os.path.join(work, "horton.npz")], check=True)
    hz = np.load(os.path.join(work, "horton.npz"))

    mo_only = dict(wfx, edf_center=np.zeros(0, dtype=np.int64), edf_exp=np.zeros(0), edf_coef=np.zeros(0))
    # MO-only on purpose (HORTON has no EDF), so ECP atoms without EDF are expected here
    eng = point_properties(pts, mo_only, prepare_basis(wfx), allow_missing_edf=True)
    rho_h = hz["rho_a"] + hz["rho_b"]
    grad_h = hz["grad_a"] + hz["grad_b"]
    hess_h = hz["hess_a"] + hz["hess_b"]
    G_h = hz["G_a"] + hz["G_b"]
    lap_h = np.trace(hess_h, axis1=1, axis2=2)
    big = rho_h > 1e-10
    out = {"name": name, "natoms": len(wfx["coords"]), "points": len(pts), "points_rho_gt_1e-10": int(big.sum())}
    # relative errors on points that matter, plus the absolute error over all points
    for key, e, h, floor in (
        ("rho", eng["rho"], rho_h, 1e-10),
        ("rho_alpha", eng["rho_alpha"], hz["rho_a"], 1e-10),
        ("rho_beta", eng["rho_beta"], hz["rho_b"], 1e-10),
        ("gradient", eng["gradient"], grad_h, 1e-10),
        ("hessian", eng["hessian"], hess_h, 1e-10),
        ("laplacian", eng["laplacian"], lap_h, 1e-10),
        ("G", eng["G"], G_h, 1e-10),
        ("K_vs_G_minus_lap/4", eng["K"], G_h - lap_h / 4, 1e-10),
    ):
        out[key] = dict(zip(("rel", "abs"), rel_err(e, h, floor)))
    # ELF, LOL, ALIE from HORTON's ingredients; only where rho > 1e-6
    elf_h, lol_h = elf_lol_reference(hz["rho_a"], hz["rho_b"], hz["grad_a"], hz["grad_b"], G_h,
                                     spin_polarized=wavefunction_type(wfx) not in (0, 3))
    with np.errstate(divide="ignore", invalid="ignore"):
        alie_h = (hz["alie_num_a"] + hz["alie_num_b"]) / rho_h
    dense = rho_h > 1e-6
    out["points_rho_gt_1e-6"] = int(dense.sum())
    for key, e, h in (("elf", eng["elf"], elf_h), ("lol", eng["lol"], lol_h), ("alie", eng["alie"], alie_h)):
        out[key] = dict(zip(("rel", "abs"), rel_err(e[dense], h[dense], 1e-10)))
    spin_h = hz["rho_a"] - hz["rho_b"]
    d = np.abs(eng["spin"] - spin_h)
    # floor 1e-6 rho as compare_props.SPIN_FLOOR: spin roundoff scales with rho
    out["spin"] = {"rel": float(np.max(d / np.maximum(np.abs(spin_h), 1e-6 * np.maximum(rho_h, 1e-10)))),
                   "abs": float(np.max(d))}
    out["edf"] = edf_check(wfx, rng)
    with open(os.path.join(args.out_dir, "results.jsonl"), "a") as f:
        f.write(json.dumps(out) + "\n")
    return out


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--horton_eval":
        horton_eval(*sys.argv[2:5])
        return
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--wfx_list", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--horton_python", default=os.path.expanduser("~/miniconda3/envs/horton/bin/python"))
    p.add_argument("--cpprop_name", default=None, help="CPprop file name next to each wfx, if any")
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    print("job | atoms | points | rho rel | grad rel | hess rel | lap rel | G rel | K (via G-lap/4) rel | spin rel | ELF | LOL | ALIE rel | max abs (rho) | EDF grad/hess vs FD")
    for line in open(args.wfx_list):
        if not line.strip():
            continue
        r = check_one(line.strip(), args)
        edf = r["edf"]
        print(" | ".join(str(x) for x in (
            r["name"][:30], r["natoms"], r["points"],
            *(f"{r[k]['rel']:.1e}" for k in ("rho", "gradient", "hessian", "laplacian", "G", "K_vs_G_minus_lap/4",
                                             "spin", "elf", "lol", "alie")),
            f"{r['rho']['abs']:.1e}", f"{edf['grad_rel']:.1e}/{edf['hess_rel']:.1e}" if edf else "-",
        )), flush=True)


if __name__ == "__main__":
    main()
