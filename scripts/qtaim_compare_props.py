"""Criterion A of the QTAIM engine plan: at Multiwfn's own CP positions (from
CPprop.txt, printed to 12 decimals in Bohr), compare every property the engine
evaluates (qtaim_engine.point_properties) with Multiwfn's printed value.

Reports per property the largest relative difference |e - m| / |m| (absolute
for ellipticity and eta, printed with 6 decimals, and for spin; the gradient
against the position-rounding bound), split by CP type. A NaN on either side,
or a property missing from CPprop.txt, reports inf, so it fails any threshold.

    python scripts/qtaim_compare_props.py --pairs pairs.txt
where each line of pairs.txt is "path/to/CPprop.txt path/to/orca.wfx".
"""

import argparse

import numpy as np

from qtaim_gen.source.core.charge_engine import prepare_basis, read_wfx
from qtaim_gen.source.core.parse_qtaim import load_cpprop_full
from qtaim_gen.source.core.qtaim_engine import point_properties

SCALARS = {
    "Density of all electrons": "rho",
    "Density of Alpha electrons": "rho_alpha",
    "Density of Beta electrons": "rho_beta",
    "Spin density of electrons": "spin",
    "Lagrangian kinetic energy G(r)": "G",
    "Hamiltonian kinetic energy K(r)": "K",
    "Potential energy density V(r)": "V",
    "Energy density E(r) or H(r)": "H",
    "Laplacian of electron density": "laplacian",
    "Electron localization function (ELF)": "elf",
    "Localized orbital locator (LOL)": "lol",
    "Average local ionization energy (ALIE)": "alie",
    "Determinant of Hessian": "det_hessian",
    "Ellipticity of electron density": "ellipticity",
    "eta index": "eta",
}
ABSOLUTE = {"ellipticity", "eta", "spin"}


def _worst(d):
    """Largest difference, inf if any is NaN (np.nanmax would drop it)."""
    d = np.asarray(d, dtype=float)
    return float(np.inf) if np.isnan(d).any() else float(np.max(d))


def compare_cps(cps, wfx, basis):
    """Per CP type: (count, {property: max difference}) between Multiwfn's
    printed values for cps (load_cpprop_full) and the engine at the printed
    positions. gradient_bound is max |g_engine - g_mwfn| over the rounding
    bound ||H||_2 * 8.7e-13 + 5e-11 |g| + 1e-20 (positions printed to 1e-12
    Bohr per coordinate, values to E18.10); <= 1 means consistent."""
    pts = np.array([c["pos_bohr"] for c in cps])
    eng = point_properties(pts, wfx, basis)
    out = {}
    for lab in ("NCP", "BCP", "RCP", "CCP"):
        idx = [i for i, c in enumerate(cps) if c["label"] == lab]
        if not idx:
            continue
        res = {}
        for name, key in SCALARS.items():
            ref = np.array([cps[i]["props"].get(name, np.nan) for i in idx])
            val = eng[key][idx]
            if key in ABSOLUTE:
                d = np.abs(val - ref)
            else:
                d = np.abs(val - ref) / np.maximum(np.abs(ref), 1e-300)
            res[key] = _worst(d)
        ev_ref = np.array([cps[i]["eigenvalues"] for i in idx])
        res["eigenvalues"] = _worst(np.abs(eng["eigenvalues"][idx] - np.sort(ev_ref, axis=1)) / np.maximum(np.abs(ev_ref), 1e-300))
        g_ref = np.array([cps[i]["gradient"] for i in idx])
        dg = np.linalg.norm(eng["gradient"][idx] - g_ref, axis=1)
        bound = np.abs(eng["eigenvalues"][idx]).max(axis=1) * 8.7e-13 + 5e-11 * np.linalg.norm(g_ref, axis=1) + 1e-20
        res["gradient_bound"] = _worst(dg / bound)
        out[lab] = (len(idx), res)
    return out


def compare(cpprop, wfx_path):
    wfx = read_wfx(wfx_path)
    return compare_cps(load_cpprop_full(cpprop), wfx, prepare_basis(wfx))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pairs", required=True)
    args = p.parse_args()
    worst = {}
    for line in open(args.pairs):
        if not line.strip():
            continue
        cpprop, wfx = line.split()
        res = compare(cpprop, wfx)
        name = cpprop.split("/")[-2]
        for lab, (n, r) in res.items():
            print(f"{name[:34]} | {lab} x{n} | " + " ".join(f"{k} {v:.1e}" for k, v in r.items()))
            for k, v in r.items():
                worst[(lab, k)] = max(worst.get((lab, k), 0.0), v)  # v is never NaN (_worst)
    print("\nworst over all jobs (relative; absolute for ellipticity, eta, spin; gradient_bound <= 1 is consistent):")
    for lab in ("NCP", "BCP", "RCP", "CCP"):
        row = {k: v for (l2, k), v in worst.items() if l2 == lab}
        if row:
            print(f"{lab}: " + " ".join(f"{k} {v:.1e}" for k, v in row.items()))


if __name__ == "__main__":
    main()
