"""QTAIM engine (docs/engine_roadmap/qtaim_engine_plan.md).

Phase P1: the per-point properties Multiwfn's showptprop prints for each
critical point (MW/sub.f90:2347-2510), evaluated from MO values, gradients and
Hessians (charge_engine.orbital_derivatives) plus the EDF core density:

  rho (MOs + EDF), alpha/beta = (rho +- spin)/2, spin (MO types only)
  G = 1/2 sum n_i |grad phi_i|^2 and its x/y/z parts (Lagkin, MOs only)
  K = -1/2 sum n_i phi_i lap phi_i (Hamkin, MOs only); V = -K - G; H = -K
  Laplacian, gradient and Hessian of rho (MOs + EDF)
  ELF, LOL (Becke forms, MOs only; restricted or spin-polarized by Multiwfn's
  wfntype rule; ELF_addminimal=1), ALIE
  Hessian eigenvalues, determinant, ellipticity, eta

ESP and delta-g follow in P2; CP search in P3.
"""

import numpy as np

from qtaim_gen.source.core.charge_engine import limit_blas_threads, orbital_derivatives

FC = 2.871234000  # Thomas-Fermi constant (3/10)(3 pi^2)^(2/3), as Multiwfn
FC_POL = 4.557799872  # spin-polarized (3/10)(6 pi^2)^(2/3)
ELF_ADDMINIMAL = 1e-5


def wavefunction_type(wfx):
    """Multiwfn's wfntype for a wfx (readwfx, fileIO.f90): with integer
    occupations, 1 (unrestricted) when there are as many MOs as electrons, 0
    (restricted closed shell) when half as many, else 2 (restricted open
    shell); 3/4 for fractional occupations (natural orbitals)."""
    occ = wfx["occ"]
    nelec = occ.sum()
    if np.all(occ == np.rint(occ)):
        if len(occ) == int(round(nelec)):
            return 1
        if len(occ) == int(round(nelec)) // 2:
            return 0
        return 2
    alpha = sum(o for o, t in zip(occ, wfx["spin_types"]) if t == "Alpha")
    beta = sum(o for o, t in zip(occ, wfx["spin_types"]) if t == "Beta")
    return 3 if alpha == beta else 4


def _edf_derivatives(pts, coords, wfx):
    """EDF core density (s Gaussians) with gradient and Hessian."""
    n = len(pts)
    rho = np.zeros(n)
    grad = np.zeros((n, 3))
    hess = np.zeros((n, 3, 3))
    for c, a, coef in zip(wfx["edf_center"], wfx["edf_exp"], wfx["edf_coef"]):
        d = pts - coords[c]
        ar2 = a * (d**2).sum(axis=1)
        keep = ar2 <= 40.0
        v = np.where(keep, coef * np.exp(-np.where(keep, ar2, 0.0)), 0.0)
        rho += v
        grad += (-2 * a * v)[:, None] * d
        hess += (4 * a * a * v)[:, None, None] * d[:, :, None] * d[:, None, :]
        hess -= (2 * a * v)[:, None, None] * np.eye(3)[None]
    return rho, grad, hess


@limit_blas_threads
def point_properties(pts, wfx, basis):
    """Every P1 property at each point, as arrays keyed like the CPprop.txt
    labels' meaning (see module docstring)."""
    coords = wfx["coords"]
    D = orbital_derivatives(pts, coords, basis)
    phi, dphi = D[0], D[1:4]
    occ, occ_a, occ_b = basis["occ"], basis["occ_a"], basis["occ_b"]
    # second-derivative index pairs: xx yy zz xy xz yz
    pairs = ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))

    rho_mo = (phi**2) @ occ
    spin = (phi**2) @ (occ_a - occ_b)
    grad_mo = 2 * np.einsum("kpi,pi,i->pk", dphi, phi, occ)
    hess_mo = np.zeros((len(pts), 3, 3))
    for q, (i, j) in enumerate(pairs):
        v = 2 * ((dphi[i] * dphi[j] + phi * D[4 + q]) @ occ)
        hess_mo[:, i, j] = v
        hess_mo[:, j, i] = v
    rho, grad, hess = rho_mo, grad_mo, hess_mo
    if len(wfx["edf_center"]):
        r_e, g_e, h_e = _edf_derivatives(pts, coords, wfx)
        rho, grad, hess = rho_mo + r_e, grad_mo + g_e, hess_mo + h_e

    g_xyz = 0.5 * np.stack([(dphi[k] ** 2) @ occ for k in range(3)], axis=1)
    G = g_xyz.sum(axis=1)
    K = -0.5 * ((phi * (D[4] + D[5] + D[6])) @ occ)

    # ELF / LOL (MOs only)
    tau = G
    if wavefunction_type(wfx) in (0, 3):
        with np.errstate(divide="ignore", invalid="ignore"):
            pauli = tau - np.where(rho_mo != 0, (grad_mo**2).sum(axis=1) / rho_mo / 8, 0.0)
        dh = FC * rho_mo ** (5 / 3)
    else:
        rho_a = (phi**2) @ occ_a
        rho_b = (phi**2) @ occ_b
        ga = 2 * np.einsum("kpi,pi,i->pk", dphi, phi, occ_a)
        gb = 2 * np.einsum("kpi,pi,i->pk", dphi, phi, occ_b)
        with np.errstate(divide="ignore", invalid="ignore"):
            pauli = (tau - np.where(rho_a != 0, (ga**2).sum(axis=1) / rho_a / 8, 0.0)
                     - np.where(rho_b != 0, (gb**2).sum(axis=1) / rho_b / 8, 0.0))
        dh = FC_POL * (rho_a ** (5 / 3) + rho_b ** (5 / 3))
    with np.errstate(divide="ignore", invalid="ignore"):
        elf = 1 / (1 + ((pauli + ELF_ADDMINIMAL) / dh) ** 2)
        t = np.where(tau != 0, dh / tau, 0.0)
        lol = 1 / (1 / t + 1)

    with np.errstate(divide="ignore", invalid="ignore"):
        alie = np.where(rho_mo != 0, (phi**2) @ (basis["abs_energy"] * occ) / rho_mo, 0.0)

    eig = np.linalg.eigvalsh(hess)  # ascending, as Multiwfn sorts them before ellipticity/eta
    return {
        "rho": rho, "rho_alpha": (rho + spin) / 2, "rho_beta": (rho - spin) / 2, "spin": spin,
        "G": G, "G_xyz": g_xyz, "K": K, "V": -K - G, "H": -K,
        "laplacian": hess[:, 0, 0] + hess[:, 1, 1] + hess[:, 2, 2],
        "elf": elf, "lol": lol, "alie": alie,
        "gradient": grad, "hessian": hess, "eigenvalues": eig,
        "det_hessian": np.linalg.det(hess),
        "ellipticity": eig[:, 0] / eig[:, 1] - 1, "eta": np.abs(eig[:, 0]) / eig[:, 2],
    }
