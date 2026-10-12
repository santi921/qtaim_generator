"""QTAIM engine (docs/engine_roadmap/qtaim_engine_plan.md).

Phase P1: the per-point properties Multiwfn's showptprop prints for each
critical point (MW/sub.f90:2347-2510), evaluated from primitive values and
derivatives (density_derivatives) plus the EDF core density:

  rho (MOs + EDF), alpha/beta = (rho +- spin)/2, spin (MO types only)
  G = 1/2 sum n_i |grad phi_i|^2 and its x/y/z parts (Lagkin, MOs only)
  K = -1/2 sum n_i phi_i lap phi_i (Hamkin, MOs only); V = -K - G; H = -K
  Laplacian, gradient and Hessian of rho (MOs + EDF)
  ELF, LOL (Becke forms, MOs only; restricted or spin-polarized by Multiwfn's
  wfntype rule; ELF_addminimal=1), ALIE
  Hessian eigenvalues, determinant, ellipticity, eta

Phase P2: delta_g_hirsh (IGMH, delta_g_Hirsh, MW/function.f90:4043) and, with
esp=True, esp_nuc / esp_e / esp_total (core/esp.py). delta_g_promolecular is
dropped (docs/engine_roadmap/README.md, decision 7). CP search follows in P3.
"""

from concurrent.futures import ThreadPoolExecutor

import numba
import numpy as np
from numba import njit

from qtaim_gen.source.core import esp as esp_engine
from qtaim_gen.source.core.charge_engine import (
    EXP_CUTOFF,
    _morton_order,
    _primitive_block_derivs,
    limit_blas_threads,
    mwfn_proatom_tables,
)

DERIV_BLOCK = 32  # points per block; a function of the points only, so output does not depend on threads

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


@njit(nogil=True, cache=True)
def _edf_kernel(pts, coords, e_center, e_alpha, e_coef, rho, grad, hess):
    """Adds the EDF core density (s Gaussians, exp cutoff 40) with its gradient
    and Hessian (xx yy zz xy xz yz) into rho, grad, hess."""
    for ip in range(pts.shape[0]):
        for k in range(e_center.shape[0]):
            c = e_center[k]
            dx = pts[ip, 0] - coords[c, 0]
            dy = pts[ip, 1] - coords[c, 1]
            dz = pts[ip, 2] - coords[c, 2]
            a = e_alpha[k]
            ar2 = a * (dx * dx + dy * dy + dz * dz)
            if ar2 > EXP_CUTOFF:
                continue
            v = e_coef[k] * np.exp(-ar2)
            t = -2.0 * a * v
            u = 4.0 * a * a * v
            rho[ip] += v
            grad[ip, 0] += t * dx
            grad[ip, 1] += t * dy
            grad[ip, 2] += t * dz
            hess[ip, 0] += u * dx * dx + t
            hess[ip, 1] += u * dy * dy + t
            hess[ip, 2] += u * dz * dz + t
            hess[ip, 3] += u * dx * dy
            hess[ip, 4] += u * dx * dz
            hess[ip, 5] += u * dy * dz


def _edf_derivatives(pts, coords, wfx):
    """EDF core density (s Gaussians) with gradient and Hessian (n, 3, 3)."""
    pts = np.ascontiguousarray(pts, dtype=float)
    n = len(pts)
    rho, grad, h6 = np.zeros(n), np.zeros((n, 3)), np.zeros((n, 6))
    _edf_kernel(pts, coords, wfx["edf_center"], wfx["edf_exp"], wfx["edf_coef"], rho, grad, h6)
    return rho, grad, _hess_from6(h6)


def _hess_from6(h6):
    i, j = _HESS_PAIRS[:, 0], _HESS_PAIRS[:, 1]
    hess = np.empty((len(h6), 3, 3))
    hess[:, i, j] = h6
    hess[:, j, i] = h6
    return hess


_HESS_PAIRS = np.array([(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)])


@limit_blas_threads
def density_derivatives(pts, wfx, basis, block=DERIV_BLOCK, hessian=True):
    """rho, its gradient and Hessian (MOs + EDF) at every point, and the MO-only
    parts point_properties needs. hessian=False returns only rho and its gradient
    (grad rho = 2 w . grad chi), which needs MO values but no MO gradients (rho_mo, grad_mo, spin, alpha/beta
    densities and gradients, G components, K, the ALIE numerator).

    Per block of spatially compact points: primitive values and derivatives for
    the groups whose exp-cutoff sphere reaches the block; MO values and gradients
    from one product with the coefficients; then the back-projection
    w = (n phi) C^T, so sum_i n_i phi_i d2 phi_i = w . d2 chi without forming MO
    second derivatives. Each block is reduced to per-point scalars in its worker."""
    pts = np.ascontiguousarray(pts, dtype=float)
    n = len(pts)
    coords = wfx["coords"]
    ct, occ = basis["ct"], basis["occ"]
    dspin = basis["occ_a"] - basis["occ_b"]
    open_shell = wavefunction_type(wfx) not in (0, 3)
    out = {"rho": np.zeros(n), "gradient": np.zeros((n, 3)), "h6": np.zeros((n, 6))}
    extras = () if not hessian else (
        ("rho_mo", ()), ("spin", ()), ("K", ()), ("alie_num", ()), ("grad_mo", (3,)),
        ("G_xyz", (3,)), ("rho_a", ()), ("rho_b", ()), ("grad_a", (3,)), ("grad_b", (3,)))
    for key, shape in extras:
        out[key] = np.zeros((n,) + shape)
    if not n:
        out.pop("h6")
        if hessian:
            out["hessian"] = np.zeros((0, 3, 3))
        return out
    order = _morton_order(pts)
    sorted_pts = np.ascontiguousarray(pts[order])
    gs, ge, ga, gc = basis["g_start"], basis["g_end"], basis["g_alpha"], basis["g_center"]
    glen = ge - gs
    alie_w = None if basis["abs_energy"] is None else basis["abs_energy"] * occ

    def one_block(s0):
        blk = sorted_pts[s0 : s0 + block]
        idx = order[s0 : s0 + block]
        nb = len(blk)
        lo, hi = blk.min(axis=0), blk.max(axis=0)
        d2 = (np.maximum(0.0, np.maximum(lo - coords, coords - hi)) ** 2).sum(axis=1)
        groups = np.flatnonzero(ga * d2[gc] <= EXP_CUTOFF)
        nc = glen[groups]
        cols = np.arange(nc.sum()) + np.repeat(gs[groups] - (np.cumsum(nc) - nc), nc)
        prim = _primitive_block_derivs(blk, coords, groups, gc, ga, gs, ge, basis["lmn"], len(cols))
        C = ct[cols]
        if not hessian:
            phi = prim[0] @ C
            w = (phi * occ) @ C.T
            out["rho"][idx] = (phi * phi) @ occ
            out["gradient"][idx] = 2.0 * np.einsum("pm,kpm->pk", w, prim[1:4])
            return
        m4 = (prim[:4].reshape(4 * nb, len(cols)) @ C).reshape(4, nb, C.shape[1])
        phi, dphi = m4[0], m4[1:]
        nphi = phi * occ
        w = nphi @ C.T
        rho_mo = (phi * nphi).sum(axis=1)
        grad_mo = 2.0 * np.einsum("kpi,pi->pk", dphi, nphi)
        h6 = np.empty((nb, 6))
        for q, (i, j) in enumerate(_HESS_PAIRS):
            h6[:, q] = 2.0 * ((dphi[i] * dphi[j]) @ occ + (w * prim[4 + q]).sum(axis=1))
        out["rho"][idx] = rho_mo
        out["gradient"][idx] = grad_mo
        out["h6"][idx] = h6
        phi2 = phi * phi
        out["rho_mo"][idx] = rho_mo
        out["grad_mo"][idx] = grad_mo
        out["spin"][idx] = phi2 @ dspin
        out["G_xyz"][idx] = 0.5 * np.einsum("kpi,i->pk", dphi * dphi, occ)
        out["K"][idx] = -0.5 * (w * (prim[4] + prim[5] + prim[6])).sum(axis=1)
        if alie_w is not None:
            out["alie_num"][idx] = phi2 @ alie_w
        if open_shell:
            out["rho_a"][idx] = phi2 @ basis["occ_a"]
            out["rho_b"][idx] = phi2 @ basis["occ_b"]
            out["grad_a"][idx] = 2.0 * np.einsum("kpi,pi,i->pk", dphi, phi, basis["occ_a"])
            out["grad_b"][idx] = 2.0 * np.einsum("kpi,pi,i->pk", dphi, phi, basis["occ_b"])

    starts = range(0, n, block)
    nthreads = numba.get_num_threads()
    if nthreads == 1 or len(starts) == 1:
        for s0 in starts:
            one_block(s0)
    else:
        with ThreadPoolExecutor(nthreads) as pool:
            list(pool.map(one_block, starts))
    if len(wfx["edf_center"]):
        _edf_kernel(pts, coords, wfx["edf_center"], wfx["edf_exp"], wfx["edf_coef"],
                    out["rho"], out["gradient"], out["h6"])
    h6 = out.pop("h6")
    if hessian:
        out["hessian"] = _hess_from6(h6)
    return out


@njit(nogil=True, cache=True)
def _lagintpol_d1(radpos, vals, npt, r):
    """Multiwfn's lagintpol (MW/util.f90:2682) with itype 2: value and first
    derivative of the 4-point Lagrange interpolant; linear inside the first
    point, zero from the last point on."""
    if r <= radpos[0]:
        d1 = (vals[1] - vals[0]) / (radpos[1] - radpos[0])
        return vals[0] - (radpos[0] - r) * d1, d1
    if r >= radpos[npt - 1]:
        return 0.0, 0.0
    i = np.searchsorted(radpos[:npt], r, side="right") + 1  # 1-based first point beyond r
    istart = i - 2
    iend = i + 1
    if istart < 1:
        istart += 1
        iend += 1
    elif iend > npt:
        istart -= 1
        iend -= 1
    val = 0.0
    der1 = 0.0
    for m in range(istart - 1, iend):
        poly = 1.0
        for j in range(istart - 1, iend):
            if j != m:
                poly *= (r - radpos[j]) / (radpos[m] - radpos[j])
        val += vals[m] * poly
        suml = 0.0
        for l in range(istart - 1, iend):
            if l == m:
                continue
            poly = 1.0
            for j in range(istart - 1, iend):
                if j != m and j != l:
                    poly *= (r - radpos[j]) / (radpos[m] - radpos[j])
            suml += poly / (radpos[m] - radpos[l])
        der1 += vals[m] * suml
    return val, der1


@njit(nogil=True, cache=True)
def _delta_g_hirsh(pts, rho, grad, atcoords, atnums, radpos, table, npts_of, cut_of):
    """IGMH delta-g (IGMgrad_Hirsh over all atoms, MW/function.f90:4061-4134):
    free-atom densities and gradients by proatmgrad (built-in radial densities,
    zero beyond atmrhocut), g_A = w_A grad rho - (rho/rho0) grad rho_A0
    + rho rho_A0 / rho0^2 grad rho0, delta-g = sum_A |g_A| - |sum_A g_A|."""
    n = pts.shape[0]
    nat = atcoords.shape[0]
    out = np.empty(n)
    ra = np.zeros(nat)
    ga = np.zeros((nat, 3))
    for ip in range(n):
        pro = 0.0
        pg0 = pg1 = pg2 = 0.0
        for a in range(nat):
            z = atnums[a]
            dx = pts[ip, 0] - atcoords[a, 0]
            dy = pts[ip, 1] - atcoords[a, 1]
            dz = pts[ip, 2] - atcoords[a, 2]
            r2 = dx * dx + dy * dy + dz * dz
            ra[a] = 0.0
            ga[a, 0] = ga[a, 1] = ga[a, 2] = 0.0
            if r2 < cut_of[z] * cut_of[z]:
                r = np.sqrt(r2)
                v, d1 = _lagintpol_d1(radpos, table[z], npts_of[z], r)
                ra[a] = v
                if r != 0.0:
                    ga[a, 0] = d1 / r * dx
                    ga[a, 1] = d1 / r * dy
                    ga[a, 2] = d1 / r * dz
            pro += ra[a]
            pg0 += ga[a, 0]
            pg1 += ga[a, 1]
            pg2 += ga[a, 2]
        igm = 0.0
        s0 = s1 = s2 = 0.0
        for a in range(nat):
            if pro == 0.0:
                g0 = g1 = g2 = 0.0
            else:
                w = ra[a] / pro
                f = rho[ip] / pro
                h = rho[ip] * ra[a] / (pro * pro)
                g0 = w * grad[ip, 0] - f * ga[a, 0] + h * pg0
                g1 = w * grad[ip, 1] - f * ga[a, 1] + h * pg1
                g2 = w * grad[ip, 2] - f * ga[a, 2] + h * pg2
            igm += np.sqrt(g0 * g0 + g1 * g1 + g2 * g2)
            s0 += g0
            s1 += g1
            s2 += g2
        out[ip] = igm - np.sqrt(s0 * s0 + s1 * s1 + s2 * s2)
    return out


def delta_g_hirsh(pts, rho, grad, wfx):
    """delta_g_Hirsh at each point from rho and its gradient (MOs + EDF)."""
    tables, missing = mwfn_proatom_tables(wfx["atnums"])
    if missing:
        raise ValueError(f"no built-in radial density for Z = {missing}")
    atnums, radpos, table, npts_of, cut_of = tables
    return _delta_g_hirsh(np.ascontiguousarray(pts, dtype=float), rho, np.ascontiguousarray(grad),
                          wfx["coords"], atnums, radpos, table, npts_of, cut_of)


def ecp_atoms_without_edf(wfx):
    """Indices of ECP atoms (zeff < Z) with no EDF primitive: their core density
    is missing, so rho, its Laplacian and the nuclear CPs there are wrong (a
    uranium ECP job without EDF lost a nuclear CP and had every bond unpaired)."""
    ecp = np.flatnonzero(wfx["zeff"] != wfx["atnums"])
    return [int(i) for i in ecp if not np.any(wfx["edf_center"] == i)]


@limit_blas_threads
def point_properties(pts, wfx, basis, allow_missing_edf=False, esp=False, esp_pairs=None, boys_mode="exact"):
    """Every P1/P2 property at each point, as arrays keyed like the CPprop.txt
    labels' meaning (see module docstring). Raises when an ECP atom has no EDF
    core density, unless allow_missing_edf (an MO-only comparison on purpose).
    esp=True adds esp_nuc, esp_e, esp_total; pass esp_pairs (esp.esp_pairs) to
    reuse the point-independent setup across calls."""
    missing = ecp_atoms_without_edf(wfx)
    if missing and not allow_missing_edf:
        raise ValueError(f"ECP atoms without EDF core density (0-based): {missing[:10]}")
    d = density_derivatives(pts, wfx, basis)
    rho, grad, hess = d["rho"], d["gradient"], d["hessian"]
    rho_mo, grad_mo, spin = d["rho_mo"], d["grad_mo"], d["spin"]
    g_xyz = d["G_xyz"]
    G = g_xyz.sum(axis=1)
    K = d["K"]

    # ELF / LOL (MOs only)
    tau = G
    if wavefunction_type(wfx) in (0, 3):
        with np.errstate(divide="ignore", invalid="ignore"):
            pauli = tau - np.where(rho_mo != 0, (grad_mo**2).sum(axis=1) / rho_mo / 8, 0.0)
        dh = FC * rho_mo ** (5 / 3)
    else:
        rho_a, rho_b, ga, gb = d["rho_a"], d["rho_b"], d["grad_a"], d["grad_b"]
        with np.errstate(divide="ignore", invalid="ignore"):
            pauli = (tau - np.where(rho_a != 0, (ga**2).sum(axis=1) / rho_a / 8, 0.0)
                     - np.where(rho_b != 0, (gb**2).sum(axis=1) / rho_b / 8, 0.0))
        dh = FC_POL * (rho_a ** (5 / 3) + rho_b ** (5 / 3))
    with np.errstate(divide="ignore", invalid="ignore"):
        elf = 1 / (1 + ((pauli + ELF_ADDMINIMAL) / dh) ** 2)
        t = np.where(tau != 0, dh / tau, 0.0)
        lol = 1 / (1 / t + 1)

    with np.errstate(divide="ignore", invalid="ignore"):
        alie = np.where(rho_mo != 0, d["alie_num"] / rho_mo, 0.0)

    eig = np.linalg.eigvalsh(hess)  # ascending, as Multiwfn sorts them before ellipticity/eta
    extra = {"delta_g_hirsh": delta_g_hirsh(pts, rho, grad, wfx)}
    if esp:
        nuc, ele, tot = esp_engine.electrostatic_potential(pts, wfx, basis, esp_pairs, boys_mode=boys_mode)
        extra.update(esp_nuc=nuc, esp_e=ele, esp_total=tot)
    return extra | {
        "rho": rho, "rho_alpha": (rho + spin) / 2, "rho_beta": (rho - spin) / 2, "spin": spin,
        "G": G, "G_xyz": g_xyz, "K": K, "V": -K - G, "H": -K,
        "laplacian": hess[:, 0, 0] + hess[:, 1, 1] + hess[:, 2, 2],
        "elf": elf, "lol": lol, "alie": alie,
        "gradient": grad, "hessian": hess, "eigenvalues": eig,
        "det_hessian": np.linalg.det(hess),
        "ellipticity": eig[:, 0] / eig[:, 1] - 1, "eta": np.abs(eig[:, 0]) / eig[:, 2],
    }
