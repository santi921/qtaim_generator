"""One-pass charge and fuzzy-space engine following Multiwfn 3.8 conventions.

Needs only numpy and numba; the data tables are generated modules in
qtaim_gen/source/data (multiwfn_tables, multiwfn_atmraddens, lebedev_tables).
Reads a Multiwfn-written .wfx (EDF block included), evaluates the density once
on Multiwfn's per-atom grids, and derives every full_set=0 charge, fuzzy and
fuzzy-bond quantity from that pass (plus one coarser grid for bond orders):

    hirshfeld, adch, cm5, becke (ADC-corrected, as shipped)  charge.json schema
    becke/hirsh_fuzzy_density, becke/hirsh_fuzzy_spin         fuzzy schema
    fuzzy_bond                                                 bond schema

With full_set >= 1 it adds vdd (same grid pass), mbis (its own 30-60 x 302 grid,
as MBIS_wrapper) and mbis_fuzzy_density/spin (MBIS refit on the main grid).

    charge-engine --wfx orca.wfx --out engine.json [--full_set 1]

The Multiwfn routines reproduced here, and the source lines they come from, are
listed in docs/plans/2026-10-04-feat-one-pass-charge-engine-plan.md.
"""

import argparse
import functools
import json
import os
import re
import sys
import tempfile
import time

import numba
import numpy as np
from numba import njit, prange

from qtaim_gen.source.data import multiwfn_tables as T

B2A = 0.529177210903  # Multiwfn's Bohr -> Angstrom (2018 CODATA)
EXP_CUTOFF = 40.0  # Multiwfn expcutoff=-40: skip exp(-x) for x > 40
BECKE_TOL = 1e-14  # drop P_j once its partial product is this far below the max


# ---------------------------------------------------------------- wfx reading


def _section(text, tag):
    i = text.find(f"<{tag}>")
    if i < 0:
        return None
    j = text.index(f"</{tag}>", i)
    return text[i + len(tag) + 2 : j]


def read_wfx(path):
    with open(path) as f:
        text = f.read()

    def nums(tag, dtype=float):
        return np.array(_section(text, tag).split(), dtype=dtype)

    natoms = int(_section(text, "Number of Nuclei"))
    coeff_text = re.sub(
        r"<MO Number>\s*\d+\s*</MO Number>",
        " ",
        _section(text, "Molecular Orbital Primitive Coefficients"),
    )
    occ = nums("Molecular Orbital Occupation Numbers")
    exps = nums("Primitive Exponents")
    coeffs = np.array(coeff_text.split(), dtype=float).reshape(len(occ), len(exps))
    spin_types = [
        s.strip() for s in _section(text, "Molecular Orbital Spin Types").strip().splitlines()
    ]

    wfx = {
        "atnums": nums("Atomic Numbers", int),
        "zeff": nums("Nuclear Charges"),
        "coords": nums("Nuclear Cartesian Coordinates").reshape(natoms, 3),
        "prim_center": nums("Primitive Centers", int) - 1,
        "prim_type": nums("Primitive Types", int),
        "prim_exp": exps,
        "occ": occ,
        "spin_types": spin_types,
        "coeffs": coeffs,
        "mult": int(_section(text, "Electronic Spin Multiplicity")),
        "mo_energy": nums("Molecular Orbital Energies") if _section(text, "Molecular Orbital Energies") else None,
        "edf_center": np.zeros(0, dtype=np.int64),
        "edf_exp": np.zeros(0),
        "edf_coef": np.zeros(0),
    }
    if _section(text, "Number of EDF Primitives") is not None:
        wfx["edf_center"] = nums("EDF Primitive Centers", int) - 1
        wfx["edf_exp"] = nums("EDF Primitive Exponents")
        wfx["edf_coef"] = nums("EDF Primitive Coefficients")
        if (nums("EDF Primitive Types", int) > 1).any():
            raise ValueError("EDF primitives must be s type")
    return wfx


def prepare_basis(wfx):
    """Occupied MOs, alpha/beta occupations, and primitives grouped by
    (center, exponent) so each exp() is evaluated once per group."""
    keep = wfx["occ"] != 0
    occ = wfx["occ"][keep]
    types = [t for t, k in zip(wfx["spin_types"], keep) if k]
    occ_a = np.array([o / 2 if t == "Alpha and Beta" else (o if t == "Alpha" else 0.0) for o, t in zip(occ, types)])
    occ_b = np.array([o / 2 if t == "Alpha and Beta" else (o if t == "Beta" else 0.0) for o, t in zip(occ, types)])

    mtype = np.array([T.WFX_G_TO_MWFN.get(int(t), int(t)) for t in wfx["prim_type"]])
    if mtype.max() > 56:
        raise ValueError(f"primitive type {mtype.max()} beyond h functions")
    order = np.lexsort((wfx["prim_exp"], wfx["prim_center"]))
    center = wfx["prim_center"][order]
    alpha = wfx["prim_exp"][order]
    lmn = np.array(
        [(T.TYPE2IX[t - 1], T.TYPE2IY[t - 1], T.TYPE2IZ[t - 1]) for t in mtype[order]],
        dtype=np.int64,
    ).reshape(-1, 3)
    ct = np.ascontiguousarray(wfx["coeffs"][keep][:, order].T)

    new_group = np.ones(len(order), dtype=bool)
    new_group[1:] = (center[1:] != center[:-1]) | (alpha[1:] != alpha[:-1])
    g_start = np.flatnonzero(new_group)
    g_end = np.append(g_start[1:], len(order))
    g_center = center[g_start]
    g_alpha = alpha[g_start]
    energy = wfx.get("mo_energy")
    return {
        "occ": occ, "abs_energy": None if energy is None else np.abs(energy[keep]),
        "occ_a": occ_a, "occ_b": occ_b, "ct": ct, "lmn": lmn,
        "g_alpha": g_alpha, "g_start": g_start, "g_end": g_end, "g_center": g_center,
        "nelec": float(occ.sum()),
    }


# ---------------------------------------------------------------- grids


def multiwfn_atom_grid(nrad=75, nang=434, radcut=10.0):
    """Origin-centred atom grid of gen1cintgrid (sub.f90): second-kind
    Gauss-Chebyshev radial points under the Becke map with scale 1, Lebedev
    angular points, points at r >= radcut Bohr dropped."""
    from qtaim_gen.source.data.lebedev_tables import LEBEDEV

    if nang not in LEBEDEV:
        raise ValueError(f"no Lebedev table with {nang} points; have {sorted(LEBEDEV)}")
    ang = np.array(LEBEDEV[nang]).reshape(nang, 4)
    i = np.arange(1, nrad + 1)
    x = np.cos(i * np.pi / (nrad + 1))
    r = (1 + x) / (1 - x)
    w = 2 * np.pi / (nrad + 1) * (1 + x) ** 2.5 / (1 - x) ** 3.5 * 4 * np.pi
    keep = r < radcut
    pts = (r[keep, None, None] * ang[None, :, :3]).reshape(-1, 3)
    wts = (w[keep, None] * ang[None, :, 3]).reshape(-1)
    return pts, wts


# ---------------------------------------------------------------- kernels


@njit(parallel=True, cache=True)
def _primitive_block(pts, atcoords, groups, g_center, g_alpha, g_start, g_end, lmn, ncols):
    """Values of the primitives of `groups` (one column each, in group order) at
    every point (orbderv's GTF part), zero where Multiwfn's exp cutoff drops
    them. Callers pass only the groups whose cutoff sphere reaches the block."""
    n = pts.shape[0]
    out = np.zeros((n, ncols))
    for ip in prange(n):
        x, y, z = pts[ip, 0], pts[ip, 1], pts[ip, 2]
        col = 0
        last = -1
        dx = dy = dz = r2 = 0.0
        for gi in range(groups.shape[0]):
            g = groups[gi]
            a = g_center[g]
            if a != last:
                dx = x - atcoords[a, 0]
                dy = y - atcoords[a, 1]
                dz = z - atcoords[a, 2]
                r2 = dx * dx + dy * dy + dz * dz
                last = a
            ar2 = g_alpha[g] * r2
            if ar2 > EXP_CUTOFF:
                col += g_end[g] - g_start[g]
                continue
            e = np.exp(-ar2)
            for p in range(g_start[g], g_end[g]):
                v = e
                for _ in range(lmn[p, 0]):
                    v *= dx
                for _ in range(lmn[p, 1]):
                    v *= dy
                for _ in range(lmn[p, 2]):
                    v *= dz
                out[ip, col] = v
                col += 1
    return out


@njit(parallel=True, cache=True)
def _edf(pts, atcoords, e_center, e_alpha, e_coef):
    """EDF core density (EDFrho): s-type Gaussians on ECP atoms."""
    npts = pts.shape[0]
    out = np.zeros(npts)
    for ip in prange(npts):
        v = 0.0
        for k in range(e_center.shape[0]):
            c = e_center[k]
            dx = pts[ip, 0] - atcoords[c, 0]
            dy = pts[ip, 1] - atcoords[c, 1]
            dz = pts[ip, 2] - atcoords[c, 2]
            ar2 = e_alpha[k] * (dx * dx + dy * dy + dz * dz)
            if ar2 <= 40.0:
                v += e_coef[k] * np.exp(-ar2)
        out[ip] = v
    return out


def _morton_order(pts, cell=1.0):
    """Points sorted along a Z-order curve on a `cell`-Bohr lattice, so each
    block of consecutive points is spatially compact."""
    q = np.floor((pts - pts.min(axis=0)) / cell).astype(np.int64)
    key = np.zeros(len(pts), dtype=np.int64)
    for bit in range(20):
        for k in range(3):
            key |= ((q[:, k] >> bit) & 1) << (3 * bit + k)
    return np.argsort(key, kind="stable")


def orbital_values(pts, coords, basis, block=512):
    """Occupied MO values at every point. Points are taken in spatially compact
    blocks; per block only primitive groups whose exp-cutoff sphere reaches the
    block's bounding box are evaluated, and the MOs follow from one dense
    matrix product with their coefficients (BLAS)."""
    pts = np.asarray(pts, dtype=float)
    ct = basis["ct"]
    out = np.empty((len(pts), ct.shape[1]))
    if not len(pts):
        return out
    order = _morton_order(pts)
    sorted_pts = np.ascontiguousarray(pts[order])
    gs, ge, ga, gc = basis["g_start"], basis["g_end"], basis["g_alpha"], basis["g_center"]
    glen = ge - gs
    for s0 in range(0, len(pts), block):
        blk = sorted_pts[s0 : s0 + block]
        lo, hi = blk.min(axis=0), blk.max(axis=0)
        d2 = (np.maximum(0.0, np.maximum(lo - coords, coords - hi)) ** 2).sum(axis=1)
        groups = np.flatnonzero(ga * d2[gc] <= EXP_CUTOFF)
        n = glen[groups]
        cols = np.arange(n.sum()) + np.repeat(gs[groups] - (np.cumsum(n) - n), n)
        prim = _primitive_block(blk, coords, groups, gc, ga, gs, ge, basis["lmn"], len(cols))
        out[order[s0 : s0 + block]] = prim @ ct[cols]
    return out


def limit_blas_threads(fn):
    """Run fn with BLAS (numpy matmul) threads capped at numba's thread count,
    so the two thread pools do not oversubscribe the cores."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            from threadpoolctl import threadpool_limits
        except ImportError:
            return fn(*args, **kwargs)
        with threadpool_limits(numba.get_num_threads()):
            return fn(*args, **kwargs)

    return wrapper


def density(pts, coords, basis, wfx):
    """rho (MOs + EDF core) and spin density (fdens / fspindens)."""
    phi2 = orbital_values(pts, coords, basis) ** 2
    rho_a = phi2 @ basis["occ_a"]
    rho_b = phi2 @ basis["occ_b"]
    rho = rho_a + rho_b
    if len(wfx["edf_center"]):
        rho += _edf(pts, coords, wfx["edf_center"], wfx["edf_exp"], wfx["edf_coef"])
    return rho, rho_a - rho_b


@njit(parallel=True, cache=True)
def _becke(pts, atcoords, rinv, aij, tol):
    """Normalized Becke weights of every atom at every point (BeckePvec).

    Atoms are visited nearest first. The pair factors s are <= 1, so a partial
    product already below tol * (largest P so far) bounds the full product;
    that atom's P is set to zero without looping over the remaining atoms."""
    npts = pts.shape[0]
    nat = atcoords.shape[0]
    w = np.zeros((npts, nat))
    for ip in prange(npts):
        r = np.empty(nat)
        for j in range(nat):
            dx = pts[ip, 0] - atcoords[j, 0]
            dy = pts[ip, 1] - atcoords[j, 1]
            dz = pts[ip, 2] - atcoords[j, 2]
            r[j] = np.sqrt(dx * dx + dy * dy + dz * dz)
        order = np.argsort(r)
        pmax = 0.0
        ptot = 0.0
        for jj in range(nat):
            j = order[jj]
            pj = 1.0
            for kk in range(nat):
                k = order[kk]
                if k == j:
                    continue
                mu = (r[j] - r[k]) * rinv[j, k]
                a = aij[j, k]
                if a != 0.0:
                    mu = mu + a * (1.0 - mu * mu)
                t = 1.5 * mu - 0.5 * mu * mu * mu
                t = 1.5 * t - 0.5 * t * t * t
                t = 1.5 * t - 0.5 * t * t * t
                pj *= 0.5 * (1.0 - t)
                if pj < tol * pmax:
                    pj = 0.0
                    break
            w[ip, j] = pj
            ptot += pj
            if pj > pmax:
                pmax = pj
        for j in range(nat):
            w[ip, j] /= ptot
    return w


@njit(parallel=True, cache=True)
def _hirshfeld_mwfn(pts, atcoords, atnums, radpos, table, npts_of, cut_of):
    """Hirshfeld weights from Multiwfn's built-in radial densities, evaluated
    exactly as eleraddens/lagintpol do: zero beyond atmrhocut or the last grid
    point, linear extrapolation inside the first point, otherwise 4-point
    Lagrange interpolation on the stencil lagintpol picks. Also returns the
    unnormalized promolecular density (VDD needs rho - promol).

    atnums only indexes the tables, so per-atom tables (MBIS) work with
    atnums = arange(nat) and cut_of = inf (fdens_rad has no atmrhocut test)."""
    npts = pts.shape[0]
    nat = atcoords.shape[0]
    h = np.zeros((npts, nat))
    empty = np.zeros(npts, dtype=np.bool_)
    promol = np.zeros(npts)
    for ip in prange(npts):
        tot = 0.0
        for a in range(nat):
            z = atnums[a]
            dx = pts[ip, 0] - atcoords[a, 0]
            dy = pts[ip, 1] - atcoords[a, 1]
            dz = pts[ip, 2] - atcoords[a, 2]
            r = np.sqrt(dx * dx + dy * dy + dz * dz)
            npt = npts_of[z]
            if r > cut_of[z] or r >= radpos[npt - 1]:
                continue
            if r <= radpos[0]:
                d1 = (table[z, 1] - table[z, 0]) / (radpos[1] - radpos[0])
                v = table[z, 0] - (radpos[0] - r) * d1
            else:
                # i (1-based) = first point beyond r; stencil i-2..i+1
                i = np.searchsorted(radpos[:npt], r, side="right") + 1
                istart = i - 2
                iend = i + 1
                if istart < 1:
                    istart += 1
                    iend += 1
                elif iend > npt:
                    istart -= 1
                    iend -= 1
                v = 0.0
                for m in range(istart - 1, iend):
                    poly = 1.0
                    for j in range(istart - 1, iend):
                        if j != m:
                            poly *= (r - radpos[j]) / (radpos[m] - radpos[j])
                    v += table[z, m] * poly
            h[ip, a] = v
            tot += v
        promol[ip] = tot
        if tot != 0.0:
            for a in range(nat):
                h[ip, a] /= tot
        else:
            empty[ip] = True
    return h, empty, promol


@njit(parallel=True, cache=True)
def _voronoi_mask(pts, atcoords, center):
    """VDD weight (spacecharge, chgtype 2): 1 unless another atom is strictly
    closer to the point than the centre atom (ties stay with the centre)."""
    npts = pts.shape[0]
    nat = atcoords.shape[0]
    mask = np.ones(npts)
    for ip in prange(npts):
        dx = pts[ip, 0] - atcoords[center, 0]
        dy = pts[ip, 1] - atcoords[center, 1]
        dz = pts[ip, 2] - atcoords[center, 2]
        dc2 = dx * dx + dy * dy + dz * dz
        for j in range(nat):
            if j == center:
                continue
            dx = pts[ip, 0] - atcoords[j, 0]
            dy = pts[ip, 1] - atcoords[j, 1]
            dz = pts[ip, 2] - atcoords[j, 2]
            if dx * dx + dy * dy + dz * dz < dc2:
                mask[ip] = 0.0
                break
    return mask


@njit(parallel=True, cache=True)
def _mbis_cycle(grel, atcoords, tmpden, cut2, mshell, pop, sig):
    """One MBIS iteration (MBIS, imode 0): new shell populations and the
    integral part of the new widths (Eqs. 18-19 of the MBIS paper). tmpden[c, i]
    is rho * quadrature weight * Becke weight of centre c at its point i; atoms
    beyond atmrhocut are ignored (ignorefar=1), shell densities below 1e-10 are
    zeroed and points with tmpden <= 1e-14 skipped, as Multiwfn does."""
    nat = atcoords.shape[0]
    m = grel.shape[0]
    npts = nat * m
    nblk = min(npts, 256)
    chunk = (npts + nblk - 1) // nblk
    ppop = np.zeros((nblk, nat, 6))
    psig = np.zeros((nblk, nat, 6))
    for blk in prange(nblk):
        rho0sh = np.zeros((nat, 6))
        dist = np.empty(nat)
        for ip in range(blk * chunk, min(npts, (blk + 1) * chunk)):
            c = ip // m
            i = ip - c * m
            td = tmpden[c, i]
            if not td > 1e-14:
                continue
            x = grel[i, 0] + atcoords[c, 0]
            y = grel[i, 1] + atcoords[c, 1]
            z = grel[i, 2] + atcoords[c, 2]
            rho0 = 0.0
            for j in range(nat):
                dx = x - atcoords[j, 0]
                dy = y - atcoords[j, 1]
                dz = z - atcoords[j, 2]
                d2 = dx * dx + dy * dy + dz * dz
                if d2 > cut2[j]:
                    dist[j] = -1.0
                    continue
                d = np.sqrt(d2)
                dist[j] = d
                for k in range(mshell[j]):
                    sv = sig[j, k]
                    t = pop[j, k] / sv**3 / 8 / np.pi * np.exp(-d / sv)
                    if t < 1e-10:
                        t = 0.0
                    rho0sh[j, k] = t
                    rho0 += t
            if rho0 > 0.0:
                for j in range(nat):
                    if dist[j] < 0.0:
                        continue
                    for k in range(mshell[j]):
                        f = td * rho0sh[j, k] / rho0
                        ppop[blk, j, k] += f
                        psig[blk, j, k] += f * dist[j]
    return ppop.sum(axis=0), psig.sum(axis=0)


# ---------------------------------------------------------------- partitions


def becke_pair_tables(atnums, coords):
    """1/R_ij and the clipped heteronuclear size adjustment a_ij of BeckePvec
    (only for unlike elements closer than 8 Bohr), modified-CSD radii."""
    rad = np.array([T.COVR_TIANLU[z] for z in atnums]) / B2A
    d = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        rinv = np.where(d > 0, 1.0 / d, 0.0)
        chi = rad[:, None] / rad[None, :]
        u = (chi - 1) / (chi + 1)
        a = u / (u * u - 1)
    a = np.clip(a, -0.5, 0.5)
    a[(d >= 8.0) | (atnums[:, None] == atnums[None, :])] = 0.0
    return rinv, a


def mwfn_proatom_tables(atnums):
    """Multiwfn built-in radial densities as dense arrays indexed by Z."""
    from qtaim_gen.source.data import multiwfn_atmraddens as R

    missing = sorted(set(int(z) for z in atnums) - set(R.RHO))
    if missing:
        return None, missing
    radpos = np.array(R.RADPOS)
    table = np.zeros((max(R.RHO) + 1, len(radpos)))
    npts_of = np.zeros(max(R.RHO) + 1, dtype=np.int64)
    for z, vals in R.RHO.items():
        table[z, : len(vals)] = vals
        npts_of[z] = len(vals)
    cut_of = np.array(R.ATMRHOCUT[: max(R.RHO) + 1])
    return (atnums.astype(np.int64), radpos, table, npts_of, cut_of), []


def mbis_grid_size(atnums):
    """MBIS_wrapper's own grid under iautointgrid=1: 302 angular points and
    30 radial points, raised to 40/50/60 when any Z exceeds 18/36/54."""
    zmax = int(atnums.max())
    nrad = 60 if zmax > 54 else 50 if zmax > 36 else 40 if zmax > 18 else 30
    return nrad, 302


def mbis_initial_shells(atnums):
    """MBIS initial guess (icore=1): shell count by period, populations 2/8/8/
    18/18 for the core shells and the rest in the valence shell, widths from
    1/(2Z) for the innermost to 1/2 for the valence shell. Uses the true Z."""
    nat = len(atnums)
    mshell = np.zeros(nat, dtype=np.int64)
    pop = np.zeros((nat, 6))
    sig = np.zeros((nat, 6))
    core = {1: [], 2: [2], 3: [2, 8], 4: [2, 8, 8], 5: [2, 8, 8, 18], 6: [2, 8, 8, 18, 18]}
    for a, z in enumerate(atnums):
        z = int(z)
        m = 1 if z <= 2 else 2 if z <= 10 else 3 if z <= 18 else 4 if z <= 36 else 5 if z <= 54 else 6
        mshell[a] = m
        pop[a, : m - 1] = core[m]
        pop[a, m - 1] = z - sum(core[m])
        sig[a, 0] = 1.0 / (2 * z)
        if m == 3:
            sig[a, 1] = 1.0 / (2 * np.sqrt(float(z)))
        elif m > 3:
            for k in range(1, m - 1):
                sig[a, k] = 1.0 / (2 * z ** (1 - k / (m - 1)))
        if m > 1:
            sig[a, m - 1] = 0.5
    return mshell, pop, sig


def mbis_fit(grel, coords, tmpden, atnums, qbase, crit=1e-4, maxcyc=500):
    """MBIS iterations until no charge moves by crit or maxcyc is reached.

    Returns the last charges (qbase - population; not normalized, as printed
    for imode 0), the shell parameters of the previous cycle (the loop exits
    before replacing them, and those are what the radial tables are built
    from), and the number of cycles."""
    from qtaim_gen.source.data import multiwfn_atmraddens as R

    mshell, pop, sig = mbis_initial_shells(atnums)
    cut2 = np.array([R.ATMRHOCUT[int(z)] ** 2 for z in atnums])
    last = np.zeros(len(atnums))
    for icyc in range(1, maxcyc + 1):
        popnew, signew = _mbis_cycle(grel, coords, tmpden, cut2, mshell, pop, sig)
        pos = popnew > 0
        signew[pos] /= 3 * popnew[pos]
        charge = qbase - popnew.sum(axis=1)
        varmax = np.abs(charge - last).max()
        if varmax < crit or icyc == maxcyc:
            return charge, mshell, pop, sig, icyc
        last = charge
        pop, sig = popnew, signew


def mbis_radial_tables(mshell, pop, sig):
    """Atomic radial densities MBIS(2, .) builds for fuzzy partitioning: the
    shell sum on Multiwfn's radial positions, truncated after the first point
    below 1e-8 (that point kept)."""
    from qtaim_gen.source.data import multiwfn_atmraddens as R

    radpos = np.array(R.RADPOS)
    nat = len(mshell)
    table = np.zeros((nat, len(radpos)))
    npts_of = np.full(nat, len(radpos), dtype=np.int64)
    for a in range(nat):
        for ipt, r in enumerate(radpos):
            v = 0.0
            for k in range(mshell[a]):
                v += pop[a, k] / sig[a, k] ** 3 / 8 / np.pi * np.exp(-r / sig[a, k])
            table[a, ipt] = v
            if v < 1e-8:
                npts_of[a] = ipt + 1
                break
    return np.arange(nat, dtype=np.int64), radpos, table, npts_of, np.full(nat, np.inf)


# ---------------------------------------------------------------- corrections


def adc(charge, atm_dip, coords, atnums):
    """Atomic-dipole-corrected charges (doADC, population.f90)."""
    vdw = np.array([T.VDWR[z] for z in atnums]) / B2A
    out = charge.copy()
    for i in range(len(charge)):
        r = coords - coords[i]
        d = np.linalg.norm(r, axis=1)
        rmax = vdw[i] + vdw
        tr = d / (rmax / 2) - 1
        tr = 1.5 * tr - 0.5 * tr**3
        tr = 1.5 * tr - 0.5 * tr**3
        w = 0.5 * (1 - (1.5 * tr - 0.5 * tr**3))
        w[d > rmax] = 0.0
        wtot = w.sum()
        avgr = (w[:, None] * r).sum(0) / wtot
        avgrr = np.einsum("j,ja,jb->ab", w, r, r) / wtot
        ev, vec = np.linalg.eigh(avgrr - np.outer(avgr, avgr))
        inv = np.where(np.abs(ev) > 1e-5, 1.0 / np.where(ev == 0, 1.0, ev), 0.0)
        rp = r @ vec - avgr @ vec
        out += w / wtot * ((rp * inv) @ (vec.T @ atm_dip[i]))
    return out


def cm5(charge, coords, atnums):
    """CM5 correction of Hirshfeld charges (doCM5 / getCM5Tval)."""
    rad = np.array([(T.COVR[z] + T.COVR_PYY[z]) / 2 if z <= 96 else T.COVR_PYY[z] for z in atnums])
    d_ang = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1) * B2A
    out = charge.copy()
    for i, zi in enumerate(atnums):
        for j, zj in enumerate(atnums):
            if i == j:
                continue
            if (zi, zj) in T.CM5_T_PAIRS:
                tval = T.CM5_T_PAIRS[(zi, zj)]
            elif (zj, zi) in T.CM5_T_PAIRS:
                tval = -T.CM5_T_PAIRS[(zj, zi)]
            else:
                tval = T.CM5_D.get(int(zi), 0.0) - T.CM5_D.get(int(zj), 0.0)
            out[i] += tval * np.exp(-T.CM5_ALPHA * (d_ang[i, j] - rad[i] - rad[j]))
    return out


def fuzzy_bond_orders(basis, coords, rinv, aij, radcut, labels, nrad=45, nang=170, threshold=0.05):
    """Fuzzy bond order = delocalization index in Becke fuzzy space (fuzzyana,
    iwork=1). AOMs are integrated on each atom's own grid, by default 45 x 170
    points (what iautointgrid=1 switches this task to); then for each spin
    DI_AB = 2 sum_ij sqrt(n_i n_j) S^A_ij S^B_ij over that spin's orbitals. The
    restricted, ROHF, UHF and natural-orbital branches all reduce to this."""
    pts0, wts0 = multiwfn_atom_grid(nrad, nang, radcut)
    nat = len(coords)
    # (orbital indices, (n_i n_j)^(1/4), flattened weighted AOMs, spins it covers)
    occs = [basis["occ_a"]] if np.array_equal(basis["occ_a"], basis["occ_b"]) else [basis["occ_a"], basis["occ_b"]]
    spins = []
    for occ in occs:
        idx = np.flatnonzero(occ > 0)
        if len(idx):
            q = np.sqrt(np.sqrt(np.outer(occ[idx], occ[idx])))
            spins.append((idx, q, np.empty((nat, len(idx) * len(idx))), 3 - len(occs)))
    for a in range(nat):
        pts = pts0 + coords[a]
        phi = orbital_values(pts, coords, basis)
        w = _becke(pts, coords, rinv, aij, BECKE_TOL)[:, a] * wts0
        for idx, q, y, _ in spins:
            p = phi[:, idx]
            y[a] = ((p.T @ (p * w[:, None])) * q).ravel()
    di = sum(2 * mult * (y @ y.T) for _, _, y, mult in spins)
    bonds = {}
    for i in range(nat):
        for j in range(i + 1, nat):
            if di[i, j] >= threshold:
                bonds[f"{labels[i]}_to_{labels[j]}"] = float(di[i, j])
    return bonds


def normalize(charge, zeff, nelec):
    """normalize_atmchg: scale atomic populations to the electron count."""
    pop = zeff - charge
    return zeff - nelec / pop.sum() * pop


# ---------------------------------------------------------------- driver


@limit_blas_threads
def run(wfx_path, nrad=75, nang=434, radcut=10.0, bond_nrad=45, bond_nang=170, full_set=0):
    t = {}
    t0 = time.perf_counter()
    wfx = read_wfx(wfx_path)
    basis = prepare_basis(wfx)
    t["load"] = time.perf_counter() - t0

    atnums = wfx["atnums"]
    coords = wfx["coords"]
    zeff = wfx["zeff"]
    nat = len(atnums)
    # production runs the spin steps only when mult != 1 (check_spin), including
    # for UKS singlets, so the same rule decides which keys are emitted
    open_shell = wfx["mult"] != 1
    has_ecp = bool((zeff != atnums).any())
    has_edf = len(wfx["edf_center"]) > 0
    grid_pts, grid_wts = multiwfn_atom_grid(nrad, nang, radcut)
    rinv, aij = becke_pair_tables(atnums, coords)

    skipped = []
    pro, missing = mwfn_proatom_tables(atnums)
    if missing:
        skipped.append({"schemes": "hirshfeld family", "reason": "no_builtin_proatom", "elements": missing})
    if has_ecp and not has_edf:
        # valence-only density against all-electron proatoms would be inconsistent
        pro = None
        skipped.append({"schemes": "hirshfeld family", "reason": "ecp_without_edf"})
    # full_set >= 1 adds vdd, mbis and the MBIS fuzzy integrals
    level1 = full_set >= 1
    mbis_ok = level1 and int(atnums.max()) <= 86 and not (has_ecp and not has_edf)
    if level1 and not mbis_ok:
        reason = "z_above_86" if int(atnums.max()) > 86 else "ecp_without_edf"
        skipped.append({"schemes": "mbis family", "reason": reason})
    if mbis_ok:
        td_rho = np.zeros((nat, len(grid_wts)))
        td_spin = np.zeros((nat, len(grid_wts))) if open_shell else None
    q_vdd = np.zeros(nat)

    pop_b = np.zeros(nat)
    pop_h = np.zeros(nat)
    dip_b = np.zeros((nat, 3))
    dip_h = np.zeros((nat, 3))
    fz = {k: np.zeros(nat) for k in ("bd", "bs", "hd", "hs")}
    t.update({"density": 0.0, "becke": 0.0, "hirshfeld": 0.0, "accumulate": 0.0})
    nelec_grid = 0.0

    for b in range(nat):
        pts = grid_pts + coords[b]
        t0 = time.perf_counter()
        rho, spin = density(pts, coords, basis, wfx)
        t["density"] += time.perf_counter() - t0
        t0 = time.perf_counter()
        w = _becke(pts, coords, rinv, aij, BECKE_TOL)
        t["becke"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        rel = grid_pts
        wb = w[:, b] * grid_wts
        pop_b[b] = wb @ rho
        dip_b[b] = -(rel * (wb * rho)[:, None]).sum(0)
        nelec_grid += wb @ rho
        fz["bd"] += w.T @ (wb * rho)
        if open_shell:
            fz["bs"] += w.T @ (wb * spin)
        if mbis_ok:
            td_rho[b] = wb * rho
            if open_shell:
                td_spin[b] = wb * spin
        t["accumulate"] += time.perf_counter() - t0

        if pro is not None:
            t0 = time.perf_counter()
            h, empty, promol = _hirshfeld_mwfn(pts, coords, *pro)
            t["hirshfeld"] += time.perf_counter() - t0
            if level1:
                # VDD: Voronoi cell, every point (no promol != 0 test)
                mask = _voronoi_mask(pts, coords, b)
                q_vdd[b] = -(mask * (rho - promol) * grid_wts).sum()
            t0 = time.perf_counter()
            # charges skip zero-promolecule points; fuzzy gives them to the centre atom
            hw = h[:, b] * grid_wts
            pop_h[b] = hw @ rho
            dip_h[b] = -(rel * (hw * rho)[:, None]).sum(0)
            h[empty, b] = 1.0
            fz["hd"] += h.T @ (wb * rho)
            if open_shell:
                fz["hs"] += h.T @ (wb * spin)
            t["accumulate"] += time.perf_counter() - t0

    if mbis_ok:
        # q = Z_eff + EDF core electrons - population
        qbase = atnums.astype(float) if has_edf else zeff
        t0 = time.perf_counter()
        # MBIS charges: MBIS_wrapper's own grid, Becke-weighted rho
        mgrid_pts, mgrid_wts = multiwfn_atom_grid(*mbis_grid_size(atnums), radcut)
        td_m = np.zeros((nat, len(mgrid_wts)))
        for b in range(nat):
            pts = mgrid_pts + coords[b]
            rho, _ = density(pts, coords, basis, wfx)
            td_m[b] = _becke(pts, coords, rinv, aij, BECKE_TOL)[:, b] * mgrid_wts * rho
        q_mbis, _, _, _, ncyc_charge = mbis_fit(mgrid_pts, coords, td_m, atnums, qbase)
        del td_m
        t["mbis_charge"] = time.perf_counter() - t0
        # MBIS fuzzy: refit on the main grid (fuzzyana calls MBIS(2,0) unwrapped),
        # radial tables from the fit, then integrate like the Hirshfeld fuzzy steps
        t0 = time.perf_counter()
        _, mshell, spop, ssig, ncyc_fuzzy = mbis_fit(grid_pts, coords, td_rho, atnums, qbase)
        mtab = mbis_radial_tables(mshell, spop, ssig)
        fz["md"] = np.zeros(nat)
        fz["ms"] = np.zeros(nat)
        for b in range(nat):
            h, empty, _ = _hirshfeld_mwfn(grid_pts + coords[b], coords, *mtab)
            h[empty] = 0.0
            h[empty, b] = 1.0
            fz["md"] += h.T @ td_rho[b]
            if open_shell:
                fz["ms"] += h.T @ td_spin[b]
        t["mbis_fuzzy"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    labels = [f"{i + 1}_{T_SYMBOL[z]}" for i, z in enumerate(atnums)]

    def chg(q):
        return {k: float(v) for k, v in zip(labels, q)}

    def atomic_dip(d):
        return {k: [float(x) for x in v] for k, v in zip(labels, d)}

    def mol_dip(q):
        v = (coords * q[:, None]).sum(0)
        return {"mag": float(np.linalg.norm(v)), "xyz": [float(x) for x in v]}

    def fuzzy(name, vals):
        d = chg(vals)
        d["sum"] = float(vals.sum())
        d["abs_sum"] = float(np.abs(vals).sum())
        return {name: d}

    nelec = basis["nelec"]
    out = {}
    # q = Z_eff + EDF core electrons - population = Z - population
    q_b = atnums - pop_b
    q_b_adc = adc(q_b, dip_b, coords, atnums)
    out["becke"] = {
        "charge": chg(normalize(q_b_adc, zeff, nelec)),
        "dipole": mol_dip(q_b_adc),
        "atomic_dipole": atomic_dip(dip_b),
    }
    out["becke_fuzzy_density"] = fuzzy("becke_fuzzy_density", fz["bd"])
    if open_shell:
        out["becke_fuzzy_spin"] = fuzzy("becke_fuzzy_spin", fz["bs"])

    if pro is not None:
        q_h = atnums - pop_h
        q_adch = adc(q_h, dip_h, coords, atnums)
        q_cm5 = cm5(q_h, coords, atnums)
        out["hirshfeld"] = {"charge": chg(normalize(q_h, zeff, nelec)), "dipole": mol_dip(q_h)}
        out["adch"] = {
            "charge": chg(normalize(q_adch, zeff, nelec)),
            "dipole": mol_dip(q_adch),
            "atomic_dipole": atomic_dip(dip_h),
        }
        # shipped cm5 "dipole" is the pre-correction Hirshfeld dipole (parser quirk)
        out["cm5"] = {"charge": chg(normalize(q_cm5, zeff, nelec)), "dipole": mol_dip(q_h)}
        out["hirsh_fuzzy_density"] = fuzzy("hirsh_fuzzy_density", fz["hd"])
        if open_shell:
            out["hirsh_fuzzy_spin"] = fuzzy("hirsh_fuzzy_spin", fz["hs"])
        if level1:
            # printed VDD dipole comes from the raw charges, before normalization
            out["vdd"] = {"charge": chg(normalize(q_vdd, zeff, nelec)), "dipole": mol_dip(q_vdd)}
    if mbis_ok:
        out["mbis"] = {"charge": chg(q_mbis)}
        out["mbis_fuzzy_density"] = fuzzy("mbis_fuzzy_density", fz["md"])
        if open_shell:
            out["mbis_fuzzy_spin"] = fuzzy("mbis_fuzzy_spin", fz["ms"])
    t["corrections"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    out["fuzzy_bond"] = fuzzy_bond_orders(basis, coords, rinv, aij, radcut, labels, bond_nrad, bond_nang)
    t["fuzzy_bond"] = time.perf_counter() - t0

    out["_meta"] = {
        "engine": "charge_engine_worker",
        "grid": {"nrad": nrad, "nang": nang, "radcut": radcut, "points_per_atom": int(len(grid_wts)),
                 "bond_nrad": bond_nrad, "bond_nang": bond_nang},
        "nelec_wfx": nelec,
        "nelec_grid_becke": float(nelec_grid),
        "edf_electrons": float((atnums - zeff).sum()) if has_edf else 0.0,
        "open_shell": bool(open_shell),
        "has_ecp": has_ecp,
        "full_set": full_set,
        "mbis_cycles": {"charge": ncyc_charge, "fuzzy": ncyc_fuzzy} if mbis_ok else None,
        "skipped": skipped,
        "timings_s": {k: round(v, 3) for k, v in t.items()},
    }
    return out


T_SYMBOL = (
    "X H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn "
    "Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce "
    "Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn "
    "Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl "
    "Mc Lv Ts Og"
).split()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--wfx", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--nrad", type=int, default=75, help="radial points per atom (Multiwfn radpot)")
    p.add_argument("--nang", type=int, default=434, help="Lebedev points per shell (Multiwfn sphpot)")
    p.add_argument("--radcut", type=float, default=10.0, help="drop grid points beyond this, Bohr")
    p.add_argument("--bond_nrad", type=int, default=45, help="fuzzy_bond radial points (Multiwfn: 45)")
    p.add_argument("--bond_nang", type=int, default=170, help="fuzzy_bond Lebedev points (Multiwfn: 170)")
    p.add_argument("--nthreads", type=int, default=None)
    p.add_argument("--full_set", type=int, default=0, help="1 adds vdd, mbis, mbis_fuzzy_density/spin")
    args = p.parse_args(argv)

    if args.nthreads:
        import numba

        numba.set_num_threads(args.nthreads)

    t0 = time.perf_counter()
    out = run(args.wfx, args.nrad, args.nang, args.radcut, args.bond_nrad, args.bond_nang, args.full_set)
    out["_meta"]["wall_s"] = round(time.perf_counter() - t0, 3)

    out_abs = os.path.abspath(args.out)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(out_abs), prefix=os.path.basename(out_abs) + ".", suffix=".tmp")
    os.fchmod(fd, 0o644)
    with os.fdopen(fd, "w") as f:
        json.dump(out, f, indent=1)
    os.replace(tmp, out_abs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
