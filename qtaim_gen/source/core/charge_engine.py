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
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numba
import numpy as np
from numba import njit, prange
from threadpoolctl import ThreadpoolController

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


@njit(nogil=True, cache=True)
def _primitive_block(pts, atcoords, groups, g_center, g_alpha, g_start, g_end, lmn, ncols):
    """Values of the primitives of `groups` (one column each, in group order) at
    every point (orbderv's GTF part), zero where Multiwfn's exp cutoff drops
    them. Callers pass only the groups whose cutoff sphere reaches the block.
    Serial and GIL-free: orbital_values runs one block per worker thread."""
    n = pts.shape[0]
    out = np.zeros((n, ncols))
    for ip in range(n):
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
    matrix product with their coefficients. Blocks run on numba.get_num_threads()
    worker threads, each with single-threaded BLAS (see limit_blas_threads)."""
    pts = np.asarray(pts, dtype=float)
    ct = basis["ct"]
    out = np.empty((len(pts), ct.shape[1]))
    if not len(pts):
        return out
    order = _morton_order(pts)
    sorted_pts = np.ascontiguousarray(pts[order])
    gs, ge, ga, gc = basis["g_start"], basis["g_end"], basis["g_alpha"], basis["g_center"]
    glen = ge - gs

    def one_block(s0):
        blk = sorted_pts[s0 : s0 + block]
        lo, hi = blk.min(axis=0), blk.max(axis=0)
        d2 = (np.maximum(0.0, np.maximum(lo - coords, coords - hi)) ** 2).sum(axis=1)
        groups = np.flatnonzero(ga * d2[gc] <= EXP_CUTOFF)
        n = glen[groups]
        cols = np.arange(n.sum()) + np.repeat(gs[groups] - (np.cumsum(n) - n), n)
        prim = _primitive_block(blk, coords, groups, gc, ga, gs, ge, basis["lmn"], len(cols))
        out[order[s0 : s0 + block]] = prim @ ct[cols]

    starts = range(0, len(pts), block)
    nthreads = numba.get_num_threads()
    if nthreads == 1 or len(starts) == 1:
        for s0 in starts:
            one_block(s0)
    else:
        with ThreadPoolExecutor(nthreads) as pool:
            list(pool.map(one_block, starts))
    return out


@njit(nogil=True, cache=True)
def _primitive_block_derivs(pts, atcoords, groups, g_center, g_alpha, g_start, g_end, lmn, ncols):
    """_primitive_block plus first and second derivatives: out[q, point, column]
    with q = value, d/dx, d/dy, d/dz, d2/dx2, d2/dy2, d2/dz2, d2/dxdy, d2/dxdz,
    d2/dydz. For x^l exp(-a r^2) the 1D factors are f0 = x^l,
    f1 = l x^(l-1) - 2a x^(l+1), f2 = l(l-1) x^(l-2) - 2a(2l+1) x^l + 4a^2 x^(l+2)."""
    n = pts.shape[0]
    out = np.zeros((10, n, ncols))
    f = np.empty((3, 3))
    d = np.zeros(3)
    for ip in range(n):
        x, y, z = pts[ip, 0], pts[ip, 1], pts[ip, 2]
        col = 0
        last = -1
        r2 = 0.0
        for gi in range(groups.shape[0]):
            g = groups[gi]
            a = g_center[g]
            if a != last:
                d[0] = x - atcoords[a, 0]
                d[1] = y - atcoords[a, 1]
                d[2] = z - atcoords[a, 2]
                r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2]
                last = a
            alpha = g_alpha[g]
            ar2 = alpha * r2
            if ar2 > EXP_CUTOFF:
                col += g_end[g] - g_start[g]
                continue
            e = np.exp(-ar2)
            for p in range(g_start[g], g_end[g]):
                for k in range(3):
                    lk = lmn[p, k]
                    u = d[k]
                    pk = 1.0
                    for _ in range(lk):
                        pk *= u
                    pkm1 = 0.0
                    if lk >= 1:
                        pkm1 = 1.0
                        for _ in range(lk - 1):
                            pkm1 *= u
                    pkm2 = 0.0
                    if lk >= 2:
                        pkm2 = 1.0
                        for _ in range(lk - 2):
                            pkm2 *= u
                    f[k, 0] = pk
                    f[k, 1] = lk * pkm1 - 2.0 * alpha * pk * u
                    f[k, 2] = lk * (lk - 1) * pkm2 - 2.0 * alpha * (2 * lk + 1) * pk + 4.0 * alpha * alpha * pk * u * u
                out[0, ip, col] = f[0, 0] * f[1, 0] * f[2, 0] * e
                out[1, ip, col] = f[0, 1] * f[1, 0] * f[2, 0] * e
                out[2, ip, col] = f[0, 0] * f[1, 1] * f[2, 0] * e
                out[3, ip, col] = f[0, 0] * f[1, 0] * f[2, 1] * e
                out[4, ip, col] = f[0, 2] * f[1, 0] * f[2, 0] * e
                out[5, ip, col] = f[0, 0] * f[1, 2] * f[2, 0] * e
                out[6, ip, col] = f[0, 0] * f[1, 0] * f[2, 2] * e
                out[7, ip, col] = f[0, 1] * f[1, 1] * f[2, 0] * e
                out[8, ip, col] = f[0, 1] * f[1, 0] * f[2, 1] * e
                out[9, ip, col] = f[0, 0] * f[1, 1] * f[2, 1] * e
                col += 1
    return out


_THREADPOOLS = None
_THREADPOOLS_LOCK = threading.Lock()


def _threadpools():
    """One ThreadpoolController per process. threadpool_limits() builds a new one on
    every call, which rescans the loaded libraries (dl_iterate_phdr); with several
    engine runs in threads of one process (parsl's local ThreadPoolExecutor) that
    rescan deadlocked against the other runs' thread starts."""
    global _THREADPOOLS
    with _THREADPOOLS_LOCK:
        if _THREADPOOLS is None:
            _THREADPOOLS = ThreadpoolController()
    return _THREADPOOLS


def pin_blas_single_thread():
    """Single-threaded BLAS for the rest of the process, not restored. Called once from
    the main thread before engine runs start as threads, so the library scan has no
    other threads to deadlock against, and the per-call limits below save and restore
    1: one run leaving its limit cannot turn BLAS threading back on under another."""
    _threadpools().limit(limits=1, user_api="blas")


def limit_blas_threads(fn):
    """Run fn with single-threaded BLAS. orbital_values parallelizes over blocks
    itself; a multithreaded BLAS inside those workers, or next to numba's
    spin-waiting OpenMP threads, oversubscribes the cores (7x slower on 4
    pinned cores)."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with _threadpools().limit(limits=1):
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


# routines a skip removes; every skip recorded in _meta["skipped"] is permanent for the
# wavefunction (it depends only on its elements, ECP/EDF and the run options)
from qtaim_gen.source.data.multiwfn import HIRSHFELD_FAMILY, MBIS_FAMILY  # noqa: E402
MBIS_DENCUT = 1e-10  # MBIS dencut: shell densities below this are zeroed
MBIS_MAXSHELL = 6  # shells of the initial guess (period 6 rule, also used for period 7)
MBIS_MAX_Z = 118
MBIS_NRAD_PERIOD7 = 70  # MBIS grid radial points when any Z > 86 (Multiwfn's rule stops at 60)
MBIS_MODE = "converged"  # "multiwfn" reproduces Multiwfn's stopping rule (parity tests)
MBIS_GUESS = "fitted"  # converged mode's initial guess: free-atom fits, or "multiwfn" (mbis_initial_shells)
MBIS_TOL = 1e-8  # converged mode: largest population change (e) and relative width change per step
MBIS_MAXCYC = 20000
MBIS_ANDERSON = 6  # Anderson history length
MBIS_ANDERSON_MAXCYC = 1000  # map evaluations before an accelerated fit is redone by plain iteration (see mbis_select)
MBIS_ANDERSON_TRUST = 2.0  # largest factor an extrapolated parameter may differ from the plain step
MBIS_MERGE_TOL = 1e-3  # shells of one atom whose widths agree this closely (relative) have merged
MBIS_MERGE_DETECT = 1e-2  # plain fallback: shells this close in width (relative) are merged and refit
MBIS_MERGE_FIRST = 1000  # plain cycles before the first merge check
MBIS_MERGE_EVERY = 500  # plain cycles between later merge checks
MBIS_EMPTY_POP = 1e-6  # e: a shell below this has emptied
MBIS_CHECK_SIGMAS = (0.6, 1.5)  # width scales of the extra guesses mbis_select fits from
MBIS_UNIQUE_TOL = 0.005  # e: larger guess spreads mark the MBIS record as not unique


@njit(parallel=True, cache=True)
def _mbis_pair_scan(grel, atcoords, tmpden, cut2, cand_ptr, cand, pptr, pj, pd, fill):
    """Per flat point (centre * len(grel) + point) with tmpden > 1e-14, the
    atoms within atmrhocut (ignorefar=1) and their distance. fill False counts
    them into pptr[1:]; fill True writes them from pptr."""
    m = grel.shape[0]
    for c in prange(atcoords.shape[0]):
        for i in range(m):
            ip = c * m + i
            if not tmpden[c, i] > 1e-14:
                continue
            x = grel[i, 0] + atcoords[c, 0]
            y = grel[i, 1] + atcoords[c, 1]
            z = grel[i, 2] + atcoords[c, 2]
            s = pptr[ip] if fill else 0
            for r in range(cand_ptr[c], cand_ptr[c + 1]):
                j = cand[r]
                dx = x - atcoords[j, 0]
                dy = y - atcoords[j, 1]
                dz = z - atcoords[j, 2]
                d2 = dx * dx + dy * dy + dz * dz
                if d2 > cut2[j]:
                    continue
                if fill:
                    pj[s] = j
                    pd[s] = np.sqrt(d2)
                s += 1
            if not fill:
                pptr[ip + 1] = s


def mbis_pairs(grel, coords, tmpden, atnums):
    """The (atom, distance) pairs every MBIS map evaluation on this grid reads,
    built once: point ip's pairs are pj/pd[pptr[ip]:pptr[ip + 1]], atoms in the
    order of _mbis_setup's candidate lists; the fourth item is the most pairs of
    any point. About 12 bytes per pair (0.12 GB
    for 48 atoms, 0.54 GB for 274 atoms on the MBIS charge grid)."""
    cut2, cand_ptr, cand = _mbis_setup(grel, coords, atnums)
    pptr = np.zeros(len(coords) * grel.shape[0] + 1, dtype=np.int64)
    dummy_j, dummy_d = np.empty(0, dtype=np.int32), np.empty(0)
    _mbis_pair_scan(grel, coords, tmpden, cut2, cand_ptr, cand, pptr, dummy_j, dummy_d, False)
    pptr = np.cumsum(pptr)
    pj, pd = np.empty(pptr[-1], dtype=np.int32), np.empty(pptr[-1])
    _mbis_pair_scan(grel, coords, tmpden, cut2, cand_ptr, cand, pptr, pj, pd, True)
    return pptr, pj, pd, int(np.diff(pptr).max())


def _mbis_amp_dmax(mshell, pop, sig):
    """Each shell's density at the nucleus, pop / (8 pi sigma^3), and the
    distance beyond which it is below MBIS_DENCUT (padded so a skipped term is
    certainly below it; -1 for shells that never reach it)."""
    amp = np.zeros_like(pop)
    dmax = np.full(pop.shape, -1.0)
    for j in range(len(mshell)):
        for k in range(mshell[j]):
            sv = sig[j, k]
            if not sv > 0:
                # a shell whose population went to exactly 0 keeps sigma 0, where
                # Multiwfn would form 0/0; it contributes nothing here
                continue
            amp[j, k] = pop[j, k] / sv**3 / 8 / np.pi
            if amp[j, k] > MBIS_DENCUT:
                dmax[j, k] = sv * np.log(amp[j, k] / MBIS_DENCUT) * (1 + 1e-9) + 1e-12
    return amp, dmax, dmax.max(axis=1)


@njit(parallel=True, cache=True)
def _mbis_map_blocks(tmpden, pptr, pj, pd, mshell, amp, sig, dmax, dmax_atom, block, maxterms):
    """New shell populations and the integral part of the new widths (Eqs.
    18-19 of the MBIS paper), per block of flat points: each point's shell
    terms amp exp(-d/sigma) at or above MBIS_DENCUT (atoms and shells beyond
    their dmax skipped), their sum rho0, then w term / rho0. One partial per
    block, so the total does not depend on the thread count."""
    nat, m = tmpden.shape
    npts = nat * m
    nb = (npts + block - 1) // block
    ppop = np.zeros((nb, nat, MBIS_MAXSHELL))
    psig = np.zeros((nb, nat, MBIS_MAXSHELL))
    for b in prange(nb):
        tj = np.empty(maxterms, dtype=np.int64)
        tk = np.empty(maxterms, dtype=np.int64)
        tt = np.empty(maxterms)
        tdd = np.empty(maxterms)
        for ip in range(b * block, min(npts, (b + 1) * block)):
            n = 0
            rho0 = 0.0
            for s in range(pptr[ip], pptr[ip + 1]):
                j = pj[s]
                d = pd[s]
                if d > dmax_atom[j]:
                    continue
                for k in range(mshell[j]):
                    if d > dmax[j, k]:
                        continue
                    t = amp[j, k] * np.exp(-d / sig[j, k])
                    if t < MBIS_DENCUT:
                        continue
                    tj[n] = j
                    tk[n] = k
                    tt[n] = t
                    tdd[n] = d
                    n += 1
                    rho0 += t
            if rho0 > 0.0:
                c = ip // m
                w = tmpden[c, ip - c * m]
                for u in range(n):
                    f = w * tt[u] / rho0
                    ppop[b, tj[u], tk[u]] += f
                    psig[b, tj[u], tk[u]] += f * tdd[u]
    return ppop, psig


def _mbis_cycle(tmpden, pairs, mshell, pop, sig, block=4096):
    """One MBIS iteration (MBIS, imode 0) on numba.get_num_threads() threads,
    from the mbis_pairs cache."""
    pptr, pj, pd, maxpairs = pairs
    amp, dmax, dmax_atom = _mbis_amp_dmax(mshell, pop, sig)
    ppop, psig = _mbis_map_blocks(tmpden, pptr, pj, pd, mshell, amp, sig, dmax, dmax_atom, block,
                                  maxpairs * MBIS_MAXSHELL)
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
    30 radial points, raised to 40/50/60 when any Z exceeds 18/36/54.
    Multiwfn stops at Rn; beyond it the engine continues the rule with
    MBIS_NRAD_PERIOD7 (no Multiwfn reference)."""
    zmax = int(atnums.max())
    nrad = (MBIS_NRAD_PERIOD7 if zmax > 86 else 60 if zmax > 54 else 50 if zmax > 36
            else 40 if zmax > 18 else 30)
    return nrad, 302


def mbis_initial_shells(atnums):
    """MBIS initial guess (icore=1): shell count by period, populations 2/8/8/
    18/18 for the core shells and the rest in the valence shell, widths from
    1/(2Z) for the innermost to 1/2 for the valence shell. Uses the true Z.
    Multiwfn stops at Rn; period 7 (Z 87-118) keeps its 6-shell rule (valence
    Z - 54): a seventh shell merges with its neighbour or empties at the fixed
    point and makes the solution depend on the guess."""
    nat = len(atnums)
    mshell = np.zeros(nat, dtype=np.int64)
    pop = np.zeros((nat, MBIS_MAXSHELL))
    sig = np.zeros((nat, MBIS_MAXSHELL))
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


def mbis_fit(grel, coords, tmpden, atnums, qbase, crit=1e-4, maxcyc=500, mode=None, guess=None):
    """MBIS shell fit from one guess (run() uses mbis_select in converged mode);
    returns the charges (qbase - population; not normalized,
    as printed for imode 0), the shell parameters the radial tables are built
    from, and the number of cycles.

    mode "multiwfn": Multiwfn's loop, until no charge moves by crit in a cycle or
    maxcyc; the shell parameters returned are those of the previous cycle (the
    loop exits before replacing them). It can stop far from the fixed point: on
    30 local jobs, 4 ended more than 0.03 e from it (up to 0.25 e).
    mode "converged" (default, MBIS_MODE): Anderson-accelerated iteration to the
    fixed point (_mbis_converged). guess: (mshell, pop, sig) in place of
    mbis_initial_shells."""
    mode = mode or MBIS_MODE
    if guess is not None:
        mshell, pop, sig = guess
    elif mode == "converged":
        mshell, pop, sig = mbis_guesses(atnums)[0][1]
    else:
        mshell, pop, sig = mbis_initial_shells(atnums)
    if mode not in ("converged", "multiwfn"):
        raise ValueError(f"unknown MBIS mode {mode!r}")
    pairs = mbis_pairs(grel, coords, tmpden, atnums)
    if mode == "converged":
        step = lambda mm, pp, ss: _mbis_step(tmpden, pairs, mm, pp, ss)  # noqa: E731
        objective = lambda mm, pp, ss: mbis_objective(grel, coords, tmpden, atnums, mm, pp, ss, pairs=pairs)  # noqa: E731
        return _mbis_fit_converged(step, mshell, pop, sig, qbase, objective=objective)[0]
    last = np.zeros(len(atnums))
    for icyc in range(1, maxcyc + 1):
        popnew, signew = _mbis_cycle(tmpden, pairs, mshell, pop, sig)
        pos = popnew > 0
        signew[pos] /= 3 * popnew[pos]
        charge = qbase - popnew.sum(axis=1)
        varmax = np.abs(charge - last).max()
        if varmax < crit or icyc == maxcyc:
            return charge, mshell, pop, sig, icyc
        last = charge
        pop, sig = popnew, signew


def _mbis_setup(grel, coords, atnums):
    """atmrhocut squared per atom, and per grid centre the atoms some of its
    points can be within atmrhocut of (cand[cand_ptr[c]:cand_ptr[c + 1]])."""
    from qtaim_gen.source.data import multiwfn_atmraddens as R

    cut2 = np.array([R.ATMRHOCUT[int(z)] ** 2 for z in atnums])
    reach = np.sqrt((grel**2).sum(axis=1)).max() + np.sqrt(cut2)
    dist = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)
    cand_lists = [np.flatnonzero(dist[c] <= reach * (1 + 1e-9)) for c in range(len(coords))]
    cand_ptr = np.concatenate([[0], np.cumsum([len(x) for x in cand_lists])]).astype(np.int64)
    return cut2, cand_ptr, np.concatenate(cand_lists).astype(np.int64)


def _mbis_step(tmpden, pairs, mshell, pop, sig):
    """The MBIS map: new populations and widths from the current ones."""
    popnew, signew = _mbis_cycle(tmpden, pairs, mshell, pop, sig)
    pos = popnew > 0
    signew[pos] /= 3 * popnew[pos]
    return popnew, signew


def mbis_merge_shells(mshell, pop, sig):
    """Each atom's shells, sorted by width, with adjacent ones whose widths agree
    within MBIS_MERGE_DETECT (relative) combined into one: populations summed,
    width population-weighted."""
    ms, p, s = mshell.copy(), np.zeros_like(pop), np.zeros_like(sig)
    for a, m in enumerate(mshell):
        order = np.argsort(sig[a, :m])
        groups = [[order[0]]] if m else []
        for k in order[1:]:
            if abs(np.log(sig[a, k] / sig[a, groups[-1][-1]])) < MBIS_MERGE_DETECT:
                groups[-1].append(k)
            else:
                groups.append([k])
        for i, g in enumerate(groups):
            p[a, i] = pop[a, g].sum()
            s[a, i] = (pop[a, g] * sig[a, g]).sum() / p[a, i]
        ms[a] = len(groups)
    return ms, p, s


def _mbis_fit_converged(step, mshell, pop, sig, qbase, fallback=True, objective=None, fallback_above=None):
    """Accelerated fit; with fallback, redone by plain iteration from the guess
    when acceleration stalls (MBIS_ANDERSON_MAXCYC evaluations) or settles on a
    merged/emptied-shell fixed point plain iteration leaves. Without fallback,
    the fallback still runs when fallback_above is given and the failed
    accelerated fit's objective exceeds it.

    With objective (mbis_objective of (mshell, pop, sig)), the plain fallback
    stops after MBIS_MERGE_FIRST and then every MBIS_MERGE_EVERY cycles to merge
    shells of one atom that have come within MBIS_MERGE_DETECT in width: the
    valence shell of a cation (K+, Ca2+, Ti, Y, Hf, ...) settles next to the
    outermost core shell, and plain iteration then moves population between the
    two by up to 1e-6 e per cycle without converging while the charges stay
    fixed to 1e-11 e. The merged model is refit with acceleration and kept when
    it converges, is not degenerate and its objective is not below that of the
    plain state it came from; otherwise plain iteration continues. step is
    called as step(mshell, pop, sig). Returns (charge, mshell, pop, sig,
    evaluations), whether the returned fit met the stopping rule, and whether
    the fallback ran."""
    res = _mbis_converged(step, mshell, pop, sig, qbase, maxcyc=MBIS_ANDERSON_MAXCYC)
    converged = res[4] < MBIS_ANDERSON_MAXCYC
    if converged and not mbis_degenerate(res[1], res[2], res[3]):
        return res, converged, False
    if not fallback and (fallback_above is None or not objective(res[1], res[2], res[3]) > fallback_above):
        return res, converged, False
    evals, done, p, s = res[4], 0, pop, sig
    while done < MBIS_MAXCYC:
        chunk = MBIS_MAXCYC - done
        if objective is not None:
            chunk = min(chunk, MBIS_MERGE_FIRST if done == 0 else MBIS_MERGE_EVERY)
        plain = _mbis_converged(step, mshell, p, s, qbase, maxcyc=chunk, history=0)
        evals, done, p, s = evals + plain[4], done + plain[4], plain[2], plain[3]
        if plain[4] < chunk:
            return plain[:4] + (evals,), True, True
        if objective is None:
            break
        m2, p2, s2 = mbis_merge_shells(mshell, p, s)
        if (m2 == mshell).all():
            continue
        fit = _mbis_converged(step, m2, p2, s2, qbase, maxcyc=MBIS_ANDERSON_MAXCYC)
        evals += fit[4]
        if (fit[4] < MBIS_ANDERSON_MAXCYC and not mbis_degenerate(fit[1], fit[2], fit[3])
                and objective(fit[1], fit[2], fit[3]) >= objective(mshell, p, s)):
            return fit[:4] + (evals,), True, True
    return plain[:4] + (evals,), False, True


def _mbis_converged(step, mshell, pop, sig, qbase, tol=None, maxcyc=None, history=None):
    """Fixed point of the MBIS map by Anderson acceleration (type II, on the log
    populations and log widths of the active shells). An extrapolated point that
    moves any parameter by more than a factor MBIS_ANDERSON_TRUST from the plain
    step is replaced by the plain step and the history reset: unguarded, the
    extrapolation can empty a shell and land on a degenerate fixed point plain
    iteration never reaches (Li 1s population to 0). A shell whose population is
    exactly 0 stays empty, as in plain iteration. Converged when no population
    moves by more than tol (e) and no width by more than tol relative in one plain
    step; returns charge, mshell, pop, sig, the number of map evaluations."""
    tol = MBIS_TOL if tol is None else tol
    maxcyc = MBIS_MAXCYC if maxcyc is None else maxcyc
    history = MBIS_ANDERSON if history is None else history
    act = np.zeros(pop.shape, dtype=bool)
    for a, m in enumerate(mshell):
        act[a, :m] = True
    trust = np.log(MBIS_ANDERSON_TRUST)

    xs, gs = [], []
    p, s = pop, sig
    for icyc in range(1, maxcyc + 1):
        pn, sn = step(mshell, p, s)
        dpop = np.abs(pn - p)[act].max()
        with np.errstate(divide="ignore", invalid="ignore"):
            dsig = np.where(sn[act] > 0, np.abs(sn[act] - s[act]) / sn[act], 0.0).max()
        if dpop < tol and dsig < tol:
            return qbase - pn.sum(axis=1), mshell, pn, sn, icyc
        live = act & (pn > 0) & (p > 0)
        x = np.concatenate([np.log(p[live]), np.log(s[live])])
        g = np.concatenate([np.log(pn[live]), np.log(sn[live])])
        if xs and len(xs[-1]) != len(x):
            xs, gs = [], []
        xs.append(x)
        gs.append(g)
        xs, gs = xs[-(history + 1):], gs[-(history + 1):]
        p, s = pn, sn
        if len(xs) > 1:
            f = np.array(gs) - np.array(xs)
            gamma = np.linalg.lstsq(np.diff(f, axis=0).T, f[-1], rcond=None)[0]
            nxt = g - np.diff(np.array(gs), axis=0).T @ gamma
            if np.isfinite(nxt).all() and np.abs(nxt - g).max() <= trust:
                n = int(live.sum())
                p, s = pn.copy(), sn.copy()
                p[live], s[live] = np.exp(nxt[:n]), np.exp(nxt[n:])
            else:
                xs, gs = [], []
    return qbase - pn.sum(axis=1), mshell, pn, sn, maxcyc


def mbis_degenerate(mshell, pop, sig):
    """True when some atom has two shells whose widths agree within
    MBIS_MERGE_TOL (relative) or a shell below MBIS_EMPTY_POP electrons: a fixed
    point of a model with fewer shells, which the MBIS map leaves invariant."""
    for a, m in enumerate(mshell):
        if m and (pop[a, :m] < MBIS_EMPTY_POP).any():
            return True
        ls = np.log(sig[a, :m][sig[a, :m] > 0])
        if len(ls) > 1 and np.min(np.abs(ls[:, None] - ls[None, :]) + np.eye(len(ls))) < MBIS_MERGE_TOL:
            return True
    return False


@njit(parallel=True, cache=True)
def _mbis_logsum_blocks(tmpden, pptr, pj, pd, mshell, amp, sig, dmax, dmax_atom, block):
    """Per block of flat points, the sum of w ln(rho0), rho0 the promolecule
    density with the map's MBIS_DENCUT truncation; points with rho0 = 0 add
    w ln(1e-300)."""
    nat, m = tmpden.shape
    npts = nat * m
    nb = (npts + block - 1) // block
    part = np.zeros(nb)
    for b in prange(nb):
        tot = 0.0
        for ip in range(b * block, min(npts, (b + 1) * block)):
            c = ip // m
            w = tmpden[c, ip - c * m]
            if not w > 1e-14:
                continue
            rho0 = 0.0
            for s in range(pptr[ip], pptr[ip + 1]):
                j = pj[s]
                d = pd[s]
                if d > dmax_atom[j]:
                    continue
                for k in range(mshell[j]):
                    if d > dmax[j, k]:
                        continue
                    t = amp[j, k] * np.exp(-d / sig[j, k])
                    if t >= MBIS_DENCUT:
                        rho0 += t
            tot += w * np.log(max(rho0, 1e-300))
        part[b] = tot
    return part


def mbis_objective(grel, coords, tmpden, atnums, mshell, pop, sig, block=4096, pairs=None):
    """sum_i w_i ln rho0(r_i) on the fit's grid: the part of the MBIS
    Kullback-Leibler divergence int rho ln(rho / rho0) that differs between
    fixed points (their total populations all equal the grid's electron count).
    Larger is a better fit. pairs: the grid's mbis_pairs, built when not given."""
    pptr, pj, pd, _ = pairs if pairs is not None else mbis_pairs(grel, coords, tmpden, atnums)
    amp, dmax, dmax_atom = _mbis_amp_dmax(mshell, pop, sig)
    return float(_mbis_logsum_blocks(tmpden, pptr, pj, pd, mshell, amp, sig, dmax, dmax_atom, block).sum())


def mbis_fitted_shells(atnums):
    """Converged mode's initial guess: each atom's free-atom MBIS fit
    (data/mbis_free_atom.py, fitted to the Hirshfeld proatom densities by
    scripts/fit_mbis_free_atom_table.py; Z 1-103). Elements beyond the table
    fall back to mbis_initial_shells."""
    from qtaim_gen.source.data.mbis_free_atom import SHELLS

    mshell, pop, sig = mbis_initial_shells(atnums)
    for a, z in enumerate(atnums):
        shells = SHELLS.get(int(z))
        if shells is None:
            continue
        mshell[a] = len(shells)
        pop[a], sig[a] = 0.0, 0.0
        pop[a, : len(shells)] = [n for n, _ in shells]
        sig[a, : len(shells)] = [w for _, w in shells]
    return mshell, pop, sig


def mbis_guesses(atnums):
    """(name, (mshell, pop, sig)) of the guesses mbis_select fits from: the
    free-atom fits (MBIS_GUESS "fitted") or Multiwfn's rule ("multiwfn"), then
    that guess with every width scaled by each MBIS_CHECK_SIGMAS."""
    m0, p0, s0 = mbis_fitted_shells(atnums) if MBIS_GUESS == "fitted" else mbis_initial_shells(atnums)
    return [("default", (m0, p0, s0))] + [(f"sigma_x{f:g}", (m0, p0, s0 * f)) for f in MBIS_CHECK_SIGMAS]


def mbis_select(grel, coords, tmpden, atnums, qbase):
    """MBIS can have several fixed points (on 27 local jobs below Rn, 8 had
    another more than 0.005 e away; every actinide tested did, up to 0.9 e).
    Fit from the default guess and from it with every width scaled by each of
    MBIS_CHECK_SIGMAS, keep the fit with the largest mbis_objective (smallest
    divergence), non-degenerate fits first, ties to the earlier guess. Returns
    charge, mshell, pop, sig, the total cycles, and a record of the choice.

    The check guesses fall back to plain iteration (with shell merging, see
    _mbis_fit_converged) when the default fit needed the fallback, or when
    their failed accelerated fit already has a larger objective than every
    converged, non-degenerate fit so far; otherwise a check guess whose
    accelerated fit stalls or ends degenerate is left out of the selection (on
    Sm3Ir3, tm_react, the two check fallbacks took 515 of 539 s and found
    nothing; on rmechdb_750, whose default fit needs the fallback, the x1.5
    fallback gives the fit with the best objective; on 11-15 of 440 NERSC
    actinide jobs a stalled x1.5 fit was better than the selection by up to
    5e-3 relative, and whether it converged in MBIS_ANDERSON_MAXCYC evaluations
    flipped the stored charges by up to 0.3 e between runs). spread and
    solutions count only converged, non-degenerate fits; merged lists [atom,
    shells in the guess, shells in the fit] per guess."""
    pairs = mbis_pairs(grel, coords, tmpden, atnums)
    step = lambda mm, pp, ss: _mbis_step(tmpden, pairs, mm, pp, ss)  # noqa: E731
    objective = lambda mm, pp, ss: mbis_objective(grel, coords, tmpden, atnums, mm, pp, ss, pairs=pairs)  # noqa: E731
    fits = []
    for i, (name, (m0, p0, s0)) in enumerate(mbis_guesses(atnums)):
        fallback = i == 0 or fits[0][9]
        good_so_far = [f[6] for f in fits if f[8] and not f[7]]
        above = max(good_so_far) + 1e-12 * abs(max(good_so_far)) if good_so_far else None
        (q, m, p, s, n), conv, fell_back = _mbis_fit_converged(step, m0, p0, s0, qbase, fallback=fallback,
                                                               objective=objective, fallback_above=above)
        merged = [[int(a), int(m0[a]), int(m[a])] for a in np.flatnonzero(m < m0)]
        fits.append((name, q, m, p, s, n, objective(m, p, s), mbis_degenerate(m, p, s), conv, fell_back, merged))
    good = [f for f in fits if f[8] and not f[7]]
    tol = 1e-12 * max(abs(f[6]) for f in fits)
    best = fits[0]
    for f in good:
        if (best[7] and not f[7]) or (f[7] == best[7] and f[6] > best[6] + tol):
            best = f
    kept = [best] + [f for f in good if f is not best]
    distinct = []
    for f in kept:
        if all(np.abs(f[1] - d).max() > MBIS_UNIQUE_TOL for d in distinct):
            distinct.append(f[1])
    info = {"selected": best[0], "objective": {f[0]: f[6] for f in fits},
            "degenerate": {f[0]: bool(f[7]) for f in fits}, "converged": {f[0]: bool(f[8]) for f in fits},
            "merged": {f[0]: f[10] for f in fits},
            "spread": float(max(np.abs(f[1] - best[1]).max() for f in kept)), "solutions": len(distinct)}
    return best[1], best[2], best[3], best[4], sum(f[5] for f in fits), info


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
def run(wfx_path, nrad=75, nang=434, radcut=10.0, bond_nrad=45, bond_nang=170, full_set=0, mbis_mode=None,
        mbis_ecp=True):
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
        skipped.append({"schemes": "hirshfeld family", "reason": "no_builtin_proatom", "elements": missing,
                        "routines": list(HIRSHFELD_FAMILY)})
    if has_ecp and not has_edf:
        # valence-only density against all-electron proatoms would be inconsistent
        pro = None
        skipped.append({"schemes": "hirshfeld family", "reason": "ecp_without_edf", "routines": list(HIRSHFELD_FAMILY)})
    # full_set >= 1 adds vdd, mbis and the MBIS fuzzy integrals
    level1 = full_set >= 1
    # MBIS is defined for all-electron densities; with an ECP the EDF core density stands in
    # for the replaced electrons, and mbis_ecp=False skips MBIS for such records instead
    mbis_ok = (level1 and int(atnums.max()) <= MBIS_MAX_Z and not (has_ecp and not has_edf)
               and not (has_ecp and not mbis_ecp))
    if level1 and not mbis_ok:
        reason = (f"z_above_{MBIS_MAX_Z}" if int(atnums.max()) > MBIS_MAX_Z
                  else "ecp_without_edf" if not has_edf else "ecp_disabled")
        skipped.append({"schemes": "mbis family", "reason": reason, "routines": list(MBIS_FAMILY)})
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
        mbis_mode = mbis_mode or MBIS_MODE
        selection = {}
        if mbis_mode == "converged":
            q_mbis, cm, cp, cs, ncyc_charge, selection["charge"] = mbis_select(mgrid_pts, coords, td_m, atnums, qbase)
        else:
            q_mbis, cm, cp, cs, ncyc_charge = mbis_fit(mgrid_pts, coords, td_m, atnums, qbase, mode=mbis_mode)
        degenerate = {"charge": mbis_degenerate(cm, cp, cs)}
        del td_m
        t["mbis_charge"] = time.perf_counter() - t0
        # MBIS fuzzy: refit on the main grid (fuzzyana calls MBIS(2,0) unwrapped),
        # radial tables from the fit, then integrate like the Hirshfeld fuzzy steps
        t0 = time.perf_counter()
        if mbis_mode == "converged":
            # one fit, from the charge-grid fit's own shells (same density, finer grid), so both
            # land on the same fixed point: from the raw guess, the refit of an x1.5 selection
            # could return to the default's fixed point (0.25 e away on a U nitrate)
            guess = dict(mbis_guesses(atnums))[selection["charge"]["selected"]]
            _, mshell, spop, ssig, ncyc_fuzzy = mbis_fit(grid_pts, coords, td_rho, atnums, qbase, mode=mbis_mode,
                                                        guess=(cm, cp, cs))
        else:
            guess = mbis_initial_shells(atnums)
            _, mshell, spop, ssig, ncyc_fuzzy = mbis_fit(grid_pts, coords, td_rho, atnums, qbase, mode=mbis_mode)
        degenerate["fuzzy"] = mbis_degenerate(mshell, spop, ssig)
        merged = {"charge": selection["charge"]["merged"][selection["charge"]["selected"]] if selection else [],
                  "fuzzy": [[int(a), int(guess[0][a]), int(mshell[a])] for a in np.flatnonzero(mshell < guess[0])]}
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
        "mbis_mode": mbis_mode if mbis_ok else None,
        # converged mode: which guess's fixed point the charge-grid fit kept and why (mbis_select;
        # the fuzzy refit starts from that fit); not unique when another guess reached a fixed
        # point more than MBIS_UNIQUE_TOL away, None (unknown) when a guess did not converge
        "mbis_selection": selection.get("charge") if mbis_ok else None,
        "mbis_unique": (selection["charge"]["solutions"] == 1 if all(selection["charge"]["converged"].values())
                        else None) if mbis_ok and selection else None,
        # a merged or emptied shell in the stored fit (see mbis_degenerate)
        "mbis_degenerate": degenerate if mbis_ok else None,
        # shells the plain fallback merged, as [atom, shells in the guess, shells in the fit]
        "mbis_merged": merged if mbis_ok else None,
        # Multiwfn has no MBIS beyond Rn: these values come from the period-7 guess and grid
        "mbis_beyond_multiwfn": bool(mbis_ok and int(atnums.max()) > 86),
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
    p.add_argument("--mbis_mode", choices=("converged", "multiwfn"), default=None,
                   help="MBIS stopping: converged fixed point (default) or Multiwfn's rule (parity checks)")
    p.add_argument("--no_mbis_ecp", action="store_true",
                   help="skip MBIS when the wavefunction has ECP atoms (MBIS is defined for all-electron densities)")
    args = p.parse_args(argv)

    if args.nthreads:
        import numba

        numba.set_num_threads(args.nthreads)

    t0 = time.perf_counter()
    out = run(args.wfx, args.nrad, args.nang, args.radcut, args.bond_nrad, args.bond_nang, args.full_set,
              mbis_mode=args.mbis_mode, mbis_ecp=not args.no_mbis_ecp)
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
