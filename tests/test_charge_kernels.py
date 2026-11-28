"""The charge engine's restructured kernels (perf plan, phase 2.2) against the point-by-point
versions they replaced, kept here as references: the strip kernels and the two-phase Becke weights
must give the same bits, signed zeros included. Also the shared-evaluation options
orbital_values(cts=...) and density_derivatives(phi2=..., cts=...)."""

import importlib.util
import os

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(importlib.util.find_spec("numba") is None, reason="needs numba")

from numba import njit, prange  # noqa: E402

FIXTURES = os.path.join(os.path.dirname(__file__), "test_files", "charge_engine")
EXP_CUTOFF = 40.0  # charge_engine.EXP_CUTOFF


@njit(nogil=True, cache=False)
def _ref_primitive_block(pts, atcoords, groups, g_center, g_alpha, g_start, g_end, lmn, ncols):
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


@njit(nogil=True, cache=False)
def _ref_primitive_block_derivs(pts, atcoords, groups, g_center, g_alpha, g_start, g_end, lmn, ncols, nderiv=2):
    """_primitive_block plus first and second derivatives: out[q, point, column]
    with q = value, d/dx, d/dy, d/dz, d2/dx2, d2/dy2, d2/dz2, d2/dxdy, d2/dxdz,
    d2/dydz. For x^l exp(-a r^2) the 1D factors are f0 = x^l,
    f1 = l x^(l-1) - 2a x^(l+1), f2 = l(l-1) x^(l-2) - 2a(2l+1) x^l + 4a^2 x^(l+2).
    nderiv=1 returns only the first four rows (same values)."""
    n = pts.shape[0]
    out = np.zeros((4 if nderiv == 1 else 10, n, ncols))
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
                    f[k, 0] = pk
                    f[k, 1] = lk * pkm1 - 2.0 * alpha * pk * u
                    if nderiv == 1:
                        continue
                    pkm2 = 0.0
                    if lk >= 2:
                        pkm2 = 1.0
                        for _ in range(lk - 2):
                            pkm2 *= u
                    f[k, 2] = lk * (lk - 1) * pkm2 - 2.0 * alpha * (2 * lk + 1) * pk + 4.0 * alpha * alpha * pk * u * u
                out[0, ip, col] = f[0, 0] * f[1, 0] * f[2, 0] * e
                out[1, ip, col] = f[0, 1] * f[1, 0] * f[2, 0] * e
                out[2, ip, col] = f[0, 0] * f[1, 1] * f[2, 0] * e
                out[3, ip, col] = f[0, 0] * f[1, 0] * f[2, 1] * e
                if nderiv == 1:
                    col += 1
                    continue
                out[4, ip, col] = f[0, 2] * f[1, 0] * f[2, 0] * e
                out[5, ip, col] = f[0, 0] * f[1, 2] * f[2, 0] * e
                out[6, ip, col] = f[0, 0] * f[1, 0] * f[2, 2] * e
                out[7, ip, col] = f[0, 1] * f[1, 1] * f[2, 0] * e
                out[8, ip, col] = f[0, 1] * f[1, 0] * f[2, 1] * e
                out[9, ip, col] = f[0, 0] * f[1, 1] * f[2, 1] * e
                col += 1
    return out


@njit(parallel=True, cache=False)
def _ref_becke(pts, atcoords, rinv, aij, tol):
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


def _system(name):
    from qtaim_gen.source.core.charge_engine import prepare_basis, read_wfx

    wfx = read_wfx(os.path.join(FIXTURES, name, "orca.wfx"))
    return wfx, prepare_basis(wfx)


def _points(coords):
    """Atom-grid points around the first atom, the nuclei, midpoints of atom pairs (equal distances,
    so tied atoms in the Becke ordering) and far points where every primitive is cut."""
    from qtaim_gen.source.core.charge_engine import multiwfn_atom_grid

    grid, _ = multiwfn_atom_grid(45, 170, 10.0)
    mid = (coords[:, None, :] + coords[None, :, :]).reshape(-1, 3) / 2
    far = coords.mean(axis=0) + np.array([[25.0, 0.0, 0.0], [0.0, -40.0, 3.0]])
    return np.vstack([grid + coords[0], coords, mid, far])


def _blocks(pts, coords, basis, size):
    from qtaim_gen.source.core.charge_engine import _morton_order

    gs, ge, ga, gc = basis["g_start"], basis["g_end"], basis["g_alpha"], basis["g_center"]
    order = _morton_order(pts)
    sp = np.ascontiguousarray(pts[order])
    for s0 in range(0, len(sp), size):
        blk = sp[s0 : s0 + size]
        lo, hi = blk.min(axis=0), blk.max(axis=0)
        d2 = (np.maximum(0.0, np.maximum(lo - coords, coords - hi)) ** 2).sum(axis=1)
        groups = np.flatnonzero(ga * d2[gc] <= EXP_CUTOFF)
        yield blk, groups, int((ge - gs)[groups].sum())


def _same_bits(a, b):
    return np.array_equal(a, b) and np.array_equal(np.signbit(a), np.signbit(b))


@pytest.mark.parametrize("name", ["XeCl4_step2_0_1", "rmechdb_652_step10_0_2"])
def test_strip_kernels_give_the_point_by_point_bits(name):
    from qtaim_gen.source.core import charge_engine as ce

    wfx, basis = _system(name)
    coords = wfx["coords"]
    args = (basis["g_center"], basis["g_alpha"], basis["g_start"], basis["g_end"], basis["lmn"])
    pts = _points(coords)
    for blk, groups, nc in _blocks(pts, coords, basis, 512):
        assert _same_bits(ce._primitive_block(blk, coords, groups, *args, nc),
                          _ref_primitive_block(blk, coords, groups, *args, nc))
    for blk, groups, nc in _blocks(pts, coords, basis, 32):
        for nd in (1, 2):
            assert _same_bits(ce._primitive_block_derivs(blk, coords, groups, *args, nc, nd),
                              _ref_primitive_block_derivs(blk, coords, groups, *args, nc, nd))


@pytest.mark.parametrize("name", ["XeCl4_step2_0_1", "rmechdb_652_step10_0_2"])
def test_two_phase_becke_gives_the_reference_bits(name):
    from qtaim_gen.source.core import charge_engine as ce

    wfx, _ = _system(name)
    coords, atnums = wfx["coords"], wfx["atnums"]
    rinv, aij = ce.becke_pair_tables(atnums, coords)
    pts = _points(coords)
    for tol in (ce.BECKE_TOL, 1e-3, 0.0):
        assert _same_bits(ce._becke(pts, coords, rinv, aij, tol), _ref_becke(pts, coords, rinv, aij, tol))


def test_orbital_values_cts_are_the_separate_products():
    from qtaim_gen.source.core.charge_engine import orbital_values

    wfx, basis = _system("XeCl4_step2_0_1")
    pts = _points(wfx["coords"])
    other = np.random.default_rng(0).normal(size=(basis["ct"].shape[0], 7))
    phi, extra = orbital_values(pts, wfx["coords"], basis, cts=(other,))
    assert _same_bits(phi, orbital_values(pts, wfx["coords"], basis))
    assert _same_bits(extra, orbital_values(pts, wfx["coords"], {**basis, "ct": other}))


def test_density_derivatives_phi2_and_cts():
    from qtaim_gen.source.core.charge_engine import orbital_values
    from qtaim_gen.source.core.qtaim_engine import density_derivatives

    wfx, basis = _system("rmechdb_652_step10_0_2")
    pts = _points(wfx["coords"])
    other = np.random.default_rng(1).normal(size=(basis["ct"].shape[0], 5))
    for kw in ({"elf_only": True}, {"hessian": True}):
        d = density_derivatives(pts, wfx, basis, phi2=True, cts=(other,), **kw)
        phi, extra = orbital_values(pts, wfx["coords"], basis, cts=(other,))
        # the derivative kernel forms the same values in another order: roundoff against the scale
        for got, want in ((d["phi2"], phi**2), (d["cts"][0], extra), (d["phi2"] @ basis["occ"], d["rho_mo"])):
            assert np.abs(got - want).max() <= 1e-14 * np.abs(want).max()
        plain = density_derivatives(pts, wfx, basis, **kw)
        for k in plain:
            assert np.array_equal(d[k], plain[k]), k
    with pytest.raises(ValueError):
        density_derivatives(pts, wfx, basis, hessian=False, phi2=True)


def _ref_morton_order(pts, cell=1.0):
    q = np.floor((pts - pts.min(axis=0)) / cell).astype(np.int64)
    key = np.zeros(len(pts), dtype=np.int64)
    for bit in range(20):
        for k in range(3):
            key |= ((q[:, k] >> bit) & 1) << (3 * bit + k)
    return np.argsort(key, kind="stable")


def test_morton_order_matches_the_numpy_key():
    from qtaim_gen.source.core.charge_engine import _morton_order

    rng = np.random.default_rng(3)
    for n, scale in ((0, 1.0), (1, 1.0), (2, 1.0), (33, 0.2), (1000, 5.0), (5000, 3e6)):
        pts = rng.normal(scale=scale, size=(n, 3))
        if n > 4:
            pts[n // 2:] = np.round(pts[n // 2:])  # ties on the lattice
        if n == 0:
            assert len(_morton_order(pts)) == 0
            continue
        assert np.array_equal(_morton_order(pts), _ref_morton_order(pts)), (n, scale)
