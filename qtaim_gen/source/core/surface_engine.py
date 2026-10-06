"""ALIE molecular-surface analysis following Multiwfn 3.8 surfana, as run by the
production other_alie step ("12\\n2\\n2\\n0\\n-1\\n-1\\nq\\n": main function 12,
mapped function 2 = average local ionization energy, default rho = 0.001 a.u.
isosurface).

Reproduces, in Multiwfn's order:
  - grid box: atoms +- 1.7 * vdW radius, 0.2 Bohr spacing, rho incl. EDF at
    D20.13-rounded coordinates (savecubmat)
  - marching tetrahedra (marchtetra/genvertex: main-axis decomposition, six
    tetrahedra sharing the 4-3 diagonal), one vertex per cut edge placed by three
    bisections plus linear interpolation (vertexinterpolate), and the enclosed
    volume from the pre-merge vertices
  - vertex elimination: greedy merge within 0.1 Bohr, then removal of vertices
    with two and then three live neighbours, all serial in index order
  - ALIE = sum n_i |e_i| phi_i^2 / sum n_i phi_i^2 at the surviving vertices (MOs
    only, no EDF), area-weighted facet statistics

and returns the eight values the production parser keeps (other.json ALIE_*).

    surface-engine --wfx orca.wfx --out alie.json
"""

import argparse
import json
import os
import sys
import tempfile
import time

import numpy as np
from numba import njit, prange

from qtaim_gen.source.core.charge_engine import B2A, _edf, prepare_basis, read_wfx
from qtaim_gen.source.data import multiwfn_tables as T

ISOVALUE = 0.001  # surfisoval
SPACING = 0.2  # grdspc for ALIE
CRITMERGE = SPACING * 0.5  # grdspc * spcmergeratio
VDWMULTI = 1.7
NBISEC = 3
AU2EV = 27.2113838  # Multiwfn's au2eV
AVOGADRO = 6.02214179e23  # Multiwfn's avogacst

# getvertind: cube corner 1..8 -> (dx, dy, dz) offset from the cube origin
CORNER_OFFSET = np.array(
    [(0, 1, 0), (1, 1, 0), (0, 0, 0), (1, 1, 1), (1, 0, 0), (1, 0, 1), (0, 0, 1), (0, 1, 1)],
    dtype=np.int64,
)
# marchtetra: the six tetrahedra of the main-axis decomposition (1-based corners)
TETRAHEDRA = np.array(
    [(4, 3, 5, 2), (4, 3, 5, 6), (4, 3, 7, 6), (4, 3, 7, 8), (4, 3, 1, 8), (4, 3, 1, 2)],
    dtype=np.int64,
) - 1


# ---------------------------------------------------------------- functions


@njit(parallel=True, cache=True)
def _rho_alie(pts, atcoords, a_gstart, a_gend, a_r2max, g_alpha, g_start, g_end, lmn, ct, occ, wene, with_alie):
    """MO density sum n_i phi_i^2 and, with with_alie, the ALIE numerator
    sum |e_i| n_i phi_i^2 (avglocion), point by point without storing the MO
    matrix. Orbitals are evaluated as charge_engine._orbitals does."""
    npts = pts.shape[0]
    nat = atcoords.shape[0]
    nmo = ct.shape[1]
    rho = np.zeros(npts)
    num = np.zeros(npts if with_alie else 0)
    nblk = (npts + 63) // 64
    for blk in prange(nblk):
        phi = np.empty(nmo)
        for ip in range(blk * 64, min(npts, (blk + 1) * 64)):
            x, y, z = pts[ip, 0], pts[ip, 1], pts[ip, 2]
            for i in range(nmo):
                phi[i] = 0.0
            for a in range(nat):
                dx = x - atcoords[a, 0]
                dy = y - atcoords[a, 1]
                dz = z - atcoords[a, 2]
                r2 = dx * dx + dy * dy + dz * dz
                if r2 > a_r2max[a]:
                    continue
                for g in range(a_gstart[a], a_gend[a]):
                    ar2 = g_alpha[g] * r2
                    if ar2 > 40.0:
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
                        for i in range(nmo):
                            phi[i] += ct[p, i] * v
            s = 0.0
            t = 0.0
            for i in range(nmo):
                p2 = phi[i] * phi[i]
                s += occ[i] * p2
                if with_alie:
                    t += wene[i] * p2
            rho[ip] = s
            if with_alie:
                num[ip] = t
    return rho, num


def _evaluate(pts, coords, basis, wfx, with_alie=False):
    """fdens (MO density + EDF) and, optionally, the ALIE numerator."""
    wene = basis["abs_energy"] * basis["occ"] if with_alie else np.zeros(0)
    rho, num = _rho_alie(
        np.ascontiguousarray(pts), coords, basis["a_gstart"], basis["a_gend"], basis["a_r2max"],
        basis["g_alpha"], basis["g_start"], basis["g_end"], basis["lmn"], basis["ct"],
        basis["occ"], wene, with_alie,
    )
    rho_mo = rho
    if len(wfx["edf_center"]):
        rho = rho + _edf(np.ascontiguousarray(pts), coords, wfx["edf_center"], wfx["edf_exp"], wfx["edf_coef"])
    return rho, rho_mo, num


# ---------------------------------------------------------------- polygonization


@njit(cache=True)
def _tetravol(ax, ay, az, bx, by, bz, cx, cy, cz, dx_, dy_, dz_):
    """gettetravol: |(a-d).((b-d)x(c-d))|/6, with vecprod's operation order."""
    v1x = ax - dx_
    v1y = ay - dy_
    v1z = az - dz_
    v2x = bx - dx_
    v2y = by - dy_
    v2z = bz - dz_
    v3x = cx - dx_
    v3y = cy - dy_
    v3z = cz - dz_
    px = v2y * v3z - v2z * v3y
    py = -(v2x * v3z - v2z * v3x)
    pz = v2x * v3y - v2y * v3x
    return abs(v1x * px + v1y * py + v1z * pz) / 6.0


@njit(cache=True)
def _add_conn(conn, cpos, i, j):
    """addvtxconn: append j to i and i to j unless i already lists j. Returns
    False when a connection list is full."""
    for k in range(cpos[i]):
        if conn[i, k] == j:
            return True
    if cpos[i] >= conn.shape[1] or cpos[j] >= conn.shape[1]:
        return False
    conn[i, cpos[i]] = j
    cpos[i] += 1
    conn[j, cpos[j]] = i
    cpos[j] += 1
    return True


@njit(cache=True)
def _march(val, iso, org, d, offset, tetra, pos, width):
    """Marching tetrahedra over the boundary cubes in Multiwfn's loop order.

    Builds the vertex list (each as the ordered corner pair vertexinterpolate
    was first called with), facets and vertex connections. Positions are not
    needed for that topology; when pos holds them (second call), also returns the
    enclosed volume accumulated per tetrahedron type as genvertex does.
    Returns ok = False when a connection list overflows `width`."""
    nx, ny, nz = val.shape
    use_pos = pos.shape[0] > 0
    # boundary cubes and their corners (corpos: inside when >= iso)
    inside = val >= iso
    cid = np.full(nx * ny * nz, -1, dtype=np.int64)
    nint = 0
    ncor = 0
    nbnd = 0
    for ix in range(nx - 1):
        for iy in range(ny - 1):
            for iz in range(nz - 1):
                n_in = 0
                for c in range(8):
                    if inside[ix + offset[c, 0], iy + offset[c, 1], iz + offset[c, 2]]:
                        n_in += 1
                if n_in == 8:
                    nint += 1
                elif n_in > 0:
                    nbnd += 1
                    for c in range(8):
                        lin = ((ix + offset[c, 0]) * ny + iy + offset[c, 1]) * nz + iz + offset[c, 2]
                        if cid[lin] < 0:
                            cid[lin] = ncor
                            ncor += 1
    cache = np.full((ncor, 7), -1, dtype=np.int64)
    cap = 7 * nbnd + 16
    vtx_a = np.empty(cap, dtype=np.int64)
    vtx_b = np.empty(cap, dtype=np.int64)
    tri = np.empty((12 * nbnd + 16, 3), dtype=np.int64)
    conn = np.zeros((cap, width), dtype=np.int64)
    cpos = np.zeros(cap, dtype=np.int64)
    nv = 0
    nt = 0
    vol = np.zeros(4)
    ok = True
    vix = np.empty(4, dtype=np.int64)
    viy = np.empty(4, dtype=np.int64)
    viz = np.empty(4, dtype=np.int64)
    itest = np.empty(4, dtype=np.int64)
    newv = np.empty(4, dtype=np.int64)
    pa = np.empty(4, dtype=np.int64)
    pb = np.empty(4, dtype=np.int64)
    for ix in range(nx - 1):
        for iy in range(ny - 1):
            for iz in range(nz - 1):
                n_in = 0
                for c in range(8):
                    if inside[ix + offset[c, 0], iy + offset[c, 1], iz + offset[c, 2]]:
                        n_in += 1
                if n_in == 0 or n_in == 8:
                    continue
                for t in range(6):
                    tot = 0
                    for k in range(4):
                        c = tetra[t, k]
                        vix[k] = ix + offset[c, 0]
                        viy[k] = iy + offset[c, 1]
                        viz[k] = iz + offset[c, 2]
                        itest[k] = 1 if val[vix[k], viy[k], viz[k]] > iso else 0
                        tot += itest[k]
                    if tot == 0:
                        continue
                    if tot == 4:
                        if use_pos:
                            vol[0] += _tetravol(
                                org[0] + vix[0] * d, org[1] + viy[0] * d, org[2] + viz[0] * d,
                                org[0] + vix[1] * d, org[1] + viy[1] * d, org[2] + viz[1] * d,
                                org[0] + vix[2] * d, org[1] + viy[2] * d, org[2] + viz[2] * d,
                                org[0] + vix[3] * d, org[1] + viy[3] * d, org[2] + viz[3] * d,
                            )
                        continue
                    # cut edges in genvertex's call order; the first corner of each
                    # pair is vertexinterpolate's "a" (inside for 1 and 2 inside
                    # corners, the outside one for 3)
                    nnew = 0
                    ione = 0
                    if tot == 1 or tot == 3:
                        want = 1 if tot == 1 else 0
                        for k in range(4):
                            if itest[k] == want:
                                ione = k
                                break
                        for k in range(4):
                            if k != ione:
                                pa[nnew] = ione
                                pb[nnew] = k
                                nnew += 1
                    else:
                        for k in range(4):
                            if itest[k] == 1:
                                for o in range(4):
                                    if itest[o] == 0:
                                        pa[nnew] = k
                                        pb[nnew] = o
                                        nnew += 1
                    # vertexinterpolate for each cut edge, in call order
                    for m in range(nnew):
                        ka = pa[m]
                        kb = pb[m]
                        la_ = (vix[ka] * ny + viy[ka]) * nz + viz[ka]
                        lb_ = (vix[kb] * ny + viy[kb]) * nz + viz[kb]
                        if la_ < lb_:
                            lo = la_
                            code = (vix[kb] - vix[ka]) * 4 + (viy[kb] - viy[ka]) * 2 + (viz[kb] - viz[ka]) - 1
                        else:
                            lo = lb_
                            code = (vix[ka] - vix[kb]) * 4 + (viy[ka] - viy[kb]) * 2 + (viz[ka] - viz[kb]) - 1
                        slot = cid[lo]
                        if cache[slot, code] >= 0:
                            newv[m] = cache[slot, code]
                        else:
                            if nv >= cap:
                                return vtx_a[:0], vtx_b[:0], tri[:0], conn[:0], cpos[:0], vol, nint, False
                            vtx_a[nv] = la_
                            vtx_b[nv] = lb_
                            cache[slot, code] = nv
                            newv[m] = nv
                            nv += 1
                    if tot == 1 or tot == 3:
                        ok = ok and _add_conn(conn, cpos, newv[0], newv[1])
                        ok = ok and _add_conn(conn, cpos, newv[0], newv[2])
                        ok = ok and _add_conn(conn, cpos, newv[1], newv[2])
                        tri[nt, 0] = newv[0]
                        tri[nt, 1] = newv[1]
                        tri[nt, 2] = newv[2]
                        nt += 1
                        if use_pos:
                            k0 = ione
                            cx_ = org[0] + vix[k0] * d
                            cy_ = org[1] + viy[k0] * d
                            cz_ = org[2] + viz[k0] * d
                            p1 = pos[newv[0]]
                            p2 = pos[newv[1]]
                            p3 = pos[newv[2]]
                            v = _tetravol(cx_, cy_, cz_, p1[0], p1[1], p1[2], p2[0], p2[1], p2[2], p3[0], p3[1], p3[2])
                            if tot == 1:
                                vol[1] += v
                            else:
                                whole = _tetravol(
                                    org[0] + vix[0] * d, org[1] + viy[0] * d, org[2] + viz[0] * d,
                                    org[0] + vix[1] * d, org[1] + viy[1] * d, org[2] + viz[1] * d,
                                    org[0] + vix[2] * d, org[1] + viy[2] * d, org[2] + viz[2] * d,
                                    org[0] + vix[3] * d, org[1] + viy[3] * d, org[2] + viz[3] * d,
                                )
                                vol[2] += whole - v
                    else:
                        ok = ok and _add_conn(conn, cpos, newv[0], newv[1])
                        ok = ok and _add_conn(conn, cpos, newv[0], newv[2])
                        ok = ok and _add_conn(conn, cpos, newv[1], newv[3])
                        ok = ok and _add_conn(conn, cpos, newv[2], newv[3])
                        ok = ok and _add_conn(conn, cpos, newv[0], newv[3])
                        tri[nt, 0] = newv[0]
                        tri[nt, 1] = newv[1]
                        tri[nt, 2] = newv[3]
                        nt += 1
                        tri[nt, 0] = newv[0]
                        tri[nt, 1] = newv[2]
                        tri[nt, 2] = newv[3]
                        nt += 1
                        if use_pos:
                            t1 = pa[0]
                            t2 = pa[2]
                            t1x = org[0] + vix[t1] * d
                            t1y = org[1] + viy[t1] * d
                            t1z = org[2] + viz[t1] * d
                            t2x = org[0] + vix[t2] * d
                            t2y = org[1] + viy[t2] * d
                            t2z = org[2] + viz[t2] * d
                            s1 = pos[newv[0]]
                            s2 = pos[newv[1]]
                            s3 = pos[newv[2]]
                            s4 = pos[newv[3]]
                            vol[3] += _tetravol(t1x, t1y, t1z, s1[0], s1[1], s1[2], s3[0], s3[1], s3[2], s4[0], s4[1], s4[2])
                            vol[3] += _tetravol(t1x, t1y, t1z, t2x, t2y, t2z, s1[0], s1[1], s1[2], s4[0], s4[1], s4[2])
                            vol[3] += _tetravol(t2x, t2y, t2z, s1[0], s1[1], s1[2], s2[0], s2[1], s2[2], s4[0], s4[1], s4[2])
    return vtx_a[:nv], vtx_b[:nv], tri[:nt], conn[:nv], cpos[:nv], vol, nint, ok


@njit(cache=True)
def _eliminate(pos, conn, cpos, tri, crit2):
    """Eliminate redundant vertices exactly as surfana (ifelim=1), serial in
    index order: merge each vertex's neighbours closer than critmerge into it
    (moving it to the midpoint; the neighbour loop's trip count is fixed when
    it starts, as in Fortran), remap facets through the merge chain and drop
    degenerate ones, remove vertices with exactly two live neighbours, then
    those with exactly three (adding the triangle of those neighbours), and drop
    facets touching any removed vertex. pos and conn are modified in place."""
    nv = pos.shape[0]
    width = conn.shape[1]
    elim = np.zeros(nv, dtype=np.int64)
    merge = np.arange(nv)
    ok = True
    for i1 in range(nv):
        if elim[i1] == 1:
            continue
        x1 = pos[i1, 0]
        y1 = pos[i1, 1]
        z1 = pos[i1, 2]
        n1 = cpos[i1]
        for j in range(n1):
            i2 = conn[i1, j]
            if elim[i2] == 1:
                continue
            x2 = pos[i2, 0]
            y2 = pos[i2, 1]
            z2 = pos[i2, 2]
            d2 = (x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2
            if d2 < crit2:
                n2 = cpos[i2]
                for k in range(n2):
                    inei = conn[i2, k]
                    if elim[inei] == 1 or inei == i1:
                        continue
                    has = False
                    for t in range(cpos[inei]):
                        nn = conn[inei, t]
                        if elim[nn] == 1:
                            continue
                        if nn == i1:
                            has = True
                    if not has:
                        if cpos[inei] >= width or cpos[i1] >= width:
                            ok = False
                            continue
                        conn[inei, cpos[inei]] = i1
                        cpos[inei] += 1
                        conn[i1, cpos[i1]] = inei
                        cpos[i1] += 1
                merge[i2] = i1
                elim[i2] = 1
                pos[i1, 0] = (x1 + x2) / 2.0
                pos[i1, 1] = (y1 + y2) / 2.0
                pos[i1, 2] = (z1 + z2) / 2.0
                x1 = pos[i1, 0]
                y1 = pos[i1, 1]
                z1 = pos[i1, 2]

    ntri = tri.shape[0]
    alltri = np.empty((ntri + nv, 3), dtype=np.int64)
    alltri[:ntri] = tri
    elimtri = np.zeros(ntri + nv, dtype=np.int64)
    for it in range(ntri):
        for m in range(3):
            while elim[alltri[it, m]] == 1:
                alltri[it, m] = merge[alltri[it, m]]
        if alltri[it, 0] == alltri[it, 1] or alltri[it, 0] == alltri[it, 2] or alltri[it, 1] == alltri[it, 2]:
            elimtri[it] = 1

    for i1 in range(nv):
        if elim[i1] == 1:
            continue
        nlink = 0
        for j in range(cpos[i1]):
            if elim[conn[i1, j]] == 0:
                nlink += 1
            if nlink > 2:
                break
        if nlink == 2:
            elim[i1] = 1
    nt = ntri
    three = np.empty(3, dtype=np.int64)
    for i1 in range(nv):
        if elim[i1] == 1:
            continue
        nlink = 0
        for j in range(cpos[i1]):
            i2 = conn[i1, j]
            if elim[i2] == 0:
                nlink += 1
                if nlink > 3:
                    break
                three[nlink - 1] = i2
        if nlink == 3:
            elim[i1] = 1
            alltri[nt] = three
            elimtri[nt] = 0
            nt += 1
    for it in range(nt):
        if elimtri[it] == 1:
            continue
        for m in range(3):
            if elim[alltri[it, m]] == 1:
                elimtri[it] = 1
                break
    return elim, alltri[:nt], elimtri[:nt], ok


def _bisect_vertices(vtx_a, vtx_b, val, org, coords, basis, wfx, shape):
    """vertexinterpolate with nbisec=3: bisect the corner pair (a = first
    argument) three times on fdens, then interpolate linearly on the last
    bracket. Corner positions are org + index * spacing (not the rounded grid
    coordinates the corner values came from), as in Multiwfn."""
    _, ny, nz = shape
    ia = np.stack([vtx_a // (ny * nz), (vtx_a // nz) % ny, vtx_a % nz], axis=1)
    ib = np.stack([vtx_b // (ny * nz), (vtx_b // nz) % ny, vtx_b % nz], axis=1)
    a = org + ia * SPACING
    b = org + ib * SPACING
    flat = val.ravel()
    va = flat[vtx_a].copy()
    vb = flat[vtx_b].copy()
    for _ in range(NBISEC):
        mid = (a + b) / 2.0
        vm, _, _ = _evaluate(mid, coords, basis, wfx)
        to_b = (vm - ISOVALUE) * (va - ISOVALUE) < 0
        b[to_b] = mid[to_b]
        vb[to_b] = vm[to_b]
        a[~to_b] = mid[~to_b]
        va[~to_b] = vm[~to_b]
    with np.errstate(divide="ignore", invalid="ignore"):
        use_a = (vm - ISOVALUE) * (va - ISOVALUE) < 0
        scl_a = (va - ISOVALUE) / (va - vm)
        scl_b = (vb - ISOVALUE) / (vb - vm)
        # both branches are formed; the one np.where discards can be 0/0
        pa = (1 - scl_a[:, None]) * a + scl_a[:, None] * mid
        pb = (1 - scl_b[:, None]) * b + scl_b[:, None] * mid
    return np.where(use_a[:, None], pa, pb)


def _d20_13(v):
    """savecubmat writes each grid coordinate with D20.13 and reads it back."""
    return float(f"{v:.12E}")


def alie_surface(wfx_path, chunk_points=262144):
    t = {}
    t0 = time.perf_counter()
    wfx = read_wfx(wfx_path)
    if wfx["mo_energy"] is None:
        raise ValueError("wfx has no <Molecular Orbital Energies>")
    basis = prepare_basis(wfx)
    coords = wfx["coords"]
    atnums = wfx["atnums"]
    t["load"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    vdw = np.array([T.VDWR[z] for z in atnums]) / B2A
    org = (coords - VDWMULTI * vdw[:, None]).min(axis=0)
    end = (coords + VDWMULTI * vdw[:, None]).max(axis=0)
    shape = tuple(int(np.floor((end[k] - org[k]) / SPACING + 0.5)) + 1 for k in range(3))
    axes = [np.array([_d20_13(org[k] + i * SPACING) for i in range(shape[k])]) for k in range(3)]
    val = np.empty(shape)
    slab = max(1, chunk_points // (shape[1] * shape[2]))
    yz = np.stack(np.meshgrid(axes[1], axes[2], indexing="ij"), axis=-1).reshape(-1, 2)
    for i0 in range(0, shape[0], slab):
        xs = axes[0][i0 : i0 + slab]
        pts = np.empty((len(xs) * len(yz), 3))
        pts[:, 0] = np.repeat(xs, len(yz))
        pts[:, 1:] = np.tile(yz, (len(xs), 1))
        rho, _, _ = _evaluate(pts, coords, basis, wfx)
        val[i0 : i0 + len(xs)] = rho.reshape(len(xs), shape[1], shape[2])
    t["grid"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    width = 32
    while True:
        vtx_a, vtx_b, tri, conn, cpos, _, nint, ok = _march(
            val, ISOVALUE, org, SPACING, CORNER_OFFSET, TETRAHEDRA, np.zeros((0, 3)), width
        )
        if ok:
            break
        width *= 2
    t["polygonize"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    pos = _bisect_vertices(vtx_a, vtx_b, val, org, coords, basis, wfx, shape)
    t["bisect"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    *_, vol, _, _ = _march(val, ISOVALUE, org, SPACING, CORNER_OFFSET, TETRAHEDRA, pos, conn.shape[1])
    totvol = vol[0] + vol[1] + vol[2] + vol[3] + nint * SPACING**3
    n_before = (len(pos), int(cpos.sum()) // 2, len(tri))

    while True:
        conn_w = np.zeros((len(pos), width), dtype=np.int64)
        conn_w[:, : conn.shape[1]] = conn
        pos_m = pos.copy()
        cpos_m = cpos.copy()
        elim, alltri, elimtri, ok = _eliminate(pos_m, conn_w, cpos_m, tri, CRITMERGE**2)
        if ok:
            break
        width *= 2
    live_tri = alltri[elimtri == 0]
    t["eliminate"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    p1, p2, p3 = pos_m[live_tri[:, 0]], pos_m[live_tri[:, 1]], pos_m[live_tri[:, 2]]
    va, vb = p2 - p1, p3 - p1
    cross = np.stack([
        va[:, 1] * vb[:, 2] - va[:, 2] * vb[:, 1],
        -(va[:, 0] * vb[:, 2] - va[:, 2] * vb[:, 0]),
        va[:, 0] * vb[:, 1] - va[:, 1] * vb[:, 0],
    ], axis=1)
    area = 0.5 * np.sqrt(cross[:, 0] ** 2 + cross[:, 1] ** 2 + cross[:, 2] ** 2)

    live = np.flatnonzero(elim == 0)
    _, rho_mo, num = _evaluate(pos_m[live], coords, basis, wfx, with_alie=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        alie_live = np.where(rho_mo == 0.0, 0.0, num / rho_mo)
    alie = np.zeros(len(pos_m))
    alie[live] = alie_live
    t["alie"] = time.perf_counter() - t0

    # statistics (first live vertex wins ties, as the strict comparisons do)
    vmin = alie_live[np.argmin(alie_live)]
    vmax = alie_live[np.argmax(alie_live)]
    fval = (alie[live_tri[:, 0]] + alie[live_tri[:, 1]] + alie[live_tri[:, 2]]) / 3.0
    area_all = area.sum()
    pos_f = fval >= 0
    area_pos = area[pos_f].sum()
    area_neg = area[~pos_f].sum()
    sum_pos = (fval[pos_f] * area[pos_f]).sum()
    sum_neg = (fval[~pos_f] * area[~pos_f]).sum()
    avg_all = (sum_pos + sum_neg) / area_all
    avg_pos = sum_pos / area_pos if area_pos else 0.0
    avg_neg = sum_neg / area_neg if area_neg else 0.0
    var_pos = (area[pos_f] * (fval[pos_f] - avg_pos) ** 2).sum() / area_pos if area_pos else 0.0
    var_neg = (area[~pos_f] * (fval[~pos_f] - avg_neg) ** 2).sum() / area_neg if area_neg else 0.0
    var_all = var_pos + var_neg
    skew_all = (area * (fval - avg_all) ** 3).sum() / area_all / var_all**1.5
    totmass = sum(T.ATMWEI[z] for z in atnums)
    density = totmass / AVOGADRO * (1e24 / (totvol * B2A**3))

    out = {
        "ALIE_Volume": float(totvol),
        "ALIE_Surface_Density": float(density),
        "ALIE_Minimal_value": float(vmin * AU2EV),
        "ALIE_Maximal_value": float(vmax * AU2EV),
        "ALIE_Overall_surface_area": float(area_all),
        "ALIE_Positive_surface_area": float(area_pos),
        "ALIE_Negative_surface_area": float(area_neg),
        "ALIE_Overall_skewness": float(skew_all),
    }
    slots = np.arange(conn_w.shape[1])[None, :] < cpos_m[live][:, None]
    e_after = ((elim[conn_w[live]] == 0) & slots).sum() // 2
    meta = {
        "grid": list(shape),
        "vef_before": list(n_before),
        "vef_after": [int(len(live)), int(e_after), int(len(live_tri))],
        "average_au": float(avg_pos),
        "variance_au2": float(var_all),
        "timings_s": {k: round(v, 3) for k, v in t.items()},
    }
    return out, meta


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--wfx", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--nthreads", type=int, default=None)
    args = p.parse_args(argv)
    if args.nthreads:
        import numba

        numba.set_num_threads(args.nthreads)
    t0 = time.perf_counter()
    out, meta = alie_surface(args.wfx)
    meta["wall_s"] = round(time.perf_counter() - t0, 3)
    out_abs = os.path.abspath(args.out)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(out_abs), prefix=os.path.basename(out_abs) + ".", suffix=".tmp")
    os.fchmod(fd, 0o644)
    with os.fdopen(fd, "w") as f:
        json.dump({"other_alie": out, "_meta": meta}, f, indent=1)
    os.replace(tmp, out_abs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
