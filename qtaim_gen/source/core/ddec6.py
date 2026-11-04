"""DDEC6 atomic population analysis, ported from Chargemol 09_26_2017 (T. A. Manz and
N. Gabaldon Limas, ddec.sourceforge.net; port approved by T. A. Manz). Nonperiodic wfx
input only. Methods: Manz and Gabaldon Limas, RSC Adv. 6 (2016) 47771 and 45727.

The port follows Chargemol's arithmetic order so the density grids, and through them
the charges, reproduce its output; parity is checked against a patched reference build
(scripts/ddec6_reference). Fortran sources cited below are in
chargemol_FORTRAN_09_26_2017/sourcecode_linux.

Implemented so far (docs/engine_roadmap/ddec6_port_plan.md, P2 map):
  K0 primitive order and single blocks   module_read_wfx.f08:730-819
  K1 uniform grid                        gen_dens_grids Step 1; charge_center_positions,
                                         parallelpiped
  K2 primitive density matrices          Step 2
  K3-K5 pair screening, data, sort       Steps 3-6
  K6-K7 point loop, coarse-grid          Step 7 (parallel: blocks, then z-planes;
        interpolation                    bit-identical to the serial order)
  K8 core grid, missing core, spacing    Step 8; add_missing_core_density;
     check                               check_grid_spacing
"""

import math

import numba
import numpy as np

from qtaim_gen.source.core.charge_engine import read_wfx

# module_global_parameters.f08
NSHELLS = 100
CUTOFF_RADIUS = 500.0  # pm
BOHRPERANGSTROM = 1.889725989
SCALEFACTOR = 100.0 * NSHELLS / (CUTOFF_RADIUS * BOHRPERANGSTROM)
PREFERRED_GRID_SPACING = 0.14
GAUSSIAN_OVERLAP_TOLERANCE = 1.0e-12
ANALYTIC_ALPHA_CUTOFF = 5.0
COARSER_GRID_ALPHA_CUTOFF = 0.4
ZERO_TOLERANCE = 1.0e-10
PI = 3.14159265358979323846

# wfx primitive type -> (Lx, Ly, Lz), types 1-56 (module_read_wfx.f08 '<primitive types>')
WFX_LXYZ = np.array(((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (2, 0, 0), (0, 2, 0), (0, 0, 2), (1, 1, 0),
                     (1, 0, 1), (0, 1, 1), (3, 0, 0), (0, 3, 0), (0, 0, 3), (2, 1, 0), (2, 0, 1), (0, 2, 1),
                     (1, 2, 0), (1, 0, 2), (0, 1, 2), (1, 1, 1), (4, 0, 0), (0, 4, 0), (0, 0, 4), (3, 1, 0),
                     (3, 0, 1), (1, 3, 0), (0, 3, 1), (1, 0, 3), (0, 1, 3), (2, 2, 0), (2, 0, 2), (0, 2, 2),
                     (2, 1, 1), (1, 2, 1), (1, 1, 2), (0, 0, 5), (0, 1, 4), (0, 2, 3), (0, 3, 2), (0, 4, 1),
                     (0, 5, 0), (1, 0, 4), (1, 1, 3), (1, 2, 2), (1, 3, 1), (1, 4, 0), (2, 0, 3), (2, 1, 2),
                     (2, 2, 1), (2, 3, 0), (3, 0, 2), (3, 1, 1), (3, 2, 0), (4, 0, 1), (4, 1, 0), (5, 0, 0)),
                    dtype=np.int64)


# ---------------------------------------------------------------------------------
# Fortran intrinsics with gfortran's semantics
# ---------------------------------------------------------------------------------

@numba.njit(cache=True)
def nint(x):
    """Fortran NINT: round half away from zero."""
    return int(math.floor(x + 0.5)) if x >= 0.0 else -int(math.floor(-x + 0.5))


@numba.njit(cache=True)
def powi(x, m):
    """real**integer as gfortran compiles it for a variable exponent (libgcc __powidf2)."""
    n = -m if m < 0 else m
    y = x if n % 2 else 1.0
    n >>= 1
    while n:
        x = x * x
        if n % 2:
            y = y * x
        n >>= 1
    return 1.0 / y if m < 0 else y


@numba.njit(cache=True)
def pow10(x):
    """x**10 with a constant exponent, as gfortran expands it at compile time (GCC's
    powi table: x^10 = x^5 * x^5, x^5 = x^3 * x^2, x^3 = x^2 * x)."""
    x2 = x * x
    x5 = (x2 * x) * x2
    return x5 * x5


@numba.njit(cache=True)
def _factorial(a):
    f = 1
    for i in range(1, a + 1):
        f = f * i
    return f


@numba.njit(cache=True)
def gaussian_integration(alpha, lx):
    """module_gaussian_functions.f08 gaussian_integration."""
    if lx % 2 == 1:
        return 0.0
    t = 1
    for i in range(1, lx + 1, 2):
        t = t * i
    return math.sqrt(PI / alpha) * t / ((2 * alpha) ** (lx / 2.0))


@numba.njit(cache=True)
def _overlap_1d(l1, l2, d1, d2, asum):
    f = 0.0
    for i in range(l1 + 1):
        for j in range(l2 + 1):
            if (i + j) % 2 == 1:
                continue
            temp = float(_factorial(l1) // (_factorial(l1 - i) * _factorial(i)))
            temp = temp * float(_factorial(l2) // (_factorial(l2 - j) * _factorial(j)))
            temp = temp * powi(d1, l1 - i) * powi(d2, l2 - j)
            f = f + temp * gaussian_integration(asum, i + j)
    return f


@numba.njit(cache=True)
def gaussian_overlap_integral(a1, lx1, ly1, lz1, x1, y1, z1, a2, lx2, ly2, lz2, x2, y2, z2):
    """module_gaussian_functions.f08 gaussian_overlap_integral (unnormalized Cartesian
    Gaussians), same operation order."""
    factor1 = math.exp(-a1 * a2 / (a1 + a2) * ((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2))
    xd = (a1 * x1 + a2 * x2) / (a1 + a2)
    yd = (a1 * y1 + a2 * y2) / (a1 + a2)
    zd = (a1 * z1 + a2 * z2) / (a1 + a2)
    asum = a1 + a2
    f2 = _overlap_1d(lx1, lx2, xd - x1, xd - x2, asum)
    f3 = _overlap_1d(ly1, ly2, yd - y1, yd - y2, asum)
    f4 = _overlap_1d(lz1, lz2, zd - z1, zd - z2, asum)
    return factor1 * f2 * f3 * f4


# ---------------------------------------------------------------------------------
# K0: wavefunction in Chargemol's primitive order
# ---------------------------------------------------------------------------------

@numba.njit(cache=True)
def _sort_order(center, exps):
    """Permutation from read_wfx's two selection sorts with swaps (by centre, then by
    exponent within each centre). module_read_wfx.f08:730-786."""
    n = center.shape[0]
    perm = np.arange(n)
    c = center.copy()
    e = exps.copy()
    for i in range(n - 1):
        min_c = c[i]
        t = i
        for j in range(i, n):
            if c[j] < min_c:
                min_c = c[j]
                t = j
        if t != i:
            c[i], c[t] = c[t], c[i]
            e[i], e[t] = e[t], e[i]
            perm[i], perm[t] = perm[t], perm[i]
    start = 0
    while start < n:
        stop = start
        while stop + 1 < n and c[stop + 1] == c[start]:
            stop += 1
        for i in range(start, stop):
            min_e = e[i]
            t = i
            for j in range(i, stop + 1):
                if e[j] < min_e:
                    min_e = e[j]
                    t = j
            if t != i:
                e[i], e[t] = e[t], e[i]
                perm[i], perm[t] = perm[t], perm[i]
        start = stop + 1
    return perm


def prepare(wfx):
    """Chargemol's view of a read_wfx dict: primitives sorted and grouped into single
    blocks, alpha/beta occupations, atoms (centres with charge > 0.5)."""
    center = wfx["prim_center"].astype(np.int64)  # 0-based
    perm = _sort_order(center, wfx["prim_exp"].astype(np.float64))
    center = center[perm]
    exps = wfx["prim_exp"][perm].astype(np.float64)
    lxyz = WFX_LXYZ[wfx["prim_type"][perm] - 1]
    coeffs = np.ascontiguousarray(wfx["coeffs"][:, perm])

    starts = []
    t_center, t_exp = -1, 0.0
    for i in range(len(exps)):
        if center[i] != t_center or abs(t_exp - exps[i]) > ZERO_TOLERANCE:
            starts.append(i)
            t_center, t_exp = center[i], exps[i]
    blocks = np.array([(s, e) for s, e in zip(starts, starts[1:] + [len(exps)])], dtype=np.int64)

    occ = np.asarray(wfx["occ"], dtype=np.float64)
    occ_a = np.zeros(len(occ))
    occ_b = np.zeros(len(occ))
    spin_available = False
    for k, (o, t) in enumerate(zip(occ, wfx["spin_types"])):
        if t == "Alpha":
            if o >= 1.01:
                raise ValueError("Alpha orbital occupancy above 1")
            occ_a[k] = o
            spin_available = True
        elif t == "Beta":
            occ_b[k] = o
        else:
            occ_a[k] = o / 2.0
            occ_b[k] = o / 2.0

    zeff = np.asarray(wfx["zeff"], dtype=np.float64)
    is_atom = zeff > 0.5
    atom_of_center = np.full(len(zeff), -1, dtype=np.int64)
    atom_of_center[is_atom] = np.arange(int(is_atom.sum()))
    return {
        "center": center, "exps": exps, "lxyz": lxyz, "coeffs": coeffs, "blocks": blocks,
        "occ_a": occ_a, "occ_b": occ_b, "spin_available": spin_available,
        "included_electrons": int(round(float(occ_a.sum() + occ_b.sum()))),
        "center_xyz": np.asarray(wfx["coords"], dtype=np.float64),
        "center_z": np.asarray(wfx["atnums"], dtype=np.float64), "center_zeff": zeff,
        "atom_of_center": atom_of_center, "atom_z": np.asarray(wfx["atnums"])[is_atom],
        "atom_xyz": np.asarray(wfx["coords"], dtype=np.float64)[is_atom],
        "core_electrons": (np.asarray(wfx["atnums"], dtype=np.float64) - zeff)[is_atom],
        "wfx": wfx,
    }


# ---------------------------------------------------------------------------------
# K1: uniform grid
# ---------------------------------------------------------------------------------

def _det3(a):
    """module_matrix_operations.f08 determinant (a[i][j] = a(i+1, j+1))."""
    return (a[0][0] * a[1][1] * a[2][2] + a[0][1] * a[1][2] * a[2][0] + a[0][2] * a[1][0] * a[2][1]
            - a[0][2] * a[1][1] * a[2][0] - a[0][1] * a[1][0] * a[2][2] - a[0][0] * a[1][2] * a[2][1])


def _inverse3(a):
    """module_matrix_operations.f08 inverse = adj(a) * (1./det(a))."""
    detr = 1.0 / _det3(a)
    adj = [[a[1][1] * a[2][2] - a[1][2] * a[2][1], a[0][2] * a[2][1] - a[0][1] * a[2][2],
            a[0][1] * a[1][2] - a[0][2] * a[1][1]],
           [a[1][2] * a[2][0] - a[1][0] * a[2][2], a[0][0] * a[2][2] - a[0][2] * a[2][0],
            a[0][2] * a[1][0] - a[0][0] * a[1][2]],
           [a[1][0] * a[2][1] - a[1][1] * a[2][0], a[0][1] * a[2][0] - a[0][0] * a[2][1],
            a[0][0] * a[1][1] - a[0][1] * a[1][0]]]
    return [[adj[i][j] * detr for j in range(3)] for i in range(3)]


def _matvec(m, v):
    """Fortran matmul(m, v) for 3x3 times 3: sum over j in order 1..3 from 0."""
    out = []
    for i in range(3):
        s = 0.0
        for j in range(3):
            s = s + m[i][j] * v[j]
        out.append(s)
    return out


def _dot(a, b):
    s = 0.0
    for i in range(3):
        s = s + a[i] * b[i]
    return s


def _nint(x):
    return math.floor(x + 0.5) if x >= 0.0 else -math.floor(-x + 0.5)


def _center_positions(atom_xyz, boundary, origin):
    """charge_center_positions: pixel index of each atom and its shift (Bohr)."""
    t = [[boundary[j][i] for j in range(3)] for i in range(3)]
    inv_b = _inverse3(t)
    nabc, shift = [], []
    for xyz in atom_xyz:
        tv = _matvec(inv_b, [xyz[k] - origin[k] for k in range(3)])
        n = [_nint(v) for v in tv]
        nabc.append(n)
        shift.append(_matvec(t, [tv[k] - n[k] for k in range(3)]))
    return inv_b, nabc, shift


def grid_setup(atom_xyz):
    """Step 1 for a nonperiodic system: boundary, origin, totnum, pixelvolume, delta_n."""
    g = PREFERRED_GRID_SPACING
    boundary = [[g, 0.0, 0.0], [0.0, g, 0.0], [0.0, 0.0, g]]
    pixelvolume = _det3(boundary)
    origin = [0.0, 0.0, 0.0]
    _, nabc, _ = _center_positions(atom_xyz, boundary, origin)
    mins = [min(n[k] for n in nabc) for k in range(3)]
    maxs = [max(n[k] for n in nabc) for k in range(3)]

    # parallelpiped
    a, b, c = boundary
    aa, ab, ac, bb, bc, cc = _dot(a, a), _dot(a, b), _dot(a, c), _dot(b, b), _dot(b, c), _dot(c, c)
    r = NSHELLS / SCALEFACTOR
    sin_ab = math.sqrt(1.0 - (ab / math.sqrt(aa * bb)) * (ab / math.sqrt(aa * bb)))
    sin_ac = math.sqrt(1.0 - (ac / math.sqrt(aa * cc)) * (ac / math.sqrt(aa * cc)))
    sin_bc = math.sqrt(1.0 - (bc / math.sqrt(bb * cc)) * (bc / math.sqrt(bb * cc)))
    delta = [int(math.ceil(max(r / sin_ab, r / sin_ac) / math.sqrt(aa)) + 1.0),
             int(math.ceil(max(r / sin_ab, r / sin_bc) / math.sqrt(bb)) + 1.0),
             int(math.ceil(max(r / sin_ac, r / sin_bc) / math.sqrt(cc)) + 1.0)]

    totnum = []
    for k in range(3):
        t = maxs[k] - mins[k] + 2 * delta[k] + 11
        totnum.append(int(t + 23.99 - math.fmod(t + 23.99, 24.0)))  # modulo of positives
        shift = mins[k] - delta[k] - 6
        origin = [origin[m] + shift * boundary[k][m] for m in range(3)]
    inv_b, nabc, shift = _center_positions(atom_xyz, boundary, origin)
    return {"boundary": np.array(boundary), "inv_boundary": np.array(inv_b), "origin": np.array(origin),
            "pixelvolume": pixelvolume, "totnum": tuple(totnum), "delta": tuple(delta),
            "center_nabc": np.array(nabc, dtype=np.int64), "center_shift": np.array(shift)}


# ---------------------------------------------------------------------------------
# K2: primitive density matrices
# ---------------------------------------------------------------------------------

@numba.njit(parallel=True, cache=True)
def density_matrices(coeffs, occ_a, occ_b):
    """Step 2: upper triangles (k >= j) of the alpha and beta primitive 1PDMs, each
    element summed over orbitals in order as (occ * c_ij) * c_ik."""
    norb, nprim = coeffs.shape
    pa = np.zeros((nprim, nprim))
    pb = np.zeros((nprim, nprim))
    for j in numba.prange(nprim):
        for k in range(j, nprim):
            sa = 0.0
            sb = 0.0
            for i in range(norb):
                sa = sa + occ_a[i] * coeffs[i, j] * coeffs[i, k]
                sb = sb + occ_b[i] * coeffs[i, j] * coeffs[i, k]
            pa[j, k] = sa
            pb[j, k] = sb
    return pa, pb


# ---------------------------------------------------------------------------------
# K3-K5: pair screening, pair data, sort (Steps 3-6)
# ---------------------------------------------------------------------------------

@numba.njit(cache=True)
def _screen(blocks, center, exps, lxyz, cxyz, czeff, pa, pb):
    """Step 3. Returns the inclusion code of every (i, j >= i) pair in loop order
    (0 excluded, 1 analytic, 2 numeric) and the counts."""
    nb = blocks.shape[0]
    n_total = 0
    for bi in range(nb):
        for bj in range(bi, nb):
            for i in range(blocks[bi, 0], blocks[bi, 1]):
                for j in range(blocks[bj, 0], blocks[bj, 1]):
                    if j >= i:
                        n_total += 1
    code = np.zeros(n_total, dtype=np.int8)
    total = 0
    n_blocks = 0
    n_analytic = 0
    n_numeric = 0
    for bi in range(nb):
        a1 = exps[blocks[bi, 0]]
        c1 = center[blocks[bi, 0]]
        x1, y1, z1 = cxyz[c1, 0], cxyz[c1, 1], cxyz[c1, 2]
        for bj in range(bi, nb):
            a2 = exps[blocks[bj, 0]]
            c2 = center[blocks[bj, 0]]
            x2, y2, z2 = cxyz[c2, 0], cxyz[c2, 1], cxyz[c2, 2]
            include_block = False
            asum = a1 + a2
            for i in range(blocks[bi, 0], blocks[bi, 1]):
                for j in range(blocks[bj, 0], blocks[bj, 1]):
                    if j < i:
                        continue
                    total += 1
                    if c1 == c2:
                        if asum > 1.0:
                            lx1 = (lxyz[i, 0] + lxyz[j, 0]) - (lxyz[i, 0] + lxyz[j, 0]) % 2
                            ly1 = (lxyz[i, 1] + lxyz[j, 1]) - (lxyz[i, 1] + lxyz[j, 1]) % 2
                            lz1 = (lxyz[i, 2] + lxyz[j, 2]) - (lxyz[i, 2] + lxyz[j, 2]) % 2
                        else:
                            lx1 = 2 * ((lxyz[i, 0] + lxyz[j, 0] + 1) // 2)
                            ly1 = 2 * ((lxyz[i, 1] + lxyz[j, 1] + 1) // 2)
                            lz1 = 2 * ((lxyz[i, 2] + lxyz[j, 2] + 1) // 2)
                        lx2 = 0
                        ly2 = 0
                        lz2 = 0
                    else:
                        if asum > 1.0:
                            lx1 = lxyz[i, 0] - lxyz[i, 0] % 2
                            ly1 = lxyz[i, 1] - lxyz[i, 1] % 2
                            lz1 = lxyz[i, 2] - lxyz[i, 2] % 2
                            lx2 = lxyz[j, 0] - lxyz[j, 0] % 2
                            ly2 = lxyz[j, 1] - lxyz[j, 1] % 2
                            lz2 = lxyz[j, 2] - lxyz[j, 2] % 2
                        else:
                            lx1 = 2 * ((lxyz[i, 0] + 1) // 2)
                            ly1 = 2 * ((lxyz[i, 1] + 1) // 2)
                            lz1 = 2 * ((lxyz[i, 2] + 1) // 2)
                            lx2 = 2 * ((lxyz[j, 0] + 1) // 2)
                            ly2 = 2 * ((lxyz[j, 1] + 1) // 2)
                            lz2 = 2 * ((lxyz[j, 2] + 1) // 2)
                    max_coeff = abs(pa[i, j]) + abs(pb[i, j])
                    test = max_coeff * gaussian_overlap_integral(a1, lx1, ly1, lz1, x1, y1, z1,
                                                                 a2, lx2, ly2, lz2, x2, y2, z2)
                    if test < GAUSSIAN_OVERLAP_TOLERANCE:
                        continue
                    include_block = True
                    if a1 * min(czeff[center[i]], 1.0) + a2 * min(czeff[center[j]], 1.0) > ANALYTIC_ALPHA_CUTOFF:
                        n_analytic += 1
                        code[total - 1] = 1
                    else:
                        n_numeric += 1
                        code[total - 1] = 2
            if include_block:
                n_blocks += 1
    return code, n_blocks, n_analytic, n_numeric


@numba.njit(cache=True)
def _pair_data(blocks, center, exps, lxyz, cxyz, czeff, atom_of_center, pa, pb, code, n_pairs, n_blocks,
               core_electrons, inv_b, boundary, origin, delta):
    """Steps 4-5: electron check over all pairs, then the 40-column row of every included
    pair (Chargemol's included_primitive_data, 0-based columns) and the analytic-pair
    corrections to core_electrons and occupancy_correction."""
    natoms = core_electrons.shape[0]
    data = np.zeros((n_pairs, 40))
    pbd = np.zeros((n_blocks, 2))
    occ = np.zeros((11, natoms))
    core = core_electrons.copy()
    check = 0.0
    nb = blocks.shape[0]
    total = 0
    for bi in range(nb):
        a1 = exps[blocks[bi, 0]]
        c1 = center[blocks[bi, 0]]
        x1, y1, z1 = cxyz[c1, 0], cxyz[c1, 1], cxyz[c1, 2]
        for bj in range(bi, nb):
            a2 = exps[blocks[bj, 0]]
            c2 = center[blocks[bj, 0]]
            x2, y2, z2 = cxyz[c2, 0], cxyz[c2, 1], cxyz[c2, 2]
            for i in range(blocks[bi, 0], blocks[bi, 1]):
                for j in range(blocks[bj, 0], blocks[bj, 1]):
                    if j < i:
                        continue
                    s = gaussian_overlap_integral(a1, lxyz[i, 0], lxyz[i, 1], lxyz[i, 2], x1, y1, z1,
                                                  a2, lxyz[j, 0], lxyz[j, 1], lxyz[j, 2], x2, y2, z2)
                    if j > i:
                        check = check + 2.0 * (pa[i, j] + pb[i, j]) * s
                    else:
                        check = check + (pa[i, j] + pb[i, j]) * s
                    total += 1
    pair = 0
    block = 0
    total = 0
    for bi in range(nb):
        a1 = exps[blocks[bi, 0]]
        c1 = center[blocks[bi, 0]]
        x1, y1, z1 = cxyz[c1, 0], cxyz[c1, 1], cxyz[c1, 2]
        for bj in range(bi, nb):
            a2 = exps[blocks[bj, 0]]
            c2 = center[blocks[bj, 0]]
            x2, y2, z2 = cxyz[c2, 0], cxyz[c2, 1], cxyz[c2, 2]
            asum = a1 + a2
            in_block = 0
            for i in range(blocks[bi, 0], blocks[bi, 1]):
                lx1, ly1, lz1 = lxyz[i, 0], lxyz[i, 1], lxyz[i, 2]
                for j in range(blocks[bj, 0], blocks[bj, 1]):
                    if j < i:
                        continue
                    total += 1
                    if code[total - 1] == 0:
                        continue
                    pair += 1
                    in_block += 1
                    p = pair - 1
                    if in_block == 1:
                        block += 1
                        pbd[block - 1, 0] = pair
                    lx2, ly2, lz2 = lxyz[j, 0], lxyz[j, 1], lxyz[j, 2]
                    if i == j:
                        data[p, 2] = pa[i, j] + pb[i, j]
                        data[p, 3] = pa[i, j] - pb[i, j]
                    else:
                        data[p, 2] = 2.0 * (pa[i, j] + pb[i, j])
                        data[p, 3] = 2.0 * (pa[i, j] - pb[i, j])
                    s = gaussian_overlap_integral(a1, lx1, ly1, lz1, x1, y1, z1, a2, lx2, ly2, lz2, x2, y2, z2)
                    if code[total - 1] == 1:
                        zi = czeff[center[i]]
                        zj = czeff[center[j]]
                        if zi > 0.0 and zj > 0.0:
                            wi = pow10(a1) / (pow10(a1) + pow10(a2))
                            wj = pow10(a2) / (pow10(a1) + pow10(a2))
                        elif zi > 0.0:
                            wi = 1.0
                            wj = 0.0
                        elif zj > 0.0:
                            wi = 0.0
                            wj = 1.0
                        else:
                            wi = 0.5
                            wj = 0.5
                        data[p, 36] = wi
                        data[p, 37] = wj
                        full = data[p, 2]
                        for side in range(2):
                            ai = atom_of_center[center[i]] if side == 0 else atom_of_center[center[j]]
                            if ai < 0:
                                continue
                            w = wi if side == 0 else wj
                            core[ai] = core[ai] + full * s * w
                            occ[1, ai] = occ[1, ai] + data[p, 3] * s * w
                            for row, dx, dy, dz in ((2, 1, 0, 0), (3, 0, 1, 0), (4, 0, 0, 1), (5, 2, 0, 0),
                                                    (6, 0, 2, 0), (7, 0, 0, 2), (8, 1, 1, 0), (9, 1, 0, 1),
                                                    (10, 0, 1, 1)):
                                if side == 0:
                                    m = gaussian_overlap_integral(a1, lx1 + dx, ly1 + dy, lz1 + dz, x1, y1, z1,
                                                                  a2, lx2, ly2, lz2, x2, y2, z2)
                                else:
                                    m = gaussian_overlap_integral(a1, lx1, ly1, lz1, x1, y1, z1,
                                                                  a2, lx2 + dx, ly2 + dy, lz2 + dz, x2, y2, z2)
                                occ[row, ai] = occ[row, ai] - full * w * m
                    data[p, 0] = i + 1
                    data[p, 1] = j + 1
                    data[p, 4] = s
                    data[p, 5] = a1
                    data[p, 6] = lx1
                    data[p, 7] = ly1
                    data[p, 8] = lz1
                    data[p, 9] = x1
                    data[p, 10] = y1
                    data[p, 11] = z1
                    data[p, 12] = a2
                    data[p, 13] = lx2
                    data[p, 14] = ly2
                    data[p, 15] = lz2
                    data[p, 16] = x2
                    data[p, 17] = y2
                    data[p, 18] = z2
                    r12 = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)
                    radius = math.sqrt(max((-math.log(GAUSSIAN_OVERLAP_TOLERANCE) / asum
                                            - a1 * a2 * (r12 ** 2) / (asum ** 2)), 0.25))
                    for k in range(3):
                        data[p, 19 + k] = math.ceil(radius * 100 * delta[k] / (CUTOFF_RADIUS * BOHRPERANGSTROM)) + 1.0
                    xd = (a1 * x1 + a2 * x2) / asum
                    yd = (a1 * y1 + a2 * y2) / asum
                    zd = (a1 * z1 + a2 * z2) / asum
                    data[p, 22] = xd
                    data[p, 23] = yd
                    data[p, 24] = zd
                    v0 = xd - origin[0]
                    v1 = yd - origin[1]
                    v2 = zd - origin[2]
                    tv = np.empty(3)
                    for k in range(3):
                        tv[k] = 0.0 + inv_b[k, 0] * v0 + inv_b[k, 1] * v1 + inv_b[k, 2] * v2
                    nv = np.empty(3)
                    for k in range(3):
                        nv[k] = nint(tv[k])
                        data[p, 25 + k] = nv[k]
                    for k in range(3):
                        data[p, 28 + k] = (0.0 + boundary[0, k] * (tv[0] - nv[0]) + boundary[1, k] * (tv[1] - nv[1])
                                           + boundary[2, k] * (tv[2] - nv[2]))
                    data[p, 31] = 1.0
                    if code[total - 1] == 1:
                        data[p, 32] = 1.0
                    data[p, 33] = radius * radius
                    data[p, 34] = math.exp(-a1 * a2 * (r12 ** 2) / asum)
                    if data[p, 32] == 0.0:
                        if asum > COARSER_GRID_ALPHA_CUTOFF:
                            data[p, 35] = 1.0
                        elif asum > COARSER_GRID_ALPHA_CUTOFF * 4.0 / 9.0:
                            data[p, 35] = 2.0
                        elif asum > COARSER_GRID_ALPHA_CUTOFF / 4.0:
                            data[p, 35] = 3.0
                        elif asum > COARSER_GRID_ALPHA_CUTOFF / 9.0:
                            data[p, 35] = 4.0
                        elif asum > COARSER_GRID_ALPHA_CUTOFF / 16.0:
                            data[p, 35] = 6.0
                        elif asum > COARSER_GRID_ALPHA_CUTOFF / 36.0:
                            data[p, 35] = 8.0
                        else:
                            data[p, 35] = 12.0
            if in_block > 0:
                pbd[block - 1, 1] = pbd[block - 1, 0] + in_block - 1
    return data, pbd, occ, core, check


@numba.njit(cache=True)
def _sort_blocks(data, pbd):
    """Step 6: within each pair block, selection sort by cutoff radius (descending)."""
    for b in range(pbd.shape[0]):
        start = int(pbd[b, 0]) - 1
        stop = int(pbd[b, 1]) - 1
        for i in range(start, stop):
            r2 = data[i, 33]
            t = i
            for j in range(i, stop + 1):
                if data[j, 33] > r2:
                    r2 = data[j, 33]
                    t = j
            if t != i:
                row = data[i, :].copy()
                data[i, :] = data[t, :]
                data[t, :] = row


def pair_data(prep, grid):
    """Steps 2-6: the pair list Chargemol builds before its point loop."""
    pa, pb = density_matrices(prep["coeffs"], prep["occ_a"], prep["occ_b"])
    args = (prep["blocks"], prep["center"], prep["exps"], prep["lxyz"], prep["center_xyz"], prep["center_zeff"])
    code, n_blocks, n_analytic, n_numeric = _screen(*args, pa, pb)
    n_pairs = n_analytic + n_numeric
    data, pbd, occ, core, check = _pair_data(*args, prep["atom_of_center"], pa, pb, code, n_pairs, n_blocks,
                                             prep["core_electrons"], grid["inv_boundary"], grid["boundary"],
                                             grid["origin"], np.array(grid["delta"], dtype=np.float64))
    if abs(check - prep["included_electrons"]) > 0.001:
        raise ValueError(f"primitive pairs hold {check} electrons, expected {prep['included_electrons']}")
    _sort_blocks(data, pbd)
    return {"n_pairs": n_pairs, "n_analytic": n_analytic, "n_numeric": n_numeric, "n_blocks": n_blocks,
            "nsingleblocks": len(prep["blocks"]), "check_included_electrons": check,
            "included_electrons": prep["included_electrons"], "pair_block_data": pbd,
            "included_primitive_data": data, "occupancy_correction": occ, "core_electrons": core}


# ---------------------------------------------------------------------------------
# K6-K7: point loop and coarse-grid interpolation (Step 7)
# ---------------------------------------------------------------------------------

COARSE = (2, 3, 4, 6, 8, 12)


@numba.njit(cache=True)
def _coarse_slot(gs):
    for k in range(6):
        if COARSE[k] == gs:
            return k
    return -1


@numba.njit(cache=True)
def _axis_points(dn, nc, gs, totn):
    """Pixel indices (1-based) for offsets -dn..dn around centre pixel nc, and the first
    and last offset that lands on the grid (and on the coarse grid when gs > 1)."""
    pts = np.empty(2 * dn + 1, dtype=np.int64)
    lo = dn
    hi = -dn
    for n in range(-dn, dn + 1):
        k = n + nc + 1
        pts[dn + n] = k
        if k < 1 or k > totn:
            continue
        if gs > 1 and k % gs != gs - 1:
            continue
        lo = min(lo, n)
        hi = max(hi, n)
    return pts, lo, hi


@numba.njit(cache=True)
def _point_loop(data, pbd, center, atom_of_center, boundary, na_tot, nb_tot, nc_tot, pv, spin, occ,
                val, spn, core, cval, cspn):
    """Step 7 point loop over pair blocks, serial and in Chargemol's order. data columns
    31 (renormalization), 38 and 39 (numeric overlap sums) are updated in place."""
    mults = (8.0, 27.0, 64.0, 216.0, 512.0, 1728.0)
    max_corr = 0.0
    for b in range(pbd.shape[0]):
        start = int(pbd[b, 0]) - 1
        stop = int(pbd[b, 1]) - 1
        x1, y1, z1 = data[start, 9], data[start, 10], data[start, 11]
        x2, y2, z2 = data[start, 16], data[start, 17], data[start, 18]
        asum = data[start, 5] + data[start, 12]
        xd, yd, zd = data[start, 22], data[start, 23], data[start, 24]
        t1 = data[start, 22] - data[start, 28]
        t2 = data[start, 23] - data[start, 29]
        t3 = data[start, 24] - data[start, 30]
        r2cut = data[start, 33]
        common = data[start, 34]
        gs = nint(data[start, 35])
        wi = data[start, 36]
        wj = data[start, 37]
        ai = atom_of_center[center[nint(data[start, 0]) - 1]]
        aj = atom_of_center[center[nint(data[start, 1]) - 1]]
        ap, lo_a, hi_a = _axis_points(nint(data[start, 19]), nint(data[start, 25]), gs, na_tot)
        bp, lo_b, hi_b = _axis_points(nint(data[start, 20]), nint(data[start, 26]), gs, nb_tot)
        cp, lo_c, hi_c = _axis_points(nint(data[start, 21]), nint(data[start, 27]), gs, nc_tot)
        dna, dnb, dnc = nint(data[start, 19]), nint(data[start, 20]), nint(data[start, 21])
        if lo_a > hi_a or lo_b > hi_b or lo_c > hi_c:
            continue
        stride = gs if gs > 1 else 1
        slot = _coarse_slot(gs)
        for step in range(1, 3):
            if step == 2 and gs == 0:
                continue
            for nc in range(lo_c, hi_c + 1, stride):
                kc = cp[dnc + nc]
                for nb in range(lo_b, hi_b + 1, stride):
                    kb = bp[dnb + nb]
                    for na in range(lo_a, hi_a + 1, stride):
                        ka = ap[dna + na]
                        X = na * boundary[0, 0] + nb * boundary[1, 0] + nc * boundary[2, 0] + t1
                        Y = na * boundary[0, 1] + nb * boundary[1, 1] + nc * boundary[2, 1] + t2
                        Z = na * boundary[0, 2] + nb * boundary[1, 2] + nc * boundary[2, 2] + t3
                        d2 = (X - xd) ** 2 + (Y - yd) ** 2 + (Z - zd) ** 2
                        if d2 > r2cut:
                            continue
                        dx1 = X - x1
                        dy1 = Y - y1
                        dz1 = Z - z1
                        dx2 = X - x2
                        dy2 = Y - y2
                        dz2 = Z - z2
                        t4 = math.exp(-asum * d2) * common
                        t5 = t4 * pv
                        if slot >= 0:
                            t5 = t5 * mults[slot]
                        t7 = 0.0
                        t8 = 0.0
                        for p in range(start, stop + 1):
                            if d2 > data[p, 33]:
                                break
                            if gs != 0 and data[p, 31] == 0.0:
                                continue
                            t6 = (powi(dx1, nint(data[p, 6])) * powi(dy1, nint(data[p, 7])) * powi(dz1, nint(data[p, 8]))
                                  * powi(dx2, nint(data[p, 13])) * powi(dy2, nint(data[p, 14]))
                                  * powi(dz2, nint(data[p, 15])))
                            if gs != 0:
                                if step == 1:
                                    data[p, 38] = data[p, 38] + t5 * t6
                                    data[p, 39] = data[p, 39] + t5 * abs(t6)
                                else:
                                    t6 = abs(t6)
                            t7 = t7 + t6 * data[p, 31] * data[p, 2]
                            if spin:
                                t8 = t8 + t6 * data[p, 31] * data[p, 3]
                        if gs == 0:
                            core[ka - 1, kb - 1, kc - 1] = core[ka - 1, kb - 1, kc - 1] + t7 * t4
                            if spin:
                                sc = t8 * t4
                                spn[ka - 1, kb - 1, kc - 1] = spn[ka - 1, kb - 1, kc - 1] + sc
                                if ai >= 0:
                                    occ[1, ai] = occ[1, ai] - sc * pv * wi
                                if aj >= 0:
                                    occ[1, aj] = occ[1, aj] - sc * pv * wj
                        elif gs == 1:
                            val[ka - 1, kb - 1, kc - 1] = val[ka - 1, kb - 1, kc - 1] + t7 * t4
                            if spin:
                                spn[ka - 1, kb - 1, kc - 1] = spn[ka - 1, kb - 1, kc - 1] + t8 * t4
                        else:
                            ia = nint((ka + 1) / float(gs)) - 1
                            ib = nint((kb + 1) / float(gs)) - 1
                            ic = nint((kc + 1) / float(gs)) - 1
                            g = cval[slot]
                            g[ia, ib, ic] = g[ia, ib, ic] + t7 * t4
                            if spin:
                                h = cspn[slot]
                                h[ia, ib, ic] = h[ia, ib, ic] + t8 * t4
            if gs > 0 and step == 1:
                for p in range(start, stop + 1):
                    max_coeff = abs((data[p, 2] + data[p, 3]) / 2.0) + abs((data[p, 2] - data[p, 3]) / 2.0)
                    ov = data[p, 4]
                    s = data[p, 38]
                    abs_s = data[p, 39]
                    if max_coeff * abs_s > GAUSSIAN_OVERLAP_TOLERANCE:
                        data[p, 31] = max(min(((ov - s) / abs_s), 0.05), -0.05)
                        max_corr = max(abs((ov - s) / abs_s), max_corr)
                    else:
                        data[p, 31] = 0.0
    return max_corr


@numba.njit(parallel=True, cache=True)
def _halve(fine, coarse):
    """fine(k) += 1/8 * sum of the 8 coarse cells at nint((k -+ 0.5)/2) (12->6, 8->4,
    6->3, 4->2 in Step 7)."""
    n0, n1, n2 = fine.shape
    m0, m1, m2 = coarse.shape
    for c in numba.prange(1, n2 + 1):
        cl = (nint((c - 0.5) / 2.0) - 1) % m2
        cu = (nint((c + 0.5) / 2.0) - 1) % m2
        for b in range(1, n1 + 1):
            bl = (nint((b - 0.5) / 2.0) - 1) % m1
            bu = (nint((b + 0.5) / 2.0) - 1) % m1
            for a in range(1, n0 + 1):
                al = (nint((a - 0.5) / 2.0) - 1) % m0
                au = (nint((a + 0.5) / 2.0) - 1) % m0
                fine[a - 1, b - 1, c - 1] = fine[a - 1, b - 1, c - 1] + (1.0 / 8.0) * (
                    coarse[al, bl, cl] + coarse[au, bl, cl] + coarse[al, bu, cl] + coarse[al, bl, cu]
                    + coarse[au, bu, cl] + coarse[au, bl, cu] + coarse[al, bu, cu] + coarse[au, bu, cu])


@numba.njit(parallel=True, cache=True)
def _from_grid2(fine, g2):
    n0, n1, n2 = fine.shape
    m0, m1, m2 = g2.shape
    for c in numba.prange(1, n2 + 1):
        cl = (nint((c + 0.5) / 2.0) - 1) % m2
        cu = (nint((c + 1.5) / 2.0) - 1) % m2
        for b in range(1, n1 + 1):
            bl = (nint((b + 0.5) / 2.0) - 1) % m1
            bu = (nint((b + 1.5) / 2.0) - 1) % m1
            for a in range(1, n0 + 1):
                al = (nint((a + 0.5) / 2.0) - 1) % m0
                au = (nint((a + 1.5) / 2.0) - 1) % m0
                fine[a - 1, b - 1, c - 1] = fine[a - 1, b - 1, c - 1] + (1.0 / 8.0) * (
                    g2[al, bl, cl] + g2[au, bl, cl] + g2[al, bu, cl] + g2[al, bl, cu]
                    + g2[au, bu, cl] + g2[au, bl, cu] + g2[al, bu, cu] + g2[au, bu, cu])


@numba.njit(parallel=True, cache=True)
def _from_grid3(fine, g3):
    n0, n1, n2 = fine.shape
    m0, m1, m2 = g3.shape
    for c in numba.prange(1, n2 + 1):
        cl = (nint(c / 3.0) - 1) % m2
        cu = (nint((c + 2.0) / 3.0) - 1) % m2
        fc = ((c + 1) % 3) / 3.0
        for b in range(1, n1 + 1):
            bl = (nint(b / 3.0) - 1) % m1
            bu = (nint((b + 2.0) / 3.0) - 1) % m1
            fb = ((b + 1) % 3) / 3.0
            for a in range(1, n0 + 1):
                al = (nint(a / 3.0) - 1) % m0
                au = (nint((a + 2.0) / 3.0) - 1) % m0
                fa = ((a + 1) % 3) / 3.0
                fine[a - 1, b - 1, c - 1] = (fine[a - 1, b - 1, c - 1]
                                             + ((1.0 - fa) * (1.0 - fb) * (1.0 - fc)) * g3[al, bl, cl]
                                             + (fa * (1.0 - fb) * (1.0 - fc)) * g3[au, bl, cl]
                                             + ((1.0 - fa) * fb * (1.0 - fc)) * g3[al, bu, cl]
                                             + ((1.0 - fa) * (1.0 - fb) * fc) * g3[al, bl, cu]
                                             + (fa * fb * (1.0 - fc)) * g3[au, bu, cl]
                                             + (fa * (1.0 - fb) * fc) * g3[au, bl, cu]
                                             + ((1.0 - fa) * fb * fc) * g3[al, bu, cu]
                                             + (fa * fb * fc) * g3[au, bu, cu])


def _interpolate(fine, coarse, max_gs):
    """Coarse grids back onto the fine grid, in Chargemol's order."""
    g2, g3, g4, g6, g8, g12 = coarse
    if max_gs > 8:
        _halve(g6, g12)
    if max_gs > 6:
        _halve(g4, g8)
    if max_gs > 4:
        _halve(g3, g6)
    if max_gs > 3:
        _halve(g2, g4)
    if max_gs > 2:
        _from_grid3(fine, g3)
    if max_gs > 1:
        _from_grid2(fine, g2)


@numba.njit(cache=True)
def _block_geometry(data, pbd, na_tot, nb_tot, nc_tot):
    """Per pair block: grid spacing, first/last valid offset along a, b, c, and the
    centre pixel (columns 0-9), with column 10 = 1 when the block has grid points."""
    nb = pbd.shape[0]
    geo = np.zeros((nb, 11), dtype=np.int64)
    tots = (na_tot, nb_tot, nc_tot)
    for b in range(nb):
        start = int(pbd[b, 0]) - 1
        gs = nint(data[start, 35])
        geo[b, 0] = gs
        valid = 1
        for ax in range(3):
            dn = nint(data[start, 19 + ax])
            cen = nint(data[start, 25 + ax])
            lo = dn
            hi = -dn
            for n in range(-dn, dn + 1):
                k = n + cen + 1
                if k < 1 or k > tots[ax]:
                    continue
                if gs > 1 and k % gs != gs - 1:
                    continue
                lo = min(lo, n)
                hi = max(hi, n)
            geo[b, 1 + 2 * ax] = lo
            geo[b, 2 + 2 * ax] = hi
            geo[b, 7 + ax] = cen
            if lo > hi:
                valid = 0
        geo[b, 10] = valid
    return geo


@numba.njit(cache=True)
def _block_value(data, start, stop, X, Y, Z, step, gs, spin, mult_pv, sums):
    """Pair sum at one point of a block: (t4, t7, t8); with sums=True also accumulates
    the pass-1 numeric overlap sums (columns 38, 39). Returns t4 < 0 when the point is
    outside the block cutoff."""
    x1, y1, z1 = data[start, 9], data[start, 10], data[start, 11]
    x2, y2, z2 = data[start, 16], data[start, 17], data[start, 18]
    d2 = (X - data[start, 22]) ** 2 + (Y - data[start, 23]) ** 2 + (Z - data[start, 24]) ** 2
    if d2 > data[start, 33]:
        return -1.0, 0.0, 0.0
    dx1 = X - x1
    dy1 = Y - y1
    dz1 = Z - z1
    dx2 = X - x2
    dy2 = Y - y2
    dz2 = Z - z2
    t4 = math.exp(-(data[start, 5] + data[start, 12]) * d2) * data[start, 34]
    t5 = t4 * mult_pv[0]
    if mult_pv[1] != 1.0:
        t5 = t5 * mult_pv[1]
    t7 = 0.0
    t8 = 0.0
    for p in range(start, stop + 1):
        if d2 > data[p, 33]:
            break
        renorm = 1.0 if (step == 1 or gs == 0) else data[p, 31]
        if gs != 0 and step == 2 and renorm == 0.0:
            continue
        t6 = (powi(dx1, nint(data[p, 6])) * powi(dy1, nint(data[p, 7])) * powi(dz1, nint(data[p, 8]))
              * powi(dx2, nint(data[p, 13])) * powi(dy2, nint(data[p, 14])) * powi(dz2, nint(data[p, 15])))
        if gs != 0:
            if step == 1:
                if sums:
                    data[p, 38] = data[p, 38] + t5 * t6
                    data[p, 39] = data[p, 39] + t5 * abs(t6)
            else:
                t6 = abs(t6)
        t7 = t7 + t6 * renorm * data[p, 2]
        if spin:
            t8 = t8 + t6 * renorm * data[p, 3]
    return t4, t7, t8


@numba.njit(cache=True)
def _mult(gs, pv):
    m = np.empty(2)
    m[0] = pv
    m[1] = 1.0
    if gs == 2:
        m[1] = 8.0
    elif gs == 3:
        m[1] = 27.0
    elif gs == 4:
        m[1] = 64.0
    elif gs == 6:
        m[1] = 216.0
    elif gs == 8:
        m[1] = 512.0
    elif gs == 12:
        m[1] = 1728.0
    return m


@numba.njit(parallel=True, cache=True)
def _pass1_parallel(data, pbd, geo, boundary, pv):
    """Phase A: pass-1 numeric overlap sums and the renormalization factors, one block
    per task (a block's pairs only see its own points, so the order is the serial one)."""
    nb = pbd.shape[0]
    corr = np.zeros(nb)
    for b in numba.prange(nb):
        gs = geo[b, 0]
        if gs == 0 or geo[b, 10] == 0:
            continue
        start = int(pbd[b, 0]) - 1
        stop = int(pbd[b, 1]) - 1
        t1 = data[start, 22] - data[start, 28]
        t2 = data[start, 23] - data[start, 29]
        t3 = data[start, 24] - data[start, 30]
        mp = _mult(gs, pv)
        for nc in range(geo[b, 5], geo[b, 6] + 1, gs):
            for nbb in range(geo[b, 3], geo[b, 4] + 1, gs):
                for na in range(geo[b, 1], geo[b, 2] + 1, gs):
                    X = na * boundary[0, 0] + nbb * boundary[1, 0] + nc * boundary[2, 0] + t1
                    Y = na * boundary[0, 1] + nbb * boundary[1, 1] + nc * boundary[2, 1] + t2
                    Z = na * boundary[0, 2] + nbb * boundary[1, 2] + nc * boundary[2, 2] + t3
                    _block_value(data, start, stop, X, Y, Z, 1, gs, False, mp, True)
        mc = 0.0
        for p in range(start, stop + 1):
            max_coeff = abs((data[p, 2] + data[p, 3]) / 2.0) + abs((data[p, 2] - data[p, 3]) / 2.0)
            if max_coeff * data[p, 39] > GAUSSIAN_OVERLAP_TOLERANCE:
                data[p, 31] = max(min(((data[p, 4] - data[p, 38]) / data[p, 39]), 0.05), -0.05)
                mc = max(abs((data[p, 4] - data[p, 38]) / data[p, 39]), mc)
            else:
                data[p, 31] = 0.0
        corr[b] = mc
    return corr


@numba.njit(parallel=True, cache=True)
def _accumulate_parallel(data, pbd, geo, boundary, nc_tot, pv, spin, val, spn, core, cval, cspn):
    """Phase B: both passes' contributions, one fine z-plane per task. Every fine or
    coarse cell is fed by points of a single plane, and each task visits blocks in
    order, so each cell sums in the serial order."""
    nb = pbd.shape[0]
    for kc in numba.prange(1, nc_tot + 1):
        for b in range(nb):
            if geo[b, 10] == 0:
                continue
            gs = geo[b, 0]
            stride = gs if gs > 1 else 1
            nc = kc - geo[b, 9] - 1
            if nc < geo[b, 5] or nc > geo[b, 6] or (nc - geo[b, 5]) % stride != 0:
                continue
            start = int(pbd[b, 0]) - 1
            stop = int(pbd[b, 1]) - 1
            t1 = data[start, 22] - data[start, 28]
            t2 = data[start, 23] - data[start, 29]
            t3 = data[start, 24] - data[start, 30]
            mp = _mult(gs, pv)
            slot = _coarse_slot(gs)
            for step in range(1, 3):
                if step == 2 and gs == 0:
                    continue
                for nbb in range(geo[b, 3], geo[b, 4] + 1, stride):
                    kb = nbb + geo[b, 8] + 1
                    for na in range(geo[b, 1], geo[b, 2] + 1, stride):
                        ka = na + geo[b, 7] + 1
                        X = na * boundary[0, 0] + nbb * boundary[1, 0] + nc * boundary[2, 0] + t1
                        Y = na * boundary[0, 1] + nbb * boundary[1, 1] + nc * boundary[2, 1] + t2
                        Z = na * boundary[0, 2] + nbb * boundary[1, 2] + nc * boundary[2, 2] + t3
                        t4, t7, t8 = _block_value(data, start, stop, X, Y, Z, step, gs, spin, mp, False)
                        if t4 < 0.0:
                            continue
                        if gs == 0:
                            core[ka - 1, kb - 1, kc - 1] = core[ka - 1, kb - 1, kc - 1] + t7 * t4
                            if spin:
                                spn[ka - 1, kb - 1, kc - 1] = spn[ka - 1, kb - 1, kc - 1] + t8 * t4
                        elif gs == 1:
                            val[ka - 1, kb - 1, kc - 1] = val[ka - 1, kb - 1, kc - 1] + t7 * t4
                            if spin:
                                spn[ka - 1, kb - 1, kc - 1] = spn[ka - 1, kb - 1, kc - 1] + t8 * t4
                        else:
                            ia = nint((ka + 1) / float(gs)) - 1
                            ib = nint((kb + 1) / float(gs)) - 1
                            ic = nint((kc + 1) / float(gs)) - 1
                            g = cval[slot]
                            g[ia, ib, ic] = g[ia, ib, ic] + t7 * t4
                            if spin:
                                h = cspn[slot]
                                h[ia, ib, ic] = h[ia, ib, ic] + t8 * t4


@numba.njit(cache=True)
def _analytic_spin_occ(data, pbd, geo, center, atom_of_center, boundary, pv, occ):
    """Phase C: spin of the analytic pairs taken out of occupancy_correction row 2, in
    the serial block and point order."""
    for b in range(pbd.shape[0]):
        if geo[b, 0] != 0 or geo[b, 10] == 0:
            continue
        start = int(pbd[b, 0]) - 1
        stop = int(pbd[b, 1]) - 1
        ai = atom_of_center[center[nint(data[start, 0]) - 1]]
        aj = atom_of_center[center[nint(data[start, 1]) - 1]]
        wi = data[start, 36]
        wj = data[start, 37]
        t1 = data[start, 22] - data[start, 28]
        t2 = data[start, 23] - data[start, 29]
        t3 = data[start, 24] - data[start, 30]
        mp = _mult(0, pv)
        for nc in range(geo[b, 5], geo[b, 6] + 1):
            for nbb in range(geo[b, 3], geo[b, 4] + 1):
                for na in range(geo[b, 1], geo[b, 2] + 1):
                    X = na * boundary[0, 0] + nbb * boundary[1, 0] + nc * boundary[2, 0] + t1
                    Y = na * boundary[0, 1] + nbb * boundary[1, 1] + nc * boundary[2, 1] + t2
                    Z = na * boundary[0, 2] + nbb * boundary[1, 2] + nc * boundary[2, 2] + t3
                    t4, t7, t8 = _block_value(data, start, stop, X, Y, Z, 1, 0, True, mp, False)
                    if t4 < 0.0:
                        continue
                    sc = t8 * t4
                    if ai >= 0:
                        occ[1, ai] = occ[1, ai] - sc * pv * wi
                    if aj >= 0:
                        occ[1, aj] = occ[1, aj] - sc * pv * wj


def density_grids(prep, grid, pairs, parallel=True):
    """Step 7: valence, spin and analytic-pair core grids before the EDF/core step."""
    na, nb, nc = grid["totnum"]
    spin = prep["spin_available"]
    data = pairs["included_primitive_data"].copy()
    occ = pairs["occupancy_correction"].copy()
    val = np.zeros((na, nb, nc), order="F")
    core = np.zeros((na, nb, nc), order="F")
    spn = np.zeros((na, nb, nc) if spin else (1, 1, 1), order="F")
    numeric = data[:, 32] == 0.0
    max_gs = int(data[numeric, 35].max()) if numeric.any() else 0
    cval = numba.typed.List()
    cspn = numba.typed.List()
    for s in COARSE:
        # unused grids get a (2, 2, 2) placeholder: a (1, 1, 1) array is also C-contiguous,
        # which numba types as a different layout than the F-ordered grids
        shape = (na // s, nb // s, nc // s) if max_gs >= s else (2, 2, 2)
        cval.append(np.zeros(shape, order="F"))
        cspn.append(np.zeros(shape if spin else (2, 2, 2), order="F"))
    pbd = pairs["pair_block_data"]
    if parallel:
        geo = _block_geometry(data, pbd, na, nb, nc)
        corr = _pass1_parallel(data, pbd, geo, grid["boundary"], grid["pixelvolume"])
        max_corr = float(corr.max()) if corr.size else 0.0
        _accumulate_parallel(data, pbd, geo, grid["boundary"], nc, grid["pixelvolume"], spin, val, spn, core,
                             cval, cspn)
        if spin:
            _analytic_spin_occ(data, pbd, geo, prep["center"], prep["atom_of_center"], grid["boundary"],
                               grid["pixelvolume"], occ)
    else:
        max_corr = _point_loop(data, pbd, prep["center"], prep["atom_of_center"], grid["boundary"], na, nb, nc,
                               grid["pixelvolume"], spin, occ, val, spn, core, cval, cspn)
    _interpolate(val, list(cval), max_gs)
    if spin:
        _interpolate(spn, list(cspn), max_gs)
    return {"valence_density": val, "spin_density": spn if spin else None, "core_density": core,
            "included_primitive_data": data, "occupancy_correction": occ,
            "core_electrons": pairs["core_electrons"], "max_pair_correction": max_corr,
            "max_grid_spacing": max_gs}


# ---------------------------------------------------------------------------------
# K8: core grid, missing core, grid-spacing check (Step 8)
# ---------------------------------------------------------------------------------

MAXPIXELVOLUME = 0.0157
INTEGRATION_TOLERANCE = 0.10


@numba.njit(cache=True)
def _edf_core(core, ncen, shift, xyz, alpha, coef, boundary, delta, totnum):
    """Step 8 EDF loop: each EDF s primitive added on the fine grid inside the 500 pm
    sphere of its centre (shell index <= nshells)."""
    for j in range(alpha.shape[0]):
        for nc in range(-delta[2], delta[2] + 1):
            kc = nc + ncen[j, 2] + 1
            if kc < 1 or kc > totnum[2]:
                continue
            for nb in range(-delta[1], delta[1] + 1):
                kb = nb + ncen[j, 1] + 1
                if kb < 1 or kb > totnum[1]:
                    continue
                for na in range(-delta[0], delta[0] + 1):
                    ka = na + ncen[j, 0] + 1
                    if ka < 1 or ka > totnum[0]:
                        continue
                    v0 = na * boundary[0, 0] + nb * boundary[1, 0] + nc * boundary[2, 0] - shift[j, 0]
                    v1 = na * boundary[0, 1] + nb * boundary[1, 1] + nc * boundary[2, 1] - shift[j, 1]
                    v2 = na * boundary[0, 2] + nb * boundary[1, 2] + nc * boundary[2, 2] - shift[j, 2]
                    distance = math.sqrt(v0 * v0 + v1 * v1 + v2 * v2)
                    if math.ceil(SCALEFACTOR * distance + ZERO_TOLERANCE) <= NSHELLS:
                        ax, ay, az = xyz[j, 0], xyz[j, 1], xyz[j, 2]
                        x = v0 + ax
                        y = v1 + ay
                        z = v2 + az
                        g = (coef[j] * powi(x - ax, 0) * powi(y - ay, 0) * powi(z - az, 0)
                             * math.exp(-alpha[j] * ((x - ax) ** 2 + (y - ay) ** 2 + (z - az) ** 2)))
                        core[ka - 1, kb - 1, kc - 1] = core[ka - 1, kb - 1, kc - 1] + g


@numba.njit(parallel=True, cache=True)
def _clamp(core, val):
    n0, n1, n2 = core.shape
    for c in numba.prange(n2):
        for b in range(n1):
            for a in range(n0):
                core[a, b, c] = max(core[a, b, c], 0.0)
                val[a, b, c] = max(val[a, b, c], -core[a, b, c])


@numba.njit(cache=True)
def _shell_points(nabc, shift, boundary, lo, hi):
    """check_grid_spacing: points per radial shell for each atom, and the population of
    Chargemol's guess density used for the grid check."""
    natoms = nabc.shape[0]
    sum_points = np.zeros((NSHELLS, natoms), dtype=np.int64)
    sum_density = np.zeros((NSHELLS, natoms))
    guess = np.empty(NSHELLS)
    for k in range(NSHELLS):
        guess[k] = 0.05 * ((1.0 - (k + 1 - 1.0) / (NSHELLS - 1.0))) ** 3
    for j in range(natoms):
        for nc in range(lo[j, 2], hi[j, 2] + 1):
            for nb in range(lo[j, 1], hi[j, 1] + 1):
                for na in range(lo[j, 0], hi[j, 0] + 1):
                    v0 = na * boundary[0, 0] + nb * boundary[1, 0] + nc * boundary[2, 0] - shift[j, 0]
                    v1 = na * boundary[0, 1] + nb * boundary[1, 1] + nc * boundary[2, 1] - shift[j, 1]
                    v2 = na * boundary[0, 2] + nb * boundary[1, 2] + nc * boundary[2, 2] - shift[j, 2]
                    s = int(math.ceil(SCALEFACTOR * math.sqrt(v0 * v0 + v1 * v1 + v2 * v2) + ZERO_TOLERANCE))
                    if s <= NSHELLS:
                        sum_density[s - 1, j] = sum_density[s - 1, j] + guess[s - 1]
                        sum_points[s - 1, j] = sum_points[s - 1, j] + 1
    return sum_points, sum_density


@numba.njit(cache=True)
def _missing_core(core, nabc, shift, boundary, lo, hi, ref, missing):
    """add_missing_core_density: core reference density by radial shell, for atoms whose
    core electrons the wfx leaves out."""
    for j in range(nabc.shape[0]):
        if missing[j] == 0:
            continue
        for nc in range(lo[j, 2], hi[j, 2] + 1):
            kc = nc + nabc[j, 2] + 1
            for nb in range(lo[j, 1], hi[j, 1] + 1):
                kb = nb + nabc[j, 1] + 1
                for na in range(lo[j, 0], hi[j, 0] + 1):
                    ka = na + nabc[j, 0] + 1
                    v0 = na * boundary[0, 0] + nb * boundary[1, 0] + nc * boundary[2, 0] - shift[j, 0]
                    v1 = na * boundary[0, 1] + nb * boundary[1, 1] + nc * boundary[2, 1] - shift[j, 1]
                    v2 = na * boundary[0, 2] + nb * boundary[1, 2] + nc * boundary[2, 2] - shift[j, 2]
                    s = int(math.ceil(SCALEFACTOR * math.sqrt(v0 * v0 + v1 * v1 + v2 * v2) + ZERO_TOLERANCE))
                    if s <= NSHELLS:
                        core[ka - 1, kb - 1, kc - 1] = core[ka - 1, kb - 1, kc - 1] + ref[s - 1, j]


def core_grids(prep, grid, grids):
    """Step 8: EDF core density, clamps, zero-core atoms, missing core from the core
    reference densities (only when the wfx has no EDF block, as Chargemol decides per
    file), and check_grid_spacing. Updates grids in place and adds the per-atom shell
    geometry the iterators use."""
    from qtaim_gen.source.data import chargemol_refs

    wfx = prep["wfx"]
    boundary, origin, inv_b = grid["boundary"], grid["origin"], grid["inv_boundary"]
    delta = np.array(grid["delta"], dtype=np.int64)
    totnum = np.array(grid["totnum"], dtype=np.int64)
    core, val = grids["core_density"], grids["valence_density"]
    edf_available = len(wfx["edf_exp"]) > 0
    if edf_available:
        xyz = prep["center_xyz"][wfx["edf_center"]]
        ncen, shift = [], []
        for p in xyz:
            tv = _matvec(inv_b.tolist(), [p[k] - origin[k] for k in range(3)])
            n = [_nint(v) for v in tv]
            ncen.append(n)
            shift.append(_matvec(boundary.T.tolist(), [tv[k] - n[k] for k in range(3)]))
        _edf_core(core, np.array(ncen, dtype=np.int64), np.array(shift), xyz, wfx["edf_exp"].astype(np.float64),
                  wfx["edf_coef"].astype(np.float64), boundary, delta, totnum)
    _clamp(core, val)

    occ = grids["occupancy_correction"]
    core_e = grids["core_electrons"].copy()
    for j in range(len(core_e)):
        if core_e[j] < ZERO_TOLERANCE:
            occ[0, j] = core_e[j]
            core_e[j] = 0.0

    if grid["pixelvolume"] > MAXPIXELVOLUME:
        raise ValueError("grid pixel volume above Chargemol's maximum")
    nabc, shift = grid["center_nabc"], grid["center_shift"]
    natoms = len(nabc)
    lo = np.zeros((natoms, 3), dtype=np.int64)
    hi = np.zeros((natoms, 3), dtype=np.int64)
    for j in range(natoms):
        for k in range(3):
            ks = np.arange(-delta[k], delta[k] + 1) + nabc[j, k] + 1
            ok = np.nonzero((ks >= 1) & (ks <= totnum[k]))[0]
            if ok.size == 0:
                raise ValueError(f"atom {j + 1} has no grid points")
            lo[j, k], hi[j, k] = ok[0] - delta[k], ok[-1] - delta[k]
    sum_points, sum_density = _shell_points(nabc, shift, boundary, lo, hi)
    population = sum_density.sum(axis=0) * grid["pixelvolume"]
    if np.any(population - population.sum() / natoms > INTEGRATION_TOLERANCE):
        raise ValueError("grid spacing check failed")

    missing = np.zeros(natoms, dtype=np.int64) if edf_available else np.rint(prep["core_electrons"]).astype(np.int64)
    if missing.any():
        ref = np.zeros((NSHELLS, natoms))
        for j in range(natoms):
            if missing[j]:
                ref[:, j] = chargemol_refs.core_density(prep["atom_z"][j], missing[j])
        _missing_core(core, nabc, shift, boundary, lo, hi, ref, missing)

    grids.update(core_electrons=core_e, missing_core=missing, sum_points=sum_points, shell_lo=lo, shell_hi=hi)
    return grids


def load(wfx_path):
    return prepare(read_wfx(wfx_path))
