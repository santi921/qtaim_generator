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
    # align_periodic_atoms (end of Step 1) rebuilt the coordinates as
    # nabc*boundary + shift + origin, which is not bit-identical to the input, and
    # add_missing_core_density recomputes the centres from them: every later step
    # uses these shifts
    aligned = [[grid["center_nabc"][j][0] * boundary[0][i] + grid["center_nabc"][j][1] * boundary[1][i]
                + grid["center_nabc"][j][2] * boundary[2][i] + grid["center_shift"][j][i] + origin[i]
                for i in range(3)] for j in range(len(grid["center_nabc"]))]
    _, nabc, shift = _center_positions(aligned, boundary.tolist(), origin.tolist())
    nabc = np.array(nabc, dtype=np.int64)
    shift = np.array(shift)
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

    grids.update(core_electrons=core_e, missing_core=missing, sum_points=sum_points, shell_lo=lo, shell_hi=hi,
                 center_nabc=nabc, center_shift=shift, atom_xyz_aligned=np.array(aligned))
    return grids


# ---------------------------------------------------------------------------------
# P3: reference ions (module_oxidation_density, module_update_atomic_densities)
# ---------------------------------------------------------------------------------

CHARGE_CONVERGENCE_TOLERANCE = 1.0e-5
MAX_CORE_ELECTRONS_PER_PIXEL = 30.0
LOCALIZING_POWER = 4.0
MINIMUM_BURIED_TAIL_EXPONENT = 1.75
MAXIMUM_BURIED_TAIL_EXPONENT = 2.5
WA_RENORMALIZATION_MAX = 10.0

# available_reference_ion_range(Z, 1:2): lowest and highest reference-ion charge per element
REF_ION_LO = (-2, -2, -2, -2, -2, -5, -4, -3, -2, -2, -2, -2, -2, -5, -4, -3, -3, -2, -2, -2, -2, -2, -2, -3, -4,
              -3, -2, -2, -2, -2, -2, -5, -4, -3, -2, -2, -2, -2, -2, -2, -2, -3, -4, -3, -2, -2, -2, -2, -2, -5,
              -4, -3, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -3, -4,
              -3, -4, -3, -2, -2, -2, -5, -4, -3, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2,
              -2, -2, -2, -2, -2, -2, -2, -2, -2)
REF_ION_HI = (1, 2, 2, 3, 4, 5, 6, 3, 2, 2, 2, 3, 4, 5, 6, 7, 8, 2, 2, 3, 4, 5, 6, 7, 8, 7, 6, 5, 5, 3, 4, 5, 6, 7,
              8, 3, 2, 3, 4, 5, 6, 7, 8, 9, 7, 7, 5, 3, 4, 5, 6, 7, 8, 9, 2, 3, 4, 5, 5, 5, 4, 4, 4, 4, 5, 5, 4, 4,
              4, 4, 4, 5, 6, 7, 8, 9, 9, 7, 6, 5, 4, 5, 6, 7, 8, 7, 2, 3, 4, 5, 6, 7, 8, 9, 8, 9, 5, 5, 5, 4, 4, 4,
              4, 5, 6, 7, 8, 9, 9)


def radial_shell_volume():
    """chargemol.f08: 4 pi (k^3 - (k-1)^3) / (3 scalefactor^3) for shells k = 1..nshells."""
    return np.array([4.0 * PI * (k ** 3 - (k - 1) ** 3) / (3.0 * ((SCALEFACTOR * SCALEFACTOR) * SCALEFACTOR))
                     for k in range(1, NSHELLS + 1)])


def oxidation_density(z, atom_charge, rsv):
    """module_oxidation_density.f08: reference density of element z at integer charge
    atom_charge, built up from the neutral atom one charge state at a time (cusp fix,
    monotonic decrease, exact normalization, monotonic density difference)."""
    from qtaim_gen.source.data import chargemol_refs

    sign = -1 if atom_charge < 0 else 1
    prev = np.zeros(NSHELLS)
    for q in range(0, atom_charge + sign, sign):
        nel = z - q
        t = chargemol_refs.c2_density(z, nel).astype(np.float64).copy()
        s = 0.0
        for k in range(1, NSHELLS):
            t[k] = max(t[k], 1.0e-16)
            s = s + rsv[k] * t[k]
        t[0] = max((nel - s) / rsv[0], 1.0e-16)
        t1 = 1.0e-16
        s = 0.0
        for k in range(NSHELLS - 1, -1, -1):
            if t[k] > t1:
                t1 = t[k]
            else:
                t[k] = t1
            s = s + rsv[k] * t[k]
        for k in range(NSHELLS):
            t[k] = max(t[k] * nel / s, 1.0e-16)
        s = 0.0
        for k in range(NSHELLS - 1, -1, -1):
            s = s + rsv[k] * t[k]
        if q != 0:
            for _ in range(100):
                sd = 0.0
                dmin = 0.0
                for k in range(NSHELLS - 1, -1, -1):
                    dmin = max(dmin, sign * (prev[k] - t[k]))
                    shift = prev[k] - sign * dmin - t[k]
                    t[k] = t[k] + shift
                    sd = sd + shift * rsv[k]
                for k in range(NSHELLS):
                    t[k] = t[k] * s / (s + sd)
                if abs(sd) < 1.0e-16:
                    break
        prev = t
    return prev


def update_atomic_densities(atom_z, ref_charge, rsv, combined):
    """module_update_atomic_densities.f08: interpolate between the bracketing reference
    ions, or scale the end-of-range ion. combined (nshells, natoms) is updated in place."""
    for j, (z, q) in enumerate(zip(atom_z, ref_charge)):
        lo, hi = REF_ION_LO[z - 1], REF_ION_HI[z - 1]
        if lo < q < hi:
            c = int(math.ceil(q))
            lower = oxidation_density(z, c, rsv)
            upper = oxidation_density(z, c - 1, rsv)
            f = math.ceil(q) - q
            for k in range(NSHELLS):
                combined[k, j] = upper[k] * f + lower[k] * (1.0 - f)
        elif q <= lo or q >= hi:
            c = lo if q <= lo else hi
            if (z - c) > ZERO_TOLERANCE:
                combined[:, j] = ((z - q) / (z - c)) * oxidation_density(z, c, rsv)
            else:
                combined[:, j] = 0.0
    return combined


# ---------------------------------------------------------------------------------
# P3: atom-sphere kernels shared by the iterators
# Grid accumulations run one z-plane per task with atoms in order; per-atom shell sums
# run one atom per task over its points in order: both keep Chargemol's serial sums.
# ---------------------------------------------------------------------------------

@numba.njit(inline="always")
def _shell(na, nb, nc, j, boundary, shift):
    v0 = na * boundary[0, 0] + nb * boundary[1, 0] + nc * boundary[2, 0] - shift[j, 0]
    v1 = na * boundary[0, 1] + nb * boundary[1, 1] + nc * boundary[2, 1] - shift[j, 1]
    v2 = na * boundary[0, 2] + nb * boundary[1, 2] + nc * boundary[2, 2] - shift[j, 2]
    return int(math.ceil(SCALEFACTOR * math.sqrt(v0 * v0 + v1 * v1 + v2 * v2) + ZERO_TOLERANCE))


@numba.njit(parallel=True, cache=True)
def _cap_grid(grid, cap, nabc, shift, boundary, lo, hi):
    """grid = min(grid, cap) inside every atom's sphere."""
    nct = grid.shape[2]
    for kc in numba.prange(1, nct + 1):
        for j in range(nabc.shape[0]):
            nc = kc - nabc[j, 2] - 1
            if nc < lo[j, 2] or nc > hi[j, 2]:
                continue
            for nb in range(lo[j, 1], hi[j, 1] + 1):
                kb = nb + nabc[j, 1] + 1
                for na in range(lo[j, 0], hi[j, 0] + 1):
                    ka = na + nabc[j, 0] + 1
                    if _shell(na, nb, nc, j, boundary, shift) <= NSHELLS:
                        grid[ka - 1, kb - 1, kc - 1] = min(grid[ka - 1, kb - 1, kc - 1], cap)


@numba.njit(parallel=True, cache=True)
def _pseudo(out, dens, power, use, nabc, shift, boundary, lo, hi):
    """out += dens(shell, j)**power summed over atoms j with use[j] (power 1 or 4)."""
    nct = out.shape[2]
    for kc in numba.prange(1, nct + 1):
        for j in range(nabc.shape[0]):
            if not use[j]:
                continue
            nc = kc - nabc[j, 2] - 1
            if nc < lo[j, 2] or nc > hi[j, 2]:
                continue
            for nb in range(lo[j, 1], hi[j, 1] + 1):
                kb = nb + nabc[j, 1] + 1
                for na in range(lo[j, 0], hi[j, 0] + 1):
                    ka = na + nabc[j, 0] + 1
                    s = _shell(na, nb, nc, j, boundary, shift)
                    if s <= NSHELLS:
                        if power == 1:
                            out[ka - 1, kb - 1, kc - 1] = out[ka - 1, kb - 1, kc - 1] + dens[s - 1, j]
                        else:
                            out[ka - 1, kb - 1, kc - 1] = out[ka - 1, kb - 1, kc - 1] + dens[s - 1, j] ** LOCALIZING_POWER


@numba.njit(parallel=True, cache=True)
def _share_sums(dens, num, pseudo, thresh, use, nabc, shift, boundary, lo, hi):
    """Per atom and shell: sum of dens(s, j) * num / pseudo over points with pseudo > thresh,
    and the number of such points (the core and conditioned-density shell sums)."""
    natoms = nabc.shape[0]
    sums = np.zeros((NSHELLS, natoms))
    pts = np.zeros((NSHELLS, natoms), dtype=np.int64)
    for j in numba.prange(natoms):
        if not use[j]:
            continue
        for nc in range(lo[j, 2], hi[j, 2] + 1):
            kc = nc + nabc[j, 2] + 1
            for nb in range(lo[j, 1], hi[j, 1] + 1):
                kb = nb + nabc[j, 1] + 1
                for na in range(lo[j, 0], hi[j, 0] + 1):
                    ka = na + nabc[j, 0] + 1
                    s = _shell(na, nb, nc, j, boundary, shift)
                    if s <= NSHELLS and pseudo[ka - 1, kb - 1, kc - 1] > thresh:
                        sums[s - 1, j] = sums[s - 1, j] + dens[s - 1, j] * num[ka - 1, kb - 1, kc - 1] / pseudo[
                            ka - 1, kb - 1, kc - 1]
                        pts[s - 1, j] = pts[s - 1, j] + 1
    return sums, pts


@numba.njit(parallel=True, cache=True)
def _core_cubes(pcd, core, pseudo, use, nabc, shift, boundary, lo, hi):
    natoms = nabc.shape[0]
    cubes = np.zeros(natoms)
    maxd = np.zeros(natoms)
    for j in numba.prange(natoms):
        if not use[j]:
            continue
        for nc in range(lo[j, 2], hi[j, 2] + 1):
            kc = nc + nabc[j, 2] + 1
            for nb in range(lo[j, 1], hi[j, 1] + 1):
                kb = nb + nabc[j, 1] + 1
                for na in range(lo[j, 0], hi[j, 0] + 1):
                    ka = na + nabc[j, 0] + 1
                    s = _shell(na, nb, nc, j, boundary, shift)
                    if s <= NSHELLS and pseudo[ka - 1, kb - 1, kc - 1] > ZERO_TOLERANCE:
                        loc = pcd[s - 1, j] * core[ka - 1, kb - 1, kc - 1] / pseudo[ka - 1, kb - 1, kc - 1]
                        cubes[j] = cubes[j] + (loc * loc) * loc
                        maxd[j] = max(maxd[j], loc)
    return cubes, maxd


@numba.njit(parallel=True, cache=True)
def _core_rescale(out, pcd, core, pseudo, kf, use, nabc, shift, boundary, lo, hi):
    nct = out.shape[2]
    for kc in numba.prange(1, nct + 1):
        for j in range(nabc.shape[0]):
            if not use[j]:
                continue
            nc = kc - nabc[j, 2] - 1
            if nc < lo[j, 2] or nc > hi[j, 2]:
                continue
            for nb in range(lo[j, 1], hi[j, 1] + 1):
                kb = nb + nabc[j, 1] + 1
                for na in range(lo[j, 0], hi[j, 0] + 1):
                    ka = na + nabc[j, 0] + 1
                    s = _shell(na, nb, nc, j, boundary, shift)
                    if s <= NSHELLS and pseudo[ka - 1, kb - 1, kc - 1] > ZERO_TOLERANCE:
                        loc = pcd[s - 1, j] * core[ka - 1, kb - 1, kc - 1] / pseudo[ka - 1, kb - 1, kc - 1]
                        if loc > ZERO_TOLERANCE * ZERO_TOLERANCE:
                            out[ka - 1, kb - 1, kc - 1] = out[ka - 1, kb - 1, kc - 1] + loc / math.sqrt(
                                1 - 2 * kf[j] * (loc * loc))


def _update_profile(dens, avg, pts, exp_const):
    """New atomic shell profile from spherical averages, decaying no faster than
    exp_const per shell where both shells have points (core iterator update)."""
    for j in range(dens.shape[1]):
        dens[0, j] = avg[0, j]
        t = dens[0, j]
        for k in range(1, NSHELLS):
            if pts[k, j] > 0 and t > 0.0:
                dens[k, j] = max(min(avg[k, j], t * exp_const), 0.0)
            else:
                dens[k, j] = max(avg[k, j], 0.0)
            t = dens[k, j]


def _shell_average(sums, pts):
    avg = np.zeros_like(sums)
    nz = pts > 0
    avg[nz] = sums[nz] / pts[nz]
    return avg


def _column_sums(a):
    """Fortran sum(a, 1): sequential over shells for each atom."""
    out = np.zeros(a.shape[1])
    for j in range(a.shape[1]):
        s = 0.0
        for k in range(a.shape[0]):
            s = s + a[k, j]
        out[j] = s
    return out


def core_iterator(state):
    """module_core_iterator.f08: partition the core density among atoms, then rescale the
    core grid so each atom's core integrates to its core electron count."""
    g = state["geom"]
    args = (g["nabc"], g["shift"], g["boundary"], g["lo"], g["hi"])
    pv = g["pixelvolume"]
    core = state["core_density"]
    core_e = state["core_electrons"]
    use = core_e > ZERO_TOLERANCE
    pcd = state["neutral_density"].copy()
    core_pop = np.zeros(len(core_e))
    _cap_grid(core, MAX_CORE_ELECTRONS_PER_PIXEL / pv, *args)
    exp_const = math.exp(-2.0 * CUTOFF_RADIUS * BOHRPERANGSTROM / (100.0 * NSHELLS))
    pseudo = np.zeros(core.shape, order="F")
    for it in range(1, 201):
        pseudo[...] = 0.0
        _pseudo(pseudo, pcd, 1, use, *args)
        sums, pts = _share_sums(pcd, core, pseudo, ZERO_TOLERANCE, use, *args)
        avg = _shell_average(sums, pts)
        old = core_pop
        core_pop = pv * _column_sums(sums)
        change = float(np.max(np.abs(core_pop - old))) if len(core_pop) else 0.0
        if change < CHARGE_CONVERGENCE_TOLERANCE and it > 5:
            break
        _update_profile(pcd, avg, pts, exp_const)

    na_corr = core_e - core_pop
    for it in range(1, 41):
        cubes, maxd = _core_cubes(pcd, core, pseudo, use, *args)
        kf = np.zeros(len(core_e))
        for j in range(len(core_e)):
            if cubes[j] > (ZERO_TOLERANCE * ZERO_TOLERANCE) * ZERO_TOLERANCE and maxd[j] > ZERO_TOLERANCE:
                kf[j] = na_corr[j] / (cubes[j] * pv)
                kf[j] = min(kf[j], 0.25 / (maxd[j] * maxd[j]))
        new = np.zeros(core.shape, order="F")
        _core_rescale(new, pcd, core, pseudo, kf, use, *args)
        core = new
        sums, pts = _share_sums(pcd, core, pseudo, ZERO_TOLERANCE, use, *args)
        avg = _shell_average(sums, pts)
        core_pop = pv * _column_sums(sums)
        na_corr = core_e - core_pop
        if float(np.max(np.abs(na_corr))) < CHARGE_CONVERGENCE_TOLERANCE:
            break
        _update_profile(pcd, avg, pts, exp_const)
        pseudo[...] = 0.0
        _pseudo(pseudo, pcd, 1, use, *args)
    state.update(core_density=core, core_pseudodensity=pseudo, core_population=core_pop,
                 partial_core_density=pcd, spherical_avg_core_density=avg, sum_points=pts)
    return state


# ---------------------------------------------------------------------------------
# P3: radial reshaping (module_reshaping_functions.f08)
# ---------------------------------------------------------------------------------

RESHAPING_TOLERANCE = 1.0e-10


@numba.njit(cache=True)
def _constrain(r, u, add, sq, w, fe, tail):
    """r = u + add*sqrt(u), made non-negative and non-increasing (tail: decaying at
    least by fe per shell) wherever the shell has points; returns its integral."""
    n = r.shape[0]
    for k in range(n):
        r[k] = u[k] + add * sq[k]
    r[0] = max(r[0], 0.0)
    t = r[0]
    for k in range(1, n):
        r[k] = max(r[k], 0.0)
        if w[k] > 0.0 and t > 0.0:
            r[k] = min(r[k], t * fe[k] if tail else t)
        t = r[k]
    s = 0.0
    for k in range(n):
        s = s + r[k] * w[k]
    return s


@numba.njit(cache=True)
def reshape_profile(u, w, fe, se, tail):
    """monotonic_decay_function (tail=False) and tail_exponential_decay_function
    (tail=True): add c*sqrt(u) to the radial profile u so that, after the shape
    constraints, it keeps the integral of u; c by bracketing and interpolation."""
    n = u.shape[0]
    r = u.copy()
    sq = np.empty(n)
    weighted = 0.0
    first = 0.0
    for k in range(n):
        sq[k] = math.sqrt(u[k])
        weighted = weighted + sq[k] * w[k]
        first = first + u[k] * w[k]
    iters = 1
    add = np.zeros(5)
    err = np.zeros(5)
    if weighted < RESHAPING_TOLERANCE or first < RESHAPING_TOLERANCE:
        return r, iters
    # the unshifted profile: _constrain with add = 0 reproduces u + 0*sqrt(u) = u
    second = _constrain(r, u, 0.0, sq, w, fe, tail)
    re = first - second
    err[0] = re
    if re < RESHAPING_TOLERANCE:
        return r, iters
    for _ in range(50):
        add[1] = 2.0 * add[0] + re / weighted
        second = _constrain(r, u, add[1], sq, w, fe, tail)
        re = first - second
        if re > 0.0:
            add[0] = add[1]
            err[0] = re
        else:
            add[4] = add[1]
            err[4] = re
            break
    for trial in range(1, 51):
        if abs(err[4] - err[0]) < RESHAPING_TOLERANCE:
            break
        iters = trial
        add[2] = 0.5 * (add[0] + add[4])
        err[2] = first - _constrain(r, u, add[2], sq, w, fe, tail)
        if abs(err[2]) < RESHAPING_TOLERANCE:
            break
        m = 2.0 - 4.0 * (err[0] - err[2]) / (err[0] - err[4])
        if abs(m) < RESHAPING_TOLERANCE:
            add[3] = add[0] + (add[4] - add[0]) * (err[0] / (err[0] - err[4]))
        else:
            t = err[0] / (err[0] - err[4])
            add[3] = add[0] + (add[4] - add[0]) * (m - 1.0 + math.sqrt((1.0 - m) ** 2 + 4.0 * m * t)) / (2.0 * m)
        err[3] = first - _constrain(r, u, add[3], sq, w, fe, tail)
        if abs(err[3]) < RESHAPING_TOLERANCE:
            break
        if err[2] > 0.0 and err[3] > 0.0:
            if 3.0 * abs(err[4]) < abs(err[3]):
                add[1] = add[4] + (add[4] - add[3]) * (2.0 * err[4]) / (err[3] - err[4])
            elif 3.0 * abs(err[3]) < abs(err[4]):
                add[1] = add[3] + (add[4] - add[3]) * (2.0 * err[3]) / (err[3] - err[4])
            else:
                add[1] = 0.5 * (add[3] + add[4])
        elif err[2] > 0.0 and err[3] <= 0.0:
            if 3.0 * abs(err[2]) < abs(err[3]):
                add[1] = add[2] + (add[3] - add[2]) * (2.0 * err[2]) / (err[2] - err[3])
            elif 3.0 * abs(err[3]) < abs(err[2]):
                add[1] = add[3] + (add[3] - add[2]) * (2.0 * err[3]) / (err[2] - err[3])
            else:
                add[1] = 0.5 * (add[3] + add[2])
        elif err[2] <= 0.0 and err[3] > 0.0:
            if 3.0 * abs(err[2]) < abs(err[3]):
                add[1] = add[2] + (add[2] - add[3]) * (2.0 * err[2]) / (err[3] - err[2])
            elif 3.0 * abs(err[3]) < abs(err[2]):
                add[1] = add[3] + (add[2] - add[3]) * (2.0 * err[3]) / (err[3] - err[2])
            else:
                add[1] = 0.5 * (add[3] + add[2])
        else:
            if 3.0 * abs(err[0]) < abs(err[3]):
                add[1] = add[0] + (add[3] - add[0]) * (2.0 * err[0]) / (err[0] - err[3])
            elif 3.0 * abs(err[3]) < abs(err[0]):
                add[1] = add[3] + (add[3] - add[0]) * (2.0 * err[3]) / (err[0] - err[3])
            else:
                add[1] = 0.5 * (add[3] + add[0])
        err[1] = first - _constrain(r, u, add[1], sq, w, fe, tail)
        if abs(err[1]) < RESHAPING_TOLERANCE:
            break
        if err[1] > 0.0:
            add[0] = add[1]
            err[0] = err[1]
        else:
            add[4] = add[1]
            err[4] = err[1]
        if err[2] > 0.0:
            add[0] = max(add[2], add[0])
            err[0] = min(err[2], err[0])
        else:
            add[4] = min(add[2], add[4])
            err[4] = max(err[2], err[4])
        if err[3] > 0.0:
            add[0] = max(add[3], add[0])
            err[0] = min(err[3], err[0])
        else:
            add[4] = min(add[3], add[4])
            err[4] = max(err[3], err[4])
    if tail:
        r[0] = max(r[0], 0.0)
        t = r[0]
        for k in range(1, n):
            r[k] = max(r[k], t * se[k])
            t = r[k]
        s = 0.0
        for k in range(n):
            s = s + r[k] * w[k]
        if abs(s) > RESHAPING_TOLERANCE:
            for k in range(n):
                r[k] = r[k] * first / s
    return r, iters


# ---------------------------------------------------------------------------------
# P3: DDEC6 valence iterator (module_DDEC6_valence_iterator.f08)
# ---------------------------------------------------------------------------------

@numba.njit(parallel=True, cache=True)
def _reference_pops(comb, total, ref, loc_ref, pv, nabc, shift, boundary, lo, hi):
    """Steps 1-2: per-atom populations of the reference-ion and localized (power 4)
    partitions, pixel volume inside each term as in Chargemol."""
    natoms = nabc.shape[0]
    vpop = np.zeros(natoms)
    lpop = np.zeros(natoms)
    for j in numba.prange(natoms):
        for nc in range(lo[j, 2], hi[j, 2] + 1):
            kc = nc + nabc[j, 2] + 1
            for nb in range(lo[j, 1], hi[j, 1] + 1):
                kb = nb + nabc[j, 1] + 1
                for na in range(lo[j, 0], hi[j, 0] + 1):
                    ka = na + nabc[j, 0] + 1
                    s = _shell(na, nb, nc, j, boundary, shift)
                    if s <= NSHELLS and ref[ka - 1, kb - 1, kc - 1] > ZERO_TOLERANCE:
                        c = comb[s - 1, j]
                        t = total[ka - 1, kb - 1, kc - 1]
                        vpop[j] = vpop[j] + pv * c * t / ref[ka - 1, kb - 1, kc - 1]
                        lpop[j] = lpop[j] + pv * (c ** LOCALIZING_POWER) * t / loc_ref[ka - 1, kb - 1, kc - 1]
    return vpop, lpop


@numba.njit(parallel=True, cache=True)
def _tau_sums(cond, cref, nabc, shift, boundary, lo, hi):
    natoms = nabc.shape[0]
    pts = np.zeros((NSHELLS, natoms), dtype=np.int64)
    s_sqrt = np.zeros((NSHELLS, natoms))
    s_ratio = np.zeros((NSHELLS, natoms))
    thresh = ZERO_TOLERANCE * ZERO_TOLERANCE
    for j in numba.prange(natoms):
        for nc in range(lo[j, 2], hi[j, 2] + 1):
            kc = nc + nabc[j, 2] + 1
            for nb in range(lo[j, 1], hi[j, 1] + 1):
                kb = nb + nabc[j, 1] + 1
                for na in range(lo[j, 0], hi[j, 0] + 1):
                    ka = na + nabc[j, 0] + 1
                    s = _shell(na, nb, nc, j, boundary, shift)
                    if s <= NSHELLS and cref[ka - 1, kb - 1, kc - 1] > thresh:
                        pts[s - 1, j] = pts[s - 1, j] + 1
                        s_sqrt[s - 1, j] = s_sqrt[s - 1, j] + math.sqrt(cref[ka - 1, kb - 1, kc - 1])
                        s_ratio[s - 1, j] = s_ratio[s - 1, j] + cond[s - 1, j] / math.sqrt(cref[ka - 1, kb - 1, kc - 1])
    return pts, s_sqrt, s_ratio


@numba.njit(parallel=True, cache=True)
def _iter_sums(partial, total, tpseudo, nabc, shift, boundary, lo, hi):
    """Steps 4-7: per atom and shell, the density share, its weighted share and the
    (weighted) point counts."""
    natoms = nabc.shape[0]
    sd = np.zeros((NSHELLS, natoms))
    swd = np.zeros((NSHELLS, natoms))
    pts = np.zeros((NSHELLS, natoms), dtype=np.int64)
    swp = np.zeros((NSHELLS, natoms))
    for j in numba.prange(natoms):
        for nc in range(lo[j, 2], hi[j, 2] + 1):
            kc = nc + nabc[j, 2] + 1
            for nb in range(lo[j, 1], hi[j, 1] + 1):
                kb = nb + nabc[j, 1] + 1
                for na in range(lo[j, 0], hi[j, 0] + 1):
                    ka = na + nabc[j, 0] + 1
                    s = _shell(na, nb, nc, j, boundary, shift)
                    if s <= NSHELLS and tpseudo[ka - 1, kb - 1, kc - 1] > ZERO_TOLERANCE:
                        t = partial[s - 1, j] / tpseudo[ka - 1, kb - 1, kc - 1]
                        loc = t * total[ka - 1, kb - 1, kc - 1]
                        sd[s - 1, j] = sd[s - 1, j] + loc
                        swd[s - 1, j] = swd[s - 1, j] + t * loc
                        pts[s - 1, j] = pts[s - 1, j] + 1
                        swp[s - 1, j] = swp[s - 1, j] + t
    return sd, swd, pts, swp


def _seq_sum(v):
    s = 0.0
    for x in v:
        s = s + x
    return s


def ddec6_valence_iterator(state):
    """DDEC6_valence_iterator: steps 1-2 against reference ions, step 3 conditioned
    reference density with tau and the tail constants, steps 4-7 iterative, with the
    kappa (wA renormalization) updates when a valence population goes negative."""
    g = state["geom"]
    args = (g["nabc"], g["shift"], g["boundary"], g["lo"], g["hi"])
    pv = g["pixelvolume"]
    z = state["atom_z"].astype(np.float64)
    core_e = state["core_electrons"]
    occ1 = state["occupancy_correction"][0]
    core_pop = state["core_population"]
    nvalence = state["nvalence"]
    rsv = state["radial_shell_volume"]
    natoms = len(z)
    everyone = np.ones(natoms, dtype=np.bool_)
    total = state["valence_density"] + state["core_density"]
    comb = state["neutral_density"].copy()
    ref = np.zeros(total.shape, order="F")
    loc_ref = np.zeros(total.shape, order="F")
    net = np.zeros(natoms)
    ref_charge = np.zeros(natoms)
    hirshfeld = None
    for it in (1, 2):
        ref[...] = 0.0
        loc_ref[...] = 0.0
        _pseudo(ref, comb, 1, everyone, *args)
        _pseudo(loc_ref, comb, 4, everyone, *args)
        vpop, lpop = _reference_pops(comb, total, ref, loc_ref, pv, *args)
        vpop = vpop + occ1 - core_pop
        lpop = lpop + occ1 - core_e
        norm = nvalence / _seq_sum(vpop)
        vpop = norm * vpop
        lpop = norm * lpop
        net = z - core_e - vpop
        lnet = z - core_e - lpop
        if it == 1:
            hirshfeld = net.copy()
        ref_charge = (2.0 * lnet + net) / 3.0
        update_atomic_densities(state["atom_z"], ref_charge, rsv, comb)

    # step 3: conditioned reference density
    ref[...] = 0.0
    _pseudo(ref, comb, 1, everyone, *args)
    del loc_ref
    scond, pts = _share_sums(comb, total, ref, ZERO_TOLERANCE * ZERO_TOLERANCE, everyone, *args)
    sigma = _shell_average(scond, pts)
    crp = _column_sums(scond) * pv
    ve = 0.0
    for j in range(natoms):
        ve = ve + crp[j] - core_pop[j] + occ1[j]
    norm = nvalence / ve
    for j in range(natoms):
        target = (z[j] - core_e[j] - ref_charge[j]) / norm + core_pop[j] - occ1[j]
        if crp[j] > CHARGE_CONVERGENCE_TOLERANCE:
            for k in range(NSHELLS):
                sigma[k, j] = max(sigma[k, j] * target / crp[j], 0.0)
    cond = sigma.copy()
    w = pts * pv
    zeros = np.zeros(NSHELLS)
    for j in range(natoms):
        cond[:, j], _ = reshape_profile(sigma[:, j].copy(), w[:, j].astype(np.float64), zeros, zeros, False)

    cref = ref
    cref[...] = 0.0
    _pseudo(cref, cond, 1, everyone, *args)
    pts, s_sqrt, s_ratio = _tau_sums(cond, cref, *args)
    tau = np.ones((NSHELLS, natoms))
    nz = pts > 0
    tau[nz] = s_ratio[nz] / s_sqrt[nz]
    fe = np.zeros((NSHELLS, natoms))
    se = np.zeros((NSHELLS, natoms))
    for j in range(natoms):
        for k in range(NSHELLS):
            t2 = tau[k, j] * tau[k, j]
            first = MINIMUM_BURIED_TAIL_EXPONENT * (1.0 - t2)
            fe[k, j] = math.exp(-first / SCALEFACTOR)
            second = (1.0 - t2) / MAXIMUM_BURIED_TAIL_EXPONENT
            se[k, j] = math.exp(-1.0 / (SCALEFACTOR * second)) if second > 0.01 else 0.0
    del cref
    partial = cond.copy()

    # steps 4-7
    wa = np.ones(natoms)
    nonoverlap = CHARGE_CONVERGENCE_TOLERANCE / (10.0 * WA_RENORMALIZATION_MAX)
    completed = 3
    update_kappa = False
    max_change = 1.0
    wsad = np.zeros((NSHELLS, natoms))
    tpseudo = np.zeros(total.shape, order="F")
    n_iter = 0
    for it in range(4, 501):
        n_iter = it
        tpseudo[...] = 0.0
        _pseudo(tpseudo, partial, 1, everyone, *args)
        sd, swd, pts, swp = _iter_sums(partial, total, tpseudo, *args)
        slope = np.zeros(natoms)
        sad = np.zeros((NSHELLS, natoms))
        new_wsad = np.zeros((NSHELLS, natoms))
        for j in range(natoms):
            for k in range(NSHELLS):
                if pts[k, j] > 0:
                    sad[k, j] = sd[k, j] / pts[k, j]
                    new_wsad[k, j] = (sd[k, j] - swd[k, j] + 0.2 * swp[k, j] * sad[k, j]) / (pts[k, j] - 0.8 * swp[k, j])
                    slope[j] = slope[j] + pv * (sd[k, j] - swd[k, j])
        wsad = new_wsad
        old_net = net
        old_change = max_change
        vpop = _column_sums(sd) * pv
        vpop = vpop + occ1 - core_pop
        norm = nvalence / _seq_sum(vpop)
        vpop = norm * vpop
        net = z - core_e - vpop
        max_change = float(np.max(np.abs(net - old_net)))
        if it == 4:
            update_kappa = False
        else:
            if np.any(vpop < -CHARGE_CONVERGENCE_TOLERANCE):
                update_kappa = True
            if update_kappa and max(max_change, old_change) < CHARGE_CONVERGENCE_TOLERANCE:
                update_kappa = False
                wa[:] = 1.0
        if update_kappa:
            for j in range(natoms):
                partial[:, j] = partial[:, j] / wa[j]
                if slope[j] > nonoverlap:
                    wa[j] = max(1.0, (wa[j] * math.exp(-vpop[j] / slope[j])))
                else:
                    wa[j] = 1.0
                partial[:, j] = partial[:, j] * wa[j]
        else:
            completed += 1
            if completed == 7:
                break
            w = pts * pv
            partial = wsad.copy()
            for j in range(natoms):
                partial[:, j], _ = reshape_profile(wsad[:, j].copy(), w[:, j].astype(np.float64), fe[:, j].copy(),
                                                   se[:, j].copy(), True)
    state.update(net_atomic_charge=net, hirshfeld_net_atomic_charge=hirshfeld, n_iterations=n_iter, normalization=norm,
                 partial_density=partial, total_pseudodensity=tpseudo, total_density=total, tau=tau,
                 sum_points=pts, valence_population=vpop, spherical_average_density=sad)
    return state


# ---------------------------------------------------------------------------------
# P4: atomic multipoles (module_local_multipole_moment_analysis.f08)
# ---------------------------------------------------------------------------------

@numba.njit(parallel=True, cache=True)
def _multipole_sums(partial, total, tpseudo, pcd, core, cpseudo, norm, pv, nabc, shift, boundary, lo, hi):
    """Per atom: dipole (x, y, z) and quadrupole (xy, xz, yz, x2-y2, 3z2-r2) sums of the
    atom's valence density, relative to its nucleus."""
    natoms = nabc.shape[0]
    out = np.zeros((8, natoms))
    for j in numba.prange(natoms):
        for nc in range(lo[j, 2], hi[j, 2] + 1):
            kc = nc + nabc[j, 2] + 1
            for nb in range(lo[j, 1], hi[j, 1] + 1):
                kb = nb + nabc[j, 1] + 1
                for na in range(lo[j, 0], hi[j, 0] + 1):
                    ka = na + nabc[j, 0] + 1
                    x = na * boundary[0, 0] + nb * boundary[1, 0] + nc * boundary[2, 0] - shift[j, 0]
                    y = na * boundary[0, 1] + nb * boundary[1, 1] + nc * boundary[2, 1] - shift[j, 1]
                    z = na * boundary[0, 2] + nb * boundary[1, 2] + nc * boundary[2, 2] - shift[j, 2]
                    s = int(math.ceil(SCALEFACTOR * math.sqrt(x * x + y * y + z * z) + ZERO_TOLERANCE))
                    if s <= NSHELLS and tpseudo[ka - 1, kb - 1, kc - 1] > ZERO_TOLERANCE:
                        lad = partial[s - 1, j] * total[ka - 1, kb - 1, kc - 1] / tpseudo[ka - 1, kb - 1, kc - 1]
                        if cpseudo[ka - 1, kb - 1, kc - 1] >= ZERO_TOLERANCE:
                            lcd = pcd[s - 1, j] * core[ka - 1, kb - 1, kc - 1] / cpseudo[ka - 1, kb - 1, kc - 1]
                        else:
                            lcd = 0.0
                        t = norm * (lad - lcd) * pv
                        out[0, j] = out[0, j] - x * t
                        out[1, j] = out[1, j] - y * t
                        out[2, j] = out[2, j] - z * t
                        out[3, j] = out[3, j] - x * y * t
                        out[4, j] = out[4, j] - x * z * t
                        out[5, j] = out[5, j] - y * z * t
                        out[6, j] = out[6, j] - (x * x - y * y) * t
                        out[7, j] = out[7, j] - (3.0 * z * z - (x * x + y * y + z * z)) * t
    return out


def _eig3(a):
    """module_matrix_operations.f08 eig: sorted eigenvalues of a symmetric 3x3 matrix
    (trigonometric method). The diagonal test uses a default-REAL literal, so the
    threshold is the single-precision value of 1e-10."""
    p1 = a[0][1] * a[0][1] + a[0][2] * a[0][2] + a[1][2] * a[1][2]
    q = (a[0][0] + a[1][1] + a[2][2]) / 3.0
    if p1 < float(np.float32(0.0000000001)):
        e1 = min(a[0][0], a[1][1], a[2][2])
        e3 = max(a[0][0], a[1][1], a[2][2])
        return [e1, (q * 3.0) - e1 - e3, e3]
    p2 = (a[0][0] - q) * (a[0][0] - q) + (a[1][1] - q) * (a[1][1] - q) + (a[2][2] - q) * (a[2][2] - q) + 2.0 * p1
    p = math.sqrt(p2 / 6.0)
    b = [[(1.0 / p) * (a[i][k] - q * (1.0 if i == k else 0.0)) for k in range(3)] for i in range(3)]
    r = _det3(b) / 2.0
    if r <= -1.0:
        phi = PI / 3.0
    elif r >= 1.0:
        phi = 0.0
    else:
        phi = math.acos(r) / 3.0
    e3 = q + 2.0 * p * math.cos(phi)
    e1 = q + 2.0 * p * math.cos(phi + (2.0 * PI / 3.0))
    return [e1, 3.0 * q - e1 - e3, e3]


def atomic_multipoles(state):
    """Atomic dipoles and traceless quadrupoles in atomic units (analytic-pair corrections
    included), and the eigenvalues of each quadrupole tensor (from the grid sums only, as
    in Chargemol)."""
    g = state["geom"]
    m = _multipole_sums(state["partial_density"], state["total_density"], state["total_pseudodensity"],
                        state["partial_core_density"], state["core_density"], state["core_pseudodensity"],
                        state["normalization"], g["pixelvolume"], g["nabc"], g["shift"], g["boundary"], g["lo"], g["hi"])
    occ = state["occupancy_correction"]
    natoms = m.shape[1]
    dip = np.zeros((natoms, 3))
    quad = np.zeros((natoms, 5))
    eigs = np.zeros((natoms, 3))
    for j in range(natoms):
        dip[j] = [m[0, j] + occ[2, j], m[1, j] + occ[3, j], m[2, j] + occ[4, j]]
        quad[j] = [m[3, j] + occ[8, j], m[4, j] + occ[9, j], m[5, j] + occ[10, j],
                   m[6, j] + occ[5, j] - occ[6, j], m[7, j] + 2.0 * occ[7, j] - occ[5, j] - occ[6, j]]
        t = [[0.5 * (m[6, j] - m[7, j] / 3.0), m[3, j], m[4, j]],
             [m[3, j], 0.5 * (-m[6, j] - m[7, j] / 3.0), m[5, j]],
             [m[4, j], m[5, j], m[7, j] / 3.0]]
        eigs[j] = _eig3(t)
    mag = np.array([math.sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]) for d in dip])
    state.update(atomic_dipole=dip, atomic_dipole_mag=mag, atomic_quadrupole=quad, quadrupole_eigenvalues=eigs)
    return state


def cloud_penetration(state):
    """module_cloud_penetration.f08: fit ln(spherical average density) = a - b r over the
    shells beyond 200 pm; returns (a, b, R^2) per atom."""
    sad = state["spherical_average_density"]
    kmin = int(math.ceil(200.0 * NSHELLS / CUTOFF_RADIUS))
    out = np.zeros((sad.shape[1], 3))
    for j in range(sad.shape[1]):
        sx = sxx = sxy = sy = syy = 0.0
        n = 0
        for k in range(kmin, NSHELLS + 1):
            x = (k - 0.5) * (CUTOFF_RADIUS * BOHRPERANGSTROM / (100.0 * NSHELLS))
            if sad[k - 1, j] < ZERO_TOLERANCE ** 1.5:
                continue
            y = math.log(sad[k - 1, j])
            sx = sx + x
            sxx = sxx + x * x
            sxy = sxy + x * y
            sy = sy + y
            syy = syy + y * y
            n = n + 1
        slope = (n * sxy - sx * sy) / (n * sxx - sx * sx)
        intercept = (sy - slope * sx) / n
        r2 = (n * sxy - sx * sy) * (n * sxy - sx * sy) / ((n * sxx - sx * sx) * (n * syy - sy * sy))
        out[j] = [intercept, -slope, r2]
    state["tail_fit"] = out
    return state


# ---------------------------------------------------------------------------------
# P4: collinear spin moments (module_spin_functions, module_collinear_spin_moments_iterator)
# ---------------------------------------------------------------------------------

NUM_LOOKUP_POINTS = 10000
XI_THRESHOLD = 0.001
XI_ZERO_TOLERANCE = 0.000001
SPIN_CONVERGENCE_TOLERANCE = 5.0e-5
SPIN_REF_FRACTION = 1.0 / 2.0


def _xi(tau):
    if tau < 0.0:
        return 0.0
    if tau > 1.0:
        return 2.0 * PI
    if tau < XI_THRESHOLD:
        return 4.0 * PI * tau / 3.0
    if tau > (1.0 - XI_THRESHOLD):
        return 2.0 * PI + (tau - 1.0) * 49.956 + ((tau - 1.0) * (tau - 1.0)) * 8370.6
    return PI * ((1.0 / (tau * tau) - 1.0) * math.log((1.0 - tau) / (1.0 + tau)) + 2.0 / tau)


def _xi_derivative(tau):
    if tau < XI_THRESHOLD:
        return 4.0 * PI / 3.0
    if tau > (1.0 - XI_THRESHOLD):
        return 50.0
    return 2.0 * PI * (math.log((1.0 + tau) / (1.0 - tau)) - 2.0 * tau) / ((tau * tau) * tau)


def _inverse_xi(x):
    if x < 0.0:
        return 0.0
    if x < XI_ZERO_TOLERANCE:
        return 3.0 * x / (4.0 * PI)
    if x > 2.0 * PI - XI_ZERO_TOLERANCE:
        return 1.0
    t = (x - 0.036835 * x * x) / (4.178319 - 0.136129 * x + 0.038148 * x * x)
    d = _xi_derivative(t)
    for _ in range(5):
        t = t + (x - _xi(t)) / d
    return t


def spin_lookup_tables():
    """generate_spin_lookup_tables: Xi(i/N) and inverse Xi(2 pi i/N), i = 1..N."""
    xi = np.array([_xi(i / NUM_LOOKUP_POINTS) for i in range(1, NUM_LOOKUP_POINTS + 1)])
    inv = np.array([_inverse_xi(i * 2.0 * PI / NUM_LOOKUP_POINTS) for i in range(1, NUM_LOOKUP_POINTS + 1)])
    return xi, inv


@numba.njit(inline="always")
def _fast_xi(tau, table):
    thr = 1.0 / NUM_LOOKUP_POINTS
    if tau <= 0.0:
        return 0.0
    if tau >= 1.0:
        return 2.0 * PI
    if tau < thr:
        return 4.0 * PI * tau / 3.0
    i = int(math.floor(tau * NUM_LOOKUP_POINTS))
    f = tau * NUM_LOOKUP_POINTS - i
    return (1.0 - f) * table[i - 1] + f * table[i]


@numba.njit(inline="always")
def _fast_inverse_xi(x, table):
    thr = 1.0 / NUM_LOOKUP_POINTS
    sx = x / (2.0 * PI)
    if sx <= 0.0:
        return 0.0
    if sx >= 1.0:
        return 1.0
    if sx < thr:
        return x * 0.75 / PI
    i = int(math.floor(sx * NUM_LOOKUP_POINTS))
    f = sx * NUM_LOOKUP_POINTS - i
    return (1.0 - f) * table[i - 1] + f * table[i]


@numba.njit(inline="always")
def _theta(a, b, table):
    if abs(b) < 1.0e-10 or a < 1.0e-10:
        return 0.0
    return b * _fast_xi(abs(b) / a, table) / (2.0 * abs(b))


@numba.njit(inline="always")
def _local_spin(s, j, ka, kb, kc, it, partial, total, tpseudo, spin, sad, saas, ups, omega, trial, xi, inv):
    """One atom's spin at one point (0-based pixel) and its L value: the proportional
    spin at spin_iter 1, the Xi-function update afterwards."""
    lad = partial[s - 1, j] * total[ka, kb, kc] / tpseudo[ka, kb, kc]
    prop = partial[s - 1, j] * spin[ka, kb, kc] / tpseudo[ka, kb, kc]
    ptheta = _theta(lad, prop, xi)
    if it == 1:
        return prop, ptheta
    theta = _theta(sad[s - 1, j], saas[s - 1, j], xi)
    L = (ups[ka, kb, kc] - omega[ka, kb, kc] + PI * (spin[ka, kb, kc] - trial[ka, kb, kc]) / (total[ka, kb, kc])
         + (1.0 - SPIN_REF_FRACTION) * theta + SPIN_REF_FRACTION * ptheta)
    mag_l = abs(L)
    sign = -1 if L < 0.0 else 1
    if mag_l > PI:
        mag = lad
    else:
        mag = lad * _fast_inverse_xi(2.0 * mag_l, inv)
    return mag * sign, L


@numba.njit(parallel=True, cache=True)
def _spin_atom_sums(it, partial, total, tpseudo, spin, sad, saas, ups, omega, trial, xi, inv, nabc, shift, boundary,
                    lo, hi):
    natoms = nabc.shape[0]
    ssd = np.zeros((NSHELLS, natoms))
    pts = np.zeros((NSHELLS, natoms), dtype=np.int64)
    for j in numba.prange(natoms):
        for nc in range(lo[j, 2], hi[j, 2] + 1):
            kc = nc + nabc[j, 2]
            for nb in range(lo[j, 1], hi[j, 1] + 1):
                kb = nb + nabc[j, 1]
                for na in range(lo[j, 0], hi[j, 0] + 1):
                    ka = na + nabc[j, 0]
                    s = _shell(na, nb, nc, j, boundary, shift)
                    if s <= NSHELLS and tpseudo[ka, kb, kc] > ZERO_TOLERANCE and total[ka, kb, kc] > ZERO_TOLERANCE:
                        pts[s - 1, j] = pts[s - 1, j] + 1
                        las, L = _local_spin(s, j, ka, kb, kc, it, partial, total, tpseudo, spin, sad, saas, ups,
                                             omega, trial, xi, inv)
                        ssd[s - 1, j] = ssd[s - 1, j] + las
    return ssd, pts


@numba.njit(parallel=True, cache=True)
def _spin_grid_sums(it, partial, total, tpseudo, spin, sad, saas, ups, omega, trial, xi, inv, nabc, shift, boundary,
                    lo, hi, new_ups, new_trial):
    nct = total.shape[2]
    for kc in numba.prange(nct):
        for j in range(nabc.shape[0]):
            nc = kc - nabc[j, 2]
            if nc < lo[j, 2] or nc > hi[j, 2]:
                continue
            for nb in range(lo[j, 1], hi[j, 1] + 1):
                kb = nb + nabc[j, 1]
                for na in range(lo[j, 0], hi[j, 0] + 1):
                    ka = na + nabc[j, 0]
                    s = _shell(na, nb, nc, j, boundary, shift)
                    if s <= NSHELLS and tpseudo[ka, kb, kc] > ZERO_TOLERANCE and total[ka, kb, kc] > ZERO_TOLERANCE:
                        las, L = _local_spin(s, j, ka, kb, kc, it, partial, total, tpseudo, spin, sad, saas, ups,
                                             omega, trial, xi, inv)
                        new_ups[ka, kb, kc] = new_ups[ka, kb, kc] + L * partial[s - 1, j] / tpseudo[ka, kb, kc]
                        new_trial[ka, kb, kc] = new_trial[ka, kb, kc] + las


@numba.njit(parallel=True, cache=True)
def _spin_omega(partial, total, tpseudo, spin, sad, saas, xi, nabc, shift, boundary, lo, hi, omega):
    nct = total.shape[2]
    for kc in numba.prange(nct):
        for j in range(nabc.shape[0]):
            nc = kc - nabc[j, 2]
            if nc < lo[j, 2] or nc > hi[j, 2]:
                continue
            for nb in range(lo[j, 1], hi[j, 1] + 1):
                kb = nb + nabc[j, 1]
                for na in range(lo[j, 0], hi[j, 0] + 1):
                    ka = na + nabc[j, 0]
                    s = _shell(na, nb, nc, j, boundary, shift)
                    if s <= NSHELLS and tpseudo[ka, kb, kc] > ZERO_TOLERANCE and total[ka, kb, kc] > ZERO_TOLERANCE:
                        lad = partial[s - 1, j] * total[ka, kb, kc] / tpseudo[ka, kb, kc]
                        prop = partial[s - 1, j] * spin[ka, kb, kc] / tpseudo[ka, kb, kc]
                        theta = _theta(sad[s - 1, j], saas[s - 1, j], xi)
                        ptheta = _theta(lad, prop, xi)
                        omega[ka, kb, kc] = omega[ka, kb, kc] + ((1.0 - SPIN_REF_FRACTION) * theta
                                                                 + SPIN_REF_FRACTION * ptheta) * partial[s - 1, j] / \
                            tpseudo[ka, kb, kc]


def spin_moments(state):
    """collinear_spin_moments_iterator: DDEC atomic spin moments. Like Chargemol, the
    spin density is not renormalized, so the moments sum to the grid integral of the
    spin density plus the analytic-pair corrections, not exactly to Na - Nb."""
    g = state["geom"]
    args = (g["nabc"], g["shift"], g["boundary"], g["lo"], g["hi"])
    pv = g["pixelvolume"]
    xi, inv = spin_lookup_tables()
    partial, total, tpseudo = state["partial_density"], state["total_density"], state["total_pseudodensity"]
    spin, sad = state["spin_density"], state["spherical_average_density"]
    occ2 = state["occupancy_correction"][1]
    natoms = sad.shape[1]
    saas = np.zeros((NSHELLS, natoms))
    ups = np.zeros(total.shape, order="F")
    omega = np.zeros(total.shape, order="F")
    trial = np.zeros(total.shape, order="F")
    spin_pop = np.zeros(natoms)
    n_iter = 0
    for it in range(1, 501):
        n_iter = it
        ssd, pts = _spin_atom_sums(it, partial, total, tpseudo, spin, sad, saas, ups, omega, trial, xi, inv, *args)
        new_ups = np.zeros(total.shape, order="F")
        new_trial = np.zeros(total.shape, order="F")
        _spin_grid_sums(it, partial, total, tpseudo, spin, sad, saas, ups, omega, trial, xi, inv, *args,
                        new_ups, new_trial)
        ups, trial = new_ups, new_trial
        saas = _shell_average(ssd, pts)
        old = spin_pop
        spin_pop = _column_sums(ssd) * pv
        spin_pop = spin_pop + occ2
        max_change = float(np.max(np.abs(spin_pop - old)))
        if ((max_change < SPIN_CONVERGENCE_TOLERANCE and it > 6)
                or (max(float(np.max(np.abs(spin_pop))), max_change) < SPIN_CONVERGENCE_TOLERANCE and it > 1)):
            break
        omega[...] = 0.0
        _spin_omega(partial, total, tpseudo, spin, sad, saas, xi, *args, omega)
    state.update(spin_moment=spin_pop, spin_iterations=n_iter, total_spin_moment=_seq_sum(spin_pop))
    return state


def run(wfx_path, parallel=True):
    """DDEC6 analysis of one wfx: net atomic charges, atomic multipoles, cloud
    penetration fits and (open shell) atomic spin moments."""
    prep = load(wfx_path)
    grid = grid_setup(prep["atom_xyz"])
    pairs = pair_data(prep, grid)
    grids = core_grids(prep, grid, density_grids(prep, grid, pairs, parallel=parallel))
    net_charge = float(prep["center_zeff"].sum() - (prep["occ_a"].sum() + prep["occ_b"].sum()))
    core_e = grids["core_electrons"]
    nvalence = float(prep["atom_z"].sum()) - net_charge - _seq_sum(core_e)
    int_tol = max(INTEGRATION_TOLERANCE, 0.10 * nvalence / 100.0)
    check = abs(nvalence - grids["valence_density"].sum() * grid["pixelvolume"] - _seq_sum(grids["occupancy_correction"][0]))
    if check > int_tol:
        raise ValueError(f"valence electrons on the grid off by {check:.4f} e (Chargemol stops here)")
    rsv = radial_shell_volume()
    neutral = np.column_stack([oxidation_density(int(z), 0, rsv) for z in prep["atom_z"]])
    state = {"geom": {"nabc": grids["center_nabc"], "shift": grids["center_shift"], "boundary": grid["boundary"],
                      "lo": grids["shell_lo"], "hi": grids["shell_hi"], "pixelvolume": grid["pixelvolume"]},
             "atom_z": np.asarray(prep["atom_z"], dtype=np.int64), "core_electrons": core_e,
             "occupancy_correction": grids["occupancy_correction"], "valence_density": grids["valence_density"],
             "core_density": grids["core_density"], "spin_density": grids["spin_density"], "nvalence": nvalence,
             "radial_shell_volume": rsv, "neutral_density": neutral, "net_charge": net_charge}
    state = core_iterator(state)
    state = ddec6_valence_iterator(state)
    state = cloud_penetration(atomic_multipoles(state))
    if prep["spin_available"]:
        state = spin_moments(state)
    return state


def load(wfx_path):
    return prepare(read_wfx(wfx_path))
