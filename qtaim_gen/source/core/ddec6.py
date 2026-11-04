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


def load(wfx_path):
    return prepare(read_wfx(wfx_path))
