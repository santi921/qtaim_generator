"""Electrostatic potential at points (docs/engine_roadmap/qtaim_engine_plan.md, P2).

Multiwfn's showptprop prints, per critical point (MW/sub.f90:2405-2420):
  esp_nuc   sum_A Zeff_A / |r - R_A|, exactly 1000 at a nucleus (nucesp, MW/function.f90:4250)
  esp_e     -sum_ij P_ij <g_i| 1/|r' - r| |g_j> over the wfx primitives, no EDF term
  esp_total esp_nuc + esp_e
Production gets esp_e from eleesp1 (MW/function.f90:4283-4420), not libreta: the
QTAIM stdin never initializes libreta. eleesp1 takes the Boys function from Fmch
(MW/function.f90:5011), whose series stops at 1e-8 and whose asymptotic branch uses
sqrt(pi)/2 truncated to 0.88622692, so its values carry ~1e-8 relative error. The
engine uses an exact Boys function by default; boys="multiwfn" reproduces Fmch for
validation only (docs/engine_roadmap/README.md, decision 9).
"""

import math
from concurrent.futures import ThreadPoolExecutor

import numba
import numpy as np
from numba import njit

BOYS_SERIES_MAX_X = 30.0  # series below, erf + upward recursion above
BOYS_TABLE_STEP = 0.05
BOYS_TABLE_TERMS = 8  # Taylor terms; error ~ (step/2)^8 / 8! ~ 4e-17 relative
BOYS_TABLE_MMAX = 12  # highest order served from the table (h-h pairs need 10)


@njit(cache=True)
def boys_reference(mmax, x, out):
    """F_0..F_mmax(x) into out, to ~1e-15 relative. Below BOYS_SERIES_MAX_X:
    F_mmax(x) = e^-x sum_k (2x)^k / ((2mmax+1)(2mmax+3)...(2mmax+2k+1)), then
    downward recursion; above: F_0 from erf, then upward recursion (stable there)."""
    ex = math.exp(-x)
    if x < BOYS_SERIES_MAX_X:
        term = 1.0 / (2 * mmax + 1)
        s = term
        k = 0
        while term > 1e-17 * s:
            k += 1
            term *= 2.0 * x / (2 * mmax + 2 * k + 1)
            s += term
        out[mmax] = ex * s
        for m in range(mmax, 0, -1):
            out[m - 1] = (2.0 * x * out[m] + ex) / (2 * m - 1)
    else:
        out[0] = 0.5 * math.sqrt(math.pi / x) * math.erf(math.sqrt(x))
        for m in range(mmax):
            out[m + 1] = ((2 * m + 1) * out[m] - ex) / (2.0 * x)


def _boys_table():
    n = int(round(BOYS_SERIES_MAX_X / BOYS_TABLE_STEP)) + 1
    mtop = BOYS_TABLE_MMAX + BOYS_TABLE_TERMS
    table = np.empty((n, mtop + 1))
    for i in range(n):
        boys_reference(mtop, i * BOYS_TABLE_STEP, table[i])
    return table


BOYS_TABLE = _boys_table()
_INV_FACT = np.array([1.0 / math.factorial(k) for k in range(BOYS_TABLE_TERMS)])


@njit(cache=True)
def boys(mmax, x, out, table, inv_fact):
    """As boys_reference, but F_mmax below BOYS_SERIES_MAX_X comes from a Taylor
    expansion about the nearest table point (dF_m/dx = -F_{m+1})."""
    ex = math.exp(-x)
    if x < BOYS_SERIES_MAX_X:
        i = int(x / BOYS_TABLE_STEP + 0.5)
        dx = i * BOYS_TABLE_STEP - x
        s = 0.0
        p = 1.0
        for k in range(BOYS_TABLE_TERMS):
            s += table[i, mmax + k] * p * inv_fact[k]
            p *= dx
        out[mmax] = s
        for m in range(mmax, 0, -1):
            out[m - 1] = (2.0 * x * out[m] + ex) / (2 * m - 1)
    else:
        out[0] = 0.5 * math.sqrt(math.pi / x) * math.erf(math.sqrt(x))
        for m in range(mmax):
            out[m + 1] = ((2 * m + 1) * out[m] - ex) / (2.0 * x)


@njit(cache=True)
def fmch(m, x, expnx):
    """Multiwfn's Fmch (MW/function.f90:5011), line for line: F_m(x) with
    expnx = exp(-x); returns 0 where Multiwfn reports non-convergence."""
    eps = 1e-8
    if x <= 10.0:
        if expnx == 0.0:
            return 0.0
        a = m + 0.5
        term = 1.0 / a
        partsum = term
        for _ in range(2, 51):
            a += 1.0
            term = term * x / a
            partsum += term
            if term / partsum < eps:
                return 0.5 * partsum * expnx
        return 0.0
    a = float(m)
    b = a + 0.5
    a = a - 0.5
    xd = 1.0 / x
    approx = 0.88622692 * (math.sqrt(xd) * xd**m)
    for _ in range(m):
        b -= 1.0
        approx *= b
    fimult = 0.5 * expnx * xd
    if fimult == 0.0:
        return approx
    fiprop = fimult / approx
    term = 1.0
    partsum = term
    # DO I=2,NOTRMS with a real NOTRMS = x + m: the bound truncates to an integer
    for _ in range(2, int(x + m) + 1):
        term = term * a * xd
        partsum += term
        if abs(term * fiprop / partsum) < eps:
            return approx - fimult * partsum
        a -= 1.0
    return 0.0


@njit(cache=True)
def boys_multiwfn(mmax, x, out):
    """eleesp1's use of Fmch: F_mmax from Fmch, then downward recursion."""
    ex = math.exp(-x)
    out[mmax] = fmch(mmax, x, ex)
    for m in range(mmax, 0, -1):
        out[m - 1] = (2.0 * x * out[m] + ex) / (2 * m - 1)


# ---------------------------------------------------------------- integrals

BOYS_EXACT, BOYS_REFERENCE, BOYS_MULTIWFN = 0, 1, 2
_BOYS_MODES = {"exact": BOYS_EXACT, "reference": BOYS_REFERENCE, "multiwfn": BOYS_MULTIWFN}
LMAX_PAIR = 10  # h + h
SCREEN_TOL = 1e-15  # drop a group pair whose contribution is provably below this at every point


def _hermite_index():
    """Hermite indices (t, u, v) with t + u + v <= LMAX_PAIR, ordered by total
    order, so the indices of order <= s are a prefix of length NTRI[s]. For each
    index of order >= 1, the MD recursion step that builds it:
    R^m[i] = X_dir R^{m+1}[s1] + c R^{m+1}[s2] (dir = first nonzero of t, u, v)."""
    tuv = [(t, u, s - t - u) for s in range(LMAX_PAIR + 1) for t in range(s, -1, -1) for u in range(s - t, -1, -1)]
    idx = np.full((LMAX_PAIR + 1,) * 3, -1, dtype=np.int64)
    for i, (t, u, v) in enumerate(tuv):
        idx[t, u, v] = i
    n = len(tuv)
    op_dir = np.zeros(n, dtype=np.int64)
    op_s1 = np.zeros(n, dtype=np.int64)
    op_s2 = np.full(n, -1, dtype=np.int64)
    op_c = np.zeros(n)
    order = np.zeros(n, dtype=np.int64)
    for i, c in enumerate(tuv):
        order[i] = sum(c)
        if i == 0:
            continue
        d = next(k for k in range(3) if c[k] > 0)
        lo = list(c)
        lo[d] -= 1
        op_dir[i], op_s1[i] = d, idx[tuple(lo)]
        if c[d] > 1:
            lo[d] -= 1
            op_s2[i], op_c[i] = idx[tuple(lo)], c[d] - 1
    ntri = np.array([(s + 1) * (s + 2) * (s + 3) // 6 for s in range(LMAX_PAIR + 1)], dtype=np.int64)
    return idx, ntri, order, op_dir, op_s1, op_s2, op_c


HIDX, NTRI, HORDER, OP_DIR, OP_S1, OP_S2, OP_C = _hermite_index()


@njit(nogil=True, cache=True)
def _hermite_r(N, X, Y, Z, two_p, F, R, ntri, op_dir, op_s1, op_s2, op_c):
    """Hermite Coulomb integrals R^0_tuv (packed by total order, t + u + v <= N)
    from F_0..F_N: R^m_000 = (-2p)^m F_m, then the MD recursion downwards in m."""
    fac = 1.0
    for m in range(N + 1):
        R[m, 0] = fac * F[m]
        fac *= -two_p
    for m in range(N - 1, -1, -1):
        for i in range(1, ntri[N - m]):
            d = op_dir[i]
            x = X if d == 0 else (Y if d == 1 else Z)
            v = x * R[m + 1, op_s1[i]]
            if op_s2[i] >= 0:
                v += op_c[i] * R[m + 1, op_s2[i]]
            R[m, i] = v


def _hermite_bound_weights(npt=4001, rmax=12.0):
    """Screening weights: R^0_tuv(p, PC) is a coordinate derivative of order
    s = t + u + v of F_0(p |PC|^2), so |R^0_tuv| <= W[s] (2p)^(s/2) with W[s] the
    largest |R^0_tuv| at p = 1/2. Sampled on radial lines along axes, face and
    body diagonals and a few skew directions, then doubled for safety."""
    W = np.zeros(LMAX_PAIR + 1)
    F = np.zeros(LMAX_PAIR + 1)
    R = np.zeros((LMAX_PAIR + 1, NTRI[LMAX_PAIR]))
    dirs = [d / np.linalg.norm(d) for d in (np.array(v, float) for v in
            [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1), (2, 1, 0), (2, 1, 1), (1, 2, 3)])]
    for r in np.linspace(0.0, rmax, npt):
        for d in dirs:
            X, Y, Z = r * d
            boys_reference(LMAX_PAIR, 0.5 * r * r, F)
            _hermite_r(LMAX_PAIR, X, Y, Z, 1.0, F, R, NTRI, OP_DIR, OP_S1, OP_S2, OP_C)
            a = np.abs(R[0])
            for s in range(LMAX_PAIR + 1):
                lo = 0 if s == 0 else NTRI[s - 1]
                W[s] = max(W[s], a[lo:NTRI[s]].max())
    return 2.0 * W


HBOUND_W = _hermite_bound_weights()


@njit(nogil=True, cache=True)
def _hermite_1d(imax, jmax, pa, pb, p, e):
    """McMurchie-Davidson expansion coefficients E^{ij}_t of a 1D Gaussian
    product (overlap prefactor left out): e[i, j, t] for i <= imax, j <= jmax."""
    e[:, :, :] = 0.0
    e[0, 0, 0] = 1.0
    h = 0.5 / p
    for i in range(imax + 1):
        for j in range(jmax + 1):
            if i == 0 and j == 0:
                continue
            if i > 0:
                src, x = i - 1, pa
                for t in range(i + j + 1):
                    v = x * e[src, j, t]
                    if t > 0:
                        v += h * e[src, j, t - 1]
                    if t + 1 <= src + j:
                        v += (t + 1) * e[src, j, t + 1]
                    e[i, j, t] = v
            else:
                src, x = j - 1, pb
                for t in range(i + j + 1):
                    v = x * e[i, src, t]
                    if t > 0:
                        v += h * e[i, src, t - 1]
                    if t + 1 <= i + src:
                        v += (t + 1) * e[i, src, t + 1]
                    e[i, j, t] = v


@njit(cache=True)
def _group_pairs(coords, g_center, g_alpha, g_start, g_end, lmn, dmat, cutoff, tol, hidx, ntri, horder, wbound):
    """Every group pair g1 <= g2 whose overlap factor exp(-mu AB^2) is above
    exp(-cutoff) and whose contribution bound pref sum_i |D_i| W[s_i] p^(s_i/2)
    reaches tol: exponent sum p, centre P, prefactor 2 pi / p K_AB (doubled off
    the diagonal, as P is symmetric), highest Hermite order L, and the
    density-weighted Hermite coefficients D (packed by total order) flat."""
    ng = g_center.shape[0]
    glmax = np.zeros(ng, dtype=np.int64)
    for g in range(ng):
        for a in range(g_start[g], g_end[g]):
            glmax[g] = max(glmax[g], lmn[a, 0] + lmn[a, 1] + lmn[a, 2])
    cand1 = []
    cand2 = []
    for g1 in range(ng):
        A = coords[g_center[g1]]
        for g2 in range(g1, ng):
            B = coords[g_center[g2]]
            ab2 = (A[0] - B[0]) ** 2 + (A[1] - B[1]) ** 2 + (A[2] - B[2]) ** 2
            if g_alpha[g1] * g_alpha[g2] / (g_alpha[g1] + g_alpha[g2]) * ab2 <= cutoff:
                cand1.append(g1)
                cand2.append(g2)
    ncand = len(cand1)
    p_exp = np.empty(ncand)
    p_ctr = np.empty((ncand, 3))
    pref = np.empty(ncand)
    L = np.empty(ncand, dtype=np.int64)
    off = np.empty(ncand + 1, dtype=np.int64)
    off[0] = 0
    D = np.zeros(ncand * 8)
    ex = np.empty((6, 6, 11))
    ey = np.empty((6, 6, 11))
    ez = np.empty((6, 6, 11))
    k = 0
    for c in range(ncand):
        g1, g2 = cand1[c], cand2[c]
        A = coords[g_center[g1]]
        B = coords[g_center[g2]]
        a, b = g_alpha[g1], g_alpha[g2]
        p = a + b
        P = (a * A + b * B) / p
        ab2 = (A[0] - B[0]) ** 2 + (A[1] - B[1]) ** 2 + (A[2] - B[2]) ** 2
        lk = glmax[g1] + glmax[g2]
        n = ntri[lk]
        base = off[k]
        if base + n > D.shape[0]:
            grown = np.zeros(max(2 * D.shape[0], base + n))
            grown[: D.shape[0]] = D
            D = grown
        D[base : base + n] = 0.0
        _hermite_1d(glmax[g1], glmax[g2], P[0] - A[0], P[0] - B[0], p, ex)
        _hermite_1d(glmax[g1], glmax[g2], P[1] - A[1], P[1] - B[1], p, ey)
        _hermite_1d(glmax[g1], glmax[g2], P[2] - A[2], P[2] - B[2], p, ez)
        for ia in range(g_start[g1], g_end[g1]):
            for ib in range(g_start[g2], g_end[g2]):
                w = dmat[ia, ib]
                if w == 0.0:
                    continue
                i0, i1, i2 = lmn[ia, 0], lmn[ia, 1], lmn[ia, 2]
                j0, j1, j2 = lmn[ib, 0], lmn[ib, 1], lmn[ib, 2]
                for t in range(i0 + j0 + 1):
                    wt = w * ex[i0, j0, t]
                    for u in range(i1 + j1 + 1):
                        wu = wt * ey[i1, j1, u]
                        for v in range(i2 + j2 + 1):
                            D[base + hidx[t, u, v]] += wu * ez[i2, j2, v]
        pk = (2.0 if g1 != g2 else 1.0) * 2.0 * math.pi / p * math.exp(-a * b / p * ab2)
        bound = 0.0
        for i in range(n):
            bound += abs(D[base + i]) * wbound[horder[i]] * (2.0 * p) ** (0.5 * horder[i])
        if pk * bound < tol:
            continue
        p_exp[k] = p
        p_ctr[k] = P
        pref[k] = pk
        L[k] = lk
        off[k + 1] = base + n
        k += 1
    return p_exp[:k].copy(), p_ctr[:k].copy(), pref[:k].copy(), L[:k].copy(), off[: k + 1].copy(), D[: off[k]].copy()


@njit(nogil=True, cache=True)
def _esp_e_points(pts, p_exp, p_ctr, pref, L, off, D, mode, table, inv_fact, ntri, op_dir, op_s1, op_s2, op_c):
    """-sum_ab P_ab <a|1/|r-C||b> at each point C: per group pair, Boys values
    F_n(p |PC|^2), the Hermite Coulomb integrals R^0_tuv, contracted with D."""
    n = pts.shape[0]
    out = np.zeros(n)
    R = np.zeros((LMAX_PAIR + 1, ntri[LMAX_PAIR]))
    F = np.zeros(LMAX_PAIR + 1)
    for ip in range(n):
        cx, cy, cz = pts[ip, 0], pts[ip, 1], pts[ip, 2]
        total = 0.0
        for k in range(p_exp.shape[0]):
            p = p_exp[k]
            X = p_ctr[k, 0] - cx
            Y = p_ctr[k, 1] - cy
            Z = p_ctr[k, 2] - cz
            T = p * (X * X + Y * Y + Z * Z)
            N = L[k]
            if mode == BOYS_EXACT:
                boys(N, T, F, table, inv_fact)
            elif mode == BOYS_REFERENCE:
                boys_reference(N, T, F)
            else:
                boys_multiwfn(N, T, F)
            base = off[k]
            if N == 0:
                total += pref[k] * D[base] * F[0]
                continue
            _hermite_r(N, X, Y, Z, 2.0 * p, F, R, ntri, op_dir, op_s1, op_s2, op_c)
            s = 0.0
            for i in range(ntri[N]):
                s += D[base + i] * R[0, i]
            total += pref[k] * s
        out[ip] = -total
    return out


def nuclear_esp(pts, coords, zeff):
    """nucesp: sum_A Zeff_A / |r - R_A|, exactly 1000 at a nucleus."""
    d2 = ((np.asarray(pts, dtype=float)[:, None, :] - coords[None, :, :]) ** 2).sum(axis=2)
    with np.errstate(divide="ignore"):
        v = (zeff / np.sqrt(d2)).sum(axis=1)
    v[(d2 == 0.0).any(axis=1)] = 1000.0
    return v


def esp_pairs(wfx, basis, cutoff=60.0, tol=SCREEN_TOL):
    """Point-independent part of the electronic ESP: the screened group pairs
    with their density-weighted Hermite coefficients."""
    ct = basis["ct"]
    dmat = (ct * basis["occ"]) @ ct.T
    return _group_pairs(wfx["coords"], basis["g_center"], basis["g_alpha"], basis["g_start"],
                        basis["g_end"], basis["lmn"], dmat, cutoff, tol, HIDX, NTRI, HORDER, HBOUND_W)


def electrostatic_potential(pts, wfx, basis, pairs=None, boys_mode="exact", block=None):
    """(esp_nuc, esp_e, esp_total) at each point, as showptprop prints them.
    The EDF core density is left out, as Multiwfn does. Points are split into
    blocks (default: 4 per worker thread, at most 64 points) run on
    numba.get_num_threads() workers; each point's sum is independent of the
    blocking, so the output does not depend on the thread count."""
    pts = np.ascontiguousarray(pts, dtype=float)
    if pairs is None:
        pairs = esp_pairs(wfx, basis)
    mode = _BOYS_MODES[boys_mode]
    ele = np.empty(len(pts))
    nthreads = numba.get_num_threads()
    if block is None:
        block = max(1, min(64, -(-len(pts) // (4 * nthreads))))

    def one_block(s0):
        ele[s0 : s0 + block] = _esp_e_points(pts[s0 : s0 + block], *pairs, mode, BOYS_TABLE, _INV_FACT,
                                             NTRI, OP_DIR, OP_S1, OP_S2, OP_C)

    starts = range(0, len(pts), block)
    if nthreads == 1 or len(starts) <= 1:
        for s0 in starts:
            one_block(s0)
    else:
        with ThreadPoolExecutor(nthreads) as pool:
            list(pool.map(one_block, starts))
    nuc = nuclear_esp(pts, wfx["coords"], wfx["zeff"])
    return nuc, ele, nuc + ele
