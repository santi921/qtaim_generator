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

from qtaim_gen.source.core.esp_contract import contract3, contract4, contract5, contract6
from qtaim_gen.source.core.esp_hermite import HIDX, HORDER, LMAX_PAIR, NTRI, OP_C, OP_DIR, OP_S1, OP_S2  # noqa: F401

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
SCREEN_TOL = 1e-15  # dropped group pairs together contribute provably less than this at every point


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


# _hermite_bound_weights() output, stored: computing it takes about 0.7 s at every import.
# Pinned on purpose, so every machine screens with the same weights; tests/test_esp.py checks
# them against a fresh computation to 1e-12 relative.
HBOUND_W = np.array([2.0, 0.5364161998644659, 0.6666666666666666, 0.6593964254193881, 1.2000000000000002,
                     1.8826633707144467, 4.285714285714286, 8.75707560408186, 23.333333333333332,
                     57.13693150788334, 171.8181818181818])


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
    # tol bounds the sum over every dropped pair, not each one
    tol = tol / max(ncand, 1)
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


# ------------------------------------------------- blocked kernel (boys_mode="exact")

# T at and above which F_N(T) equals its asymptotic form (2N-1)!! / (2T)^N sqrt(pi / T) / 2 to
# 3e-17 relative: the dropped part is the regularized upper incomplete gamma Q(N + 1/2, T),
# so the smallest T on a 0.5 grid with Q <= 3e-17. Pinned; tests/test_esp.py recomputes them.
FAR_T = np.array([36.0, 40.5, 43.5, 46.5, 49.5, 52.0, 54.5, 57.0, 59.5, 62.0, 64.0])
NEAR_TMAX = 72.0  # the near-field table covers 0 <= T < NEAR_TMAX > FAR_T.max()
NEAR_MTOP = LMAX_PAIR + BOYS_TABLE_TERMS + 2
_HALF_SQRT_PI = 0.5 * math.sqrt(math.pi)
_RDEN = np.array([1.0 / (2 * m - 1) if m > 0 else 0.0 for m in range(LMAX_PAIR + 2)])


@njit(cache=True)
def _boys_series(mmax, x, out):
    """F_0..F_mmax(x) by the series for F_mmax (stopped at 1e-18 relative) and downward
    recursion; the near-field table is built from it."""
    ex = math.exp(-x)
    term = 1.0 / (2 * mmax + 1)
    s = term
    k = 0
    while term > 1e-18 * s:
        k += 1
        term *= 2.0 * x / (2 * mmax + 2 * k + 1)
        s += term
    out[mmax] = ex * s
    for m in range(mmax, 0, -1):
        out[m - 1] = (2.0 * x * out[m] + ex) / (2 * m - 1)


def _near_tables():
    """F_m(i BOYS_TABLE_STEP), m <= NEAR_MTOP, as rows per m (contiguous in i), and exp(-i step)."""
    n = int(round(NEAR_TMAX / BOYS_TABLE_STEP)) + 2
    tab = np.empty((n, NEAR_MTOP + 1))
    for i in range(n):
        _boys_series(NEAR_MTOP, i * BOYS_TABLE_STEP, tab[i])
    return np.ascontiguousarray(tab.T), np.exp(-np.arange(n) * BOYS_TABLE_STEP)


NEAR_TABLE, NEAR_EXP = _near_tables()


@njit(nogil=True, cache=True, error_model="numpy")
def _esp_e_blocked(pts, p_exp, p_ctr, pref, L, off, D, tabt, etab, inv_fact, rden, far_t, ntri,
                   op_dir, op_s1, op_s2, op_c, B):
    """As _esp_e_points with the exact Boys function, organized for SIMD: group pairs outside,
    blocks of B points inside (pts spatially sorted so a block is compact). Per point, F_m is
    the asymptotic form where T >= far_t[N] and otherwise an 8-term Taylor expansion of F_N
    about the nearest table point with exp(-T) from the table, then downward recursion; the
    contraction is written out for Hermite orders up to 6. Every point's sum runs over the
    pairs in the same order whatever block it is in, so the result does not depend on the
    blocking or the thread count."""
    n = pts.shape[0]
    out = np.zeros(n)
    X = np.empty(B)
    Y = np.empty(B)
    Z = np.empty(B)
    T = np.empty(B)
    h = np.empty(B)
    acc = np.empty(B)
    near = np.empty(B, dtype=np.int64)
    cx = np.empty(B)
    cy = np.empty(B)
    cz = np.empty(B)
    F = np.zeros((LMAX_PAIR + 1, B))
    R = np.zeros((LMAX_PAIR + 1, ntri[LMAX_PAIR], B))
    i7, i6, i5, i4 = inv_fact[7], inv_fact[6], inv_fact[5], inv_fact[4]
    i3, i2 = inv_fact[3], inv_fact[2]
    inv_step = 1.0 / BOYS_TABLE_STEP
    for s0 in range(0, n, B):
        nb = min(B, n - s0)
        bcx = 0.0
        bcy = 0.0
        bcz = 0.0
        for i in range(nb):
            cx[i] = pts[s0 + i, 0]
            cy[i] = pts[s0 + i, 1]
            cz[i] = pts[s0 + i, 2]
            acc[i] = 0.0
            bcx += cx[i]
            bcy += cy[i]
            bcz += cz[i]
        bcx /= nb
        bcy /= nb
        bcz /= nb
        brad = 0.0
        for i in range(nb):
            brad = max(brad, (cx[i] - bcx) ** 2 + (cy[i] - bcy) ** 2 + (cz[i] - bcz) ** 2)
        # a bounding sphere of the block, padded: it only decides which branches are run,
        # each point still takes the branch its own T selects
        brad = math.sqrt(brad) * (1.0 + 1e-12) + 1e-12
        for k in range(p_exp.shape[0]):
            p = p_exp[k]
            N = L[k]
            tf = far_t[N]
            px = p_ctr[k, 0]
            py = p_ctr[k, 1]
            pz = p_ctr[k, 2]
            dc = math.sqrt((px - bcx) ** 2 + (py - bcy) ** 2 + (pz - bcz) ** 2)
            lo = max(dc - brad, 0.0)
            hi = dc + brad
            all_far = p * lo * lo >= tf
            all_near = p * hi * hi < tf
            for i in range(nb):
                x = px - cx[i]
                y = py - cy[i]
                z = pz - cz[i]
                X[i] = x
                Y[i] = y
                Z[i] = z
                T[i] = p * (x * x + y * y + z * z)
            if not all_near:
                for i in range(nb):
                    hh = 1.0 / T[i]
                    F[0, i] = _HALF_SQRT_PI * math.sqrt(hh)
                    h[i] = 0.5 * hh
                for m in range(N):
                    c = 2.0 * m + 1.0
                    for i in range(nb):
                        F[m + 1, i] = c * F[m, i] * h[i]
            if not all_far:
                nn = 0
                for i in range(nb):
                    if T[i] < tf:
                        near[nn] = i
                        nn += 1
                cN, c1, c2, c3 = tabt[N], tabt[N + 1], tabt[N + 2], tabt[N + 3]
                c4, c5, c6, c7 = tabt[N + 4], tabt[N + 5], tabt[N + 6], tabt[N + 7]
                for u in range(nn):
                    i = near[u]
                    t = T[i]
                    j = int(t * inv_step + 0.5)
                    dx = j * BOYS_TABLE_STEP - t
                    s = c7[j] * i7
                    s = c6[j] * i6 + dx * s
                    s = c5[j] * i5 + dx * s
                    s = c4[j] * i4 + dx * s
                    s = c3[j] * i3 + dx * s
                    s = c2[j] * i2 + dx * s
                    s = c1[j] + dx * s
                    f = cN[j] + dx * s
                    F[N, i] = f
                    if N > 0:
                        e = i7
                        e = i6 + dx * e
                        e = i5 + dx * e
                        e = i4 + dx * e
                        e = i3 + dx * e
                        e = i2 + dx * e
                        e = 1.0 + dx * e
                        e = etab[j] * (1.0 + dx * e)
                        t2 = 2.0 * t
                        for m in range(N, 0, -1):
                            f = (t2 * f + e) * rden[m]
                            F[m - 1, i] = f
            base = off[k]
            pk = pref[k]
            if N == 0:
                d0 = pk * D[base]
                for i in range(nb):
                    acc[i] += d0 * F[0, i]
            elif N == 1:
                a1 = -2.0 * p * pk
                d0 = pk * D[base]
                d1 = a1 * D[base + 1]
                d2 = a1 * D[base + 2]
                d3 = a1 * D[base + 3]
                for i in range(nb):
                    acc[i] += d0 * F[0, i] + F[1, i] * (d1 * X[i] + d2 * Y[i] + d3 * Z[i])
            elif N == 2:
                a1 = -2.0 * p * pk
                a2 = 4.0 * p * p * pk
                d0 = pk * D[base]
                d1 = a1 * D[base + 1]
                d2 = a1 * D[base + 2]
                d3 = a1 * D[base + 3]
                dt = a1 * (D[base + 4] + D[base + 7] + D[base + 9])
                e4 = a2 * D[base + 4]
                e5 = a2 * D[base + 5]
                e6 = a2 * D[base + 6]
                e7 = a2 * D[base + 7]
                e8 = a2 * D[base + 8]
                e9 = a2 * D[base + 9]
                for i in range(nb):
                    x = X[i]
                    y = Y[i]
                    z = Z[i]
                    acc[i] += (d0 * F[0, i] + F[1, i] * (d1 * x + d2 * y + d3 * z + dt)
                               + F[2, i] * (x * (e4 * x + e5 * y + e6 * z) + y * (e7 * y + e8 * z) + e9 * z * z))
            elif N == 3:
                contract3(nb, X, Y, Z, F, D, base, pk, p, acc)
            elif N == 4:
                contract4(nb, X, Y, Z, F, D, base, pk, p, acc)
            elif N == 5:
                contract5(nb, X, Y, Z, F, D, base, pk, p, acc)
            elif N == 6:
                contract6(nb, X, Y, Z, F, D, base, pk, p, acc)
            else:
                fac = 1.0
                for m in range(N + 1):
                    for i in range(nb):
                        R[m, 0, i] = fac * F[m, i]
                    fac *= -2.0 * p
                for m in range(N - 1, -1, -1):
                    for q in range(1, ntri[N - m]):
                        d = op_dir[q]
                        s1 = op_s1[q]
                        s2 = max(op_s2[q], 0)
                        c = op_c[q] if op_s2[q] >= 0 else 0.0
                        if d == 0:
                            for i in range(nb):
                                R[m, q, i] = X[i] * R[m + 1, s1, i] + c * R[m + 1, s2, i]
                        elif d == 1:
                            for i in range(nb):
                                R[m, q, i] = Y[i] * R[m + 1, s1, i] + c * R[m + 1, s2, i]
                        else:
                            for i in range(nb):
                                R[m, q, i] = Z[i] * R[m + 1, s1, i] + c * R[m + 1, s2, i]
                for q in range(ntri[N]):
                    dq = pk * D[base + q]
                    for i in range(nb):
                        acc[i] += dq * R[0, q, i]
        for i in range(nb):
            out[s0 + i] = -acc[i]
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
    with their density-weighted Hermite coefficients. tol bounds the total
    contribution of all dropped pairs at any point."""
    lmax = int(basis["lmn"].sum(axis=1).max()) if len(basis["lmn"]) else 0
    if lmax > LMAX_PAIR // 2:
        # the Hermite tables are sized for h + h; numba does not bounds-check them
        raise ValueError(f"primitive angular momentum {lmax} beyond h (5)")
    ct = basis["ct"]
    dmat = (ct * basis["occ"]) @ ct.T
    return _group_pairs(wfx["coords"], basis["g_center"], basis["g_alpha"], basis["g_start"],
                        basis["g_end"], basis["lmn"], dmat, cutoff, tol, HIDX, NTRI, HORDER, HBOUND_W)


def electrostatic_potential(pts, wfx, basis, pairs=None, boys_mode="exact", block=None):
    """(esp_nuc, esp_e, esp_total) at each point, as showptprop prints them.
    The EDF core density is left out, as Multiwfn does. boys_mode="exact" runs the blocked
    kernel (_esp_e_blocked) on the points in Morton order; "reference" and "multiwfn" run the
    per-point kernel (_esp_e_points). Points are split into blocks run on
    numba.get_num_threads() workers; each point's sum is independent of the blocking, so the
    output does not depend on the thread count."""
    from qtaim_gen.source.core.charge_engine import _morton_order

    pts = np.ascontiguousarray(pts, dtype=float)
    if pairs is None:
        pairs = esp_pairs(wfx, basis)
    mode = _BOYS_MODES[boys_mode]
    ele = np.empty(len(pts))
    nthreads = numba.get_num_threads()
    if mode == BOYS_EXACT:
        order = _morton_order(pts) if len(pts) else np.arange(0)
        spts = np.ascontiguousarray(pts[order])
        sele = np.empty(len(pts))
        if block is None:
            block = max(64, 64 * -(-len(pts) // (4 * nthreads * 64)))

        def one_block(s0):
            sele[s0 : s0 + block] = _esp_e_blocked(spts[s0 : s0 + block], *pairs, NEAR_TABLE, NEAR_EXP, _INV_FACT,
                                                   _RDEN, FAR_T, NTRI, OP_DIR, OP_S1, OP_S2, OP_C, 64)
    else:
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
    if mode == BOYS_EXACT:
        ele[order] = sele
    nuc = nuclear_esp(pts, wfx["coords"], wfx["zeff"])
    return nuc, ele, nuc + ele
