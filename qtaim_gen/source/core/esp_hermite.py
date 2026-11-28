"""Hermite index tables of the McMurchie-Davidson ESP kernels (core/esp.py), in a module of their
own so scripts/gen_esp_contract.py can build core/esp_contract.py without importing esp, which
imports the generated file."""

import numpy as np

LMAX_PAIR = 10  # h + h


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
