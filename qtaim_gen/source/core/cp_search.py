"""Critical point search for the QTAIM engine (docs/engine_roadmap/qtaim_engine_plan.md, P3).

Multiwfn-compatible mode (seeds="multiwfn") reproduces production's topology
stdin `2 2 3 4 5` (MW/topology.f90:1013-1305) with findcp (MW/topology.f90:2080):
  stage 2  nuclei, with the gradient criterion switched off
  stage 3  midpoints of atom pairs within vdwsumcrit x (vdW_i + vdW_j)
  stage 4  centroids of atom triples whose three pair distances pass that test
  stage 5  centroids of atom quadruples whose six pair distances pass
Each seed runs plain Newton, disp = -H^-1 g, until |disp| < dispconv and
|g| < gradconv (g at the point before the step; the CP is stored after it),
stopping when |det H| < singularcrit or after topomaxcyc steps. A converged
point within minicpdis of an earlier CP is dropped; the type is the number of
positive Hessian eigenvalues (Hessian of the last step). Seeds are deduplicated
in loop order, which is Multiwfn's order on one thread (with several threads
Multiwfn's first-come order depends on scheduling), and each stage is sorted as
sortCP does. All seeds of a stage iterate together, one density_derivatives
call per Newton step.
"""

import numpy as np
from numba import njit

from qtaim_gen.source.core.charge_engine import B2A
from qtaim_gen.source.core.qtaim_engine import density_derivatives

GRADCONV = 1e-6
DISPCONV = 1e-7
MINICPDIS = 0.03  # Bohr
VDWSUMCRIT = 1.5
SINGULARCRIT = 5e-22
TOPOMAXCYC = 120
CP_TYPES = {0: "NCP", 1: "BCP", 2: "RCP", 3: "CCP"}  # by number of positive eigenvalues

# Bondi vdW radii (Angstrom) as Multiwfn's vdwr (MW/define.f90:85-90), index = Z
_VDWR_ANG = np.array(
    [0.3, 1.2, 1.4, 1.82, 1.77, 1.74, 1.7, 1.55, 1.52, 1.47, 1.54, 2.27, 1.73, 1.73, 2.1, 1.8, 1.8, 1.75, 1.88]
    + [2.0] * 9 + [1.63, 1.4, 1.39, 1.87, 2.0, 1.85, 1.9, 1.85, 2.02]
    + [2.0] * 9 + [1.63, 1.72, 1.58, 1.93, 2.17, 2.0, 2.06, 1.98, 2.16]
    + [2.0] * 23 + [1.72, 1.66, 1.55, 1.96, 2.02]
    + [2.0] * 38
)
VDWR = _VDWR_ANG / B2A


@njit(cache=True)
def _close_matrix(coords, radii, crit):
    n = coords.shape[0]
    close = np.zeros((n, n), dtype=np.bool_)
    for i in range(n):
        for j in range(n):
            if i != j:
                d = np.sqrt(((coords[i] - coords[j]) ** 2).sum())
                close[i, j] = d <= crit * (radii[i] + radii[j])
    return close


@njit(cache=True)
def _tuples(close, k):
    """Index tuples i < j < ... in Multiwfn's nested loop order whose atoms are
    pairwise close (k = 2, 3 or 4)."""
    n = close.shape[0]
    out = []
    for i in range(n):
        for j in range(i + 1, n):
            if not close[i, j]:
                continue
            if k == 2:
                out.append((i, j, -1, -1))
                continue
            for m in range(j + 1, n):
                if not (close[i, m] and close[j, m]):
                    continue
                if k == 3:
                    out.append((i, j, m, -1))
                    continue
                for l in range(m + 1, n):
                    if close[i, l] and close[j, l] and close[m, l]:
                        out.append((i, j, m, l))
    res = np.empty((len(out), k), dtype=np.int64)
    for r in range(len(out)):
        for c in range(k):
            res[r, c] = out[r][c]
    return res


def multiwfn_seeds(coords, atnums):
    """The four seed stages of topology options 2-5, each in Multiwfn's loop order."""
    close = _close_matrix(coords, VDWR[atnums], VDWSUMCRIT)
    stages = [("nuclei", coords.copy())]
    for name, k in (("pairs", 2), ("triads", 3), ("quads", 4)):
        idx = _tuples(close, k)
        stages.append((name, coords[idx].mean(axis=1) if len(idx) else np.zeros((0, 3))))
    return stages


def newton(seeds, wfx, basis, gradconv=GRADCONV, dispconv=DISPCONV, maxcyc=TOPOMAXCYC, singular=SINGULARCRIT):
    """findcp's Newton iteration for every seed at once. Returns final positions,
    status (1 converged, -1 singular Hessian, -2 out of cycles), the Hessian of
    the last step and the iteration count."""
    x = np.array(seeds, dtype=float)
    n = len(x)
    status = np.zeros(n, dtype=np.int64)
    hess = np.zeros((n, 3, 3))
    niter = np.zeros(n, dtype=np.int64)
    active = np.arange(n)
    for _ in range(maxcyc):
        if not len(active):
            break
        d = density_derivatives(x[active], wfx, basis)
        g, H = d["gradient"], d["hessian"]
        niter[active] += 1
        sing = np.abs(np.linalg.det(H)) < singular
        status[active[sing]] = -1
        ok = ~sing
        a, g, H = active[ok], g[ok], H[ok]
        disp = -np.einsum("pij,pj->pi", np.linalg.inv(H), g)
        x[a] += disp
        conv = (np.linalg.norm(disp, axis=1) < dispconv) & (np.linalg.norm(g, axis=1) < gradconv)
        status[a[conv]] = 1
        hess[a[conv]] = H[conv]
        active = a[~conv]
    status[active] = -2
    return x, status, hess, niter


def _sort_key(pos):
    # sortCP's "unique code" (MW/topology.f90:2279)
    return pos[:, 0] * 0.234134 + pos[:, 1] * 1.9837322 - pos[:, 2] * 0.5413578924


def search(wfx, basis, seeds="multiwfn", minicpdis=MINICPDIS):
    """CPs as a dict: pos (n, 3) Bohr, type (number of positive eigenvalues),
    eigenvalues, stage, and per-stage seed and Newton statistics."""
    if seeds != "multiwfn":
        raise ValueError(f"unknown seed mode {seeds!r}")
    found_pos, found_type, found_eig, found_stage, stats = [], [], [], [], []
    for stage, pts in multiwfn_seeds(wfx["coords"], wfx["atnums"]):
        new_pos, new_type, new_eig = [], [], []
        if len(pts):
            gconv = 1.0 if stage == "nuclei" else GRADCONV
            x, status, hess, niter = newton(pts, wfx, basis, gradconv=gconv)
            eig = np.linalg.eigvalsh(hess)
            existing = np.array(found_pos).reshape(-1, 3)
            for i in np.flatnonzero(status == 1):
                if np.all(eig[i] == 0):
                    continue
                if len(existing) and np.min(np.linalg.norm(existing - x[i], axis=1)) <= minicpdis:
                    continue
                if new_pos and np.min(np.linalg.norm(np.array(new_pos) - x[i], axis=1)) <= minicpdis:
                    continue
                new_pos.append(x[i])
                new_type.append(int((eig[i] > 0).sum()))
                new_eig.append(eig[i])
            stats.append({"stage": stage, "seeds": len(pts), "converged": int((status == 1).sum()),
                          "singular": int((status == -1).sum()), "maxcyc": int((status == -2).sum()),
                          "new_cps": len(new_pos), "newton_steps": int(niter.sum())})
        else:
            stats.append({"stage": stage, "seeds": 0, "converged": 0, "singular": 0, "maxcyc": 0,
                          "new_cps": 0, "newton_steps": 0})
        if new_pos:
            order = np.argsort(_sort_key(np.array(new_pos)), kind="stable")
            found_pos += [new_pos[i] for i in order]
            found_type += [new_type[i] for i in order]
            found_eig += [new_eig[i] for i in order]
            found_stage += [stage] * len(order)
    return {
        "pos": np.array(found_pos).reshape(-1, 3), "type": np.array(found_type, dtype=np.int64),
        "eigenvalues": np.array(found_eig).reshape(-1, 3), "stage": found_stage, "stats": stats,
    }
