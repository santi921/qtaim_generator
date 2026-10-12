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

Engine mode (seeds="delaunay") replaces the O(N k^3) triads and quads with
O(N) seeds from the Delaunay triangulation of the nuclei: nuclei; points at
0.3, 0.5, 0.7 along every vdW-close pair; centroids of triangles and
tetrahedra whose edges all pass the vdW test. Newton steps are capped at
TRUST_RADIUS. Converged points are deduplicated by ascending |g| (then
position), so the CP kept from a cluster does not depend on seed order.
seeds="superset" runs Multiwfn's seeds and the Delaunay ones together: on 88
local jobs the Delaunay seeds alone missed 4 weak (rho 8e-4 to 4e-3) ring and
cage CPs that Multiwfn's triads and quads find.
"""

import numpy as np
from numba import njit
from scipy.spatial import Delaunay, QhullError

from qtaim_gen.source.core.charge_engine import B2A
from qtaim_gen.source.core.qtaim_engine import density_derivatives

GRADCONV = 1e-6
DISPCONV = 1e-7
MINICPDIS = 0.03  # Bohr
VDWSUMCRIT = 1.5
SINGULARCRIT = 5e-22
TOPOMAXCYC = 120
TRUST_RADIUS = None  # Bohr cap on Newton steps (engine mode); off: seeds that run into vacuum then stop fast on det H
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


def delaunay_seeds(coords, atnums):
    """Engine seeds: nuclei, 3 points per vdW-close pair, and centroids of the
    Delaunay triangles and tetrahedra of the nuclei whose edges are all close.
    Fewer than 4 atoms, or nuclei on a plane or line, use Multiwfn's seeds."""
    close = _close_matrix(coords, VDWR[atnums], VDWSUMCRIT)
    try:
        tets = np.sort(Delaunay(coords).simplices, axis=1) if len(coords) >= 4 else None
    except QhullError:
        tets = None
    if tets is None:
        return np.concatenate([p for _, p in multiwfn_seeds(coords, atnums)[1:]])
    pairs = _tuples(close, 2)
    a, b = coords[pairs[:, 0]], coords[pairs[:, 1]]
    along = np.concatenate([a + f * (b - a) for f in (0.3, 0.5, 0.7)])
    faces = np.unique(np.concatenate([tets[:, [0, 1, 2]], tets[:, [0, 1, 3]], tets[:, [0, 2, 3]], tets[:, [1, 2, 3]]]), axis=0)
    faces = faces[close[faces[:, 0], faces[:, 1]] & close[faces[:, 0], faces[:, 2]] & close[faces[:, 1], faces[:, 2]]]
    ok = np.ones(len(tets), dtype=bool)
    for i, j in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)):
        ok &= close[tets[:, i], tets[:, j]]
    tets = np.unique(tets[ok], axis=0)
    return np.concatenate([along, coords[faces].mean(axis=1), coords[tets].mean(axis=1)])


def newton(seeds, wfx, basis, gradconv=GRADCONV, dispconv=DISPCONV, maxcyc=TOPOMAXCYC, singular=SINGULARCRIT,
           trust=None):
    """findcp's Newton iteration for every seed at once, steps optionally capped
    at trust Bohr. Returns final positions, status (1 converged, -1 singular
    Hessian, -2 out of cycles), the Hessian and gradient of the last step and
    the iteration count."""
    x = np.array(seeds, dtype=float)
    n = len(x)
    status = np.zeros(n, dtype=np.int64)
    hess = np.zeros((n, 3, 3))
    gnorm = np.full(n, np.inf)
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
        if trust is not None:
            dn = np.linalg.norm(disp, axis=1)
            big = dn > trust
            disp[big] *= (trust / dn[big])[:, None]
        x[a] += disp
        gn = np.linalg.norm(g, axis=1)
        conv = (np.linalg.norm(disp, axis=1) < dispconv) & (gn < gradconv)
        status[a[conv]] = 1
        hess[a[conv]] = H[conv]
        gnorm[a[conv]] = gn[conv]
        active = a[~conv]
    status[active] = -2
    return x, status, hess, gnorm, niter


def _sort_key(pos):
    # sortCP's "unique code" (MW/topology.f90:2279)
    return pos[:, 0] * 0.234134 + pos[:, 1] * 1.9837322 - pos[:, 2] * 0.5413578924


def _engine_search(wfx, basis, minicpdis, superset):
    stats = []
    cand_x, cand_eig, cand_g, cand_stage = [], [], [], []
    stages = [("nuclei", wfx["coords"].copy(), 1.0)]
    if superset:
        stages += [(name, pts, GRADCONV) for name, pts in multiwfn_seeds(wfx["coords"], wfx["atnums"])[1:]]
    stages.append(("delaunay", delaunay_seeds(wfx["coords"], wfx["atnums"]), GRADCONV))
    for stage, pts, gconv in stages:
        if not len(pts):
            stats.append({"stage": stage, "seeds": 0, "converged": 0, "singular": 0, "maxcyc": 0, "newton_steps": 0})
            continue
        x, status, hess, gnorm, niter = newton(pts, wfx, basis, gradconv=gconv, trust=TRUST_RADIUS)
        ok = np.flatnonzero(status == 1)
        eig = np.linalg.eigvalsh(hess[ok])
        keep = ~np.all(eig == 0, axis=1)
        cand_x.append(x[ok][keep]); cand_eig.append(eig[keep]); cand_g.append(gnorm[ok][keep])
        cand_stage += [stage] * int(keep.sum())
        stats.append({"stage": stage, "seeds": len(pts), "converged": len(ok), "singular": int((status == -1).sum()),
                      "maxcyc": int((status == -2).sum()), "newton_steps": int(niter.sum())})
    x, eig, g = np.concatenate(cand_x), np.concatenate(cand_eig), np.concatenate(cand_g)
    # nuclear CPs first (their gradient criterion is off), then ascending |g|, then position
    rank = np.lexsort((x[:, 2], x[:, 1], x[:, 0], g, np.array([s != "nuclei" for s in cand_stage])))
    kept = []
    for i in rank:
        if kept and np.min(np.linalg.norm(x[kept] - x[i], axis=1)) <= minicpdis:
            continue
        kept.append(i)
    kept = np.array(kept, dtype=np.int64)
    types = (eig[kept] > 0).sum(axis=1)
    order = np.lexsort((_sort_key(x[kept]), types))
    kept = kept[order]
    return {"pos": x[kept], "type": types[order], "eigenvalues": eig[kept],
            "stage": [cand_stage[i] for i in kept], "stats": stats}


def search(wfx, basis, seeds="multiwfn", minicpdis=MINICPDIS):
    """CPs as a dict: pos (n, 3) Bohr, type (number of positive eigenvalues),
    eigenvalues, stage, and per-stage seed and Newton statistics."""
    if seeds in ("delaunay", "superset"):
        return _engine_search(wfx, basis, minicpdis, superset=seeds == "superset")
    if seeds != "multiwfn":
        raise ValueError(f"unknown seed mode {seeds!r}")
    found_pos, found_type, found_eig, found_stage, stats = [], [], [], [], []
    for stage, pts in multiwfn_seeds(wfx["coords"], wfx["atnums"]):
        new_pos, new_type, new_eig = [], [], []
        if len(pts):
            gconv = 1.0 if stage == "nuclei" else GRADCONV
            x, status, hess, _, niter = newton(pts, wfx, basis, gradconv=gconv)
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


# ---------------------------------------------------------------- bond paths

PATH_STEP = 0.03  # Bohr, pathstepsize
PATH_END = 0.05  # Bohr, discritpathfin
PATH_MAXPT = 451  # maxpathpttry
NCP_ATOM_DIST = 0.15 / B2A  # NCP2atmidx: a nucleus within 0.15 Angstrom
ENGINE_PATH_MAXPT = 2000


def ncp_atoms(cps, coords):
    """NCP2atmidx: for each CP, the first atom (0-based) within 0.15 Angstrom
    if it is a nuclear CP, else -1."""
    out = np.full(len(cps["pos"]), -1, dtype=np.int64)
    for i in np.flatnonzero(cps["type"] == 0):
        near = np.flatnonzero(np.linalg.norm(coords - cps["pos"][i], axis=1) < NCP_ATOM_DIST)
        if len(near):
            out[i] = near[0]
    return out


def bond_paths(cps, wfx, basis, mode="multiwfn"):
    """findpath (MW/topology.f90:1903) from every BCP, both directions at once:
    a first step of PATH_STEP along the eigenvector of the positive Hessian
    eigenvalue, then RK2 midpoint steps along grad rho / |grad rho|. A path
    ends at the first CP (in list order, other than its BCP) within PATH_END;
    it fails when rho decreases or after PATH_MAXPT points. Returns, per BCP,
    the two end CP indices (-1 for a failed path) and the path lengths in
    points, and per BCP the connected atoms as BCP2atmidx assigns them (both
    paths must end at nuclear CPs mapped to atoms, else (-1, -1)).

    mode="engine" keeps every trajectory identical until Multiwfn would stop,
    so every pair Multiwfn assigns is kept, and removes its three failure
    modes: only nuclear CPs end a path (Multiwfn also stops at any other CP
    within PATH_END), a rho decrease halves that path's step and retries from
    the last point (down to PATH_STEP / 64), and up to ENGINE_PATH_MAXPT points."""
    engine = mode == "engine"
    if mode not in ("multiwfn", "engine"):
        raise ValueError(f"unknown path mode {mode!r}")
    maxpt = ENGINE_PATH_MAXPT if engine else PATH_MAXPT
    pos = cps["pos"]
    targets = np.flatnonzero(cps["type"] == 0) if engine else np.arange(len(pos))
    bcps = np.flatnonzero(cps["type"] == 1)
    nb = len(bcps)
    ends = np.full((nb, 2), -1, dtype=np.int64)
    npts = np.zeros((nb, 2), dtype=np.int64)
    if nb:
        d = density_derivatives(pos[bcps], wfx, basis)
        eigval, eigvec = np.linalg.eigh(d["hessian"])
        v = eigvec[np.arange(nb), :, np.argmax(eigval > 0, axis=1)]
        x = np.concatenate([pos[bcps] + PATH_STEP * v, pos[bcps] - PATH_STEP * v])
        x_prev = np.concatenate([pos[bcps], pos[bcps]])
        rho_old = np.concatenate([d["rho"], d["rho"]])
        h = np.full(2 * nb, PATH_STEP)
        owner = np.concatenate([bcps, bcps])
        slot = np.concatenate([np.zeros(nb, dtype=np.int64), np.ones(nb, dtype=np.int64)])
        row = np.concatenate([np.arange(nb), np.arange(nb)])
        active = np.arange(2 * nb)
        tpos = pos[targets]
        for ipt in range(2, maxpt + 1):
            if not len(active):
                break
            npts[row[active], slot[active]] = ipt
            dist = np.linalg.norm(x[active, None, :] - tpos[None, :, :], axis=2)
            dist[targets[None, :] == owner[active, None]] = np.inf
            hit = dist < PATH_END
            reached = hit.any(axis=1)
            ends[row[active[reached]], slot[active[reached]]] = targets[np.argmax(hit, axis=1)[reached]]
            active = active[~reached]
            if ipt == maxpt or not len(active):
                break
            g = density_derivatives(x[active], wfx, basis, hessian=False)
            rho, k1 = g["rho"], g["gradient"]
            ok = rho >= rho_old[active]
            if engine:
                back = active[~ok]
                h[back] *= 0.5
                x[back] = x_prev[back]
                retry = back[h[back] >= PATH_STEP / 64]
                # a retried path steps again from its last accepted point, whose rho is rho_old
                if len(retry):
                    gr = density_derivatives(x[retry], wfx, basis, hessian=False)["gradient"]
                    k1 = np.concatenate([k1[ok], gr])
                    active = np.concatenate([active[ok], retry])
                    rho = np.concatenate([rho[ok], rho_old[retry]])
                else:
                    active, rho, k1 = active[ok], rho[ok], k1[ok]
            else:
                active, rho, k1 = active[ok], rho[ok], k1[ok]
            rho_old[active] = rho
            k1 = k1 / np.linalg.norm(k1, axis=1)[:, None]
            k2 = density_derivatives(x[active] + 0.5 * h[active, None] * k1, wfx, basis, hessian=False)["gradient"]
            x_prev[active] = x[active]
            x[active] += h[active, None] * k2 / np.linalg.norm(k2, axis=1)[:, None]
    atom = ncp_atoms(cps, wfx["coords"])
    pairs = np.full((nb, 2), -1, dtype=np.int64)
    for b in range(nb):
        got = [atom[e] for e in ends[b] if e >= 0 and cps["type"][e] == 0]
        # BCP2atmidx: the first NCP end fills i, the second j; an unmapped NCP (atom -1) leaves i empty
        i = j = -1
        for a in got:
            if i == -1:
                i = a
            else:
                j = a
        if i >= 0 and j >= 0:
            pairs[b] = (i, j)
    return {"bcp": bcps, "ends": ends, "npts": npts, "atoms": pairs, "ncp_atom": atom}
