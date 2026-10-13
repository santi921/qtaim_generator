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

Engine mode (seeds="delaunay" or "superset") merges only CPs of the same type
within minicpdis (Multiwfn's rule ignores the type and loses one CP of a close
BCP-RCP pair). seeds="delaunay" replaces the O(N k^3) triads and quads with
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
                for q in range(m + 1, n):
                    if close[i, q] and close[j, q] and close[m, q]:
                        out.append((i, j, m, q))
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
           trust=None, accept_maximum=False):
    """findcp's Newton iteration for every seed at once, steps optionally capped
    at trust Bohr. Returns final positions, status (1 converged, 2 converged at a
    maximum by displacement only, -1 singular Hessian, -2 out of cycles), the
    Hessian and gradient of the last step and the iteration count.

    accept_maximum: a seed still unconverged after maxcyc whose last step was
    below dispconv with a negative-definite Hessian gets status 2 instead of -2.
    Seeds that converge normally are untouched. At a heavy all-electron nucleus
    (X2C uranium: exponents to 5.7e7, Hessian -3e15) Newton lands within 5e-15
    Bohr of the maximum, but the gradient there is still ~16 because no closer
    point is representable in double precision; findcp's nuclear criterion
    (|g| < 1) then never holds and the nuclear CP is lost."""
    x = np.array(seeds, dtype=float)
    n = len(x)
    status = np.zeros(n, dtype=np.int64)
    hess = np.zeros((n, 3, 3))
    gnorm = np.full(n, np.inf)
    niter = np.zeros(n, dtype=np.int64)
    active = np.arange(n)
    at_max = np.zeros(n, dtype=bool)
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
        small = np.linalg.norm(disp, axis=1) < dispconv
        conv = small & (gn < gradconv)
        status[a[conv]] = 1
        hess[a[conv]] = H[conv]
        gnorm[a[conv]] = gn[conv]
        if accept_maximum:
            at_max[a] = small & ~conv & np.all(np.linalg.eigvalsh(H) < 0, axis=1)
            hess[a[~conv]] = H[~conv]
            gnorm[a[~conv]] = gn[~conv]
        active = a[~conv]
    status[active] = -2
    status[active[at_max[active]]] = 2
    return x, status, hess, gnorm, niter


def _sort_key(pos):
    # sortCP's "unique code" (MW/topology.f90:2279)
    return pos[:, 0] * 0.234134 + pos[:, 1] * 1.9837322 - pos[:, 2] * 0.5413578924


DEGENERATE_CP = 1e-4  # Bohr: converged points this close are one CP, whatever their type


def _final_eigenvalues(pos, wfx, basis):
    """Hessian eigenvalues at the converged points, the ones CPprop.txt prints:
    typing from the Hessian before Newton's last step (findcp) can disagree in
    sign with them for a near-degenerate CP, which cpprop_integrity rejects."""
    if not len(pos):
        return np.zeros((0, 3))
    return np.linalg.eigvalsh(density_derivatives(pos, wfx, basis)["hessian"])


def _engine_search(wfx, basis, minicpdis, superset):
    stats = []
    cand_x, cand_eig, cand_g, cand_stage = [], [], [], []
    stages = [("nuclei", wfx["coords"].copy(), 1.0)]
    if superset:
        stages += [(name, pts, GRADCONV) for name, pts in multiwfn_seeds(wfx["coords"], wfx["atnums"])[1:]]
    stages.append(("delaunay", delaunay_seeds(wfx["coords"], wfx["atnums"]), GRADCONV))
    for stage, pts, gconv in stages:
        if not len(pts):
            stats.append({"stage": stage, "seeds": 0, "converged": 0, "singular": 0, "maxcyc": 0, "newton_steps": 0,
                          "precision_limited": 0})
            continue
        x, status, hess, gnorm, niter = newton(pts, wfx, basis, gradconv=gconv, trust=TRUST_RADIUS,
                                               accept_maximum=stage == "nuclei")
        ok = np.flatnonzero(status >= 1)
        eig = _final_eigenvalues(x[ok], wfx, basis)
        keep = ~np.all(eig == 0, axis=1)
        cand_x.append(x[ok][keep])
        cand_eig.append(eig[keep])
        cand_g.append(gnorm[ok][keep])
        cand_stage += [stage] * int(keep.sum())
        stats.append({"stage": stage, "seeds": len(pts), "converged": len(ok), "singular": int((status == -1).sum()),
                      "maxcyc": int((status == -2).sum()), "newton_steps": int(niter.sum()),
                      "precision_limited": int((status == 2).sum())})
    x, eig, g = np.concatenate(cand_x), np.concatenate(cand_eig), np.concatenate(cand_g)
    ctype = (eig > 0).sum(axis=1)
    # nuclear CPs first (their gradient criterion is off), then ascending |g|, then position;
    # only CPs of the same type merge: a BCP and an RCP 0.01 Bohr apart (near a bifurcation)
    # are two CPs, and Multiwfn's type-blind rule drops one of them
    rank = np.lexsort((x[:, 2], x[:, 1], x[:, 0], g, np.array([s != "nuclei" for s in cand_stage])))
    kept = []
    for i in rank:
        if kept and np.min(np.linalg.norm(x[kept] - x[i], axis=1)) <= DEGENERATE_CP:
            continue
        same = [k for k in kept if ctype[k] == ctype[i]]
        if same and np.min(np.linalg.norm(x[same] - x[i], axis=1)) <= minicpdis:
            continue
        kept.append(i)
    kept = np.array(kept, dtype=np.int64)
    types = ctype[kept]
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


# ---------------------------------------------------------------- reseeding (P4)

SPHERE_RADIUS = 3.0  # Bohr, Multiwfn's toposphrad
RESEED_ROUNDS = (128, 512, 1000)  # sphere points per atom per round; Multiwfn's exhaustive search uses 1000


def sphere_seeds(coords, n, radius=SPHERE_RADIUS):
    """n deterministic points per atom inside a radius-Bohr ball: a scrambled
    Sobol sequence (fixed seed) in the cube, kept inside the ball, the same
    pattern translated to every nucleus. Multiwfn's sphere search (topology
    option 6) draws 1000 random points per atom in the same ball."""
    from scipy.stats import qmc

    m = 1 << int(np.ceil(np.log2(4 * n)))
    cube = qmc.Sobol(d=3, scramble=True, seed=0).random(m) * 2 - 1
    ball = cube[(cube**2).sum(axis=1) <= 1][:n] * radius
    return (coords[:, None, :] + ball[None, :, :]).reshape(-1, 3)


def _poincare_hopf(types):
    c = np.bincount(types, minlength=4)
    return int(c[0] - c[1] + c[2] - c[3])


def reseed(cps, wfx, basis, rounds=RESEED_ROUNDS, minicpdis=MINICPDIS):
    """While Poincare-Hopf fails on the CPs with rho >= WEAK_DROP (the ones
    analyze keeps; README decision 10), add CPs from sphere seeds around every
    atom, with more points per round. New CPs within minicpdis of a known one
    of the same type are dropped (known CPs are never replaced); among new ones
    the lowest |g| is kept. Rounds stop once the kept CPs or the full set satisfy
    Poincare-Hopf. Returns the updated cps and one stats entry per round run."""
    stats = []
    if not rounds:
        return cps, stats
    rho = density_derivatives(cps["pos"], wfx, basis, hessian=False)["rho"]
    for n in rounds:
        # stop once the kept CPs satisfy it, or the full set does (then the kept set fails only
        # because a CP lies below WEAK_DROP, which more seeds cannot change)
        if _poincare_hopf(cps["type"][rho >= WEAK_DROP]) == 1 or _poincare_hopf(cps["type"]) == 1:
            break
        x, status, hess, gnorm, niter = newton(sphere_seeds(wfx["coords"], n), wfx, basis)
        ok = np.flatnonzero(status == 1)
        eig = _final_eigenvalues(x[ok], wfx, basis)
        good = ~np.all(eig == 0, axis=1)
        ok, eig = ok[good], eig[good]
        added = 0
        for j in np.lexsort((x[ok, 2], x[ok, 1], x[ok, 0], gnorm[ok])):
            p, t = x[ok[j]], int((eig[j] > 0).sum())
            if np.min(np.linalg.norm(cps["pos"] - p, axis=1)) <= DEGENERATE_CP:
                continue
            same = cps["type"] == t
            if same.any() and np.min(np.linalg.norm(cps["pos"][same] - p, axis=1)) <= minicpdis:
                continue
            cps["pos"] = np.vstack([cps["pos"], p])
            cps["type"] = np.append(cps["type"], t)
            cps["eigenvalues"] = np.vstack([cps["eigenvalues"], eig[j]])
            cps["stage"] = cps["stage"] + [f"sphere{n}"]
            rho = np.append(rho, density_derivatives(p[None], wfx, basis, hessian=False)["rho"])
            added += 1
        stats.append({"stage": f"sphere{n}", "seeds": len(x), "converged": len(ok), "singular": int((status == -1).sum()),
                      "maxcyc": int((status == -2).sum()), "new_cps": added, "newton_steps": int(niter.sum()),
                      "poincare_hopf_after": _poincare_hopf(cps["type"][rho >= WEAK_DROP]),
                      "poincare_hopf_full_set_after": _poincare_hopf(cps["type"])})
    return cps, stats


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
            # findpath compares rho here with the value of its last density call, which
            # is the RK2 midpoint of the step that led here (rho_old), not the last point
            ok = rho >= rho_old[active]
            if engine:
                back = active[~ok]
                h[back] *= 0.5
                x[back] = x_prev[back]
                retry = back[h[back] >= PATH_STEP / 64]
                # a retried path steps again from its last accepted point
                if len(retry):
                    gr = density_derivatives(x[retry], wfx, basis, hessian=False)["gradient"]
                    k1 = np.concatenate([k1[ok], gr])
                    active = np.concatenate([active[ok], retry])
                else:
                    active, k1 = active[ok], k1[ok]
            else:
                active, k1 = active[ok], k1[ok]
            k1 = k1 / np.linalg.norm(k1, axis=1)[:, None]
            mid = density_derivatives(x[active] + 0.5 * h[active, None] * k1, wfx, basis, hessian=False)
            k2 = mid["gradient"]
            rho_old[active] = mid["rho"]
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


# ---------------------------------------------------------------- full analysis and CPprop.txt

WEAK_DROP = 1e-5  # rho below this is dropped, after the Poincare-Hopf check (README decision 10)
WEAK_FLAG = 1e-3  # rho below this is flagged
AU2EV, AU2KCAL = 27.2113838, 627.51  # Multiwfn's constants (MW/define.f90:29)
CP_LABELS = {0: "(3,-3)", 1: "(3,-1)", 2: "(3,+1)", 3: "(3,+3)"}


def _canonical_order(cps, ncp_atom, bcp_pairs, bcp_index, coords):
    """NCPs by atom (non-nuclear attractors after them), BCPs by sorted atom
    pair (unpaired after them), RCPs and CCPs by their 3 / 4 nearest atoms;
    ties by position rounded to 1e-6 Bohr."""
    pair_of = {int(b): tuple(sorted(map(int, p))) if p[0] >= 0 else (10**9, 10**9) for b, p in zip(bcp_index, bcp_pairs)}
    keys = []
    for i, (p, t) in enumerate(zip(cps["pos"], cps["type"])):
        rounded = tuple(np.round(p, 6))
        if t == 0:
            k = (ncp_atom[i] if ncp_atom[i] >= 0 else 10**9,)
        elif t == 1:
            k = pair_of[i]
        else:
            k = tuple(sorted(np.argsort(np.linalg.norm(coords - p, axis=1), kind="stable")[: t + 1].tolist()))
        keys.append((int(t), k, rounded))
    return np.array(sorted(range(len(keys)), key=lambda i: keys[i]), dtype=np.int64)


NEAR_DEGENERATE = 0.1  # Bohr: CP pairs closer than this are flagged (plan section 4)


def _near_pairs(pos):
    """Index pairs (i < j) of CPs closer than NEAR_DEGENERATE, e.g. a BCP and an
    RCP about to annihilate; their properties are ill-conditioned."""
    if len(pos) < 2:
        return []
    d = np.linalg.norm(pos[:, None] - pos[None], axis=2)
    i, j = np.where(np.triu(d < NEAR_DEGENERATE, k=1))
    return [(int(a), int(b)) for a, b in zip(i, j)]


def analyze(wfx, basis, seeds="superset", paths="engine", boys_mode="exact", reseed_rounds=None):
    """CP search, properties at every CP, bond paths and atom mapping.

    When Poincare-Hopf fails on the CPs with rho >= WEAK_DROP, reseed adds
    sphere-seed rounds (reseed_rounds, default RESEED_ROUNDS; () disables, and
    seeds="multiwfn" disables it unless given). Both sums are returned:
    poincare_hopf (kept CPs) and poincare_hopf_full_set; then CPs with rho <
    WEAK_DROP are dropped and those with rho < WEAK_FLAG flagged.
    seeds="multiwfn" keeps Multiwfn's CP order (so cp_num matches); other modes
    use the canonical order."""
    from qtaim_gen.source.core.qtaim_engine import point_properties

    cps = search(wfx, basis, seeds=seeds)
    if reseed_rounds is None:
        reseed_rounds = () if seeds == "multiwfn" else RESEED_ROUNDS
    cps, reseed_stats = reseed(cps, wfx, basis, rounds=reseed_rounds)
    cps["stats"] = cps["stats"] + reseed_stats
    props = point_properties(cps["pos"], wfx, basis, esp=True, boys_mode=boys_mode)
    counts = np.bincount(cps["type"], minlength=4)
    ph = int(counts[0] - counts[1] + counts[2] - counts[3])
    keep = np.flatnonzero(props["rho"] >= WEAK_DROP)
    ph_kept = _poincare_hopf(cps["type"][keep])
    cps = {"pos": cps["pos"][keep], "type": cps["type"][keep], "stage": [cps["stage"][i] for i in keep],
           "stats": cps["stats"]}
    props = {k: v[keep] for k, v in props.items()}
    bp = bond_paths(cps, wfx, basis, mode=paths)
    order = np.arange(len(keep)) if seeds == "multiwfn" else _canonical_order(
        cps, bp["ncp_atom"], bp["atoms"], bp["bcp"], wfx["coords"])
    pair = np.full((len(keep), 2), -1, dtype=np.int64)
    pair[bp["bcp"]] = bp["atoms"]
    return {
        "pos": cps["pos"][order], "type": cps["type"][order], "props": {k: v[order] for k, v in props.items()},
        "ncp_atom": bp["ncp_atom"][order], "bcp_atoms": pair[order],
        "weak": np.flatnonzero(props["rho"][order] < WEAK_FLAG),
        "near_pairs": _near_pairs(cps["pos"][order]),
        "poincare_hopf": ph_kept, "poincare_hopf_full_set": ph, "dropped": int(counts.sum() - len(keep)),
        "counts_full_set": counts.tolist(), "stats": cps["stats"],
    }


def _fe(x, w=18, d=10):
    """Fortran Ew.d: 0.dddddddddd E+xx, the E dropped for a three-digit exponent."""
    if not np.isfinite(x):
        return ("NaN" if np.isnan(x) else ("-Infinity" if x < 0 else "Infinity")).rjust(w)
    if x == 0:
        body, exp = "0." + "0" * d, 0
    else:
        s = f"{abs(x):.{d - 1}e}"
        body, exp = "0." + s[0] + s[2 : d + 1], int(s.split("e")[1]) + 1
    sign = "-" if x < 0 else ""
    tail = f"E{exp:+03d}" if abs(exp) <= 99 else f"{exp:+04d}"
    return (sign + body + tail).rjust(w)


def write_cpprop(path, res, wfx):
    """res (analyze) as Multiwfn's CPprop.txt (topology option 7 0, showptprop),
    with the fields the engine computes; parse_qtaim, load_cpprop_full and
    validation.cpprop_integrity read it like Multiwfn's."""
    from qtaim_gen.source.utils.element_classes import SYMBOL_TO_Z

    sym = {z: s for s, z in SYMBOL_TO_Z.items()}
    name = [sym[int(z)].ljust(2) for z in wfx["atnums"]]
    P = res["props"]
    e3 = lambda v: "".join(_fe(x) for x in v)  # noqa: E731
    lines = []
    for i, (p, t) in enumerate(zip(res["pos"], res["type"])):
        lines.append(f" ----------------   CP{i + 1:6d},     Type {CP_LABELS[int(t)]}   ----------------")
        if t == 0:
            a = res["ncp_atom"][i]
            lines.append(" Corresponding nucleus:  Unknown" if a < 0 else f" Corresponding nucleus:{a + 1:6d}({name[a]})")
        elif t == 1 and res["bcp_atoms"][i, 0] >= 0:
            a, b = res["bcp_atoms"][i]
            lines.append(f" Connected atoms:{a + 1:6d}({name[a]})   --{b + 1:6d}({name[b]})")
        lines.append(" Position (Bohr):    " + "".join(f"{x:18.12f}" for x in p))
        lines.append(" Position (Angstrom):" + "".join(f"{x:18.12f}" for x in p * B2A))
        H = P["hessian"][i]
        eig, vec = np.linalg.eigh(H)
        tot = P["esp_total"][i]
        lines += [
            " Density of all electrons:" + _fe(P["rho"][i]),
            " Density of Alpha electrons:" + _fe(P["rho_alpha"][i]),
            " Density of Beta electrons:" + _fe(P["rho_beta"][i]),
            " Spin density of electrons:" + _fe(P["spin"][i]),
            " Lagrangian kinetic energy G(r):" + _fe(P["G"][i]),
            " G(r) in X,Y,Z:" + e3(P["G_xyz"][i]),
            " Hamiltonian kinetic energy K(r):" + _fe(P["K"][i]),
            " Potential energy density V(r):" + _fe(P["V"][i]),
            " Energy density E(r) or H(r):" + _fe(P["H"][i]),
            " Laplacian of electron density:" + _fe(P["laplacian"][i]),
            " Electron localization function (ELF):" + _fe(P["elf"][i]),
            " Localized orbital locator (LOL):" + _fe(P["lol"][i]),
            " Average local ionization energy (ALIE):" + _fe(P["alie"][i]),
            " Delta-g (under Hirshfeld partition):" + _fe(P["delta_g_hirsh"][i]),
            " ESP from nuclear charges:" + _fe(P["esp_nuc"][i]),
            " ESP from electrons:" + _fe(P["esp_e"][i]),
            " Total ESP:" + _fe(tot) + " a.u. (" + _fe(tot * AU2EV, 14, 7) + " eV," + _fe(tot * AU2KCAL, 14, 7) + " kcal/mol)",
            " ",
            " Note: Below information is for electron density",
            " ",
            " Components of gradient in x/y/z are:",
            e3(P["gradient"][i]),
            " Norm of gradient is:" + _fe(np.linalg.norm(P["gradient"][i])),
            " ",
            " Components of Laplacian in x/y/z are:",
            e3(np.diag(H)),
            " Total:" + _fe(np.trace(H)),
            " ",
            " Hessian matrix:",
            *(e3(H[:, j]) for j in range(3)),
            " Eigenvalues of Hessian:" + e3(eig),
            " Eigenvectors (columns) of Hessian:",
            *(e3(vec[r]) for r in range(3)),
            " Determinant of Hessian:" + _fe(np.linalg.det(H)),
            f" Ellipticity of electron density:{eig[0] / eig[1] - 1:12.6f}",
            f" eta index:{abs(eig[0]) / eig[2]:12.6f}",
            " ",
        ]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
