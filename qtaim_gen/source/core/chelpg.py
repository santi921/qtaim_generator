"""CHELPG charges as Multiwfn's production step computes them ("7\\n12\\n1\\nn\\n0\\n0\\nq\\n",
or the radii-file variant for molecules with an actinide): fitESP(2), MW/population.f90:2743.

  points   setCHELPGpt (population.f90:3345): a cube around the nuclei extended by 2.8
           Angstrom on each side, spacing 0.3 Angstrom, int(len/spacing)+1 steps per axis
           taken 0..n inclusive; a point is kept when it lies outside every atom's radius
           (dropped at d <= r) and within 2.8 Angstrom (d <= extension) of some atom
  radii    setESPfitvdwr: Breneman-Wiberg for H-Ar, vdwr_UFF/1.2 above under ispecial=1;
           a molecule with an actinide reads every radius from chelpg_radii.txt
           (data/multiwfn.py chelpg_radius, Batsanov radii for the actinides)
  ESP      totesp: effective-charge nuclear term + electronic term, no EDF (core/esp.py);
           production Multiwfn evaluates it with libreta (iESPcode=2), the engine exactly
           (boys_mode="multiwfn" reproduces eleesp1, iESPcode=1, for parity checks)
  fit      least squares of sum_A q_A / |r_k - R_A| against the ESP with one Lagrange row
           fixing sum q = sum Zeff - nelec; Multiwfn inverts the bordered matrix (KROUT),
           the engine solves it with LU (np.linalg.solve)
"""

import time

import numpy as np
from numba import njit, prange

from qtaim_gen.source.core.charge_engine import B2A
from qtaim_gen.source.data.multiwfn import CHELPG_ACTINIDES, CHELPG_BRENEMAN_WIBERG, chelpg_radius
from qtaim_gen.source.data.multiwfn_tables import VDWR_UFF

FIT_SPACING = 0.566917796573677  # Bohr, Multiwfn's literal for 0.3 Angstrom
FIT_EXTENSION = 5.29123276802099  # Bohr, Multiwfn's literal for 2.8 Angstrom
POINT_BLOCK = 32768  # fitting points per block of the normal equations


def chelpg_radii(atnums):
    """Point-exclusion radius of each atom in Bohr, computed in Multiwfn's order of
    operations: stock radii are Angstrom / b2a, UFF ones (Angstrom / b2a) / 1.2
    (vdwr_UFF is converted to Bohr at startup, Multiwfn.f90:186); with an actinide every
    radius comes from the file, written in Angstrom and divided by b2a on reading."""
    zs = [int(z) for z in atnums]
    if any(z in CHELPG_ACTINIDES for z in zs):
        return np.array([chelpg_radius(z) / B2A for z in zs])
    return np.array([CHELPG_BRENEMAN_WIBERG[z] / B2A if z in CHELPG_BRENEMAN_WIBERG else VDWR_UFF[z] / B2A / 1.2
                     for z in zs])


@njit(cache=True)
def _keep(x, y, z, coords, radii, extdis):
    near = False
    for a in range(coords.shape[0]):
        d = np.sqrt((coords[a, 0] - x) ** 2 + (coords[a, 1] - y) ** 2 + (coords[a, 2] - z) ** 2)
        if d <= radii[a]:
            return False
        if d <= extdis:
            near = True
    return near


@njit(parallel=True, cache=True)
def _grid_points(coords, radii, lo, n, spacing, extdis):
    counts = np.zeros(n[0] + 1, dtype=np.int64)
    for ix in prange(n[0] + 1):
        x = lo[0] + ix * spacing
        c = 0
        for iy in range(n[1] + 1):
            y = lo[1] + iy * spacing
            for iz in range(n[2] + 1):
                if _keep(x, y, lo[2] + iz * spacing, coords, radii, extdis):
                    c += 1
        counts[ix] = c
    start = np.zeros(n[0] + 2, dtype=np.int64)
    start[1:] = np.cumsum(counts)
    pts = np.empty((start[-1], 3))
    for ix in prange(n[0] + 1):
        x = lo[0] + ix * spacing
        k = start[ix]
        for iy in range(n[1] + 1):
            y = lo[1] + iy * spacing
            for iz in range(n[2] + 1):
                z = lo[2] + iz * spacing
                if _keep(x, y, z, coords, radii, extdis):
                    pts[k, 0] = x
                    pts[k, 1] = y
                    pts[k, 2] = z
                    k += 1
    return pts


def chelpg_points(coords, radii, spacing=FIT_SPACING, extdis=FIT_EXTENSION):
    """setCHELPGpt: the fitting points (Bohr), in Multiwfn's x-y-z loop order."""
    coords = np.ascontiguousarray(coords, dtype=float)
    lo = coords.min(axis=0) - extdis
    length = (coords.max(axis=0) + extdis) - lo
    n = (length / spacing).astype(np.int64) + 1
    return _grid_points(coords, np.ascontiguousarray(radii, dtype=float), lo, n, spacing, extdis)


def _inverse_distances(pts, coords):
    return 1.0 / np.sqrt(((pts[:, None, :] - coords[None, :, :]) ** 2).sum(axis=2))


def fit_charges(pts, esp, coords, total_charge):
    """Charges minimizing sum_k (esp_k - sum_A q_A / |r_k - R_A|)^2 with sum_A q_A =
    total_charge, plus the fit diagnostics Multiwfn prints (RMSE, RRMSE) and the
    condition number of the bordered matrix."""
    nat = len(coords)
    amat = np.zeros((nat + 1, nat + 1))
    bvec = np.zeros(nat + 1)
    for s in range(0, len(pts), POINT_BLOCK):
        dinv = _inverse_distances(pts[s : s + POINT_BLOCK], coords)
        amat[:nat, :nat] += dinv.T @ dinv
        bvec[:nat] += dinv.T @ esp[s : s + POINT_BLOCK]
    amat[nat, :nat] = 1.0
    amat[:nat, nat] = 1.0
    bvec[nat] = total_charge
    q = np.linalg.solve(amat, bvec)[:nat]
    sq_err = 0.0
    for s in range(0, len(pts), POINT_BLOCK):
        sq_err += ((esp[s : s + POINT_BLOCK] - _inverse_distances(pts[s : s + POINT_BLOCK], coords) @ q) ** 2).sum()
    diag = {
        "rmse_au": float(np.sqrt(sq_err / len(pts))),
        "rrmse": float(np.sqrt(sq_err / (esp**2).sum())),
        "cond": float(np.linalg.cond(amat)),
    }
    return q, diag


def chelpg(wfx, basis, boys_mode="exact"):
    """(CHELPG charges, _meta entry) for a wavefunction from charge_engine.read_wfx /
    prepare_basis."""
    from qtaim_gen.source.core.esp import electrostatic_potential

    t = {}
    t0 = time.perf_counter()
    coords = wfx["coords"]
    pts = chelpg_points(coords, chelpg_radii(wfx["atnums"]))
    t["points"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    esp = electrostatic_potential(pts, wfx, basis, boys_mode=boys_mode)[2]
    t["esp"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    q, diag = fit_charges(pts, esp, coords, float(wfx["zeff"].sum()) - basis["nelec"])
    t["fit"] = time.perf_counter() - t0
    meta = {"n_points": int(len(pts)), **diag, "boys_mode": boys_mode,
            "timings_s": {k: round(v, 3) for k, v in t.items()}}
    return q, meta
