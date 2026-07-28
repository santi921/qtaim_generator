"""Self-contained HORTON charge worker.

Runs inside the dedicated `horton` environment (qc-iodata, qc-grid with
scipy<1.15, gbasis, horton-part, qc-atomdb) and must never import qtaim_gen.
Invoked by qtaim_gen.source.core.horton.run_horton_analysis via subprocess:

    <horton_python> horton_worker.py --wfx FILE --out horton.json \
        --schemes becke,hirshfeld,is --grid fine

Input wfx must already have the Multiwfn EDF block stripped (the orchestrator
does this). Output is charge.json-schema entries under *_horton keys plus a
_meta block. Exit code is nonzero on any failure, including the electron-count
validation gate.
"""

import argparse
import contextlib
import io
import json
import logging
import os
import sys

import numpy as np

NELEC_TOLERANCE = 0.01
PROATOM_RMIN = 1e-5
PROATOM_RMAX = 20.0
PROATOM_NPOINT = 300

# Ground-state multiplicities H-Xe for the atomdb proatom lookup
GROUND_STATE_MULT = {
    1: 2, 2: 1, 3: 2, 4: 1, 5: 2, 6: 3, 7: 4, 8: 3, 9: 2, 10: 1,
    11: 2, 12: 1, 13: 2, 14: 3, 15: 4, 16: 3, 17: 2, 18: 1, 19: 2, 20: 1,
    21: 2, 22: 3, 23: 4, 24: 7, 25: 6, 26: 5, 27: 4, 28: 3, 29: 2, 30: 1,
    31: 2, 32: 3, 33: 4, 34: 3, 35: 2, 36: 1, 37: 2, 38: 1, 39: 2, 40: 3,
    41: 6, 42: 7, 43: 6, 44: 5, 45: 4, 46: 1, 47: 2, 48: 1, 49: 2, 50: 3,
    51: 4, 52: 3, 53: 2, 54: 1,
}

CACHE_DIR = os.path.join(
    os.path.expanduser(os.environ.get("XDG_CACHE_HOME", "~/.cache")),
    "qtaim_gen",
    "proatoms_slater",
)


# Multiwfn's default fuzzy-atom radii ("Modified CSD", covr_tianlu in the
# Multiwfn source, Angstrom): CSD covalent radii, but every main-group element
# (except H, He) takes the group-IVA radius of its row. Used by the becke_csd
# scheme so HORTON Becke cells match Multiwfn's Becke partition.
COVR_TIANLU = {
    1: 0.31, 2: 0.28,
    **{z: 0.76 for z in range(3, 11)},
    **{z: 1.11 for z in range(11, 19)},
    19: 1.2, 20: 1.2,
    21: 1.70, 22: 1.60, 23: 1.53, 24: 1.39, 25: 1.39, 26: 1.32, 27: 1.26,
    28: 1.24, 29: 1.32, 30: 1.22,
    **{z: 1.2 for z in range(31, 37)},
    37: 1.42, 38: 1.42,
    39: 1.90, 40: 1.75, 41: 1.64, 42: 1.54, 43: 1.47, 44: 1.46, 45: 1.42,
    46: 1.39, 47: 1.45, 48: 1.44,
    **{z: 1.39 for z in range(49, 55)},
    55: 1.46, 56: 1.46,
    57: 2.07, 58: 2.04, 59: 2.03, 60: 2.01, 61: 1.99, 62: 1.98, 63: 1.98,
    64: 1.96, 65: 1.94, 66: 1.92, 67: 1.92, 68: 1.89, 69: 1.90, 70: 1.87,
    71: 1.87,
    72: 1.75, 73: 1.70, 74: 1.62, 75: 1.51, 76: 1.44, 77: 1.41, 78: 1.36,
    79: 1.36, 80: 1.32,
    **{z: 1.46 for z in range(81, 89)},
}


def make_tianlu_becke_wpart():
    """BeckeWPart subclass using Multiwfn's modified-CSD radii.

    Defined lazily so importing this module never needs the horton stack.
    """
    from grid.becke import BeckeWeights
    from horton_part import BeckeWPart
    from horton_part.utils import ANGSTROM

    class TianluBeckeWPart(BeckeWPart):
        name = "becke_csd"

        def update_at_weights(self):
            self.logger.info("Computing Becke weights (modified CSD radii).")
            radii_dict = {
                int(z): COVR_TIANLU[int(z)] * ANGSTROM
                for z in set(self.numbers.tolist())
            }
            bw_helper = BeckeWeights(radii_dict, self._k)
            # Multiwfn clips the size-adjustment parameter a_ij at +-0.5
            # (Becke's monotonicity bound); qc-grid hardcodes +-0.45 and its
            # compute_atom_weight cutoff argument is never forwarded, so patch
            # the staticmethod for the duration of the computation.
            orig_alpha = BeckeWeights._calculate_alpha
            BeckeWeights._calculate_alpha = staticmethod(
                lambda radii, cutoff=0.5: orig_alpha(radii, cutoff=0.5)
            )
            try:
                for index in range(self.natom):
                    grid = self.get_grid(index)
                    at_weights = self.cache.load(
                        f"at_weights_{index}", alloc=grid.size
                    )[0]
                    at_weights[:] = bw_helper.compute_atom_weight(
                        grid.points, self.coordinates, self.numbers, index
                    )
            finally:
                BeckeWeights._calculate_alpha = orig_alpha

    return TianluBeckeWPart


def proatom_radial_density(atnum: int) -> tuple:
    """Neutral-atom (r, rho) on an exponential radial grid, cached to disk.

    Densities come from qc-AtomDB's Slater dataset (Koga et al. HF atoms);
    first use per element downloads a small file from the AtomDB data repo.
    """
    cache_file = os.path.join(CACHE_DIR, f"{atnum}.npz")
    if os.path.isfile(cache_file):
        data = np.load(cache_file)
        return data["r"], data["rho"]

    import atomdb
    from iodata.periodic import num2sym

    mult = GROUND_STATE_MULT.get(atnum)
    if mult is None:
        raise ValueError(f"No ground-state multiplicity for Z={atnum}")
    with contextlib.redirect_stderr(io.StringIO()):
        species = atomdb.load(num2sym[atnum], 0, mult, dataset="slater")
    r = np.geomspace(PROATOM_RMIN, PROATOM_RMAX, PROATOM_NPOINT)
    rho = species.dens_func(spin="t")(r)

    os.makedirs(CACHE_DIR, exist_ok=True)
    tmp = cache_file + ".tmp.npz"
    with open(tmp, "wb") as f:
        np.savez(f, r=r, rho=rho)
    os.replace(tmp, cache_file)
    return r, rho


FALLBACK_ANGULAR_DEGREE = 29  # Lebedev 302-point shells


def build_molgrid(atnums, atcoords, preset):
    """Molecular integration grid, with a fallback for elements the qc-grid
    presets do not cover (lanthanides Z=58-71, Z=83-86 as of qc-grid 0.0.9).

    The fallback builds per-atom grids with radial parameters borrowed from
    the nearest covered element (La for the lanthanides, Pb beyond) and a
    uniform angular degree. The electron-count gate downstream still validates
    the result.
    """
    import scipy.constants
    from grid import AtomGrid, BeckeWeights, MolGrid, UniformInteger
    from grid.rtransform import PowerRTransform
    from grid.utils import _DEFAULT_POWER_RTRANSFORM_PARAMS as RGRID_PARAMS

    try:
        return MolGrid.from_preset(atnums, atcoords, preset, store=True), preset
    except (ValueError, KeyError):
        pass

    ang = scipy.constants.angstrom / scipy.constants.value("atomic unit of length")

    def radial_params(z):
        if z in RGRID_PARAMS:
            return RGRID_PARAMS[z]
        if 57 < z < 72:
            return RGRID_PARAMS[57]
        if z > 82:
            return RGRID_PARAMS[82]
        return RGRID_PARAMS[min(RGRID_PARAMS, key=lambda c: abs(c - z))]

    atgrids = []
    for z, coord in zip(atnums, atcoords):
        rmin, rmax, npt = radial_params(int(z))
        rgrid = PowerRTransform(rmin * ang, rmax * ang).transform_1d_grid(
            UniformInteger(npt)
        )
        atgrids.append(
            AtomGrid(rgrid, degrees=[FALLBACK_ANGULAR_DEGREE], center=coord)
        )
    return MolGrid(atnums, atgrids, BeckeWeights(), store=True), "fallback_uniform"


def build_proatomdb(atnums):
    from grid import OneDGrid
    from horton_part.core.proatomdb import ProAtomDB, ProAtomRecord

    records = []
    for atnum in sorted(set(int(n) for n in atnums)):
        r, rho = proatom_radial_density(atnum)
        # trapezoid weights so record moments integrate correctly
        w = np.gradient(r)
        rgrid = OneDGrid(r, w)
        records.append(
            ProAtomRecord(
                number=atnum,
                charge=0,
                energy=0.0,
                rgrid=rgrid,
                rho=rho,
                pseudo_number=atnum,
            )
        )
    return ProAtomDB(records)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wfx", required=True, help="wavefunction file (EDF-free wfx)")
    parser.add_argument("--out", required=True, help="output JSON path")
    parser.add_argument("--schemes", default="becke,hirshfeld,is")
    parser.add_argument("--grid", default="fine", help="MolGrid preset")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    if not args.verbose:
        logging.disable(logging.CRITICAL)

    import warnings

    warnings.filterwarnings("ignore")

    from gbasis.evals.density import evaluate_density
    from gbasis.wrappers import from_iodata
    from iodata import load_one
    from iodata.periodic import num2sym

    mol = load_one(args.wfx)
    basis = from_iodata(mol)
    dm = np.dot(mol.mo.coeffs * mol.mo.occs, mol.mo.coeffs.T)
    molgrid, grid_used = build_molgrid(mol.atnums, mol.atcoords, args.grid)
    rho = evaluate_density(dm, basis, molgrid.points)

    nelec_grid = float(molgrid.integrate(rho))
    nelec_expected = float(mol.nelec)
    if abs(nelec_grid - nelec_expected) > NELEC_TOLERANCE:
        print(
            f"electron-count validation failed: grid={nelec_grid:.4f} "
            f"expected={nelec_expected:.4f} (tol {NELEC_TOLERANCE})",
            file=sys.stderr,
        )
        return 2

    has_ecp = bool((mol.atcorenums != mol.atnums).any())
    atom_keys = [f"{i + 1}_{num2sym[int(n)]}" for i, n in enumerate(mol.atnums)]

    result = {}
    skipped = []
    for scheme in [s.strip() for s in args.schemes.split(",") if s.strip()]:
        if scheme == "becke":
            from horton_part import BeckeWPart

            part = BeckeWPart(mol.atcoords, mol.atnums, mol.atcorenums, molgrid, rho)
        elif scheme == "becke_csd":
            cls = make_tianlu_becke_wpart()
            part = cls(mol.atcoords, mol.atnums, mol.atcorenums, molgrid, rho)
        elif scheme == "is":
            from horton_part import ISAWPart

            part = ISAWPart(mol.atcoords, mol.atnums, mol.atcorenums, molgrid, rho)
        elif scheme == "hirshfeld":
            if has_ecp:
                # all-electron proatoms vs valence-only molecular density
                # would be inconsistent
                skipped.append({"scheme": "hirshfeld", "reason": "ecp_atoms_present"})
                continue
            from horton_part import HirshfeldWPart

            proatomdb = build_proatomdb(mol.atnums)
            part = HirshfeldWPart(
                mol.atcoords, mol.atnums, mol.atcorenums, molgrid, rho, proatomdb
            )
        else:
            print(f"unknown scheme: {scheme}", file=sys.stderr)
            return 2

        part.do_charges()
        charges = part.cache["charges"]
        result[f"{scheme}_horton"] = {
            "charge": {k: round(float(q), 8) for k, q in zip(atom_keys, charges)}
        }

    import gbasis
    import grid as grid_pkg
    import horton_part
    import iodata

    result["_meta"] = {
        "engine": "horton",
        "grid": grid_used,
        "nelec_grid": round(nelec_grid, 6),
        "nelec_expected": round(nelec_expected, 6),
        "has_ecp": has_ecp,
        "schemes_skipped": skipped,
        "versions": {
            "iodata": getattr(iodata, "__version__", "?"),
            "grid": getattr(grid_pkg, "__version__", "?"),
            "gbasis": getattr(gbasis, "__version__", "?"),
            "horton_part": getattr(horton_part, "__version__", "?"),
        },
    }

    tmp = args.out + ".tmp"
    with open(tmp, "w") as f:
        json.dump(result, f, indent=1)
    os.replace(tmp, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
