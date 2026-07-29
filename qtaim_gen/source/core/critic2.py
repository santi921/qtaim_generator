"""Critic2 QTAIM bond-critical-point engine (cross-validation benchmark).

Runs Critic2 on a job's orca.wfx and writes a `critic2.json` sidecar whose bond
entries use the same field names and `"<i>_<j>"` 0-based atom-pair keys as the
shipped Multiwfn `qtaim.json`, so comparing the two codes is a dict lookup.

Deliberately narrower than the HORTON charge engine (core/horton.py): this is a
standalone benchmark, not a pipeline stage. Nothing here is wired into
gbw_analysis or the LMDB/converter path, and it never merges into qtaim.json.

Two facts drive the implementation, both verified against Multiwfn on real jobs:

- Critic2 consumes the **raw** wfx, unlike HORTON which needs the EDF block
  stripped. Do not reuse core.horton.strip_edf here.
- On ECP jobs Critic2 does not consume the wfx EDF core density
  ("Use core densities? F"), which spawns a spurious non-nuclear attractor
  near the ECP atom. BCP *properties* are unaffected (rho/Laplacian match
  Multiwfn to 5 decimals, since BCPs sit in the valence region), but one
  attractor of the affected BCP points at the phantom CP instead of the
  nucleus. Such attractors are remapped to the nearest real atom and recorded
  in `_meta`.
"""

import json
import logging
import os
import re
import subprocess
import time
from typing import Optional

import numpy as np

BOHR_TO_ANG = 0.529177249

# Critic2 POINTPROP shorthands -> shipped qtaim.json field names. All verified
# to match Multiwfn to 6 decimals, sign conventions included, except elf/lol
# which use a different uniform-gas reference convention in each code.
POINTPROP_MAP = {
    "gkin": "Lagrangian_K",
    "kkin": "Hamiltonian_K",
    "he": "energy_density",
    "elf": "e_loc_func",
    "lol": "lol",
}
DEFAULT_POINTPROPS = tuple(POINTPROP_MAP)

# Fields whose definition differs between the two codes; comparable but not
# expected to agree numerically (report as convention differences).
CONVENTION_DIVERGENT = ("e_loc_func", "lol")

# Critic2 assigns a BCP to the atoms its gradient path terminates at, which is
# more rigorous than a distance heuristic - but in metal/ECP systems, where the
# core density is missing, the integration can terminate at a distant atom. A
# claimed attractor is overridden (in favour of the two nearest atoms, which is
# Multiwfn's own add_closest_atoms_to_bond heuristic) only when it is
# implausible by this factor relative to the second-nearest atom, so genuinely
# curved bond paths keep Critic2's assignment.
PAIR_IMPLAUSIBLE_FACTOR = 1.5

DECK_NAME = "critic2_run.cri"
CRO_NAME = "critic2_run.cro"
CPREPORT_NAME = "critic2_cps.json"


def write_critic2_deck(
    wfx_name: str,
    cpreport_name: str = CPREPORT_NAME,
    discard: float = 1e-5,
    pointprops: tuple = DEFAULT_POINTPROPS,
) -> str:
    """Build a molecular CP-search deck. Paths are relative; run with cwd=folder.

    `discard` prunes spurious low-density CPs; 1e-5 a.u. is the value the
    Critic2 manual recommends for molecules.
    """
    lines = [
        f"molecule {wfx_name}",
        f"load {wfx_name} id rho",
        "reference rho",
    ]
    lines += [f"pointprop {p}" for p in pointprops]
    lines += [
        f'auto discard "$rho < {discard:g}"',
        f"cpreport {cpreport_name}",
        "end",
        "",
    ]
    return "\n".join(lines)


def _pos_ang(cp_or_atom: dict, centering_vector) -> list:
    """Cartesian coords in Angstrom. The cpreport JSON is in bohr and molecular
    coordinates are offset by the supercell centering vector."""
    return [
        (c + centering_vector[i]) * BOHR_TO_ANG
        for i, c in enumerate(cp_or_atom["cartesian_coordinates"])
    ]


def parse_critic2_cps(cpreport_path: str, validate_pairs: bool = True) -> dict:
    """Parse a Critic2 cpreport JSON into qtaim.json-compatible BCP entries.

    Returns {"<i>_<j>": {<multiwfn field names>}, ..., "_meta": {...}} with
    0-based atom-pair keys sorted ascending, matching the shipped schema.

    With validate_pairs, implausible attractor assignments are corrected to the
    two nearest atoms and recorded in `_meta.pair_corrections`; see
    PAIR_IMPLAUSIBLE_FACTOR.
    """
    with open(cpreport_path, "r") as f:
        data = json.load(f)

    struct = data["structure"]
    cv = struct["molecule_centering_vector"]
    atom_pos = np.array([_pos_ang(a, cv) for a in struct["cell_atoms"]])
    n_atoms = len(atom_pos)

    neq = data["critical_points"]["nonequivalent_cps"]
    cell = data["critical_points"]["cell_cps"]
    by_id = {c["id"]: c for c in neq}
    # attractors' cell_id indexes the cell CP list; for a molecule that list is
    # 1:1 with nonequivalent_cps, but resolve it properly rather than assuming.
    cell_by_id = {c["id"]: c for c in cell}

    counts = {"nucleus": 0, "bond": 0, "ring": 0, "cage": 0}
    for c in neq:
        counts[{-3: "nucleus", -1: "bond", 1: "ring", 3: "cage"}[c["signature"]]] += 1
    # Poincare-Hopf: must be 1 for a molecule; anything else flags a CP search
    # that missed or invented critical points.
    ph_sum = counts["nucleus"] - counts["bond"] + counts["ring"] - counts["cage"]

    result = {}
    nna_remapped = []
    collisions = []
    pair_corrections = []
    for c in cell:
        if c["signature"] != -1:
            continue
        props = by_id.get(c["nonequivalent_id"])
        if props is None:
            continue

        pair = []
        for att in c.get("attractors", []):
            idx = att["cell_id"] - 1
            if idx < n_atoms:
                pair.append(idx)
                continue
            # Non-nuclear attractor (ECP core artifact): attribute the bond to
            # the nearest real atom rather than discarding a physical BCP.
            phantom_cell = cell_by_id.get(att["cell_id"])
            phantom = (
                by_id.get(phantom_cell["nonequivalent_id"])
                if phantom_cell
                else by_id.get(att["cell_id"])
            )
            if phantom is None:
                continue
            dists = np.linalg.norm(
                atom_pos - np.array(_pos_ang(phantom, cv)), axis=1
            )
            nearest = int(np.argmin(dists))
            pair.append(nearest)
            nna_remapped.append(
                {
                    "cp_id": att["cell_id"],
                    "mapped_to_atom": nearest,
                    "distance_ang": round(float(dists[nearest]), 6),
                }
            )
        if len(pair) != 2 or pair[0] == pair[1]:
            continue

        cp_pos = np.array(_pos_ang(props, cv))
        if validate_pairs:
            d = np.linalg.norm(atom_pos - cp_pos, axis=1)
            order = np.argsort(d)
            two_nearest = sorted(int(i) for i in order[:2])
            if sorted(pair) != two_nearest and len(atom_pos) > 2:
                # is a claimed attractor implausibly far, given a closer atom?
                if max(d[pair]) > PAIR_IMPLAUSIBLE_FACTOR * d[order[1]]:
                    pair_corrections.append(
                        {
                            "critic2_pair": sorted(pair),
                            "corrected_pair": two_nearest,
                            "claimed_distances_ang": [
                                round(float(d[i]), 6) for i in sorted(pair)
                            ],
                            "nearest_distances_ang": [
                                round(float(d[i]), 6) for i in two_nearest
                            ],
                        }
                    )
                    pair = two_nearest

        lam = props["hessian_eigenvalues"]  # ascending
        entry = {
            "density_all": props["field"],
            "lap_e_density": props["laplacian"],
            "grad_norm": props["gradient_norm"],
            "eig_hess": float(sum(lam)),  # Multiwfn stores the sum, not the triple
            "det_hessian": float(lam[0] * lam[1] * lam[2]),
            "ellip_e_dens": float(lam[0] / lam[1] - 1) if lam[1] else float("nan"),
            "eta": float(abs(lam[0]) / lam[2]) if lam[2] else float("nan"),
            "pos_ang": [float(x) for x in cp_pos],
            "hessian_eigenvalues": list(lam),
        }
        for pp in props.get("pointprops", []):
            name = POINTPROP_MAP.get(pp["name"])
            if name:
                entry[name] = pp["value"]
        # spin-resolved density is emitted natively for open-shell wavefunctions
        if "field_spin_up" in props and "field_spin_down" in props:
            up, dn = props["field_spin_up"], props["field_spin_down"]
            entry["density_alpha"] = up
            entry["density_beta"] = dn
            entry["spin_density"] = up - dn
        # Critic2-only bonus: true bond-path length (Multiwfn ships no BPL)
        paths = [a.get("path_length") for a in c.get("attractors", [])]
        if len(paths) == 2 and all(p is not None for p in paths):
            entry["bond_path_length_ang"] = float(sum(paths) * BOHR_TO_ANG)

        key = f"{min(pair)}_{max(pair)}"
        if key in result:
            # Two BCPs claiming one pair (possible after NNA remapping): keep
            # the denser one, which is the physical bond.
            collisions.append(key)
            if entry["density_all"] <= result[key]["density_all"]:
                continue
        result[key] = entry

    result["_meta"] = {
        "engine": "critic2",
        "n_atoms": n_atoms,
        "cp_counts": counts,
        "poincare_hopf_sum": ph_sum,
        "poincare_hopf_ok": ph_sum == 1,
        "n_bcps_resolved": sum(1 for k in result if k != "_meta"),
        "nna_remapped": nna_remapped,
        "pair_corrections": pair_corrections,
        "pair_collisions": collisions,
        "source_units": data.get("units"),
        "field_type": data.get("field", {}).get("type"),
    }
    return result


def _critic2_version(cro_path: str) -> Optional[str]:
    if not os.path.isfile(cro_path):
        return None
    with open(cro_path, "r", errors="replace") as f:
        for _ in range(40):
            line = f.readline()
            if not line:
                break
            # e.g. "+ critic2 (development), version 1.1"
            m = re.search(r"critic2\s*(\([^)]*\))?,?\s*version\s*(\S+)", line, re.I)
            if m:
                build = (m.group(1) or "").strip("()")
                ver = m.group(2).strip().rstrip(".,")
                return f"{ver} ({build})" if build else ver
    return None


def run_critic2_analysis(
    folder: str,
    critic2_cmd: str = "critic2",
    discard: float = 1e-5,
    pointprops: tuple = DEFAULT_POINTPROPS,
    timeout: int = 1800,
    overwrite: bool = False,
    keep_intermediates: bool = True,
    logger: Optional[logging.Logger] = None,
) -> bool:
    """Run Critic2 on folder's orca.wfx and write critic2.json. True on success.

    Never raises: every failure path logs and returns False, matching
    run_horton_analysis's contract.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    from qtaim_gen.source.core.horton import find_wfx

    out_path = os.path.join(folder, "critic2.json")
    if (
        not overwrite
        and os.path.isfile(out_path)
        and os.path.getsize(out_path) > 0
    ):
        logger.info("critic2.json already present in %s -- skipping", folder)
        return True

    wfx_path = find_wfx(folder)
    if wfx_path is None:
        logger.info("No orca.wfx found in %s -- skipping Critic2", folder)
        return False

    # Critic2 writes its outputs relative to cwd; run in the wfx's directory.
    work_dir = os.path.dirname(wfx_path)
    deck_path = os.path.join(work_dir, DECK_NAME)
    cro_path = os.path.join(work_dir, CRO_NAME)
    cpreport_path = os.path.join(work_dir, CPREPORT_NAME)

    t_start = time.time()
    try:
        with open(deck_path, "w") as f:
            f.write(
                write_critic2_deck(
                    os.path.basename(wfx_path),
                    cpreport_name=CPREPORT_NAME,
                    discard=discard,
                    pointprops=pointprops,
                )
            )

        result = subprocess.run(
            [critic2_cmd, DECK_NAME, CRO_NAME],
            cwd=work_dir,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode != 0:
            logger.error(
                "Critic2 failed for %s (rc=%d): %s",
                folder,
                result.returncode,
                (result.stderr or result.stdout or "").strip()[-500:],
            )
            return False
        if not os.path.isfile(cpreport_path):
            logger.error("Critic2 wrote no cpreport JSON for %s", folder)
            return False

        parsed = parse_critic2_cps(cpreport_path)
        parsed["_meta"]["critic2_version"] = _critic2_version(cro_path)
        parsed["_meta"]["discard"] = discard
        parsed["_meta"]["elapsed_s"] = round(time.time() - t_start, 2)

        tmp = out_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(parsed, f, indent=1)
        os.replace(tmp, out_path)

        meta = parsed["_meta"]
        logger.info(
            "Critic2 %s: %d BCPs, PH=%d%s%s in %.1f s",
            folder,
            meta["n_bcps_resolved"],
            meta["poincare_hopf_sum"],
            f", {len(meta['nna_remapped'])} NNA remapped"
            if meta["nna_remapped"]
            else "",
            "" if meta["poincare_hopf_ok"] else " [PH != 1]",
            meta["elapsed_s"],
        )

        if not keep_intermediates:
            for p in (deck_path, cro_path, cpreport_path):
                try:
                    os.remove(p)
                except OSError:
                    pass
        return True

    except subprocess.TimeoutExpired:
        logger.error("Critic2 timed out after %d s for %s", timeout, folder)
        return False
    except Exception as e:
        logger.error("Error in Critic2 analysis for %s: %s", folder, e)
        return False
