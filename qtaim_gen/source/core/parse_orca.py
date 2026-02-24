"""
Parser for ORCA .out files.

Single-pass, enum-based state machine that extracts electronic structure
properties from ORCA output files. Produces a dict suitable for writing
to orca.json and merging charge/bond data into existing charge.json/bond.json.
"""

import json
import logging
import os
from enum import Enum, auto
from typing import List, Optional, Tuple

from qtaim_gen.source.utils.atomic_write import atomic_json_write

logger = logging.getLogger(__name__)


class OrcaParseState(Enum):
    IDLE = auto()
    SCF_ENERGY_BLOCK = auto()
    SCF_CONVERGENCE = auto()
    ORBITAL_ENERGIES = auto()
    MULLIKEN_CHARGES = auto()
    LOEWDIN_CHARGES = auto()
    LOEWDIN_BOND_ORDERS = auto()
    MAYER_POPULATION = auto()
    MAYER_BOND_ORDERS = auto()
    HIRSHFELD = auto()
    MBIS = auto()
    MBIS_VALENCE = auto()
    GRADIENT = auto()
    DIPOLE = auto()
    QUADRUPOLE = auto()
    ROTATIONAL_CONSTANTS = auto()


# ── Utilities ──────────────────────────────────────────────────────────

def parse_orca_float(value_str: str) -> Optional[float]:
    """Parse a float that may use Fortran D notation or be overflowed."""
    s = value_str.strip()
    if not s or "***" in s:
        return None
    s = s.replace("D", "E").replace("d", "e")
    try:
        return float(s)
    except ValueError:
        return None


def _atom_key(index_0based: int, element: str) -> str:
    """Format atom key as '{1-indexed}_{Element}'."""
    return f"{index_0based + 1}_{element.strip()}"


def _bond_key(i: int, elem_i: str, j: int, elem_j: str) -> str:
    """Format bond key as '{ind}_{Elem}_to_{ind}_{Elem}' (1-indexed)."""
    return f"{i + 1}_{elem_i.strip()}_to_{j + 1}_{elem_j.strip()}"


def _parse_charge_line(line: str) -> Tuple[Optional[int], Optional[str], List[float]]:
    """Parse a Mulliken/Loewdin charge line.

    Handles two ORCA formats:
      1-char element:  "   0 O :   -0.740774"         (parts: 0, O, :, -0.740774)
      2-char element:  "   0 Po:    1.306089"          (parts: 0, Po:, 1.306089)
      UKS with spin:   "   0 Po:   -0.465047  0.342"  (extra spin column)

    Returns (index_0based, element, [charge, spin...]) or (None, None, []).
    """
    parts = line.split()
    if len(parts) < 3:
        return None, None, []
    try:
        idx = int(parts[0])
    except ValueError:
        return None, None, []
    # Format 1: "0 O : -0.74" -> parts[2] == ":"
    if len(parts) >= 4 and parts[2] == ":":
        elem = parts[1]
        vals = [parse_orca_float(p) for p in parts[3:]]
        vals = [v for v in vals if v is not None]
        return idx, elem, vals
    # Format 2: "0 Po: 1.30" -> parts[1] ends with ":"
    if parts[1].endswith(":"):
        elem = parts[1][:-1]
        vals = [parse_orca_float(p) for p in parts[2:]]
        vals = [v for v in vals if v is not None]
        return idx, elem, vals
    return None, None, []


def _parse_run_time(line: str) -> Optional[float]:
    """Parse ORCA run time line to total seconds.

    Format: 'TOTAL RUN TIME: 0 days 0 hours 24 minutes 5 seconds 906 msec'
    """
    try:
        parts = line.split(":")
        if len(parts) < 2:
            return None
        tokens = parts[1].split()
        # tokens: ['0', 'days', '0', 'hours', '24', 'minutes', '5', 'seconds', '906', 'msec']
        d = int(tokens[0])
        h = int(tokens[2])
        m = int(tokens[4])
        s = int(tokens[6])
        ms = int(tokens[8])
        return d * 86400 + h * 3600 + m * 60 + s + ms / 1000.0
    except (ValueError, IndexError):
        return None


def _parse_bond_pairs(line: str) -> List[Tuple[str, float]]:
    """Parse bond order pairs from ORCA format: B(  0-N ,  2-C ) :   1.5313 ..."""
    pairs = []
    parts = line.split("B(")
    for part in parts[1:]:  # skip first empty split
        try:
            bond_part, value_part = part.split(":", 1)
            # parse "  0-N ,  2-C ) "
            bond_part = bond_part.replace(")", "").strip()
            atoms = bond_part.split(",")
            idx_elem_i = atoms[0].strip().split("-")
            idx_elem_j = atoms[1].strip().split("-")
            i = int(idx_elem_i[0].strip())
            elem_i = idx_elem_i[1].strip()
            j = int(idx_elem_j[0].strip())
            elem_j = idx_elem_j[1].strip()
            val = parse_orca_float(value_part.strip().split()[0])
            if val is not None:
                key = _bond_key(i, elem_i, j, elem_j)
                pairs.append((key, val))
        except (ValueError, IndexError):
            continue
    return pairs


def _finalize_orbitals(
    result: dict,
    last_occupied_energy: Optional[float],
    last_occupied_ev: Optional[float],
    first_virtual_energy: Optional[float],
    first_virtual_ev: Optional[float],
    n_electrons: float,
    n_orbitals: int,
) -> None:
    """Write orbital energy results into *result* dict."""
    result["homo_eh"] = last_occupied_energy
    result["homo_ev"] = last_occupied_ev
    result["lumo_eh"] = first_virtual_energy
    result["lumo_ev"] = first_virtual_ev
    if last_occupied_energy is not None and first_virtual_energy is not None:
        result["homo_lumo_gap_eh"] = first_virtual_energy - last_occupied_energy
    result["n_electrons"] = n_electrons
    result["n_orbitals"] = n_orbitals


# ── Main parser ────────────────────────────────────────────────────────

def parse_orca_output(orca_out_path: str) -> dict:
    """
    Parse an ORCA .out file and return a dict of extracted properties.

    Single-pass, line-by-line, enum-based state machine.
    Handles missing sections gracefully (absent keys).
    Handles truncated files (returns partial dict).
    For duplicate sections, last occurrence wins.
    """
    result = {}
    state = OrcaParseState.IDLE

    # Accumulators for multi-line sections
    mulliken_charges = {}
    mulliken_spins = {}
    mulliken_has_spin = False
    loewdin_charges = {}
    loewdin_spins = {}
    loewdin_has_spin = False
    loewdin_bonds = {}
    mayer_population = {}
    mayer_charges = {}
    mayer_bonds = {}
    hirshfeld_charges = {}
    hirshfeld_spins = {}
    mbis_charges = {}
    mbis_populations = {}
    mbis_spins = {}
    mbis_valence_pop = {}
    mbis_valence_width = {}
    gradient = {}
    energy_components = {}

    # Orbital energy tracking
    last_occupied_energy = None
    last_occupied_ev = None
    first_virtual_energy = None
    first_virtual_ev = None
    n_electrons = 0
    n_orbitals = 0

    # SCF convergence
    scf_convergence = {}

    # Dipole/quadrupole accumulators
    dipole_lines = []
    quadrupole_lines = []
    quadrupole_has_tot = False
    rotational_lines = []

    # Mayer state tracking
    mayer_header_skipped = False

    # Hirshfeld state tracking
    hirshfeld_header_done = False

    # MBIS state tracking
    mbis_header_done = False

    # Section line counters for skipping headers
    section_line_count = 0

    try:
        with open(orca_out_path, "r", buffering=8 * 1024 * 1024,
                  errors="replace") as f:
            for line in f:

                # ── IDLE: check triggers ──────────────────────────
                if state == OrcaParseState.IDLE:
                    stripped = line.strip()

                    if "FINAL SINGLE POINT ENERGY" in line and "---" not in line:
                        # Single-line parse, stay IDLE. Last occurrence wins.
                        parts = line.split()
                        for p in parts:
                            val = parse_orca_float(p)
                            if val is not None and val < 0:
                                result["final_energy_eh"] = val
                                result["scf_converged"] = True
                                break

                    elif stripped == "TOTAL SCF ENERGY":
                        state = OrcaParseState.SCF_ENERGY_BLOCK
                        energy_components = {}
                        section_line_count = 0

                    elif stripped == "SCF CONVERGENCE":
                        state = OrcaParseState.SCF_CONVERGENCE
                        scf_convergence = {}
                        section_line_count = 0

                    elif stripped == "ORBITAL ENERGIES":
                        state = OrcaParseState.ORBITAL_ENERGIES
                        last_occupied_energy = None
                        last_occupied_ev = None
                        first_virtual_energy = None
                        first_virtual_ev = None
                        n_electrons = 0
                        n_orbitals = 0
                        section_line_count = 0

                    elif stripped.startswith("MULLIKEN ATOMIC CHARGES"):
                        state = OrcaParseState.MULLIKEN_CHARGES
                        mulliken_charges = {}
                        mulliken_spins = {}
                        mulliken_has_spin = "SPIN" in stripped

                    elif stripped.startswith("LOEWDIN ATOMIC CHARGES"):
                        state = OrcaParseState.LOEWDIN_CHARGES
                        loewdin_charges = {}
                        loewdin_spins = {}
                        loewdin_has_spin = "SPIN" in stripped

                    elif "LOEWDIN BOND ORDERS" in stripped:
                        state = OrcaParseState.LOEWDIN_BOND_ORDERS
                        loewdin_bonds = {}

                    elif "MAYER POPULATION ANALYSIS" in stripped:
                        state = OrcaParseState.MAYER_POPULATION
                        mayer_population = {}
                        mayer_charges = {}
                        mayer_header_skipped = False
                        section_line_count = 0

                    elif "Mayer bond orders larger than" in line:
                        state = OrcaParseState.MAYER_BOND_ORDERS
                        mayer_bonds = {}

                    elif stripped == "HIRSHFELD ANALYSIS":
                        state = OrcaParseState.HIRSHFELD
                        hirshfeld_charges = {}
                        hirshfeld_spins = {}
                        hirshfeld_header_done = False

                    elif stripped == "MBIS ANALYSIS":
                        state = OrcaParseState.MBIS
                        mbis_charges = {}
                        mbis_populations = {}
                        mbis_spins = {}
                        mbis_header_done = False

                    elif "MBIS VALENCE-SHELL" in stripped:
                        state = OrcaParseState.MBIS_VALENCE
                        mbis_valence_pop = {}
                        mbis_valence_width = {}
                        section_line_count = 0

                    elif stripped == "CARTESIAN GRADIENT":
                        state = OrcaParseState.GRADIENT
                        gradient = {}
                        section_line_count = 0

                    elif stripped == "DIPOLE MOMENT":
                        state = OrcaParseState.DIPOLE
                        dipole_lines = []
                        section_line_count = 0

                    elif stripped == "QUADRUPOLE MOMENT":
                        state = OrcaParseState.QUADRUPOLE
                        quadrupole_lines = []
                        quadrupole_has_tot = False
                        section_line_count = 0

                    elif stripped == "Rotational spectrum":
                        state = OrcaParseState.ROTATIONAL_CONSTANTS
                        rotational_lines = []
                        section_line_count = 0

                    # ── Single-line IDLE parsers ──────────────────

                    elif "SCF CONVERGED AFTER" in line and "CYCLES" in line:
                        for part in line.split():
                            try:
                                result["scf_cycles"] = int(part)
                                break
                            except ValueError:
                                continue

                    elif "Norm of the Cartesian gradient" in line and "..." in line:
                        result["gradient_norm"] = parse_orca_float(line.split("...")[1].strip())

                    elif line.startswith("RMS gradient") and "..." in line:
                        result["gradient_rms"] = parse_orca_float(line.split("...")[1].strip())

                    elif line.startswith("MAX gradient") and "..." in line:
                        result["gradient_max"] = parse_orca_float(line.split("...")[1].strip())

                    elif "TOTAL RUN TIME:" in line:
                        result["total_run_time_s"] = _parse_run_time(line)

                    continue  # always continue after IDLE processing

                # ── Non-IDLE: dispatch to section parsers ─────────
                section_line_count += 1

                # ── TOTAL SCF ENERGY ──────────────────────────────
                if state == OrcaParseState.SCF_ENERGY_BLOCK:
                    # Terminate on --- delimiter (next section header)
                    if line.startswith("---") and energy_components:
                        result["energy_components"] = energy_components
                        state = OrcaParseState.IDLE
                        # Don't continue - let IDLE re-check this line next iteration
                        # Actually we need to fall through to IDLE on next line
                        continue
                    if "Nuclear Repulsion" in line:
                        energy_components["nuclear_repulsion_eh"] = parse_orca_float(line.split(":")[1].split()[0])
                    elif "Electronic Energy" in line:
                        energy_components["electronic_energy_eh"] = parse_orca_float(line.split(":")[1].split()[0])
                    elif "One Electron Energy" in line:
                        energy_components["one_electron_energy_eh"] = parse_orca_float(line.split(":")[1].split()[0])
                    elif "Two Electron Energy" in line:
                        energy_components["two_electron_energy_eh"] = parse_orca_float(line.split(":")[1].split()[0])
                    elif "Virial Ratio" in line:
                        energy_components["virial_ratio"] = parse_orca_float(line.split(":")[1].strip())
                    elif "E(XC)" in line:
                        energy_components["xc_energy_eh"] = parse_orca_float(line.split(":")[1].split()[0])
                    elif "NL Energy" in line or "E(C,NL)" in line:
                        energy_components["nl_energy_eh"] = parse_orca_float(line.split(":")[1].split()[0])

                # ── SCF CONVERGENCE ───────────────────────────────
                elif state == OrcaParseState.SCF_CONVERGENCE:
                    if line.strip() == "" and section_line_count > 2:
                        result["scf_convergence"] = scf_convergence
                        state = OrcaParseState.IDLE
                        continue
                    if "Energy change" in line:
                        scf_convergence["energy_change"] = parse_orca_float(line.split("...")[1].split()[0])
                    elif "MAX-Density change" in line:
                        scf_convergence["max_density_change"] = parse_orca_float(line.split("...")[1].split()[0])
                    elif "RMS-Density change" in line:
                        scf_convergence["rms_density_change"] = parse_orca_float(line.split("...")[1].split()[0])
                    elif "DIIS Error" in line:
                        scf_convergence["diis_error"] = parse_orca_float(line.split("...")[1].split()[0])

                # ── ORBITAL ENERGIES ──────────────────────────────
                elif state == OrcaParseState.ORBITAL_ENERGIES:
                    stripped = line.strip()
                    if (stripped == "" or stripped.startswith("----")) and section_line_count > 3:
                        _finalize_orbitals(result, last_occupied_energy, last_occupied_ev, first_virtual_energy, first_virtual_ev, n_electrons, n_orbitals)
                        state = OrcaParseState.IDLE
                        continue
                    parts = line.split()
                    # Orbital lines: "  NO   OCC   E(Eh)   E(eV)" with exactly 4 columns
                    # and OCC is 0.0000, 1.0000, or 2.0000
                    if len(parts) == 4 and parts[0] != "NO":
                        try:
                            occ = float(parts[1])
                            if occ not in (0.0, 1.0, 2.0):
                                # Not a valid orbital line; terminate
                                _finalize_orbitals(result, last_occupied_energy, last_occupied_ev, first_virtual_energy, first_virtual_ev, n_electrons, n_orbitals)
                                state = OrcaParseState.IDLE
                                continue
                            e_eh = float(parts[2])
                            e_ev = float(parts[3])
                            n_orbitals += 1
                            if occ > 0:
                                n_electrons += occ
                                last_occupied_energy = e_eh
                                last_occupied_ev = e_ev
                            elif first_virtual_energy is None:
                                first_virtual_energy = e_eh
                                first_virtual_ev = e_ev
                        except (ValueError, IndexError):
                            pass

                # ── MULLIKEN ATOMIC CHARGES ────────────────────────
                elif state == OrcaParseState.MULLIKEN_CHARGES:
                    if "Sum of atomic charges" in line:
                        result["mulliken_charges"] = mulliken_charges
                        if mulliken_has_spin and mulliken_spins:
                            result["mulliken_spins"] = mulliken_spins
                        state = OrcaParseState.IDLE
                        continue
                    if line.strip() == "" or line.strip().startswith("---"):
                        if line.strip() == "" and mulliken_charges:
                            result["mulliken_charges"] = mulliken_charges
                            if mulliken_has_spin and mulliken_spins:
                                result["mulliken_spins"] = mulliken_spins
                            state = OrcaParseState.IDLE
                        continue
                    idx, elem, vals = _parse_charge_line(line)
                    if idx is not None and vals:
                        mulliken_charges[_atom_key(idx, elem)] = vals[0]
                        if mulliken_has_spin and len(vals) >= 2:
                            mulliken_spins[_atom_key(idx, elem)] = vals[1]

                # ── LOEWDIN ATOMIC CHARGES ─────────────────────────
                elif state == OrcaParseState.LOEWDIN_CHARGES:
                    if line.strip() == "" or line.strip().startswith("---"):
                        if line.strip() == "" and loewdin_charges:
                            result["loewdin_charges"] = loewdin_charges
                            if loewdin_has_spin and loewdin_spins:
                                result["loewdin_spins"] = loewdin_spins
                            state = OrcaParseState.IDLE
                        continue
                    idx, elem, vals = _parse_charge_line(line)
                    if idx is not None and vals:
                        loewdin_charges[_atom_key(idx, elem)] = vals[0]
                        if loewdin_has_spin and len(vals) >= 2:
                            loewdin_spins[_atom_key(idx, elem)] = vals[1]

                # ── LOEWDIN BOND ORDERS ────────────────────────────
                elif state == OrcaParseState.LOEWDIN_BOND_ORDERS:
                    if line.strip() == "" and loewdin_bonds:
                        result["loewdin_bond_orders"] = loewdin_bonds
                        state = OrcaParseState.IDLE
                        continue
                    if "B(" in line:
                        for key, val in _parse_bond_pairs(line):
                            loewdin_bonds[key] = val

                # ── MAYER POPULATION ANALYSIS ──────────────────────
                elif state == OrcaParseState.MAYER_POPULATION:
                    if "Mayer bond orders" in line:
                        # Save population data and transition to bond orders
                        if mayer_population:
                            result["mayer_population"] = mayer_population
                        if mayer_charges:
                            result["mayer_charges"] = mayer_charges
                        state = OrcaParseState.MAYER_BOND_ORDERS
                        mayer_bonds = {}
                        continue
                    if line.strip() == "" and mayer_population:
                        result["mayer_population"] = mayer_population
                        if mayer_charges:
                            result["mayer_charges"] = mayer_charges
                        state = OrcaParseState.IDLE
                        continue
                    # Skip header lines until we see the atom table
                    parts = line.split()
                    if not mayer_header_skipped:
                        if "ATOM" in line and "NA" in line:
                            mayer_header_skipped = True
                        continue
                    # Parse atom rows: "  0 N     7.5606     7.0000    -0.5606     1.8457     1.8457    -0.0000"
                    # Columns:          idx elem   NA         ZA         QA         VA         BVA        FA
                    if len(parts) >= 7:
                        try:
                            idx = int(parts[0])
                            elem = parts[1]
                            qa = parse_orca_float(parts[4])
                            va = parse_orca_float(parts[5])
                            bva = parse_orca_float(parts[6])
                            key = _atom_key(idx, elem)
                            if va is not None and bva is not None:
                                mayer_population[key] = {
                                    "va": va, "bva": bva
                                }
                            if qa is not None:
                                mayer_charges[key] = qa
                        except (ValueError, IndexError):
                            pass

                # ── MAYER BOND ORDERS ──────────────────────────────
                elif state == OrcaParseState.MAYER_BOND_ORDERS:
                    if line.strip() == "" and mayer_bonds:
                        result["mayer_bond_orders"] = mayer_bonds
                        state = OrcaParseState.IDLE
                        continue
                    if "B(" in line:
                        for key, val in _parse_bond_pairs(line):
                            mayer_bonds[key] = val

                # ── HIRSHFELD ANALYSIS ─────────────────────────────
                elif state == OrcaParseState.HIRSHFELD:
                    if line.strip() == "" and hirshfeld_charges:
                        result["hirshfeld_charges"] = hirshfeld_charges
                        result["hirshfeld_spins"] = hirshfeld_spins
                        state = OrcaParseState.IDLE
                        continue
                    # Skip header until we see "ATOM     CHARGE      SPIN"
                    if not hirshfeld_header_done:
                        if "ATOM" in line and "CHARGE" in line and "SPIN" in line:
                            hirshfeld_header_done = True
                        continue
                    # Parse: "  0 Ac   0.187646    0.000000"
                    parts = line.split()
                    if len(parts) >= 4:
                        try:
                            idx = int(parts[0])
                            elem = parts[1]
                            charge = parse_orca_float(parts[2])
                            spin = parse_orca_float(parts[3])
                            key = _atom_key(idx, elem)
                            if charge is not None:
                                hirshfeld_charges[key] = charge
                            if spin is not None:
                                hirshfeld_spins[key] = spin
                        except (ValueError, IndexError):
                            pass

                # ── MBIS ANALYSIS ──────────────────────────────────
                elif state == OrcaParseState.MBIS:
                    if line.strip() == "" and mbis_charges:
                        result["mbis_charges"] = mbis_charges
                        result["mbis_populations"] = mbis_populations
                        result["mbis_spins"] = mbis_spins
                        state = OrcaParseState.IDLE
                        continue
                    if "MBIS VALENCE-SHELL" in line:
                        result["mbis_charges"] = mbis_charges
                        result["mbis_populations"] = mbis_populations
                        result["mbis_spins"] = mbis_spins
                        state = OrcaParseState.MBIS_VALENCE
                        mbis_valence_pop = {}
                        mbis_valence_width = {}
                        section_line_count = 0
                        continue
                    if not mbis_header_done:
                        if "ATOM" in line and "CHARGE" in line:
                            mbis_header_done = True
                        continue
                    # Parse: "  0 Ac   2.439318   26.560682    0.000000"
                    parts = line.split()
                    if len(parts) >= 5:
                        try:
                            idx = int(parts[0])
                            elem = parts[1]
                            charge = parse_orca_float(parts[2])
                            pop = parse_orca_float(parts[3])
                            spin = parse_orca_float(parts[4])
                            key = _atom_key(idx, elem)
                            if charge is not None:
                                mbis_charges[key] = charge
                            if pop is not None:
                                mbis_populations[key] = pop
                            if spin is not None:
                                mbis_spins[key] = spin
                        except (ValueError, IndexError):
                            pass

                # ── MBIS VALENCE SHELL ─────────────────────────────
                elif state == OrcaParseState.MBIS_VALENCE:
                    if line.strip() == "" and mbis_valence_pop:
                        result["mbis_valence_populations"] = mbis_valence_pop
                        result["mbis_valence_widths"] = mbis_valence_width
                        state = OrcaParseState.IDLE
                        continue
                    # Skip header: "ATOM   POPULATION   WIDTH(A.U.)"
                    if "ATOM" in line and "POPULATION" in line:
                        continue
                    parts = line.split()
                    if len(parts) >= 4:
                        try:
                            idx = int(parts[0])
                            elem = parts[1]
                            pop = parse_orca_float(parts[2])
                            width = parse_orca_float(parts[3])
                            key = _atom_key(idx, elem)
                            if pop is not None:
                                mbis_valence_pop[key] = pop
                            if width is not None:
                                mbis_valence_width[key] = width
                        except (ValueError, IndexError):
                            pass

                # ── CARTESIAN GRADIENT ─────────────────────────────
                elif state == OrcaParseState.GRADIENT:
                    if line.strip() == "" and gradient:
                        result["gradient"] = gradient
                        state = OrcaParseState.IDLE
                        continue
                    # Format: "   1   O   :    0.000385692    0.000381587    0.000766511"
                    parts = line.split()
                    if len(parts) >= 6 and ":" in line:
                        try:
                            idx = int(parts[0]) - 1  # 1-indexed in gradient output
                            elem = parts[1]
                            gx = parse_orca_float(parts[3])
                            gy = parse_orca_float(parts[4])
                            gz = parse_orca_float(parts[5])
                            if gx is not None and gy is not None and gz is not None:
                                gradient[_atom_key(idx, elem)] = [gx, gy, gz]
                        except (ValueError, IndexError):
                            pass

                # ── DIPOLE MOMENT ──────────────────────────────────
                elif state == OrcaParseState.DIPOLE:
                    # Collect lines until we see the next section delimiter
                    if line.startswith("---") and section_line_count > 3:
                        _parse_dipole_block(dipole_lines, result)
                        state = OrcaParseState.IDLE
                        continue
                    # Check for next section trigger (e.g., QUADRUPOLE MOMENT)
                    if "QUADRUPOLE MOMENT" in line.strip():
                        _parse_dipole_block(dipole_lines, result)
                        state = OrcaParseState.QUADRUPOLE
                        quadrupole_lines = []
                        quadrupole_has_tot = False
                        section_line_count = 0
                        continue
                    if "Rotational spectrum" in line:
                        _parse_dipole_block(dipole_lines, result)
                        state = OrcaParseState.ROTATIONAL_CONSTANTS
                        rotational_lines = []
                        section_line_count = 0
                        continue
                    dipole_lines.append(line)

                # ── QUADRUPOLE MOMENT ──────────────────────────────
                elif state == OrcaParseState.QUADRUPOLE:
                    if "TOT" in line:
                        quadrupole_has_tot = True
                    if line.strip() == "" and quadrupole_has_tot:
                        _parse_quadrupole_block(quadrupole_lines, result)
                        state = OrcaParseState.IDLE
                        continue
                    if "Rotational spectrum" in line:
                        _parse_quadrupole_block(quadrupole_lines, result)
                        state = OrcaParseState.ROTATIONAL_CONSTANTS
                        rotational_lines = []
                        section_line_count = 0
                        continue
                    if line.startswith("---") and quadrupole_has_tot:
                        _parse_quadrupole_block(quadrupole_lines, result)
                        state = OrcaParseState.IDLE
                        continue
                    quadrupole_lines.append(line)

                # ── ROTATIONAL CONSTANTS ───────────────────────────
                elif state == OrcaParseState.ROTATIONAL_CONSTANTS:
                    if line.strip() == "" and section_line_count > 2:
                        _parse_rotational_block(rotational_lines, result)
                        state = OrcaParseState.IDLE
                        continue
                    rotational_lines.append(line)

    except (OSError, IOError) as e:
        logger.warning(f"Error reading {orca_out_path}: {e}")

    # Handle case where file ends without blank line terminators
    if state == OrcaParseState.DIPOLE and dipole_lines:
        _parse_dipole_block(dipole_lines, result)
    elif state == OrcaParseState.QUADRUPOLE and quadrupole_lines:
        _parse_quadrupole_block(quadrupole_lines, result)
    elif state == OrcaParseState.ROTATIONAL_CONSTANTS and rotational_lines:
        _parse_rotational_block(rotational_lines, result)
    elif state == OrcaParseState.MULLIKEN_CHARGES and mulliken_charges:
        result["mulliken_charges"] = mulliken_charges
        if mulliken_has_spin and mulliken_spins:
            result["mulliken_spins"] = mulliken_spins
    elif state == OrcaParseState.LOEWDIN_CHARGES and loewdin_charges:
        result["loewdin_charges"] = loewdin_charges
        if loewdin_has_spin and loewdin_spins:
            result["loewdin_spins"] = loewdin_spins
    elif state == OrcaParseState.LOEWDIN_BOND_ORDERS and loewdin_bonds:
        result["loewdin_bond_orders"] = loewdin_bonds
    elif state == OrcaParseState.MAYER_BOND_ORDERS and mayer_bonds:
        result["mayer_bond_orders"] = mayer_bonds
    elif state == OrcaParseState.MAYER_POPULATION and mayer_population:
        result["mayer_population"] = mayer_population
        if mayer_charges:
            result["mayer_charges"] = mayer_charges
    elif state == OrcaParseState.HIRSHFELD and hirshfeld_charges:
        result["hirshfeld_charges"] = hirshfeld_charges
        result["hirshfeld_spins"] = hirshfeld_spins
    elif state == OrcaParseState.MBIS and mbis_charges:
        result["mbis_charges"] = mbis_charges
        result["mbis_populations"] = mbis_populations
        result["mbis_spins"] = mbis_spins
    elif state == OrcaParseState.MBIS_VALENCE and mbis_valence_pop:
        result["mbis_valence_populations"] = mbis_valence_pop
        result["mbis_valence_widths"] = mbis_valence_width
    elif state == OrcaParseState.GRADIENT and gradient:
        result["gradient"] = gradient
    elif state == OrcaParseState.SCF_ENERGY_BLOCK and energy_components:
        result["energy_components"] = energy_components
    elif state == OrcaParseState.SCF_CONVERGENCE and scf_convergence:
        result["scf_convergence"] = scf_convergence
    elif state == OrcaParseState.ORBITAL_ENERGIES:
        _finalize_orbitals(result, last_occupied_energy, last_occupied_ev, first_virtual_energy, first_virtual_ev, n_electrons, n_orbitals)

    return result


# ── Block parsers for accumulated lines ────────────────────────────────

def _parse_dipole_block(lines: list, result: dict) -> None:
    """Parse collected dipole section lines."""
    for line in lines:
        if "Total Dipole Moment" in line and ":" in line:
            parts = line.split(":")[1].split()
            if len(parts) >= 3:
                x = parse_orca_float(parts[0])
                y = parse_orca_float(parts[1])
                z = parse_orca_float(parts[2])
                if x is not None and y is not None and z is not None:
                    result["dipole_au"] = [x, y, z]
        elif "Magnitude (a.u.)" in line:
            parts = line.split(":")
            if len(parts) >= 2:
                val = parse_orca_float(parts[1].strip())
                if val is not None:
                    result["dipole_magnitude_au"] = val


def _parse_quadrupole_block(lines: list, result: dict) -> None:
    """Parse collected quadrupole section lines."""
    for line in lines:
        if line.strip().startswith("TOT"):
            parts = line.split()
            # Format: "TOT     -275.047  -255.296  -290.176   1.619   0.318  -0.681 (a.u.)"
            vals = []
            for p in parts[1:]:
                if p.startswith("("):
                    break
                v = parse_orca_float(p)
                if v is not None:
                    vals.append(v)
            if len(vals) >= 6:
                result["quadrupole_au"] = vals[:6]  # xx, yy, zz, xy, xz, yz


def _parse_rotational_block(lines: list, result: dict) -> None:
    """Parse collected rotational constants lines."""
    for line in lines:
        if "Rotational constants in cm-1" in line:
            parts = line.split(":")
            if len(parts) >= 2:
                vals = []
                for p in parts[1].split():
                    v = parse_orca_float(p)
                    if v is not None:
                        vals.append(v)
                if len(vals) >= 3:
                    result["rotational_constants_cm1"] = vals[:3]


# ── Validation ────────────────────────────────────────────────────────


def validate_parse_completeness(orca_dict: dict) -> bool:
    """Check that the parse produced enough data to justify merging/deletion.

    A truncated orca.out may parse successfully but produce a nearly-empty dict.
    Require at least final energy + one charge population.
    """
    has_energy = "final_energy_eh" in orca_dict
    has_any_charges = any(
        k in orca_dict
        for k in (
            "mulliken_charges",
            "loewdin_charges",
            "hirshfeld_charges",
            "mbis_charges",
        )
    )
    return has_energy and has_any_charges


# ── Shared utilities ──────────────────────────────────────────────────


def find_orca_output_file(folder: str) -> Optional[str]:
    """Find orca.out or output.out in a folder.

    Returns the full path if found, None otherwise.
    Used by both the pipeline (_run_orca_parse in omol.py) and the
    retroactive CLI (parse_orca_out.py) to avoid duplicating this logic.
    """
    for name in ("orca.out", "output.out"):
        path = os.path.join(folder, name)
        if os.path.isfile(path):
            return path
    return None


# ── JSON I/O with atomic writes ───────────────────────────────────────


def write_orca_json(folder: str, orca_dict: dict) -> None:
    """Write orca.json to the specified folder with compact atomic write."""
    path = os.path.join(folder, "orca.json")
    atomic_json_write(path, orca_dict, indent=None)


def merge_orca_into_charge_json(orca_dict: dict, charge_json_path: str):
    """
    Merge ORCA charge data into existing charge.json.
    Adds _orca suffixed method keys. Idempotent.
    Skips if charge.json doesn't exist.
    """
    if not os.path.isfile(charge_json_path):
        return

    try:
        with open(charge_json_path, "r") as f:
            charge_data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return

    modified = False

    # Mulliken charges from ORCA
    if "mulliken_charges" in orca_dict:
        charge_data["mulliken_orca"] = {"charge": orca_dict["mulliken_charges"]}
        modified = True

    # Loewdin charges from ORCA
    if "loewdin_charges" in orca_dict:
        charge_data["loewdin_orca"] = {"charge": orca_dict["loewdin_charges"]}
        modified = True

    # Hirshfeld charges from ORCA
    if "hirshfeld_charges" in orca_dict:
        entry = {"charge": orca_dict["hirshfeld_charges"]}
        if "hirshfeld_spins" in orca_dict:
            entry["spin"] = orca_dict["hirshfeld_spins"]
        charge_data["hirshfeld_orca"] = entry
        modified = True

    # Mayer gross atomic charges from ORCA (QA column from Mayer population)
    # Unifies with charge.json alongside Multiwfn-derived charge data
    if "mayer_charges" in orca_dict:
        charge_data["mayer_orca"] = {"charge": orca_dict["mayer_charges"]}
        modified = True

    # MBIS charges from ORCA
    if "mbis_charges" in orca_dict:
        entry = {"charge": orca_dict["mbis_charges"]}
        if "mbis_spins" in orca_dict:
            entry["spin"] = orca_dict["mbis_spins"]
        if "mbis_populations" in orca_dict:
            entry["population"] = orca_dict["mbis_populations"]
        charge_data["mbis_orca"] = entry
        modified = True

    if modified:
        atomic_json_write(charge_json_path, charge_data)


def merge_orca_into_bond_json(orca_dict: dict, bond_json_path: str):
    """
    Merge ORCA bond order data into existing bond.json.
    Adds _orca suffixed method keys. Idempotent.
    Skips if bond.json doesn't exist.
    """
    if not os.path.isfile(bond_json_path):
        return

    try:
        with open(bond_json_path, "r") as f:
            bond_data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return

    modified = False

    if "mayer_bond_orders" in orca_dict:
        bond_data["mayer_orca"] = orca_dict["mayer_bond_orders"]
        modified = True

    if "loewdin_bond_orders" in orca_dict:
        bond_data["loewdin_orca"] = orca_dict["loewdin_bond_orders"]
        modified = True

    if modified:
        atomic_json_write(bond_json_path, bond_data)
