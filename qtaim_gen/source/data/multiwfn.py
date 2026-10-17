"""
Helpers to generate the data strings for the multiwfn program
"""

from typing import Optional, List

# routines the charge-engine mode of gbw_analysis computes itself instead of
# sending them to Multiwfn: charges, fuzzy integrals and fuzzy_bond in one pass
# (core/charge_engine.py; vdd, mbis and mbis_fuzzy_* exist only at full_set >= 1)
# and the ALIE surface analysis (core/surface_engine.py)
ENGINE_LEVEL1_ROUTINES = frozenset({"vdd", "mbis", "mbis_fuzzy_density", "mbis_fuzzy_spin"})
SURFACE_ENGINE_ROUTINES = frozenset({"other_alie"})
ENGINE_ROUTINES = frozenset({
    "hirshfeld", "adch", "cm5", "becke",
    "becke_fuzzy_density", "hirsh_fuzzy_density",
    "becke_fuzzy_spin", "hirsh_fuzzy_spin",
    "fuzzy_bond",
}) | ENGINE_LEVEL1_ROUTINES | SURFACE_ENGINE_ROUTINES


def charge_data() -> str:
    """
    charge analysis data input - single jobs
    """
    # string_ret = "7\n1\n1\nn\n2\n1\nn\n10\n0\nn\n11\n1\nn\n12\n1\nn\n0\n13\n1\nn\n0\n16\n1\nn\n18\n1\nn\n0\n19\nn\n20\n1\nn\n0\nq\n"
    string_ret = (
        "7\n1\n1\nn\n2\n1\nn\n10\n0\nn\n11\n1\nn\n13\n1\nn\n0\n20\n1\nn\n0\nq\n"
    )
    return string_ret


# CHELPG point-exclusion radii (Angstrom). Multiwfn defines them only for H-Ar
# (Breneman-Wiberg, population.f90 setESPfitvdwr) and gives heavier elements
# vdwr_UFF / 1.2 under ispecial=1 (U 1.415 A, below the 1.70 of O). Actinides get
# Batsanov's van der Waals radii (Inorg. Mater. 2001, 37, 871: Th 2.4, U 2.3) and
# CHELPG_ACTINIDE_DEFAULT where Batsanov gives none (ORCA's default CHELPG set uses
# about 2.34 for every actinide). A molecule with an actinide is fitted with these
# radii from a file (menu 10 -> 3); every other element keeps the radius Multiwfn
# would use, so molecules without actinides keep the stock input and results.
CHELPG_BRENEMAN_WIBERG = {1: 1.45, 2: 1.45, 3: 1.5, 4: 1.5, 5: 1.5, 6: 1.5, 7: 1.7, 8: 1.7, 9: 1.7, 10: 1.7,
                          **{z: 2.0 for z in range(11, 19)}}
CHELPG_ACTINIDE_RADII = {90: 2.4, 92: 2.3}
CHELPG_ACTINIDE_DEFAULT = 2.34
CHELPG_ACTINIDES = range(89, 104)
CHELPG_RADII_FILE = "chelpg_radii.txt"
CHELPG_INPUT = "7\n12\n1\nn\n0\n0\nq\n"
CHELPG_INPUT_RADII_FILE = f"7\n12\n10\n3\n1\n{CHELPG_RADII_FILE}\nn\n0\n0\nq\n"


def chelpg_radius(z: int) -> float:
    """CHELPG radius (Angstrom) of element z, as written to CHELPG_RADII_FILE."""
    from qtaim_gen.source.data.multiwfn_tables import VDWR_UFF

    if z in CHELPG_BRENEMAN_WIBERG:
        return CHELPG_BRENEMAN_WIBERG[z]
    if z in CHELPG_ACTINIDES:
        return CHELPG_ACTINIDE_RADII.get(z, CHELPG_ACTINIDE_DEFAULT)
    return VDWR_UFF[z] / 1.2


def chelpg_radii_file(atnums) -> "str | None":
    """Contents of CHELPG_RADII_FILE for a molecule with these atomic numbers:
    "Symbol radius" per element, or None when it has no actinide (stock input)."""
    from rdkit import Chem

    zs = sorted({int(z) for z in atnums})
    if not any(z in CHELPG_ACTINIDES for z in zs):
        return None
    pt = Chem.GetPeriodicTable()
    return "".join(f"{pt.GetElementSymbol(z)} {chelpg_radius(z)!r}\n" for z in zs)


# routines a permanent skip removes (see TIMINGS_SKIPPED_KEY in utils/validation.py):
# the charge engine skips the Hirshfeld family without free-atom densities or with an
# ECP atom but no EDF, and the MBIS family per its own limits; Multiwfn refuses MBIS
# beyond MULTIWFN_MBIS_MAX_Z (it exits 64 for mbis and segfaults for the fuzzy steps)
HIRSHFELD_FAMILY = ("hirshfeld", "adch", "cm5", "vdd", "hirsh_fuzzy_density", "hirsh_fuzzy_spin")
MBIS_FAMILY = ("mbis", "mbis_fuzzy_density", "mbis_fuzzy_spin")
MULTIWFN_MBIS_MAX_Z = 86


def charge_data_dict(full_set: int = 0) -> dict:
    """
    charge analysis data input - dictionary form
    Takes full_set to decide which methods to include
    Returns dictionary of method names to input strings
    """
    # cut resp and chelpg and peoe
    string_dict = {}

    string_dict["hirshfeld"] = "7\n1\n1\nn\n0\nq\n"
    string_dict["adch"] = "7\n11\n1\nn\n0\nq\n"
    string_dict["cm5"] = "7\n16\n1\nn\n0\nq\n"  # might cut later
    string_dict["becke"] = "7\n10\n0\nn\n0\nq\n"

    if full_set > 0:
        string_dict["vdd"] = "7\n2\n1\nn\n0\nq\n"
        string_dict["mbis"] = "7\n20\n1\nn\n0\nq\n"
        string_dict["chelpg"] = CHELPG_INPUT  # might cut later

    if full_set > 1:
        string_dict["bader"] = "17\n1\n1\n2\n7\n1\n1\n7\n1\n5\n-10\nq\n"

    return string_dict


def bond_order_data() -> str:  # separate out into dictionary
    """
    bond order data input - single job
    """
    string_ret = "9\n7\nn\n8\nn\n10\n1\n1\n0\n0\nq\n"
    return string_ret


def bond_order_dict(full_set: int = 0) -> dict:
    """
    bond order data input - dictionary form
    Takes full_set to decide which methods to include
    Returns dictionary of method names to input strings
    """
    string_dict = {}
    string_dict["fuzzy_bond"] = "9\n7\nn\n0\nq\n"

    if full_set > 0:
        string_dict["ibsi_bond"] = "9\n10\n1\n1\n0\n0\nq\n"

    if full_set > 1:
        string_dict["laplacian_bond"] = "9\n8\nn\n0\nq\n"  # expensive

    return string_dict


def fuzzy_data(spin: bool = True, full_set: int = 0) -> dict:
    """
    fuzzy analysis data input - multiple jobs
    Takes:
        spin: whether to include spin fuzzy analysis
        full_set: level of calculation detail (0-baseline, 1-baseline+mbis/elf, 2-full)
    Returns
        dictionary of method names to input strings
    """
    # string_ret = "15\n1\n1\n1\n2\n1\n3\n1\n9\n4\nn\n0\nq\n"
    # string_ret = "15\n1\n1\nn1\n2\n1\n3\n1\n9\n4\nn\n0\nq\n"
    # Each string must run in its own Multiwfn process: the -1 partition choice
    # persists within a session, so a becke_* step chained after a hirsh_* step
    # silently uses Hirshfeld (docs/solutions/multiwfn-gotchas/).
    string_dict = {}
    string_dict["becke_fuzzy_density"] = "15\n1\n1\n0\nq\n"
    # Hirshfeld (-1 -> 3) returns straight to the fuzzy menu; only MBIS (-1 -> 5)
    # has a "1 Start calculation!" submenu needing the extra "1". With it, the
    # density string integrated rho twice (second block all zeros) and the spin
    # string integrated rho (function 1) instead of spin density (function 5).
    string_dict["hirsh_fuzzy_density"] = "15\n-1\n3\n1\n1\n0\nq\n"

    if spin:
        string_dict["hirsh_fuzzy_spin"] = "15\n-1\n3\n1\n5\n0\nq\n"
        string_dict["becke_fuzzy_spin"] = "15\n1\n5\n0\nq\n"
        if full_set > 0:
            string_dict["mbis_fuzzy_spin"] = "15\n-1\n5\n1\n1\n5\n0\nq\n"

    if full_set > 0:
        string_dict["elf_fuzzy"] = "15\n1\n9\n0\nq\n"
        string_dict["mbis_fuzzy_density"] = "15\n-1\n5\n1\n1\n1\n0\n0\nq\n"

    if full_set > 1:
        string_dict["laplacian_rho_fuzzy"] = "15\n1\n3\n0\nq\n"
        string_dict["grad_norm_rho_fuzzy"] = "15\n1\n2\n0\nq\n"

    # return string_ret
    return string_dict


def other_data() -> str:
    """
    other properties - ellip, esp, lol, eta, e_loc, lagrangian
    """
    string_ret = "26\n3\na\nn\n3\nh\nn\n8\n0\n0\n12\n0\n-1\n2\n2\n0\n-1\n-1\nq\n"
    return string_ret


def other_data_dict(full_set: int) -> dict:
    """
    other properties - ellip, esp, lol, eta, e_loc, lagrangian
    """
    string_dict = {}
    string_dict["other_geometry"] = "26\n3\na\nn\n3\nh\nn\n8\n0\n0\nq\n"
    string_dict["other_alie"] = "12\n2\n2\n0\n-1\n-1\nq\n"

    if full_set > 1:
        string_dict["other_esp"] = "12\n0\n-1\n-1\nq\n"
    return string_dict
 

def qtaim_data(exhaustive: bool = False) -> str:  # can work in one go
    """
    qtaim data - critical points search and analysis
    """
    if exhaustive:
        string_ret = "2\n2\n3\n4\n5\n6\n-1\n-9\n8\n7\n0\n-10\nq\n"
    else:  # skips spherical search around atoms
        string_ret = "2\n2\n3\n4\n5\n8\n7\n0\n-10\nq\n"
    return string_ret
