"""Element class definitions shared across manifest building, held-out split
construction, and dataset-coverage figures."""

from __future__ import annotations

TRANSITION_METALS = (
    frozenset(range(21, 31))    # Sc-Zn
    | frozenset(range(39, 49))  # Y-Cd
    | frozenset(range(72, 81))  # Hf-Hg
)
LANTHANIDES = frozenset(range(57, 72))   # La-Lu
ACTINIDES = frozenset(range(89, 104))    # Ac-Lr


# Symbol -> atomic number, Z=1..103. Hardcoded to avoid a pymatgen import
# in the hot path of manifest building (pymatgen imports are slow).
_SYMBOLS = [
    "H", "He",
    "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar",
    "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr",
    "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "In", "Sn", "Sb", "Te", "I", "Xe",
    "Cs", "Ba",
    "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er",
    "Tm", "Yb", "Lu",
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn",
    "Fr", "Ra",
    "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm",
    "Md", "No", "Lr",
]
SYMBOL_TO_Z = {sym: i + 1 for i, sym in enumerate(_SYMBOLS)}


def symbol_to_z(symbol: str) -> int:
    return SYMBOL_TO_Z[symbol]
