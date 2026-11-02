"""ORCA wavefunctions without Multiwfn: read the .molden file orca_2mkl writes and
return the dict charge_engine.read_wfx returns for the orca.wfx Multiwfn would write
from it (Multiwfn 3.8 fileIO.f90 readmolden, ORCA path, and readinfile's EDF step).

What the ORCA path does, in order:
  - [Atoms]: element from the symbol, nuclear charge from the third column, AU or
    Angstrom coordinates; [Pseudo] (ORCA >= 6) overrides the charge per atom. ORCA 5
    moldens list the full Z, so their effective charges come from orca.out
    (zeff_override, see orca5_effective_charges).
  - [GTO]: primitives with a zero contraction coefficient are dropped; ORCA's
    coefficients are divided by its normalization (renormgau_ORCA) and every shell
    is renormalized (renormmoldengau). All shells are spherical.
  - [MO]: ORCA normalizes f(+3,-3), g(+3,-3,+4,-4) and h(+3,-3,+4,-4) to -1, so their
    coefficients change sign. Multiwfn does this for h on alpha orbitals only
    (fileIO.f90:4613-4615); here it is done for both spins, as ORCA's own orca_plot
    requires, unless mwfn_h_compat=True reproduces Multiwfn.
  - spherical -> Cartesian with gensphcartab(iprog=2) (data/multiwfn_sphcart.py);
    primitive coefficient = Cartesian MO coefficient x contraction x normgau.
  - unrestricted when the first occupation is below 1.05; otherwise "Alpha and Beta",
    with singly occupied orbitals of a restricted open shell as "Alpha".
  - EDF: when any atom's charge differs from Z, every such atom gets the core
    density for (Z, Z - charge) from data/multiwfn_edf.py, as Multiwfn's built-in
    library supplies it (isupplyEDF=2); a pair the library lacks gets none.

Primitive types follow the wfx convention (g functions reordered by outwfx's
convGseq), so prepare_basis and every engine take the dict unchanged. Only the
round trip through outwfx's E20.12 text differs from reading Multiwfn's orca.wfx.
"""

import logging
import math
import os

import numpy as np

from qtaim_gen.source.data import multiwfn_sphcart as SC
from qtaim_gen.source.data import multiwfn_tables as T
from qtaim_gen.source.data.multiwfn_edf import EDF
from qtaim_gen.source.utils.io import pull_ecp_dict

logger = logging.getLogger(__name__)

B2A = 0.529177210903  # Multiwfn's Bohr -> Angstrom (2018 CODATA)

SYMBOLS = (
    "Bq H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr "
    "Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb "
    "Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf "
    "Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og"
).split()
Z_OF_SYMBOL = {s: z for z, s in enumerate(SYMBOLS)}

SHELL_L = {"s": 0, "p": 1, "d": 2, "f": 3, "g": 4, "h": 5}
# renormgau_ORCA / renormmoldengau constants per l: (n1, n2, nf)
_RENORM = {0: (3, 3, 1), 1: (7, 5, 1), 2: (11, 7, 9), 3: (15, 9, 225), 4: (19, 11, 11025), 5: (23, 13, 893025)}
# ORCA's -1 normalized components, 0-based within the shell (0, +1, -1, +2, -2, ...)
_ORCA_FLIP = {3: slice(5, 7), 4: slice(5, 9), 5: slice(5, 9)}
_SPH2CART = {2: np.array(SC.D), 3: np.array(SC.F), 4: np.array(SC.G), 5: np.array(SC.H)}
# molden Cartesian position -> Multiwfn primitive type (fileIO.f90 s2f)
_MOLDEN_TO_MWFN = {
    0: (1,), 1: (2, 3, 4), 2: (5, 6, 7, 8, 9, 10), 3: (11, 12, 13, 17, 14, 15, 18, 19, 16, 20),
    4: (35, 25, 21, 34, 33, 29, 24, 26, 22, 32, 30, 23, 31, 28, 27), 5: tuple(range(36, 57)),
}
_MWFN_TO_WFX = {m: w for w, m in T.WFX_G_TO_MWFN.items()}


def _normgau(t, e):
    """Multiwfn normgau: normalization of a Cartesian Gaussian of type t (1-based)."""
    ix, iy, iz = T.TYPE2IX[t - 1], T.TYPE2IY[t - 1], T.TYPE2IZ[t - 1]
    ft = math.factorial
    return (2 * e / math.pi) ** 0.75 * math.sqrt(
        (8 * e) ** (ix + iy + iz) * ft(ix) * ft(iy) * ft(iz) / (ft(2 * ix) * ft(2 * iy) * ft(2 * iz))
    )


def _renorm_shell(l, exps, coefs):
    """Divide by renormgau_ORCA, then renormmoldengau (sub.f90:1457, 1485)."""
    n1, n2, nf = _RENORM[l]
    fc = 2.0**n1 / (math.pi**3 * nf)
    con = [c / math.sqrt(math.sqrt(fc * e**n2)) for e, c in zip(exps, coefs)]
    ctmp = [c * math.sqrt(math.sqrt(fc * e**n2)) for e, c in zip(exps, con)]
    s = 0.0
    for i in range(len(exps)):
        for j in range(i + 1):
            add = ctmp[i] * ctmp[j] / math.sqrt(fc * ((exps[i] + exps[j]) / 2) ** n2)
            s += add if i == j else 2 * add
    if s > 1e-10:
        s = 1 / math.sqrt(s)
    return [c * s for c in con]


def orca5_effective_charges(orca_out):
    """{0-based atom index: effective nuclear charge} for ECP atoms, from the starred
    rows of orca.out's CARTESIAN COORDINATES (A.U.) block (ORCA 5 moldens carry no
    [Pseudo] block)."""
    return {int(k): float(v["number"]) for k, v in pull_ecp_dict(orca_out).items()}


def _section_name(line):
    return line.strip().split("]")[0][1:].strip().lower()


def read_orca_molden(path, zeff_override=None, mwfn_h_compat=False, keep_virtuals=False):
    """The read_wfx dict for an ORCA .molden file, streamed line by line.

    zeff_override: {0-based atom: effective charge}, for ORCA 5 ECP jobs.
    mwfn_h_compat: flip h(+3,-3,+4,-4) on alpha orbitals only, as Multiwfn does.
    keep_virtuals: keep unoccupied orbitals (Multiwfn's wfx has occupied only).
    """
    atnums, zeff, coords, shells, mos = [], [], [], [], []
    is_orca, section, unit_au, nsph = False, None, True, 0
    cur_atom = shell = mo = coef = None
    with open(path, errors="replace") as fh:
        for line in fh:
            s = line.strip()
            if s.startswith("["):
                section = _section_name(s)
                if section == "atoms":
                    unit_au = "angs" not in s.lower()
                elif section == "mo":
                    nsph = sum(2 * sh[1] + 1 for sh in shells)
                continue
            if section in ("molden format", "title") or section is None:
                is_orca = is_orca or "orca" in s.lower()
            elif section == "atoms" and s:
                p = s.split()
                sym = p[0][0].upper() + (p[0][1].lower() if len(p[0]) > 1 and p[0][1].isalpha() else "")
                if sym not in Z_OF_SYMBOL:
                    raise ValueError(f"{path}: unknown element {p[0]}")
                atnums.append(Z_OF_SYMBOL[sym])
                zeff.append(float(p[2]))
                coords.append([float(x) for x in p[3:6]])
            elif section == "pseudo" and s:
                p = s.split()
                zeff[int(p[1]) - 1] = float(p[2])
            elif section == "gto":
                p = s.split()
                if not p:
                    cur_atom = shell = None
                elif cur_atom is None:
                    cur_atom = int(p[0]) - 1
                elif p[0].lower() in SHELL_L or p[0].lower() == "i":
                    if p[0].lower() not in SHELL_L:
                        raise ValueError(f"{path}: {p[0]} shells are not supported")
                    shell = [cur_atom, SHELL_L[p[0].lower()], [], []]
                    shells.append(shell)
                else:
                    e, c = (float(x.replace("D", "E").replace("d", "e")) for x in p[:2])
                    if c != 0.0:
                        shell[2].append(e)
                        shell[3].append(c)
            elif section == "mo" and s:
                if "=" in s:
                    if mo is None or "_body" in mo:
                        mo = {}
                        mos.append(mo)
                    k, v = s.split("=", 1)
                    mo[k.strip().lower()] = v.strip()
                else:
                    # the header precedes the coefficients, so unoccupied orbitals are
                    # skipped without storing them (most of a large molden)
                    if "_body" not in mo:
                        mo["_body"] = True
                        coef = mo["coef"] = np.zeros(nsph) if keep_virtuals or float(mo["occup"]) != 0 else None
                    if coef is not None:
                        i, v = s.split()[:2]
                        coef[int(i) - 1] = float(v.replace("D", "E"))
    if not is_orca:
        raise ValueError(f"{path}: not an ORCA molden (no 'orca' before [GTO]); only ORCA conventions are implemented")
    if not atnums or not shells or not mos:
        raise ValueError(f"{path}: missing [Atoms], [GTO] or [MO]")
    atnums = np.array(atnums, dtype=np.int64)
    zeff = np.array(zeff)
    coords = np.array(coords)
    if not unit_au:
        coords = coords / B2A
    for k, v in (zeff_override or {}).items():
        zeff[k] = v

    occ = np.array([float(m["occup"]) for m in mos])
    energy = np.array([float(m["ene"]) for m in mos])
    spin = np.array(["beta" in m.get("spin", "alpha").lower() for m in mos])
    keep = np.array([m.get("coef") is not None for m in mos])
    C = np.array([m["coef"] for m in mos if m.get("coef") is not None])
    alpha_rows = np.flatnonzero(~spin[keep])
    off = 0
    for _, l, _, _ in shells:
        if l in _ORCA_FLIP:
            cols = slice(off + _ORCA_FLIP[l].start, off + _ORCA_FLIP[l].stop)
            C[alpha_rows if (l == 5 and mwfn_h_compat) else slice(None), cols] *= -1
        off += 2 * l + 1

    nprim = sum(len(ex) * len(_MOLDEN_TO_MWFN[l]) for _, l, ex, _ in shells)
    coeffs = np.empty((len(C), nprim))
    centers, types, exps = [], [], []
    off = 0
    for iat, l, ex, co in shells:
        block = C[:, off : off + 2 * l + 1]
        cart = block if l < 2 else block @ _SPH2CART[l].T
        con = _renorm_shell(l, ex, co)
        for j, t in enumerate(_MOLDEN_TO_MWFN[l]):
            for e, c in zip(ex, con):
                coeffs[:, len(exps)] = cart[:, j] * (c * _normgau(t, e))
                centers.append(iat)
                types.append(_MWFN_TO_WFX.get(t, t))
                exps.append(e)
        off += 2 * l + 1
    del C

    if occ[0] < 1.05:
        spin_types = ["Beta" if b else "Alpha" for b in spin]
    elif all(o == round(o) for o in occ) and not any(o == 1.0 for o in occ):
        spin_types = ["Alpha and Beta"] * len(occ)
    else:
        spin_types = ["Alpha" if o == 1.0 else "Alpha and Beta" for o in occ]
    n_a = sum(o if t == "Alpha" else (o / 2 if t == "Alpha and Beta" else 0.0) for o, t in zip(occ, spin_types))
    n_b = sum(o if t == "Beta" else (o / 2 if t == "Alpha and Beta" else 0.0) for o, t in zip(occ, spin_types))

    ec, ee, eco = [], [], []
    if np.any((atnums != np.rint(zeff)) & (atnums != 0)):
        for a, (z, q) in enumerate(zip(atnums, zeff)):
            ncore = int(z - round(q))
            if z == 0 or ncore == 0:
                continue
            if (int(z), ncore) not in EDF:
                logger.warning("%s: no EDF core density for Z=%d with %d core electrons", path, z, ncore)
                continue
            al, co = EDF[(int(z), ncore)]
            ec += [a] * len(al)
            ee += list(al)
            eco += list(co)
    return {
        "atnums": atnums, "zeff": zeff, "coords": coords,
        "prim_center": np.array(centers, dtype=np.int64), "prim_type": np.array(types, dtype=np.int64),
        "prim_exp": np.array(exps), "occ": occ[keep], "spin_types": [t for t, k in zip(spin_types, keep) if k],
        "coeffs": coeffs, "mult": int(round(n_a - n_b)) + 1, "mo_energy": energy[keep],
        "edf_center": np.array(ec, dtype=np.int64), "edf_exp": np.array(ee), "edf_coef": np.array(eco),
    }


def _rows(values, per_line, fmt):
    v = [fmt(x) for x in values]
    return ["".join(v[i : i + per_line]) for i in range(0, len(v), per_line)]


def write_wfx(wfx, path, title="Generated by qtaim_generator (core/wavefunction.py)"):
    """Write a read_wfx dict as Multiwfn 3.8's outwfx does (fileIO.f90:7836): occupied
    MOs only, values in E20.12, energy 0 and virial ratio 2 (a molden carries neither),
    <Number of Core Electrons> = the electrons the EDF stands in for."""
    from qtaim_gen.source.core.cp_search import _fe

    def e20(x):
        return _fe(float(x), w=20, d=12)

    def i20(x):
        return f"{int(x):20d}"

    occ_mask = wfx["occ"] != 0
    occ = wfx["occ"][occ_mask]
    types = [t for t, k in zip(wfx["spin_types"], occ_mask) if k]
    n_a = sum(o if t == "Alpha" else (o / 2 if t == "Alpha and Beta" else 0.0) for o, t in zip(occ, types))
    n_b = sum(o if t == "Beta" else (o / 2 if t == "Alpha and Beta" else 0.0) for o, t in zip(occ, types))
    nelec = n_a + n_b
    atnums, zeff = wfx["atnums"], wfx["zeff"]
    edf_atoms = sorted(set(int(a) for a in wfx["edf_center"]))
    n_core = sum(int(atnums[a] - round(zeff[a])) for a in edf_atoms)

    tmp = f"{path}.tmp"

    def emit(*lines):
        fh.write("\n".join(lines) + "\n")

    def block(tag, lines):
        emit(f"<{tag}>", *lines, f"</{tag}>")

    with open(tmp, "w") as fh:
        emit("<Title>", f" {title}", "</Title>", "<Keywords>", " GTO", "</Keywords>")

        block("Number of Nuclei", [f"{len(atnums):6d}"])
        block("Number of Occupied Molecular Orbitals", [f"{len(occ):6d}"])
        block("Number of Perturbations", [f"{0:6d}"])
        block("Net Charge", [f"{round(float(zeff.sum()) - nelec):6d}"])
        block("Number of Electrons", [f"{round(nelec):6d}"])
        block("Number of Alpha Electrons", [f"{round(n_a):6d}"])
        block("Number of Beta Electrons", [f"{round(n_b):6d}"])
        block("Electronic Spin Multiplicity", [f"{round(n_a - n_b) + 1:6d}"])
        block("Number of Core Electrons", [f"{n_core:6d}"])
        block("Nuclear Names", [f" {SYMBOLS[z]}{i + 1}" for i, z in enumerate(atnums)])
        block("Atomic Numbers", [f"{int(z):3d}" for z in atnums])
        block("Nuclear Charges", [e20(q) for q in zeff])
        block("Nuclear Cartesian Coordinates", ["".join(e20(x) for x in xyz) for xyz in wfx["coords"]])
        block("Number of Primitives", [f"{len(wfx['prim_exp']):6d}"])
        block("Primitive Centers", _rows(wfx["prim_center"] + 1, 5, i20))
        block("Primitive Types", _rows(wfx["prim_type"], 5, i20))
        block("Primitive Exponents", _rows(wfx["prim_exp"], 5, e20))
        if len(wfx["edf_exp"]):
            emit("<Additional Electron Density Function (EDF)>")
            block("Number of EDF Primitives", [f"{len(wfx['edf_exp']):6d}"])
            block("EDF Primitive Centers", _rows(wfx["edf_center"] + 1, 5, i20))
            # Multiwfn never sets b_EDF%type and writes 0; the EDF is s functions, and the
            # Multiwfn routines that read this file back expect what Multiwfn writes
            block("EDF Primitive Types", _rows(np.zeros(len(wfx["edf_exp"]), dtype=int), 5, i20))
            block("EDF Primitive Exponents", _rows(wfx["edf_exp"], 5, e20))
            block("EDF Primitive Coefficients", _rows(wfx["edf_coef"], 5, e20))
            emit("</Additional Electron Density Function (EDF)>")
        block("Molecular Orbital Occupation Numbers", [e20(o) for o in occ])
        block("Molecular Orbital Energies", [e20(e) for e in wfx["mo_energy"][occ_mask]])
        block("Molecular Orbital Spin Types", [f" {t}" for t in types])
        emit("<Molecular Orbital Primitive Coefficients>")
        for i, c in enumerate(wfx["coeffs"][occ_mask]):
            emit("<MO Number>", f"{i + 1:6d}", "</MO Number>", *_rows(c, 4, e20))
        emit("</Molecular Orbital Primitive Coefficients>")
        block("Energy = T + Vne + Vee + Vnn", [e20(0.0)])
        block("Virial Ratio (-V/T)", [e20(2.0)])
    os.replace(tmp, path)
