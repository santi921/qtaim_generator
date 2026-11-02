"""ORCA molden reader (core/wavefunction.py) against Multiwfn's orca.wfx and ORCA's orca_plot.

Fixtures (tests/test_files/wavefunction):
  iodide_droplet  ORCA 6.0.0 UKS (mult 5) with an iodine ECP and EDF: molden from
                  orca_2mkl and the orca.wfx Multiwfn 3.8 writes from it
  no_cc_pv5z      NO UKS PBE0/cc-pV5Z (ORCA 6.0.1, s-h shells), molden trimmed to the
                  occupied MOs plus three virtual MOs dominated by f, g and h functions,
                  Multiwfn's orca.wfx, and orca_plot cube values for those three MOs
  orca5_coordinates_au.out  the CARTESIAN COORDINATES (A.U.) block of an ORCA 5 job
                  with Y and Sb ECPs (starred effective charges)
"""

import importlib.util
import json
import os

import numpy as np
import pytest

FIXTURES = os.path.join(os.path.dirname(__file__), "test_files", "wavefunction")
HAS_NUMBA = importlib.util.find_spec("numba") is not None
EXACT_KEYS = ("atnums", "zeff", "prim_center", "prim_type", "edf_center")
# Multiwfn writes the wfx with E20.12, so 12 significant digits is all a match can show
CLOSE_KEYS = ("coords", "prim_exp", "occ", "mo_energy", "edf_exp", "edf_coef")


def _paths(name):
    d = os.path.join(FIXTURES, name)
    return os.path.join(d, "orca.molden.input"), os.path.join(d, "orca.wfx")


@pytest.mark.parametrize("name,compat", [("iodide_droplet", False), ("no_cc_pv5z", True)])
def test_reader_reproduces_multiwfn_wfx(name, compat):
    """Multiwfn-compatible output equals the wfx Multiwfn writes, up to its E20.12 text."""
    from qtaim_gen.source.core.charge_engine import read_wfx
    from qtaim_gen.source.core.wavefunction import read_orca_molden

    molden, wfx = _paths(name)
    a, b = read_wfx(wfx), read_orca_molden(molden, mwfn_h_compat=compat)
    for k in EXACT_KEYS:
        assert np.array_equal(a[k], b[k]), k
    assert a["spin_types"] == b["spin_types"] and a["mult"] == b["mult"]
    for k in CLOSE_KEYS:
        if len(a[k]):
            assert np.allclose(a[k], b[k], rtol=1e-11, atol=0), k
    scale = np.abs(a["coeffs"]).max(axis=1, keepdims=True)
    assert np.max(np.abs(a["coeffs"] - b["coeffs"]) / scale) < 1e-10


@pytest.mark.skipif(not HAS_NUMBA, reason="numba not installed")
@pytest.mark.parametrize("name,compat", [("iodide_droplet", False), ("no_cc_pv5z", True)])
def test_density_matches_multiwfn_wfx(name, compat):
    from qtaim_gen.source.core.charge_engine import density, prepare_basis, read_wfx
    from qtaim_gen.source.core.wavefunction import read_orca_molden

    molden, wfx = _paths(name)
    a, b = read_wfx(wfx), read_orca_molden(molden, mwfn_h_compat=compat)
    rng = np.random.default_rng(0)
    c = a["coords"]
    pts = np.concatenate([c[rng.integers(len(c), size=2000)] + rng.normal(scale=1.5, size=(2000, 3)),
                          c[rng.integers(len(c), size=500)] + rng.normal(scale=0.05, size=(500, 3))])
    ra, sa = density(pts, c, prepare_basis(a), a)
    rb, sb = density(pts, c, prepare_basis(b), b)
    keep = ra > 1e-8
    assert np.max(np.abs(ra - rb)[keep] / ra[keep]) < 1e-10
    assert np.max(np.abs(sa - sb)) < 1e-8 * np.max(ra)


@pytest.mark.skipif(not HAS_NUMBA, reason="numba not installed")
def test_orbital_signs_match_orca_plot():
    """f(+3,-3), g and h(+3,-3,+4,-4) flipped on both spins reproduce ORCA's own MO
    values; Multiwfn's alpha-only h flip does not for a beta h orbital."""
    from qtaim_gen.source.core.charge_engine import orbital_values, prepare_basis
    from qtaim_gen.source.core.wavefunction import read_orca_molden

    molden, _ = _paths("no_cc_pv5z")
    with open(os.path.join(FIXTURES, "no_cc_pv5z", "orca_plot_reference.json")) as fh:
        ref = json.load(fh)["mos"]

    def mo_values(w, r):
        i = int(np.argmin([abs(e - r["energy"]) + (0 if t == r["spin"] else 1e9)
                           for e, t in zip(w["mo_energy"], w["spin_types"])]))
        one = dict(w, occ=np.ones(1), spin_types=["Alpha"], coeffs=w["coeffs"][i : i + 1],
                   mo_energy=w["mo_energy"][i : i + 1])
        return orbital_values(np.array(r["points_bohr"]), w["coords"], prepare_basis(one))[:, 0]

    fixed = read_orca_molden(molden, keep_virtuals=True)
    compat = read_orca_molden(molden, keep_virtuals=True, mwfn_h_compat=True)
    for name, r in ref.items():
        # orca_plot prints 5 significant digits
        assert np.max(np.abs(mo_values(fixed, r) - np.array(r["orca_plot"]))) < 1e-5, name
    assert np.max(np.abs(mo_values(compat, ref["beta_109_h"]) - np.array(ref["beta_109_h"]["orca_plot"]))) > 0.1


def test_h_fix_changes_only_beta_h_coefficients():
    from qtaim_gen.source.core.wavefunction import read_orca_molden

    molden, _ = _paths("no_cc_pv5z")
    fixed = read_orca_molden(molden)
    compat = read_orca_molden(molden, mwfn_h_compat=True)
    beta = np.array([t == "Beta" for t in fixed["spin_types"]])
    h = fixed["prim_type"] >= 36
    diff = fixed["coeffs"] != compat["coeffs"]
    assert diff.any()
    assert not diff[~beta].any() and not diff[:, ~h].any()


def test_unoccupied_orbitals_are_dropped_unless_kept():
    from qtaim_gen.source.core.wavefunction import read_orca_molden

    molden, _ = _paths("no_cc_pv5z")
    occ_only = read_orca_molden(molden)
    everything = read_orca_molden(molden, keep_virtuals=True)
    assert (occ_only["occ"] != 0).all()
    assert len(everything["occ"]) == len(occ_only["occ"]) + 3
    assert occ_only["mult"] == everything["mult"] == 2


def test_orca5_effective_charges_replace_a_missing_pseudo_block(tmp_path):
    """ORCA 5 moldens list the full Z and no [Pseudo]; the charges from orca.out give
    the same wavefunction, EDF included, as the [Pseudo] block of ORCA 6."""
    from qtaim_gen.source.core.wavefunction import orca5_effective_charges, read_orca_molden

    assert orca5_effective_charges(os.path.join(FIXTURES, "orca5_coordinates_au.out")) == {0: 11.0, 72: 23.0}

    molden, _ = _paths("iodide_droplet")
    with open(molden) as fh:
        text = fh.read()
    start = text.index("[Pseudo]")
    stripped = tmp_path / "orca.molden.input"
    stripped.write_text(text[:start] + text[text.index("[GTO]", start):])
    with_pseudo = read_orca_molden(molden)
    iodine = int(np.flatnonzero(with_pseudo["atnums"] == 53)[0])
    no_pseudo = read_orca_molden(str(stripped))
    assert no_pseudo["zeff"][iodine] == 53 and not len(no_pseudo["edf_center"])
    override = read_orca_molden(str(stripped), zeff_override={iodine: 25.0})
    for k in ("zeff", "edf_center", "edf_exp", "edf_coef"):
        assert np.array_equal(override[k], with_pseudo[k]), k


@pytest.mark.parametrize("name,compat", [("iodide_droplet", False), ("no_cc_pv5z", True)])
def test_written_wfx_matches_multiwfn_file(tmp_path, name, compat):
    """Every section but the title is Multiwfn's text; MO coefficients may differ in
    the last printed digit (operation order in the spherical -> Cartesian products)."""
    import re

    from qtaim_gen.source.core.charge_engine import read_wfx
    from qtaim_gen.source.core.wavefunction import read_orca_molden, write_wfx

    molden, wfx = _paths(name)
    out = tmp_path / "orca.wfx"
    write_wfx(read_orca_molden(molden, mwfn_h_compat=compat), str(out))

    def sections(text):
        return dict(re.findall(r"<([^/>][^>]*)>\n(.*?)\n</\1>", text, re.S))

    with open(wfx) as fh:
        ref = sections(fh.read())
    mine = sections(out.read_text())
    assert set(ref) == set(mine)
    coeff = "Molecular Orbital Primitive Coefficients"
    for k in set(ref) - {"Title", coeff}:
        assert ref[k] == mine[k], k
    a, b = ref[coeff].split(), mine[coeff].split()
    assert len(a) == len(b) and sum(x == y for x, y in zip(a, b)) / len(a) > 0.999
    assert np.allclose(read_wfx(str(out))["coeffs"], read_wfx(wfx)["coeffs"], rtol=1e-10, atol=1e-300)


@pytest.mark.parametrize("edit,message", [
    (lambda t: t.replace("orca", "other code").replace("ORCA", "other code"), "not an ORCA molden"),
    (lambda t: t.replace("\nf ", "\ni ", 1), "shells are not supported"),
])
def test_unsupported_moldens_are_refused(tmp_path, edit, message):
    from qtaim_gen.source.core.wavefunction import read_orca_molden

    molden, _ = _paths("no_cc_pv5z")
    with open(molden) as fh:
        text = fh.read()
    bad = tmp_path / "orca.molden.input"
    bad.write_text(edit(text))
    with pytest.raises(ValueError, match=message):
        read_orca_molden(str(bad))
