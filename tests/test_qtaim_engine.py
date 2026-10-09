"""QTAIM engine (core/qtaim_engine.py), phase P1: at Multiwfn's own CP positions
the per-point properties match Multiwfn 3.8's CPprop.txt to print precision,
and the full CPprop loader keeps every CP type."""

import importlib.util
import os

import numpy as np
import pytest

from qtaim_gen.source.core.parse_qtaim import load_cpprop_full, poincare_hopf

FIXTURES = os.path.join(os.path.dirname(__file__), "test_files")
HAS_NUMBA = importlib.util.find_spec("numba") is not None
CPPROP = os.path.join(FIXTURES, "charge_engine", "rmechdb_652_step10_0_2", "multiwfn_cpprop.txt")


def test_loader_keeps_every_cp_type():
    cps = load_cpprop_full(os.path.join(FIXTURES, "CPprop_w_bond_paths.txt"))
    counts = {lab: sum(c["label"] == lab for c in cps) for lab in ("NCP", "BCP", "RCP", "CCP")}
    assert counts == {"NCP": 13, "BCP": 13, "RCP": 1, "CCP": 0}
    assert poincare_hopf(cps) == 1
    first = cps[0]
    assert first["nucleus"] == 8
    assert first["pos_bohr"] == pytest.approx([-3.310421416761, -7.514490165356, -0.242075252851])
    assert first["props"]["Density of all electrons"] == pytest.approx(301.9076788)
    assert first["props"]["Corr. hole for alpha, ref."] == pytest.approx(-4.07793488e-05)
    assert len(first["hessian"]) == 3 and len(first["eigenvectors"]) == 3
    assert all(c["connected"] is not None for c in cps if c["label"] == "BCP")


@pytest.mark.skipif(not HAS_NUMBA, reason="numba not installed")
def test_point_properties_match_multiwfn_at_its_cps():
    from qtaim_gen.source.core.charge_engine import prepare_basis, read_wfx
    from qtaim_gen.source.core.qtaim_engine import point_properties

    cps = load_cpprop_full(CPPROP)
    wfx = read_wfx(os.path.join(os.path.dirname(CPPROP), "orca.wfx"))
    eng = point_properties(np.array([c["pos_bohr"] for c in cps]), wfx, prepare_basis(wfx))
    # printed E18.10; positions printed to 12 decimals add ~1e-10 near nuclei
    rel = {
        "Density of all electrons": "rho", "Density of Alpha electrons": "rho_alpha",
        "Density of Beta electrons": "rho_beta", "Lagrangian kinetic energy G(r)": "G",
        "Hamiltonian kinetic energy K(r)": "K", "Potential energy density V(r)": "V",
        "Energy density E(r) or H(r)": "H", "Laplacian of electron density": "laplacian",
        "Electron localization function (ELF)": "elf", "Localized orbital locator (LOL)": "lol",
        "Average local ionization energy (ALIE)": "alie", "Determinant of Hessian": "det_hessian",
    }
    for i, c in enumerate(cps):
        for name, key in rel.items():
            assert eng[key][i] == pytest.approx(c["props"][name], rel=1e-9), (c["index"], name)
        assert eng["spin"][i] == pytest.approx(c["props"]["Spin density of electrons"], abs=1e-10)
        # printed f12.6
        assert eng["ellipticity"][i] == pytest.approx(c["props"]["Ellipticity of electron density"], abs=1e-6)
        assert eng["eta"][i] == pytest.approx(c["props"]["eta index"], abs=1e-6)
        assert eng["eigenvalues"][i] == pytest.approx(sorted(c["eigenvalues"]), rel=1e-9)
