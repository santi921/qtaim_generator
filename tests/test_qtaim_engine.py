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


def test_loader_reads_three_digit_exponents_and_both_eigenvector_labels(tmp_path):
    # Multiwfn prints 1.999917076e-167 as 0.1999917076-166 (no E); current
    # builds write "Eigenvectors (columns)", older ones "Eigenvectors(columns)"
    with open(CPPROP) as f:
        lines = f.read().splitlines()
    i = next(k for k, ln in enumerate(lines) if "Components of gradient in x/y/z are:" in ln)
    vals = lines[i + 1].split()
    lines[i + 1] = "  " + "  ".join(["0.1999917076-166", "-0.3132422160+150", vals[2]])
    path = tmp_path / "CPprop.txt"
    path.write_text("\n".join(lines) + "\n")
    cps = load_cpprop_full(str(path))
    assert cps[0]["gradient"][:2] == [pytest.approx(0.1999917076e-166), pytest.approx(-0.3132422160e150)]
    assert all(len(c["eigenvectors"]) == 3 for c in cps)
    assert len(cps) == len(load_cpprop_full(CPPROP))


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


XECL4_WFX = os.path.join(FIXTURES, "charge_engine", "XeCl4_step2_0_1", "orca.wfx")


@pytest.mark.skipif(not HAS_NUMBA, reason="numba not installed")
def test_edf_path_against_charge_engine_and_finite_differences():
    """Closed shell with an EDF core (Xe): rho matches charge_engine.density, an
    independent EDF implementation, and the gradient matches central differences."""
    from qtaim_gen.source.core.charge_engine import density, prepare_basis, read_wfx
    from qtaim_gen.source.core.qtaim_engine import point_properties, wavefunction_type

    wfx = read_wfx(XECL4_WFX)
    assert len(wfx["edf_center"]) and wavefunction_type(wfx) == 0
    basis = prepare_basis(wfx)
    rng = np.random.default_rng(0)
    xe = wfx["coords"][int(np.argmax(wfx["atnums"]))]
    pts = xe + rng.normal(scale=0.5, size=(40, 3))
    eng = point_properties(pts, wfx, basis)
    rho_ref, _ = density(pts, wfx["coords"], basis, wfx)
    assert eng["rho"] == pytest.approx(rho_ref, rel=1e-13)
    h = 1e-5
    fd = np.stack([(point_properties(pts + h * e, wfx, basis)["rho"] - point_properties(pts - h * e, wfx, basis)["rho"])
                   / (2 * h) for e in np.eye(3)], axis=1)
    assert np.abs(eng["gradient"] - fd).max() <= 1e-5 * np.abs(eng["gradient"]).max()
    # the restricted ELF/LOL branch: bounded
    assert np.all((eng["elf"] >= 0) & (eng["elf"] <= 1)) and np.all((eng["lol"] >= 0) & (eng["lol"] <= 1))


def test_ecp_without_edf_is_refused():
    """An ECP atom with no EDF core density gives a wrong rho near it; only an
    MO-only comparison (tier 1 against HORTON) may ask for that on purpose."""
    from qtaim_gen.source.core.charge_engine import prepare_basis, read_wfx
    from qtaim_gen.source.core.qtaim_engine import ecp_atoms_without_edf, point_properties

    wfx = read_wfx(XECL4_WFX)
    assert ecp_atoms_without_edf(wfx) == []
    stripped = dict(wfx, edf_center=np.zeros(0, dtype=np.int64), edf_exp=np.zeros(0), edf_coef=np.zeros(0))
    xe = int(np.argmax(wfx["atnums"]))
    assert ecp_atoms_without_edf(stripped) == [xe]
    pts = wfx["coords"][[xe]] + 0.1
    with pytest.raises(ValueError, match="ECP atoms without EDF"):
        point_properties(pts, stripped, prepare_basis(stripped))
    if HAS_NUMBA:
        assert point_properties(pts, stripped, prepare_basis(stripped), allow_missing_edf=True)["rho"][0] > 0


def test_sampler_element_classes():
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
    from qtaim_verify_sample import ELEMENTS, element_class

    cls = {s: element_class(ELEMENTS.index(s) + 1) for s in ("H", "Br", "Fe", "Zn", "Rb", "Cs", "I", "Ag", "Hf", "Hg", "La", "Lu", "U")}
    assert cls == {"H": 0, "Br": 0, "Fe": 1, "Zn": 1, "Rb": 2, "Cs": 2, "I": 2, "Ag": 3, "Hf": 3, "Hg": 3,
                   "La": 4, "Lu": 4, "U": 4}
