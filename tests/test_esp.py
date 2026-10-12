"""ESP engine (core/esp.py), QTAIM phase P2: Boys functions against mpmath, the
nuclear-attraction integrals against an independent quadrature, and the ESP at
Multiwfn's CP positions against its CPprop.txt."""

import importlib.util
import math
import os

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(importlib.util.find_spec("numba") is None, reason="needs numba")

FIXTURES = os.path.join(os.path.dirname(__file__), "test_files")
CPPROP = os.path.join(FIXTURES, "charge_engine", "rmechdb_652_step10_0_2", "multiwfn_cpprop.txt")
XS = np.concatenate([[0.0, 1e-10, 1e-4], np.linspace(0.013, 29.97, 41), [29.9999, 30.0, 30.0001, 47.3, 210.0, 1e4]])


def _boys_mp(m, x):
    mpmath = pytest.importorskip("mpmath")
    mpmath.mp.dps = 40
    return float(mpmath.hyp1f1(m + 0.5, m + 1.5, -x) / (2 * m + 1))


def test_boys_functions_against_mpmath():
    from qtaim_gen.source.core import esp

    out_ref, out_tab, out_mw = np.empty(13), np.empty(13), np.empty(13)
    worst = {"reference": 0.0, "table": 0.0, "multiwfn": 0.0}
    for x in XS:
        exact = [_boys_mp(m, x) for m in range(13)]
        for mmax in (0, 4, 10, 12):
            esp.boys_reference(mmax, x, out_ref)
            esp.boys(mmax, x, out_tab, esp.BOYS_TABLE, esp._INV_FACT)
            esp.boys_multiwfn(mmax, x, out_mw)
            for m in range(mmax + 1):
                for key, out in (("reference", out_ref), ("table", out_tab), ("multiwfn", out_mw)):
                    worst[key] = max(worst[key], abs(out[m] - exact[m]) / exact[m])
    assert worst["reference"] < 5e-15 and worst["table"] < 5e-15
    # Fmch: series cut at 1e-8, sqrt(pi)/2 truncated to 0.88622692 above x = 10
    assert 1e-9 < worst["multiwfn"] < 1e-7


def _two_primitive_system(lmn_a, lmn_b, A, B, alpha, beta):
    wfx = {"coords": np.array([A, B], dtype=float), "zeff": np.array([1.0, 1.0])}
    basis = {
        "ct": np.ones((2, 1)), "occ": np.array([1.0]), "lmn": np.array([lmn_a, lmn_b], dtype=np.int64),
        "g_center": np.array([0, 1]), "g_alpha": np.array([alpha, beta]),
        "g_start": np.array([0, 1]), "g_end": np.array([1, 2]),
    }
    return wfx, basis


def _attraction_quadrature(la, lb, A, B, alpha, beta, C):
    """<a|1/|r-C||b> = 2/sqrt(pi) int_0^inf dt prod_k int (x-A)^i (x-B)^j
    exp(-alpha (x-A)^2 - beta (x-B)^2 - t^2 (x-C)^2) dx, inner integrals by
    Gauss-Hermite (exact for these polynomial degrees), outer by scipy quad."""
    from scipy.integrate import quad

    nodes, weights = np.polynomial.hermite.hermgauss(30)

    def inner(t):
        val = 1.0
        for k in range(3):
            q = alpha + beta + t * t
            Q = (alpha * A[k] + beta * B[k] + t * t * C[k]) / q
            const = alpha * A[k] ** 2 + beta * B[k] ** 2 + t * t * C[k] ** 2 - q * Q * Q
            x = Q + nodes / math.sqrt(q)
            val *= math.exp(-const) / math.sqrt(q) * np.sum(weights * (x - A[k]) ** la[k] * (x - B[k]) ** lb[k])
        return val

    return 2.0 / math.sqrt(math.pi) * quad(inner, 0.0, np.inf, epsabs=0.0, epsrel=1e-12, limit=400)[0]


@pytest.mark.parametrize("lmn_a, lmn_b, alpha, beta", [
    ((0, 0, 0), (0, 0, 0), 1.3, 0.4),
    ((1, 0, 0), (0, 1, 1), 0.9, 2.1),
    ((2, 1, 0), (0, 0, 2), 0.35, 1.7),
    ((3, 0, 1), (1, 2, 0), 1.1, 0.6),
    ((2, 2, 0), (1, 1, 2), 0.8, 0.5),
    ((5, 0, 0), (0, 3, 2), 0.7, 1.2),
    ((1, 2, 2), (2, 1, 2), 1.5, 0.9),
])
def test_attraction_integrals_against_quadrature(lmn_a, lmn_b, alpha, beta):
    from qtaim_gen.source.core import esp

    A, B = np.array([0.1, -0.3, 0.2]), np.array([0.9, 0.4, -0.5])
    wfx, basis = _two_primitive_system(lmn_a, lmn_b, A, B, alpha, beta)
    pairs = esp.esp_pairs(wfx, basis, cutoff=1e9, tol=0.0)
    for C in (np.array([0.3, 0.1, -0.2]), np.array([2.5, -1.0, 0.7]), np.array([-6.0, 4.0, 9.0])):
        # one MO with both coefficients 1: esp_e = -(V_aa + 2 V_ab + V_bb)
        ref = -sum(f * _attraction_quadrature(la, lb, pa, pb, xa, xb, C) for f, la, lb, pa, pb, xa, xb in (
            (1, lmn_a, lmn_a, A, A, alpha, alpha), (2, lmn_a, lmn_b, A, B, alpha, beta), (1, lmn_b, lmn_b, B, B, beta, beta)))
        _, ele, _ = esp.electrostatic_potential(C[None], wfx, basis, pairs)
        assert ele[0] == pytest.approx(ref, rel=1e-11, abs=1e-14)


def test_s_gaussian_closed_form():
    from qtaim_gen.source.core import esp

    # rho = exp(-2 a r^2): V_e(r) = -(pi / 2a)^(3/2) erf(sqrt(2a) r) / r
    a = 0.8
    wfx, basis = _two_primitive_system((0, 0, 0), (0, 0, 0), [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], a, a)
    basis = dict(basis, ct=np.array([[1.0], [0.0]]))
    pts = np.array([[0.0, 0.0, 1e-9], [0.3, 0.0, 0.0], [1.0, 2.0, -0.5], [40.0, 0.0, 0.0]])
    _, ele, _ = esp.electrostatic_potential(pts, wfx, basis, esp.esp_pairs(wfx, basis, tol=0.0))
    r = np.linalg.norm(pts, axis=1)
    ref = -((math.pi / (2 * a)) ** 1.5) * np.array([math.erf(math.sqrt(2 * a) * x) for x in r]) / r
    np.testing.assert_allclose(ele, ref, rtol=2e-14)


def _fixture():
    from qtaim_gen.source.core.charge_engine import prepare_basis, read_wfx
    from qtaim_gen.source.core.parse_qtaim import load_cpprop_full

    cps = load_cpprop_full(CPPROP)
    wfx = read_wfx(os.path.join(os.path.dirname(CPPROP), "orca.wfx"))
    return cps, wfx, prepare_basis(wfx)


def test_esp_matches_multiwfn_at_its_cps():
    from qtaim_gen.source.core import esp

    cps, wfx, basis = _fixture()
    pts = np.array([c["pos_bohr"] for c in cps])
    ref_e = np.array([c["props"]["ESP from electrons"] for c in cps])
    _, ele_mw, _ = esp.electrostatic_potential(pts, wfx, basis, boys_mode="multiwfn")
    _, ele, _ = esp.electrostatic_potential(pts, wfx, basis)
    # printed E18.10; the 12-decimal positions add ~1e-10
    assert np.max(np.abs(ele_mw - ref_e) / np.abs(ref_e)) < 1e-9
    # the exact Boys function differs from Fmch by ~6e-9 (divergence, decision 9)
    assert 1e-9 < np.max(np.abs(ele - ref_e) / np.abs(ref_e)) < 2e-8
    # nuclear ESP away from nuclei; at NCPs (1e-8 to 1e-5 Bohr from a nucleus) it is position-rounding limited
    bcp = np.array([c["label"] != "NCP" for c in cps])
    nuc = esp.nuclear_esp(pts[bcp], wfx["coords"], wfx["zeff"])
    ref_n = np.array([c["props"]["ESP from nuclear charges"] for c in cps])[bcp]
    assert np.max(np.abs(nuc - ref_n) / ref_n) < 1e-9


def test_screening_and_blocking_do_not_change_the_result():
    from qtaim_gen.source.core import esp

    cps, wfx, basis = _fixture()
    pts = np.array([c["pos_bohr"] for c in cps])
    full = esp.esp_pairs(wfx, basis, cutoff=1e9, tol=0.0)
    screened = esp.esp_pairs(wfx, basis)
    assert len(screened[0]) < len(full[0])
    _, a, _ = esp.electrostatic_potential(pts, wfx, basis, full)
    _, b, _ = esp.electrostatic_potential(pts, wfx, basis, screened)
    _, c, _ = esp.electrostatic_potential(pts, wfx, basis, screened, block=1)
    assert np.max(np.abs(a - b)) < 1e-13
    assert np.array_equal(b, c)


def test_nuclear_esp_is_1000_at_a_nucleus():
    from qtaim_gen.source.core import esp

    coords = np.array([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]])
    v = esp.nuclear_esp(np.array([[0.0, 0.0, 0.0], [0.7, 0.0, 0.0]]), coords, np.array([1.0, 1.0]))
    assert v[0] == 1000.0 and v[1] == pytest.approx(2 / 0.7)
