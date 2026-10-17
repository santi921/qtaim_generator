"""QTAIM engine CP search (core/cp_search.py), phase P3, Multiwfn-compatible
mode: the same seeds, the same CP set and positions as Multiwfn 3.8's default
search (topology options 2-5), and output independent of the thread count."""

import importlib.util
import os

import numpy as np
import pytest

from qtaim_gen.source.core.parse_qtaim import load_cpprop_full

pytestmark = pytest.mark.skipif(importlib.util.find_spec("numba") is None, reason="needs numba")

FIXTURES = os.path.join(os.path.dirname(__file__), "test_files", "charge_engine")
# multiwfn_cpprop.txt: Multiwfn 3.8(dev) 2024-Oct-6, stdin 2 2 3 4 5 8 7 0 -10 q
JOBS = ["rmechdb_652_step10_0_2", "XeCl4_step2_0_1"]
LABEL_TYPE = {"NCP": 0, "BCP": 1, "RCP": 2, "CCP": 3}


def _load(job):
    from qtaim_gen.source.core.charge_engine import prepare_basis, read_wfx

    wfx = read_wfx(os.path.join(FIXTURES, job, "orca.wfx"))
    return wfx, prepare_basis(wfx), load_cpprop_full(os.path.join(FIXTURES, job, "multiwfn_cpprop.txt"))


def test_seed_counts_match_multiwfn():
    from qtaim_gen.source.core.cp_search import multiwfn_seeds

    wfx, _, _ = _load("XeCl4_step2_0_1")
    # "Number of starting points" printed by Multiwfn for stages 3, 4, 5
    assert [len(p) for _, p in multiwfn_seeds(wfx["coords"], wfx["atnums"])] == [5, 10, 10, 5]


@pytest.mark.parametrize("job", JOBS)
def test_cp_set_and_positions_match_multiwfn(job):
    from qtaim_gen.source.core.cp_search import search

    wfx, basis, ref = _load(job)
    res = search(wfx, basis)
    assert len(res["pos"]) == len(ref)
    for lab, t in LABEL_TYPE.items():
        rp = np.array([c["pos_bohr"] for c in ref if c["label"] == lab]).reshape(-1, 3)
        ep = res["pos"][res["type"] == t]
        assert len(rp) == len(ep), lab
        if len(rp):
            # positions printed to 12 decimals
            assert np.linalg.norm(rp[:, None] - ep[None], axis=2).min(axis=1).max() < 5e-12, lab


def test_search_is_independent_of_thread_count():
    import numba

    from qtaim_gen.source.core.cp_search import search

    wfx, basis, _ = _load("XeCl4_step2_0_1")
    n0 = numba.get_num_threads()
    try:
        numba.set_num_threads(1)
        a = search(wfx, basis)
        numba.set_num_threads(min(2, numba.config.NUMBA_NUM_THREADS))
        b = search(wfx, basis)
    finally:
        numba.set_num_threads(n0)
    assert np.array_equal(a["pos"], b["pos"]) and np.array_equal(a["type"], b["type"])


def test_tuples_follow_multiwfn_loop_order():
    from qtaim_gen.source.core.cp_search import _tuples

    close = np.ones((4, 4), dtype=bool)
    np.fill_diagonal(close, False)
    close[0, 3] = close[3, 0] = False
    assert _tuples(close, 2).tolist() == [[0, 1], [0, 2], [1, 2], [1, 3], [2, 3]]
    assert _tuples(close, 3).tolist() == [[0, 1, 2], [1, 2, 3]]
    assert _tuples(close, 4).tolist() == []


@pytest.mark.parametrize("job", JOBS)
def test_superset_seeds_keep_every_multiwfn_cp(job):
    from qtaim_gen.source.core.cp_search import search

    wfx, basis, ref = _load(job)
    res = search(wfx, basis, seeds="superset")
    for lab, t in LABEL_TYPE.items():
        rp = np.array([c["pos_bohr"] for c in ref if c["label"] == lab]).reshape(-1, 3)
        ep = res["pos"][res["type"] == t]
        if len(rp):
            assert len(ep) and np.linalg.norm(rp[:, None] - ep[None], axis=2).min(axis=1).max() < 5e-12, lab


@pytest.mark.parametrize("job", JOBS)
@pytest.mark.parametrize("mode", ["multiwfn", "engine"])
def test_bond_path_pairs_match_multiwfn(job, mode):
    from qtaim_gen.source.core.cp_search import bond_paths, search

    wfx, basis, ref = _load(job)
    res = search(wfx, basis)
    bp = bond_paths(res, wfx, basis, mode=mode)
    epos = res["pos"][bp["bcp"]]
    for c in (c for c in ref if c["label"] == "BCP"):
        k = int(np.argmin(np.linalg.norm(epos - c["pos_bohr"], axis=1)))
        assert c["connected"] is not None
        assert sorted(int(a) + 1 for a in bp["atoms"][k]) == sorted(c["connected"])
    assert (bp["ncp_atom"][res["type"] == 0] >= 0).all()


def test_fortran_e_format():
    from qtaim_gen.source.core.cp_search import _fe

    assert _fe(3102.474076) == "  0.3102474076E+04"
    assert _fe(-0.3937299428e8) == " -0.3937299428E+08"
    assert _fe(0.0) == "  0.0000000000E+00"
    assert _fe(0.99999999999999) == "  0.1000000000E+01"
    assert _fe(1.999917076e-167) == "  0.1999917076-166"


def test_engine_cpprop_parses_like_multiwfn(tmp_path):
    import contextlib
    import io

    from qtaim_gen.source.core.cp_search import analyze, write_cpprop
    from qtaim_gen.source.core.parse_multiwfn import parse_qtaim
    from qtaim_gen.source.utils.validation import cpprop_integrity

    job = "XeCl4_step2_0_1"
    wfx, basis, _ = _load(job)
    res = analyze(wfx, basis, seeds="multiwfn", paths="multiwfn", boys_mode="multiwfn")
    out = tmp_path / "CPprop.txt"
    write_cpprop(out, res, wfx)
    assert cpprop_integrity(out.read_bytes()) == []
    inp = os.path.join(FIXTURES, job, "orca.inp")
    with contextlib.redirect_stdout(io.StringIO()):
        ref = parse_qtaim(os.path.join(FIXTURES, job, "multiwfn_cpprop.txt"), inp, orca_tf=True)
        eng = parse_qtaim(str(out), inp, orca_tf=True)
    assert set(ref) == set(eng)
    for k in ref:
        for f, v in ref[k].items():
            if f in ("cp_num", "delta_g_promolecular") or not isinstance(v, float):
                continue
            if f == "grad_norm":
                assert abs(eng[k][f] - v) < 1e-12, (k, f)
            elif f in ("ellip_e_dens", "eta"):
                assert abs(eng[k][f] - v) <= 2e-6, (k, f)
            else:
                assert eng[k][f] == pytest.approx(v, rel=1e-9, abs=1e-12 * abs(ref[k]["density_all"])), (k, f)


def test_sphere_seeds_are_deterministic_and_inside_the_ball():
    from qtaim_gen.source.core.cp_search import SPHERE_RADIUS, sphere_seeds

    coords = np.array([[0.0, 0.0, 0.0], [5.0, 1.0, -2.0]])
    a, b = sphere_seeds(coords, 128), sphere_seeds(coords, 128)
    assert a.shape == (256, 3) and np.array_equal(a, b)
    assert np.linalg.norm(a[:128] - coords[0], axis=1).max() <= SPHERE_RADIUS
    assert np.allclose(a[128:] - coords[1], a[:128] - coords[0])


def test_reseed_recovers_a_missing_cp():
    from qtaim_gen.source.core.cp_search import _poincare_hopf, reseed, search

    wfx, basis, _ = _load("XeCl4_step2_0_1")
    full = search(wfx, basis, seeds="superset")
    assert _poincare_hopf(full["type"]) == 1
    drop = int(np.flatnonzero(full["type"] == 1)[0])
    keep = [i for i in range(len(full["pos"])) if i != drop]
    cps = {"pos": full["pos"][keep], "type": full["type"][keep], "eigenvalues": full["eigenvalues"][keep],
           "stage": [full["stage"][i] for i in keep], "stats": []}
    assert _poincare_hopf(cps["type"]) == 2
    cps, stats = reseed(cps, wfx, basis)
    assert _poincare_hopf(cps["type"]) == 1 and stats
    assert np.linalg.norm(cps["pos"] - full["pos"][drop], axis=1).min() < 1e-8


def test_near_pairs_flags_close_cps():
    from qtaim_gen.source.core.cp_search import _near_pairs

    pos = np.array([[0.0, 0.0, 0.0], [0.0115, 0.0, 0.0], [1.0, 0.0, 0.0]])
    assert _near_pairs(pos) == [(0, 1)]


@pytest.mark.parametrize("job", JOBS)
def test_engine_types_agree_with_printed_eigenvalues(job, tmp_path):
    # the type must come from the Hessian CPprop.txt prints, or cpprop_integrity
    # rejects a near-degenerate CP on every rerun
    from qtaim_gen.source.core.cp_search import analyze, write_cpprop
    from qtaim_gen.source.utils.validation import cpprop_integrity

    wfx, basis, _ = _load(job)
    res = analyze(wfx, basis)
    assert np.array_equal(res["type"], (res["props"]["eigenvalues"] > 0).sum(axis=1))
    write_cpprop(tmp_path / "CPprop.txt", res, wfx)
    assert cpprop_integrity((tmp_path / "CPprop.txt").read_bytes()) == []


def test_reseed_skips_when_only_weak_cps_break_the_kept_set():
    # the full set satisfies Poincare-Hopf and a CP below WEAK_DROP breaks the kept
    # set: more seeds cannot change that, so no round runs
    from qtaim_gen.source.core.cp_search import _poincare_hopf, reseed, search

    wfx, basis, _ = _load("XeCl4_step2_0_1")
    full = search(wfx, basis, seeds="superset")
    drop = int(np.flatnonzero(full["type"] == 1)[0])
    keep = [i for i in range(len(full["pos"])) if i != drop]
    far = wfx["coords"].max(axis=0) + 40.0  # rho ~ 0 there
    cps = {"pos": np.vstack([full["pos"][keep], far]), "type": np.append(full["type"][keep], 1),
           "eigenvalues": np.vstack([full["eigenvalues"][keep], [-1.0, 1.0, 2.0]]),
           "stage": [full["stage"][i] for i in keep] + ["test"], "stats": []}
    assert _poincare_hopf(cps["type"]) == 1
    _, stats = reseed(cps, wfx, basis)
    assert stats == []


def test_engine_check_survives_a_corrupt_input_tarball(tmp_path):
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
    import qtaim_verify_stored as V

    wfx, basis, ref = _load("XeCl4_step2_0_1")
    src, res, work = tmp_path / "src", tmp_path / "res", tmp_path / "work"
    for d in (src, res, work):
        d.mkdir()
    (src / "orca.tar.zst").write_bytes(b"not a zstd tarball")
    out = V.engine_check(ref, os.path.join(FIXTURES, "XeCl4_step2_0_1", "multiwfn_cpprop.txt"), wfx, basis,
                         str(work), str(src), str(res))
    assert out["inp"] is False
    assert "error" not in out["multiwfn"] and out["multiwfn"]["counts"]["BCP"][0] == out["multiwfn"]["counts"]["BCP"][1]


def _fake_maximum(eigs, floor):
    # density_derivatives near a sharp maximum at the origin with Hessian diag(eigs) and a gradient that
    # never drops below floor, as at a heavy all-electron nucleus in double precision
    H = np.diag(eigs)
    calls = [0]

    def fake(pts, wfx, basis, **kw):
        calls[0] += 1
        g = pts @ H + floor * (-1) ** calls[0]  # roundoff Newton cannot cancel
        return {"gradient": g, "hessian": np.broadcast_to(H, (len(pts), 3, 3)).copy()}

    return fake


@pytest.mark.parametrize("eigs,accept,expected", [
    ((-3e15, -3e15, -3e15), False, -2),
    ((-3e15, -3e15, -3e15), True, 2),
    ((-3e15, -3e15, 3e15), True, -2),
])
def test_newton_accepts_a_precision_limited_maximum_only_when_asked(monkeypatch, eigs, accept, expected):
    import qtaim_gen.source.core.cp_search as C

    monkeypatch.setattr(C, "density_derivatives", _fake_maximum(eigs, 16.0))
    x, status, hess, gnorm, _ = C.newton([[1e-3, 0.0, 0.0]], {}, None, gradconv=1.0, accept_maximum=accept)
    assert status[0] == expected
    assert np.abs(x[0]).max() < 1e-13
    if expected == 2:
        assert np.allclose(hess[0], np.diag(eigs)) and gnorm[0] > 1.0


def test_newton_accept_maximum_leaves_normal_convergence_alone(monkeypatch):
    import qtaim_gen.source.core.cp_search as C

    monkeypatch.setattr(C, "density_derivatives", _fake_maximum((-3e15, -3e15, -3e15), 0.0))
    runs = [C.newton([[1e-3, 2e-3, 0.0]], {}, None, gradconv=1.0, accept_maximum=a) for a in (False, True)]
    assert runs[0][1][0] == runs[1][1][0] == 1
    for a, b in zip(runs[0], runs[1]):
        assert np.array_equal(a, b)


UCL6 = "/home/santiagovargas/dev/qtaim_generator/data/joe_collab/lrc_results/ijb_05_121a_UCl6_1/orca.wfx"


@pytest.mark.skipif(not os.path.exists(UCL6), reason="local X2C UCl6 wavefunction")
def test_x2c_uranium_nuclear_cp_is_found():
    from qtaim_gen.source.core.charge_engine import prepare_basis, read_wfx
    from qtaim_gen.source.core.cp_search import analyze

    wfx = read_wfx(UCL6)
    r = analyze(wfx, prepare_basis(wfx))
    u = int(np.flatnonzero(wfx["atnums"] == 92)[0])
    assert r["counts_full_set"] == [7, 6, 0, 0] and r["poincare_hopf"] == 1
    assert np.count_nonzero(r["ncp_atom"] == u) == 1
    bcp = r["bcp_atoms"][r["type"] == 1]
    assert (bcp >= 0).all() and np.isin(bcp, [u]).any(axis=1).all()
    nuc = [s for s in r["stats"] if s["stage"] == "nuclei"][0]
    # U sits 26 Bohr from the origin: its seed converges in the nucleus frame, not by the rescue
    assert nuc["nucleus_frame"] == 1 and nuc["precision_limited"] == 0


NITRATE_U = ("/home/santiagovargas/dev/qtaim_generator/data/actinide_dev/work/multiwfn/TZVPP/"
             "x2c_UN3O11_spin_1_PBE0_X2C-TZVPPall/orca.wfx")


@pytest.mark.skipif(not os.path.exists(NITRATE_U), reason="local X2C nitrate wavefunction")
def test_heavy_nucleus_cp_is_translation_invariant():
    """The U nucleus of a centred nitrate, and the same molecule shifted 40 and 100 Bohr:
    the shifted copies converge in the nucleus frame with the same offset from the nucleus,
    a gradient at the 1e-13 level and the same density and Laplacian."""
    from qtaim_gen.source.core.charge_engine import prepare_basis, read_wfx
    from qtaim_gen.source.core.cp_search import analyze

    out = {}
    for shift in (0.0, 40.0, 100.0):
        wfx = read_wfx(NITRATE_U)
        wfx["coords"] = wfx["coords"] + np.array([shift, -0.7 * shift, 0.4 * shift]) / np.sqrt(1.65)
        r = analyze(wfx, prepare_basis(wfx))
        u = int(np.flatnonzero(wfx["atnums"] == 92)[0])
        i = int(np.flatnonzero(r["ncp_atom"] == u)[0])
        nuc = [s for s in r["stats"] if s["stage"] == "nuclei"][0]
        out[shift] = (r["pos"][i] - wfx["coords"][u], {k: np.asarray(v)[i] for k, v in r["props"].items()}, nuc, r)
    for shift in (40.0, 100.0):
        off, props, nuc, r = out[shift]
        assert r["poincare_hopf"] == 1 and nuc["nucleus_frame"] == 1 and nuc["precision_limited"] == 0
        assert np.linalg.norm(props["gradient"]) < 1e-10
        assert abs(props["rho"] - out[0.0][1]["rho"]) <= 1e-12 * abs(out[0.0][1]["rho"])
        assert abs(props["laplacian"] - out[0.0][1]["laplacian"]) <= 1e-12 * abs(out[0.0][1]["laplacian"])
        assert np.linalg.norm(off) < 1e-12
