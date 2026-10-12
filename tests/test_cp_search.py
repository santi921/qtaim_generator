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
