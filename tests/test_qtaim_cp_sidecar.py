"""QTAIM P4: the qtaim_cp.json sidecar (ring and cage CPs, non-nuclear attractors,
unpaired BCPs), the engine's nna / nna_vacuum / bcp_unpaired metadata, and the
permanent QTAIM skip for an ECP atom without EDF core density."""

import importlib.util
import json
import logging
import os
import shutil

import numpy as np
import pytest

from qtaim_gen.source.core.parse_qtaim import (
    NNA_VACUUM_RHO,
    get_qtaim_descs,
    only_atom_cps,
    qtaim_cp_sidecar,
)

LOG = logging.getLogger("test_qtaim_cp_sidecar")
TEST_FILES = os.path.join(os.path.dirname(__file__), "test_files")
FIXTURES = os.path.join(TEST_FILES, "charge_engine")
HAS_NUMBA = importlib.util.find_spec("numba") is not None


def test_ring_cps_are_kept_only_on_request():
    path = os.path.join(TEST_FILES, "CPprop.txt")
    default = get_qtaim_descs(path)
    full = get_qtaim_descs(path, keep_all=True)
    rings = [k for k in full if k.endswith("_ring")]
    assert len(rings) == 2
    assert not any(k.endswith(("_ring", "_cage")) for k in default)
    assert set(full) - set(default) == set(rings)


def test_vacuum_threshold_matches_the_engine_weak_flag():
    from qtaim_gen.source.core.cp_search import WEAK_FLAG

    assert NNA_VACUUM_RHO == WEAK_FLAG


@pytest.mark.skipif(not HAS_NUMBA, reason="numba not installed")
def test_bcp_ends_point_at_the_nuclear_cps_of_the_pair():
    from qtaim_gen.source.core.charge_engine import prepare_basis, read_wfx
    from qtaim_gen.source.core.cp_search import analyze

    wfx = read_wfx(os.path.join(FIXTURES, "XeCl4_step2_0_1", "orca.wfx"))
    res = analyze(wfx, prepare_basis(wfx))
    bcps = np.flatnonzero(res["type"] == 1)
    assert len(bcps)
    for b in bcps:
        ends = res["bcp_ends"][b]
        assert (res["type"][ends] == 0).all()
        assert sorted(res["ncp_atom"][ends]) == sorted(res["bcp_atoms"][b])
    assert (res["bcp_ends"][res["type"] != 1] == -1).all()


def _engine_cpprop_with_nna(tmp_path):
    """The XeCl4 engine result with one Cl nuclear CP relabeled as a non-nuclear
    attractor (so its Xe-Cl BCP is unpaired and ends there), written as CPprop.txt
    and qtaim.out; returns the folder, the result and the relabeled CP number."""
    from qtaim_gen.source.core.charge_engine import prepare_basis, read_wfx
    from qtaim_gen.source.core.cp_search import analyze, qtaim_engine_out, write_cpprop

    wfx = read_wfx(os.path.join(FIXTURES, "XeCl4_step2_0_1", "orca.wfx"))
    res = analyze(wfx, prepare_basis(wfx))
    cl = int(np.flatnonzero((res["type"] == 0) & (wfx["atnums"][np.maximum(res["ncp_atom"], 0)] == 17))[0])
    res["ncp_atom"] = res["ncp_atom"].copy()
    res["bcp_atoms"] = res["bcp_atoms"].copy()
    res["ncp_atom"][cl] = -1
    for b in np.flatnonzero(res["type"] == 1):
        if cl in res["bcp_ends"][b]:
            res["bcp_atoms"][b] = -1
    write_cpprop(str(tmp_path / "CPprop.txt"), res, wfx)
    (tmp_path / "qtaim.out").write_text(qtaim_engine_out(res, 1.0, {"seeds": "superset"}))
    shutil.copy(os.path.join(FIXTURES, "XeCl4_step2_0_1", "orca.inp"), tmp_path / "orca.inp")
    return tmp_path, res, cl + 1


@pytest.mark.skipif(not HAS_NUMBA, reason="numba not installed")
def test_engine_meta_lists_nna_and_unpaired_bcps(tmp_path):
    from qtaim_gen.source.utils.validation import qtaim_engine_meta

    folder, res, nna = _engine_cpprop_with_nna(tmp_path)
    meta = qtaim_engine_meta((folder / "qtaim.out").read_text())
    assert meta["nna"] == [nna]
    assert meta["nna_vacuum"] == []  # a nuclear position, rho far above the threshold
    unpaired = meta["bcp_unpaired"]
    assert len(unpaired) == 1
    bcp, e1, e2 = unpaired[0]
    assert res["type"][bcp - 1] == 1 and nna in (e1, e2)


@pytest.mark.skipif(not HAS_NUMBA, reason="numba not installed")
def test_sidecar_holds_nna_and_unpaired_bcp_with_path_ends(tmp_path):
    from qtaim_gen.source.utils.validation import qtaim_engine_meta

    folder, res, nna = _engine_cpprop_with_nna(tmp_path)
    meta = qtaim_engine_meta((folder / "qtaim.out").read_text())
    side = qtaim_cp_sidecar(str(folder / "CPprop.txt"), str(folder / "orca.inp"), orca_tf=True,
                            engine_meta=meta)
    assert side["_meta"]["source"] == "qtaim_engine"
    assert side["_meta"]["counts"] == {"ring": 0, "cage": 0, "nna": 1, "unpaired_bcp": 1}
    entry = side["nna"][str(nna)]
    assert entry["vacuum"] is False and entry["density_all"] > 1
    assert entry["nearest_dist_ang"][0] < 1e-3  # it sits on the Cl nucleus
    (bcp_key, bcp), = side["unpaired_bcp"].items()
    assert bcp["path_ends"] == meta["bcp_unpaired"][0][1:]
    assert bcp["nearest_atoms"][0] != bcp["nearest_atoms"][1]
    # qtaim.json is unchanged: the NNA and the unpaired BCP are not in it
    atoms, bonds = only_atom_cps(get_qtaim_descs(str(folder / "CPprop.txt")))
    assert not any("Unknown" in k for k in atoms)
    json.dumps(side)  # serializable as written


def test_sidecar_from_a_multiwfn_cpprop_has_rings_and_no_path_ends():
    side = qtaim_cp_sidecar(os.path.join(TEST_FILES, "CPprop.txt"), os.path.join(TEST_FILES, "input.in"))
    assert side["_meta"]["source"] == "cpprop"
    assert len(side["ring"]) == 2 and not side["cage"]
    for entry in side["ring"].values():
        assert len(entry["nearest_atoms"]) == 3
        assert entry["nearest_dist_ang"] == sorted(entry["nearest_dist_ang"])
    assert all("path_ends" not in e for e in side["unpaired_bcp"].values())


def test_nna_keys_are_unique_per_cp(tmp_path):
    text = open(os.path.join(TEST_FILES, "CPprop.txt")).read()
    # relabel the first two nuclear CPs as non-nuclear attractors
    import re

    n = 0

    def repl(m):
        nonlocal n
        n += 1
        return " Corresponding nucleus:  Unknown" if n <= 2 else m.group(0)

    (tmp_path / "CPprop.txt").write_text(re.sub(r" Corresponding nucleus:.*", repl, text))
    descs = get_qtaim_descs(str(tmp_path / "CPprop.txt"))
    assert len([k for k in descs if k.endswith("_Unknown")]) == 2


@pytest.mark.skipif(not HAS_NUMBA, reason="numba not installed")
def test_ecp_atom_without_edf_is_a_permanent_qtaim_skip(tmp_path):
    from qtaim_gen.source.core.horton import strip_edf
    from qtaim_gen.source.core.omol import _run_qtaim_engine
    from qtaim_gen.source.utils import provenance
    from qtaim_gen.source.utils.io import check_results_exist

    # XeCl4: the Xe ECP's core density comes from the EDF block
    text = open(os.path.join(FIXTURES, "XeCl4_step2_0_1", "orca.wfx")).read()
    assert "EDF" in text
    (tmp_path / "orca.wfx").write_text(strip_edf(text))
    assert _run_qtaim_engine(str(tmp_path), n_threads=1, logger=LOG)
    skipped = provenance.load(str(tmp_path))["skipped"]
    assert skipped["qtaim"]["reason"] == "ecp_without_edf" and skipped["qtaim"]["by"] == "qtaim_engine"
    assert not (tmp_path / "CPprop.txt").exists()
    # a second run does not try again
    assert _run_qtaim_engine(str(tmp_path), n_threads=1, logger=LOG)

    # the restart pre-check does not ask for the qtaim.json the skip never wrote
    gen = tmp_path / "generator"
    gen.mkdir()
    for name in ("timings.json", "other.json", "fuzzy_full.json", "charge.json", "bond.json"):
        (gen / name).write_text("{}\n")
    assert check_results_exist(str(tmp_path), move_results=True)
    provenance.drop(str(tmp_path), ["qtaim"])
    assert "qtaim" not in provenance.skipped(provenance.load(str(tmp_path)))
    assert not check_results_exist(str(tmp_path), move_results=True)

