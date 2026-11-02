"""Planarity (other_geometry) from core/surface_engine.py against Multiwfn 3.8.

References: tests/test_files/charge_engine/planarity_reference.json (Multiwfn prints
MPP and SDP with f12.6 Angstrom, so agreement is bounded by that rounding).
"""

import importlib.util
import json
import logging
import os
import shutil

import numpy as np
import pytest

FIXTURES = os.path.join(os.path.dirname(__file__), "test_files", "charge_engine")
HAS_NUMBA = importlib.util.find_spec("numba") is not None
PRINT_TOL = 5.1e-7
KEYS = ("mpp_full", "sdp_full", "mpp_heavy", "sdp_heavy")

with open(os.path.join(FIXTURES, "planarity_reference.json")) as f:
    REFERENCE = json.load(f)["molecules"]


@pytest.mark.skipif(not HAS_NUMBA, reason="numba not installed")
@pytest.mark.parametrize("name", sorted(REFERENCE))
def test_planarity_matches_multiwfn(name):
    from qtaim_gen.source.core.surface_engine import planarity

    mol = REFERENCE[name]
    got = planarity(np.array(mol["coords_bohr"]), mol["atnums"])
    for k in KEYS:
        # Multiwfn prints MPP NaN for an empty heavy-atom set; the engine gives 0
        want = 0.0 if mol["multiwfn"][k] is None else mol["multiwfn"][k]
        assert got[k] == pytest.approx(want, abs=PRINT_TOL), (name, k)


@pytest.mark.skipif(not HAS_NUMBA, reason="numba not installed")
def test_planarity_is_independent_of_atom_order_and_translation():
    from qtaim_gen.source.core.surface_engine import planarity

    mol = REFERENCE["MR_169236_0_18_0_1"]
    coords, atnums = np.array(mol["coords_bohr"]), np.array(mol["atnums"])
    perm = np.random.default_rng(0).permutation(len(atnums))
    base = planarity(coords, atnums)
    moved = planarity(coords[perm] + np.array([30.0, -12.0, 5.0]), atnums[perm])
    for k in KEYS:
        assert moved[k] == pytest.approx(base[k], abs=1e-12), k


@pytest.mark.skipif(not HAS_NUMBA, reason="numba not installed")
def test_surface_engine_writes_planarity_into_other_json(tmp_path):
    """Engine mode: other_geometry comes from the surface engine, not Multiwfn, and
    compiles into other.json next to the ALIE fields."""
    from qtaim_gen.source.core.omol import _run_surface_engine, parse_multiwfn
    from qtaim_gen.source.data.multiwfn import ENGINE_ROUTINES

    assert "other_geometry" in ENGINE_ROUTINES
    src = os.path.join(FIXTURES, "XeCl4_step2_0_1")
    for f in ("orca.wfx", "orca.inp"):
        shutil.copy(os.path.join(src, f), tmp_path / f)
    logger = logging.getLogger("test_planarity")
    assert _run_surface_engine(str(tmp_path), n_threads=2, logger=logger)
    parse_multiwfn(str(tmp_path), separate=True, logger=logger, full_set=0, skip_routines=ENGINE_ROUTINES)

    other = json.loads((tmp_path / "other.json").read_text())
    # XeCl4 is square planar: local Multiwfn prints 0.000000 for all four
    for k in KEYS:
        assert other[k] == pytest.approx(0.0, abs=PRINT_TOL), k
    assert "ALIE_Volume" in other


def _write_generator(folder, other):
    gen = folder / "generator"
    gen.mkdir()
    (gen / "timings.json").write_text(json.dumps({"surface_engine": 3.0}))
    (gen / "other.json").write_text(json.dumps(other))


def test_restart_skips_only_when_planarity_and_alie_are_present(tmp_path):
    """A folder whose other.json has ALIE but no planarity (a run before
    other_geometry moved into the surface engine, with Multiwfn skipped) is rerun."""
    from qtaim_gen.source.core.omol import _run_surface_engine

    logger = logging.getLogger("test_planarity")
    done = tmp_path / "done"
    done.mkdir()
    _write_generator(done, {"ALIE_Volume": 1.0, "mpp_full": 0.1})
    assert _run_surface_engine(str(done), n_threads=1, logger=logger, restart=True)

    alie_only = tmp_path / "alie_only"
    alie_only.mkdir()
    _write_generator(alie_only, {"ALIE_Volume": 1.0})
    # not skipped, so it looks for a wavefunction and finds none
    assert not _run_surface_engine(str(alie_only), n_threads=1, logger=logger, restart=True)
