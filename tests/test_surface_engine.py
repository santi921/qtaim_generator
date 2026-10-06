"""End-to-end check of core/surface_engine.py (the other_alie step): on the two
charge-engine wfx fixtures it must build the same surface mesh Multiwfn 3.8 does
(grid size and vertex/edge/facet counts after elimination) and reproduce the
eight parsed ALIE_* values to Multiwfn's print precision."""

import importlib.util
import json
import os
import subprocess
import sys

import pytest

FIXTURES = os.path.join(os.path.dirname(__file__), "test_files", "charge_engine")
HAS_NUMBA = importlib.util.find_spec("numba") is not None

# printed precision: volume/area f12.5, min/max f13.5 (eV), density f10.4, skewness f20.10
TOL = {
    "ALIE_Volume": 1e-5, "ALIE_Overall_surface_area": 1e-5, "ALIE_Positive_surface_area": 1e-5,
    "ALIE_Negative_surface_area": 1e-5, "ALIE_Minimal_value": 1e-5, "ALIE_Maximal_value": 1e-5,
    "ALIE_Surface_Density": 1e-4, "ALIE_Overall_skewness": 1e-9,
}


def _fixture_names():
    if not os.path.isdir(FIXTURES):
        return []
    return sorted(
        d for d in os.listdir(FIXTURES) if os.path.isfile(os.path.join(FIXTURES, d, "multiwfn_reference_alie.json"))
    )


@pytest.mark.skipif(not HAS_NUMBA, reason="numba not installed")
@pytest.mark.parametrize("name", _fixture_names())
def test_surface_engine_reproduces_multiwfn(tmp_path, name):
    folder = os.path.join(FIXTURES, name)
    out = tmp_path / "alie.json"
    subprocess.run(
        [sys.executable, "-m", "qtaim_gen.source.core.surface_engine",
         "--wfx", os.path.join(folder, "orca.wfx"), "--out", str(out), "--nthreads", "2"],
        check=True,
        capture_output=True,
    )
    eng = json.loads(out.read_text())
    with open(os.path.join(folder, "multiwfn_reference_alie.json")) as f:
        ref = json.load(f)

    assert eng["_meta"]["grid"] == ref["grid"]
    assert eng["_meta"]["vef_after"] == ref["vef_after"]
    assert set(eng["other_alie"]) == set(ref["other_alie"])
    for k, v in ref["other_alie"].items():
        assert eng["other_alie"][k] == pytest.approx(v, abs=TOL[k]), k
