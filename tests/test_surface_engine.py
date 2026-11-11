"""End-to-end check of core/surface_engine.py (the other_alie and other_esp steps):
on the two charge-engine wfx fixtures it must build the same surface mesh Multiwfn
3.8 does (grid size and vertex/edge/facet counts after elimination) and reproduce
the parsed ALIE_* and ESP_* values to Multiwfn's print precision. ESP parity is
with Multiwfn's exact code (iESPcode=1) and the engine's copy of its Boys function;
production Multiwfn uses libreta (iESPcode=2), which the exact kernel departs from
(docs/engine_roadmap/README.md, divergence changelog)."""

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


# half a unit in the last digit surfana prints (MW/surfana.f90:1193-1323)
ESP_TOL = {
    "Volume": 5e-6, "Surface_Density": 5e-5, "Minimal_value": 5e-6, "Maximal_value": 5e-6,
    "Overall_surface_area": 5e-6, "Positive_surface_area": 5e-6, "Negative_surface_area": 5e-6,
    "Overall_average_value": 5e-9, "Positive_average_value": 5e-9, "Negative_average_value": 5e-9,
    "Overall_variance": 5e-9, "Positive_variance": 5e-9, "Negative_variance": 5e-9,
    "Balance_of_charges": 5e-9, "Product_of_sigma": 5e-9, "Internal_charge_separation": 5e-9,
    "Molecular_polarity_index": 5e-9, "Nonpolar_surface_area": 5e-3, "Polar_surface_area": 5e-3,
    "Overall_skewness": 5e-11,
}


def _esp_fixture_names():
    if not os.path.isdir(FIXTURES):
        return []
    return sorted(
        d for d in os.listdir(FIXTURES) if os.path.isfile(os.path.join(FIXTURES, d, "multiwfn_reference_esp.json"))
    )


@pytest.mark.skipif(not HAS_NUMBA, reason="numba not installed")
@pytest.mark.parametrize("name", _esp_fixture_names())
def test_esp_surface_reproduces_multiwfn_exact_esp(tmp_path, name):
    folder = os.path.join(FIXTURES, name)
    out = tmp_path / "esp.json"
    subprocess.run(
        [sys.executable, "-m", "qtaim_gen.source.core.surface_engine", "--function", "esp", "--boys_mode", "multiwfn",
         "--wfx", os.path.join(folder, "orca.wfx"), "--out", str(out), "--nthreads", "2"],
        check=True,
        capture_output=True,
    )
    eng = json.loads(out.read_text())
    with open(os.path.join(folder, "multiwfn_reference_esp.json")) as f:
        ref = json.load(f)["eleesp1"]

    assert eng["_meta"]["vef_before"] == ref["vef_before"]
    assert eng["_meta"]["vef_after"] == ref["vef_after"]
    assert set(eng["other_esp"]) == set(ref["other_esp"])
    for k, v in ref["other_esp"].items():
        assert eng["other_esp"][k] == pytest.approx(v, abs=ESP_TOL[k[4:]]), k


@pytest.mark.skipif(not HAS_NUMBA, reason="numba not installed")
def test_esp_statistics_of_a_one_signed_surface():
    """An all-positive surface: averages over the empty sign are NaN (0/0) and
    its variance 0, as Multiwfn prints them; the balance of charges is then 0."""
    import numpy as np

    from qtaim_gen.source.core.surface_engine import _facet_statistics

    val = np.array([0.1, 0.2, 0.3, 0.4])
    st = _facet_statistics(val, np.arange(4), np.array([[0, 1, 2], [1, 2, 3]]), np.array([1.0, 2.0]))
    assert np.isnan(st["avg_neg"]) and st["var_neg"] == 0.0 and st["area_neg"] == 0.0
    assert st["nu"] == 0.0 and st["avg_pos"] == pytest.approx((0.2 + 2 * 0.3) / 3)


@pytest.mark.skipif(not HAS_NUMBA, reason="numba not installed")
def test_full_set_2_surface_engine_writes_esp_into_other_json(tmp_path):
    """gbw_analysis at full_set 2 with the engines: the surface engine writes
    other_esp.json next to other_alie.json, parse_multiwfn compiles both into
    other.json, which then validates at full_set 2; each routine gets a provenance entry
    at the engine version, and full_set 1 leaves the ESP surface out."""
    import logging
    import shutil

    from qtaim_gen.source.core.omol import _run_surface_engine, parse_multiwfn
    from qtaim_gen.source.data.engine_versions import ENGINE_VERSION
    from qtaim_gen.source.data.multiwfn import ENGINE_ROUTINES
    from qtaim_gen.source.utils import provenance
    from qtaim_gen.source.utils.validation import validate_other_dict

    src = os.path.join(FIXTURES, "rmechdb_652_step10_0_2")
    logger = logging.getLogger("test_surface_engine")
    for level in (1, 2):
        folder = tmp_path / f"l{level}"
        folder.mkdir()
        for f in ("orca.wfx", "orca.inp"):
            shutil.copy(os.path.join(src, f), folder / f)
        assert _run_surface_engine(str(folder), n_threads=2, logger=logger, full_set=level)
        assert (folder / "other_esp.json").exists() is (level == 2)
        parse_multiwfn(str(folder), separate=True, logger=logger, full_set=level, skip_routines=ENGINE_ROUTINES)
        other = json.loads((folder / "other.json").read_text())
        assert validate_other_dict(str(folder / "other.json"), full_set=level)
        assert ("ESP_Volume" in other) is (level == 2)
        prov = provenance.load(str(folder))
        want = {"other_alie", "other_geometry"} | ({"other_esp"} if level == 2 else set())
        assert provenance.engine_routines(prov) == want
        assert {prov["routines"][r]["engine_version"] for r in want} == {ENGINE_VERSION}
        assert set(prov["engines"]["surface_engine"]["surfaces"]) == want - {"other_geometry"}
    with open(os.path.join(src, "multiwfn_reference_esp.json")) as f:
        ref = json.load(f)["eleesp1"]["other_esp"]
    # geometry-only fields do not depend on the ESP kernel
    for k in ("ESP_Volume", "ESP_Overall_surface_area"):
        assert other[k] == pytest.approx(ref[k], abs=ESP_TOL[k[4:]]), k


def _restart_folder(tmp_path, other, routines, timings):
    """A finished job with generator/other.json, timings.json and provenance.json; the root
    timings.json is seeded from generator/ as run_jobs does on a restart."""
    import shutil

    from qtaim_gen.source.utils import provenance

    src = os.path.join(FIXTURES, "rmechdb_652_step10_0_2")
    folder = tmp_path / "job"
    gen = folder / "generator"
    gen.mkdir(parents=True)
    for f in ("orca.wfx", "orca.inp"):
        shutil.copy(os.path.join(src, f), folder / f)
    (gen / "other.json").write_text(json.dumps(other))
    (gen / "timings.json").write_text(json.dumps(timings))
    prov = provenance.empty()
    prov["routines"] = routines
    (gen / provenance.PROVENANCE_FILE).write_text(json.dumps(prov))
    shutil.copy(gen / "timings.json", folder / "timings.json")
    return folder


@pytest.mark.skipif(not HAS_NUMBA, reason="numba not installed")
def test_restart_at_full_set_2_adds_only_the_esp_surface_and_validates(tmp_path):
    """A full_set 1 folder from engine version 2, restarted at full_set 2: the surface
    engine computes only other_esp (ALIE and planarity stay as stored, with their version 2
    records), writes its 'other_esp' timing, and the folder validates at full_set 2. A
    second restart finds every routine and skips."""
    import logging

    from qtaim_gen.source.core.omol import _run_surface_engine, move_results_to_folder, parse_multiwfn
    from qtaim_gen.source.data.engine_versions import ENGINE_VERSION, ROUTINE_ENGINE, routines_at
    from qtaim_gen.source.data.multiwfn import ENGINE_ROUTINES
    from qtaim_gen.source.utils import provenance
    from qtaim_gen.source.utils.validation import get_expected_timing_keys, validate_timing_dict

    logger = logging.getLogger("test_surface_engine")
    stored = {"ALIE_Volume": 123.0, "mpp_full": 0.5}
    # Multiwfn's timings for the full_set 2 routines no engine covers; no other_esp timing
    keys, _ = get_expected_timing_keys(full_set=2)
    v2 = routines_at(2) - {"qtaim", "convert"}
    timings = {k: 1.0 for k in keys if k not in v2 and k != "other_esp"}
    timings.update({"charge_engine": 1.0, "surface_engine": 1.0})
    folder = _restart_folder(tmp_path, stored, {r: {"by": ROUTINE_ENGINE[r], "engine_version": 2} for r in v2},
                             timings)
    gen = folder / "generator"
    assert not validate_timing_dict(str(gen / "timings.json"), full_set=2, charge_engine=True)

    assert _run_surface_engine(str(folder), n_threads=2, logger=logger, restart=True, full_set=2)
    assert (folder / "other_esp.json").exists()
    assert not (folder / "other_alie.json").exists() and not (folder / "other_geometry.json").exists()
    parse_multiwfn(str(folder), separate=True, logger=logger, full_set=2, skip_routines=ENGINE_ROUTINES)
    move_results_to_folder(str(folder), logger=logger, clean=True)

    other = json.loads((gen / "other.json").read_text())
    assert {k: other[k] for k in stored} == stored
    with open(os.path.join(FIXTURES, "rmechdb_652_step10_0_2", "multiwfn_reference_esp.json")) as f:
        ref = json.load(f)["eleesp1"]["other_esp"]
    assert other["ESP_Volume"] == pytest.approx(ref["ESP_Volume"], abs=ESP_TOL["Volume"])
    assert json.loads((gen / "timings.json").read_text())["other_esp"] > 0
    prov = json.loads((gen / provenance.PROVENANCE_FILE).read_text())
    assert prov["routines"]["other_esp"]["engine_version"] == ENGINE_VERSION
    assert prov["routines"]["other_alie"]["engine_version"] == 2 and prov["routines"]["hirshfeld"]["engine_version"] == 2
    assert validate_timing_dict(str(gen / "timings.json"), full_set=2, charge_engine=True)

    assert not (folder / "other_esp.json").exists()  # parse_multiwfn removed the intermediate
    assert _run_surface_engine(str(folder), n_threads=2, logger=logger, restart=True, full_set=2)
    assert not (folder / "other_esp.json").exists()


@pytest.mark.skipif(not HAS_NUMBA, reason="numba not installed")
def test_restart_recomputes_surface_values_the_engine_did_not_write(tmp_path):
    """ESP values in other.json that Multiwfn computed are recomputed by the engine, not
    reused: provenance, not the presence of the values, says what the engine wrote."""
    import logging

    from qtaim_gen.source.core.omol import _run_surface_engine
    from qtaim_gen.source.data.engine_versions import ENGINE_VERSION

    routines = {"other_alie": {"by": "surface_engine", "engine_version": ENGINE_VERSION},
                "other_geometry": {"by": "surface_engine", "engine_version": ENGINE_VERSION},
                "other_esp": {"by": "multiwfn"}}
    folder = _restart_folder(tmp_path, {"ALIE_Volume": 123.0, "mpp_full": 0.5, "ESP_Volume": 1.0}, routines,
                             {"surface_engine": 1.0, "other_esp": 9.0})
    assert _run_surface_engine(str(folder), n_threads=2, logger=logging.getLogger("t"), restart=True, full_set=2)
    assert (folder / "other_esp.json").exists() and not (folder / "other_alie.json").exists()


@pytest.mark.skipif(not HAS_NUMBA, reason="numba not installed")
def test_esp_failure_keeps_alie_and_is_retried_once_then_skipped(tmp_path, monkeypatch):
    """An ESP surface failure keeps the ALIE and planarity results; the next pass retries
    only the ESP surface, and a second failure records other_esp as permanently skipped,
    after which validation no longer expects it."""
    import logging
    import shutil

    from qtaim_gen.source.core import surface_engine
    from qtaim_gen.source.core.omol import _run_surface_engine, move_results_to_folder
    from qtaim_gen.source.utils import provenance

    def boom(*a, **k):
        raise MemoryError("surface too large")

    monkeypatch.setattr(surface_engine, "esp_surface", boom)
    logger = logging.getLogger("test_surface_engine")
    folder = tmp_path / "job"
    folder.mkdir()
    for f in ("orca.wfx", "orca.inp"):
        shutil.copy(os.path.join(FIXTURES, "rmechdb_652_step10_0_2", f), folder / f)
    assert not _run_surface_engine(str(folder), n_threads=2, logger=logger, full_set=2)
    assert (folder / "other_alie.json").exists() and (folder / "other_geometry.json").exists()
    prov = provenance.load(str(folder))
    assert provenance.engine_routines(prov) == {"other_alie", "other_geometry"} and not provenance.skipped(prov)
    (folder / "other.json").write_text(json.dumps({"ALIE_Volume": 1.0, "mpp_full": 0.1}))
    move_results_to_folder(str(folder), logger=logger, clean=True)
    for f in ("other_alie.json", "other_geometry.json"):
        os.remove(folder / f)

    assert _run_surface_engine(str(folder), n_threads=2, logger=logger, restart=True, full_set=2)
    assert not (folder / "other_alie.json").exists()  # only the ESP surface was retried
    prov = provenance.load(str(folder))
    assert prov["skipped"]["other_esp"]["by"] == "surface_engine"
    assert "MemoryError" in prov["skipped"]["other_esp"]["reason"]
