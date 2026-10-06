"""Tests for the one-pass charge/fuzzy engine (core/charge_engine.py).

Tests cover the generated data modules (stdlib only) and an end-to-end check,
skipped when numba is missing: the worker's output on two small wfx fixtures must
reproduce the Multiwfn reference values (parsed from Multiwfn 3.8 per-step
outputs with the production parsers) to Multiwfn's print precision.
"""

import importlib.util
import json
import os
import subprocess
import sys

import pytest

SOURCE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "qtaim_gen", "source"
)
DATA = os.path.join(SOURCE, "data")
HELPERS = os.path.join(SOURCE, "scripts", "helpers")
FIXTURES = os.path.join(os.path.dirname(__file__), "test_files", "charge_engine")
HAS_NUMBA = importlib.util.find_spec("numba") is not None

# Multiwfn prints charges and fuzzy integrals with 8 decimals and open-shell
# bond orders with 6, so agreement is bounded by that rounding
CHARGE_TOL = 1e-7
BOND_TOL = 1e-6


def _load(name, folder=DATA):
    spec = importlib.util.spec_from_file_location(name, os.path.join(folder, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


tables = _load("multiwfn_tables")
raddens = _load("multiwfn_atmraddens")


class TestTables:
    def test_lengths(self):
        for t in (tables.VDWR, tables.COVR, tables.COVR_PYY, tables.COVR_TIANLU):
            assert len(t) == 151
        assert len(tables.TYPE2IX) == len(tables.TYPE2IY) == len(tables.TYPE2IZ) == 56

    def test_spot_values(self):
        assert tables.VDWR[6] == 1.7
        assert tables.COVR[26] == 1.32
        assert tables.COVR_PYY[97] == 1.68
        assert tables.COVR_TIANLU[8] == 0.76  # row-2 main group takes carbon's radius

    def test_tianlu_matches_horton_worker(self):
        worker = _load("horton_worker", HELPERS)
        for z, r in worker.COVR_TIANLU.items():
            assert tables.COVR_TIANLU[z] == pytest.approx(r)

    def test_primitive_types_are_cartesian_shells(self):
        # s, p, d, f, g, h blocks by total angular momentum
        sizes = {0: 1, 1: 3, 2: 6, 3: 10, 4: 15, 5: 21}
        ltot = [x + y + z for x, y, z in zip(tables.TYPE2IX, tables.TYPE2IY, tables.TYPE2IZ)]
        for ang, n in sizes.items():
            assert ltot.count(ang) == n

    def test_wfx_g_map_is_a_permutation_of_g(self):
        assert sorted(tables.WFX_G_TO_MWFN) == list(range(21, 36))
        assert sorted(tables.WFX_G_TO_MWFN.values()) == list(range(21, 36))

    def test_cm5_pair_overrides(self):
        assert tables.CM5_T_PAIRS[(1, 8)] == 0.1671
        assert tables.CM5_ALPHA == 2.474


class TestRadialDensities:
    def test_covers_h_to_lr(self):
        assert sorted(raddens.RHO) == list(range(1, 104))

    def test_grid_is_increasing(self):
        r = raddens.RADPOS
        assert len(r) == 200
        assert all(b > a for a, b in zip(r, r[1:]))

    def test_cutoff_is_last_grid_point(self):
        # atmrhocut(Z) == atmradpos(npt(Z)), per the atmraddens.f90 comment
        for z, rho in raddens.RHO.items():
            assert raddens.ATMRHOCUT[z] == pytest.approx(raddens.RADPOS[len(rho) - 1], abs=0.01)

    def test_densities_positive_and_decaying(self):
        for z, rho in raddens.RHO.items():
            assert rho[0] > 0
            assert rho[-1] < rho[0]


def _fixture_names():
    if not os.path.isdir(FIXTURES):
        return []
    return sorted(d for d in os.listdir(FIXTURES) if os.path.isfile(os.path.join(FIXTURES, d, "orca.wfx")))


@pytest.mark.skipif(not HAS_NUMBA, reason="numba not installed")
@pytest.mark.parametrize("name", _fixture_names())
def test_worker_reproduces_multiwfn(tmp_path, name):
    folder = os.path.join(FIXTURES, name)
    out = tmp_path / "engine.json"
    subprocess.run(
        [sys.executable, "-m", "qtaim_gen.source.core.charge_engine",
         "--wfx", os.path.join(folder, "orca.wfx"), "--out", str(out), "--nthreads", "2"],
        check=True,
        capture_output=True,
    )
    with open(out) as f:
        eng = json.load(f)
    with open(os.path.join(folder, "multiwfn_reference.json")) as f:
        ref = json.load(f)

    assert set(ref) <= set(eng), f"engine lacks {set(ref) - set(eng)}"
    assert not {"vdd", "mbis", "mbis_fuzzy_density", "mbis_fuzzy_spin"} & set(eng), "full_set 0 ran level-1 schemes"
    for step, ref_d in ref.items():
        if step == "fuzzy_bond":
            assert set(eng[step]) == set(ref_d)
            for k, v in ref_d.items():
                assert eng[step][k] == pytest.approx(v, abs=BOND_TOL), (step, k)
        elif "charge" in ref_d:
            for k, v in ref_d["charge"].items():
                assert eng[step]["charge"][k] == pytest.approx(v, abs=CHARGE_TOL), (step, k)
        else:
            for k, v in ref_d[step].items():
                assert eng[step][step][k] == pytest.approx(v, abs=CHARGE_TOL), (step, k)


@pytest.mark.skipif(not HAS_NUMBA, reason="numba not installed")
@pytest.mark.parametrize("name", _fixture_names())
def test_worker_reproduces_multiwfn_level1(tmp_path, name):
    """full_set 1 adds vdd, mbis and the MBIS fuzzy integrals; references are the
    Multiwfn 3.8 outputs of the production inputs, parsed with the production parsers."""
    folder = os.path.join(FIXTURES, name)
    out = tmp_path / "engine.json"
    subprocess.run(
        [sys.executable, "-m", "qtaim_gen.source.core.charge_engine",
         "--wfx", os.path.join(folder, "orca.wfx"), "--out", str(out), "--nthreads", "2",
         "--full_set", "1"],
        check=True,
        capture_output=True,
    )
    with open(out) as f:
        eng = json.load(f)
    with open(os.path.join(folder, "multiwfn_reference_level1.json")) as f:
        ref = json.load(f)

    assert set(ref) <= set(eng), f"engine lacks {set(ref) - set(eng)}"
    for step, ref_d in ref.items():
        if "charge" in ref_d:
            for k, v in ref_d["charge"].items():
                assert eng[step]["charge"][k] == pytest.approx(v, abs=CHARGE_TOL), (step, k)
            if "dipole" in ref_d:
                # dipoles are printed with 6 decimals
                assert eng[step]["dipole"]["mag"] == pytest.approx(ref_d["dipole"]["mag"], abs=BOND_TOL)
        else:
            for k, v in ref_d[step].items():
                assert eng[step][step][k] == pytest.approx(v, abs=CHARGE_TOL), (step, k)


class TestTimingValidation:
    """A charge_engine timing stands in for the ENGINE_ROUTINES timings and a
    surface_engine timing for 'other'; engine mode requires both."""

    BASE = {"qtaim": 5.0, "other": 3.0}
    MWFN = {
        "hirshfeld": 1.0, "becke": 1.0, "adch": 1.0, "cm5": 1.0, "fuzzy_bond": 1.0,
        "becke_fuzzy_density": 1.0, "hirsh_fuzzy_density": 1.0,
    }

    def _write(self, tmp_path, timings):
        path = tmp_path / "timings.json"
        path.write_text(json.dumps(timings))
        return str(path)

    def _validate(self, path, **kw):
        from qtaim_gen.source.utils.validation import validate_timing_dict

        return validate_timing_dict(path, full_set=0, **kw)

    def test_multiwfn_timings_pass(self, tmp_path):
        assert self._validate(self._write(tmp_path, {**self.BASE, **self.MWFN}))

    def test_engine_timing_replaces_routine_keys(self, tmp_path):
        path = self._write(tmp_path, {**self.BASE, "charge_engine": 2.0})
        assert self._validate(path, spin_tf=True)
        # engine mode also needs the ALIE from the surface engine
        assert not self._validate(path, charge_engine=True)
        path = self._write(tmp_path, {**self.BASE, "charge_engine": 2.0, "surface_engine": 1.0})
        assert self._validate(path, charge_engine=True)

    def test_surface_engine_timing_stands_in_for_other(self, tmp_path):
        path = self._write(tmp_path, {"qtaim": 5.0, "charge_engine": 2.0, "surface_engine": 1.0})
        assert self._validate(path, charge_engine=True)
        assert not self._validate(self._write(tmp_path, {"qtaim": 5.0, "charge_engine": 2.0}))

    def test_engine_timing_covers_level1_routines(self, tmp_path):
        from qtaim_gen.source.utils.validation import validate_timing_dict

        timings = {**self.BASE, "charge_engine": 2.0, "surface_engine": 1.0,
                   "chelpg": 1.0, "ibsi_bond": 1.0, "elf_fuzzy": 1.0}
        path = self._write(tmp_path, timings)
        assert validate_timing_dict(path, full_set=1, charge_engine=True)
        # chelpg, ibsi_bond and elf_fuzzy still come from Multiwfn
        del timings["chelpg"]
        assert not validate_timing_dict(self._write(tmp_path, timings), full_set=1, charge_engine=True)

    def test_engine_mode_rejects_multiwfn_only(self, tmp_path):
        path = self._write(tmp_path, {**self.BASE, **self.MWFN})
        assert not self._validate(path, charge_engine=True)

    def test_zero_engine_timing_does_not_count(self, tmp_path):
        path = self._write(tmp_path, {**self.BASE, "charge_engine": 0.0})
        assert not self._validate(path)


@pytest.mark.skipif(not HAS_NUMBA, reason="numba not installed")
@pytest.mark.parametrize("name", _fixture_names())
def test_engine_jsons_compile_like_multiwfn(tmp_path, name):
    """gbw_analysis's charge-engine step: per-step jsons from the engine compile
    through parse_multiwfn into the same charge/bond/fuzzy_full content Multiwfn
    produces, and a leftover Multiwfn .out for an engine routine is not parsed."""
    import logging
    import shutil

    from qtaim_gen.source.core.omol import _run_charge_engine, parse_multiwfn
    from qtaim_gen.source.data.multiwfn import ENGINE_ROUTINES

    src = os.path.join(FIXTURES, name)
    for f in ("orca.wfx", "orca.inp"):
        shutil.copy(os.path.join(src, f), tmp_path / f)
    planted = os.path.join(src, "multiwfn_hirshfeld.out")
    if os.path.isfile(planted):
        # altered so a parse of it would be visible in charge.json
        text = open(planted).read().replace("0.14026913", "9.99999999")
        (tmp_path / "hirshfeld.out").write_text(text)

    logger = logging.getLogger("test_charge_engine")
    assert _run_charge_engine(str(tmp_path), n_threads=2, logger=logger)
    parse_multiwfn(str(tmp_path), separate=True, logger=logger, skip_routines=ENGINE_ROUTINES)

    with open(os.path.join(src, "multiwfn_reference.json")) as f:
        ref = json.load(f)
    charge = json.loads((tmp_path / "charge.json").read_text())
    bond = json.loads((tmp_path / "bond.json").read_text())
    fuzzy = json.loads((tmp_path / "fuzzy_full.json").read_text())
    timings = json.loads((tmp_path / "timings.json").read_text())

    assert timings["charge_engine"] > 0
    for step in ("hirshfeld", "adch", "cm5", "becke"):
        for k, v in ref[step]["charge"].items():
            assert charge[step]["charge"][k] == pytest.approx(v, abs=CHARGE_TOL), (step, k)
    assert set(bond["fuzzy_bond"]) == set(ref["fuzzy_bond"])
    for step in (s for s in ref if "fuzzy" in s and s != "fuzzy_bond"):
        for k, v in ref[step][step].items():
            assert fuzzy[step][k] == pytest.approx(v, abs=CHARGE_TOL), (step, k)
    # per-step intermediates are removed once compiled
    assert not any((tmp_path / f"{r}.json").exists() for r in ENGINE_ROUTINES)


@pytest.mark.skipif(not HAS_NUMBA, reason="numba not installed")
@pytest.mark.parametrize("name", _fixture_names())
def test_engine_jsons_compile_like_multiwfn_level1_and_alie(tmp_path, name):
    """full_set 1: vdd/mbis/mbis_fuzzy_* from the charge engine and other_alie
    from the surface engine compile into charge.json, fuzzy_full.json and
    other.json with the values Multiwfn produces."""
    import logging
    import shutil

    from qtaim_gen.source.core.omol import _run_charge_engine, _run_surface_engine, parse_multiwfn
    from qtaim_gen.source.data.multiwfn import ENGINE_ROUTINES

    src = os.path.join(FIXTURES, name)
    for f in ("orca.wfx", "orca.inp"):
        shutil.copy(os.path.join(src, f), tmp_path / f)
    logger = logging.getLogger("test_charge_engine")
    assert _run_charge_engine(str(tmp_path), n_threads=2, logger=logger, full_set=1)
    assert _run_surface_engine(str(tmp_path), n_threads=2, logger=logger)
    parse_multiwfn(str(tmp_path), separate=True, logger=logger, full_set=1, skip_routines=ENGINE_ROUTINES)

    with open(os.path.join(src, "multiwfn_reference_level1.json")) as f:
        ref = json.load(f)
    with open(os.path.join(src, "multiwfn_reference_alie.json")) as f:
        ref_alie = json.load(f)["other_alie"]
    charge = json.loads((tmp_path / "charge.json").read_text())
    fuzzy = json.loads((tmp_path / "fuzzy_full.json").read_text())
    other = json.loads((tmp_path / "other.json").read_text())
    timings = json.loads((tmp_path / "timings.json").read_text())

    assert timings["charge_engine"] > 0 and timings["surface_engine"] > 0
    for step in ("vdd", "mbis"):
        for k, v in ref[step]["charge"].items():
            assert charge[step]["charge"][k] == pytest.approx(v, abs=CHARGE_TOL), (step, k)
    for step in (s for s in ref if "fuzzy" in s):
        for k, v in ref[step][step].items():
            assert fuzzy[step][k] == pytest.approx(v, abs=CHARGE_TOL), (step, k)
    assert other["ALIE_Volume"] == pytest.approx(ref_alie["ALIE_Volume"], abs=1e-5)
    assert other["ALIE_Overall_skewness"] == pytest.approx(ref_alie["ALIE_Overall_skewness"], abs=1e-9)
