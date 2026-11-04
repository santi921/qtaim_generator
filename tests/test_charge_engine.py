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
         "--full_set", "1", "--mbis_mode", "multiwfn"],
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
def test_engine_jsons_compile_like_multiwfn_level1_and_alie(tmp_path, name, monkeypatch):
    """full_set 1: vdd/mbis/mbis_fuzzy_* from the charge engine and other_alie
    from the surface engine compile into charge.json, fuzzy_full.json and
    other.json with the values Multiwfn produces."""
    import logging
    import shutil

    from qtaim_gen.source.core import charge_engine
    from qtaim_gen.source.core.omol import _run_charge_engine, _run_surface_engine, parse_multiwfn
    from qtaim_gen.source.data.multiwfn import ENGINE_ROUTINES

    # the references are Multiwfn's MBIS, which stops by its own rule
    monkeypatch.setattr(charge_engine, "MBIS_MODE", "multiwfn")
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


def _mbis_grid_density(wfx_path):
    """The MBIS charge grid's Becke-weighted density, as run() builds it."""
    import numpy as np

    from qtaim_gen.source.core import charge_engine as CE

    wfx = CE.read_wfx(wfx_path)
    basis = CE.prepare_basis(wfx)
    atnums, coords = wfx["atnums"], wfx["coords"]
    rinv, aij = CE.becke_pair_tables(atnums, coords)
    pts0, wts = CE.multiwfn_atom_grid(*CE.mbis_grid_size(atnums), 10.0)
    td = np.zeros((len(atnums), len(wts)))
    for b in range(len(atnums)):
        pts = pts0 + coords[b]
        rho, _ = CE.density(pts, coords, basis, wfx)
        td[b] = CE._becke(pts, coords, rinv, aij, CE.BECKE_TOL)[:, b] * wts * rho
    qbase = atnums.astype(float) if len(wfx["edf_center"]) else wfx["zeff"]
    return pts0, coords, td, atnums, qbase


@pytest.mark.skipif(not HAS_NUMBA, reason="numba not installed")
@pytest.mark.parametrize("name", _fixture_names())
def test_mbis_converged_mode_reaches_the_plain_fixed_point(name):
    """Converged (accelerated) MBIS lands where plain iteration run far past
    Multiwfn's stopping rule does, from the default and the check guess."""
    import numpy as np

    from qtaim_gen.source.core import charge_engine as CE

    args = _mbis_grid_density(os.path.join(FIXTURES, name, "orca.wfx"))
    q_plain = CE.mbis_fit(*args, crit=1e-10, maxcyc=30000, mode="multiwfn")[0]
    q_conv, m, p, s, ncyc = CE.mbis_fit(*args, mode="converged")
    assert np.abs(q_conv - q_plain).max() < 1e-6
    assert not CE.mbis_degenerate(m, p, s)
    q_sel, *_, info = CE.mbis_select(*args)
    assert info["solutions"] == 1 and np.abs(q_sel - q_conv).max() < 1e-6


def test_mbis_degenerate_detects_merged_and_empty_shells():
    import numpy as np

    from qtaim_gen.source.core.charge_engine import mbis_degenerate

    mshell = np.array([2, 3])
    pop = np.array([[1.9, 4.1, 0.0], [2.0, 7.5, 6.5]])
    sig = np.array([[0.07, 0.49, 0.0], [0.03, 0.2, 0.8]])
    assert not mbis_degenerate(mshell, pop, sig)
    merged = sig.copy()
    merged[1, 2] = merged[1, 1] * (1 + 1e-4)
    assert mbis_degenerate(mshell, pop, merged)
    empty = pop.copy()
    empty[0, 0] = 1e-9
    assert mbis_degenerate(mshell, empty, sig)


def test_mbis_fallback_only_when_asked(monkeypatch):
    """A stalled accelerated fit is redone by plain iteration only with fallback;
    either way a fit that never meets the stopping rule reports converged False."""
    import numpy as np

    from qtaim_gen.source.core import charge_engine as CE

    monkeypatch.setattr(CE, "MBIS_ANDERSON_MAXCYC", 5)
    monkeypatch.setattr(CE, "MBIS_MAXCYC", 7)
    calls = []

    def drifting(m, p, s):
        calls.append(1)
        return p * 1.01, s

    mshell, pop, sig = np.array([1]), np.array([[1.0]]), np.array([[0.5]])
    (_, _, _, _, n), conv, fell = CE._mbis_fit_converged(drifting, mshell, pop, sig, np.array([1.0]), fallback=False)
    assert (len(calls), n, conv, fell) == (5, 5, False, False)
    calls.clear()
    (_, _, _, _, n), conv, fell = CE._mbis_fit_converged(drifting, mshell, pop, sig, np.array([1.0]))
    assert (len(calls), n, conv, fell) == (12, 12, False, True)
    calls.clear()
    (_, _, _, _, n), conv, fell = CE._mbis_fit_converged(lambda m, p, s: (p, s), mshell, pop, sig, np.array([1.0]))
    assert (n, conv, fell) == (1, True, False)


def test_mbis_merge_shells_combines_equal_widths():
    import numpy as np

    from qtaim_gen.source.core.charge_engine import mbis_merge_shells

    mshell = np.array([3, 2])
    pop = np.array([[2.0, 8.0, 0.5], [1.0, 3.0, 0.0]])
    sig = np.array([[0.02, 0.400, 0.402], [0.05, 0.5, 0.0]])
    m, p, s = mbis_merge_shells(mshell, pop, sig)
    assert m.tolist() == [2, 2]
    assert np.allclose(p[0], [2.0, 8.5, 0.0]) and np.allclose(p.sum(axis=1), pop.sum(axis=1))
    assert np.isclose(s[0, 1], (8.0 * 0.400 + 0.5 * 0.402) / 8.5) and s[0, 2] == 0.0
    assert np.array_equal(p[1], pop[1]) and np.array_equal(s[1], sig[1])


def test_mbis_fallback_merges_a_stalled_shell_pair(monkeypatch):
    """Plain iteration that moves population between two shells of equal width
    without converging: the merged one-shell model is kept when its objective is
    not worse, and plain iteration runs on to MBIS_MAXCYC when it is."""
    import numpy as np

    from qtaim_gen.source.core import charge_engine as CE

    monkeypatch.setattr(CE, "MBIS_ANDERSON_MAXCYC", 5)
    monkeypatch.setattr(CE, "MBIS_MAXCYC", 7)
    monkeypatch.setattr(CE, "MBIS_MERGE_FIRST", 3)
    monkeypatch.setattr(CE, "MBIS_MERGE_EVERY", 2)

    def step(m, p, s):  # two shells: population drifts from the second to the first; one shell: fixed
        return (p * np.array([[1.001, 0.999]]), s) if m[0] == 2 else (p, s)

    mshell, pop, sig = np.array([2]), np.array([[1.0, 1.0]]), np.array([[0.5, 0.501]])
    (q, m, p, s, n), conv, fell = CE._mbis_fit_converged(step, mshell, pop, sig, np.array([2.0]),
                                                         objective=lambda mm, pp, ss: -float(mm[0]))
    assert (m.tolist(), conv, fell, n) == ([1], True, True, 5 + 3 + 1)
    assert np.isclose(p[0, 0], 1.001**3 + 0.999**3) and q[0] == 2.0 - p[0, 0]

    (q, m, p, s, n), conv, fell = CE._mbis_fit_converged(step, mshell, pop, sig, np.array([2.0]),
                                                         objective=lambda mm, pp, ss: float(mm[0]))
    assert (m.tolist(), conv, fell, n) == ([2], False, True, 5 + 7 + 3)


def test_mbis_chunked_fallback_matches_plain_without_merges(monkeypatch):
    """With nothing to merge, the chunked plain fallback returns exactly what
    one uninterrupted plain run does."""
    import numpy as np

    from qtaim_gen.source.core import charge_engine as CE

    monkeypatch.setattr(CE, "MBIS_ANDERSON_MAXCYC", 1)
    monkeypatch.setattr(CE, "MBIS_MERGE_FIRST", 3)
    monkeypatch.setattr(CE, "MBIS_MERGE_EVERY", 2)
    target = np.array([[1.5, 0.5]])

    def step(m, p, s):
        return target + 0.6 * (p - target), s

    mshell, pop, sig = np.array([2]), np.array([[1.0, 1.0]]), np.array([[0.1, 0.5]])
    plain = CE._mbis_fit_converged(step, mshell, pop, sig, np.array([2.0]))
    chunked = CE._mbis_fit_converged(step, mshell, pop, sig, np.array([2.0]), objective=lambda mm, pp, ss: 0.0)
    assert plain[1:] == chunked[1:] == (True, True)
    # 1 accelerated evaluation, then plain convergence inside a chunk (chunks end at 3, 5, 7, ...)
    assert plain[0][4] == chunked[0][4] and (plain[0][4] - 1) % 2 == 0
    assert all(np.array_equal(a, b) for a, b in zip(plain[0][:4], chunked[0][:4]))


def test_mbis_select_drops_failed_check_guesses(monkeypatch):
    """A check guess that stalls or ends degenerate cannot be selected, even
    with a better objective, and does not count as another solution."""
    import numpy as np

    from qtaim_gen.source.core import charge_engine as CE

    m = np.array([2, 1])
    p = np.array([[0.5, 0.5], [1.0, 0.0]])
    s = np.array([[0.5, 0.6], [0.5, 0.0]])
    guesses = [("default", (m, p, s)), ("sigma_x0.6", (m, p, s * 0.6)), ("sigma_x1.5", (m, p, s * 1.5))]
    results = {  # (fit, converged, objective, degenerate, fell back); default: converged; x0.6: stalled;
        # x1.5: converged but degenerate
        0.5: ((np.array([0.1, -0.1]), m, p, s, 40), True, 1.0, False, False),
        0.3: ((np.array([0.6, -0.6]), m, p, s * 0.6, 5), False, 9.0, False, False),
        0.75: ((np.array([0.4, -0.4]), m, p, s * 1.5, 30), True, 5.0, True, False),
    }
    seen = []

    def fake_fit(step, mshell, pop, sig, qbase, fallback=True, objective=None):
        seen.append(fallback)
        r = results[round(float(sig[0, 0]), 2)]
        return r[0], r[1], r[4]

    monkeypatch.setattr(CE, "mbis_guesses", lambda atnums: guesses)
    monkeypatch.setattr(CE, "_mbis_setup", lambda *a: (None, None, None))
    monkeypatch.setattr(CE, "_mbis_fit_converged", fake_fit)
    monkeypatch.setattr(CE, "mbis_objective", lambda g, c, t, a, mm, pp, ss: results[round(float(ss[0, 0]), 2)][2])
    monkeypatch.setattr(CE, "mbis_degenerate", lambda mm, pp, ss: results[round(float(ss[0, 0]), 2)][3])
    q, *_, ncyc, info = CE.mbis_select(None, None, None, np.array([1, 1]), np.array([1.0, 1.0]))
    assert seen == [True, False, False]
    assert info["selected"] == "default" and np.allclose(q, [0.1, -0.1])
    assert info["converged"] == {"default": True, "sigma_x0.6": False, "sigma_x1.5": True}
    assert info["solutions"] == 1 and info["spread"] == 0.0
    assert info["merged"] == {"default": [], "sigma_x0.6": [], "sigma_x1.5": []}
    assert ncyc == 75

    # default fit degenerate: the check guesses get the fallback, and a converged,
    # non-degenerate check fit wins over it
    seen.clear()
    results[0.5] = ((np.array([0.1, -0.1]), m, p, s, 21000), False, 1.0, True, True)
    results[0.75] = ((np.array([0.4, -0.4]), m, p, s * 1.5, 1519), True, 5.0, False, True)
    q, *_, info = CE.mbis_select(None, None, None, np.array([1, 1]), np.array([1.0, 1.0]))
    assert seen == [True, True, True]
    assert info["selected"] == "sigma_x1.5" and np.allclose(q, [0.4, -0.4])
    assert info["solutions"] == 1

    # default fit converged only after the fallback merged a shell pair of atom 0: the check
    # guesses still get the fallback (rmechdb_750), and the merge is recorded
    seen.clear()
    results[0.5] = ((np.array([0.1, -0.1]), np.array([1, 1]), p, s, 2010), True, 1.0, False, True)
    q, *_, info = CE.mbis_select(None, None, None, np.array([1, 1]), np.array([1.0, 1.0]))
    assert seen == [True, True, True]
    assert info["merged"]["default"] == [[0, 2, 1]] and info["selected"] == "sigma_x1.5"


def test_mbis_guess_beyond_rn_keeps_six_shells():
    """Period 7 uses Multiwfn's Rn rule: 6 shells, valence Z - 54."""
    import numpy as np

    from qtaim_gen.source.core.charge_engine import MBIS_MAX_Z, mbis_initial_shells

    atnums = np.array([86, 90, 92, 94, 103])
    mshell, pop, sig = mbis_initial_shells(atnums)
    assert (mshell == 6).all()
    assert np.allclose(pop.sum(axis=1), atnums)
    assert np.allclose(pop[:, 5], atnums - 54)
    assert np.allclose(sig[:, 0], 1 / (2 * atnums)) and np.allclose(sig[:, 5], 0.5)
    assert MBIS_MAX_Z >= 103


@pytest.mark.skipif(not HAS_NUMBA, reason="numba not installed")
def test_run_records_mbis_mode_and_uniqueness():
    from qtaim_gen.source.core.charge_engine import run

    out = run(os.path.join(FIXTURES, "XeCl4_step2_0_1", "orca.wfx"), full_set=1)
    meta = out["_meta"]
    assert meta["mbis_mode"] == "converged"
    assert meta["mbis_selection"]["selected"] == "default"
    assert set(meta["mbis_selection"]["objective"]) == {"default", "sigma_x0.6", "sigma_x1.5"}
    assert meta["mbis_selection"]["converged"] == {"default": True, "sigma_x0.6": True, "sigma_x1.5": True}
    assert meta["mbis_unique"] is True
    assert meta["mbis_degenerate"] == {"charge": False, "fuzzy": False}
    assert meta["mbis_merged"] == {"charge": [], "fuzzy": []}
    assert meta["mbis_beyond_multiwfn"] is False


def test_mbis_fitted_guess_covers_the_proatom_range():
    import numpy as np

    from qtaim_gen.source.core.charge_engine import mbis_fitted_shells
    from qtaim_gen.source.data.mbis_free_atom import SHELLS

    assert sorted(SHELLS) == list(range(1, 104))
    atnums = np.array([1, 8, 26, 54, 92, 103])
    mshell, pop, sig = mbis_fitted_shells(atnums)
    assert np.allclose(pop.sum(axis=1), atnums, atol=0.05)
    for a in range(len(atnums)):
        widths = sig[a, : mshell[a]]
        assert (widths > 0).all() and (np.diff(widths) > 0).all()


@pytest.mark.skipif(not HAS_NUMBA, reason="numba not installed")
def test_mbis_ecp_switch():
    """XeCl4 has an ECP on Xe with EDF: MBIS runs by default and is skipped,
    with its reason recorded, under mbis_ecp=False."""
    from qtaim_gen.source.core.charge_engine import run

    wfx = os.path.join(FIXTURES, "XeCl4_step2_0_1", "orca.wfx")
    on = run(wfx, full_set=1)
    off = run(wfx, full_set=1, mbis_ecp=False)
    assert on["_meta"]["has_ecp"] and "mbis" in on
    assert "mbis" not in off and "mbis_fuzzy_density" not in off
    assert {"schemes": "mbis family", "reason": "ecp_disabled",
            "routines": ["mbis", "mbis_fuzzy_density", "mbis_fuzzy_spin"]} in off["_meta"]["skipped"]
