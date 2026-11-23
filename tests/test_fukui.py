"""Condensed Fukui functions in the charge engine (charge_engine.fukui_values, run(molden_path=)).

The fixtures' orca_window.molden.input hold the orbitals within FUKUI_WINDOW of the frontier
of the orca_2mkl molden (ORCA 6.0.1) of each orca.wfx; the engine keeps exactly these, so the
results equal those from the full molden. multiwfn_reference_fukui.json holds Multiwfn's
Hirshfeld compositions of these orbitals (menu 8 option 8) and, for the closed shell, CDFT
option 6, both on the engine's Fukui grid: 75 x 434 with every radial point (radcut=0), as
CDFT option 6 integrates (scripts/fukui_reference/campaign.py).
"""

import importlib.util
import json
import logging
import os
import shutil

import numpy as np
import pytest

from qtaim_gen.source.data.multiwfn import FUKUI_ROUTINES

FIXTURES = os.path.join(os.path.dirname(__file__), "test_files", "charge_engine")
HAS_NUMBA = importlib.util.find_spec("numba") is not None
NAMES = ("XeCl4_step2_0_1", "rmechdb_652_step10_0_2")
# Multiwfn prints compositions as f11.3 % and CDFT option 6 as f15.5
PRINT_TOL = 5e-6 + 1e-9
# run() against the same integrals over every window orbital from the molden: FUKUI_DROP and
# the wfx's occupied orbitals (E20.12 text) instead of the molden's
PATH_TOL = 1e-9
LOG = logging.getLogger("test_fukui")


def _paths(name):
    return os.path.join(FIXTURES, name, "orca.wfx"), os.path.join(FIXTURES, name, "orca_window.molden.input")


def _compositions(name, fw, cut=None):
    """Per-orbital Hirshfeld compositions on the full radial grid, the integrals run() takes."""
    from qtaim_gen.source.core import charge_engine as ce

    w = ce.read_wfx(_paths(name)[0])
    fb = ce.prepare_basis(fw, keep=np.ones(len(fw["occ"]), dtype=bool))
    pro, _ = ce.mwfn_proatom_tables(w["atnums"])
    gp, gw = ce.multiwfn_atom_grid(75, 434, np.inf)
    comp = np.zeros((len(w["atnums"]), len(fw["occ"])))
    for b in range(len(w["atnums"])):
        pts = gp + w["coords"][b]
        h, _, _ = ce._hirshfeld_mwfn(pts, w["coords"], *pro)
        hw = h[:, b] * gw
        if cut is not None:
            hw = np.where(hw < cut, 0.0, hw)
        comp[b] = hw @ ce.orbital_values(pts, w["coords"], fb) ** 2
    return comp


def _values(out, labels, key):
    return np.array([out[key][key][lab] for lab in labels])


# ---------------------------------------------------------------- reader window


def test_frontier_window_selects_each_side_from_its_own_frontier():
    from qtaim_gen.source.core.wavefunction import frontier_window

    # (energy, occupation): a 3 Hartree gap, so no orbital lies within the window of mu
    headers = [(-5.0, 2.0), (-3.3, 2.0), (-3.0, 2.0), (0.01, 0.0), (0.5, 0.0), (0.7, 0.0)]
    assert frontier_window(headers, 0.6) == {1, 2, 3, 4}
    # unrestricted, alpha then beta in file order
    headers = [(-0.5, 1.0), (-0.3, 1.0), (0.1, 0.0), (-0.45, 1.0), (-0.05, 0.0), (1.0, 0.0)]
    assert frontier_window(headers, 0.3) == {0, 1, 2, 3, 4}
    # ORCA's placeholders for removed linearly dependent functions: unoccupied, energy exactly 0,
    # written after the real orbitals; with a LUMO above 0 they must not become the frontier
    headers = [(-0.4, 2.0), (-0.27, 2.0), (0.003, 0.0), (0.2, 0.0), (0.9, 0.0), (0.0, 0.0), (0.0, 0.0)]
    assert frontier_window(headers, 0.6) == {0, 1, 2, 3}


def test_window_read_keeps_the_selected_orbitals_unchanged():
    from qtaim_gen.source.core.wavefunction import frontier_window, molden_mo_headers, read_orca_molden

    path = _paths("rmechdb_652_step10_0_2")[1]
    full = read_orca_molden(path, keep_virtuals=True)
    win = read_orca_molden(path, orbital_window=0.2)
    idx = sorted(frontier_window(molden_mo_headers(path), 0.2))
    assert 0 < len(idx) < len(full["occ"])
    assert np.array_equal(win["coeffs"], full["coeffs"][idx])
    assert np.array_equal(win["mo_energy"], full["mo_energy"][idx])
    assert win["spin_types"] == [full["spin_types"][i] for i in idx]
    assert win["mult"] == full["mult"] == 2


# ---------------------------------------------------------------- fukui_values


def _synthetic(seed=0, nat=4, nocc=5, nvirt=4):
    rng = np.random.default_rng(seed)
    comp = rng.random((nat, nocc + nvirt))
    energy = np.sort(np.concatenate([rng.uniform(-1.0, -0.3, nocc), rng.uniform(-0.1, 0.5, nvirt)]))
    occ = np.array([2.0] * nocc + [0.0] * nvirt)
    return comp, energy, occ


def test_closed_shell_equals_its_spin_orbital_expansion():
    """A restricted closed shell gives the values of the same orbitals listed once per spin."""
    from qtaim_gen.source.core.charge_engine import fukui_values

    comp, energy, occ = _synthetic()
    restricted, meta_r = fukui_values(comp, energy, occ, ["Alpha and Beta"] * len(occ))
    spin = ["Alpha"] * len(occ) + ["Beta"] * len(occ)
    expanded, meta_u = fukui_values(np.hstack([comp, comp]), np.concatenate([energy, energy]),
                                    np.concatenate([occ / 2, occ / 2]), spin)
    for k in restricted:
        np.testing.assert_allclose(expanded[k], restricted[k], rtol=1e-14, atol=0)
    assert meta_r["minus_fmo_spin_orbitals"] == meta_u["minus_fmo_spin_orbitals"] == 2


def test_closed_shell_orbital_weights_are_multiwfns():
    """For a closed shell the orbital-weighted values are Multiwfn's formula (CDFT.f90:758-866)."""
    from qtaim_gen.source.core.charge_engine import fukui_values

    comp, energy, occ = _synthetic(1)
    out, _ = fukui_values(comp, energy, occ, ["Alpha and Beta"] * len(occ))
    homo = int(np.flatnonzero(occ > 0).max())
    mu = (energy[homo] + energy[homo + 1]) / 2
    x = np.exp(-(((mu - energy) / 0.1) ** 2))
    np.testing.assert_allclose(out["plus_ow"], comp[:, homo + 1:] @ x[homo + 1:] / x[homo + 1:].sum(), rtol=1e-12)
    np.testing.assert_allclose(out["minus_ow"], comp[:, :homo + 1] @ x[:homo + 1] / x[:homo + 1].sum(), rtol=1e-12)
    np.testing.assert_array_equal(out["plus_fmo"], comp[:, homo + 1])
    np.testing.assert_array_equal(out["minus_fmo"], comp[:, homo])


def test_open_shell_frontier_is_taken_over_both_spins():
    """f- comes from the highest occupied spin-orbital and f+ from the lowest unoccupied,
    whichever spin they have; orbitals within the tolerance are averaged."""
    from qtaim_gen.source.core.charge_engine import FUKUI_DEGEN_TOL, fukui_values

    comp = np.eye(6)[:3] + 0.1  # 3 atoms, 6 spin-orbitals with distinct compositions
    energy = np.array([-0.6, -0.30, 0.20, -0.5, -0.30 + FUKUI_DEGEN_TOL / 2, 0.05])
    occ = np.array([1.0, 1.0, 0.0, 1.0, 1.0, 0.0])
    spin = ["Alpha"] * 3 + ["Beta"] * 3
    out, meta = fukui_values(comp, energy, occ, spin)
    np.testing.assert_allclose(out["minus_fmo"], (comp[:, 1] + comp[:, 4]) / 2)  # alpha and beta HOMO, degenerate
    np.testing.assert_array_equal(out["plus_fmo"], comp[:, 5])  # beta LUMO is lower
    assert meta["minus_fmo_spin_orbitals"] == 2 and meta["plus_fmo_spin_orbitals"] == 1
    assert meta["e_lumo"] == 0.05


def test_large_gap_does_not_underflow():
    """Weights relative to the frontier orbital: a 6 Hartree gap (exp(-900) underflows) still
    gives the frontier orbitals."""
    from qtaim_gen.source.core.charge_engine import fukui_values

    comp = np.array([[0.2, 0.7, 0.4], [0.8, 0.3, 0.6]])
    out, _ = fukui_values(comp, np.array([-8.0, -6.0, 0.0]), np.array([2.0, 2.0, 0.0]), ["Alpha and Beta"] * 3)
    assert np.isfinite(out["minus_ow"]).all() and np.isfinite(out["plus_ow"]).all()
    np.testing.assert_allclose(out["minus_ow"], comp[:, 1])
    np.testing.assert_allclose(out["plus_ow"], comp[:, 2])


# ---------------------------------------------------------------- against Multiwfn


@pytest.mark.skipif(not HAS_NUMBA, reason="numba not installed")
@pytest.mark.parametrize("name", NAMES)
def test_compositions_match_multiwfn(name):
    """Every window orbital's Hirshfeld composition, with Multiwfn's 1e-9 weight cut,
    against menu 8 option 8 on the same grid; and fukui_values on the compositions of every
    window orbital without the cut is run()'s output."""
    from qtaim_gen.source.core import charge_engine as ce
    from qtaim_gen.source.core.wavefunction import read_orca_molden

    wfx, molden = _paths(name)
    ref = json.load(open(os.path.join(FIXTURES, name, "multiwfn_reference_fukui.json")))
    fw = read_orca_molden(molden, orbital_window=ce.FUKUI_WINDOW)
    comp = _compositions(name, fw, cut=1e-9)
    R = np.array(ref["orbcomp_window"]).T
    assert R.shape == comp.shape
    assert np.abs(comp / comp.sum(0) - R).max() <= PRINT_TOL

    out = ce.run(wfx, molden_path=molden)
    labels = ce.atom_labels(ce.read_wfx(wfx)["atnums"])
    expect, _ = ce.fukui_values(_compositions(name, fw), fw["mo_energy"], fw["occ"], fw["spin_types"])
    for k, v in expect.items():
        assert np.abs(_values(out, labels, f"fukui_{k}") - v).max() <= PATH_TOL, k


@pytest.mark.skipif(not HAS_NUMBA, reason="numba not installed")
def test_orbital_weighted_matches_multiwfn_cdft():
    """Closed shell: run()'s orbital-weighted f+ and f- against Multiwfn's CDFT option 6."""
    from qtaim_gen.source.core import charge_engine as ce

    wfx, molden = _paths("XeCl4_step2_0_1")
    ref = np.array(json.load(open(os.path.join(FIXTURES, "XeCl4_step2_0_1", "multiwfn_reference_fukui.json")))["ow"])
    out = ce.run(wfx, molden_path=molden)
    labels = ce.atom_labels(ce.read_wfx(wfx)["atnums"])
    assert np.abs(_values(out, labels, "fukui_plus_ow") - ref[:, 0]).max() <= PRINT_TOL
    assert np.abs(_values(out, labels, "fukui_minus_ow") - ref[:, 1]).max() <= PRINT_TOL
    meta = out["_meta"]["fukui"]
    # XeCl4's LUMO is a degenerate pair: 4 spin-orbitals in f+
    assert meta["plus_fmo_spin_orbitals"] == 4 and meta["minus_fmo_spin_orbitals"] == 2
    assert set(FUKUI_ROUTINES) <= set(out)


@pytest.mark.skipif(not HAS_NUMBA, reason="numba not installed")
def test_frontier_values_do_not_depend_on_rotations_within_a_degenerate_set():
    """Rotating XeCl4's degenerate LUMO pair changes each orbital's composition but not f+."""
    from qtaim_gen.source.core import charge_engine as ce
    from qtaim_gen.source.core.wavefunction import read_orca_molden

    fw = read_orca_molden(_paths("XeCl4_step2_0_1")[1], orbital_window=ce.FUKUI_WINDOW)
    virt = np.flatnonzero(fw["occ"] == 0)
    pair = virt[np.argsort(fw["mo_energy"][virt])[:2]]
    assert abs(fw["mo_energy"][pair[1]] - fw["mo_energy"][pair[0]]) <= ce.FUKUI_DEGEN_TOL
    base = _compositions("XeCl4_step2_0_1", fw)
    c, s = np.cos(0.5), np.sin(0.5)
    rot = dict(fw, coeffs=fw["coeffs"].copy())
    rot["coeffs"][pair] = np.array([[c, s], [-s, c]]) @ fw["coeffs"][pair]
    turned = _compositions("XeCl4_step2_0_1", rot)
    assert np.abs(turned[:, pair] - base[:, pair]).max() > 1e-3
    a, _ = ce.fukui_values(base, fw["mo_energy"], fw["occ"], fw["spin_types"])
    b, _ = ce.fukui_values(turned, fw["mo_energy"], fw["occ"], fw["spin_types"])
    np.testing.assert_allclose(b["plus_fmo"], a["plus_fmo"], atol=1e-12)
    np.testing.assert_allclose(b["plus_ow"], a["plus_ow"], atol=1e-12)


# ---------------------------------------------------------------- skips and runner


@pytest.mark.skipif(not HAS_NUMBA, reason="numba not installed")
def test_skips_without_molden_or_with_another_structures():
    from qtaim_gen.source.core import charge_engine as ce

    wfx, _ = _paths("XeCl4_step2_0_1")
    for molden, reason in ((None, "no_molden"), (_paths("rmechdb_652_step10_0_2")[1], "molden_mismatch")):
        out = ce.run(wfx, molden_path=molden)
        assert not FUKUI_ROUTINES & set(out)
        assert {"schemes": "fukui", "reason": reason, "routines": sorted(FUKUI_ROUTINES)} in out["_meta"]["skipped"]
        assert out["_meta"]["fukui"] is None


def test_runner_passes_the_molden_and_records_a_missing_one(tmp_path, monkeypatch):
    from qtaim_gen.source.core import charge_engine, omol
    from qtaim_gen.source.utils import provenance

    monkeypatch.setattr(omol, "check_spin", lambda folder: False)
    wanted = omol._engine_routines(0, False)
    seen = {}

    def fake_run(path, full_set=0, molden_path=None, **kw):
        seen["molden"] = molden_path
        out = {r: {r: {"1_H": 1.0}} for r in wanted}
        if molden_path is None:
            for r in FUKUI_ROUTINES:
                out.pop(r)
            out["_meta"] = {"skipped": [{"schemes": "fukui", "reason": "no_molden", "routines": sorted(FUKUI_ROUTINES)}]}
        else:
            out["_meta"] = {"skipped": [], "fukui": {"delta": 0.1, "degeneracy_tol": 3.7e-4}}
        return out

    monkeypatch.setattr(charge_engine, "run", fake_run)
    with_molden = tmp_path / "a"
    with_molden.mkdir()
    (with_molden / "orca.wfx").write_text("wfx")
    (with_molden / "orca.molden.input").write_text("molden")
    assert omol._run_charge_engine(str(with_molden), n_threads=1, logger=LOG)
    assert seen["molden"] == str(with_molden / "orca.molden.input")
    prov = provenance.load(str(with_molden))
    assert FUKUI_ROUTINES <= provenance.engine_routines(prov)
    assert prov["routines"]["fukui_plus_ow"]["spec"]["fukui"] == {"delta": 0.1, "degeneracy_tol": 3.7e-4}

    without = tmp_path / "b"
    without.mkdir()
    (without / "orca.wfx").write_text("wfx")
    assert omol._run_charge_engine(str(without), n_threads=1, logger=LOG)
    assert seen["molden"] is None
    prov = provenance.load(str(without))
    assert prov["skipped"] == {r: {"reason": "no_molden", "by": "charge_engine"} for r in FUKUI_ROUTINES}


def test_engine_validation_expects_the_fukui_keys_unless_skipped(tmp_path):
    from qtaim_gen.source.utils.validation import validate_fuzzy_dict

    path = tmp_path / "fuzzy_full.json"
    entry = {"1_H": 0.5, "2_H": 0.5, "sum": 1.0, "abs_sum": 1.0}
    path.write_text(json.dumps({"becke_fuzzy_density": entry, "hirsh_fuzzy_density": entry}))
    assert validate_fuzzy_dict(str(path), n_atoms=2)
    assert not validate_fuzzy_dict(str(path), n_atoms=2, charge_engine=True)
    assert validate_fuzzy_dict(str(path), n_atoms=2, charge_engine=True, skipped=set(FUKUI_ROUTINES))
    path.write_text(json.dumps({"becke_fuzzy_density": entry, "hirsh_fuzzy_density": entry,
                                **{r: entry for r in FUKUI_ROUTINES}}))
    assert validate_fuzzy_dict(str(path), n_atoms=2, charge_engine=True)


@pytest.mark.skipif(not HAS_NUMBA, reason="numba not installed")
def test_fukui_jsons_compile_into_fuzzy_full(tmp_path):
    """The engine's per-step Fukui jsons compile into fuzzy_full.json and are removed."""
    from qtaim_gen.source.core.omol import _run_charge_engine, parse_multiwfn
    from qtaim_gen.source.data.multiwfn import ENGINE_ROUTINES

    src = os.path.join(FIXTURES, "rmechdb_652_step10_0_2")
    for f in ("orca.wfx", "orca.inp"):
        shutil.copy(os.path.join(src, f), tmp_path / f)
    shutil.copy(os.path.join(src, "orca_window.molden.input"), tmp_path / "orca.molden.input")
    assert _run_charge_engine(str(tmp_path), n_threads=2, logger=LOG)
    parse_multiwfn(str(tmp_path), separate=True, logger=LOG, skip_routines=ENGINE_ROUTINES)
    from qtaim_gen.source.core.charge_engine import read_wfx

    nat = len(read_wfx(str(tmp_path / "orca.wfx"))["atnums"])
    fuzzy = json.loads((tmp_path / "fuzzy_full.json").read_text())
    for r in FUKUI_ROUTINES:
        assert len(fuzzy[r]) == nat + 2  # per atom plus sum and abs_sum
        assert fuzzy[r]["sum"] == pytest.approx(1.0, abs=1e-3)
        assert not (tmp_path / f"{r}.json").exists()


def test_truncation_keeps_the_frontier_and_bounds_the_dropped_share():
    from qtaim_gen.source.core.charge_engine import _ow_weights, fukui_orbitals

    rng = np.random.default_rng(3)
    energy = np.sort(rng.uniform(-3.0, 2.0, 400))
    occ = np.where(energy < -0.2, 2.0, 0.0)
    spin = ["Alpha and Beta"] * len(occ)
    keep = fukui_orbitals(energy, occ, spin, drop=1e-10)
    _, x, _, _ = _ow_weights(energy, occ, spin, 0.1)
    for sel in (occ > 0, occ == 0):
        assert x[sel & ~keep].sum() / x[sel].sum() < 1e-10
        assert keep[np.flatnonzero(sel)[np.argmax(x[sel])]]
    assert keep.sum() < len(occ) / 4
