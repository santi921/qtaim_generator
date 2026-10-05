"""recheck_fuzzy: value checks, reparse from archived output, invalidation, wavefunction guard."""

import json
import os
import zipfile
from pathlib import Path

import pytest

from qtaim_gen.source.utils.fuzzy_recheck import (
    fuzzy_value_failures,
    hirsh_density_from_charges,
    out_electron_counts,
    recheck_fuzzy,
)

MWFN = Path(__file__).parent / "test_files" / "multiwfn"
HIRSH_TWO_BLOCK = (MWFN / "hirsh_two_block" / "hirsh_fuzzy_density.out").read_text()
FUZZY_BOND_WFX = (MWFN / "open_shell_wfx" / "fuzzy_bond.out").read_text()
# same output as if Multiwfn had read the wavefunction as all-alpha (.wfn era)
FUZZY_BOND_ALL_ALPHA = FUZZY_BOND_WFX.replace(
    "Total/Alpha/Beta electrons:     95.0000     48.0000     47.0000",
    "Total/Alpha/Beta electrons:     95.0000     95.0000      0.0000",
)

ATOMS = ["1_H", "2_Br", "3_H", "4_C", "5_C", "6_C", "7_Br", "8_H", "9_H", "10_H", "11_H", "12_H"]


def _table(values, total=None):
    d = dict(zip(ATOMS, values))
    s = sum(values) if total is None else total
    d.update({"sum": s, "abs_sum": abs(s)})
    return d


def _job(tmp_path, fuzzy, bond=None, outs=None, loose=()):
    """Completed-job layout: compiled JSONs + out_files.zip in generator/."""
    gen = tmp_path / "generator"
    gen.mkdir()
    (gen / "fuzzy_full.json").write_text(json.dumps(fuzzy))
    (gen / "bond.json").write_text(json.dumps(bond or {"mayer_orca": {"4_C_to_5_C": 1.0}}))
    with zipfile.ZipFile(gen / "out_files.zip", "w") as zf:
        for name, text in (outs or {}).items():
            zf.writestr(name, text)
    for name in loose:
        (tmp_path / name).write_text("stale\n")
    return tmp_path


GOOD_DENSITY = _table([0.94, 35.15, 0.97, 6.01, 6.08, 6.03, 35.04, 0.96, 0.96, 0.96, 0.95, 0.95])
GOOD_SPIN = _table([0.0, 0.3, 0.0, 0.2, 0.1, 0.1, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0])  # sums to 1.0
ZEROS = _table([0.0] * 12)
ALL_ALPHA_SPIN = GOOD_DENSITY  # spin == density: sums to ~95


class TestValueChecks:
    def test_zero_density_and_wrong_spin_flagged(self):
        fz = {"becke_fuzzy_density": GOOD_DENSITY, "hirsh_fuzzy_density": ZEROS,
              "becke_fuzzy_spin": ALL_ALPHA_SPIN, "hirsh_fuzzy_spin": GOOD_SPIN}
        assert sorted(fuzzy_value_failures(fz, mult=2)) == ["becke_fuzzy_spin", "hirsh_fuzzy_density"]

    def test_summary_rows_excluded(self):
        # a stored "sum" of 95 must not count toward the spin total
        spin = dict(GOOD_SPIN, sum=95.0, abs_sum=95.0)
        assert fuzzy_value_failures({"becke_fuzzy_spin": spin}, mult=2) == []

    def test_spin_ignored_for_closed_shell(self):
        assert fuzzy_value_failures({"becke_fuzzy_spin": ALL_ALPHA_SPIN}, mult=1) == []

    def test_nonfinite_flagged(self):
        bad = dict(GOOD_DENSITY, **{"4_C": float("nan")})
        assert fuzzy_value_failures({"becke_fuzzy_density": bad}, mult=1) == ["becke_fuzzy_density"]

    def test_electron_counts(self):
        assert out_electron_counts(FUZZY_BOND_WFX) == (95.0, 48.0, 47.0)
        assert out_electron_counts(FUZZY_BOND_ALL_ALPHA) == (95.0, 95.0, 0.0)
        assert out_electron_counts("no banner") is None


class TestRecheck:
    def test_zero_density_reparsed_from_zip(self, tmp_path):
        job = _job(tmp_path, {"becke_fuzzy_density": GOOD_DENSITY, "hirsh_fuzzy_density": ZEROS},
                   outs={"hirsh_fuzzy_density.out": HIRSH_TWO_BLOCK}, loose=["hirsh_fuzzy_density.json"])
        rep = recheck_fuzzy(str(job), mult=1)
        assert rep == {"reparse": ["hirsh_fuzzy_density"], "rerun": [], "derived": [],
                       "wavefunction": "not needed", "ok": True}
        fz = json.loads((job / "generator" / "fuzzy_full.json").read_text())
        assert fz["hirsh_fuzzy_density"]["sum"] == pytest.approx(94.99993976)
        assert fz["becke_fuzzy_density"] == GOOD_DENSITY
        assert not (job / "hirsh_fuzzy_density.json").exists()

    def test_alpha_only_fuzzy_bond_reparsed_to_total(self, tmp_path):
        bond = {"fuzzy_bond": {"4_C_to_5_C": 0.643844}, "mayer_orca": {"4_C_to_5_C": 1.0}}
        job = _job(tmp_path, {"becke_fuzzy_spin": GOOD_SPIN, "hirsh_fuzzy_spin": GOOD_SPIN},
                   bond=bond, outs={"fuzzy_bond.out": FUZZY_BOND_WFX})
        rep = recheck_fuzzy(str(job), mult=2)
        assert rep["reparse"] == ["fuzzy_bond"] and rep["rerun"] == []
        b = json.loads((job / "generator" / "bond.json").read_text())
        assert b["fuzzy_bond"]["4_C_to_5_C"] == pytest.approx(1.251987)
        assert b["mayer_orca"] == {"4_C_to_5_C": 1.0}

    def test_correct_wfx_fuzzy_bond_left_alone(self, tmp_path):
        from qtaim_gen.source.core.parse_multiwfn import parse_bond_order_fuzzy
        total = parse_bond_order_fuzzy(str(MWFN / "open_shell_wfx" / "fuzzy_bond.out"))
        job = _job(tmp_path, {"becke_fuzzy_spin": GOOD_SPIN, "hirsh_fuzzy_spin": GOOD_SPIN},
                   bond={"fuzzy_bond": total}, outs={"fuzzy_bond.out": FUZZY_BOND_WFX})
        assert recheck_fuzzy(str(job), mult=2)["reparse"] == []

    def test_all_alpha_steps_invalidated_and_wfn_removed(self, tmp_path):
        bond = {"fuzzy_bond": {"4_C_to_5_C": 2.43}, "mayer_orca": {"4_C_to_5_C": 1.0}}
        job = _job(tmp_path, {"becke_fuzzy_density": GOOD_DENSITY, "becke_fuzzy_spin": ALL_ALPHA_SPIN,
                              "hirsh_fuzzy_spin": ALL_ALPHA_SPIN},
                   bond=bond, outs={"fuzzy_bond.out": FUZZY_BOND_ALL_ALPHA},
                   loose=["becke_fuzzy_spin.out", "becke_fuzzy_spin.json", "orca.wfn", "orca.gbw"])
        (job / "generator" / "hirsh_fuzzy_spin.json").write_text("{}")
        rep = recheck_fuzzy(str(job), mult=2)
        assert rep["rerun"] == ["becke_fuzzy_spin", "fuzzy_bond", "hirsh_fuzzy_spin"]
        assert rep["ok"] and "gbw source" in rep["wavefunction"]
        fz = json.loads((job / "generator" / "fuzzy_full.json").read_text())
        assert sorted(fz) == ["becke_fuzzy_density"]
        b = json.loads((job / "generator" / "bond.json").read_text())
        assert "fuzzy_bond" not in b and "mayer_orca" in b
        for gone in ("becke_fuzzy_spin.out", "becke_fuzzy_spin.json", "orca.wfn",
                     "generator/hirsh_fuzzy_spin.json"):
            assert not (job / gone).exists(), gone
        assert (job / "orca.gbw").exists()

    def test_refuses_open_shell_rerun_from_wfn_only(self, tmp_path):
        job = _job(tmp_path, {"becke_fuzzy_spin": ALL_ALPHA_SPIN}, loose=["orca.wfn"])
        before = (job / "generator" / "fuzzy_full.json").read_text()
        rep = recheck_fuzzy(str(job), mult=2)
        assert rep["ok"] is False and rep["rerun"] == ["becke_fuzzy_spin"]
        assert (job / "orca.wfn").exists()
        assert (job / "generator" / "fuzzy_full.json").read_text() == before

    def test_dry_run_writes_nothing(self, tmp_path):
        job = _job(tmp_path, {"hirsh_fuzzy_density": ZEROS, "becke_fuzzy_spin": ALL_ALPHA_SPIN},
                   outs={"hirsh_fuzzy_density.out": HIRSH_TWO_BLOCK}, loose=["orca.wfn", "orca.gbw"])
        snap = {p: p.read_bytes() for p in job.rglob("*") if p.is_file()}
        rep = recheck_fuzzy(str(job), mult=2, dry_run=True)
        assert rep["reparse"] == ["hirsh_fuzzy_density"] and rep["rerun"] == ["becke_fuzzy_spin"]
        assert {p: p.read_bytes() for p in job.rglob("*") if p.is_file()} == snap

    def test_clean_record_untouched(self, tmp_path):
        job = _job(tmp_path, {"becke_fuzzy_density": GOOD_DENSITY, "hirsh_fuzzy_density": GOOD_DENSITY})
        assert recheck_fuzzy(str(job), mult=1) == {
            "reparse": [], "rerun": [], "derived": [], "wavefunction": "not needed", "ok": True}



class TestSingletFuzzyBond:
    """UKS singlets: the fuzzy_bond check used to run only for mult > 1."""

    BANNER = "Total/Alpha/Beta electrons:     95.0000     48.0000     47.0000"

    def _out(self, total, alpha, beta):
        return FUZZY_BOND_WFX.replace(self.BANNER, f"Total/Alpha/Beta electrons: {total:11.4f} {alpha:11.4f} {beta:11.4f}")

    def test_alpha_only_singlet_reparsed_to_total(self, tmp_path):
        bond = {"fuzzy_bond": {"4_C_to_5_C": 0.643844}, "mayer_orca": {"4_C_to_5_C": 1.0}}
        job = _job(tmp_path, {}, bond=bond, outs={"fuzzy_bond.out": self._out(96, 48, 48)})
        rep = recheck_fuzzy(str(job), mult=1)
        assert rep["reparse"] == ["fuzzy_bond"] and rep["rerun"] == [] and rep["ok"]
        b = json.loads((job / "generator" / "bond.json").read_text())
        assert b["fuzzy_bond"]["4_C_to_5_C"] == pytest.approx(1.251987)

    def test_correct_singlet_left_alone(self, tmp_path):
        from qtaim_gen.source.core.parse_multiwfn import parse_bond_order_fuzzy
        total = parse_bond_order_fuzzy(str(MWFN / "open_shell_wfx" / "fuzzy_bond.out"))
        job = _job(tmp_path, {}, bond={"fuzzy_bond": total}, outs={"fuzzy_bond.out": self._out(96, 48, 48)})
        assert recheck_fuzzy(str(job), mult=1) == {
            "reparse": [], "rerun": [], "derived": [], "wavefunction": "not needed", "ok": True}

    def test_all_alpha_singlet_reruns_from_wfx_and_drops_the_wfn(self, tmp_path):
        bond = {"fuzzy_bond": {"4_C_to_5_C": 2.5}, "mayer_orca": {"4_C_to_5_C": 1.0}}
        job = _job(tmp_path, {}, bond=bond, outs={"fuzzy_bond.out": self._out(96, 96, 0)},
                   loose=["orca.wfn", "orca.gbw"])
        rep = recheck_fuzzy(str(job), mult=1)
        assert rep["rerun"] == ["fuzzy_bond"] and rep["ok"] and "gbw source" in rep["wavefunction"]
        assert not (job / "orca.wfn").exists()
        assert "fuzzy_bond" not in json.loads((job / "generator" / "bond.json").read_text())

    def test_all_alpha_singlet_refused_without_a_wfx_source(self, tmp_path):
        bond = {"fuzzy_bond": {"4_C_to_5_C": 2.5}}
        job = _job(tmp_path, {}, bond=bond, outs={"fuzzy_bond.out": self._out(96, 96, 0)}, loose=["orca.wfn"])
        rep = recheck_fuzzy(str(job), mult=1)
        assert rep["ok"] is False and rep["rerun"] == ["fuzzy_bond"]
        assert (job / "orca.wfn").exists()
        assert json.loads((job / "generator" / "bond.json").read_text()) == bond

    def test_singlet_without_archive_is_left_alone(self, tmp_path):
        job = _job(tmp_path, {}, bond={"fuzzy_bond": {"4_C_to_5_C": 2.5}})
        assert recheck_fuzzy(str(job), mult=1) == {
            "reparse": [], "rerun": [], "derived": [], "wavefunction": "not needed", "ok": True}

    def test_open_shell_without_archive_still_reruns(self, tmp_path):
        job = _job(tmp_path, {}, bond={"fuzzy_bond": {"4_C_to_5_C": 2.5}}, loose=["orca.gbw"])
        assert recheck_fuzzy(str(job), mult=2, dry_run=True)["rerun"] == ["fuzzy_bond"]


class TestReviewFixes:
    def test_job_path_with_generator_component(self, tmp_path):
        # e.g. /lus/eagle/projects/generator/...: the merged generator/ copy must win
        root = tmp_path / "projects" / "generator" / "job"
        root.mkdir(parents=True)
        job = _job(root, {"becke_fuzzy_density": GOOD_DENSITY, "hirsh_fuzzy_density": ZEROS},
                   outs={"hirsh_fuzzy_density.out": HIRSH_TWO_BLOCK})
        (job / "fuzzy_full.json").write_text(json.dumps({"becke_fuzzy_density": GOOD_DENSITY}))
        assert recheck_fuzzy(str(job), mult=1, dry_run=True)["reparse"] == ["hirsh_fuzzy_density"]

    def test_zero_beta_open_shell_is_resolved(self, tmp_path):
        from qtaim_gen.source.core.parse_multiwfn import parse_bond_order_fuzzy
        total = parse_bond_order_fuzzy(str(MWFN / "open_shell_wfx" / "fuzzy_bond.out"))
        h_atom = FUZZY_BOND_WFX.replace(
            "Total/Alpha/Beta electrons:     95.0000     48.0000     47.0000",
            "Total/Alpha/Beta electrons:      1.0000      1.0000      0.0000",
        )
        job = _job(tmp_path, {}, bond={"fuzzy_bond": total}, outs={"fuzzy_bond.out": h_atom})
        rep = recheck_fuzzy(str(job), mult=2)
        assert rep["rerun"] == [] and rep["reparse"] == []

    def test_compressed_gbw_counts_only_with_preprocess(self, tmp_path):
        job = _job(tmp_path, {"becke_fuzzy_spin": ALL_ALPHA_SPIN}, loose=["orca.wfn", "orca.gbw.zstd0"])
        before = (job / "generator" / "fuzzy_full.json").read_text()
        assert recheck_fuzzy(str(job), mult=2)["ok"] is False
        assert (job / "orca.wfn").exists()
        assert (job / "generator" / "fuzzy_full.json").read_text() == before
        rep = recheck_fuzzy(str(job), mult=2, preprocess_compressed=True)
        assert rep["ok"] and rep["rerun"] == ["becke_fuzzy_spin"]
        assert not (job / "orca.wfn").exists()

    def test_tar_archive_is_not_a_gbw_source(self, tmp_path):
        job = _job(tmp_path, {"becke_fuzzy_spin": ALL_ALPHA_SPIN}, loose=["orca.wfn", "orca.tar.zst"])
        assert recheck_fuzzy(str(job), mult=2, preprocess_compressed=True)["ok"] is False
        assert (job / "orca.wfn").exists()

    def test_empty_gbw_is_not_a_source(self, tmp_path):
        job = _job(tmp_path, {"becke_fuzzy_spin": ALL_ALPHA_SPIN}, loose=["orca.wfn"])
        (job / "orca.gbw").write_bytes(b"")
        assert recheck_fuzzy(str(job), mult=2)["ok"] is False

    def test_no_source_applies_reparse_but_keeps_rerun_steps(self, tmp_path):
        job = _job(tmp_path, {"hirsh_fuzzy_density": ZEROS, "becke_fuzzy_spin": ALL_ALPHA_SPIN},
                   outs={"hirsh_fuzzy_density.out": HIRSH_TWO_BLOCK}, loose=["becke_fuzzy_spin.out"])
        rep = recheck_fuzzy(str(job), mult=2)
        assert rep["ok"] is False and rep["reparse"] == ["hirsh_fuzzy_density"]
        fz = json.loads((job / "generator" / "fuzzy_full.json").read_text())
        assert fz["hirsh_fuzzy_density"]["sum"] == pytest.approx(94.99993976)
        assert fz["becke_fuzzy_spin"] == ALL_ALPHA_SPIN
        assert (job / "becke_fuzzy_spin.out").exists()

    def test_wfn_mode_refuses_open_shell_spin_rerun(self, tmp_path):
        job = _job(tmp_path, {"becke_fuzzy_spin": ALL_ALPHA_SPIN}, loose=["orca.wfn", "orca.gbw"])
        rep = recheck_fuzzy(str(job), mult=2, wfx=False)
        assert rep["ok"] is False and "--wfn" in rep["wavefunction"]
        assert (job / "orca.wfn").exists()
        assert "becke_fuzzy_spin" in json.loads((job / "generator" / "fuzzy_full.json").read_text())

    def test_legacy_wfn_removed_with_source(self, tmp_path):
        job = _job(tmp_path, {"becke_fuzzy_spin": ALL_ALPHA_SPIN}, loose=["orca5.wfn", "orca.gbw"])
        assert recheck_fuzzy(str(job), mult=2)["ok"]
        assert not (job / "orca5.wfn").exists()

    def test_closed_shell_density_rerun_accepts_wfn(self, tmp_path):
        # density does not depend on spin labels, so a .wfn is fine for it
        job = _job(tmp_path, {"hirsh_fuzzy_density": ZEROS}, loose=["orca.wfn"])
        rep = recheck_fuzzy(str(job), mult=1)
        assert rep["ok"] and rep["rerun"] == ["hirsh_fuzzy_density"]
        assert (job / "orca.wfn").exists()


HIRSH_Q = [0.06, -0.15, 0.03, -0.01, -0.08, -0.03, -0.04, 0.04, 0.04, 0.04, 0.05, 0.05]
Z_ATOMS = [1, 35, 1, 6, 6, 6, 35, 1, 1, 1, 1, 1]


def _write_charges(job, charges):
    gen = job / "generator"
    (gen / "charge.json").write_text(json.dumps(
        {"hirshfeld": {"charge": dict(zip(ATOMS, charges)), "dipole": {"mag": 0.1}}}))


class TestHirshDensityFromCharges:
    def test_zero_density_rebuilt_without_archive(self, tmp_path):
        job = _job(tmp_path, {"becke_fuzzy_density": GOOD_DENSITY, "hirsh_fuzzy_density": ZEROS})
        _write_charges(job, HIRSH_Q)
        rep = recheck_fuzzy(str(job), mult=1)
        assert rep["reparse"] == ["hirsh_fuzzy_density"] and rep["derived"] == ["hirsh_fuzzy_density"]
        assert rep["rerun"] == [] and rep["wavefunction"] == "not needed"
        h = json.loads((job / "generator" / "fuzzy_full.json").read_text())["hirsh_fuzzy_density"]
        assert list(h) == ATOMS + ["sum", "abs_sum"]
        for key, z, q in zip(ATOMS, Z_ATOMS, HIRSH_Q):
            assert h[key] == pytest.approx(z - q)
        assert h["sum"] == pytest.approx(sum(Z_ATOMS) - sum(HIRSH_Q))
        assert h["abs_sum"] == pytest.approx(h["sum"])

    def test_archived_output_preferred_over_charges(self, tmp_path):
        job = _job(tmp_path, {"hirsh_fuzzy_density": ZEROS},
                   outs={"hirsh_fuzzy_density.out": HIRSH_TWO_BLOCK})
        _write_charges(job, HIRSH_Q)
        rep = recheck_fuzzy(str(job), mult=1)
        assert rep["reparse"] == ["hirsh_fuzzy_density"] and rep["derived"] == []
        h = json.loads((job / "generator" / "fuzzy_full.json").read_text())["hirsh_fuzzy_density"]
        assert h["sum"] == pytest.approx(94.99993976)

    def test_open_shell_spin_still_reruns(self, tmp_path):
        job = _job(tmp_path, {"hirsh_fuzzy_density": ZEROS, "hirsh_fuzzy_spin": ALL_ALPHA_SPIN},
                   loose=["orca.gbw"])
        _write_charges(job, HIRSH_Q)
        rep = recheck_fuzzy(str(job), mult=2, dry_run=True)
        assert rep["derived"] == ["hirsh_fuzzy_density"] and rep["rerun"] == ["hirsh_fuzzy_spin"]

    def test_no_charges_falls_back_to_rerun(self, tmp_path):
        job = _job(tmp_path, {"hirsh_fuzzy_density": ZEROS}, loose=["orca.gbw"])
        rep = recheck_fuzzy(str(job), mult=1, dry_run=True)
        assert rep["rerun"] == ["hirsh_fuzzy_density"] and rep["derived"] == []

    def test_nan_charge_falls_back_to_rerun(self, tmp_path):
        job = _job(tmp_path, {"hirsh_fuzzy_density": ZEROS}, loose=["orca.gbw"])
        _write_charges(job, HIRSH_Q[:-1] + [float("nan")])
        rep = recheck_fuzzy(str(job), mult=1, dry_run=True)
        assert rep["rerun"] == ["hirsh_fuzzy_density"] and rep["derived"] == []

    def test_unknown_element_label_returns_none(self, tmp_path):
        job = _job(tmp_path, {"hirsh_fuzzy_density": ZEROS})
        (job / "generator" / "charge.json").write_text(
            json.dumps({"hirshfeld": {"charge": {"1_Xx": 0.1}}}))
        assert hirsh_density_from_charges(str(job)) is None
