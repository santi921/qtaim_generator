"""recheck_fuzzy: value checks, reparse from archived output, invalidation, wavefunction guard."""

import json
import os
import zipfile
from pathlib import Path

import pytest

from qtaim_gen.source.utils.fuzzy_recheck import (
    _electron_count,
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


def _summing_to(total, rest=1.0):
    """A 12-atom table whose atom values sum to `total`."""
    return _table([total - 11 * rest] + [rest] * 11)


def _fuzzy_out(values):
    """A Multiwfn fuzzy integration table for ATOMS, as parse_fuzzy_real_space reads it."""
    rows = [f"    {a.split('_')[0]}({a.split('_')[1]})      {v:.10E}    0.0    0.0" for a, v in zip(ATOMS, values)]
    return ("   Atomic space        Value                % of sum            % of sum abs\n" + "\n".join(rows)
            + f"\n Summing up above values:    {sum(values):.10E}"
            + f"\n Summing up absolute value of above values:    {sum(abs(v) for v in values):.10E}\n\n")


class TestGridAlphaLoss:
    """Spin falling short of mult - 1 by the same amount as the density misses its integer: grid loss of
    a diffuse alpha electron (LRC tm_react dianions, 2026-10-19), accepted; everything else still flagged."""

    # measured: MOR21_Au3 mult 3 (spin 1.733, density 357.733 of 358), MOR21_Fe3 mult 6 (4.76, 238.76 of 239)
    AU3 = (3, 1.733, 357.733)
    FE3 = (6, 4.760, 238.760)

    def test_measured_dianions_accepted(self):
        for mult, spin, dens in (self.AU3, self.FE3):
            fz = {"becke_fuzzy_density": _summing_to(dens), "hirsh_fuzzy_density": _summing_to(dens),
                  "becke_fuzzy_spin": _summing_to(spin, 0.01), "hirsh_fuzzy_spin": _summing_to(spin, 0.01)}
            assert fuzzy_value_failures(fz, mult=mult) == []

    def test_any_partition_density_counts(self):
        # a hirsh density rebuilt from charges is an exact integer; the becke integral shows the loss
        mult, spin, dens = self.AU3
        fz = {"becke_fuzzy_density": _summing_to(dens), "hirsh_fuzzy_density": _summing_to(358.0),
              "hirsh_fuzzy_spin": _summing_to(spin, 0.01)}
        assert fuzzy_value_failures(fz, mult=mult) == []

    def test_spin_equal_to_density_still_flagged(self):
        # the #28 defect (Cu2 / Cr2 pattern): spin integrates to the whole density
        fz = {"becke_fuzzy_density": _summing_to(357.0), "becke_fuzzy_spin": _summing_to(357.0)}
        assert fuzzy_value_failures(fz, mult=2) == ["becke_fuzzy_spin"]

    def test_shortfall_without_density_loss_flagged(self):
        fz = {"becke_fuzzy_density": _summing_to(358.0), "becke_fuzzy_spin": _summing_to(1.733, 0.01)}
        assert fuzzy_value_failures(fz, mult=3) == ["becke_fuzzy_spin"]

    def test_shortfall_beyond_limit_flagged(self):
        fz = {"becke_fuzzy_density": _summing_to(357.3), "becke_fuzzy_spin": _summing_to(1.3, 0.01)}
        assert fuzzy_value_failures(fz, mult=3) == ["becke_fuzzy_spin"]

    def test_excess_spin_flagged(self):
        fz = {"becke_fuzzy_density": _summing_to(358.267), "becke_fuzzy_spin": _summing_to(2.267, 0.01)}
        assert fuzzy_value_failures(fz, mult=3) == ["becke_fuzzy_spin"]

    def test_spin_without_density_flagged(self):
        # nothing to compare the shortfall with, so no exemption
        assert fuzzy_value_failures({"becke_fuzzy_spin": _summing_to(1.733, 0.01)}, mult=3) == ["becke_fuzzy_spin"]

    def test_zero_density_cannot_vouch(self):
        fz = {"hirsh_fuzzy_density": ZEROS, "becke_fuzzy_spin": _summing_to(1.733, 0.01)}
        assert sorted(fuzzy_value_failures(fz, mult=3)) == ["becke_fuzzy_spin", "hirsh_fuzzy_density"]

    def test_electron_count_pins_the_match(self):
        # density 1.267 e short of N = 358, spin 0.267 e short: near an integer, but not the electron count
        fz = {"becke_fuzzy_density": _summing_to(356.733), "becke_fuzzy_spin": _summing_to(1.733, 0.01)}
        assert fuzzy_value_failures(fz, mult=3) == []
        assert fuzzy_value_failures(fz, mult=3, n_electrons=358) == ["becke_fuzzy_spin"]
        fz["becke_fuzzy_density"] = _summing_to(357.733)
        assert fuzzy_value_failures(fz, mult=3, n_electrons=358) == []

    def test_electron_count_from_input(self, tmp_path):
        # orca5 fixture: 118 atoms, sum of Z 645, charge -2
        assert _electron_count(str(Path(__file__).parent / "test_files" / "lmdb_tests" / "orca5")) == 647
        assert _electron_count(str(tmp_path)) is None

    def test_recheck_leaves_grid_loss_folder_alone(self, tmp_path):
        mult, spin, dens = self.FE3
        job = _job(tmp_path, {"becke_fuzzy_density": _summing_to(dens), "hirsh_fuzzy_density": _summing_to(dens),
                              "becke_fuzzy_spin": _summing_to(spin, 0.01), "hirsh_fuzzy_spin": _summing_to(spin, 0.01)})
        rep = recheck_fuzzy(str(job), mult=mult, dry_run=True)
        assert rep["reparse"] == [] and rep["rerun"] == []

    def test_reparsed_grid_loss_spin_accepted(self, tmp_path):
        # stored spin is the #28 defect; the archived .wfx-era output carries the grid loss and is reparsed
        mult, spin, dens = self.AU3
        good = [spin - 11 * 0.01] + [0.01] * 11
        job = _job(tmp_path, {"becke_fuzzy_density": _summing_to(dens), "becke_fuzzy_spin": _summing_to(dens)},
                   outs={"becke_fuzzy_spin.out": _fuzzy_out(good)})
        rep = recheck_fuzzy(str(job), mult=mult, dry_run=True)
        assert rep["reparse"] == ["becke_fuzzy_spin"] and rep["rerun"] == []

    def test_input_read_only_when_spin_off(self, tmp_path, monkeypatch):
        import qtaim_gen.source.utils.fuzzy_recheck as fr
        calls = []
        monkeypatch.setattr(fr, "_electron_count", lambda folder: calls.append(folder))
        job = _job(tmp_path, {"becke_fuzzy_density": GOOD_DENSITY, "becke_fuzzy_spin": GOOD_SPIN})
        fr.plan_recheck(str(job), 2)
        assert calls == []
        fr.plan_recheck(str(job), 3)
        assert calls == [str(job)]


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

    RESTRICTED_LINES = (
        "#    1:    1(H )    2(Br)   0.899321\n"
        "#    2:    4(C )    5(C )   1.251987\n"
    )

    def _inp(self, job, keyword):
        (job / "orca.inp").write_text(f"! {keyword} wB97M-V def2-TZVPD\n*xyz 0 1\nH 0 0 0\nH 0 0 0.74\n*\n")

    def _restricted_out(self, lines=True):
        head = FUZZY_BOND_WFX.split("Orbitals from")[0].replace(
            self.BANNER, "Total/Alpha/Beta electrons:     96.0000     48.0000     48.0000")
        return head + ("The total bond order >=  0.050000\n" + self.RESTRICTED_LINES if lines else "")

    def test_uks_singlet_without_archive_reruns(self, tmp_path):
        job = _job(tmp_path, {}, bond={"fuzzy_bond": {"4_C_to_5_C": 2.5}}, loose=["orca.gbw"])
        self._inp(job, "UKS")
        rep = recheck_fuzzy(str(job), mult=1, dry_run=True)
        assert rep["rerun"] == ["fuzzy_bond"] and rep["ok"]

    def test_rks_singlet_never_parsed_or_rerun(self, tmp_path):
        job = _job(tmp_path, {}, bond={"fuzzy_bond": {"4_C_to_5_C": 9.9}},
                   outs={"fuzzy_bond.out": self._out(96, 96, 0)})
        self._inp(job, "RKS")
        assert recheck_fuzzy(str(job), mult=1) == {
            "reparse": [], "rerun": [], "derived": [], "wavefunction": "not needed", "ok": True}

    def test_restricted_single_column_output_is_clean(self, tmp_path):
        job = _job(tmp_path, {}, bond={"fuzzy_bond": {"4_C_to_5_C": 9.9}},
                   outs={"fuzzy_bond.out": self._restricted_out()})
        assert recheck_fuzzy(str(job), mult=1)["reparse"] == []

    def test_truncated_restricted_output_is_clean_and_keeps_the_wfn(self, tmp_path):
        job = _job(tmp_path, {}, bond={"fuzzy_bond": {"4_C_to_5_C": 1.0}},
                   outs={"fuzzy_bond.out": self._restricted_out(lines=False)}, loose=["input.wfn", "orca.gbw"])
        rep = recheck_fuzzy(str(job), mult=1)
        assert rep["rerun"] == [] and rep["reparse"] == []
        assert (job / "input.wfn").exists()

    def test_truncated_unrestricted_output_reruns(self, tmp_path):
        truncated = FUZZY_BOND_WFX.split("The total bond order")[0].replace(
            self.BANNER, "Total/Alpha/Beta electrons:     96.0000     48.0000     48.0000")
        job = _job(tmp_path, {}, bond={"fuzzy_bond": {"4_C_to_5_C": 1.0}},
                   outs={"fuzzy_bond.out": truncated}, loose=["orca.gbw"])
        assert recheck_fuzzy(str(job), mult=1, dry_run=True)["rerun"] == ["fuzzy_bond"]

    def test_overflowed_archive_reruns_instead_of_crashing(self, tmp_path):
        bond = {"fuzzy_bond": {"4_C_to_5_C": 0.643844}}
        bad = FUZZY_BOND_WFX.replace("Total:  0.899321", "Total:**********", 1)
        job = _job(tmp_path, {}, bond=bond, outs={"fuzzy_bond.out": bad}, loose=["orca.gbw"])
        rep = recheck_fuzzy(str(job), mult=2, dry_run=True)
        assert rep["rerun"] == ["fuzzy_bond"] and rep["reparse"] == []

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
