"""Restart-gate fixes for the Aug 2026 OMol4M stall loops.

Ways a folder was re-queued forever:
- qtaim finished but qtaim.json was never written (killed before the final
  parse), so the gate redid the whole qtaim step on every restart;
- the "reparse before rerun" path parsed a partial CPprop.txt over a good
  qtaim.json;
- a banner-complete .out from a broken wavefunction was skipped as verified
  while its parse failed every pass;
- a Multiwfn crash was masked by tee and logged as "Completed".
"""

import json
import logging
import os
import shutil
import zipfile
from pathlib import Path

from qtaim_gen.source.core.omol import (
    _compiled_data_present,
    _expected_electrons,
    _has_ecp_atoms,
    _has_usable_step_output,
    _qtaim_output_complete,
    _reject_wavefunction_with_wrong_electron_count,
    _step_out_parses,
    _wavefunction_electrons,
    parse_multiwfn,
    write_multiwfn_exe,
)
from qtaim_gen.source.core.parse_multiwfn import parse_charge_base, parse_charge_doc
from qtaim_gen.source.core.parse_qtaim import get_qtaim_descs, only_atom_cps

TEST_FILES = Path(__file__).parent / "test_files"
CPPROP_FIXTURE = TEST_FILES / "CPprop_w_bond_paths.txt"
INP_FIXTURE = TEST_FILES / "input_bond_paths.in"
HIRSHFELD_FIXTURE = TEST_FILES / "multiwfn" / "hirshfeld.out"
CHARGE_DOC_FIXTURE = TEST_FILES / "multiwfn" / "charge.out"
CHELPG_GLUED_FIXTURE = TEST_FILES / "multiwfn" / "chelpg_glued_overflow.out"

COUNT_LINE = " Number of (3,-1) CPs:    13    Generating topology paths...\n"
EXPORT_LINE = " Done! The results have been outputted to CPprop.txt in current folder\n"
BANNER = "                   ************ Main function menu ************\n"

WATER_INP = "! B3LYP def2-SVP\n*xyz 0 1\nO 0.0 0.0 0.0\nH 0.0 0.0 0.96\nH 0.93 0.0 -0.24\n*\n"


def _n_atoms_in_fixture() -> int:
    atoms, _ = only_atom_cps(get_qtaim_descs(str(CPPROP_FIXTURE)))
    return len(atoms)


def _write_wfx(path, electrons, core=None):
    lines = [
        "<Number of Nuclei>", "   3", "</Number of Nuclei>",
        "<Number of Electrons>", f"   {electrons}", "</Number of Electrons>",
    ]
    if core is not None:
        lines += ["<Number of Core Electrons>", f"   {core}", "</Number of Core Electrons>"]
    lines += ["<Nuclear Names>", "O1", "H2", "H3", "</Nuclear Names>", "<Primitive Centers>"]
    Path(path).write_text("\n".join(lines) + "\n")
    return str(path)


class TestQtaimRawOutputAcceptedWithoutJson:
    def _folder(self, tmp_path, qtaim_out_text=COUNT_LINE + EXPORT_LINE, cpprop=True):
        if cpprop:
            shutil.copy(CPPROP_FIXTURE, tmp_path / "CPprop.txt")
        if qtaim_out_text is not None:
            (tmp_path / "qtaim.out").write_text(BANNER + qtaim_out_text + BANNER)
        return str(tmp_path)

    def test_complete_raw_output_is_usable(self, tmp_path):
        folder = self._folder(tmp_path)
        n = _n_atoms_in_fixture()
        assert _qtaim_output_complete(folder, n_atoms=n)
        assert _has_usable_step_output(folder, "qtaim", n_atoms=n)

    def test_usable_without_atom_count(self, tmp_path):
        assert _qtaim_output_complete(self._folder(tmp_path), n_atoms=None)

    def test_atom_count_mismatch_reruns(self, tmp_path):
        folder = self._folder(tmp_path)
        assert not _qtaim_output_complete(folder, n_atoms=_n_atoms_in_fixture() + 1)

    def test_export_not_finished_reruns(self, tmp_path):
        folder = self._folder(tmp_path, qtaim_out_text=COUNT_LINE)
        assert not _qtaim_output_complete(folder, n_atoms=_n_atoms_in_fixture())

    def test_search_not_finished_reruns(self, tmp_path):
        folder = self._folder(tmp_path, qtaim_out_text=EXPORT_LINE)
        assert not _qtaim_output_complete(folder, n_atoms=_n_atoms_in_fixture())

    def test_missing_cpprop_reruns(self, tmp_path):
        folder = self._folder(tmp_path, cpprop=False)
        assert not _qtaim_output_complete(folder, n_atoms=_n_atoms_in_fixture())

    def test_missing_qtaim_out_reruns(self, tmp_path):
        folder = self._folder(tmp_path, qtaim_out_text=None)
        assert not _qtaim_output_complete(folder, n_atoms=_n_atoms_in_fixture())


class TestParseMultiwfnGuardsPartialCpprop:
    def _folder(self, tmp_path, qtaim_out_text):
        shutil.copy(CPPROP_FIXTURE, tmp_path / "CPprop.txt")
        shutil.copy(INP_FIXTURE, tmp_path / "input.in")
        (tmp_path / "qtaim.out").write_text(BANNER + qtaim_out_text + BANNER)
        return str(tmp_path)

    def test_partial_export_is_not_parsed(self, tmp_path):
        folder = self._folder(tmp_path, COUNT_LINE)
        parse_multiwfn(folder, separate=False, logger=logging.getLogger("t"))
        assert not os.path.exists(os.path.join(folder, "qtaim.json"))

    def test_complete_export_is_parsed(self, tmp_path):
        folder = self._folder(tmp_path, COUNT_LINE + EXPORT_LINE)
        parse_multiwfn(folder, separate=False, logger=logging.getLogger("t"))
        assert os.path.getsize(os.path.join(folder, "qtaim.json")) > 0


class TestStepOutMustParse:
    def _fixture_sum(self):
        charges, _ = parse_charge_base(str(HIRSHFELD_FIXTURE), corrected=False)
        return len(charges), sum(charges.values())

    def test_real_output_is_usable(self, tmp_path):
        shutil.copy(HIRSHFELD_FIXTURE, tmp_path / "hirshfeld.out")
        n, total = self._fixture_sum()
        assert _has_usable_step_output(
            str(tmp_path), "hirshfeld", n_atoms=n, charge=round(total)
        )

    def test_overflowed_charges_rerun(self, tmp_path):
        # A run on a truncated wavefunction prints Fortran overflow stars.
        lines = HIRSHFELD_FIXTURE.read_text().splitlines()
        row = next(i for i, line in enumerate(lines) if "Final atomic charges:" in line) + 1
        lines[row] = lines[row].rsplit(" ", 1)[0] + " ************"
        (tmp_path / "hirshfeld.out").write_text("\n".join(lines) + "\n")
        assert not _has_usable_step_output(str(tmp_path), "hirshfeld")

    def test_charges_not_summing_to_net_charge_rerun(self, tmp_path):
        shutil.copy(HIRSHFELD_FIXTURE, tmp_path / "hirshfeld.out")
        n, total = self._fixture_sum()
        assert not _has_usable_step_output(
            str(tmp_path), "hirshfeld", n_atoms=n, charge=round(total) + 3
        )

    def test_wrong_atom_count_reruns(self, tmp_path):
        shutil.copy(HIRSHFELD_FIXTURE, tmp_path / "hirshfeld.out")
        n, _ = self._fixture_sum()
        assert not _has_usable_step_output(str(tmp_path), "hirshfeld", n_atoms=n + 1)

    def test_routine_without_parser_falls_back_to_banner(self, tmp_path):
        p = tmp_path / "nonesuch.out"
        p.write_text("anything")
        assert _step_out_parses(str(p), "nonesuch")

    def test_empty_bond_table_only_for_tiny_systems(self, tmp_path):
        # Same exemption validate_bond_dict grants: no pairs is fine for 1-2
        # atoms, suspicious for anything larger.
        p = tmp_path / "fuzzy_bond.out"
        p.write_text(BANNER + BANNER)
        assert _step_out_parses(str(p), "fuzzy_bond", n_atoms=2)
        assert not _step_out_parses(str(p), "fuzzy_bond", n_atoms=5)
        assert not _step_out_parses(str(p), "fuzzy_bond", n_atoms=None)

    def test_nested_charge_routine_checks_each_scheme(self):
        schemes, _, _ = parse_charge_doc(str(CHARGE_DOC_FIXTURE))
        n = len(next(iter(schemes.values())))
        assert _step_out_parses(str(CHARGE_DOC_FIXTURE), "charge", n_atoms=n, charge=2)
        assert not _step_out_parses(str(CHARGE_DOC_FIXTURE), "charge", n_atoms=n + 1)
        assert not _step_out_parses(str(CHARGE_DOC_FIXTURE), "charge", charge=5)

    def test_glued_overflow_charges_rerun(self):
        # Values of |q| >= 100 parse (glued to the paren) but cannot sum to
        # the net charge; the gate must not need n_atoms to reject them.
        assert not _step_out_parses(str(CHELPG_GLUED_FIXTURE), "chelpg", charge=1)
        assert not _step_out_parses(str(CHELPG_GLUED_FIXTURE), "chelpg", n_atoms=150, charge=1)

    def test_truncated_charge_row_reruns(self, tmp_path):
        p = tmp_path / "chelpg.out"
        p.write_text("   Center       Charge\n     1(C )  -0.67\n     2(C\n")
        assert not _step_out_parses(str(p), "chelpg")


class TestWavefunctionElectronCount:
    def test_reads_valence_plus_core(self, tmp_path):
        assert _wavefunction_electrons(_write_wfx(tmp_path / "orca.wfx", 106)) == 106
        assert _wavefunction_electrons(_write_wfx(tmp_path / "orca.wfx", 106, core=28)) == 134

    def test_orca_native_layout_counts_after_nuclear_names(self, tmp_path):
        p = tmp_path / "orca.wfx"
        p.write_text(
            "<Number of Nuclei>\n 3\n</Number of Nuclei>\n"
            "<Nuclear Names>\nO1\nH2\nH3\n</Nuclear Names>\n"
            "<Number of Electrons>\n 10\n</Number of Electrons>\n"
            "<Primitive Centers>\n1 1 1\n"
        )
        assert _wavefunction_electrons(str(p)) == 10

    def test_wfn_is_not_judged(self, tmp_path):
        p = tmp_path / "orca.wfn"
        p.write_text("GAUSSIAN 1 MOL ORBITALS\n")
        assert _wavefunction_electrons(str(p)) is None

    def test_expected_from_input(self):
        dft = {
            "mol": {0: {"element": "O"}, 1: {"element": "H"}, 2: {"element": "H"}},
            "charge": -1,
        }
        assert _expected_electrons(dft) == 11

    def test_ecp_systems_are_not_judged(self):
        # def2 ECPs start at Rb; the wfx core count depends on whether the EDF
        # library loaded, so the check would misfire on a correct wfx.
        dft = {"mol": {0: {"element": "I"}, 1: {"element": "H"}}, "charge": 0}
        assert _expected_electrons(dft) is None
        assert _has_ecp_atoms(dft)
        assert not _has_ecp_atoms({"mol": {0: {"element": "Br"}}, "charge": 0})


class TestRejectBadWavefunction:
    def _folder(self, tmp_path, electrons, with_gbw=True, gbw_name="orca.gbw"):
        (tmp_path / "orca.inp").write_text(WATER_INP)
        _write_wfx(tmp_path / "orca.wfx", electrons)
        if with_gbw:
            (tmp_path / gbw_name).write_bytes(b"gbw")
        (tmp_path / "orca.molden.input").write_text("[Molden]\n")
        (tmp_path / "hirshfeld.out").write_text(BANNER + BANNER)
        (tmp_path / "hirshfeld.json").write_text("{}")
        (tmp_path / "CPprop.txt").write_text("cp")
        (tmp_path / "orca.json").write_text("{}")
        (tmp_path / "timings.json").write_text(json.dumps({"hirshfeld": 1.0}))
        gen = tmp_path / "generator"
        gen.mkdir()
        (gen / "charge.json").write_text("{}")
        (gen / "qtaim.json").write_text("{}")
        with zipfile.ZipFile(gen / "out_files.zip", "w") as zf:
            zf.writestr("adch.out", "x")
        return str(tmp_path)

    def test_mismatch_discards_everything_derived(self, tmp_path):
        folder = self._folder(tmp_path, electrons=4)  # water has 10
        assert _reject_wavefunction_with_wrong_electron_count(folder, logging.getLogger("t"))
        gone = [
            "orca.wfx", "orca.molden.input", "hirshfeld.out", "hirshfeld.json",
            "CPprop.txt", "generator/charge.json", "generator/qtaim.json",
            "generator/out_files.zip",
        ]
        for rel in gone:
            assert not os.path.exists(os.path.join(folder, rel)), rel
        for rel in ("orca.inp", "orca.json", "timings.json", "orca.gbw"):
            assert os.path.exists(os.path.join(folder, rel)), rel

    def test_compressed_gbw_counts_only_with_preprocessing(self, tmp_path):
        # Without preprocessing nothing would extract the .gbw again, so the
        # wfx must stay; with it the folder can be regenerated.
        log = logging.getLogger("t")
        folder = self._folder(tmp_path, electrons=4, gbw_name="orca.gbw.zstd0")
        assert not _reject_wavefunction_with_wrong_electron_count(folder, log)
        assert os.path.exists(os.path.join(folder, "orca.wfx"))
        assert _reject_wavefunction_with_wrong_electron_count(
            folder, log, preprocess_compressed=True
        )
        assert not os.path.exists(os.path.join(folder, "orca.wfx"))

    def test_matching_count_leaves_folder_alone(self, tmp_path):
        folder = self._folder(tmp_path, electrons=10)
        assert not _reject_wavefunction_with_wrong_electron_count(folder, logging.getLogger("t"))
        assert os.path.exists(os.path.join(folder, "orca.wfx"))
        assert os.path.exists(os.path.join(folder, "generator", "charge.json"))

    def test_no_gbw_source_leaves_folder_alone(self, tmp_path):
        folder = self._folder(tmp_path, electrons=4, with_gbw=False)
        assert not _reject_wavefunction_with_wrong_electron_count(folder, logging.getLogger("t"))
        assert os.path.exists(os.path.join(folder, "orca.wfx"))


class TestCompiledChargeSum:
    MAP = {"hirshfeld": ("charge.json", "hirshfeld", "charge")}

    def _write(self, tmp_path, charges):
        (tmp_path / "charge.json").write_text(
            json.dumps({"hirshfeld": {"charge": charges}})
        )
        return str(tmp_path)

    def test_garbage_sum_is_not_verified(self, tmp_path):
        folder = self._write(tmp_path, {"1_O": 200.0, "2_H": 90.0, "3_H": 94.0})
        assert not _compiled_data_present(folder, "hirshfeld", self.MAP, n_atoms=3, charge=0)
        assert _compiled_data_present(folder, "hirshfeld", self.MAP, n_atoms=3)

    def test_good_sum_is_verified(self, tmp_path):
        folder = self._write(tmp_path, {"1_O": -0.6, "2_H": 0.3, "3_H": 0.3})
        assert _compiled_data_present(folder, "hirshfeld", self.MAP, n_atoms=3, charge=0)
        assert not _compiled_data_present(folder, "hirshfeld", self.MAP, n_atoms=3, charge=2)


def test_multiwfn_wrapper_sets_pipefail(tmp_path):
    inp = tmp_path / "hirshfeld.txt"
    inp.write_text("7\n1\n1\nn\n0\nq\n")
    write_multiwfn_exe(
        out_folder=str(tmp_path),
        read_file="orca.wfx",
        multi_wfn_cmd="Multiwfn",
        multiwfn_input_file=str(inp),
        name="props_hirshfeld.mfwn",
    )
    lines = (tmp_path / "props_hirshfeld.mfwn").read_text().splitlines()
    assert lines[0] == "#!/bin/bash"
    assert lines[1] == "set -o pipefail"
    assert "| tee" in lines[-1]
