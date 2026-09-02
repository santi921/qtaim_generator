"""Restart-gate fixes for the Aug 2026 OMol4M stall loops.

Four ways a folder was re-queued forever:
- qtaim finished but qtaim.json was never written (killed before the final
  parse), so the gate redid the whole qtaim step on every restart;
- the "reparse before rerun" path parsed a partial CPprop.txt over a good
  qtaim.json;
- a banner-complete .out from a broken wavefunction was skipped as verified
  while its parse failed every pass;
- a Multiwfn crash was masked by tee and logged as "Completed".
"""

import logging
import os
import shutil
from pathlib import Path

from qtaim_gen.source.core.omol import (
    _expected_electrons,
    _has_usable_step_output,
    _qtaim_output_complete,
    _step_out_parses,
    _wavefunction_electrons,
    parse_multiwfn,
    write_multiwfn_exe,
)
from qtaim_gen.source.core.parse_multiwfn import parse_charge_base
from qtaim_gen.source.core.parse_qtaim import get_qtaim_descs, only_atom_cps

TEST_FILES = Path(__file__).parent / "test_files"
CPPROP_FIXTURE = TEST_FILES / "CPprop_w_bond_paths.txt"
INP_FIXTURE = TEST_FILES / "input_bond_paths.in"
HIRSHFELD_FIXTURE = TEST_FILES / "multiwfn" / "hirshfeld.out"

COUNT_LINE = " Number of (3,-1) CPs:    13    Generating topology paths...\n"
EXPORT_LINE = " Done! The results have been outputted to CPprop.txt in current folder\n"
BANNER = "                   ************ Main function menu ************\n"


def _n_atoms_in_fixture() -> int:
    atoms, _ = only_atom_cps(get_qtaim_descs(str(CPPROP_FIXTURE)))
    return len(atoms)


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


class TestWavefunctionElectronCount:
    def _wfx(self, tmp_path, electrons, core=None):
        lines = [
            "<Number of Nuclei>", "   3", "</Number of Nuclei>",
            "<Number of Electrons>", f"   {electrons}", "</Number of Electrons>",
        ]
        if core is not None:
            lines += ["<Number of Core Electrons>", f"   {core}", "</Number of Core Electrons>"]
        lines += ["<Nuclear Names>", "O1", "H2", "H3", "</Nuclear Names>", "<Primitive Centers>"]
        p = tmp_path / "orca.wfx"
        p.write_text("\n".join(lines) + "\n")
        return str(p)

    def test_reads_valence_plus_core(self, tmp_path):
        assert _wavefunction_electrons(self._wfx(tmp_path, 106)) == 106
        assert _wavefunction_electrons(self._wfx(tmp_path, 106, core=28)) == 134

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
