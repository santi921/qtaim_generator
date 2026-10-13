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
import subprocess
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
    write_conversion,
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


def test_multiwfn_wrapper_passes_bare_wavefunction_name(tmp_path):
    # Multiwfn 3.8 rejects input paths over 200 chars, so the script must
    # name the wavefunction relative to its cwd even for deep job folders.
    folder = tmp_path / ("x" * 120) / ("y" * 120)
    folder.mkdir(parents=True)
    inp = folder / "hirshfeld.txt"
    inp.write_text("7\n1\n1\nn\n0\nq\n")
    for override in (False, True):
        write_multiwfn_exe(
            out_folder=str(folder),
            read_file=str(folder / "orca.wfx"),
            multi_wfn_cmd="Multiwfn",
            multiwfn_input_file=str(inp),
            name="props_hirshfeld.mfwn",
            overwrite=True,
            gbw_override=override,
        )
        cmd = (folder / "props_hirshfeld.mfwn").read_text().splitlines()[-1]
        assert cmd.startswith("Multiwfn 'orca.wfx' < ")


def _run_convert(folder, orca_2mkl_cmd):
    write_conversion(out_folder=str(folder), read_file=str(folder / "orca.gbw"), overwrite=True,
                     orca_2mkl_cmd=orca_2mkl_cmd)
    subprocess.run(["bash", "convert.in"], cwd=folder, check=False)


def test_conversion_replaces_a_stale_molden(tmp_path):
    # a molden left by a killed run is removed before orca_2mkl writes the new one;
    # removing it afterwards deleted the fresh file and convert failed on every retry
    fake = tmp_path / "fake_2mkl"
    fake.write_text('#!/bin/bash\necho fresh > "$1.molden.input"\n')
    fake.chmod(0o755)
    (tmp_path / "orca.gbw").write_text("gbw")
    (tmp_path / "orca.molden.input").write_text("stale")
    _run_convert(tmp_path, str(fake))
    assert (tmp_path / "orca.molden.input").read_text() == "fresh\n"
    assert not (tmp_path / "orca.gbw").exists()


def test_failed_conversion_keeps_the_gbw(tmp_path):
    # a stale non-empty molden must not let the gbw guard fire when orca_2mkl fails
    (tmp_path / "orca.gbw").write_text("gbw")
    (tmp_path / "orca.molden.input").write_text("stale, truncated")
    _run_convert(tmp_path, "false")
    assert (tmp_path / "orca.gbw").read_text() == "gbw"
    assert not (tmp_path / "orca.molden.input").exists()


def test_conversion_without_a_gbw_leaves_the_molden(tmp_path):
    # with no gbw the molden is the only source: nothing removes it
    (tmp_path / "orca.molden.input").write_text("source")
    _run_convert(tmp_path, "false")
    assert (tmp_path / "orca.molden.input").read_text() == "source"


def test_multiwfn_wrapper_reaches_a_wavefunction_kept_in_generator(tmp_path):
    # horton's wavefunction gate can pick generator/orca.wfx as the only copy;
    # the script runs with cwd=folder, so the path must be relative to it
    folder = tmp_path / ("x" * 120) / ("y" * 120)
    (folder / "generator").mkdir(parents=True)
    (folder / "generator" / "orca.wfx").write_text("wfx")
    inp = folder / "hirshfeld.txt"
    inp.write_text("7\n1\n1\nn\n0\nq\n")
    write_multiwfn_exe(out_folder=str(folder), read_file=str(folder / "generator" / "orca.wfx"),
                       multi_wfn_cmd="Multiwfn", multiwfn_input_file=str(inp),
                       name="props_hirshfeld.mfwn", overwrite=True)
    cmd = (folder / "props_hirshfeld.mfwn").read_text().splitlines()[-1]
    assert cmd.startswith("Multiwfn 'generator/orca.wfx' < ")
    assert (folder / "generator" / "orca.wfx").is_file()


def _damaged_cpprop_text():
    """CPprop_w_bond_paths.txt cut inside its last block, as found on LRC (2026-10-09)."""
    lines = CPPROP_FIXTURE.read_text().splitlines()
    last = max(i for i, ln in enumerate(lines) if "----------------   CP" in ln)
    cut = next(i for i in range(last, len(lines)) if "ESP from nuclear charges" in lines[i])
    return "\n".join(lines[: cut + 1]) + "\n"


class TestDamagedCpprop:
    """A CPprop.txt damaged after the run carries a complete qtaim.out, so the
    export marker cannot catch it; cpprop_integrity must, at every gate."""

    def _folder(self, tmp_path):
        (tmp_path / "CPprop.txt").write_text(_damaged_cpprop_text())
        shutil.copy(INP_FIXTURE, tmp_path / "input.in")
        (tmp_path / "qtaim.out").write_text(BANNER + COUNT_LINE + EXPORT_LINE + BANNER)
        return str(tmp_path)

    def test_not_parsed(self, tmp_path):
        folder = self._folder(tmp_path)
        parse_multiwfn(folder, separate=False, logger=logging.getLogger("t"))
        assert not os.path.exists(os.path.join(folder, "qtaim.json"))

    def test_raw_output_is_not_usable(self, tmp_path):
        assert not _qtaim_output_complete(self._folder(tmp_path), n_atoms=_n_atoms_in_fixture())

    def test_archived_under_a_name_that_cannot_replace_a_sound_copy(self, tmp_path):
        from qtaim_gen.source.core.omol import clean_jobs

        folder = self._folder(tmp_path)
        gen = tmp_path / "generator"
        gen.mkdir()
        sound = CPPROP_FIXTURE.read_bytes()
        with zipfile.ZipFile(gen / "out_files.zip", "w") as z:
            z.writestr("CPprop.txt", sound)
            z.writestr("qtaim.out", BANNER + COUNT_LINE + EXPORT_LINE + BANNER)
        archived_out = BANNER + COUNT_LINE + EXPORT_LINE + BANNER
        clean_jobs(folder, logger=logging.getLogger("t"), move_results=True)
        with zipfile.ZipFile(gen / "out_files.zip") as z:
            assert z.read("CPprop.txt") == sound
            assert z.read("CPprop.corrupt.txt").decode() == _damaged_cpprop_text()
            # the sound pair stays from one run: the damaged run's qtaim.out does not replace it
            assert z.read("qtaim.out").decode() == archived_out
            assert "qtaim.corrupt.out" in z.namelist()

    def test_sound_cpprop_is_archived_as_is(self, tmp_path):
        from qtaim_gen.source.core.omol import clean_jobs

        shutil.copy(CPPROP_FIXTURE, tmp_path / "CPprop.txt")
        (tmp_path / "qtaim.out").write_text(BANNER + COUNT_LINE + EXPORT_LINE + BANNER)
        clean_jobs(str(tmp_path), logger=logging.getLogger("t"), move_results=True)
        with zipfile.ZipFile(tmp_path / "generator" / "out_files.zip") as z:
            assert "CPprop.corrupt.txt" not in z.namelist()
            assert z.read("CPprop.txt") == CPPROP_FIXTURE.read_bytes()

    def test_damaged_root_copy_fails_without_check_bcp_count(self, tmp_path):
        from qtaim_gen.source.utils.validation import validate_qtaim_dict

        folder = self._folder(tmp_path)
        gen = tmp_path / "generator"
        gen.mkdir()
        rec = {str(i): {"density_all": 1.0} for i in range(_n_atoms_in_fixture())}
        rec.update({f"{i}_{i + 1}": {"density_all": 0.1} for i in range(13)})
        (gen / "qtaim.json").write_text(json.dumps(rec))
        n = _n_atoms_in_fixture()
        assert not validate_qtaim_dict(str(gen / "qtaim.json"), n_atoms=n, folder=folder)
        assert not _qtaim_output_complete(folder, n_atoms=n)

    def test_stale_root_record_is_not_promoted_over_a_damaged_cpprop(self, tmp_path):
        # parse_multiwfn refuses the damaged CPprop.txt and leaves the old root
        # qtaim.json; that leftover must not replace the generator/ record
        import time
        from qtaim_gen.source.core.omol import move_results_to_folder

        gen = tmp_path / "generator"
        gen.mkdir()
        (tmp_path / "qtaim.json").write_text(json.dumps({"0": {"density_all": 9.9}}))
        (gen / "qtaim.json").write_text(json.dumps({"0": {"density_all": 1.0}}))
        time.sleep(0.02)
        self._folder(tmp_path)  # damaged CPprop.txt, newer than both records
        move_results_to_folder(str(tmp_path), logger=logging.getLogger("t"), clean=False)
        assert json.loads((gen / "qtaim.json").read_text()) == {"0": {"density_all": 1.0}}

    def test_archive_outranks_a_legacy_loose_generator_copy(self, tmp_path):
        # nothing writes generator/CPprop.txt any more; read first, a damaged one
        # would outlive every rerun that refreshes the archive
        from qtaim_gen.source.utils.validation import qtaim_cpprop_problems

        gen = tmp_path / "generator"
        gen.mkdir()
        (gen / "CPprop.txt").write_text(_damaged_cpprop_text())
        with zipfile.ZipFile(gen / "out_files.zip", "w") as z:
            z.writestr("CPprop.txt", CPPROP_FIXTURE.read_bytes())
            z.writestr("qtaim.out", COUNT_LINE + EXPORT_LINE)
        assert qtaim_cpprop_problems(str(tmp_path)) == []


def test_every_validation_in_gbw_analysis_forwards_the_mode_flags():
    # merging feat/horton-charge-engine dropped charge_engine from the revalidation
    # after a timing patch, so a Multiwfn folder could pass an engine run and be cleaned
    import ast
    import inspect
    import textwrap

    from qtaim_gen.source.core.omol import gbw_analysis

    tree = ast.parse(textwrap.dedent(inspect.getsource(gbw_analysis)))
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "validation_checks"]
    assert calls
    required = {"charge_engine", "recheck_fuzzy", "recheck_allalpha_qtaim", "recheck_cp_labels", "enforce_poincare_hopf"}
    for call in calls:
        assert required <= {k.arg for k in call.keywords}, (call.lineno, required - {k.arg for k in call.keywords})


def test_patch_timings_leaves_engine_routines_alone(tmp_path):
    # a charge_engine timing stands in for ENGINE_ROUTINES and a surface_engine one for
    # 'other' (validate_timing_dict); only keys that are really missing get patched
    from qtaim_gen.source.core.omol import patch_timings_from_log
    from qtaim_gen.source.data.multiwfn import ENGINE_ROUTINES
    from qtaim_gen.source.utils.validation import get_expected_timing_keys

    expected, _ = get_expected_timing_keys(full_set=0, spin_tf=False)
    timings = {k: 1.0 for k in expected if k not in ENGINE_ROUTINES and k not in ("other", "qtaim")}
    timings.update(charge_engine=5.0, surface_engine=2.0)
    gen = tmp_path / "generator"
    gen.mkdir()
    (gen / "timings.json").write_text(json.dumps(timings))
    assert patch_timings_from_log(str(tmp_path), full_set=0, move_results=True)
    patched = json.loads((gen / "timings.json").read_text())["_timings_patched"]
    assert set(patched) == {"qtaim"}
