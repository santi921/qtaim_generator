"""Tests for QTAIM bond-CP completeness validation.

A truncated Multiwfn CPprop.txt silently loses critical points from the tail of
the CP list. The nuclear-CP count check cannot see it, because Multiwfn numbers
nuclear CPs first so any surviving prefix still satisfies
`len(NCPs) == n_atoms`. These cover the checks that do see it.
"""

import json
import os
import zipfile

import pytest

from qtaim_gen.source.utils.validation import (
    count_reported_bcps,
    qtaim_run_status,
    validate_qtaim_dict,
)

# Multiwfn prints the CP count at the end of the search, then exports the
# per-CP properties and prints a completion line. A complete run has both.
COUNT_LINE = " Number of (3,-1) CPs:    11    Generating topology paths...\n"
EXPORT_LINE = " Done! The results have been outputted to CPprop.txt in current folder\n"
QTAIM_OUT_LINE = COUNT_LINE + EXPORT_LINE


def _write_qtaim_json(path, n_atoms, n_bcps):
    d = {str(i): {"cp_num": i + 1, "density_all": 1.0} for i in range(n_atoms)}
    for b in range(n_bcps):
        d[f"{b}_{b + 1}"] = {"cp_num": n_atoms + b + 1, "density_all": 0.1}
    with open(path, "w") as f:
        json.dump(d, f)
    return path


class TestCountReportedBcps:
    def test_reads_folder_root(self, tmp_path):
        (tmp_path / "qtaim.out").write_text(QTAIM_OUT_LINE)
        assert count_reported_bcps(str(tmp_path)) == 11

    def test_reads_generator_subfolder(self, tmp_path):
        gen = tmp_path / "generator"
        gen.mkdir()
        (gen / "qtaim.out").write_text(QTAIM_OUT_LINE)
        assert count_reported_bcps(str(tmp_path)) == 11

    def test_reads_zipped_out_files(self, tmp_path):
        """After cleanup, qtaim.out only survives inside out_files.zip."""
        gen = tmp_path / "generator"
        gen.mkdir()
        with zipfile.ZipFile(gen / "out_files.zip", "w") as zf:
            zf.writestr("qtaim.out", QTAIM_OUT_LINE)
        assert count_reported_bcps(str(tmp_path)) == 11

    def test_missing_returns_none(self, tmp_path):
        assert count_reported_bcps(str(tmp_path)) is None

    def test_corrupt_zip_returns_none(self, tmp_path):
        gen = tmp_path / "generator"
        gen.mkdir()
        (gen / "out_files.zip").write_text("not a zip")
        assert count_reported_bcps(str(tmp_path)) is None

    def test_takes_last_occurrence(self, tmp_path):
        (tmp_path / "qtaim.out").write_text(
            " Number of (3,-1) CPs:    3\n" + QTAIM_OUT_LINE
        )
        assert count_reported_bcps(str(tmp_path)) == 11


class TestQtaimRunStatus:
    """Counts alone cannot prove completeness: a run killed during the search,
    or during the CPprop.txt export, needs the markers to detect."""

    def test_complete_run(self, tmp_path):
        (tmp_path / "qtaim.out").write_text(QTAIM_OUT_LINE)
        st = qtaim_run_status(str(tmp_path))
        assert st["have_qtaim_out"] and st["search_done"] and st["export_done"]
        assert st["reported_bcp"] == 11

    def test_killed_during_search(self, tmp_path):
        (tmp_path / "qtaim.out").write_text(" Generating starting points...\n")
        st = qtaim_run_status(str(tmp_path))
        assert st["have_qtaim_out"]
        assert not st["search_done"]
        assert st["reported_bcp"] is None

    def test_killed_during_export(self, tmp_path):
        """The dangerous case: the count is there, so a count-only check that
        happened to match would call this complete."""
        (tmp_path / "qtaim.out").write_text(COUNT_LINE)
        st = qtaim_run_status(str(tmp_path))
        assert st["search_done"]
        assert not st["export_done"]
        assert st["reported_bcp"] == 11

    def test_absent_is_unknown_not_complete(self, tmp_path):
        st = qtaim_run_status(str(tmp_path))
        assert st["have_qtaim_out"] is False
        assert st["search_done"] is None
        assert st["export_done"] is None


class TestValidateBcpCompleteness:
    def test_complete_record_passes(self, tmp_path):
        (tmp_path / "qtaim.out").write_text(QTAIM_OUT_LINE)
        p = _write_qtaim_json(tmp_path / "qtaim.json", n_atoms=12, n_bcps=11)
        assert validate_qtaim_dict(
            str(p), n_atoms=12, folder=str(tmp_path), check_bcp_count=True
        )

    def test_truncated_record_fails(self, tmp_path):
        (tmp_path / "qtaim.out").write_text(QTAIM_OUT_LINE)
        p = _write_qtaim_json(tmp_path / "qtaim.json", n_atoms=12, n_bcps=5)
        assert not validate_qtaim_dict(
            str(p), n_atoms=12, folder=str(tmp_path), check_bcp_count=True
        )

    def test_truncation_invisible_to_legacy_check(self, tmp_path):
        """The regression this exists for: nuclear CPs all present, BCPs lost."""
        (tmp_path / "qtaim.out").write_text(QTAIM_OUT_LINE)
        p = _write_qtaim_json(tmp_path / "qtaim.json", n_atoms=12, n_bcps=5)
        assert validate_qtaim_dict(str(p), n_atoms=12)

    def test_empty_bcp_set_fails(self, tmp_path):
        p = _write_qtaim_json(tmp_path / "qtaim.json", n_atoms=12, n_bcps=0)
        assert not validate_qtaim_dict(
            str(p), n_atoms=12, folder=str(tmp_path), check_bcp_count=True
        )

    def test_single_atom_may_have_no_bcps(self, tmp_path):
        p = _write_qtaim_json(tmp_path / "qtaim.json", n_atoms=1, n_bcps=0)
        assert validate_qtaim_dict(
            str(p), n_atoms=1, folder=str(tmp_path), check_bcp_count=True
        )

    def test_extra_bcps_do_not_fail(self, tmp_path):
        """Only a shortfall is a defect; a surplus is not."""
        (tmp_path / "qtaim.out").write_text(QTAIM_OUT_LINE)
        p = _write_qtaim_json(tmp_path / "qtaim.json", n_atoms=14, n_bcps=13)
        assert validate_qtaim_dict(
            str(p), n_atoms=14, folder=str(tmp_path), check_bcp_count=True
        )

    def test_incomplete_search_fails_even_with_matching_counts(self, tmp_path):
        (tmp_path / "qtaim.out").write_text(" Generating starting points...\n")
        p = _write_qtaim_json(tmp_path / "qtaim.json", n_atoms=12, n_bcps=11)
        assert not validate_qtaim_dict(
            str(p), n_atoms=12, folder=str(tmp_path), check_bcp_count=True
        )

    def test_incomplete_export_fails_even_with_matching_counts(self, tmp_path):
        """Counts agree, but the export never finished, so the record cannot
        be trusted as complete."""
        (tmp_path / "qtaim.out").write_text(COUNT_LINE)
        p = _write_qtaim_json(tmp_path / "qtaim.json", n_atoms=12, n_bcps=11)
        assert not validate_qtaim_dict(
            str(p), n_atoms=12, folder=str(tmp_path), check_bcp_count=True
        )

    def test_no_qtaim_out_skips_count_check(self, tmp_path):
        """Absent provenance must not fail an otherwise-valid record."""
        p = _write_qtaim_json(tmp_path / "qtaim.json", n_atoms=12, n_bcps=5)
        assert validate_qtaim_dict(
            str(p), n_atoms=12, folder=str(tmp_path), check_bcp_count=True
        )


class TestRestartSkipHonorsCompleteness:
    """The restart path must not skip a QTAIM step whose output is incomplete.

    Regression: `_has_usable_step_output` accepted any non-empty qtaim.json, so
    a record with every nuclear CP and no bond CPs was skipped as "data
    verified" while validation failed the same job for having no bond critical
    points -- the pipeline detecting a defect and then declining to fix it.
    """

    @staticmethod
    def _job(tmp_path, name, n_atoms, n_bcp, with_out=True):
        folder = tmp_path / name
        (folder / "generator").mkdir(parents=True)
        rec = {str(i): {"cp_num": i + 1} for i in range(n_atoms)}
        for b in range(n_bcp):
            rec[f"{b}_{b + 1}"] = {"cp_num": n_atoms + b + 1}
        (folder / "generator" / "qtaim.json").write_text(json.dumps(rec))
        if with_out:
            (folder / "qtaim.out").write_text(QTAIM_OUT_LINE)
        return folder

    def test_empty_bcp_set_is_never_skipped(self, tmp_path):
        from qtaim_gen.source.core.omol import _has_usable_step_output

        folder = self._job(tmp_path, "empty", n_atoms=225, n_bcp=0)
        assert not _has_usable_step_output(str(folder), "qtaim", n_atoms=225)

    def test_truncated_skipped_unless_flag_set(self, tmp_path):
        from qtaim_gen.source.core.omol import _has_usable_step_output

        # qtaim.out reports 11; the record holds 5
        folder = self._job(tmp_path, "trunc", n_atoms=12, n_bcp=5)
        assert _has_usable_step_output(str(folder), "qtaim", n_atoms=12)
        assert not _has_usable_step_output(
            str(folder), "qtaim", n_atoms=12, check_bcp_count=True
        )

    def test_complete_record_is_skipped(self, tmp_path):
        from qtaim_gen.source.core.omol import _has_usable_step_output

        folder = self._job(tmp_path, "good", n_atoms=12, n_bcp=11)
        assert _has_usable_step_output(
            str(folder), "qtaim", n_atoms=12, check_bcp_count=True
        )

    def test_nuclear_cp_mismatch_is_never_skipped(self, tmp_path):
        from qtaim_gen.source.core.omol import _has_usable_step_output

        folder = self._job(tmp_path, "ncp", n_atoms=12, n_bcp=11)
        assert not _has_usable_step_output(str(folder), "qtaim", n_atoms=20)

    def test_unfinished_export_is_not_skipped(self, tmp_path):
        from qtaim_gen.source.core.omol import _has_usable_step_output

        folder = self._job(tmp_path, "noexport", n_atoms=12, n_bcp=11, with_out=False)
        (folder / "qtaim.out").write_text(COUNT_LINE)  # no export marker
        assert not _has_usable_step_output(
            str(folder), "qtaim", n_atoms=12, check_bcp_count=True
        )

    def test_single_atom_with_no_bcps_is_fine(self, tmp_path):
        from qtaim_gen.source.core.omol import _has_usable_step_output

        folder = self._job(tmp_path, "atom", n_atoms=1, n_bcp=0, with_out=False)
        assert _has_usable_step_output(str(folder), "qtaim", n_atoms=1)


class TestCPpropArchived:
    """CPprop.txt must survive into out_files.zip.

    Regression: the cleanup loop deleted CPprop.txt before the zip was built,
    making the zip's own `endswith("CPprop.txt")` clause dead code. No archived
    job retained the per-CP property blocks, so a lost critical point could not
    be diagnosed after the fact -- qtaim.out carries only the count.
    """

    @staticmethod
    def _job(tmp_path):
        folder = tmp_path / "job"
        folder.mkdir()
        for name in (
            "qtaim.out", "charge.out", "convert.out", "orca.out",
            "CPprop.txt", "settings.ini", "qtaim.txt",
        ):
            (folder / name).write_text(f"content of {name}\n")
        (folder / "qtaim.json").write_text('{"0": {"cp_num": 1}}')
        return folder

    def _clean(self, folder):
        import logging

        from qtaim_gen.source.core.omol import clean_jobs

        clean_jobs(
            str(folder),
            separate=False,
            logger=logging.getLogger("test_clean"),
            full_set=0,
            move_results=False,
        )
        return zipfile.ZipFile(folder / "out_files.zip").namelist()

    def test_cpprop_is_archived(self, tmp_path):
        folder = self._job(tmp_path)
        assert "CPprop.txt" in self._clean(folder)

    def test_cpprop_removed_from_disk_after_archiving(self, tmp_path):
        folder = self._job(tmp_path)
        self._clean(folder)
        assert not (folder / "CPprop.txt").exists()

    def test_step_outs_archived_but_orca_out_kept_loose(self, tmp_path):
        folder = self._job(tmp_path)
        names = self._clean(folder)
        assert "qtaim.out" in names
        # orca.out is parsed separately and must not be swept into the zip
        assert "orca.out" not in names
        assert (folder / "orca.out").exists()


class TestStorableVsReportedCount:
    """Multiwfn's reported count is an upper bound, not a target.

    A 100-job repair test found 21 of 28 residual shortfalls were CPs the
    atom-pair-keyed schema cannot store -- 18 of them CPs with no
    "Connected atoms:" line. Rejecting those would livelock: validation fails,
    the restart path reruns, the rerun reproduces the identical record.
    """

    # exact Multiwfn CPprop.txt block shape; the parser is column/token
    # sensitive, so this mirrors a real file rather than approximating it
    CP_BLOCK = (
        " ----------------   CP{n:>6},     Type (3,-1)   ----------------\n"
        "{connected}"
        " Position (Bohr):        1.000000000000    0.000000000000    {z:.12f}\n"
        " Position (Angstrom):    0.529177000000    0.000000000000    {z:.12f}\n"
        " Density of all electrons:  0.1000000000E+00\n"
    )

    def _cpprop(self, tmp_path, n_with_paths, n_without_paths):
        blocks = []
        for i in range(n_with_paths):
            blocks.append(
                self.CP_BLOCK.format(
                    n=i + 1,
                    z=1.0 + i,
                    connected=(
                        f" Connected atoms: {i + 1:>5}(H )   --  {i + 2:>5}(H )\n"
                    ),
                )
            )
        for j in range(n_without_paths):
            # a CP Multiwfn found but could not attribute to an atom pair
            blocks.append(
                self.CP_BLOCK.format(n=100 + j, z=50.0 + j, connected="")
            )
        (tmp_path / "CPprop.txt").write_text("".join(blocks))

    def test_storable_excludes_cps_without_bond_paths(self, tmp_path):
        from qtaim_gen.source.utils.validation import storable_bcp_count

        self._cpprop(tmp_path, n_with_paths=5, n_without_paths=3)
        assert storable_bcp_count(str(tmp_path)) == 5

    def test_storable_collapses_duplicate_pairs(self, tmp_path):
        from qtaim_gen.source.utils.validation import storable_bcp_count

        blocks = [
            self.CP_BLOCK.format(
                n=i + 1, z=1.0 + i,
                connected=" Connected atoms:     1(H )   --      2(H )\n",
            )
            for i in range(3)
        ]
        (tmp_path / "CPprop.txt").write_text("".join(blocks))
        # three CPs, one atom pair
        assert storable_bcp_count(str(tmp_path)) == 1

    def test_missing_cpprop_returns_none(self, tmp_path):
        from qtaim_gen.source.utils.validation import storable_bcp_count

        assert storable_bcp_count(str(tmp_path)) is None

    def test_unstorable_shortfall_passes_validation(self, tmp_path):
        """The livelock case: reported 8, storable 5, stored 5 -> complete."""
        (tmp_path / "qtaim.out").write_text(
            " Number of (3,-1) CPs:     8\n"
            " Done! The results have been outputted to CPprop.txt in current folder\n"
        )
        self._cpprop(tmp_path, n_with_paths=5, n_without_paths=3)
        p = _write_qtaim_json(tmp_path / "qtaim.json", n_atoms=12, n_bcps=5)
        assert validate_qtaim_dict(
            str(p), n_atoms=12, folder=str(tmp_path), check_bcp_count=True
        )

    def test_real_loss_below_storable_still_fails(self, tmp_path):
        (tmp_path / "qtaim.out").write_text(
            " Number of (3,-1) CPs:     8\n"
            " Done! The results have been outputted to CPprop.txt in current folder\n"
        )
        self._cpprop(tmp_path, n_with_paths=5, n_without_paths=3)
        p = _write_qtaim_json(tmp_path / "qtaim.json", n_atoms=12, n_bcps=2)
        assert not validate_qtaim_dict(
            str(p), n_atoms=12, folder=str(tmp_path), check_bcp_count=True
        )

    def test_small_unattributable_gap_tolerated(self, tmp_path):
        """No CPprop.txt: an off-by-one gap must not fail, or older records
        (which never archived CPprop.txt) would rerun forever."""
        (tmp_path / "qtaim.out").write_text(QTAIM_OUT_LINE)  # reports 11
        p = _write_qtaim_json(tmp_path / "qtaim.json", n_atoms=12, n_bcps=10)
        assert validate_qtaim_dict(
            str(p), n_atoms=12, folder=str(tmp_path), check_bcp_count=True
        )

    def test_large_unattributable_gap_fails(self, tmp_path):
        (tmp_path / "qtaim.out").write_text(QTAIM_OUT_LINE)  # reports 11
        p = _write_qtaim_json(tmp_path / "qtaim.json", n_atoms=12, n_bcps=3)
        assert not validate_qtaim_dict(
            str(p), n_atoms=12, folder=str(tmp_path), check_bcp_count=True
        )


class TestBcpTolerance:
    """Wiggle room so unrepairable jobs stop requeueing.

    Some CPs have no traceable bond path and therefore no storable atom pair.
    Measured on a repair test: 17 jobs missing exactly 1 and 2 missing exactly
    2, independent of system size. Without slack those jobs fail validation,
    get requeued, rerun identically, and never clear.
    """

    @staticmethod
    def _job(tmp_path, n_bcp, reported=11, n_atoms=12):
        (tmp_path / "qtaim.out").write_text(
            f" Number of (3,-1) CPs:    {reported}\n"
            " Done! The results have been outputted to CPprop.txt in current folder\n"
        )
        return _write_qtaim_json(tmp_path / "qtaim.json", n_atoms=n_atoms, n_bcps=n_bcp)

    def test_within_tolerance_passes(self, tmp_path):
        p = self._job(tmp_path, n_bcp=9)  # 2 missing
        assert validate_qtaim_dict(
            str(p), n_atoms=12, folder=str(tmp_path), check_bcp_count=True,
            bcp_tolerance=2,
        )

    def test_beyond_tolerance_fails(self, tmp_path):
        p = self._job(tmp_path, n_bcp=8)  # 3 missing
        assert not validate_qtaim_dict(
            str(p), n_atoms=12, folder=str(tmp_path), check_bcp_count=True,
            bcp_tolerance=2,
        )

    def test_zero_tolerance_is_strict(self, tmp_path):
        p = self._job(tmp_path, n_bcp=10)  # 1 missing
        assert not validate_qtaim_dict(
            str(p), n_atoms=12, folder=str(tmp_path), check_bcp_count=True,
            bcp_tolerance=0,
        )

    def test_tolerance_is_absolute_not_fractional(self, tmp_path):
        """One missing CP must be tolerated the same whether the system has 11
        BCPs or 300; the observed loss does not scale with size."""
        for sub, reported in (("small", 11), ("large", 300)):
            d = tmp_path / sub
            d.mkdir()
            p = self._job(d, n_bcp=reported - 1, reported=reported, n_atoms=reported + 1)
            assert validate_qtaim_dict(
                str(p), n_atoms=reported + 1, folder=str(d),
                check_bcp_count=True, bcp_tolerance=2,
            ), sub

    def test_empty_bcp_set_ignores_tolerance(self, tmp_path):
        """A multi-atom system with no bond CPs is broken regardless of slack."""
        p = self._job(tmp_path, n_bcp=0)
        assert not validate_qtaim_dict(
            str(p), n_atoms=12, folder=str(tmp_path), check_bcp_count=True,
            bcp_tolerance=99,
        )

    def test_restart_gate_uses_same_tolerance(self, tmp_path):
        """The skip gate and the validator must agree, or the restart path
        reruns records validation accepts."""
        from qtaim_gen.source.core.omol import _has_usable_step_output

        folder = tmp_path / "job"
        (folder / "generator").mkdir(parents=True)
        rec = {str(i): {"cp_num": i + 1} for i in range(12)}
        for b in range(10):  # 1 short of the 11 reported
            rec[f"{b}_{b + 1}"] = {"cp_num": 13 + b}
        (folder / "generator" / "qtaim.json").write_text(json.dumps(rec))
        (folder / "qtaim.out").write_text(QTAIM_OUT_LINE)
        assert _has_usable_step_output(
            str(folder), "qtaim", n_atoms=12, check_bcp_count=True, bcp_tolerance=2
        )
        assert not _has_usable_step_output(
            str(folder), "qtaim", n_atoms=12, check_bcp_count=True, bcp_tolerance=0
        )


ORCA_INP_H2O = """! B3LYP def2-SVP
* xyz 0 1
O   0.000000  0.000000  0.117300
H   0.000000  0.757200 -0.469200
H   0.000000 -0.757200 -0.469200
*
"""


class TestMissingQtaimDiscovery:
    """Folders with no qtaim.json / no qtaim.out must be enumerable.

    Discovery used to key on qtaim.json alone, so a job whose QTAIM step never
    ran did not appear in the audit at all -- indistinguishable from a job that
    was never submitted, and reported as if the vertical were clean. And a job
    with qtaim.json but no qtaim.out left search_done/export_done null, which
    the selector read as "not False" and so classified as a known-good control.
    Both populations have to be selectable for the dataset to end up uniform.
    """

    @staticmethod
    def _job(root, name, qtaim=True, qtaim_out=True, n_bcp=2):
        d = os.path.join(root, "vert", name)
        os.makedirs(d)
        with open(os.path.join(d, "orca.inp"), "w") as f:
            f.write(ORCA_INP_H2O)
        if qtaim:
            rec = {str(i): {"cp_num": i + 1} for i in range(3)}
            for b in range(n_bcp):
                rec[f"0_{b + 1}"] = {"cp_num": 4 + b}
            with open(os.path.join(d, "qtaim.json"), "w") as f:
                json.dump(rec, f)
        if qtaim_out:
            with open(os.path.join(d, "qtaim.out"), "w") as f:
                f.write(
                    " Number of (3,-1) CPs:     2\n"
                    " Done! The results have been outputted to CPprop.txt"
                    " in current folder\n"
                )
        return d

    def test_default_walk_skips_folders_without_qtaim_json(self, tmp_path):
        from qtaim_gen.source.scripts.helpers.audit_qtaim_connectivity import (
            find_job_folders,
        )

        root = str(tmp_path)
        self._job(root, "complete")
        self._job(root, "no_json", qtaim=False)
        found = {os.path.basename(f) for f in find_job_folders(root)}
        assert found == {"complete"}

    def test_include_missing_finds_them_by_orca_input(self, tmp_path):
        from qtaim_gen.source.scripts.helpers.audit_qtaim_connectivity import (
            find_job_folders,
        )

        root = str(tmp_path)
        self._job(root, "complete")
        self._job(root, "no_json", qtaim=False)
        found = {
            os.path.basename(f)
            for f in find_job_folders(root, require_qtaim=False)
        }
        assert found == {"complete", "no_json"}

    def test_audit_folder_reports_absence_instead_of_raising(self, tmp_path):
        from qtaim_gen.source.scripts.helpers.audit_qtaim_connectivity import (
            audit_folder,
        )

        d = self._job(str(tmp_path), "no_json", qtaim=False)
        row = audit_folder(d, 1.3)
        assert row["have_qtaim_json"] is False
        assert row["n_atoms"] == 3
        # blank, not zero: zero would read as "searched and found nothing"
        assert row["n_bcp"] is None
        assert row["bcp_shortfall"] is None

    def test_audit_folder_flags_absent_provenance(self, tmp_path):
        from qtaim_gen.source.scripts.helpers.audit_qtaim_connectivity import (
            audit_folder,
        )

        d = self._job(str(tmp_path), "no_out", qtaim_out=False)
        row = audit_folder(d, 1.3)
        assert row["have_qtaim_json"] is True
        assert row["have_qtaim_out"] is False
        assert row["reported_bcp"] is None
        assert row["bcp_shortfall"] is None

    def test_selector_classifies_the_absence_modes(self, tmp_path):
        from qtaim_gen.source.scripts.helpers.audit_qtaim_connectivity import (
            audit_folder,
        )
        from qtaim_gen.source.scripts.helpers.select_qtaim_rerun import classify

        root = str(tmp_path)
        cases = {
            "complete": ("control", {}),
            "no_json": ("no_qtaim_json", {"qtaim": False}),
            "no_out": ("no_provenance", {"qtaim_out": False}),
            "short": ("shortfall", {"n_bcp": 1}),
        }
        for name, (expected, kwargs) in cases.items():
            d = self._job(root, name, **kwargs)
            row = audit_folder(d, 1.3)
            # go through the CSV round trip: DictReader yields strings
            as_csv = {k: "" if v is None else str(v) for k, v in row.items()}
            assert classify(as_csv, bcp_tolerance=0) == expected, name

    def test_selector_respects_the_runner_tolerance(self, tmp_path):
        """A shortfall the runner tolerates must not be selected, or the job
        reruns, returns identical, and is selected again forever."""
        from qtaim_gen.source.scripts.helpers.audit_qtaim_connectivity import (
            audit_folder,
        )
        from qtaim_gen.source.scripts.helpers.select_qtaim_rerun import classify

        d = self._job(str(tmp_path), "short", n_bcp=1)  # 1 missing of 2
        row = audit_folder(d, 1.3)
        as_csv = {k: "" if v is None else str(v) for k, v in row.items()}
        assert classify(as_csv, bcp_tolerance=0) == "shortfall"
        assert classify(as_csv, bcp_tolerance=2) != "shortfall"

    def test_verify_does_not_call_unverifiable_records_fixed(self):
        from qtaim_gen.source.scripts.helpers.verify_qtaim_rerun import classify_state

        assert classify_state({"have_qtaim_out": False}) == "no_provenance"
        assert classify_state({"have_qtaim_json": False}) == "no_qtaim_json"
        assert classify_state({"have_qtaim_out": True, "n_bcp": 5,
                               "n_cov_bonds": 5, "bcp_shortfall": 0}) == "ok"


class TestTristateCoercion:
    """One reader for bool-ish audit fields, whichever side they arrive from.

    Audit rows are consumed straight from audit_folder (real bools) and out of
    a CSV (the strings "True"/"False"/""). Comparing against one form silently
    mishandles the other, and here the wrong answer is the dangerous direction:
    a job with no QTAIM output reads as a known-good control and never requeues.
    """

    def test_accepts_both_forms(self):
        from qtaim_gen.source.utils.validation import as_tristate

        assert as_tristate(False) is False
        assert as_tristate("False") is False
        assert as_tristate(True) is True
        assert as_tristate("True") is True
        assert as_tristate(None) is None
        assert as_tristate("") is None
        assert as_tristate("garbage") is None

    def test_selector_and_verifier_agree_across_both_forms(self):
        from qtaim_gen.source.scripts.helpers.select_qtaim_rerun import classify
        from qtaim_gen.source.scripts.helpers.verify_qtaim_rerun import classify_state

        cases = {
            "no_qtaim_json": {"have_qtaim_json": False, "have_qtaim_out": True},
            "no_provenance": {"have_qtaim_json": True, "have_qtaim_out": False},
        }
        for expected, extra in cases.items():
            row = dict(
                n_atoms=3, n_bcp=2, n_cov_bonds=2, n_isolated_bonded=0,
                bcp_shortfall=0, **extra,
            )
            as_csv = {k: "" if v is None else str(v) for k, v in row.items()}
            assert classify(row) == expected
            assert classify(as_csv) == expected
            assert classify_state(row) == expected


class TestShortfallCheckCost:
    """The clean path must not pay for a CPprop.txt parse it cannot need.

    storable_bcp_count extracts CPprop.txt from generator/out_files.zip and
    reparses every CP block. Multiwfn's reported count is an upper bound on
    what the atom-pair schema can hold, so storable <= reported and a raw
    deficit already inside the tolerance guarantees the exact one is too --
    consulting it first is wasted I/O on the majority of folders.
    """

    @staticmethod
    def _job(tmp_path, reported, n_bcp, storable=None, zipped=False, n_atoms=12):
        (tmp_path / "generator").mkdir(exist_ok=True)
        (tmp_path / "qtaim.out").write_text(
            f" Number of (3,-1) CPs:    {reported}\n"
            " Done! The results have been outputted to CPprop.txt in current folder\n"
        )
        if storable is not None:
            blocks = []
            for k in range(1, reported + 1):
                blocks.append(
                    f" ----------------   CP{k:>6},     Type (3,-1)   ----------------"
                )
                if k <= storable:
                    blocks.append(
                        f" Connected atoms: {k:>5}(H )   --  {k + 1:>5}(H )"
                    )
            body = "\n".join(blocks) + "\n"
            if zipped:
                with zipfile.ZipFile(
                    tmp_path / "generator" / "out_files.zip", "w"
                ) as z:
                    z.writestr("CPprop.txt", body)
            else:
                (tmp_path / "CPprop.txt").write_text(body)
        return _write_qtaim_json(
            tmp_path / "qtaim.json", n_atoms=n_atoms, n_bcps=n_bcp
        )

    @pytest.mark.parametrize("deficit,expect_calls", [(0, 0), (1, 0), (2, 0), (3, 1)])
    def test_parse_only_reached_past_the_tolerance(
        self, tmp_path, monkeypatch, deficit, expect_calls
    ):
        import qtaim_gen.source.utils.validation as V

        p = self._job(tmp_path, reported=11, n_bcp=11 - deficit)
        calls = []
        real = V.storable_bcp_count
        monkeypatch.setattr(
            V, "storable_bcp_count", lambda f: (calls.append(f), real(f))[1]
        )
        V.validate_qtaim_dict(
            str(p), n_atoms=12, folder=str(tmp_path),
            check_bcp_count=True, bcp_tolerance=2,
        )
        assert len(calls) == expect_calls

    def test_restart_gate_skips_the_parse_too(self, tmp_path, monkeypatch):
        import qtaim_gen.source.utils.validation as V
        from qtaim_gen.source.core.omol import _qtaim_output_complete

        gen = tmp_path / "generator"
        gen.mkdir()
        self._job(tmp_path, reported=11, n_bcp=10)
        os.replace(tmp_path / "qtaim.json", gen / "qtaim.json")
        calls = []
        monkeypatch.setattr(
            V, "storable_bcp_count", lambda f: calls.append(f) or None
        )
        assert _qtaim_output_complete(
            str(tmp_path), n_atoms=12, check_bcp_count=True, bcp_tolerance=2
        )
        assert calls == []

    def test_zipped_cpprop_still_rescues_a_large_deficit(self, tmp_path):
        """Past the tolerance the exact count is consulted, and a record that
        is maximally complete for the schema still passes."""
        p = self._job(tmp_path, reported=14, n_bcp=11, storable=11, zipped=True)
        assert validate_qtaim_dict(
            str(p), n_atoms=12, folder=str(tmp_path),
            check_bcp_count=True, bcp_tolerance=2,
        )

    def test_gate_and_validator_agree_either_side_of_the_tolerance(self, tmp_path):
        from qtaim_gen.source.core.omol import _qtaim_output_complete

        for deficit in (1, 3):
            d = tmp_path / f"d{deficit}"
            (d / "generator").mkdir(parents=True)
            p = self._job(d, reported=11, n_bcp=11 - deficit)
            gate = _qtaim_output_complete(
                str(d), n_atoms=12, check_bcp_count=True, bcp_tolerance=2
            )
            val = validate_qtaim_dict(
                str(p), n_atoms=12, folder=str(d),
                check_bcp_count=True, bcp_tolerance=2,
            )
            assert gate == val, deficit
            assert gate is (deficit <= 2)
