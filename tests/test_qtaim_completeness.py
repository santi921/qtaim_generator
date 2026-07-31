"""Tests for QTAIM bond-CP completeness validation.

A truncated Multiwfn CPprop.txt silently loses critical points from the tail of
the CP list. The nuclear-CP count check cannot see it, because Multiwfn numbers
nuclear CPs first so any surviving prefix still satisfies
`len(NCPs) == n_atoms`. These cover the checks that do see it.
"""

import json
import zipfile

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
