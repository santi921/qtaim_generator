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
