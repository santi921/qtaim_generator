"""Tests for requalify-qtaim (scripts/helpers/requalify_qtaim_folders.py)."""

import json
import sys
import zipfile

import pytest

from qtaim_gen.source.core import parse_multiwfn
from qtaim_gen.source.scripts.helpers import requalify_qtaim_folders as rq

COUNT_LINE = " Number of (3,-1) CPs:     {n}\n"
EXPORT_LINE = " Done! The results have been outputted to CPprop.txt in current folder\n"
INP = "! wB97M-V\n*xyz 0 1\nH 0.0 0.0 0.0\nH 0.0 0.0 0.74\nO 0.0 0.9 0.0\n*\n"
REPARSED = {"0": {"density_all": 0.40}, "1": {"density_all": 0.41}, "2": {"density_all": 0.30},
            "0_2": {"density_all": 0.25}, "1_2": {"density_all": 0.26}}


def _qtaim_out(n_bcp, export=True):
    from qtaim_gen.source.utils import validation
    text = COUNT_LINE.format(n=n_bcp)
    assert validation.QTAIM_COUNT_PATTERN.findall(text), "fixture count line no longer matches the parser"
    return text + (EXPORT_LINE if export else "")


def _cpprop_text(n_ncp=3, n_bcp=2):
    """Well-formed CPprop.txt blocks (the parse itself is mocked; validation.cpprop_integrity
    reads these): n_ncp (3,-3) then n_bcp (3,-1) blocks, numbered from 1."""
    blocks = []
    for n in range(1, n_ncp + n_bcp + 1):
        kind, eig = ("(3,-3)", "-0.3E+00 -0.2E+00 -0.1E+00") if n <= n_ncp else ("(3,-1)", "-0.3E+00 -0.2E+00  0.1E+00")
        blocks.append(
            f" ----------------   CP{n:>6},     Type {kind}   ----------------\n"
            " Position (Bohr):      0.000000000000    0.000000000000    0.000000000000\n"
            " Density of all electrons:  0.1000000000E+00\n"
            " Norm of gradient is:  0.1000000000E-14\n"
            f" Eigenvalues of Hessian: {eig}\n"
            " Determinant of Hessian:  0.6000000000E-02\n"
        )
    return "".join(blocks)


def _job(tmp_path, stored, n_bcp=2, cpprop=True, qtaim_out=True, export=True, inp=True):
    job = tmp_path / "job"
    gen = job / "generator"
    gen.mkdir(parents=True)
    (gen / "qtaim.json").write_text(json.dumps(stored))
    with zipfile.ZipFile(gen / "out_files.zip", "w") as zf:
        if cpprop:
            zf.writestr("CPprop.txt", cpprop if isinstance(cpprop, str) else _cpprop_text(n_bcp=n_bcp))
        if qtaim_out:
            zf.writestr("qtaim.out", _qtaim_out(n_bcp, export))
    if inp:
        (job / "orca.inp").write_text(INP)
    return job


@pytest.fixture
def reparse(monkeypatch):
    calls = []

    def fake(cprop_file, inp_loc, orca_tf=False):
        calls.append((cprop_file, inp_loc, orca_tf))
        out = {int(k) if "_" not in k else k: v for k, v in REPARSED.items()}
        return out
    monkeypatch.setattr(parse_multiwfn, "parse_qtaim", fake)
    return calls


def _stored(job):
    return json.loads((job / "generator" / "qtaim.json").read_text())


class TestProcessFolder:

    def test_mislabeled_pairs_are_replaced(self, tmp_path, reparse):
        stale = dict(REPARSED, **{"2_2": {"density_all": 0.25}, "0_1": {"density_all": 0.26}})
        job = _job(tmp_path, stale)
        r = rq.process_folder(str(job), None, None, dry_run=False)
        assert r["status"] == rq.STATUS_REPLACED
        assert r["only_stored"] == ["0_1", "2_2"] and r["only_reparsed"] == []
        stored = _stored(job)
        # the rewrite regenerates the provenance block parse_multiwfn writes
        assert set(stored.pop("_meta")) == {"poincare_hopf", "cp_counts", "qtaim_search"}
        assert stored == REPARSED
        assert reparse[0][2] is True
        assert not (job / ".processing.lock").exists()

    def test_damaged_cpprop_is_never_used(self, tmp_path, reparse):
        # a truncated archive still parses, minus its tail, and would replace a sound record
        stale = dict(REPARSED, **{"2_2": {"density_all": 0.25}})
        text = _cpprop_text()
        job = _job(tmp_path, stale, cpprop=text[: text.rindex(" Eigenvalues")])
        r = rq.process_folder(str(job), None, None, dry_run=False)
        assert r["status"] == rq.STATUS_DAMAGED_CPPROP
        assert _stored(job) == stale and reparse == []

    def test_matching_record_is_left_alone(self, tmp_path, reparse):
        job = _job(tmp_path, REPARSED)
        assert rq.process_folder(str(job), None, None, dry_run=False)["status"] == rq.STATUS_SAME

    def test_dry_run_writes_nothing_and_takes_no_lock(self, tmp_path, reparse):
        stale = dict(REPARSED, **{"2_2": {"density_all": 0.25}})
        job = _job(tmp_path, stale)
        (job / ".processing.lock").write_text("other job")
        r = rq.process_folder(str(job), None, None, dry_run=True)
        assert r["status"] == rq.STATUS_WOULD_REPLACE and _stored(job) == stale

    def test_record_from_a_different_run_is_left_alone(self, tmp_path, reparse):
        other = dict(REPARSED, **{"0": {"density_all": 0.99}, "2_2": {"density_all": 0.25}})
        job = _job(tmp_path, other)
        assert rq.process_folder(str(job), None, None, dry_run=False)["status"] == rq.STATUS_DIFFERENT_RUN
        assert _stored(job) == other

    def test_extra_fields_in_the_stored_record_do_not_count_as_a_different_run(self, tmp_path, reparse):
        stale = {k: dict(v, legacy_field=1.0) for k, v in REPARSED.items()}
        stale["2_2"] = {"density_all": 0.25}
        job = _job(tmp_path, stale)
        assert rq.process_folder(str(job), None, None, dry_run=False)["status"] == rq.STATUS_REPLACED

    def test_more_bcps_than_reported_is_refused(self, tmp_path, reparse):
        job = _job(tmp_path, dict(REPARSED, **{"2_2": {}}), n_bcp=1)
        assert rq.process_folder(str(job), None, None, dry_run=False)["status"] == rq.STATUS_COUNT_MISMATCH

    def test_unfinished_export_is_no_provenance(self, tmp_path, reparse):
        job = _job(tmp_path, REPARSED, export=False)
        assert rq.process_folder(str(job), None, None, dry_run=False)["status"] == rq.STATUS_NO_PROVENANCE

    def test_no_qtaim_out_is_no_provenance(self, tmp_path, reparse):
        job = _job(tmp_path, REPARSED, qtaim_out=False)
        assert rq.process_folder(str(job), None, None, dry_run=False)["status"] == rq.STATUS_NO_PROVENANCE

    def test_no_cpprop(self, tmp_path, reparse):
        job = _job(tmp_path, REPARSED, cpprop=False)
        assert rq.process_folder(str(job), None, None, dry_run=False)["status"] == rq.STATUS_NO_CPPROP

    def test_atom_count_mismatch(self, tmp_path, reparse):
        job = _job(tmp_path, REPARSED)
        (job / "orca.inp").write_text("! x\n*xyz 0 1\nH 0 0 0\nH 0 0 1\nO 0 1 0\nH 1 1 1\n*\n")
        assert rq.process_folder(str(job), None, None, dry_run=False)["status"] == rq.STATUS_NCP_MISMATCH

    def test_input_folder_supplies_the_geometry(self, tmp_path, reparse):
        inputs, results = tmp_path / "in", tmp_path / "res"
        (inputs / "v" / "job").mkdir(parents=True)
        (inputs / "v" / "job" / "orca.inp").write_text(INP)
        job = _job(results / "v", dict(REPARSED, **{"2_2": {}}), inp=False)
        r = rq.process_folder(str(inputs / "v" / "job"), str(inputs), str(results), dry_run=False)
        assert r["folder"] == str(job) and r["status"] == rq.STATUS_REPLACED

    def test_locked_and_missing(self, tmp_path, reparse):
        job = _job(tmp_path, REPARSED)
        (job / ".processing.lock").write_text("other job")
        assert rq.process_folder(str(job), None, None, dry_run=False)["status"] == rq.STATUS_LOCKED
        assert rq.process_folder(str(tmp_path / "gone"), None, None, dry_run=False)["status"] == rq.STATUS_MISSING


class TestMain:

    def test_remaining_lists_only_folders_needing_a_rerun(self, tmp_path, monkeypatch, reparse):
        fixed = _job(tmp_path / "a", dict(REPARSED, **{"2_2": {}}))
        unverifiable = _job(tmp_path / "b", REPARSED, qtaim_out=False)
        lst = tmp_path / "jobs.txt"
        lst.write_text(f"{fixed}\n{unverifiable}\n")
        report, left = tmp_path / "rep.json", tmp_path / "left.txt"
        monkeypatch.setattr(sys, "argv", ["requalify-qtaim", "--folder_list", str(lst), "--workers", "1",
                                          "--report", str(report), "--list_remaining", str(left)])
        assert rq.main() == 0
        agg = json.loads(report.read_text())["aggregate"]
        assert agg[rq.STATUS_REPLACED] == 1 and agg[rq.STATUS_NO_PROVENANCE] == 1
        assert left.read_text().split() == [str(unverifiable)]
