"""Tests for the reparse-orca-json driver (scripts/helpers/reparse_orca_json.py)."""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from qtaim_gen.source.core.parse_orca import ORCA_PARSER_VERSION
from qtaim_gen.source.scripts.helpers import reparse_orca_json as rj

FIXTURE_UKS = Path(__file__).parent / "test_files" / "orca_outs" / "minimal_uks.out"
V1_ORCA_JSON = {"final_energy_eh": -75.1, "homo_eh": -0.4, "n_electrons": 5.0}


def _make_job(root: Path, name: str, with_out: bool = True, layout: str = "flat") -> Path:
    job = root / name
    job.mkdir(parents=True)
    if with_out:
        shutil.copy(FIXTURE_UKS, job / "orca.out")
    target = job / "generator" if layout == "generator" else job
    target.mkdir(exist_ok=True)
    with open(target / "orca.json", "w") as f:
        json.dump(V1_ORCA_JSON, f)
    with open(target / "charge.json", "w") as f:
        json.dump({"mulliken": {"charge": {"0": -0.3, "1": 0.3}}}, f)
    return job


def _load(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _run(folder: Path, **kw) -> dict:
    defaults = dict(move_files=False, dry_run=False, min_version=ORCA_PARSER_VERSION, force=False)
    defaults.update(kw)
    return rj.process_folder(str(folder), **defaults)


class TestProcessFolder:

    def test_stale_folder_is_reparsed(self, tmp_path):
        job = _make_job(tmp_path, "job")
        r = _run(job)
        assert r["status"] == rj.STATUS_REPARSED
        assert r["version_before"] == 1
        assert r["version_after"] == ORCA_PARSER_VERSION
        assert r["source"] == "folder_out"
        d = _load(job / "orca.json")
        assert d["orca_parser_version"] == ORCA_PARSER_VERSION
        assert d["n_electrons"] == pytest.approx(9.0)
        assert d["hf_type"] == "UHF"
        # on-disk orca.out is left alone
        assert (job / "orca.out").is_file()
        # merge into charge.json happened
        assert "mulliken_orca" in _load(job / "charge.json")

    def test_current_folder_is_skipped(self, tmp_path):
        job = _make_job(tmp_path, "job")
        assert _run(job)["status"] == rj.STATUS_REPARSED
        mtime = os.path.getmtime(job / "orca.json")
        r = _run(job)
        assert r["status"] == rj.STATUS_CURRENT
        assert os.path.getmtime(job / "orca.json") == mtime

    def test_force_reparses_current(self, tmp_path):
        job = _make_job(tmp_path, "job")
        _run(job)
        assert _run(job, force=True)["status"] == rj.STATUS_REPARSED

    def test_no_source(self, tmp_path):
        job = _make_job(tmp_path, "job", with_out=False)
        r = _run(job)
        assert r["status"] == rj.STATUS_NO_SOURCE
        assert _load(job / "orca.json") == V1_ORCA_JSON

    def test_dry_run_writes_nothing(self, tmp_path):
        job = _make_job(tmp_path, "job")
        r = _run(job, dry_run=True)
        assert r["status"] == rj.STATUS_WOULD_REPARSE
        assert r["source"] == "folder_out"
        assert _load(job / "orca.json") == V1_ORCA_JSON

    def test_missing_orca_json_counts_as_stale(self, tmp_path):
        job = tmp_path / "job"
        job.mkdir()
        shutil.copy(FIXTURE_UKS, job / "orca.out")
        r = _run(job)
        assert r["version_before"] is None
        assert r["status"] == rj.STATUS_REPARSED

    def test_generator_layout_replaces_stale_copy(self, tmp_path):
        job = _make_job(tmp_path, "job", layout="generator")
        r = _run(job, move_files=True)
        assert r["status"] == rj.STATUS_REPARSED
        assert not (job / "orca.json").exists()
        assert _load(job / "generator" / "orca.json")["orca_parser_version"] == ORCA_PARSER_VERSION
        assert "mulliken_orca" in _load(job / "generator" / "charge.json")

    def test_truncated_out_is_partial(self, tmp_path):
        job = _make_job(tmp_path, "job", with_out=False)
        text = FIXTURE_UKS.read_text()
        with open(job / "orca.out", "w") as f:
            f.write(text[: text.index("FINAL SINGLE POINT ENERGY")])
        r = _run(job)
        assert r["status"] == rj.STATUS_PARTIAL
        assert _load(job / "orca.json")["orca_parser_version"] == ORCA_PARSER_VERSION


class TestSourceRoot:

    def test_source_out_is_staged_and_removed(self, tmp_path):
        results = tmp_path / "results"
        source = tmp_path / "source"
        job = _make_job(results, "vert/job", with_out=False)
        src_job = source / "vert" / "job"
        src_job.mkdir(parents=True)
        shutil.copy(FIXTURE_UKS, src_job / "orca.out")

        r = _run(job, root_dir=str(results), source_root=str(source))
        assert r["status"] == rj.STATUS_REPARSED
        assert r["source"] == "source_out"
        assert not (job / "orca.out").exists()
        assert (src_job / "orca.out").is_file()
        assert _load(job / "orca.json")["n_electrons"] == pytest.approx(9.0)

    def test_source_archive_is_extracted_and_removed(self, tmp_path):
        if shutil.which("tar") is None or shutil.which("zstd") is None:
            pytest.skip("tar/zstd not available")
        results = tmp_path / "results"
        source = tmp_path / "source"
        job = _make_job(results, "vert/job", with_out=False)
        src_job = source / "vert" / "job"
        src_job.mkdir(parents=True)
        shutil.copy(FIXTURE_UKS, src_job / "orca.out")
        proc = subprocess.run(
            ["tar", "--zstd", "-cf", "orca.tar.zst", "orca.out"], cwd=src_job, capture_output=True
        )
        if proc.returncode != 0:
            pytest.skip(f"tar --zstd unavailable: {proc.stderr.decode(errors='replace')[:200]}")
        os.remove(src_job / "orca.out")

        r = _run(job, root_dir=str(results), source_root=str(source))
        assert r["status"] == rj.STATUS_REPARSED
        assert r["source"] == "source_archive"
        assert not (job / "orca.out").exists()
        assert not (job / "orca.tar.zst").exists()
        assert (src_job / "orca.tar.zst").is_file()
        assert _load(job / "orca.json")["orca_parser_version"] == ORCA_PARSER_VERSION

    def test_source_root_without_mirror_is_no_source(self, tmp_path):
        results = tmp_path / "results"
        job = _make_job(results, "vert/job", with_out=False)
        r = _run(job, root_dir=str(results), source_root=str(tmp_path / "empty"))
        assert r["status"] == rj.STATUS_NO_SOURCE

    def test_dry_run_reports_source_kind(self, tmp_path):
        results = tmp_path / "results"
        source = tmp_path / "source"
        job = _make_job(results, "vert/job", with_out=False)
        src_job = source / "vert" / "job"
        src_job.mkdir(parents=True)
        shutil.copy(FIXTURE_UKS, src_job / "orca.out")
        r = _run(job, dry_run=True, root_dir=str(results), source_root=str(source))
        assert r["status"] == rj.STATUS_WOULD_REPARSE
        assert r["source"] == "source_out"
        assert not (job / "orca.out").exists()


class TestDiscovery:

    def test_folder_list_skips_comments_and_missing(self, tmp_path):
        a = _make_job(tmp_path, "a")
        lst = tmp_path / "jobs.txt"
        lst.write_text(f"# comment\n\n{a}\n{tmp_path / 'missing'}\n")
        assert rj.discover_folders(None, str(lst)) == [str(a)]

    def test_root_dir_finds_folders_with_orca_artifacts(self, tmp_path):
        _make_job(tmp_path, "with_json", with_out=False)
        _make_job(tmp_path, "with_out")
        (tmp_path / "unrelated").mkdir()
        found = rj.discover_folders(str(tmp_path), None)
        assert sorted(os.path.basename(f) for f in found) == ["with_json", "with_out"]
