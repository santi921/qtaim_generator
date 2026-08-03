"""Tests for checkpointing in get_folders_from_file / refine_list_of_jobs."""

import os

import pytest

import qtaim_gen.source.utils.io as io_mod
from qtaim_gen.source.utils.io import get_folders_from_file
from qtaim_gen.source.scripts.helpers.refine_list_of_jobs import main as refine_main


@pytest.fixture
def job_folders(tmp_path):
    """Create job folders: keep_* fail validation (need rerun), skip_* pass."""
    folders = []
    for name in ["keep_a", "keep_b", "skip_c", "skip_d", "keep_e"]:
        d = tmp_path / name
        d.mkdir()
        folders.append(str(d))
    job_file = tmp_path / "jobs.txt"
    job_file.write_text("\n".join(folders) + "\n")
    return job_file, folders


@pytest.fixture
def fake_validation(monkeypatch):
    """validation_checks returns True (done) for skip_* folders, False for keep_*."""
    calls = []

    def _fake(folder_outputs, **kwargs):
        calls.append(folder_outputs)
        return os.path.basename(folder_outputs).startswith("skip_")

    monkeypatch.setattr(io_mod, "validation_checks", _fake)
    return calls


def read_checkpoint(path):
    decisions = {}
    with open(path) as f:
        for line in f:
            folder, verdict = line.rstrip("\n").split("\t")
            decisions[folder] = verdict
    return decisions


def test_fresh_run_writes_checkpoint(job_folders, fake_validation, tmp_path):
    job_file, folders = job_folders
    ckpt = tmp_path / "run.ckpt"

    result = get_folders_from_file(
        str(job_file),
        num_folders=len(folders),
        pre_validate=True,
        max_workers=1,
        checkpoint_path=str(ckpt),
    )

    keeps = {f for f in folders if os.path.basename(f).startswith("keep_")}
    assert set(result) == keeps

    decisions = read_checkpoint(ckpt)
    assert set(decisions) == set(folders)
    for folder, verdict in decisions.items():
        expected = "KEEP" if os.path.basename(folder).startswith("keep_") else "SKIP"
        assert verdict == expected


def test_resume_skips_decided_folders(job_folders, fake_validation, tmp_path):
    job_file, folders = job_folders
    ckpt = tmp_path / "run.ckpt"

    keep_a = [f for f in folders if f.endswith("keep_a")][0]
    skip_c = [f for f in folders if f.endswith("skip_c")][0]
    ckpt.write_text(f"{keep_a}\tKEEP\n{skip_c}\tSKIP\n")

    result = get_folders_from_file(
        str(job_file),
        num_folders=len(folders),
        pre_validate=True,
        max_workers=1,
        checkpoint_path=str(ckpt),
    )

    # decided folders were not revalidated
    assert keep_a not in fake_validation
    assert skip_c not in fake_validation
    # checkpointed KEEP is in the result, checkpointed SKIP is not
    keeps = {f for f in folders if os.path.basename(f).startswith("keep_")}
    assert set(result) == keeps
    # checkpoint now covers all folders
    assert set(read_checkpoint(ckpt)) == set(folders)


def test_fully_decided_checkpoint_short_circuits(job_folders, fake_validation, tmp_path):
    job_file, folders = job_folders
    ckpt = tmp_path / "run.ckpt"
    lines = []
    for f in folders:
        verdict = "KEEP" if os.path.basename(f).startswith("keep_") else "SKIP"
        lines.append(f"{f}\t{verdict}")
    ckpt.write_text("\n".join(lines) + "\n")

    result = get_folders_from_file(
        str(job_file),
        num_folders=len(folders),
        pre_validate=True,
        max_workers=1,
        checkpoint_path=str(ckpt),
    )

    assert fake_validation == []
    keeps = {f for f in folders if os.path.basename(f).startswith("keep_")}
    assert set(result) == keeps


def test_no_checkpoint_path_unchanged(job_folders, fake_validation, tmp_path):
    job_file, folders = job_folders

    result = get_folders_from_file(
        str(job_file),
        num_folders=len(folders),
        pre_validate=True,
        max_workers=1,
    )

    keeps = {f for f in folders if os.path.basename(f).startswith("keep_")}
    assert set(result) == keeps
    assert list(tmp_path.glob("*.ckpt")) == []


def test_cli_rejects_checkpoint_with_sampling(job_folders, fake_validation, tmp_path):
    job_file, folders = job_folders
    ckpt = tmp_path / "run.ckpt"

    rc = refine_main(
        [
            "--job_file",
            str(job_file),
            "--num_folders",
            "2",
            "--checkpoint_file",
            str(ckpt),
            "--quiet",
        ]
    )
    assert rc == 2
    assert not ckpt.exists()
