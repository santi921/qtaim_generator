import pytest

from qtaim_gen.source.scripts import full_runner


def test_folder_runs_only_that_folder(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        full_runner, "gbw_analysis", lambda **kw: calls.append(kw["folder"])
    )
    job = tmp_path / "job"
    job.mkdir()

    full_runner.main(["--folder", str(job), "--num_jobs", "50"])

    assert calls == [str(job)]


def test_folder_and_job_file_are_exclusive(tmp_path):
    with pytest.raises(SystemExit):
        full_runner.main(
            ["--folder", str(tmp_path), "--job_file", str(tmp_path / "jobs.txt")]
        )
