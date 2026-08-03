"""Tests for the per-vertical manifest builder."""

from __future__ import annotations

import os
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from qtaim_gen.source.utils.element_classes import (
    ACTINIDES,
    LANTHANIDES,
    SYMBOL_TO_Z,
    TRANSITION_METALS,
)
from qtaim_gen.source.utils.manifest import (
    SCHEMA,
    build_vertical,
    hill_formula,
    list_verticals,
    merge_manifests,
    process_job,
    walk_vertical,
    JobTarget,
)


FIXTURE_ROOT = Path(__file__).parent / "test_files" / "manifest_tests"


def _row_by_job(rows, job_id):
    return next(r for r in rows if r["job_id"] == job_id)


def _read_parquet_rows(path):
    return pq.read_table(path).to_pylist()


def test_element_classes_consistent():
    assert SYMBOL_TO_Z["Fe"] in TRANSITION_METALS
    assert SYMBOL_TO_Z["Eu"] in LANTHANIDES
    assert SYMBOL_TO_Z["U"] in ACTINIDES
    assert SYMBOL_TO_Z["C"] not in TRANSITION_METALS


def test_hill_formula():
    assert hill_formula(["O", "H", "H"]) == "H2O"
    assert hill_formula(["C", "C", "H", "H", "H", "H", "H", "H"]) == "C2H6"
    assert hill_formula(["Fe", "O", "O"]) == "FeO2"


def test_walk_finds_nested_jobs_and_skips_orig():
    targets = list(walk_vertical(str(FIXTURE_ROOT), "spice"))
    job_ids = sorted(os.path.basename(t.job_dir) for t in targets)
    # job_a contains a nested orca.inp under should_be_skipped/ that must not
    # be picked up — job dirs are terminal.
    assert job_ids == ["job_a", "job_b"]
    # job_b is two levels deeper than job_a
    a = next(t for t in targets if os.path.basename(t.job_dir) == "job_a")
    b = next(t for t in targets if os.path.basename(t.job_dir) == "job_b")
    assert a.rel_path.count(os.sep) + 1 == 2  # spice/job_a
    assert b.rel_path.count(os.sep) + 1 == 4  # spice/nested/sub/job_b


def test_process_job_happy_path():
    target = JobTarget(
        root=str(FIXTURE_ROOT),
        vertical="spice",
        job_dir=str(FIXTURE_ROOT / "spice" / "job_a"),
        rel_path="spice/job_a",
    )
    row, err = process_job(target)
    assert err is None
    assert row["read_status"] == "ok"
    assert row["charge"] == 0
    assert row["mult"] == 1
    assert row["spin"] == 0.0
    assert row["n_atoms"] == 3
    assert row["element_set"] == ["H", "O"]
    assert row["formula_hill"] == "H2O"
    assert row["heaviest_z"] == 8
    assert row["has_tm"] is False
    assert row["has_lanthanide"] is False
    assert row["net_charge_abs"] == 0


def test_process_job_transition_metal_high_spin():
    target = JobTarget(
        root=str(FIXTURE_ROOT),
        vertical="tm_react",
        job_dir=str(FIXTURE_ROOT / "tm_react" / "job_c"),
        rel_path="tm_react/job_c",
    )
    row, err = process_job(target)
    assert err is None
    assert row["charge"] == 2
    assert row["mult"] == 5
    assert row["spin"] == 2.0
    assert row["has_tm"] is True
    assert row["heaviest_z"] == SYMBOL_TO_Z["Fe"]


def test_process_job_negative_charge_lanthanide():
    target = JobTarget(
        root=str(FIXTURE_ROOT),
        vertical="lanth",
        job_dir=str(FIXTURE_ROOT / "lanth" / "job_d"),
        rel_path="lanth/job_d",
    )
    row, err = process_job(target)
    assert err is None
    assert row["charge"] == -2
    assert row["mult"] == 3
    assert row["net_charge_abs"] == 2
    assert row["has_lanthanide"] is True
    assert row["has_tm"] is False


def test_process_job_corrupt_inp():
    target = JobTarget(
        root=str(FIXTURE_ROOT),
        vertical="corrupt",
        job_dir=str(FIXTURE_ROOT / "corrupt" / "job_e"),
        rel_path="corrupt/job_e",
    )
    row, err = process_job(target)
    assert err is not None
    assert row["read_status"] in ("corrupt_inp", "parse_error")
    assert row["charge"] is None
    assert row["element_set"] is None


def test_process_job_missing_inp(tmp_path):
    job_dir = tmp_path / "no_inp"
    job_dir.mkdir()
    target = JobTarget(
        root=str(tmp_path), vertical="x", job_dir=str(job_dir), rel_path="x/no_inp"
    )
    row, err = process_job(target)
    assert err == "missing_inp"
    assert row["read_status"] == "missing_inp"


def test_build_vertical_writes_parquet(tmp_path):
    out_dir = tmp_path / "manifest"
    summary = build_vertical(
        root=str(FIXTURE_ROOT),
        vertical="spice",
        out_dir=str(out_dir),
        workers=1,
    )
    assert summary["ok"] == 2
    out = out_dir / "manifest_spice.parquet"
    assert out.exists()
    table = pq.read_table(out)
    assert table.schema == SCHEMA
    rows = table.to_pylist()
    assert len(rows) == 2
    job_a = _row_by_job(rows, "job_a")
    assert job_a["formula_hill"] == "H2O"


def test_build_vertical_records_corruption(tmp_path):
    out_dir = tmp_path / "manifest"
    summary = build_vertical(
        root=str(FIXTURE_ROOT),
        vertical="corrupt",
        out_dir=str(out_dir),
        workers=1,
    )
    total = sum(summary.values())
    assert total == 1
    assert summary["ok"] == 0
    failures = out_dir / "failures_corrupt.csv"
    assert failures.exists()


def test_build_vertical_empty_writes_empty_parquet(tmp_path):
    out_dir = tmp_path / "manifest"
    summary = build_vertical(
        root=str(FIXTURE_ROOT),
        vertical="empty",
        out_dir=str(out_dir),
        workers=1,
    )
    assert summary["ok"] == 0
    out = out_dir / "manifest_empty.parquet"
    assert out.exists()
    assert pq.read_table(out).num_rows == 0


def test_build_vertical_overwrite_guard(tmp_path):
    out_dir = tmp_path / "manifest"
    build_vertical(str(FIXTURE_ROOT), "spice", str(out_dir), workers=1)
    summary = build_vertical(str(FIXTURE_ROOT), "spice", str(out_dir), workers=1)
    assert summary == {"skipped": 1}
    summary = build_vertical(
        str(FIXTURE_ROOT), "spice", str(out_dir), workers=1, overwrite=True
    )
    assert summary["ok"] == 2


def test_merge_two_verticals(tmp_path):
    out_dir = tmp_path / "manifest"
    build_vertical(str(FIXTURE_ROOT), "spice", str(out_dir), workers=1)
    build_vertical(str(FIXTURE_ROOT), "tm_react", str(out_dir), workers=1)
    merged = merge_manifests(str(out_dir))
    rows = _read_parquet_rows(merged)
    assert len(rows) == 3
    verticals = sorted({r["vertical"] for r in rows})
    assert verticals == ["spice", "tm_react"]
    # No duplicate (vertical, rel_path) keys.
    keys = [(r["vertical"], r["rel_path"]) for r in rows]
    assert len(set(keys)) == len(keys)


def test_list_verticals_excludes_files(tmp_path):
    (tmp_path / "v1").mkdir()
    (tmp_path / "v2").mkdir()
    (tmp_path / "stray.txt").write_text("hi")
    assert list_verticals(str(tmp_path)) == ["v1", "v2"]
