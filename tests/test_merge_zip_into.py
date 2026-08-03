"""Tests for merge_zip_into: preserve richest out_files.zip across reruns."""

import os
import tempfile
import zipfile

import pytest

from qtaim_gen.source.utils.io import merge_zip_into


def _make_zip(path: str, entries: dict) -> None:
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in entries.items():
            zf.writestr(name, data)


def _read_zip(path: str) -> dict:
    out = {}
    with zipfile.ZipFile(path, "r") as zf:
        for name in zf.namelist():
            out[name] = zf.read(name)
    return out


def test_move_when_dest_missing():
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "src.zip")
        dest = os.path.join(tmp, "sub", "dest.zip")
        _make_zip(src, {"a.out": b"hello"})

        merge_zip_into(src, dest)

        assert not os.path.exists(src)
        assert os.path.exists(dest)
        assert _read_zip(dest) == {"a.out": b"hello"}


def test_merge_adds_missing_entries():
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "src.zip")
        dest = os.path.join(tmp, "dest.zip")
        _make_zip(dest, {"a.out": b"AAA"})
        _make_zip(src, {"b.out": b"BBB", "c.out": b"CCC"})

        merge_zip_into(src, dest)

        assert not os.path.exists(src)
        assert _read_zip(dest) == {
            "a.out": b"AAA",
            "b.out": b"BBB",
            "c.out": b"CCC",
        }


def test_collision_prefers_larger():
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "src.zip")
        dest = os.path.join(tmp, "dest.zip")
        _make_zip(dest, {"adch.out": b"short", "cm5.out": b"rich cm5 content"})
        _make_zip(src, {"adch.out": b"richer adch content here", "cm5.out": b"x"})

        merge_zip_into(src, dest)

        merged = _read_zip(dest)
        assert merged["adch.out"] == b"richer adch content here"
        assert merged["cm5.out"] == b"rich cm5 content"


def test_collision_equal_size_keeps_existing():
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "src.zip")
        dest = os.path.join(tmp, "dest.zip")
        _make_zip(dest, {"a.out": b"XXXXX"})
        _make_zip(src, {"a.out": b"YYYYY"})

        merge_zip_into(src, dest)

        assert _read_zip(dest) == {"a.out": b"XXXXX"}


def test_missing_src_is_noop():
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "missing.zip")
        dest = os.path.join(tmp, "dest.zip")
        _make_zip(dest, {"a.out": b"AAA"})

        merge_zip_into(src, dest)

        assert _read_zip(dest) == {"a.out": b"AAA"}


def test_dest_corrupt_raises_and_preserves_dest():
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "src.zip")
        dest = os.path.join(tmp, "dest.zip")
        _make_zip(src, {"a.out": b"AAA"})
        with open(dest, "wb") as f:
            f.write(b"not a zip file")

        with pytest.raises(zipfile.BadZipFile):
            merge_zip_into(src, dest)

        assert os.path.exists(src)
        with open(dest, "rb") as f:
            assert f.read() == b"not a zip file"
        assert not os.path.exists(dest + ".merge.tmp")


def test_simulated_rerun_preserves_original_entries():
    # Simulate first full run producing many .out files, then a parse_only
    # rerun producing only a subset - merged zip should retain everything.
    with tempfile.TemporaryDirectory() as tmp:
        dest = os.path.join(tmp, "out_files.zip")
        first_run = {
            "adch.out": b"first adch full output",
            "cm5.out": b"first cm5 full output",
            "hirshfeld.out": b"first hirshfeld full output",
            "fuzzy_full.out": b"first fuzzy full output",
        }
        _make_zip(dest, first_run)

        src = os.path.join(tmp, "out_files.zip.new")
        rerun = {"adch.out": b"short rerun"}
        _make_zip(src, rerun)

        merge_zip_into(src, dest)

        merged = _read_zip(dest)
        assert set(merged) == set(first_run)
        for k, v in first_run.items():
            assert merged[k] == v
