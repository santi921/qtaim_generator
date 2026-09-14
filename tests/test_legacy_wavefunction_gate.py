"""Regression tests for the orca5.wfn deadlock.

A legacy orca5.wfn satisfied every suffix-based "is there a wavefunction?"
check while failing the canonical-name check that Multiwfn steps actually use.
The folder then skipped decompression, skipped conversion, and refused every
step with "No orca.wfn or orca.wfx ... cannot run qtaim" - forever, because
the only code that removes orca5.wfn sat behind the gate it was blocking.
"""

import logging
import os

import pytest

from qtaim_gen.source.core.omol import (
    _normalize_wavefunction_name,
    _wavefunction_present,
)


@pytest.fixture
def logger():
    return logging.getLogger("test_legacy_wavefunction_gate")


class TestNormalizeWavefunctionName:
    def test_canonical_is_returned_untouched(self, tmp_path, logger):
        (tmp_path / "orca.wfn").write_text("wfn\n")
        assert _normalize_wavefunction_name(str(tmp_path), logger) == str(
            tmp_path / "orca.wfn"
        )

    def test_legacy_wfn_is_renamed(self, tmp_path, logger):
        (tmp_path / "orca5.wfn").write_text("legacy\n")
        result = _normalize_wavefunction_name(str(tmp_path), logger)
        assert result == str(tmp_path / "orca.wfn")
        assert (tmp_path / "orca.wfn").read_text() == "legacy\n"
        assert not (tmp_path / "orca5.wfn").exists()

    def test_extension_is_preserved(self, tmp_path, logger):
        (tmp_path / "orca5.wfx").write_text("legacy\n")
        assert _normalize_wavefunction_name(str(tmp_path), logger) == str(
            tmp_path / "orca.wfx"
        )

    def test_empty_legacy_is_not_promoted(self, tmp_path, logger):
        (tmp_path / "orca5.wfn").write_text("")
        assert _normalize_wavefunction_name(str(tmp_path), logger) is None
        assert (tmp_path / "orca5.wfn").exists()

    def test_nothing_to_do(self, tmp_path, logger):
        (tmp_path / "orca.gbw").write_text("gbw\n")
        assert _normalize_wavefunction_name(str(tmp_path), logger) is None

    def test_canonical_in_generator_wins(self, tmp_path, logger):
        (tmp_path / "generator").mkdir()
        (tmp_path / "generator" / "orca.wfx").write_text("gen\n")
        (tmp_path / "orca5.wfn").write_text("legacy\n")
        assert _normalize_wavefunction_name(str(tmp_path), logger) == str(
            tmp_path / "generator" / "orca.wfx"
        )
        assert (tmp_path / "orca5.wfn").exists()


class TestCompressedSourceRetention:
    """process_folder used to delete orca.gbw.zstd0/orca.tar.zst even after a
    failed validation, leaving nothing for a retry to rebuild the
    wavefunction from."""

    def _folder(self, tmp_path):
        (tmp_path / "orca.inp").write_text("! B3LYP def2-SVP\n")
        (tmp_path / "orca.gbw.zstd0").write_text("compressed\n")
        (tmp_path / "orca.tar.zst").write_text("compressed\n")
        return tmp_path

    def _run(self, tmp_path, monkeypatch, validated):
        from qtaim_gen.source.core import workflow

        monkeypatch.setattr(
            workflow, "gbw_analysis", lambda **kwargs: validated
        )
        return workflow.process_folder(
            folder=str(tmp_path),
            multiwfn_cmd="/bin/true",
            orca_2mkl_cmd="/bin/true",
        )

    def test_failed_validation_keeps_sources(self, tmp_path, monkeypatch):
        folder = self._folder(tmp_path)
        result = self._run(folder, monkeypatch, validated=False)
        assert result["status"] == "ok"
        assert (folder / "orca.gbw.zstd0").exists()
        assert (folder / "orca.tar.zst").exists()

    def test_passed_validation_removes_sources(self, tmp_path, monkeypatch):
        folder = self._folder(tmp_path)
        self._run(folder, monkeypatch, validated=True)
        assert not (folder / "orca.gbw.zstd0").exists()
        assert not (folder / "orca.tar.zst").exists()


class TestPreprocessingGate:
    """The gate used to be len(files ending in .inp/.wfn/.wfx) < 2."""

    def _folder(self, tmp_path):
        (tmp_path / "orca.inp").write_text("! B3LYP def2-SVP\n")
        (tmp_path / "orca.gbw.zstd0").write_text("compressed\n")
        (tmp_path / "orca.tar.zst").write_text("compressed\n")
        return tmp_path

    def test_legacy_wfn_no_longer_blocks_extraction(self, tmp_path, logger):
        folder = self._folder(tmp_path)
        (folder / "orca5.wfn").write_text("legacy\n")

        # orca5.wfn + orca.inp used to read as "2 uncompressed files", which
        # skipped extraction. Now it is renamed into place instead.
        assert _normalize_wavefunction_name(str(folder), logger) is not None
        assert _wavefunction_present(str(folder))

    def test_orca5_gbw_kept_without_canonical_source(self, tmp_path, logger):
        folder = self._folder(tmp_path)
        (folder / "orca5.gbw").write_text("legacy gbw\n")
        # No canonical wavefunction and no orca.gbw: orca5.gbw is the only
        # wavefunction source and must survive the sweep.
        assert _normalize_wavefunction_name(str(folder), logger) is None
