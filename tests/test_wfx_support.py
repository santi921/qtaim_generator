"""Tests for .wfx wavefunction format support.

Tests the centralized find_wavefunction_file() utility and
the create_jobs() wfx mode for Multiwfn conversion.
"""

import os
import tempfile
import shutil
import pytest

from qtaim_gen.source.utils.io import (
    find_wavefunction_file,
    has_wavefunction_file,
    WFN_EXTENSIONS,
)


class TestFindWavefunctionFile:
    """Tests for find_wavefunction_file() and has_wavefunction_file()."""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()

    def teardown_method(self):
        shutil.rmtree(self.tmpdir)

    def test_wfn_only(self):
        """Finds .wfn when it's the only wavefunction file."""
        wfn_path = os.path.join(self.tmpdir, "orca.wfn")
        open(wfn_path, "w").close()

        result = find_wavefunction_file(self.tmpdir)
        assert result is not None
        assert result.endswith(".wfn")
        assert has_wavefunction_file(self.tmpdir) is True

    def test_wfx_only(self):
        """Finds .wfx when it's the only wavefunction file."""
        wfx_path = os.path.join(self.tmpdir, "orca.wfx")
        open(wfx_path, "w").close()

        result = find_wavefunction_file(self.tmpdir)
        assert result is not None
        assert result.endswith(".wfx")
        assert has_wavefunction_file(self.tmpdir) is True

    def test_both_prefers_wfx(self):
        """Prefers .wfx over .wfn when both exist."""
        open(os.path.join(self.tmpdir, "orca.wfn"), "w").close()
        open(os.path.join(self.tmpdir, "orca.wfx"), "w").close()

        result = find_wavefunction_file(self.tmpdir)
        assert result is not None
        assert result.endswith(".wfx")

    def test_neither(self):
        """Returns None when no wavefunction file exists."""
        # Create some other file
        open(os.path.join(self.tmpdir, "orca.gbw"), "w").close()

        result = find_wavefunction_file(self.tmpdir)
        assert result is None
        assert has_wavefunction_file(self.tmpdir) is False

    def test_empty_folder(self):
        """Returns None for empty folder."""
        result = find_wavefunction_file(self.tmpdir)
        assert result is None
        assert has_wavefunction_file(self.tmpdir) is False

    def test_extensions_order(self):
        """WFN_EXTENSIONS has .wfx first for priority."""
        assert WFN_EXTENSIONS[0] == ".wfx"
        assert WFN_EXTENSIONS[1] == ".wfn"


class TestCreateJobsWfxMode:
    """Tests for create_jobs() wfx parameter affecting convert.txt content."""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        # Create a minimal .gbw file to trigger conversion
        self.gbw_path = os.path.join(self.tmpdir, "orca.gbw")
        open(self.gbw_path, "w").close()
        # Create a minimal .inp file for check_spin
        self.inp_path = os.path.join(self.tmpdir, "orca.inp")
        with open(self.inp_path, "w") as f:
            f.write("! B3LYP def2-SVP\n\n*xyz 0 1\nC 0.0 0.0 0.0\nH 1.0 0.0 0.0\n*\n")

    def teardown_method(self):
        shutil.rmtree(self.tmpdir)

    def test_create_jobs_wfn_mode(self):
        """Default mode (wfx=False) produces option 4 targeting .wfn."""
        from qtaim_gen.source.core.omol import create_jobs

        create_jobs(
            folder=self.tmpdir,
            multiwfn_cmd="/usr/bin/Multiwfn",
            orca_2mkl_cmd="/usr/bin/orca_2mkl",
            separate=False,
            debug=True,  # minimal job set
            wfx=False,
        )

        convert_txt = os.path.join(self.tmpdir, "convert.txt")
        assert os.path.exists(convert_txt), "convert.txt should be created"
        with open(convert_txt) as f:
            content = f.read()

        # Should use Multiwfn option 4 for .wfn
        assert "100\n2\n4\n" in content, f"Expected option 4 for .wfn, got: {content}"
        assert "orca.wfn" in content, f"Expected .wfn filename, got: {content}"

    def test_create_jobs_wfx_mode(self):
        """With wfx=True, produces option 5 targeting .wfx."""
        from qtaim_gen.source.core.omol import create_jobs

        create_jobs(
            folder=self.tmpdir,
            multiwfn_cmd="/usr/bin/Multiwfn",
            orca_2mkl_cmd="/usr/bin/orca_2mkl",
            separate=False,
            debug=True,  # minimal job set
            wfx=True,
        )

        convert_txt = os.path.join(self.tmpdir, "convert.txt")
        assert os.path.exists(convert_txt), "convert.txt should be created"
        with open(convert_txt) as f:
            content = f.read()

        # Should use Multiwfn option 5 for .wfx
        assert "100\n2\n4\n" in content, f"Expected option 5 for .wfx, got: {content}"
        assert "orca.wfx" in content, f"Expected .wfx filename, got: {content}"

    def test_create_jobs_skips_conversion_when_wfn_exists(self):
        """No convert.txt when .wfn already present."""
        from qtaim_gen.source.core.omol import create_jobs

        # Create a .wfn file so conversion is skipped
        open(os.path.join(self.tmpdir, "orca.wfn"), "w").close()

        create_jobs(
            folder=self.tmpdir,
            multiwfn_cmd="/usr/bin/Multiwfn",
            orca_2mkl_cmd="/usr/bin/orca_2mkl",
            separate=False,
            debug=True,
            wfx=False,
        )

        convert_txt = os.path.join(self.tmpdir, "convert.txt")
        assert not os.path.exists(convert_txt), "convert.txt should NOT be created when .wfn exists"

    def test_create_jobs_skips_conversion_when_wfx_exists(self):
        """No convert.txt when .wfx already present."""
        from qtaim_gen.source.core.omol import create_jobs

        # Create a .wfx file so conversion is skipped
        open(os.path.join(self.tmpdir, "orca.wfx"), "w").close()

        create_jobs(
            folder=self.tmpdir,
            multiwfn_cmd="/usr/bin/Multiwfn",
            orca_2mkl_cmd="/usr/bin/orca_2mkl",
            separate=False,
            debug=True,
            wfx=True,
        )

        convert_txt = os.path.join(self.tmpdir, "convert.txt")
        assert not os.path.exists(convert_txt), "convert.txt should NOT be created when .wfx exists"


class TestCleanJobsWfx:
    """Tests for clean_jobs() handling both .wfn and .wfx."""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        # Create a minimal .inp file for check_spin
        with open(os.path.join(self.tmpdir, "orca.inp"), "w") as f:
            f.write("! B3LYP def2-SVP\n\n*xyz 0 1\nC 0.0 0.0 0.0\nH 1.0 0.0 0.0\n*\n")

    def teardown_method(self):
        shutil.rmtree(self.tmpdir)

    def test_cleans_wfn_when_gbw_exists(self):
        """clean_jobs removes .wfn when .gbw exists."""
        from qtaim_gen.source.core.omol import clean_jobs

        open(os.path.join(self.tmpdir, "orca.gbw"), "w").close()
        open(os.path.join(self.tmpdir, "orca.wfn"), "w").close()

        clean_jobs(folder=self.tmpdir, separate=False, move_results=False)

        assert not os.path.exists(os.path.join(self.tmpdir, "orca.wfn"))

    def test_cleans_wfx_when_gbw_exists(self):
        """clean_jobs removes .wfx when .gbw exists."""
        from qtaim_gen.source.core.omol import clean_jobs

        open(os.path.join(self.tmpdir, "orca.gbw"), "w").close()
        open(os.path.join(self.tmpdir, "orca.wfx"), "w").close()

        clean_jobs(folder=self.tmpdir, separate=False, move_results=False)

        assert not os.path.exists(os.path.join(self.tmpdir, "orca.wfx"))

    def test_preserves_wfx_when_no_gbw(self):
        """clean_jobs preserves .wfx when no .gbw exists."""
        from qtaim_gen.source.core.omol import clean_jobs

        open(os.path.join(self.tmpdir, "orca.wfx"), "w").close()

        clean_jobs(folder=self.tmpdir, separate=False, move_results=False)

        assert os.path.exists(os.path.join(self.tmpdir, "orca.wfx"))
