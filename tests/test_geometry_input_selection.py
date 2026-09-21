"""Geometry-input selection and the parser errors behind it.

ORCA leaves orca.property.inp in every job folder and the pipeline never
deletes it. It has no "* xyz" block, so it is not a geometry input. Selection
used to be os.listdir()[0], which is arbitrary and shifts whenever a file is
added to the folder, so an unrelated write could swap the parsed molecule or
crash the parser with UnboundLocalError.
"""
import os

import pytest

from qtaim_gen.source.core.parse_qtaim import (
    dft_inp_to_dict,
    get_spin_charge_from_orca_inp,
    orca_inp_to_dict,
)
from qtaim_gen.source.utils.validation import (
    geometry_input_candidates,
    get_charge_spin_n_atoms_from_folder,
)

TEST_FILES = os.path.join(os.path.dirname(__file__), "test_files")
REAL_INP = os.path.join(TEST_FILES, "lmdb_tests", "orca6_rks", "orca.inp")

PROPERTY_INP = """\
-------------------------------------------------------------
----------------------- !PROPERTIES! ------------------------
-------------------------------------------------------------
# -----------------------------------------------------------
$ SCF_Energy
   description: The SCF energy
   geom. index: 1
   prop. index: 1
        SCF Energy:     -1234.5678901234
# -----------------------------------------------------------
"""


@pytest.fixture
def job_folder(tmp_path):
    """A job folder as the pipeline leaves it: real input plus the decoy."""
    with open(REAL_INP) as f:
        real = f.read()
    (tmp_path / "orca.inp").write_text(real)
    (tmp_path / "orca.property.inp").write_text(PROPERTY_INP)
    return tmp_path


def test_property_inp_is_never_a_candidate(job_folder):
    assert geometry_input_candidates(str(job_folder)) == ["orca.inp"]


def test_convert_in_is_never_a_candidate(job_folder):
    (job_folder / "convert.in").write_text("2\norca.wfn\n")
    assert geometry_input_candidates(str(job_folder)) == ["orca.inp"]


def test_canonical_input_wins_over_alphabetically_earlier_name(job_folder):
    (job_folder / "aaa_stale.inp").write_text("! HF\n* xyz 0 1\nH 0 0 0\n*\n")
    assert geometry_input_candidates(str(job_folder))[0] == "orca.inp"


def test_selection_is_stable_when_unrelated_files_appear(job_folder):
    before = get_charge_spin_n_atoms_from_folder(str(job_folder))
    for name in ("hirshfeld.out", "orca.tar.zst", "timings.json", "zzz.out"):
        (job_folder / name).write_text("x")
    after = get_charge_spin_n_atoms_from_folder(str(job_folder))
    assert before == after
    assert len(after["mol"]) > 0


def test_folder_with_only_the_decoy_reports_failure(tmp_path):
    (tmp_path / "orca.property.inp").write_text(PROPERTY_INP)
    assert get_charge_spin_n_atoms_from_folder(str(tmp_path)) is False


def test_unparsable_candidate_falls_through_to_the_good_one(job_folder):
    # a truncated input sorts before orca.inp but must not win
    (job_folder / "aaa_truncated.inp").write_text("! B3LYP def2-SVP\n")
    parsed = get_charge_spin_n_atoms_from_folder(str(job_folder))
    assert parsed is not False
    assert len(parsed["mol"]) > 0


def test_get_spin_charge_names_the_file_instead_of_unbound_local(tmp_path):
    bad = tmp_path / "orca.property.inp"
    bad.write_text(PROPERTY_INP)
    with pytest.raises(ValueError, match="no '\\* xyz' coordinate block"):
        get_spin_charge_from_orca_inp(str(bad))


def test_dft_inp_to_dict_raises_on_missing_xyz_block(tmp_path):
    bad = tmp_path / "orca.property.inp"
    bad.write_text(PROPERTY_INP)
    with pytest.raises(ValueError, match="no '\\* xyz' coordinate block"):
        dft_inp_to_dict(str(bad), parse_charge_spin=True)


def test_orca_inp_to_dict_raises_on_missing_xyz_block(tmp_path):
    bad = tmp_path / "orca.property.inp"
    bad.write_text(PROPERTY_INP)
    with pytest.raises(ValueError, match="no '\\* xyz' coordinate block"):
        orca_inp_to_dict(str(bad))


def test_orca_inp_to_dict_raises_on_unterminated_block(tmp_path):
    bad = tmp_path / "orca.inp"
    bad.write_text("! B3LYP def2-SVP\n* xyz 0 1\nH 0.0 0.0 0.0\n")
    with pytest.raises(ValueError, match="unterminated"):
        orca_inp_to_dict(str(bad))


def test_real_input_still_parses(job_folder):
    parsed = get_charge_spin_n_atoms_from_folder(str(job_folder))
    direct = dft_inp_to_dict(REAL_INP, parse_charge_spin=True)
    assert parsed == direct
