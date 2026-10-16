"""CHELPG radii for actinides: a radius file (Batsanov Th/U, CHELPG_ACTINIDE_DEFAULT
otherwise) for molecules with an actinide, Multiwfn's stock input for the rest."""

import logging

from qtaim_gen.source.data.multiwfn import (
    CHELPG_INPUT,
    CHELPG_INPUT_RADII_FILE,
    CHELPG_RADII_FILE,
    charge_data_dict,
    chelpg_radii_file,
    chelpg_radius,
)
from qtaim_gen.source.data.multiwfn_tables import VDWR_UFF

LOG = logging.getLogger("test_chelpg_radii")


def _inp(folder, atoms):
    lines = ["! PBE0 def2-SVP", "* xyz 0 1"] + [f"{el} 0.0 0.0 {i:.1f}" for i, el in enumerate(atoms)] + ["*"]
    (folder / "orca.inp").write_text("\n".join(lines) + "\n")


def test_no_actinide_keeps_the_stock_input():
    assert chelpg_radii_file([1, 6, 7, 8, 17, 26, 55]) is None
    assert charge_data_dict(full_set=1)["chelpg"] == CHELPG_INPUT == "7\n12\n1\nn\n0\n0\nq\n"


def test_actinide_radii_and_unchanged_other_elements():
    text = chelpg_radii_file([92, 8, 7, 55, 94, 90])
    rows = dict(line.split() for line in text.splitlines())
    assert rows == {"N": "1.7", "O": "1.7", "Cs": repr(VDWR_UFF[55] / 1.2), "Th": "2.4", "U": "2.3", "Pu": "2.34"}
    # every element of the molecule is listed: a missing one makes Multiwfn ask for the path again
    assert len(rows) == 6
    assert chelpg_radius(26) == VDWR_UFF[26] / 1.2  # Fe keeps the ispecial=1 fallback


def test_create_jobs_writes_the_radius_file_only_for_actinides(tmp_path):
    from qtaim_gen.source.core.omol import _chelpg_input

    act = tmp_path / "act"
    act.mkdir()
    _inp(act, ["U", "O", "O", "N"])
    assert _chelpg_input(str(act), CHELPG_INPUT, LOG) == CHELPG_INPUT_RADII_FILE
    assert (act / CHELPG_RADII_FILE).read_text() == "N 1.7\nO 1.7\nU 2.3\n"

    org = tmp_path / "org"
    org.mkdir()
    _inp(org, ["C", "O", "H", "H"])
    assert _chelpg_input(str(org), CHELPG_INPUT, LOG) == CHELPG_INPUT
    assert not (org / CHELPG_RADII_FILE).exists()

    empty = tmp_path / "empty"
    empty.mkdir()
    assert _chelpg_input(str(empty), CHELPG_INPUT, LOG) == CHELPG_INPUT
