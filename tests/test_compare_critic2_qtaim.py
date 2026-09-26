import pytest

from qtaim_gen.source.scripts.helpers.compare_critic2_qtaim import wfx_has_ecp


@pytest.mark.parametrize("n_core, expected", [("28", True), ("0", False)])
def test_wfx_has_ecp_reads_core_electron_count(tmp_path, n_core, expected):
    wfx = tmp_path / "orca.wfx"
    wfx.write_text(
        "<Number of Nuclei>\n 3\n</Number of Nuclei>\n"
        f"<Number of Core Electrons>\n    {n_core}\n</Number of Core Electrons>\n"
    )
    assert wfx_has_ecp(str(wfx)) is expected
