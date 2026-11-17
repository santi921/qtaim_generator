"""Chargemol reference densities (data/chargemol_refs.npz) against the source files.

The full comparison needs the Chargemol 09_26_2017 atomic_densities directory
(CHARGEMOL_ATOMIC_DENSITIES, default ~/dev/chargemol_src/chargemol_09_26_2017/atomic_densities)
and is skipped without it.
"""

import os
import re

import numpy as np
import pytest

from qtaim_gen.source.data import chargemol_refs as CR

SRC = os.environ.get("CHARGEMOL_ATOMIC_DENSITIES",
                     os.path.expanduser("~/dev/chargemol_src/chargemol_09_26_2017/atomic_densities"))


def test_coverage():
    t = CR._tables()
    assert len(t["c2"]) == 943 and len(t["core"]) == 954
    assert {z for z, _ in t["c2"]} == set(range(1, 110))
    assert all((z, z) in t["c2"] for z in range(1, 110))
    assert all(v.shape == (100,) for v in t["c2"].values())


def test_spot_values():
    # first shells of c2_006_006_006_500_100.txt and core_026_026_010_500_100.txt
    assert CR.c2_density(6, 6)[:2].tolist() == [131.642, 25.120724]
    assert CR.core_density(26, 10)[:2].tolist() == [13522.508, 119.46545]
    with pytest.raises(KeyError):
        CR.c2_density(6, 20)


@pytest.mark.skipif(not os.path.isdir(SRC), reason="Chargemol atomic_densities not available")
def test_matches_source_files():
    n = 0
    for name in sorted(os.listdir(SRC)):
        m = re.fullmatch(r"(c2|core)_(\d{3})_\d{3}_(\d{3})_500_100\.txt", name)
        if not m:
            continue
        with open(os.path.join(SRC, name)) as f:
            lines = [ln.strip() for ln in f]
        i = lines.index("density in atomic units:")
        ref = np.array([float(v) for v in lines[i + 1:i + 101]])
        get = CR.c2_density if m.group(1) == "c2" else CR.core_density
        assert np.array_equal(get(int(m.group(2)), int(m.group(3))), ref), name
        n += 1
    assert n == 943 + 954
