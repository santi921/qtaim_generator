"""DDEC6 port (core/ddec6.py), density-grid kernels K0-K5.

Unit tests run everywhere. The parity test compares against Chargemol's validation
dumps (scripts/ddec6_reference, chargemol_serial_dump) and is skipped without them:
DDEC6_DUMPS, default ~/dev/qtaim_generator/data/ddec6_ref/dumps/<fixture>/.
"""

import math
import os

import numpy as np
import pytest

from qtaim_gen.source.core import ddec6

FIXTURES = os.path.join(os.path.dirname(__file__), "test_files", "charge_engine")
DUMPS = os.environ.get("DDEC6_DUMPS", os.path.expanduser("~/dev/qtaim_generator/data/ddec6_ref/dumps"))


def test_nint_rounds_half_away_from_zero():
    assert [ddec6.nint(x) for x in (0.5, 1.5, 2.5, -0.5, -1.5, 2.49)] == [1, 2, 3, -1, -2, 2]


def test_powi_matches_square_and_multiply():
    x = 1.2345678901234567
    assert ddec6.powi(x, 0) == 1.0
    assert ddec6.powi(x, 5) == ((x * x) * (x * x)) * x
    assert ddec6.powi(0.0, 0) == 1.0
    assert ddec6.pow10(x) == ((x * x * x) * (x * x)) ** 2


def test_overlap_closed_forms():
    a, b, r = 0.7, 1.9, 1.3
    p = a + b
    ss = (math.pi / p) ** 1.5 * math.exp(-a * b / p * r * r)
    got = ddec6.gaussian_overlap_integral(a, 0, 0, 0, 0.0, 0.0, 0.0, b, 0, 0, 0, 0.0, 0.0, r)
    assert got == pytest.approx(ss, rel=1e-14)
    # <x^2> of one s Gaussian: (pi/p)^1.5 / (2p)
    got = ddec6.gaussian_overlap_integral(a, 2, 0, 0, 0.0, 0.0, 0.0, b, 0, 0, 0, 0.0, 0.0, 0.0)
    assert got == pytest.approx((math.pi / p) ** 1.5 / (2 * p), rel=1e-14)
    # odd total power on one centre integrates to zero
    assert ddec6.gaussian_overlap_integral(a, 1, 0, 0, 0.0, 0.0, 0.0, b, 0, 0, 0, 0.0, 0.0, 0.0) == 0.0


def test_sort_order_groups_centre_then_exponent():
    center = np.array([1, 0, 1, 0, 0])
    exps = np.array([3.0, 2.0, 1.0, 5.0, 2.0])
    perm = ddec6._sort_order(center, exps)
    assert center[perm].tolist() == [0, 0, 0, 1, 1]
    assert exps[perm].tolist() == [2.0, 2.0, 5.0, 1.0, 3.0]


@pytest.mark.parametrize("name", ["rmechdb_652_step10_0_2", "XeCl4_step2_0_1"])
def test_pair_data_electron_count(name):
    prep = ddec6.load(os.path.join(FIXTURES, name, "orca.wfx"))
    grid = ddec6.grid_setup(prep["atom_xyz"])
    pairs = ddec6.pair_data(prep, grid)
    assert abs(pairs["check_included_electrons"] - pairs["included_electrons"]) < 1e-8
    assert all(n % 24 == 0 for n in grid["totnum"])
    assert pairs["n_pairs"] == pairs["n_analytic"] + pairs["n_numeric"]


@pytest.mark.parametrize("name", ["rmechdb_652_step10_0_2", "XeCl4_step2_0_1"])
def test_pair_data_matches_chargemol_bitwise(name):
    dump_dir = os.path.join(DUMPS, name)
    if not os.path.exists(os.path.join(dump_dir, "dump_pairs.bin")):
        pytest.skip("Chargemol validation dumps not available")
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "read_dumps", os.path.join(os.path.dirname(__file__), "..", "scripts", "ddec6_reference", "read_dumps.py"))
    R = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(R)
    ref = R.read_pairs(os.path.join(dump_dir, "dump_pairs.bin"))
    s7 = R.read_step7(os.path.join(dump_dir, "dump_step7.bin"), ref["n_pairs"], ref["natoms"])

    prep = ddec6.load(os.path.join(FIXTURES, name, "orca.wfx"))
    grid = ddec6.grid_setup(prep["atom_xyz"])
    pairs = ddec6.pair_data(prep, grid)
    for k in ("n_pairs", "n_analytic", "n_numeric", "n_blocks", "nsingleblocks", "included_electrons"):
        assert pairs[k] == ref[k], k
    assert grid["totnum"] == (s7["totnumA"], s7["totnumB"], s7["totnumC"])
    assert np.array_equal(grid["origin"], s7["origin"])
    assert grid["pixelvolume"] == s7["pixelvolume"]
    assert pairs["check_included_electrons"] == ref["check_included_electrons"]
    assert np.array_equal(pairs["pair_block_data"], ref["pair_block_data"])
    assert np.array_equal(pairs["included_primitive_data"][:, :38], ref["included_primitive_data"][:, :38])
    assert np.array_equal(pairs["occupancy_correction"], ref["occupancy_correction"])
    assert np.array_equal(pairs["core_electrons"], ref["core_electrons"])
