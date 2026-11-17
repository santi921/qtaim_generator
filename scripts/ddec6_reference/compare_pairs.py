"""Compare the DDEC6 port's grid setup and pair data (K0-K5) with Chargemol's dumps.

    python compare_pairs.py <orca.wfx> <dump dir with dump_pairs.bin and dump_step7.bin>
"""

import os
import sys
import time

import numpy as np

import read_dumps as R
from qtaim_gen.source.core import ddec6

COLS = {0: "i", 1: "j", 2: "P_sum", 3: "P_diff", 4: "overlap", 5: "alpha1", 6: "Lx1", 7: "Ly1", 8: "Lz1",
        9: "X1", 10: "Y1", 11: "Z1", 12: "alpha2", 13: "Lx2", 14: "Ly2", 15: "Lz2", 16: "X2", 17: "Y2",
        18: "Z2", 19: "dna", 20: "dnb", 21: "dnc", 22: "XD", 23: "YD", 24: "ZD", 25: "Na", 26: "Nb",
        27: "Nc", 28: "shift_x", 29: "shift_y", 30: "shift_z", 31: "renorm", 32: "analytic",
        33: "R2_cut", 34: "common", 35: "gridspacing", 36: "w_i", 37: "w_j"}


def report(name, a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape:
        print(f"  {name:22s} SHAPE {a.shape} vs {b.shape}")
        return False
    d = np.abs(a - b)
    scale = np.maximum(np.abs(b), 1e-300)
    n = int((a != b).sum())
    print(f"  {name:22s} differ {n:8d} / {a.size:<9d} max abs {d.max() if d.size else 0:.3e}"
          f"  max rel {(d / scale).max() if d.size else 0:.3e}")
    return n == 0


def main(wfx, dump_dir):
    t0 = time.time()
    prep = ddec6.load(wfx)
    grid = ddec6.grid_setup(prep["atom_xyz"])
    t1 = time.time()
    pairs = ddec6.pair_data(prep, grid)
    t2 = time.time()
    ref = R.read_pairs(os.path.join(dump_dir, "dump_pairs.bin"))
    s7 = R.read_step7(os.path.join(dump_dir, "dump_step7.bin"), ref["n_pairs"], ref["natoms"])
    print(f"port: prepare + grid {t1 - t0:.1f} s, pair data {t2 - t1:.1f} s")

    ok = True
    for k in ("n_pairs", "n_analytic", "n_numeric", "n_blocks", "nsingleblocks", "included_electrons"):
        same = pairs[k] == ref[k]
        ok &= same
        print(f"  {k:22s} port {pairs[k]}  chargemol {ref[k]}" + ("" if same else "  MISMATCH"))
    print(f"  totnum                 port {grid['totnum']}  chargemol {(s7['totnumA'], s7['totnumB'], s7['totnumC'])}")
    ok &= grid["totnum"] == (s7["totnumA"], s7["totnumB"], s7["totnumC"])
    ok &= report("origin", grid["origin"], s7["origin"])
    ok &= report("boundary", grid["boundary"], s7["boundary"].T)
    ok &= report("pixelvolume", grid["pixelvolume"], s7["pixelvolume"])
    ok &= report("check_included_e", pairs["check_included_electrons"], ref["check_included_electrons"])
    if pairs["n_pairs"] != ref["n_pairs"]:
        print("pair counts differ; skipping row comparison")
        return 1
    ok &= report("pair_block_data", pairs["pair_block_data"], ref["pair_block_data"])
    a, b = pairs["included_primitive_data"], ref["included_primitive_data"]
    for c in range(38):
        ok &= report(f"col {c + 1} {COLS[c]}", a[:, c], b[:, c])
    ok &= report("occupancy_correction", pairs["occupancy_correction"], ref["occupancy_correction"])
    ok &= report("core_electrons", pairs["core_electrons"], ref["core_electrons"])
    print("K0-K5: BIT-IDENTICAL" if ok else "K0-K5: DIFFERENCES (see above)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:3]))
