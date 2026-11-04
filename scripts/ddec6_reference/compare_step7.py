"""Compare the DDEC6 port's Step 7 grids (K6-K7) with Chargemol's dump_step7.bin.

    python compare_step7.py <orca.wfx> <dump dir>
"""

import os
import sys
import time

import numpy as np

import read_dumps as R
from compare_pairs import report
from qtaim_gen.source.core import ddec6


def main(wfx, dump_dir):
    prep = ddec6.load(wfx)
    grid = ddec6.grid_setup(prep["atom_xyz"])
    pairs = ddec6.pair_data(prep, grid)
    t0 = time.time()
    g = ddec6.density_grids(prep, grid, pairs)
    print(f"port: Step 7 {time.time() - t0:.1f} s, max_grid_spacing {g['max_grid_spacing']}")
    ref = R.read_pairs(os.path.join(dump_dir, "dump_pairs.bin"))
    s7 = R.read_step7(os.path.join(dump_dir, "dump_step7.bin"), ref["n_pairs"], ref["natoms"])
    ok = report("max_pair_correction", g["max_pair_correction"], s7["max_pair_correction"])
    for c, name in ((31, "renorm"), (38, "overlap_sum"), (39, "abs_overlap_sum")):
        ok &= report(f"col {c + 1} {name}", g["included_primitive_data"][:, c], s7["included_primitive_data"][:, c])
    ok &= report("occupancy_correction", g["occupancy_correction"], s7["occupancy_correction"])
    for k in ("valence_density", "core_density", "spin_density"):
        if k in s7:
            ok &= report(k, g[k], s7[k])
    pv = grid["pixelvolume"]
    print(f"  sums*pv: valence {g['valence_density'].sum() * pv:.12f} core {g['core_density'].sum() * pv:.12f}")
    print("K6-K7: BIT-IDENTICAL" if ok else "K6-K7: DIFFERENCES (see above)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:3]))
