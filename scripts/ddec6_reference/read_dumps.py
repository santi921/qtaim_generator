"""Read the validation dumps chargemol_serial_dump writes (chargemol_dump.patch).

All three files are Fortran stream access: default INTEGER is int32, REAL(dp) float64,
arrays column-major. Indices in the arrays are Chargemol's (1-based) where they are
data (pair primitive indices, pair_block_data).

    pairs = read_pairs("dump_pairs.bin")   after Step 6 (pair list, analytic corrections)
    step7 = read_step7("dump_step7.bin")   after Step 7 and the coarse-grid interpolation
    final = read_final("dump_final.bin")   after the core grid and add_missing_core_density
"""

import numpy as np


class _Stream:
    def __init__(self, path):
        with open(path, "rb") as f:
            self.buf = f.read()
        self.pos = 0

    def take(self, dtype, shape=()):
        n = int(np.prod(shape)) if shape else 1
        a = np.frombuffer(self.buf, dtype=dtype, count=n, offset=self.pos)
        self.pos += a.nbytes
        return a.reshape(shape, order="F") if shape else a[0]

    def done(self):
        assert self.pos == len(self.buf), f"{len(self.buf) - self.pos} bytes left"


def read_pairs(path):
    s = _Stream(path)
    out = {k: int(s.take("<i4")) for k in ("n_pairs", "n_analytic", "n_numeric", "n_blocks", "max_grid_spacing",
                                           "natoms", "nsingleblocks")}
    out["check_included_electrons"] = float(s.take("<f8"))
    out["included_electrons"] = int(s.take("<i4"))  # an INTEGER in Chargemol
    out["pair_block_data"] = s.take("<f8", (out["n_blocks"], 2))
    out["included_primitive_data"] = s.take("<f8", (out["n_pairs"], 40))
    out["occupancy_correction"] = s.take("<f8", (11, out["natoms"]))
    out["core_electrons"] = s.take("<f8", (out["natoms"],))
    s.done()
    return out


def _grids(s, out, natoms=None):
    shape = (out["totnumA"], out["totnumB"], out["totnumC"])
    out["valence_density"] = s.take("<f8", shape)
    out["core_density"] = s.take("<f8", shape)
    if out["spin_available"]:
        out["spin_density"] = s.take("<f8", shape)


def _header(s):
    out = {k: int(s.take("<i4")) for k in ("totnumA", "totnumB", "totnumC")}
    out["spin_available"] = bool(s.take("<i4"))
    return out


def read_step7(path, n_pairs, natoms):
    s = _Stream(path)
    out = _header(s)
    out["origin"] = s.take("<f8", (3,))
    out["boundary"] = s.take("<f8", (3, 3))
    out["pixelvolume"] = float(s.take("<f8"))
    out["max_pair_correction"] = float(s.take("<f8"))
    out["included_primitive_data"] = s.take("<f8", (n_pairs, 40))
    out["occupancy_correction"] = s.take("<f8", (11, natoms))
    out["core_electrons"] = s.take("<f8", (natoms,))
    _grids(s, out)
    s.done()
    return out


def read_final(path, natoms):
    s = _Stream(path)
    out = _header(s)
    out["occupancy_correction"] = s.take("<f8", (11, natoms))
    out["core_electrons"] = s.take("<f8", (natoms,))
    out["missing_core"] = s.take("<i4", (natoms,))
    _grids(s, out)
    s.done()
    return out
