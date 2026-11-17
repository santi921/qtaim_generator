"""Generate qtaim_gen/source/data/chargemol_refs.npz from Chargemol's reference densities.

    python scripts/gen_chargemol_data.py ~/dev/chargemol_src/chargemol_09_26_2017/atomic_densities qtaim_gen/source/data

Chargemol 09_26_2017 (Manz and Gabaldon Limas, ddec.sourceforge.net; port approved by
T. A. Manz) reads two sets from atomic_densities/, both on its 100-shell, 500 pm grid:
  c2_{Z}_{Z}_{nelectron}_500_100.txt   conditioned reference ion densities
                                       (module_oxidation_density.f08)
  core_{Z}_{Z}_{ncore}_500_100.txt     core reference densities for atoms without EDF
                                       (module_add_missing_core_density.f08)
Only the raw "density in atomic units:" values are stored; the cusp correction,
monotonic smoothing and renormalization stay in code, as in Chargemol. The 400 pm /
75-shell files are not read by Chargemol (nshells = 100, cutoff_radius = 500).
"""

import os
import re
import sys

import numpy as np

NSHELLS = 100


def _read(path):
    with open(path) as fh:
        lines = fh.read().splitlines()
    head = {}
    for i, line in enumerate(lines):
        if line.strip().lower() == "density in atomic units:":
            values = np.array([float(v) for v in lines[i + 1:i + 1 + NSHELLS]])
            assert len(values) == NSHELLS, path
            return head, values
        if "=" in line:
            k, v = line.split("=", 1)
            head[k.strip()] = v.strip()
    raise ValueError(f"{path}: no density block")


def tables(src):
    out = {}
    for prefix in ("c2", "core"):
        rows = []
        for name in sorted(os.listdir(src)):
            m = re.fullmatch(rf"{prefix}_(\d{{3}})_(\d{{3}})_(\d{{3}})_500_100\.txt", name)
            if not m:
                continue
            z, z2, n = (int(g) for g in m.groups())
            assert z == z2, name
            head, values = _read(os.path.join(src, name))
            assert int(head["atomic number"]) == z, name
            assert int(head["electron count"]) == n, name
            rows.append((z, n, values))
        assert rows, f"no {prefix} files in {src}"
        out[f"{prefix}_z"] = np.array([r[0] for r in rows], dtype=np.int16)
        out[f"{prefix}_n"] = np.array([r[1] for r in rows], dtype=np.int16)
        out[f"{prefix}_density"] = np.stack([r[2] for r in rows])
    return out


def main(argv):
    src, dst = argv[1], argv[2]
    data = tables(src)
    path = os.path.join(dst, "chargemol_refs.npz")
    np.savez_compressed(path, source=np.array("Chargemol 09_26_2017 atomic_densities, *_500_100"), **data)
    print(f"wrote {path}: {len(data['c2_z'])} c2, {len(data['core_z'])} core")


if __name__ == "__main__":
    main(sys.argv)
