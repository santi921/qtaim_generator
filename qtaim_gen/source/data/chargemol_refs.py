"""Chargemol 09_26_2017 reference densities on its 100-shell, 500 pm radial grid
(Manz and Gabaldon Limas, ddec.sourceforge.net; port approved by T. A. Manz).
Raw values as in atomic_densities/*_500_100.txt, generated into chargemol_refs.npz by
scripts/gen_chargemol_data.py, so do not hand-edit. Used by the DDEC6 port.

    c2_density(z, nelectron)  conditioned reference ion density (c2 files)
    core_density(z, ncore)    core reference density (core files)
"""

import functools
import os

import numpy as np

_PATH = os.path.join(os.path.dirname(__file__), "chargemol_refs.npz")


@functools.lru_cache(maxsize=None)
def _tables():
    with np.load(_PATH) as f:
        out = {}
        for prefix in ("c2", "core"):
            keys = zip(f[f"{prefix}_z"].tolist(), f[f"{prefix}_n"].tolist())
            dens = f[f"{prefix}_density"]
            out[prefix] = {k: dens[i] for i, k in enumerate(keys)}
        return out


def c2_density(z, nelectron):
    """Reference ion density of element z with nelectron electrons; KeyError if absent."""
    return _tables()["c2"][(int(z), int(nelectron))]


def core_density(z, ncore):
    """Core reference density of element z with ncore core electrons; KeyError if absent."""
    return _tables()["core"][(int(z), int(ncore))]
