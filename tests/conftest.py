"""Shared pytest configuration and compatibility shims."""

import pickle
from pathlib import Path

import lmdb
import numpy as np
import pytest

# numpy 2.x removed np.isscalar, but pytest 9.0.x still uses it internally
# in pytest.approx. Restore it until pytest ships a fix.
if not hasattr(np, "isscalar"):
    np.isscalar = lambda x: np.ndim(x) == 0


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "slow: tests that hit real OMol4M_lmdbs verticals; skipped if data is missing",
    )


_LMDB_TYPES = ["structure", "charge", "qtaim", "bond", "fuzzy", "other", "orca", "timings"]
_DEFAULT_KEYS = ["k_alpha", "k_beta", "k_gamma"]


def _payload(lmdb_type, key, idx):
    if lmdb_type == "structure":
        if idx % 2 == 0:
            return {
                "atomic_numbers": [1, 1, 8],
                "positions": [[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [0.48, 0.83, 0.0]],
            }
        return {
            "atomic_numbers": [6, 1, 1, 1, 1],
            "positions": [
                [0.0, 0.0, 0.0],
                [0.63, 0.63, 0.63],
                [-0.63, -0.63, 0.63],
                [-0.63, 0.63, -0.63],
                [0.63, -0.63, -0.63],
            ],
        }
    if lmdb_type == "charge":
        return {
            "hirshfeld": [0.10 + idx, -0.10, 0.0],
            "cm5": [0.12 + idx, -0.11, -0.01],
            "atom_indices": [0, 1, 2],
        }
    if lmdb_type == "orca":
        return {
            "dipole_au": [0.10 + 0.01 * idx, 0.0, 0.0],
            "dipole_magnitude_au": 0.10 + 0.01 * idx,
            "energy": -76.4 + idx,
        }
    if lmdb_type == "qtaim":
        return {"bond_critical_points": [], "n_cps": idx}
    if lmdb_type == "bond":
        return {"mayer": {"0_1": 0.95}, "wiberg": {"0_1": 0.92}}
    if lmdb_type == "fuzzy":
        return {"fuzzy_volume": [1.0 + idx, 1.0, 1.0]}
    if lmdb_type == "other":
        return {"esp_min": -0.05 - idx, "esp_max": 0.05 + idx}
    if lmdb_type == "timings":
        return {"orca_seconds": 12.5 + idx}
    raise ValueError(lmdb_type)


# Real-shape payloads used by the census (Stream C) tests. The shapes mirror
# what we actually see in data/OMol4M_lmdbs/<vertical>/<type>.lmdb so census
# code is exercised against production-equivalent records.
#
# Per-record seeds for the three default keys (idx 0/1/2):
#   structure: H2O (3 atoms), H2O (3 atoms), CH4 (5 atoms) -> 2 unique formulas
#   charge:    3 schemes per record -> 9 (scheme, structure) pairs
#   qtaim:     2 BCPs, 1 BCP, 0 BCPs -> 3 BCPs total (NCPs always present)
#   bond:      3 bond entries per record across 2 schemes -> 9 bonds total
def _real_payload(lmdb_type, key, idx):
    if lmdb_type == "structure":
        from pymatgen.core import Molecule

        if idx == 2:
            return {
                "molecule": Molecule(
                    ["C", "H", "H", "H", "H"],
                    [
                        [0.0, 0.0, 0.0],
                        [0.63, 0.63, 0.63],
                        [-0.63, -0.63, 0.63],
                        [-0.63, 0.63, -0.63],
                        [0.63, -0.63, -0.63],
                    ],
                ),
                "spin": 1,
                "charge": 0,
            }
        return {
            "molecule": Molecule(
                ["H", "H", "O"],
                [[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [0.48, 0.83, 0.0]],
            ),
            "spin": 1,
            "charge": 0,
        }
    if lmdb_type == "charge":
        return {
            "hirshfeld": {"charge": {0: 0.10, 1: 0.10, 2: -0.20}, "dipole": {"x": 0.0}},
            "cm5": {"charge": {0: 0.12, 1: 0.12, 2: -0.24}, "dipole": {"x": 0.0}},
            "mulliken_orca": {"charge": {0: 0.08, 1: 0.08, 2: -0.16}},
        }
    if lmdb_type == "qtaim":
        cp_template = {"density_all": 0.1, "lap_e_density": -0.05}
        if idx == 0:
            return {
                "0": cp_template, "1": cp_template, "2": cp_template,
                "0_1": cp_template, "0_2": cp_template,
            }
        if idx == 1:
            return {"0": cp_template, "1": cp_template, "2": cp_template, "0_1": cp_template}
        return {
            "0": cp_template, "1": cp_template, "2": cp_template,
            "3": cp_template, "4": cp_template,
        }
    if lmdb_type == "bond":
        return {
            "mayer_orca": {"0_1": 1.0, "0_2": 0.5},
            "loewdin_orca": {"0_1": 0.9},
        }
    # Census does not read fuzzy/other/orca/timings; reuse the simple payloads.
    return _payload(lmdb_type, key, idx)


def _write_lmdb(path, records):
    env = lmdb.open(
        str(path),
        subdir=False,
        map_size=10 * 1024 * 1024,
        meminit=False,
    )
    with env.begin(write=True) as txn:
        for k, v in records.items():
            txn.put(k.encode("ascii"), pickle.dumps(v, protocol=-1))
        txn.put(b"length", pickle.dumps(len(records), protocol=-1))
    env.sync()
    env.close()


@pytest.fixture
def tiny_vertical(tmp_path):
    """Build a synthetic vertical with all eight LMDBs at tmp_path.

    Returns a builder fn::

        build(keys=None, orphan_lmdb=None, *,
              real_shape=False, bad_records=None, dest=None)

    - ``orphan_lmdb`` is a dict ``{lmdb_type: [extra_keys]}`` that injects
      keys into specific LMDBs only (used for inner-join orphan tests).
    - ``real_shape=True`` swaps the simplified payloads for production-shape
      records (pymatgen Molecule, nested charge schemes, BCP/NCP keys in
      qtaim, bond schemes); used by Stream C census tests.
    - ``bad_records`` is a dict ``{lmdb_type: {key: payload}}`` that injects
      explicit (possibly malformed) payloads. Bypasses the payload generator.
    - ``dest`` overrides ``tmp_path`` so multiple verticals can be written
      under a shared parent directory (corpus-mode tests).

    The builder returns ``(root_path, base_keys)``.
    """

    def build(keys=None, orphan_lmdb=None, *, real_shape=False, bad_records=None, dest=None):
        keys = list(keys) if keys is not None else list(_DEFAULT_KEYS)
        orphan_lmdb = orphan_lmdb or {}
        bad_records = bad_records or {}
        target = Path(dest) if dest is not None else tmp_path
        target.mkdir(parents=True, exist_ok=True)
        payload_fn = _real_payload if real_shape else _payload
        for t in _LMDB_TYPES:
            records = {k: payload_fn(t, k, i) for i, k in enumerate(keys)}
            for extra_idx, extra_k in enumerate(orphan_lmdb.get(t, []), start=len(keys)):
                records[extra_k] = payload_fn(t, extra_k, extra_idx)
            for bad_k, bad_v in bad_records.get(t, {}).items():
                records[bad_k] = bad_v
            _write_lmdb(target / f"{t}.lmdb", records)
        return target, keys

    return build
