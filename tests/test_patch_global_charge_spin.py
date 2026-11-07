"""
charge_dipole_global=false and patch_global_charge_spin.

The patch must reproduce a fresh build: an old-schema build (dipole globals, no
charge/spin), scaled and then patched, has to equal a build with
charge_dipole_global=false + allowed_charges/allowed_spins scaled with its own
scaler, graph by graph and in the LMDB metadata.
"""

import os
import pickle
from pathlib import Path

import lmdb
import pytest
import torch

from qtaim_gen.source.core.converter import GeneralConverter
from qtaim_gen.source.scripts.helpers.add_pos_z_to_graphs import StructureLookup
from qtaim_gen.source.scripts.helpers.patch_global_charge_spin import (
    build_feature_scaler,
    fit_onehot_stats,
    new_global_names,
    patch_lmdb,
)
from qtaim_gen.source.utils.lmdbs import parse_charge_data
from qtaim_gen.source.utils.scaling import _deserialize_graph, _is_metadata, apply_scalers_to_lmdb_inplace
from qtaim_embed.data.processing import HeteroGraphStandardScalerIterative

MERGED = Path(__file__).parent / "test_files" / "lmdb_tests" / "generator_lmdbs_merged"
GEOM = MERGED / "merged_geom.lmdb"
SCHEMES = ["hirshfeld", "adch", "cm5", "becke"]
DIPOLES = [f"charge_{s}_dipole_mag" for s in SCHEMES]
CHARGES = list(range(-10, 11))
SPINS = list(range(1, 12))

pytestmark = pytest.mark.skipif(not GEOM.exists(), reason="Test LMDB fixtures not available")


def _config(out, **kw):
    c = {
        "chunk": -1, "filter_list": ["scaled", "length"], "restart": False,
        "allowed_ring_size": [3, 4, 5, 6, 7, 8], "allowed_charges": None, "allowed_spins": None,
        "data_inputs": ["geom", "charge"], "bonding_scheme": "structural", "charge_filter": SCHEMES,
        "keys_data": {"atom": [f"charge_{s}" for s in SCHEMES], "bond": [], "global": list(DIPOLES)},
        "keys_target": {"atom": [f"charge_{s}" for s in SCHEMES], "bond": [], "global": []},
        "lmdb_path": str(out), "lmdb_name": "graphs.lmdb",
        "lmdb_locations": {"geom_lmdb": str(GEOM), "charge_lmdb": str(MERGED / "merged_charge.lmdb")},
        "n_workers": 1, "batch_size": 100, "save_scaler": True,
    }
    c.update(kw)
    return c


def _build_scaled(out, **kw):
    GeneralConverter(_config(out, **kw)).process()
    path = str(out / "graphs.lmdb")
    fs = HeteroGraphStandardScalerIterative(features_tf=True, load=True, finalized=True,
                                            load_path=str(out / "feature_scaler_iterative.pt"))
    ls = HeteroGraphStandardScalerIterative(features_tf=False, load=True, finalized=True,
                                            load_path=str(out / "label_scaler_iterative.pt"))
    apply_scalers_to_lmdb_inplace(path, fs, ls, set())
    return path


def _read(path):
    env = lmdb.open(path, subdir=False, readonly=True, lock=False)
    graphs, meta = {}, {}
    with env.begin() as t:
        for k, v in t.cursor():
            if _is_metadata(k, set()):
                meta[k] = pickle.loads(v)
            else:
                g = _deserialize_graph(v)
                graphs[g.mol_name] = g
    env.close()
    return graphs, meta


@pytest.fixture(scope="module")
def builds(tmp_path_factory):
    old = _build_scaled(tmp_path_factory.mktemp("old"))
    new_dir = tmp_path_factory.mktemp("new")
    new = _build_scaled(new_dir, charge_dipole_global=False, allowed_charges=CHARGES, allowed_spins=SPINS)
    return old, new, new_dir


def test_parse_charge_data_include_dipoles():
    rec = {"hirshfeld": {"charge": {"1_O": -0.4, "2_H": 0.4}, "dipole": {"mag": 0.7}}}
    _, glob = parse_charge_data(rec, 2, ["hirshfeld"])
    assert glob == {"charge_hirshfeld_dipole_mag": 0.7}
    atoms, glob = parse_charge_data(rec, 2, ["hirshfeld"], include_dipoles=False)
    assert glob == {} and atoms[0]["charge_hirshfeld"] == -0.4


def test_old_build_has_dipole_globals(builds):
    _, meta = _read(builds[0])
    assert meta[b"feature_names"]["global"] == ["num atoms", "num bonds", "molecule weight"] + DIPOLES


def test_charge_dipole_global_false_drops_listed_dipoles(builds):
    graphs, meta = _read(builds[1])
    assert meta[b"feature_names"]["global"] == new_global_names(CHARGES, SPINS)
    assert all(g["global"].feat.shape == (1, 35) for g in graphs.values())


def test_patch_matches_fresh_build(builds, tmp_path):
    old, new, new_dir = builds
    lookup = StructureLookup([str(GEOM)])
    try:
        stats = fit_onehot_stats(old, lookup, CHARGES, SPINS)
        old_scaler = os.path.join(os.path.dirname(old), "feature_scaler_iterative.pt")
        d = build_feature_scaler(old_scaler, str(tmp_path / "feature_scaler_iterative.pt"), stats)
        counts = patch_lmdb(old, lookup, stats)
        again = patch_lmdb(old, lookup, stats)
    finally:
        lookup.close()
    assert counts["patched"] == stats["n_graphs"] > 0
    assert again == {"patched": 0, "already_patched": True}

    patched, meta_p = _read(old)
    fresh, meta_f = _read(new)
    assert set(patched) == set(fresh)
    for k in (b"feature_names", b"feature_size", b"allowed_charges", b"allowed_spins"):
        assert meta_p[k] == meta_f[k], k
    for name, g in fresh.items():
        p = patched[name]
        torch.testing.assert_close(p["global"].feat.double(), g["global"].feat.double(), atol=1e-6, rtol=0)
        for nt, attr in (("atom", "feat"), ("bond", "feat"), ("atom", "labels")):
            torch.testing.assert_close(getattr(p[nt], attr), getattr(g[nt], attr))

    ref = torch.load(new_dir / "feature_scaler_iterative.pt", weights_only=False)
    torch.testing.assert_close(d["mean"]["global"], ref["mean"]["global"].double(), atol=1e-9, rtol=0)
    torch.testing.assert_close(d["std"]["global"], ref["std"]["global"].double(), atol=1e-9, rtol=0)
