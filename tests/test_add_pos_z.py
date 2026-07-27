"""
Tests for atom.pos/atom.z in graph LMDBs.

Covers both paths to the new schema:
1. Regeneration: the grapher now stores pos/z at build time, so converter
   output must carry them and they must match structure.lmdb exactly.
2. Patch: add_pos_z_to_graphs attaches pos/z to graphs built before the
   schema change. Verified by stripping pos/z from freshly built graphs and
   asserting the patch restores them byte-identically.
"""

import os
import pickle
import shutil

import lmdb
import pytest
import torch

from qtaim_embed.data.lmdb import load_graph_from_serialized, serialize_graph

from qtaim_gen.source.core.converter import BaseConverter
from qtaim_gen.source.scripts.helpers.add_pos_z_to_graphs import (
    StructureLookup,
    patch_lmdb,
)
from qtaim_gen.source.utils.scaling import _is_metadata

BASE_TESTS = os.path.dirname(__file__)
MERGED_FOLDER = os.path.join(BASE_TESTS, "test_files", "lmdb_tests", "generator_lmdbs_merged")
GEOM_LMDB = os.path.join(MERGED_FOLDER, "merged_geom.lmdb")

pytestmark = pytest.mark.skipif(
    not os.path.exists(GEOM_LMDB), reason="Test LMDB fixtures not available"
)


def _iter_graph_records(lmdb_path):
    """Yield (key_bytes, graph) for every graph record in a graph LMDB."""
    env = lmdb.open(lmdb_path, subdir=False, readonly=True, lock=False)
    with env.begin() as txn:
        for key_bytes, value_bytes in txn.cursor():
            if _is_metadata(key_bytes, set()):
                continue
            obj = pickle.loads(value_bytes)
            graph_bytes = obj["molecule_graph"] if isinstance(obj, dict) else obj
            yield key_bytes, load_graph_from_serialized(graph_bytes)
    env.close()


def _structure_pos_z(mol_name):
    env = lmdb.open(GEOM_LMDB, subdir=False, readonly=True, lock=False)
    with env.begin() as txn:
        raw = txn.get(mol_name.encode("ascii"))
    env.close()
    assert raw is not None
    molecule = pickle.loads(raw)["molecule_graph"].molecule
    pos = torch.tensor(molecule.cart_coords, dtype=torch.float32)
    z = torch.tensor(molecule.atomic_numbers, dtype=torch.long)
    return pos, z


@pytest.fixture(scope="module")
def built_graph_lmdb(tmp_path_factory):
    """Run BaseConverter on the geom fixture and return the graph LMDB path."""
    out_dir = tmp_path_factory.mktemp("posz_build")
    config = {
        "chunk": -1,
        "filter_list": ["length"],
        "restart": False,
        "allowed_ring_size": [3, 4, 5, 6, 7, 8],
        "allowed_charges": None,
        "allowed_spins": None,
        "keys_target": {"atom": [], "bond": [], "global": ["n_atoms"]},
        "keys_data": {"atom": [], "bond": [], "global": ["n_atoms"]},
        "lmdb_path": str(out_dir),
        "lmdb_name": "graphs.lmdb",
        "lmdb_locations": {"geom_lmdb": GEOM_LMDB},
        "n_workers": 1,
        "batch_size": 100,
        "skip_scaling": True,
        "save_unfinalized_scaler": True,
    }
    converter = BaseConverter(
        config, config_path=os.path.join(str(out_dir), "config.json")
    )
    converter.process(return_info=True)
    return os.path.join(str(out_dir), "graphs.lmdb")


def _strip_pos_z(src_path, dst_path):
    """Copy a graph LMDB with pos/z removed from every graph (pre-schema format)."""
    shutil.copy(src_path, dst_path)
    env = lmdb.open(dst_path, subdir=False, map_size=1024**3)
    updates = []
    with env.begin() as txn:
        for key_bytes, value_bytes in txn.cursor():
            if _is_metadata(key_bytes, set()):
                continue
            obj = pickle.loads(value_bytes)
            graph = load_graph_from_serialized(obj["molecule_graph"])
            del graph["atom"].pos
            del graph["atom"].z
            updates.append((
                key_bytes,
                pickle.dumps(
                    {"molecule_graph": serialize_graph(graph, ret=True)},
                    protocol=-1,
                ),
            ))
    with env.begin(write=True) as txn:
        for k, v in updates:
            txn.put(k, v)
    env.close()


class TestRegeneratedGraphs:
    def test_pos_z_present_and_match_structure(self, built_graph_lmdb):
        n_graphs = 0
        for _key, graph in _iter_graph_records(built_graph_lmdb):
            assert "pos" in graph["atom"]
            assert "z" in graph["atom"]
            assert graph["atom"].pos.dtype == torch.float32
            assert graph["atom"].z.dtype == torch.long
            pos_ref, z_ref = _structure_pos_z(graph.mol_name)
            assert torch.equal(graph["atom"].pos, pos_ref)
            assert torch.equal(graph["atom"].z, z_ref)
            n_graphs += 1
        assert n_graphs > 0


class TestPatchScript:
    def test_patch_restores_stripped_graphs(self, built_graph_lmdb, tmp_path):
        stripped = str(tmp_path / "stripped.lmdb")
        _strip_pos_z(built_graph_lmdb, stripped)
        for _key, graph in _iter_graph_records(stripped):
            assert "pos" not in graph["atom"]

        lookup = StructureLookup([GEOM_LMDB])
        counts = patch_lmdb(stripped, lookup)
        lookup.close()
        assert counts["skipped_has_pos"] == 0
        assert counts["patched"] > 0

        reference = {k: g for k, g in _iter_graph_records(built_graph_lmdb)}
        patched = {k: g for k, g in _iter_graph_records(stripped)}
        assert set(patched) == set(reference)
        for key, graph in patched.items():
            ref = reference[key]
            assert graph.mol_name == ref.mol_name
            assert torch.equal(graph["atom"].pos, ref["atom"].pos)
            assert torch.equal(graph["atom"].z, ref["atom"].z)
            assert torch.equal(graph["atom"].feat, ref["atom"].feat)

    def test_patch_is_idempotent(self, built_graph_lmdb, tmp_path):
        target = str(tmp_path / "patched.lmdb")
        shutil.copy(built_graph_lmdb, target)

        lookup = StructureLookup([GEOM_LMDB])
        counts = patch_lmdb(target, lookup)
        lookup.close()
        assert counts["patched"] == 0
        assert counts["skipped_has_pos"] > 0

    def test_patch_fails_on_missing_key(self, built_graph_lmdb, tmp_path):
        stripped = str(tmp_path / "bad_key.lmdb")
        _strip_pos_z(built_graph_lmdb, stripped)

        # corrupt one graph's mol_name so the join must miss
        env = lmdb.open(stripped, subdir=False, map_size=1024**3)
        with env.begin() as txn:
            for key_bytes, value_bytes in txn.cursor():
                if not _is_metadata(key_bytes, set()):
                    break
        obj = pickle.loads(value_bytes)
        graph = load_graph_from_serialized(obj["molecule_graph"])
        graph.mol_name = "does_not_exist_anywhere"
        with env.begin(write=True) as txn:
            txn.put(
                key_bytes,
                pickle.dumps(
                    {"molecule_graph": serialize_graph(graph, ret=True)},
                    protocol=-1,
                ),
            )
        env.close()

        lookup = StructureLookup([GEOM_LMDB])
        with pytest.raises(RuntimeError, match="not found"):
            patch_lmdb(stripped, lookup)
        lookup.close()
        # source must be intact after the failed run (atomic replace never ran)
        assert os.path.exists(stripped)
        for _key, graph in _iter_graph_records(stripped):
            assert "pos" not in graph["atom"]

    def test_dry_run_writes_nothing(self, built_graph_lmdb, tmp_path):
        stripped = str(tmp_path / "dry.lmdb")
        _strip_pos_z(built_graph_lmdb, stripped)
        mtime = os.path.getmtime(stripped)

        lookup = StructureLookup([GEOM_LMDB])
        counts = patch_lmdb(stripped, lookup, dry_run=True)
        lookup.close()
        assert counts["patched"] > 0
        assert os.path.getmtime(stripped) == mtime
        for _key, graph in _iter_graph_records(stripped):
            assert "pos" not in graph["atom"]
