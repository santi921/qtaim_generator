"""
Integration tests verifying that converter LMDB output is compatible with
qtaim_embed's LMDBBaseDataset reader.

Compatibility contract:
  - Integer string keys "0", "1", ... (not molecule ID strings)
  - "length" metadata key -> pickle.dumps(int)
  - Values are pickle.dumps({"molecule_graph": <torch.save bytes>})
  - load_graph_from_serialized(sample["molecule_graph"]) returns a valid PyG graph
"""

import pickle
import pytest
import lmdb
from pathlib import Path

try:
    from qtaim_gen.source.core.converter import BaseConverter, GeneralConverter
    from qtaim_embed.data.lmdb import load_graph_from_serialized
    DEPS_AVAILABLE = True
except ImportError:
    DEPS_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not DEPS_AVAILABLE,
    reason="qtaim_embed or converter not available"
)

MERGED = Path(__file__).parent / "test_files" / "lmdb_tests" / "generator_lmdbs_merged"


def _base_config(tmp_path):
    return {
        "chunk": -1,
        "filter_list": ["length"],
        "restart": False,
        "allowed_ring_size": [3, 4, 5, 6, 7, 8],
        "allowed_charges": None,
        "allowed_spins": None,
        "keys_target": {"atom": [], "bond": [], "global": ["n_atoms"]},
        "keys_data": {"atom": [], "bond": [], "global": ["n_atoms"]},
        "lmdb_path": str(tmp_path) + "/",
        "lmdb_name": "test_graphs.lmdb",
        "lmdb_locations": {"geom_lmdb": str(MERGED / "merged_geom.lmdb")},
        "n_workers": 1,
        "batch_size": 100,
    }


def _general_config(tmp_path):
    return {
        "chunk": -1,
        "filter_list": ["length"],
        "restart": False,
        "allowed_ring_size": [3, 4, 5, 6, 7, 8],
        "allowed_charges": None,
        "allowed_spins": None,
        "keys_target": {"atom": [], "bond": [], "global": ["n_atoms"]},
        "keys_data": {"atom": [], "bond": [], "global": ["n_atoms"]},
        "lmdb_path": str(tmp_path) + "/",
        "lmdb_name": "test_graphs.lmdb",
        "lmdb_locations": {
            "geom_lmdb": str(MERGED / "merged_geom.lmdb"),
            "charge_lmdb": str(MERGED / "merged_charge.lmdb"),
            "bonds_lmdb": str(MERGED / "merged_bond.lmdb"),
        },
        "charge_filter": ["mbis"],
        "bond_filter": ["ibsi"],
        "bond_list_definition": "ibsi",
        "bonding_scheme": "bonding",
        "n_workers": 1,
        "batch_size": 100,
    }


def _assert_lmdb_embed_compatible(lmdb_path: Path):
    """
    Open the LMDB and verify the format matches what qtaim_embed expects:
    - "length" key present and is an int
    - All data keys are consecutive integer strings "0", "1", ...
    - Values decode to {"molecule_graph": bytes}
    - load_graph_from_serialized succeeds on each graph
    """
    env = lmdb.open(str(lmdb_path), subdir=False, readonly=True, lock=False)
    skip_keys = {"length", "scaled", "scaler_finalized", "processed_source_keys"}

    with env.begin() as txn:
        length_raw = txn.get(b"length")
        assert length_raw is not None, "Missing 'length' key in LMDB"
        num_graphs = pickle.loads(length_raw)
        assert isinstance(num_graphs, int) and num_graphs > 0, \
            f"'length' must be a positive int, got {num_graphs!r}"

        for idx in range(num_graphs):
            raw = txn.get(f"{idx}".encode("ascii"))
            assert raw is not None, \
                f"Key '{idx}' not found — keys are not consecutive integer strings"

            sample = pickle.loads(raw)
            assert isinstance(sample, dict) and "molecule_graph" in sample, \
                f"Key '{idx}': value must be a dict with 'molecule_graph', got {type(sample)}"

            graph = load_graph_from_serialized(sample["molecule_graph"])
            assert graph is not None, f"Key '{idx}': load_graph_from_serialized returned None"
            assert hasattr(graph, "node_types"), \
                f"Key '{idx}': deserialized object is not a PyG HeteroData graph"

    env.close()
    return num_graphs


class TestBaseConverterEmbedFormat:
    def test_integer_keys_and_dict_format(self, tmp_path):
        config = _base_config(tmp_path)
        converter = BaseConverter(config)
        info = converter.process(return_info=True)

        lmdb_path = Path(tmp_path) / "test_graphs.lmdb"
        assert lmdb_path.exists(), "Converter did not produce output LMDB"

        n = _assert_lmdb_embed_compatible(lmdb_path)
        assert n == info["processed_count"], \
            f"'length' key ({n}) != processed_count ({info['processed_count']})"

    def test_no_molecule_id_keys(self, tmp_path):
        config = _base_config(tmp_path)
        BaseConverter(config).process()

        lmdb_path = Path(tmp_path) / "test_graphs.lmdb"
        env = lmdb.open(str(lmdb_path), subdir=False, readonly=True, lock=False)
        skip_keys = {"length", "scaled", "scaler_finalized", "processed_source_keys"}

        with env.begin() as txn:
            cursor = txn.cursor()
            for key_bytes, _ in cursor:
                key = key_bytes.decode("ascii")
                if key in skip_keys:
                    continue
                assert key.isdigit(), \
                    f"Found non-integer key '{key}' — molecule ID keys must not appear in output"
        env.close()


class TestGeneralConverterEmbedFormat:
    def test_integer_keys_and_dict_format(self, tmp_path):
        config = _general_config(tmp_path)
        converter = GeneralConverter(config)
        info = converter.process(return_info=True)

        lmdb_path = Path(tmp_path) / "test_graphs.lmdb"
        assert lmdb_path.exists(), "Converter did not produce output LMDB"

        n = _assert_lmdb_embed_compatible(lmdb_path)
        assert n == info["processed_count"]
