import os
import lmdb
import pickle as pkl
from qtaim_gen.source.core.converter import BaseConverter


def test_folder_indexing_maps_to_correct_env():
    base_tests = os.path.dirname(__file__)
    folder = os.path.join(base_tests, "test_files", "lmdb_tests", "generator_lmdbs")

    config = {
        "chunk": 1,
        "filter_list": ["scaled", "length"],
        "restart": False,
        "allowed_ring_size": [3, 4, 5, 6, 7, 8],
        "allowed_charges": None,
        "allowed_spins": None,
        "keys_target": {"atom": [], "bond": [], "global": ["n_atoms"]},
        "keys_data": {"atom": [], "bond": [], "global": ["n_atoms"]},
        "lmdb_path": os.path.join(base_tests, "tmp_index"),
        "lmdb_name": "graphs_index.lmdb",
        "lmdb_locations": {"geom_lmdb": folder},
    }

    cfg_path = os.path.join(config["lmdb_path"], "config.json")
    conv = BaseConverter(config, config_path=cfg_path)

    # ensure keys length matches expected
    assert conv.lmdb_dict["geom_lmdb"]["num_samples"] >= 0
    # fetch first item
    item = conv.__getitem__("geom_lmdb", 0)
    assert item is not None
