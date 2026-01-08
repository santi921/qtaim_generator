import os
import numpy as np
from copy import deepcopy
from qtaim_gen.source.core.converter import BaseConverter
from tests.utils_lmdb import get_first_graph, check_graph_equality


def _base_config(tmp_path, lmdb_location, lmdb_name="graphs_scale_test.lmdb"):
    return {
        "chunk": -1,
        "filter_list": ["scaled", "length"],
        "restart": False,
        "allowed_ring_size": [3, 4, 5, 6, 7, 8],
        "allowed_charges": None,
        "allowed_spins": None,
        "keys_target": {"atom": [], "bond": [], "global": ["n_atoms"]},
        "keys_data": {"atom": [], "bond": [], "global": ["n_atoms"]},
        "lmdb_path": str(tmp_path / "out_scale"),
        "lmdb_name": lmdb_name,
        "lmdb_locations": {"geom_lmdb": lmdb_location},
    }


def test_scale_single_file(tmp_path):
    base_tests = os.path.dirname(__file__)
    merged_folder = os.path.join(base_tests, "test_files", "lmdb_tests", "generator_lmdbs_merged")
    geom_lmdb = os.path.join(merged_folder, "merged_geom.lmdb")

    config = _base_config(tmp_path, geom_lmdb, lmdb_name="graphs_single_scale.lmdb")
    cfg_path = os.path.join(config["lmdb_path"], "config.json")
    conv = BaseConverter(config, config_path=cfg_path)

    conv.process(return_info=True)
    first_pre = get_first_graph(conv)

    info_scale = conv.scale_graph_lmdb(return_info=True)
    assert info_scale is not None
    assert info_scale.get("scaled_count", 0) > 0

    # second call should detect already-scaled and skip
    info_scale_restart = conv.scale_graph_lmdb(return_info=True)
    assert info_scale_restart is None

    first_post = get_first_graph(conv)
    assert first_pre is not None and first_post is not None
    check_graph_equality(first_pre, first_post)


def test_scale_folder_input(tmp_path):
    base_tests = os.path.dirname(__file__)
    folder = os.path.join(base_tests, "test_files", "lmdb_tests", "generator_lmdbs")

    config = _base_config(tmp_path, folder, lmdb_name="graphs_folder_scale.lmdb")
    cfg_path = os.path.join(config["lmdb_path"], "config.json")
    conv = BaseConverter(config, config_path=cfg_path)

    conv.process(return_info=True)
    first_pre = get_first_graph(conv)

    info_scale = conv.scale_graph_lmdb(return_info=True)
    assert info_scale is not None
    assert info_scale.get("scaled_count", 0) > 0

    info_scale_restart = conv.scale_graph_lmdb(return_info=True)
    assert info_scale_restart is None

    first_post = get_first_graph(conv)
    assert first_pre is not None and first_post is not None
    check_graph_equality(first_pre, first_post)
