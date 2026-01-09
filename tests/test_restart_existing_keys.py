import os
import copy
from qtaim_gen.source.core.converter import BaseConverter


def test_restart_existing_keys_and_scaled_flag(tmp_path):
    base_tests = os.path.dirname(__file__)
    merged_geom = os.path.join(
        base_tests,
        "test_files",
        "lmdb_tests",
        "generator_lmdbs_merged",
        "merged_geom.lmdb",
    )

    config = {
        "chunk": -1,
        "filter_list": ["scaled", "length"],
        "restart": False,
        "allowed_ring_size": [3, 4, 5, 6, 7, 8],
        "allowed_charges": None,
        "allowed_spins": None,
        "keys_target": {"atom": [], "bond": [], "global": ["n_atoms"]},
        "keys_data": {"atom": [], "bond": [], "global": ["n_atoms"]},
        "lmdb_path": str(tmp_path / "out_restart"),
        "lmdb_name": "graphs_restart.lmdb",
        "lmdb_locations": {"geom_lmdb": merged_geom},
    }

    config_path = os.path.join(config["lmdb_path"], "config.json")
    conv1 = BaseConverter(config, config_path=config_path)

    info1 = conv1.process(return_info=True)
    assert info1["processed_count"] > 0

    info_scale = conv1.scale_graph_lmdb(return_info=True)
    assert info_scale is not None and info_scale.get("scaled_count", 0) > 0

    # create restart config and new converter
    config2 = copy.deepcopy(config)
    config2["restart"] = True
    conv2 = BaseConverter(config2, config_path=config_path)

    # existing_keys should be present and scaled flag should be True
    assert hasattr(conv2, "existing_keys")
    assert len(conv2.existing_keys) == info1["processed_count"]
    assert getattr(conv2, "scaled", False) is True

    # processing again should skip existing keys
    info2 = conv2.process(return_info=True)
    assert info2["processed_count"] == 0

    # scaling should detect already-scaled and skip (return None)
    info_scale2 = conv2.scale_graph_lmdb(return_info=True)
    assert info_scale2 is None
