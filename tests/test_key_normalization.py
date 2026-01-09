import os
import lmdb
import pickle as pkl
import numpy as np
from qtaim_gen.source.core.converter import BaseConverter
from qtaim_embed.data.lmdb import load_dgl_graph_from_serialized
from tests.utils_lmdb import get_first_graph


def test_key_normalization_and_scaling(tmp_path):
    # configure a small folder-based converter output to a temporary directory
    base_tests = os.path.dirname(__file__)
    input_folder = os.path.join(
        base_tests, "test_files", "lmdb_tests", "generator_lmdbs"
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
        "lmdb_path": str(tmp_path / "out"),
        "lmdb_name": "graphs_keynorm.lmdb",
        # use the existing test fixture folder inside the tests directory as input
        "lmdb_locations": {"geom_lmdb": input_folder},
    }

    # provide a config path so overwrite_config has a valid target
    config_path = os.path.join(config["lmdb_path"], "config.json")
    conv = BaseConverter(config, config_path=config_path)

    # process and capture a pre-scale graph
    conv.process(return_info=True)
    first_pre = get_first_graph(conv)

    # scale and ensure scaler reports scaled_count
    info_scale = conv.scale_graph_lmdb(return_info=True)
    assert info_scale is not None and info_scale.get("scaled_count", 0) > 0

    first_post = get_first_graph(conv)
    assert first_pre is not None and first_post is not None

    # ensure at least one feature tensor changed after scaling
    changed = False
    for node_level in first_pre.ndata["feat"].keys():
        ft1 = first_pre.ndata["feat"][node_level]
        ft2 = first_post.ndata["feat"][node_level]
        if not np.array_equal(ft1.cpu().numpy(), ft2.cpu().numpy()):
            changed = True
            break
    assert changed, "No feature values changed after scaling"

    # verify output LMDB keys do not look like repr(byte) (e.g., "b'0'")
    out_file = os.path.join(
        config["lmdb_path"],
        (
            config["lmdb_name"]
            if config["lmdb_name"].endswith(".lmdb")
            else f"{config['lmdb_name']}.lmdb"
        ),
    )
    env = lmdb.open(
        out_file, subdir=False, readonly=True, lock=False, readahead=True, meminit=False
    )
    with env.begin(write=False) as txn:
        cursor = txn.cursor()
        for key, _ in cursor:
            k = key.decode("ascii")
            if k in config["filter_list"]:
                continue
            assert not (
                k.startswith("b'") or k.startswith('b"')
            ), f"Found non-normalized key in output LMDB: {k}"
