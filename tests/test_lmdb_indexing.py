import os
import pickle
from copy import deepcopy
from qtaim_gen.source.core.converter import BaseConverter


def test_folder_indexing_maps_to_correct_env():
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
        "lmdb_path": os.path.join(base_tests, "out_index_test"),
        "lmdb_name": "graphs_index.lmdb",
        "lmdb_locations": {"geom_lmdb": input_folder},
    }

    config_path = os.path.join(config["lmdb_path"], "config.json")
    conv = BaseConverter(config, config_path=config_path)

    # basic sanity
    lmdb_info = conv.lmdb_dict["geom_lmdb"]
    total = lmdb_info["num_samples"]
    assert total == len(lmdb_info["keys"])

    # for each global index, compute expected env and key and compare to __getitem__
    for idx in range(total):
        # use the same algorithm as __getitem__ to determine env and key
        # find db_idx via bisect on keylen_cumulative
        # but instead of re-implementing, access the stored fields
        keylen_cum = lmdb_info["keylen_cumulative"]
        # determine db_idx
        db_idx = 0
        for i, cum in enumerate(keylen_cum):
            if idx < cum:
                db_idx = i
                break
        else:
            db_idx = len(keylen_cum) - 1

        if db_idx == 0:
            idx_in_db = idx
        else:
            idx_in_db = idx - keylen_cum[db_idx - 1]

        key_in_db = lmdb_info["keys_raw"][db_idx][idx_in_db]
        env = lmdb_info["envs"][db_idx]["env"]

        expected = pickle.loads(env.begin().get(key_in_db))
        got = conv.__getitem__("geom_lmdb", idx)

        # compare a few reliable fields
        assert expected is not None
        assert got is not None
        assert expected.get("spin") == got.get("spin")
        assert expected.get("charge") == got.get("charge")
        assert len(expected.get("molecule_graph")) == len(got.get("molecule_graph"))
