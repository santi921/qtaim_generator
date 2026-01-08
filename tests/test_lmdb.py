import os
import lmdb
import json
import pickle as pkl
import numpy as np
from copy import deepcopy
from qtaim_gen.source.utils.lmdbs import json_2_lmdbs, inp_files_2_lmdbs
from qtaim_gen.source.core.converter import (
    BaseConverter,
    QTAIMConverter,
    GeneralConverter,
)
from qtaim_embed.data.lmdb import load_dgl_graph_from_serialized


def get_first_graph(converter):
    db_test = converter.connect_db(converter.file)
    graph = None
    with db_test.begin(write=False) as txn:
        cursor = txn.cursor()
        for key, value in cursor:
            try:
                # get first graph
                graph = deepcopy(load_dgl_graph_from_serialized(pkl.loads(value)))
                # print(f"Key: {key.decode('ascii')}, Graph: {graph}")
                break
            except:
                pass
    return graph


def check_graph_equality(graph1, graph2):
    for graph_level in ["feat", "labels"]:
        for node_level in graph1.ndata[graph_level].keys():
            ft_1 = graph1.ndata[graph_level][node_level]
            ft_2 = graph2.ndata[graph_level][node_level]
            # assert shape is unchanged
            assert (
                ft_1.shape == ft_2.shape
            ), f"Graph features shapes changed! {graph_level} {node_level}. Graph 1: {ft_1.shape}, Graph 2: {ft_2.shape}"
            # now assert the are not the saame values
            assert not np.array_equal(
                ft_1.cpu().numpy(), ft_2.cpu().numpy()
            ), f"Graph features are the same! {graph_level} {node_level}. Graph 1: {ft_1.cpu().numpy()}"


def get_benchmark_info(converter):
    info_process = converter.process(return_info=True)
    first_graph_pre = get_first_graph(converter)
    info_scale = converter.scale_graph_lmdb(return_info=True)
    info_scale_restart = converter.scale_graph_lmdb(return_info=True)
    first_graph_post = get_first_graph(converter)
    return (
        info_process,
        first_graph_pre,
        info_scale,
        info_scale_restart,
        first_graph_post,
    )


class TestLMDB:
    dir_data = "./test_files/lmdb_tests/"
    dir_active = "./test_files/lmdb_tests/generator_lmdbs/"
    dir_active_merged = "./test_files/lmdb_tests/generator_lmdbs_merged/"
    # create folder if it doesn't exist
    if not os.path.exists(dir_active):
        os.makedirs(dir_active)
    if not os.path.exists(dir_active_merged):
        os.makedirs(dir_active_merged)

    chunk_size = 2

    # construct merged inputs
    merge = True
    json_2_lmdbs(
        dir_data,
        dir_active_merged,
        "charge",
        "merged_charge.lmdb",
        chunk_size,
        clean=True,
        merge=merge,
    )
    json_2_lmdbs(
        dir_data,
        dir_active_merged,
        "bond",
        "merged_bond.lmdb",
        chunk_size,
        clean=True,
        merge=merge,
    )
    json_2_lmdbs(
        dir_data,
        dir_active_merged,
        "other",
        "merged_other.lmdb",
        chunk_size,
        clean=True,
        merge=merge,
    )
    json_2_lmdbs(
        dir_data,
        dir_active_merged,
        "qtaim",
        "merged_qtaim.lmdb",
        chunk_size,
        clean=True,
        merge=merge,
    )

    json_2_lmdbs(
        dir_data,
        dir_active_merged,
        "fuzzy_full",
        "merged_fuzzy.lmdb",
        chunk_size,
        clean=True,
        merge=merge,
    )

    inp_files_2_lmdbs(
        dir_data,
        dir_active_merged,
        "merged_geom.lmdb",
        chunk_size,
        clean=True,
        merge=merge,
    )

    # construct non_merged inputs
    merge = False

    json_2_lmdbs(
        dir_data,
        dir_active,
        "charge",
        "charge.lmdb",
        chunk_size,
        clean=True,
        merge=merge,
    )
    json_2_lmdbs(
        dir_data, dir_active, "bond", "bond.lmdb", chunk_size, clean=True, merge=merge
    )
    json_2_lmdbs(
        dir_data, dir_active, "other", "other.lmdb", chunk_size, clean=True, merge=merge
    )
    json_2_lmdbs(
        dir_data, dir_active, "qtaim", "qtaim.lmdb", chunk_size, clean=True, merge=merge
    )

    json_2_lmdbs(
        dir_data,
        dir_active,
        "fuzzy_full",
        "fuzzy.lmdb",
        chunk_size,
        clean=True,
        merge=merge,
    )

    inp_files_2_lmdbs(
        dir_data, dir_active, "geom.lmdb", chunk_size, clean=True, merge=merge
    )

    def read_helper(self, file, lookup):
        env = lmdb.open(
            file,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=True,
            meminit=False,
        )

        for key, value in env.begin().cursor():
            if str(key.decode("ascii"))[2:-1] == lookup:
                return pkl.loads(value)

    def test_write_read(self):

        charge_lmdb = (
            "./test_files/lmdb_tests/generator_lmdbs_merged/merged_charge.lmdb"
        )
        bond_lmdb = "./test_files/lmdb_tests/generator_lmdbs_merged/merged_bond.lmdb"
        other_lmdb = "./test_files/lmdb_tests/generator_lmdbs_merged/merged_other.lmdb"
        qtaim_lmdb = "./test_files/lmdb_tests/generator_lmdbs_merged/merged_qtaim.lmdb"
        fuzzy_lmdb = "./test_files/lmdb_tests/generator_lmdbs_merged/merged_fuzzy.lmdb"

        orca5_rks_bond = "./test_files/lmdb_tests/orca5_rks/bond.json"
        orca5_qtaim = "./test_files/lmdb_tests/orca5/qtaim.json"
        orca5_uks_charge = "./test_files/lmdb_tests/orca5_uks/charge.json"
        orca5_uks_fuzzy = "./test_files/lmdb_tests/orca5_uks/fuzzy_full.json"
        orca6_rks_other = "./test_files/lmdb_tests/orca6_rks/other.json"

        orca5_rks_bond_json = json.load(open(orca5_rks_bond, "r"))
        orca5_qtaim_json = json.load(open(orca5_qtaim, "r"))
        orca5_uks_charge_json = json.load(open(orca5_uks_charge, "r"))
        orca6_rks_other_json = json.load(open(orca6_rks_other, "r"))
        orca5_rks_fuzzy_json = json.load(open(orca5_uks_fuzzy, "r"))

        dict_orca5 = self.read_helper(bond_lmdb, "orca5_rks")
        dict_orca_5 = self.read_helper(qtaim_lmdb, "orca5")
        dict_orca5_uks = self.read_helper(charge_lmdb, "orca5_uks")
        dict_orca6 = self.read_helper(other_lmdb, "orca6_rks")
        dict_orca5_fuzzy = self.read_helper(fuzzy_lmdb, "orca5_uks")

        dict_orca5["ibsi"]["3_O_to_10_H"]
        orca5_rks_bond_json["ibsi"]["3_O_to_10_H"]

        assert (
            dict_orca5["ibsi"]["3_O_to_10_H"]
            == orca5_rks_bond_json["ibsi"]["3_O_to_10_H"]
        ), f"Expected {orca5_rks_bond_json['ibsi']['3_O_to_10_H']}, got {dict_orca5['ibsi']['3_O_to_10_H']}"
        assert (
            dict_orca_5["1"]["density_alpha"] == orca5_qtaim_json["1"]["density_alpha"]
        ), f"Expected {dict_orca_5['1']['density_alpha']}, got {orca5_qtaim_json['1']['density_alpha']}"
        assert (
            dict_orca5_uks["hirshfeld"]["charge"]["56_H"]
            == orca5_uks_charge_json["hirshfeld"]["charge"]["56_H"]
        ), f"Expected {orca5_uks_charge_json['hirshfeld']['charge']['56_H']}, got {dict_orca5_uks['hirshfeld']['charge']['56_H']}"
        assert (
            dict_orca6["mpp_full"] == orca6_rks_other_json["mpp_full"]
        ), f"Expected {orca6_rks_other_json['mpp_full'] }, got {dict_orca6['mpp_full'] }"

        assert (
            dict_orca5_fuzzy["mbis_fuzzy_density"]["45_Cl"]
            == orca5_rks_fuzzy_json["mbis_fuzzy_density"]["45_Cl"]
        ), f"Expected {orca5_rks_fuzzy_json['mbis_fuzzy_density']['45_Cl'] }, got {dict_orca5_fuzzy['mbis_fuzzy_density']['45_Cl'] }"

    def test_merge(self):
        charge_lmdb = (
            "./test_files/lmdb_tests/generator_lmdbs_merged/merged_charge.lmdb"
        )
        bond_lmdb = "./test_files/lmdb_tests/generator_lmdbs_merged/merged_bond.lmdb"
        other_lmdb = "./test_files/lmdb_tests/generator_lmdbs_merged/merged_other.lmdb"
        qtaim_lmdb = "./test_files/lmdb_tests/generator_lmdbs_merged/merged_qtaim.lmdb"
        geom_lmdb = "./test_files/lmdb_tests/generator_lmdbs_merged/merged_geom.lmdb"
        fuzzy_lmdb = "./test_files/lmdb_tests/generator_lmdbs_merged/merged_fuzzy.lmdb"

        for lmdb_file in [
            charge_lmdb,
            bond_lmdb,
            other_lmdb,
            qtaim_lmdb,
            fuzzy_lmdb,
            geom_lmdb,
        ]:
            env = lmdb.open(
                lmdb_file,
                subdir=False,
                readonly=True,
                lock=False,
                readahead=True,
                meminit=False,
            )

            for key, value in env.begin().cursor():
                if key.decode("ascii") == "length":
                    assert (
                        pkl.loads(value) == 4
                    ), f"Expected 4, got {pkl.loads(value)} on {lmdb_file}"


class TestConverters:

    base_dir = os.path.join(
        ".", "test_files", "converter", "converter_baseline_testing"
    )
    base_dir_data = os.path.join(".", "test_files", "lmdb_tests", "generator_lmdbs")
    base_dir_data_merged = os.path.join(
        ".", "test_files", "lmdb_tests", "generator_lmdbs_merged"
    )

    # default configuration
    default_config_dict = {
        "chunk": -1,
        "filter_list": ["scaled", "length"],
        "restart": False,
        "allowed_ring_size": [3, 4, 5, 6, 7, 8],
        "allowed_charges": None,
        "allowed_spins": None,
        "keys_target": {"atom": [], "bond": [], "global": ["n_atoms"]},
        "keys_data": {"atom": [], "bond": [], "global": ["n_atoms"]},
    }

    # qtaim single-file config
    config_qtaim_path = os.path.join(base_dir, "config_qtaim.json")
    config_qtaim = deepcopy(default_config_dict)
    config_qtaim.update(
        {
            "lmdb_path": os.path.join(base_dir, "qtaim_converter"),
            "lmdb_name": "graphs_qtaim.lmdb",
            "lmdb_locations": {
                "geom_lmdb": os.path.join(base_dir_data_merged, "merged_geom.lmdb"),
                "qtaim_lmdb": os.path.join(base_dir_data_merged, "merged_qtaim.lmdb"),
            },
            "keys_data": {
                "atom": ["eta", "lol"],
                "bond": ["eta", "lol"],
                "global": ["n_atoms"],
            },
        }
    )

    # qtaim folder config
    config_qtaim_folder_path = os.path.join(base_dir, "config_qtaim_folder.json")
    config_qtaim_folder = deepcopy(default_config_dict)
    config_qtaim_folder.update(
        {
            "lmdb_path": os.path.join(base_dir, "qtaim_converter_folder"),
            "lmdb_name": "graphs_folder_qtaim.lmdb",
            "lmdb_locations": {
                "geom_lmdb": os.path.join(base_dir_data_merged),
                "qtaim_lmdb": os.path.join(base_dir_data_merged),
            },
            "keys_data": {
                "atom": ["eta", "lol"],
                "bond": ["eta", "lol"],
                "global": ["n_atoms"],
            },
        }
    )

    # baseline config (single merged geom)
    config_path = os.path.join(base_dir, "config.json")
    config_baseline = deepcopy(default_config_dict)
    config_baseline.update(
        {
            "lmdb_path": os.path.join(base_dir, "baseline_converter"),
            "lmdb_name": "graphs",
            "lmdb_locations": {
                "geom_lmdb": os.path.join(base_dir_data_merged, "merged_geom.lmdb"),
            },
        }
    )

    # folder variant
    config_folder_path = os.path.join(base_dir, "config_folder.json")
    config_folder = deepcopy(default_config_dict)
    config_folder.update(
        {
            "lmdb_path": os.path.join(base_dir, "baseline_converter_folder"),
            "lmdb_name": "graphs_folder",
            "lmdb_locations": {"geom_lmdb": os.path.join(base_dir_data)},
        }
    )

    converter_baseline = BaseConverter(config_baseline, config_path=config_path)
    converter_baseline_folder = BaseConverter(
        config_folder, config_path=config_folder_path
    )
    converter_qtaim = QTAIMConverter(config_qtaim, config_path=config_qtaim_path)
    converter_qtaim_folder = QTAIMConverter(
        config_qtaim_folder, config_path=config_qtaim_folder_path
    )

    (
        info_process_baseline,
        first_graph_pre,
        info_scale_baseline,
        info_scale_baseline_restart,
        first_graph_post,
    ) = get_benchmark_info(converter_baseline)
    (
        info_process_baseline_folder,
        first_graph_pre_folder,
        info_scale_baseline_folder,
        info_scale_baseline_folder_restart,
        first_graph_post_folder,
    ) = get_benchmark_info(converter_baseline_folder)
    (
        info_process_conv_qtaim,
        first_graph_pre_qtaim,
        info_scale_qtaim,
        info_scale_qtaim_restart,
        first_graph_pre_qtaim_post,
    ) = get_benchmark_info(converter_qtaim)
    (
        info_process_conv_qtaim_folder,
        first_graph_pre_qtaim_folder,
        info_scale_qtaim_folder,
        info_scale_qtaim_folder_restart,
        first_graph_pre_qtaim_folder_post,
    ) = get_benchmark_info(converter_qtaim_folder)

    def test_restarts(self):
        # assert all the restart are None:
        assert (
            self.info_scale_qtaim_restart is None
        ), f"Expected None, got {self.info_scale_qtaim_restart}"
        assert (
            self.info_scale_qtaim_folder_restart is None
        ), f"Expected None, got {self.info_scale_qtaim_folder_restart}"
        assert (
            self.info_scale_baseline_restart is None
        ), f"Expected None, got {self.info_scale_baseline_restart}"
        assert (
            self.info_scale_baseline_folder_restart is None
        ), f"Expected None, got {self.info_scale_baseline_folder_restart}"

    def test_scaling_counts(self):
        # scaled counts should be 4 for each
        assert (
            self.info_scale_qtaim["scaled_count"] == 4
        ), f"Expected 4, got {self.info_scale_qtaim['scaled_count']}"
        assert (
            self.info_scale_qtaim_folder["scaled_count"] == 4
        ), f"Expected 4, got {self.info_scale_qtaim_folder['scaled_count']}"
        assert (
            self.info_scale_baseline["scaled_count"] == 4
        ), f"Expected 4, got {self.info_scale_baseline['scaled_count']}"
        assert (
            self.info_scale_baseline_folder["scaled_count"] == 4
        ), f"Expected 4, got {self.info_scale_baseline_folder['scaled_count']}"

    def test_process_counts(self):
        # process should also have 4 keys each
        assert (
            self.info_process_conv_qtaim["processed_count"] == 4
        ), f"Expected 4, got {self.info_process_conv_qtaim['processed_count']}"
        assert (
            self.info_process_conv_qtaim_folder["processed_count"] == 4
        ), f"Expected 4, got {self.info_process_conv_qtaim_folder['processed_count']}"
        assert (
            self.info_process_baseline["processed_count"] == 4
        ), f"Expected 4, got {self.info_process_baseline['processed_count']}"
        assert (
            self.info_process_baseline_folder["processed_count"] == 4
        ), f"Expected 4, got {self.info_process_baseline_folder['processed_count']}"

    def test_feat_names(self):
        # assert we get the same features from the qtaim converter and the qtaim folder converter
        for k, v in self.converter_qtaim.grapher.feat_names.items():
            assert len(v) == len(
                self.converter_qtaim_folder.grapher.feat_names[k]
            ), f"Expected {len(self.converter_qtaim_folder.grapher.feat_names[k])}, got {len(v)} for {k}"
            # assert feat_names are the same
            assert set(v) == set(
                self.converter_qtaim_folder.grapher.feat_names[k]
            ), f"Expected {set(self.converter_qtaim_folder.grapher.feat_names[k])}, got {set(v)} for {k}"

        # now the same for the baseline converters
        for k, v in self.converter_baseline.grapher.feat_names.items():
            assert len(v) == len(
                self.converter_baseline_folder.grapher.feat_names[k]
            ), f"Expected {len(self.converter_baseline_folder.grapher.feat_names[k])}, got {len(v)} for {k}"
            # assert feat_names are the same
            assert set(v) == set(
                self.converter_baseline_folder.grapher.feat_names[k]
            ), f"Expected {set(self.converter_baseline_folder.grapher.feat_names[k])}, got {set(v)} for {k}"

    def test_scaling_ops_single(self):
        # check that the features are not the same after scaling for qtaim converter
        check_graph_equality(
            self.first_graph_pre_qtaim, self.first_graph_pre_qtaim_post
        )
        check_graph_equality(self.first_graph_pre, self.first_graph_post)

    def test_scaling_ops_folder(self):
        # check that the features are not the same after scaling for qtaim folder converter
        check_graph_equality(
            self.first_graph_pre_qtaim_folder, self.first_graph_pre_qtaim_folder_post
        )
        # check that the features are not the same after scaling for baseline folder converter
        check_graph_equality(self.first_graph_pre_folder, self.first_graph_post_folder)


obj_lmdb = TestLMDB()
obj_lmdb.test_write_read()
obj_lmdb.test_merge()
obj_converters = TestConverters()
obj_converters.test_restarts()
obj_converters.test_scaling_counts()
obj_converters.test_process_counts()
obj_converters.test_feat_names()
obj_converters.test_scaling_ops_single()
obj_converters.test_scaling_ops_folder()
