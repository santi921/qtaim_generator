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
from tests.utils_lmdb import get_first_graph, check_graph_equality, get_benchmark_info


class TestLMDB:
    # resolve test_files relative to this test file
    from pathlib import Path

    base_tests = Path(__file__).parent
    # json_2_lmdbs and write_lmdb expect directory paths that end with a separator
    dir_data = str(base_tests / "test_files" / "lmdb_tests") + os.sep
    dir_active = str(base_tests / "test_files" / "lmdb_tests" / "generator_lmdbs") + os.sep
    dir_active_merged = str(base_tests / "test_files" / "lmdb_tests" / "generator_lmdbs_merged") + os.sep

    chunk_size = 2

    @classmethod
    def setup_class(cls):
        # create folder if it doesn't exist
        os.makedirs(cls.dir_active, exist_ok=True)
        os.makedirs(cls.dir_active_merged, exist_ok=True)

        # construct merged inputs
        merge = True
        json_2_lmdbs(
            cls.dir_data,
            cls.dir_active_merged,
            "charge",
            "merged_charge.lmdb",
            cls.chunk_size,
            clean=True,
            merge=merge,
        )
        json_2_lmdbs(
            cls.dir_data,
            cls.dir_active_merged,
            "bond",
            "merged_bond.lmdb",
            cls.chunk_size,
            clean=True,
            merge=merge,
        )
        json_2_lmdbs(
            cls.dir_data,
            cls.dir_active_merged,
            "other",
            "merged_other.lmdb",
            cls.chunk_size,
            clean=True,
            merge=merge,
        )
        json_2_lmdbs(
            cls.dir_data,
            cls.dir_active_merged,
            "qtaim",
            "merged_qtaim.lmdb",
            cls.chunk_size,
            clean=True,
            merge=merge,
        )

        json_2_lmdbs(
            cls.dir_data,
            cls.dir_active_merged,
            "fuzzy_full",
            "merged_fuzzy.lmdb",
            cls.chunk_size,
            clean=True,
            merge=merge,
        )

        inp_files_2_lmdbs(
            cls.dir_data,
            cls.dir_active_merged,
            "merged_geom.lmdb",
            cls.chunk_size,
            clean=True,
            merge=merge,
        )

        # construct non_merged inputs
        merge = False

        json_2_lmdbs(
            cls.dir_data,
            cls.dir_active,
            "charge",
            "charge.lmdb",
            cls.chunk_size,
            clean=True,
            merge=merge,
        )
        json_2_lmdbs(
            cls.dir_data, cls.dir_active, "bond", "bond.lmdb", cls.chunk_size, clean=True, merge=merge
        )
        json_2_lmdbs(
            cls.dir_data, cls.dir_active, "other", "other.lmdb", cls.chunk_size, clean=True, merge=merge
        )
        json_2_lmdbs(
            cls.dir_data, cls.dir_active, "qtaim", "qtaim.lmdb", cls.chunk_size, clean=True, merge=merge
        )

        json_2_lmdbs(
            cls.dir_data,
            cls.dir_active,
            "fuzzy_full",
            "fuzzy.lmdb",
            cls.chunk_size,
            clean=True,
            merge=merge,
        )

        inp_files_2_lmdbs(
            cls.dir_data, cls.dir_active, "geom.lmdb", cls.chunk_size, clean=True, merge=merge
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
        base = self.base_tests / "test_files"

        charge_lmdb = str(base / "lmdb_tests" / "generator_lmdbs_merged" / "merged_charge.lmdb")
        bond_lmdb = str(base / "lmdb_tests" / "generator_lmdbs_merged" / "merged_bond.lmdb")
        other_lmdb = str(base / "lmdb_tests" / "generator_lmdbs_merged" / "merged_other.lmdb")
        qtaim_lmdb = str(base / "lmdb_tests" / "generator_lmdbs_merged" / "merged_qtaim.lmdb")
        fuzzy_lmdb = str(base / "lmdb_tests" / "generator_lmdbs_merged" / "merged_fuzzy.lmdb")

        orca5_rks_bond = str(base / "lmdb_tests" / "orca5_rks" / "bond.json")
        orca5_qtaim = str(base / "lmdb_tests" / "orca5" / "qtaim.json")
        orca5_uks_charge = str(base / "lmdb_tests" / "orca5_uks" / "charge.json")
        orca5_uks_fuzzy = str(base / "lmdb_tests" / "orca5_uks" / "fuzzy_full.json")
        orca6_rks_other = str(base / "lmdb_tests" / "orca6_rks" / "other.json")

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
        base = self.base_tests / "test_files"

        charge_lmdb = str(base / "lmdb_tests" / "generator_lmdbs_merged" / "merged_charge.lmdb")
        bond_lmdb = str(base / "lmdb_tests" / "generator_lmdbs_merged" / "merged_bond.lmdb")
        other_lmdb = str(base / "lmdb_tests" / "generator_lmdbs_merged" / "merged_other.lmdb")
        qtaim_lmdb = str(base / "lmdb_tests" / "generator_lmdbs_merged" / "merged_qtaim.lmdb")
        geom_lmdb = str(base / "lmdb_tests" / "generator_lmdbs_merged" / "merged_geom.lmdb")
        fuzzy_lmdb = str(base / "lmdb_tests" / "generator_lmdbs_merged" / "merged_fuzzy.lmdb")

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

    # use test-file-relative paths so tests work when run from repo root
    from pathlib import Path

    _tests_dir = Path(__file__).parent
    base_dir = str(_tests_dir / "test_files" / "converter" / "converter_baseline_testing")
    base_dir_data = str(_tests_dir / "test_files" / "lmdb_tests" / "generator_lmdbs")
    base_dir_data_merged = str(_tests_dir / "test_files" / "lmdb_tests" / "generator_lmdbs_merged")

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

    @classmethod
    def setup_class(cls):
        # instantiate converters during test class setup (avoid running at import)
        cls.converter_baseline = BaseConverter(cls.config_baseline, config_path=cls.config_path)
        cls.converter_baseline_folder = BaseConverter(
            cls.config_folder, config_path=cls.config_folder_path
        )
        cls.converter_qtaim = QTAIMConverter(cls.config_qtaim, config_path=cls.config_qtaim_path)
        cls.converter_qtaim_folder = QTAIMConverter(
            cls.config_qtaim_folder, config_path=cls.config_qtaim_folder_path
        )

        (
            cls.info_process_baseline,
            cls.first_graph_pre,
            cls.info_scale_baseline,
            cls.info_scale_baseline_restart,
            cls.first_graph_post,
        ) = get_benchmark_info(cls.converter_baseline)
        (
            cls.info_process_baseline_folder,
            cls.first_graph_pre_folder,
            cls.info_scale_baseline_folder,
            cls.info_scale_baseline_folder_restart,
            cls.first_graph_post_folder,
        ) = get_benchmark_info(cls.converter_baseline_folder)
        (
            cls.info_process_conv_qtaim,
            cls.first_graph_pre_qtaim,
            cls.info_scale_qtaim,
            cls.info_scale_qtaim_restart,
            cls.first_graph_pre_qtaim_post,
        ) = get_benchmark_info(cls.converter_qtaim)
        (
            cls.info_process_conv_qtaim_folder,
            cls.first_graph_pre_qtaim_folder,
            cls.info_scale_qtaim_folder,
            cls.info_scale_qtaim_folder_restart,
            cls.first_graph_pre_qtaim_folder_post,
        ) = get_benchmark_info(cls.converter_qtaim_folder)

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

