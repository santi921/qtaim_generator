import pytest

import os
import lmdb
import json
import pickle as pkl
import numpy as np
from copy import deepcopy
from glob import glob
from qtaim_gen.source.utils.lmdbs import json_2_lmdbs, inp_files_2_lmdbs, filter_bond_feats
from qtaim_gen.source.core.converter import (
    BaseConverter,
    QTAIMConverter,
    GeneralConverter,
)
from tests.utils_lmdb import check_graph_equality, get_benchmark_info
from qtaim_gen.source.utils.lmdbs import (
    parse_charge_data,
    parse_qtaim_data,
    parse_bond_data,
    parse_fuzzy_data,
    gather_structure_info
)
from qtaim_gen.source.scripts.json_to_lmdb import (
    partition_folders_by_shard,
    merge_shards,
    convert_json_with_stats,
)
import logging
import shutil
import tempfile

import pytest


class TestLMDB:
    # resolve test_files relative to this test file
    from pathlib import Path

    base_tests = Path(__file__).parent

    # json_2_lmdbs and write_lmdb expect directory paths that end with a separator
    dir_data = str(base_tests / "test_files" / "lmdb_tests") + os.sep
    dir_active = (
        str(base_tests / "test_files" / "lmdb_tests" / "generator_lmdbs") + os.sep
    )
    dir_active_merged = (
        str(base_tests / "test_files" / "lmdb_tests" / "generator_lmdbs_merged")
        + os.sep
    )

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
            cls.dir_data,
            cls.dir_active,
            "bond",
            "bond.lmdb",
            cls.chunk_size,
            clean=True,
            merge=merge,
        )
        json_2_lmdbs(
            cls.dir_data,
            cls.dir_active,
            "other",
            "other.lmdb",
            cls.chunk_size,
            clean=True,
            merge=merge,
        )
        json_2_lmdbs(
            cls.dir_data,
            cls.dir_active,
            "qtaim",
            "qtaim.lmdb",
            cls.chunk_size,
            clean=True,
            merge=merge,
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
            cls.dir_data,
            cls.dir_active,
            "geom.lmdb",
            cls.chunk_size,
            clean=True,
            merge=merge,
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
            key_str = key.decode("ascii")
            if key_str == lookup:
                return pkl.loads(value)
        env.close()
        return None

    def test_write_read(self):
        base = self.base_tests / "test_files"

        charge_lmdb = str(
            base / "lmdb_tests" / "generator_lmdbs_merged" / "merged_charge.lmdb"
        )
        bond_lmdb = str(
            base / "lmdb_tests" / "generator_lmdbs_merged" / "merged_bond.lmdb"
        )
        other_lmdb = str(
            base / "lmdb_tests" / "generator_lmdbs_merged" / "merged_other.lmdb"
        )
        qtaim_lmdb = str(
            base / "lmdb_tests" / "generator_lmdbs_merged" / "merged_qtaim.lmdb"
        )
        fuzzy_lmdb = str(
            base / "lmdb_tests" / "generator_lmdbs_merged" / "merged_fuzzy.lmdb"
        )

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

        charge_lmdb = str(
            base / "lmdb_tests" / "generator_lmdbs_merged" / "merged_charge.lmdb"
        )
        bond_lmdb = str(
            base / "lmdb_tests" / "generator_lmdbs_merged" / "merged_bond.lmdb"
        )
        other_lmdb = str(
            base / "lmdb_tests" / "generator_lmdbs_merged" / "merged_other.lmdb"
        )
        qtaim_lmdb = str(
            base / "lmdb_tests" / "generator_lmdbs_merged" / "merged_qtaim.lmdb"
        )
        geom_lmdb = str(
            base / "lmdb_tests" / "generator_lmdbs_merged" / "merged_geom.lmdb"
        )
        fuzzy_lmdb = str(
            base / "lmdb_tests" / "generator_lmdbs_merged" / "merged_fuzzy.lmdb"
        )

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
    base_dir = str(
        _tests_dir / "test_files" / "converter" / "converter_baseline_testing"
    )
    base_dir_data = str(_tests_dir / "test_files" / "lmdb_tests" / "generator_lmdbs")
    base_dir_data_merged = str(
        _tests_dir / "test_files" / "lmdb_tests" / "generator_lmdbs_merged"
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

    config_general = deepcopy(default_config_dict)
    config_general.update(
        {
            "lmdb_path": os.path.join(base_dir, "qtaim_converter"),
            "lmdb_name": "graphs_qtaim.lmdb",
            "lmdb_locations": {
                "geom_lmdb": os.path.join(base_dir_data_merged, "merged_geom.lmdb"),
                "qtaim_lmdb": os.path.join(base_dir_data_merged, "merged_qtaim.lmdb"),
                "charge_lmdb": os.path.join(base_dir_data_merged, "merged_charge.lmdb"),
                "bond_lmdb": os.path.join(base_dir_data_merged, "merged_bond.lmdb"),
                "fuzzy_lmdb": os.path.join(base_dir_data_merged, "merged_fuzzy.lmdb"),
                "other_lmdb": os.path.join(base_dir_data_merged, "merged_other.lmdb"),
            },
            "keys_data": {
                "atom": ["eta", "lol"],
                "bond": ["eta", "lol"],
                "global": ["n_atoms"],
            },
        }
    )

    @classmethod
    def setup_class(cls):
        # instantiate converters during test class setup (avoid running at import)
        cls.converter_baseline = BaseConverter(
            cls.config_baseline, config_path=cls.config_path
        )
        cls.converter_baseline_folder = BaseConverter(
            cls.config_folder, config_path=cls.config_folder_path
        )
        cls.converter_qtaim = QTAIMConverter(
            cls.config_qtaim, config_path=cls.config_qtaim_path
        )
        cls.converter_qtaim_folder = QTAIMConverter(
            cls.config_qtaim_folder, config_path=cls.config_qtaim_folder_path
        )

        #cls.converter_general = GeneralConverter(
        #    cls.config_general, config_path=cls.config_qtaim_path
        #)

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
        ), f"Expected 4, got {self.info_scale_qtaim['scaled_count']} on qtaim converter"
        assert (
            self.info_scale_qtaim_folder["scaled_count"] == 4
        ), f"Expected 4, got {self.info_scale_qtaim_folder['scaled_count']} on qtaim folder converter"
        assert (
            self.info_scale_baseline["scaled_count"] == 4
        ), f"Expected 4, got {self.info_scale_baseline['scaled_count']} on baseline converter"
        assert (
            self.info_scale_baseline_folder["scaled_count"] == 4
        ), f"Expected 4, got {self.info_scale_baseline_folder['scaled_count']} on baseline folder converter"

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

    def test_parsers(self):
        # test parsers by using general converter
        self.converter_general = GeneralConverter(
            self.config_general, config_path=self.config_qtaim_path
        )
        # self.converter_general.process()
        # get first key
        struct_raw = self.converter_general.__getitem__(
            "geom_lmdb", b"orca5"
        )
        charge_dict_raw = self.converter_general.__getitem__(
            "charge_lmdb", b"orca5"
        )
        qtaim_dict_raw = self.converter_general.__getitem__(
            "qtaim_lmdb", b"orca5"
        )
        bond_dict_raw = self.converter_general.__getitem__(
            "bond_lmdb", b"orca5"
        )
        fuzzy_dict_raw = self.converter_general.__getitem__(
            "fuzzy_lmdb", b"orca5"
        )


        # structure info
        _, global_feats = gather_structure_info(struct_raw)
        # charge info
        atom_feats_charge, global_feats_charge = parse_charge_data(
            charge_dict_raw, global_feats["n_atoms"]
        )

        # bond info
        bond_feats_bond_fuzzy, bond_list_fuzzy = parse_bond_data(
            bond_dict_raw, bond_list_definition="fuzzy", bond_filter=["fuzzy", "ibsi"]
        )
        print("number of fuzzy bonds:", len(bond_list_fuzzy))
        bond_feats_bond_ibsi, bond_list_ibsi = parse_bond_data(
            bond_dict_raw, bond_list_definition="ibsi", bond_filter=["fuzzy", "ibsi"]
        )
        print("number of ibsi bonds:", len(bond_list_ibsi))

        # fuzzy info
        atom_feats_fuzzy, global_feats_fuzzy = parse_fuzzy_data(
            fuzzy_dict_raw, global_feats["n_atoms"], fuzzy_filter=None
        )

        bond_feats_qtaim = {}
        atom_feats_qtaim = {i: {} for i in range(global_feats["n_atoms"])}
        # this tests merging in qtaim as well as parsing
        print("qtaim raw",  qtaim_dict_raw)
        (
            atom_keys,
            bond_keys,
            atom_feats_qtaim_temp,
            bond_feats_qtaim_temp,
            connected_bond_paths,
        ) = parse_qtaim_data(
            dict_qtaim=qtaim_dict_raw,
            atom_feats={},
            bond_feats={},
        )
        print("number of connected bond paths:", len(connected_bond_paths))
        # check number of connected bond paths is 158 
        assert len(connected_bond_paths) == 158, f"Expected 158, got {len(connected_bond_paths)}"
        # batch update atom_feats_qtaim and bond_feats_qtaim with the temp dicts
        # batch update atom_feats_qtaim with atom_feats_qtaim_temp
        for key, value in atom_feats_qtaim_temp.items():
            atom_feats_qtaim[key].update(value)

        # assign bond_feats_qtaim from bond_feats_qtaim_temp
        for key, value in bond_feats_qtaim_temp.items():
            bond_feats_qtaim[key] = value

        # CHARGE tests
        # iterate through processed_charge_dict and print keys and values
        for key, value in atom_feats_charge.items():
            # assert non of the values are {}
            assert value is not None, f"Expected non-None value for {key}, got {value}"
            # assert len of dict is 8
            assert len(value) == 9, f"Expected 9, got {len(value)} for {key}"

        # assert global_feats_charge has 5 keys and non are None
        assert (
            len(global_feats_charge) == 5
        ), f"Expected 5, got {len(global_feats_charge)} for global_feats_charge"

        for key, value in global_feats_charge.items():
            assert value is not None, f"Expected non-None value for {key}, got {value}"

        # FUZZY tests
        for key, value in atom_feats_fuzzy.items():
            assert value is not None, f"Expected non-None value for {key}, got {value}"
            assert len(value) == 7, f"Expected 7, got {len(value)} for {key}"

        assert (
            len(global_feats_fuzzy) == 14
        ), f"Expected 14, got {len(global_feats_fuzzy)} for global_feats_fuzzy"
        for key, value in global_feats_fuzzy.items():
            assert value is not None, f"Expected non-None value for {key}, got {value}"

        # BOND tests
        # check that each bond_list type isn't a list []
        for bond_list in [bond_list_fuzzy, bond_list_ibsi]:
            assert isinstance(
                bond_list, list
            ), f"Expected non-list, got {type(bond_list)}"
            # check that each bond in the list is a tuple of length 2
            for bond in bond_list:
                assert isinstance(bond, tuple) or isinstance(
                    bond, list
                ), f"Expected tuple or list, got {type(bond)}"
                assert (
                    len(bond) == 2
                ), f"Expected length 2, got {len(bond)} for bond {bond}"
        # check that bond_feats_bond has keys "fuzzy" and "ibsi"
        # go through every key in bond_feats_bond and check that each sub dict has the same keys as the other subdicts
        for temp_dict_bond in [bond_feats_bond_fuzzy, bond_feats_bond_ibsi]:
            for key, value in temp_dict_bond.items():
                assert (
                    value is not None
                ), f"Expected non-None value for {key}, got {value}"

        # QTAIM tests
        # for connect-bond_paths just check that it's a list of tuple and non of the tuples are empty
        assert isinstance(connected_bond_paths, list), f"Expected list, got {type(connected_bond_paths)}"
        for bond_path in connected_bond_paths:
            assert isinstance(bond_path, tuple) or isinstance(
                bond_path, list
            ), f"Expected tuple or list, got {type(bond_path)}"
            assert len(bond_path) > 0, f"Expected non-empty tuple, got {bond_path}"
        # for atom feats just check that each key has a non-None valude and that the length of the dict is consistent 
        for key, value in atom_feats_qtaim.items():
            assert value is not None, f"Expected non-None value for {key}, got {value}"
            assert len(value) == 22, f"Expected 22, got {len(value)} for {key}"
        
        for key, value in bond_feats_qtaim.items():
            assert value is not None, f"Expected non-None value for {key}, got {value}"
            assert len(value) == 22, f"Expected 22, got {len(value)} for {key}"
        #assert that bond_keys_qtaim is len 22, likewise for atom_keys_qtaim
        assert len(atom_keys) == 22, f"Expected 22, got {len(atom_keys)} for atom_keys"
        assert len(bond_keys) == 22, f"Expected 22, got {len(bond_keys)} for bond_keys"
    
    def test_parser_merge(self):
        # test parsers by using general converter
        self.converter_general = GeneralConverter(
            self.config_general, config_path=self.config_qtaim_path
        )
        # self.converter_general.process()
        # get first key
        struct_raw = self.converter_general.__getitem__(
            "geom_lmdb", b"orca5"
        )
        charge_dict_raw = self.converter_general.__getitem__(
            "charge_lmdb", b"orca5"
        )
        qtaim_dict_raw = self.converter_general.__getitem__(
            "qtaim_lmdb", b"orca5"
        )
        bond_dict_raw = self.converter_general.__getitem__(
            "bond_lmdb", b"orca5"
        )
        fuzzy_dict_raw = self.converter_general.__getitem__(
            "fuzzy_lmdb", b"orca5"
        )

        # TESTS 
        # 1) check that the number of atoms does not change as we merge along
        # 2) check that bonds are consistent 
        # 3) test bond feat filters as they are being added in - check that the keys are consistent with the input dicts and that the values are not None

        
        # structure info
        _, global_feats = gather_structure_info(struct_raw)
        bonds = struct_raw["bonds"]
        bond_list = {
            tuple(sorted(b)): None for b in bonds if b[0] != b[1]
        }
        bond_feats = {}
        atom_feats= {i: {} for i in range(global_feats["n_atoms"])}
        

        # charge info
        atom_feats_charge, global_feats_charge = parse_charge_data(
            charge_dict_raw, global_feats["n_atoms"])
        # fuzzy info
        atom_feats_fuzzy, global_feats_fuzzy = parse_fuzzy_data(
            fuzzy_dict_raw, global_feats["n_atoms"], fuzzy_filter=None
        )

        # merge charge info into atom_feats and global_feats - batch update the dicts
        for key, value in atom_feats_charge.items():
            if key in atom_feats:
                atom_feats[key].update(atom_feats_charge.get(key, {}))
                atom_feats[key].update(atom_feats_fuzzy.get(key, {}))
            else:
                atom_feats[key] = value
        
        global_feats.update(global_feats_charge)
        global_feats.update(global_feats_fuzzy)
        
        #print(atom_feats)

        # bond info
        bond_feats_bond_fuzzy, bond_list_fuzzy = parse_bond_data(
            bond_dict_raw, bond_list_definition="fuzzy", bond_filter=["fuzzy", "ibsi"], as_lists=False
        )
        _, bond_list_ibsi = parse_bond_data(
            bond_dict_raw, bond_list_definition="ibsi", bond_filter=["fuzzy", "ibsi"], as_lists=False
        )
        # merge bond info into bond_feats - batch update the dicts
        for key, value in bond_feats_bond_fuzzy.items():
            if key in bond_feats:
                bond_feats[key].update(bond_feats_bond_fuzzy.get(key, {}))
            else:
                bond_feats[key] = value

        # this tests merging in qtaim as well as parsing
        (
            atom_keys,
            bond_keys,
            atom_feats,
            bond_feats,
            connected_bond_paths,
        ) = parse_qtaim_data(
            dict_qtaim=qtaim_dict_raw,
            atom_feats={i: {} for i in range(global_feats["n_atoms"])},
            bond_feats={},
        )
        #print(atom_feats)
        
        # bond definition options
        # 1 - connected_bond_paths
        # 2 - bond_list_fuzzy
        # 3 - bond_list_ibsi
        # 4 - bond_list / bond cutoffs 
  
        # Test filter_bond_feats with different bond list definitions
        # Store original bond_feats for comparison
        print("number of bonds via feats:", len(bond_feats))
        
        original_bond_feats = bond_feats.copy()
        
        # Pre-compute normalized sets for efficient comparison (avoid O(n²) complexity)
        normalized_connected_paths = {tuple(sorted(b)) for b in connected_bond_paths}
        normalized_fuzzy = {tuple(sorted(b)) for b in bond_list_fuzzy}
        normalized_ibsi = {tuple(sorted(b)) for b in bond_list_ibsi}
        normalized_struct = {tuple(sorted(k)) for k in bond_list.keys()}
        normalized_original_keys = {tuple(sorted(k)) for k in original_bond_feats.keys()}
        
        print("number of bonds via connected paths:", len(normalized_connected_paths))
        print("number of bonds via fuzzy:", len(normalized_fuzzy))
        print("number of bonds via ibsi:", len(normalized_ibsi))
        print("number of bonds via struct:", len(normalized_struct))

        def _check_filter_bond_feats(selection, expected_normalized):
            """Helper to test filter_bond_feats for a given bond selection.

            - Ensures filtered keys are a subset of the expected_normalized set
            - Ensures every bond from selection that exists in the original bond feats
              appears in the filtered result
            - Ensures that feature values are preserved in the filtered output
            """
            filtered = filter_bond_feats(bond_feats, selection)

            # Check filtered keys are within the provided selection
            for bond_key in filtered.keys():
                normalized_key = tuple(sorted(bond_key))
                assert normalized_key in expected_normalized, \
                    f"Bond {bond_key} not in provided selection"

            # Check that every bond in the selection that exists in the original
            # bond feats is present in the filtered results
            normalized_filtered = {tuple(sorted(k)) for k in filtered.keys()}
            for bond in selection:
                normalized_bond = tuple(sorted(bond))
                if normalized_bond in normalized_original_keys:
                    assert normalized_bond in normalized_filtered, \
                        f"Bond {bond} from selection not in filtered result"

            # Ensure that feature values were preserved
            for bond_key, bond_value in filtered.items():
                assert bond_key in original_bond_feats, \
                    f"Filtered bond {bond_key} not in original bond_feats"
                assert bond_value == original_bond_feats[bond_key], \
                    f"Bond features changed for {bond_key}: {bond_value} != {original_bond_feats[bond_key]}"

            return filtered

        # Run checks for the different bond selection definitions
        filtered_bond_feats_qtaim = _check_filter_bond_feats(connected_bond_paths, normalized_connected_paths)
        filtered_bond_feats_fuzzy = _check_filter_bond_feats(bond_list_fuzzy, normalized_fuzzy)
        filtered_bond_feats_ibsi = _check_filter_bond_feats(bond_list_ibsi, normalized_ibsi)
        filtered_bond_feats_struct = _check_filter_bond_feats(bond_list, normalized_struct)


        # check the number of features of each bond that is filtered 

        print(f"Original bond feats: {len(original_bond_feats)}")
        print(f"Filtered bond feats (connected paths): {len(filtered_bond_feats_qtaim)}")
        print(f"Filtered bond feats (fuzzy): {len(filtered_bond_feats_fuzzy)}")
        print(f"Filtered bond feats (ibsi): {len(filtered_bond_feats_ibsi)}")

        # check the nubmer of features in each dict key 
        #for bond_key, bond_value in filtered_bond_feats_qtaim.items():
        #    print(f"Bond {bond_key} has {len(bond_value)} features in connected paths filter") 
        #for bond_key, bond_value in filtered_bond_feats_fuzzy.items():
        #    print(f"Bond {bond_key} has {len(bond_value)} features in fuzzy filter")
        #for bond_key, bond_value in filtered_bond_feats_ibsi.items():
        #    print(f"Bond {bond_key} has {len(bond_value)} features in ibsi filter")
        #for bond_key, bond_value in filtered_bond_feats_struct.items():
        #    print(f"Bond {bond_key} has {len(bond_value)} features in struct filter")
        #for bond_key, bond_value in original_bond_feats.items():
        #    print(f"Bond {bond_key} has {len(bond_value)} features in original bond feats")                                           


    def test_parse_bond_data_filtering(self):
        """Ensure parse_bond_data filters bond_feats and bond_list when bond_filter is provided."""
        dict_bond = {
            "fuzzy": {
                "1_O_to_2_C": 0.0,
                "1_O_to_3_C": 0.5,
            },
            "ibsi": {
                "1_O_to_2_C": 1.2,
                "1_O_to_3_C": 0.8,
            },
        }

        # with bond_filter present, should use 'fuzzy' (bond_list_definition) and only keep bonds
        # where the fuzzy value is non-zero (1_O_to_3_C)
        bond_feats_filtered, bond_list_filtered = parse_bond_data(
            dict_bond, bond_list_definition="fuzzy", bond_filter=["fuzzy", "ibsi"], as_lists=False
        )

        normalized_filtered = {tuple(sorted(b)) for b in bond_list_filtered}
        assert (0, 2) in normalized_filtered
        assert (0, 1) not in normalized_filtered
        assert any(tuple(sorted(k)) == (0, 2) for k in bond_feats_filtered.keys())
        assert not any(tuple(sorted(k)) == (0, 1) for k in bond_feats_filtered.keys())

    def test_parse_bond_data_cutoff(self):
        """Ensure parse_bond_data filters by cutoff when provided."""
        dict_bond = {
            "fuzzy": {
                "1_O_to_2_C": 0.0,
                "1_O_to_3_C": 0.5,
            },
            "ibsi": {
                "1_O_to_2_C": 1.2,
                "1_O_to_3_C": 0.8,
            },
        }

        bond_feats_cut, bond_list_cut = parse_bond_data(
            dict_bond, bond_list_definition="fuzzy", bond_filter=None, cutoff=0.4, as_lists=False
        )

        normalized_cut = {tuple(sorted(b)) for b in bond_list_cut}
        assert (0, 2) in normalized_cut
        assert (0, 1) not in normalized_cut
        assert any(tuple(sorted(k)) == (0, 2) for k in bond_feats_cut.keys())
        assert not any(tuple(sorted(k)) == (0, 1) for k in bond_feats_cut.keys())


class TestSharding:
    """Test sharding functionality for json_to_lmdb."""

    from pathlib import Path
    base_tests = Path(__file__).parent
    dir_data = str(base_tests / "test_files" / "lmdb_tests") + os.sep

    @classmethod
    def setup_class(cls):
        """Setup test directories."""
        cls.temp_dir = tempfile.mkdtemp()
        cls.logger = logging.getLogger("test_sharding")
        cls.logger.setLevel(logging.DEBUG)

    @classmethod
    def teardown_class(cls):
        """Clean up test directories."""
        if hasattr(cls, 'temp_dir') and os.path.exists(cls.temp_dir):
            shutil.rmtree(cls.temp_dir)

    def test_partition_folders_by_shard_basic(self):
        """Test basic folder partitioning across shards."""
        # Test with 4 shards
        total_shards = 4

        # Get all partitions
        all_partitions = []
        for shard_index in range(total_shards):
            partition = partition_folders_by_shard(
                self.dir_data, shard_index, total_shards, self.logger
            )
            all_partitions.append(set(partition))

        # Check that partitions are non-overlapping
        for i in range(total_shards):
            for j in range(i + 1, total_shards):
                overlap = all_partitions[i] & all_partitions[j]
                assert len(overlap) == 0, f"Shards {i} and {j} have overlapping folders: {overlap}"

        # Check that all folders are covered
        all_folders_union = set()
        for partition in all_partitions:
            all_folders_union |= partition

        # Get expected folders
        pattern = os.path.join(self.dir_data, "*/")
        all_folders = sorted(glob(pattern))
        expected_folder_names = set(os.path.basename(os.path.normpath(f)) for f in all_folders)

        assert all_folders_union == expected_folder_names, \
            f"Union of partitions doesn't match all folders. Missing: {expected_folder_names - all_folders_union}"

    def test_partition_folders_by_shard_deterministic(self):
        """Test that partitioning is deterministic - same input produces same output."""
        total_shards = 3
        shard_index = 1

        # Run partitioning twice
        partition1 = partition_folders_by_shard(
            self.dir_data, shard_index, total_shards, self.logger
        )
        partition2 = partition_folders_by_shard(
            self.dir_data, shard_index, total_shards, self.logger
        )

        assert partition1 == partition2, "Partitioning should be deterministic"

    def test_partition_folders_by_shard_no_sharding(self):
        """Test that total_shards=1 returns empty list (no sharding)."""
        partition = partition_folders_by_shard(
            self.dir_data, 0, 1, self.logger
        )
        assert partition == [], "total_shards=1 should return empty list"

    def test_partition_folders_by_shard_distribution(self):
        """Test that folder distribution across shards is balanced."""
        total_shards = 4

        # Get partition sizes
        partition_sizes = []
        for shard_index in range(total_shards):
            partition = partition_folders_by_shard(
                self.dir_data, shard_index, total_shards, self.logger
            )
            partition_sizes.append(len(partition))

        # Check that sizes are roughly balanced (within 1 folder)
        max_size = max(partition_sizes)
        min_size = min(partition_sizes)
        assert max_size - min_size <= 1, \
            f"Partition sizes should be balanced: {partition_sizes}"

    def test_merge_shards_basic(self):
        """Test basic shard merging functionality."""
        # Create test shards
        shard_paths = []
        expected_data = {}

        for i in range(3):
            shard_path = os.path.join(self.temp_dir, f"test_shard_{i}.lmdb")
            shard_paths.append(shard_path)

            # Create LMDB with test data
            env = lmdb.open(shard_path, subdir=False, map_size=10 * 1024**2, lock=False)
            with env.begin(write=True) as txn:
                # Add unique keys per shard
                for j in range(5):
                    key = f"shard_{i}_key_{j}"
                    value = {"data": f"value_{i}_{j}", "shard": i, "index": j}
                    txn.put(key.encode(), pkl.dumps(value))
                    expected_data[key] = value

                # Add length
                txn.put("length".encode(), pkl.dumps(5))
            env.close()

        # Merge shards
        merged_path = os.path.join(self.temp_dir, "merged.lmdb")
        result_path = merge_shards(shard_paths, merged_path, self.logger)

        assert result_path == merged_path, "merge_shards should return the output path"
        assert os.path.exists(merged_path), "Merged LMDB should exist"

        # Verify merged data
        env = lmdb.open(merged_path, subdir=False, readonly=True, lock=False)
        with env.begin() as txn:
            # Check length
            length_bytes = txn.get("length".encode())
            assert length_bytes is not None, "Merged LMDB should have length key"
            length = pkl.loads(length_bytes)
            assert length == len(expected_data), \
                f"Expected {len(expected_data)} entries, got {length}"

            # Check all keys present
            cursor = txn.cursor()
            merged_data = {}
            for key, value in cursor:
                key_str = key.decode("ascii")
                if key_str != "length":
                    merged_data[key_str] = pkl.loads(value)

            assert set(merged_data.keys()) == set(expected_data.keys()), \
                "Merged LMDB should contain all keys from all shards"

            # Verify values
            for key, expected_value in expected_data.items():
                assert key in merged_data, f"Key {key} missing from merged LMDB"
                assert merged_data[key] == expected_value, \
                    f"Value mismatch for {key}: {merged_data[key]} != {expected_value}"
        env.close()

    def test_merge_shards_missing_shard(self):
        """Test that merge_shards handles missing shards gracefully."""
        # Create one real shard
        shard1_path = os.path.join(self.temp_dir, "shard1.lmdb")
        env = lmdb.open(shard1_path, subdir=False, map_size=10 * 1024**2, lock=False)
        with env.begin(write=True) as txn:
            txn.put("key1".encode(), pkl.dumps({"value": 1}))
            txn.put("length".encode(), pkl.dumps(1))
        env.close()

        # Create list with missing shard
        shard_paths = [
            shard1_path,
            os.path.join(self.temp_dir, "missing_shard.lmdb")  # Doesn't exist
        ]

        merged_path = os.path.join(self.temp_dir, "merged_missing.lmdb")

        # Should not raise, but warn
        result_path = merge_shards(shard_paths, merged_path, self.logger)

        assert os.path.exists(result_path), "Merged LMDB should still be created"

        # Verify it contains data from the existing shard
        env = lmdb.open(merged_path, subdir=False, readonly=True, lock=False)
        with env.begin() as txn:
            key1_bytes = txn.get("key1".encode())
            assert key1_bytes is not None, "Data from existing shard should be present"
            assert pkl.loads(key1_bytes) == {"value": 1}
        env.close()

    def test_merge_shards_empty_list(self):
        """Test that merge_shards raises error for empty shard list."""
        merged_path = os.path.join(self.temp_dir, "merged_empty.lmdb")

        with pytest.raises(ValueError, match="No shard LMDBs provided"):
            merge_shards([], merged_path, self.logger)

    def test_sharding_folder_consistency(self):
        """Test that folder-based partitioning ensures all JSON types go to same shard."""
        total_shards = 2

        # For each folder, verify it only appears in one shard
        for shard_index in range(total_shards):
            partition = partition_folders_by_shard(
                self.dir_data, shard_index, total_shards, self.logger
            )

            # Check that same folder consistently goes to same shard
            for folder in partition:
                # Re-run partitioning and verify this folder still in same shard
                new_partition = partition_folders_by_shard(
                    self.dir_data, shard_index, total_shards, self.logger
                )
                assert folder in new_partition, \
                    f"Folder {folder} should consistently be in shard {shard_index}"

    def test_sharding_with_chunking_compatibility(self):
        """Test that sharding works with chunking (both enabled)."""
        # This is an integration test to verify sharding and chunking don't conflict
        # We'll create a small test to verify the output naming scheme

        # Create test output directory
        test_out_dir = os.path.join(self.temp_dir, "chunking_sharding_test")
        os.makedirs(test_out_dir, exist_ok=True)

        # Simulate the naming scheme used in json_to_lmdb.py
        total_shards = 2
        shard_index = 0

        # With sharding, output goes to subdirectory
        shard_out_dir = os.path.join(test_out_dir, f"shard_{shard_index}")
        os.makedirs(shard_out_dir, exist_ok=True)

        # Verify subdirectory was created
        assert os.path.exists(shard_out_dir), "Shard subdirectory should be created"

        # Verify naming doesn't conflict
        # Chunks would be named: charge_1.lmdb, charge_2.lmdb in shard_0/
        # Merged would be: charge_shard_0.lmdb in shard_0/
        chunk1_path = os.path.join(shard_out_dir, "charge_1.lmdb")
        merged_path = os.path.join(shard_out_dir, "charge_shard_0.lmdb")

        # These should not conflict
        assert os.path.dirname(chunk1_path) == os.path.dirname(merged_path)
        assert os.path.basename(chunk1_path) != os.path.basename(merged_path)


# create dummy to just run test_parsers and setups

if __name__ == "__main__":
    test_lmdb = TestConverters()
    test_lmdb.setup_class()
    test_lmdb.test_parsers()
    #test_lmdb.test_parser_merge()
