import pytest

# pytest.skip("test_lmdb.py removed/archived; use tests/test_combined_lmdb_suite.py", allow_module_level=True)

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
from tests.utils_lmdb import check_graph_equality, get_benchmark_info
from qtaim_gen.source.utils.lmdbs import (
    parse_charge_data,
    parse_qtaim_data,
    parse_bond_data,
    parse_fuzzy_data,
    gather_structure_info,
    filter_bond_feats,
)

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
            if str(key.decode("ascii"))[2:-1] == lookup:
                return pkl.loads(value)

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

        cls.converter_general = GeneralConverter(
            cls.config_general, config_path=cls.config_qtaim_path
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

    def test_parsers(self):
        # test parsers by using general converter
        self.converter_general = GeneralConverter(
            self.config_general, config_path=self.config_qtaim_path
        )
        # self.converter_general.process()
        # get first key
        struct_raw = self.converter_general.__getitem__(
            "geom_lmdb", "b'orca5'".encode("ascii")
        )
        charge_dict_raw = self.converter_general.__getitem__(
            "charge_lmdb", "b'orca5'".encode("ascii")
        )
        qtaim_dict_raw = self.converter_general.__getitem__(
            "qtaim_lmdb", "b'orca5'".encode("ascii")
        )
        bond_dict_raw = self.converter_general.__getitem__(
            "bond_lmdb", "b'orca5'".encode("ascii")
        )
        fuzzy_dict_raw = self.converter_general.__getitem__(
            "fuzzy_lmdb", "b'orca5'".encode("ascii")
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
        bond_feats_bond_ibsi, bond_list_ibsi = parse_bond_data(
            bond_dict_raw, bond_list_definition="ibsi", bond_filter=["fuzzy", "ibsi"]
        )

        # fuzzy info
        atom_feats_fuzzy, global_feats_fuzzy = parse_fuzzy_data(
            fuzzy_dict_raw, global_feats["n_atoms"], fuzzy_filter=None
        )

        bond_feats_qtaim = {}
        atom_feats_qtaim = {i: {} for i in range(global_feats["n_atoms"])}
        # this tests merging in qtaim as well as parsing

        (
            atom_keys,
            bond_keys,
            atom_feats_qtaim,
            bond_feats_qtaim,
            connected_bond_paths,
        ) = parse_qtaim_data(
            dict_qtaim=qtaim_dict_raw,
            atom_feats=atom_feats_qtaim,
            bond_feats=bond_feats_qtaim,
        )

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
    
    def test_filter_bond_feats(self):
        """Test the filter_bond_feats function with various inputs."""
        # Create test data
        bond_feats = {
            (0, 1): {'ibsi': 1.2, 'fuzzy': 0.8},
            (0, 2): {'ibsi': 0.5, 'fuzzy': 0.3},
            (1, 2): {'ibsi': 0.3, 'fuzzy': 0.2},
            (2, 3): {'ibsi': 0.9, 'fuzzy': 0.6},
            (3, 4): {'ibsi': 0.7, 'fuzzy': 0.4},
        }
        
        # Test 1: Filter with a list of bonds
        bond_list_1 = [(0, 1), (1, 2), (3, 4)]
        filtered_1 = filter_bond_feats(bond_feats, bond_list_1)
        
        assert len(filtered_1) == 3, f"Expected 3 bonds, got {len(filtered_1)}"
        assert (0, 1) in filtered_1, "Bond (0, 1) should be in filtered result"
        assert (1, 2) in filtered_1, "Bond (1, 2) should be in filtered result"
        assert (3, 4) in filtered_1, "Bond (3, 4) should be in filtered result"
        assert (0, 2) not in filtered_1, "Bond (0, 2) should not be in filtered result"
        assert (2, 3) not in filtered_1, "Bond (2, 3) should not be in filtered result"
        
        # Verify feature values are preserved
        assert filtered_1[(0, 1)] == bond_feats[(0, 1)], "Features should be unchanged"
        assert filtered_1[(1, 2)] == bond_feats[(1, 2)], "Features should be unchanged"
        
        # Test 2: Filter with a dict of bonds
        bond_list_2 = {(0, 2): None, (2, 3): None}
        filtered_2 = filter_bond_feats(bond_feats, bond_list_2)
        
        assert len(filtered_2) == 2, f"Expected 2 bonds, got {len(filtered_2)}"
        assert (0, 2) in filtered_2, "Bond (0, 2) should be in filtered result"
        assert (2, 3) in filtered_2, "Bond (2, 3) should be in filtered result"
        
        # Test 3: Filter with reversed bond tuples (should still match due to normalization)
        bond_list_3 = [(1, 0), (2, 1)]  # Reversed versions of (0, 1) and (1, 2)
        filtered_3 = filter_bond_feats(bond_feats, bond_list_3)
        
        assert len(filtered_3) == 2, f"Expected 2 bonds, got {len(filtered_3)}"
        # The original keys should be preserved, not the reversed ones
        assert (0, 1) in filtered_3, "Bond (0, 1) should be in filtered result"
        assert (1, 2) in filtered_3, "Bond (1, 2) should be in filtered result"
        
        # Test 4: Filter with empty bond list
        bond_list_4 = []
        filtered_4 = filter_bond_feats(bond_feats, bond_list_4)
        
        assert len(filtered_4) == 0, f"Expected 0 bonds, got {len(filtered_4)}"
        
        # Test 5: Filter with all bonds
        bond_list_5 = list(bond_feats.keys())
        filtered_5 = filter_bond_feats(bond_feats, bond_list_5)
        
        assert len(filtered_5) == len(bond_feats), \
            f"Expected {len(bond_feats)} bonds, got {len(filtered_5)}"
        for key in bond_feats.keys():
            assert key in filtered_5, f"Bond {key} should be in filtered result"
            assert filtered_5[key] == bond_feats[key], \
                f"Features for bond {key} should be unchanged"
    
    def test_parser_merge(self):
        # test parsers by using general converter
        self.converter_general = GeneralConverter(
            self.config_general, config_path=self.config_qtaim_path
        )
        # self.converter_general.process()
        # get first key
        struct_raw = self.converter_general.__getitem__(
            "geom_lmdb", "b'orca5'".encode("ascii")
        )
        charge_dict_raw = self.converter_general.__getitem__(
            "charge_lmdb", "b'orca5'".encode("ascii")
        )
        qtaim_dict_raw = self.converter_general.__getitem__(
            "qtaim_lmdb", "b'orca5'".encode("ascii")
        )
        bond_dict_raw = self.converter_general.__getitem__(
            "bond_lmdb", "b'orca5'".encode("ascii")
        )
        fuzzy_dict_raw = self.converter_general.__getitem__(
            "fuzzy_lmdb", "b'orca5'".encode("ascii")
        )

        
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
            bond_dict_raw, bond_list_definition="fuzzy", bond_filter=["fuzzy", "ibsi"]
        )
        bond_feats_bond_ibsi, bond_list_ibsi = parse_bond_data(
            bond_dict_raw, bond_list_definition="ibsi", bond_filter=["fuzzy", "ibsi"]
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
            atom_feats=atom_feats,
            bond_feats=bond_feats,
        )

        # bond definition options
        # 1 - connected_bond_paths
        # 2 - bond_list_fuzzy
        # 3 - bond_list_ibsi
        # 4 - bond_list / bond cutoffs 
        
        # Test filter_bond_feats with different bond list definitions
        # Store original bond_feats for comparison
        original_bond_feats = bond_feats.copy()
        
        # Pre-compute normalized sets for efficient comparison (avoid O(n²) complexity)
        normalized_connected_paths = {tuple(sorted(b)) for b in connected_bond_paths}
        normalized_fuzzy = {tuple(sorted(b)) for b in bond_list_fuzzy}
        normalized_ibsi = {tuple(sorted(b)) for b in bond_list_ibsi}
        normalized_struct = {tuple(sorted(k)) for k in bond_list.keys()}
        normalized_original_keys = {tuple(sorted(k)) for k in original_bond_feats.keys()}
        
        # Test 1: Filter with connected_bond_paths (from QTAIM)
        filtered_bond_feats_qtaim = filter_bond_feats(bond_feats, connected_bond_paths)
        # Verify all keys in filtered result are in the original bond_list
        for bond_key in filtered_bond_feats_qtaim.keys():
            normalized_key = tuple(sorted(bond_key))
            assert normalized_key in normalized_connected_paths, \
                f"Bond {bond_key} not in connected_bond_paths"
        # Verify all bonds in connected_bond_paths that are in bond_feats are in filtered result
        normalized_filtered_qtaim = {tuple(sorted(k)) for k in filtered_bond_feats_qtaim.keys()}
        for bond in connected_bond_paths:
            normalized_bond = tuple(sorted(bond))
            if normalized_bond in normalized_original_keys:
                # Check if this bond appears in the filtered result
                assert normalized_bond in normalized_filtered_qtaim, \
                    f"Bond {bond} from connected_bond_paths not in filtered result"
        
        # Test 2: Filter with bond_list_fuzzy
        filtered_bond_feats_fuzzy = filter_bond_feats(bond_feats, bond_list_fuzzy)
        # Verify all keys in filtered result are in bond_list_fuzzy
        for bond_key in filtered_bond_feats_fuzzy.keys():
            normalized_key = tuple(sorted(bond_key))
            assert normalized_key in normalized_fuzzy, \
                f"Bond {bond_key} not in bond_list_fuzzy"
        # Verify all bonds in bond_list_fuzzy that exist in bond_feats are in filtered result
        normalized_filtered_fuzzy = {tuple(sorted(k)) for k in filtered_bond_feats_fuzzy.keys()}
        for bond in bond_list_fuzzy:
            normalized_bond = tuple(sorted(bond))
            if normalized_bond in normalized_original_keys:
                assert normalized_bond in normalized_filtered_fuzzy, \
                    f"Bond {bond} from bond_list_fuzzy not in filtered result"
        
        # Test 3: Filter with bond_list_ibsi
        filtered_bond_feats_ibsi = filter_bond_feats(bond_feats, bond_list_ibsi)
        # Verify all keys in filtered result are in bond_list_ibsi
        for bond_key in filtered_bond_feats_ibsi.keys():
            normalized_key = tuple(sorted(bond_key))
            assert normalized_key in normalized_ibsi, \
                f"Bond {bond_key} not in bond_list_ibsi"
        
        # Test 4: Filter with bond_list dict (from structure bonds)
        filtered_bond_feats_struct = filter_bond_feats(bond_feats, bond_list)
        # Verify all keys in filtered result are in bond_list dict
        for bond_key in filtered_bond_feats_struct.keys():
            normalized_key = tuple(sorted(bond_key))
            assert normalized_key in normalized_struct, \
                f"Bond {bond_key} not in structure bond_list"
        
        # Test 5: Verify filtered results preserve bond feature values
        for bond_key, bond_value in filtered_bond_feats_qtaim.items():
            assert bond_key in original_bond_feats, \
                f"Filtered bond {bond_key} not in original bond_feats"
            assert bond_value == original_bond_feats[bond_key], \
                f"Bond features changed for {bond_key}: {bond_value} != {original_bond_feats[bond_key]}"



# create dummy to just run test_parsers and setups
if __name__ == "__main__":
    test_lmdb = TestConverters()
    test_lmdb.setup_class()
    test_lmdb.test_parsers()
    test_lmdb.test_parser_merge()
