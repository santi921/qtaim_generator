
import os
import lmdb
import json
import pickle as pkl
import numpy as np 
from copy import deepcopy
from qtaim_gen.source.utils.lmdbs import json_2_lmdbs, inp_files_2_lmdbs
from qtaim_gen.source.core.converter import BaseConverter, QTAIMConverter, GeneralConverter
from qtaim_gen.source.utils.lmdbs import parse_config_gen_to_embed



class TestLMDB:
    
    dir_active = "./test_files/lmdb_tests/generator_lmdbs/"
    dir_active_merged = "./test_files/lmdb_tests/generator_lmdbs_merged/"
    # create folder if it doesn't exist
    if not os.path.exists(dir_active):
        os.makedirs(dir_active)
    if not os.path.exists(dir_active_merged):
        os.makedirs(dir_active_merged)

    chunk_size = 2
    
    # construct merged inputs
    merge=True
    json_2_lmdbs(
        dir_active_merged, dir_active_merged, "charge", "merged_charge.lmdb", chunk_size, clean=True, merge=merge
    )
    json_2_lmdbs(
        dir_active_merged, dir_active_merged, "bond", "merged_bond.lmdb", chunk_size, clean=True, merge=merge
    )
    json_2_lmdbs(
        dir_active_merged, dir_active_merged, "other", "merged_other.lmdb", chunk_size, clean=True, merge=merge
    )
    json_2_lmdbs(
        dir_active_merged, dir_active_merged, "qtaim", "merged_qtaim.lmdb", chunk_size, clean=True, merge=merge
    )

    json_2_lmdbs(
        dir_active_merged,
        dir_active_merged,
        "fuzzy_full",
        "merged_fuzzy.lmdb",
        chunk_size,
        clean=True,
        merge=merge
    )

    inp_files_2_lmdbs(
        dir_active_merged, dir_active_merged, "merged_geom.lmdb", chunk_size, clean=True, merge=merge
    )

    # construct non_merged inputs
    merge=False

    json_2_lmdbs(
        dir_active, dir_active, "charge", "merged_charge.lmdb", chunk_size, clean=True, merge=merge
    )
    json_2_lmdbs(
        dir_active, dir_active, "bond", "merged_bond.lmdb", chunk_size, clean=True, merge=merge
    )
    json_2_lmdbs(
        dir_active, dir_active, "other", "merged_other.lmdb", chunk_size, clean=True, merge=merge
    )
    json_2_lmdbs(
        dir_active, dir_active, "qtaim", "merged_qtaim.lmdb", chunk_size, clean=True, merge=merge
    )

    json_2_lmdbs(
        dir_active,
        dir_active,
        "fuzzy_full",
        "merged_fuzzy.lmdb",
        chunk_size,
        clean=True,
        merge=merge
    )

    inp_files_2_lmdbs(
        dir_active, dir_active, "merged_geom.lmdb", chunk_size, clean=True, merge=merge
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

        charge_lmdb = "./test_files/lmdb_tests/generator_lmdbs_merged/merged_charge.lmdb"
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
            dict_orca5_fuzzy['mbis_fuzzy_density']['45_Cl'] == orca5_rks_fuzzy_json["mbis_fuzzy_density"]["45_Cl"] 
        ), f"Expected {orca5_rks_fuzzy_json['mbis_fuzzy_density']['45_Cl'] }, got {dict_orca5_fuzzy['mbis_fuzzy_density']['45_Cl'] }"

    def test_merge(self):
        charge_lmdb = "./test_files/lmdb_tests/generator_lmdbs_merged/merged_charge.lmdb"
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


# tests
# - match to non-lmdb data, scaling number of atoms, number of output features
# - check that scaling can't happen more than once 
# - chunking matching non-chunked 
# - tests on restarts

class TestConverters:

    base_dir = os.path.join(".", "data", "converter_baseline_testing")

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
    config_qtaim.update({
        "lmdb_path": os.path.join(base_dir, "qtaim_converter"),
        "lmdb_name": "graphs_qtaim.lmdb",
        "lmdb_locations": {
            "geom_lmdb": os.path.join(base_dir, "inputs", "geom", "merged_geom.lmdb"),
            "qtaim_lmdb": os.path.join(base_dir, "inputs", "qtaim", "merged_qtaim.lmdb"),
        },
        "keys_data": {"atom": ["eta", "lol"], "bond": ["eta", "lol"], "global": ["n_atoms"]},
    })

    # qtaim folder config
    config_qtaim_folder_path = os.path.join(base_dir, "config_qtaim_folder.json")
    config_qtaim_folder = deepcopy(default_config_dict)
    config_qtaim_folder.update({
        "lmdb_path": os.path.join(base_dir, "qtaim_converter"),
        "lmdb_name": "graphs_folder_qtaim.lmdb",
        "lmdb_locations": {
            "geom_lmdb": os.path.join(base_dir, "inputs", "geom"),
            "qtaim_lmdb": os.path.join(base_dir, "inputs", "qtaim"),
        },
    })

    # baseline config (single merged geom)
    config_path = os.path.join(base_dir, "config.json")
    config_baseline = deepcopy(default_config_dict)
    config_baseline.update({
        "lmdb_path": base_dir,
        "lmdb_name": "graphs",
        "lmdb_locations": {
            "geom_lmdb": os.path.join(base_dir, "inputs", "geom", "merged_geom.lmdb"),
        },
    })

    # folder variant
    config_folder_path = os.path.join(base_dir, "config_folder.json")
    config_folder = deepcopy(default_config_dict)
    config_folder.update({
        "lmdb_path": base_dir,
        "lmdb_name": "graphs_folder",
        "lmdb_locations": {
            "geom_lmdb": os.path.join(base_dir, "inputs", "geom"),
        },
    })
