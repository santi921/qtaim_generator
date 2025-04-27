import os
import pickle
import lmdb
import json
import numpy as np
import bisect
import argparse
from copy import deepcopy
from typing import Dict, List, Tuple, Union, Any

from qtaim_embed.data.processing import (
    HeteroGraphStandardScalerIterative,
    HeteroGraphLogMagnitudeScaler,
)
from qtaim_embed.core.molwrapper import MoleculeWrapper
from qtaim_embed.utils.grapher import get_grapher

from qtaim_embed.data.lmdb import (
    serialize_dgl_graph, TransformMol, 
    load_dgl_graph_from_serialized
)

from qtaim_gen.source.core.lmdbs import (
    get_elements_from_structure_lmdb,
    get_elements_from_structure_lmdb_folder_list,
    parse_config_gen_to_embed,
    parse_charge_data,
    parse_qtaim_data,
)
from qtaim_gen.source.core.embed import (
    split_graph_labels,
    get_include_exclude_indices,
    build_and_featurize_graph
)


def clean_id(key: bytes) -> str:
    """
    Clean the ID of the molecule.

    Args:
        key (bytes): The ID of the molecule in bytes.

    Returns:
        str: The cleaned ID as a string.
    """
    if "'" in key.decode("ascii"):
        id = key.decode("ascii").split("'")[1]
    elif '"' in key.decode("ascii"):
        id = key.decode("ascii").split('"')[1]
    else:
        id = key.decode("ascii")
    return id


class QTAIMEmbedConverter:
    def __init__(
            self, 
            config_dict: Dict[str, Any]
        ):
        self.config_dict = config_dict
        self.restart = config_dict["restart"]
        
        # keys
        self.atom_keys_grapher_no_qtaim = None
        self.atom_keys = None
        self.bond_keys = None
        self.global_keys= None
        
        # indices of keys in graphs
        self.index_dict = {}
        self.index_dict_qtaim = {}
        
        # graphers
        self.grapher_qtaim = None
        self.grapher_no_qtaim = None
        
        # data
        self.single_lmdb = True
        self.lmdb_dict = self.pull_lmdbs()
        
        self.skip_keys = ["length", "scaled"]
        self.fail_log_dict = {
            "structure": [], 
            "qtaim": [],
            "charge": [], 
            "graph": [], 
            "scaler": []
        }

        ####################### Element Set ########################
        if "element_set" in self.config_dict.keys():
            element_set = self.config_dict["element_set"]
        else:
            if self.single_lmdb:
                element_set = get_elements_from_structure_lmdb(self.lmdb_dict["structure_lmdb"])
            else: 
                element_set = get_elements_from_structure_lmdb_folder_list(
                    self.lmdb_dict["structure_lmdbs"]
                )

        self.element_set = sorted(list(element_set))
 
        ####################### LMDB Dir Stuff ########################
        # make directory for lmdb
        if not os.path.exists(self.config_dict["lmdb_path_non_qtaim"]):
            os.makedirs(self.config_dict["lmdb_path_non_qtaim"])

        if not os.path.exists(self.config_dict["lmdb_path_qtaim"]):
            os.makedirs(self.config_dict["lmdb_path_qtaim"])
        
        if "filter_list" not in self.config_dict.keys():
            self.config_dict["filter_list"] = ["scaled", "length"]

        if "chunk" not in self.config_dict.keys():
            self.config_dict["chunk"] = -1

        if "lmdb_qtaim_name" not in config_dict.keys():
            assert "lmdb_non_qtaim_name" not in config_dict.keys(), "qtaim and non qtaim names must be not set at the same time"
            # TODO: folder lmdb wrapper --> this does folder outputs!!! (chunking)
            pass
            
        else: 
            self.file_qtaim = os.path.join(
                self.config_dict["lmdb_path_qtaim"], self.config_dict["lmdb_qtaim_name"]
            )
            
            self.file_non_qtaim = os.path.join(
                self.config_dict["lmdb_path_non_qtaim"], self.config_dict["lmdb_non_qtaim_name"]
            )

            if os.path.exists(self.file_qtaim) and self.restart:
                os.remove(self.file_qtaim)
            if os.path.exists(self.file_non_qtaim) and self.restart:
                os.remove(self.file_non_qtaim)

            self.db_qtaim = self.connect_db(self.file_qtaim)
            self.db_non_qtaim = self.connect_db(self.file_non_qtaim)


            self.feature_scaler_iterative = HeteroGraphStandardScalerIterative(
                features_tf=True, mean={}, std={}
            )
            self.label_scaler_iterative = HeteroGraphStandardScalerIterative(
                features_tf=False, mean={}, std={}
            )

            self.feature_scaler_iterative_qtaim = HeteroGraphStandardScalerIterative(
                features_tf=True, mean={}, std={}
            )
            self.label_scaler_iterative_qtaim = HeteroGraphStandardScalerIterative(
                features_tf=False, mean={}, std={}
            )

    
    def pull_lmdbs(self):

        lmdb_dict = {}

        assert (
            "structure_lmdb" in config_dict["lmdb_locations"].keys()
        ), "The config file must contain a key 'structure_lmdb'"

        

        for key in config_dict["lmdb_locations"].keys():
            lmdb_path = config_dict["lmdb_locations"][key]

            # chick if the path is a file or directory
            if not os.path.exists(lmdb_path):
                raise ValueError(f"Path '{lmdb_path}' does not exist.")
            
            if os.path.isdir(lmdb_path):
                # if the path is a directory, check if it contains lmdb files
                lmdb_dict[key] = self.connect_folder_of_dbs(lmdb_path, key)
                self.single_lmdb = False                
            else: 
                # single lmdb file 
                lmdb_dict[key] = self.connect_db(lmdb_path, with_meta=True)

        if self.single_lmdb:
            assert (
                lmdb_dict[key]['env'].stat()["entries"] > 0
            ), f"The LMDB file is empty: {lmdb_dict[key]}"
        else:
            for key in lmdb_dict.keys():
                assert (
                    lmdb_dict[key]['num_samples'] > 0
                ), f"The LMDB file is empty: {lmdb_dict[key]}" 

        return lmdb_dict


    def connect_db(self, lmdb_path=None, map_size=int(1099511627776 * 2), with_meta=False):
        env = lmdb.open(
            str(lmdb_path),
            subdir=False,
            readonly=False,
            lock=False,
            readahead=True,
            meminit=False,
            max_readers=1,
            map_async=True,
            map_size=map_size,
        )
        if with_meta: 
            length_entry = env.begin().get("length".encode("ascii"))
            _keys = []
            with env.begin(write=False) as txn:
                cursor = txn.cursor()
                for key, _ in cursor:
                    _keys.append(key)
            
            return {
                "env": env,
                "keys": _keys,
                "num_samples": length_entry,
            }
        return env

    
    def connect_folder_of_dbs(self, db_path, key=None):
        if not db_path.is_file():
            db_paths = sorted(db_path.glob(f"*{key}*.lmdb"))
            assert len(db_paths) > 0, f"No LMDBs found in '{db_path}'"
            # self.metadata_path = self.path / "metadata.npz"

            _keys = []
            envs = []
            for db_path in db_paths:
                cur_env = self.connect_db(db_path)
                envs.append(cur_env)

                # If "length" encoded as ascii is present, use that
                length_entry = cur_env.begin().get("length".encode("ascii"))
                if length_entry is not None:
                    num_entries = pickle.loads(length_entry)
                else:
                    # Get the number of stores data from the number of entries in the LMDB
                    num_entries = cur_env.stat()["entries"]

                # Append the keys (0->num_entries) as a list
                _keys.append(list(range(num_entries)))  # need to be dicts 

            keylens = [len(k) for k in _keys]  # need to be dicts 
            _keylen_cumulative = np.cumsum(keylens).tolist()  # need to be dicts 
            num_samples = sum(keylens)  # need to be dicts

            return{
                "envs": envs,
                "keys": _keys,
                "keylens": keylens,
                "keylen_cumulative": _keylen_cumulative,
                "num_samples": num_samples,
                "lmdb_paths": db_paths,
            }

        else: 
            # Throw error if the path is a file
            raise ValueError(f"Path '{db_path}' is a file, not a directory.")


    def scale_graphs_single(
        self, 
        lmdb_file: lmdb,
        label_scaler: Union[HeteroGraphStandardScalerIterative, HeteroGraphLogMagnitudeScaler],
        feature_scaler: Union[HeteroGraphStandardScalerIterative, HeteroGraphLogMagnitudeScaler],
    ) -> None:
        """
        Scale the graphs using the provided scaler. Read from LMDB file and apply the scaler to each graph.
        """


        with lmdb_file.begin(write=False) as txn_in:    
            cursor = txn_in.cursor()
            for key, value in cursor: 
                    
                if key.decode("ascii") not in self.config_dict["filter_list"]:
                    #print(key.decode("ascii"))
                    graph = load_dgl_graph_from_serialized(pickle.loads(value))
                    #print(graph)
                    graph = feature_scaler([graph])
                    graph = label_scaler(graph)
                    
                    txn = lmdb_file.begin(write=True)
                    txn.put(
                        f"{key}".encode("ascii"),
                        pickle.dumps(serialize_dgl_graph(graph[0], ret=True), protocol=-1),
                    )
                    txn.commit()

                if key.decode("ascii") == "length":
                    # get the length of the lmdb file
                    length = pickle.loads(value)
                    txn = lmdb_file.begin(write=True)
                    txn.put(
                        "length".encode("ascii"),
                        pickle.dumps(length, protocol=-1),
                    )
                    txn.commit()
                
                if key.decode("ascii") == "scaled":
                    # get the length of the lmdb file
                    txn = lmdb_file.begin(write=True)
                    txn.put(
                        "scaled".encode("ascii"),
                        pickle.dumps(True, protocol=-1),
                    )
                    txn.commit()

        """
        # use for testing later 
        with lmdb_file.begin(write=False) as txn_in:    
            cursor = txn_in.cursor()
            for key, value in cursor: 
                if key.decode("ascii") == "length":
                    graph = load_dgl_graph_from_serialized(pickle.loads(value))
                    #print(graph)
                    #print(graph.ndata["feat"]['global'])

        lmdb_file.close()
        """
        

    def scale_graph_lmdb(self):
        
        if self.config_dict["chunk"] == -1:
            db_qtaim = lmdb.open(
                self.file_qtaim,
                map_size=int(1099511627776 * 2),
                subdir=False,
                meminit=False,
                map_async=True,
            )

            db_non_qtaim = lmdb.open(
                self.file_non_qtaim,
                map_size=int(1099511627776 * 2),
                subdir=False,
                meminit=False,
                map_async=True,
            )

            scaler.scale_graphs_single(
                lmdb_file=db_qtaim,
                label_scaler=self.label_scaler_iterative_qtaim,
                feature_scaler=self.feature_scaler_iterative_qtaim,
            )

            scaler.scale_graphs_single(
                lmdb_file=db_non_qtaim,
                label_scaler=self.feature_scaler_iterative,
                feature_scaler=self.feature_scaler_iterative,
            )

          
        else: 
            pass
            # TODO: implement chunking


    def main_loop(
        self
    ) -> Dict[str, Union[HeteroGraphStandardScalerIterative, HeteroGraphLogMagnitudeScaler]]:
        """
        Main loop for processing the LMDB files and generating the graphs.
        """

        #with self.lmdb_dict["structure_lmdb"].begin(write=False) as txn_in:
            #cursor = txn_in.cursor()
            #for key, value in cursor:  # first loop for gathering statistics and averages
                
        if self.single_lmdb:
            keys_to_iterate = self.lmdb_dict["structure_lmdb"]["keys"]
        else:
            keys_to_iterate = []
            for key in self.lmdb_dict["structure_lmdbs"]:
                keys_to_iterate += self.lmdb_dict["structure_lmdbs"][key]["keys"]

        for key in keys_to_iterate:
            if key.decode("ascii") not in self.skip_keys:
                try: 
                    #value_structure = pickle.loads(value)
                    # structure keys
                    value_structure = self.__getitem__("structure_lmdb", key)
                    mol_graph = value_structure["molecule_graph"]
                    n_atoms = len(mol_graph)
                    spin = value_structure["spin"]
                    charge = value_structure["charge"]
                    bonds = value_structure["bonds"]

                    id = clean_id(key)

                    # global features
                    global_feats = {
                        "charge": charge,
                        "spin": spin,
                        "n_atoms": n_atoms,
                        "n_bonds": len(bonds),
                    }

                    # initialize atom_feats with dictionary whos keys are atom inds and empty dict as values
                    atom_feats = {i: {} for i in range(n_atoms)}
                    bond_feats = {}
                    bond_list = {tuple(sorted(b)): None for b in bonds if b[0] != b[1]}
                except: 
                    self.fail_log_dict["structure"].append(key.decode("ascii"))
                    continue

                try: 
                    #value_charge = pickle.loads(self.lmdb_dict["charge_lmdb"].get(key))
                    value_charge = self.__getitem__("charge_lmdb", key)
                    # parse charge data
                    atom_feats_charge, global_dipole_feats = parse_charge_data(
                        value_charge, n_atoms
                    )
                    # copy atom_feats to atom_feats_no_qtaim
                    atom_feats_no_qtaim = deepcopy(atom_feats)

                    if self.atom_keys_grapher_no_qtaim == None:
                        self.atom_keys_grapher_no_qtaim = list(atom_feats[0].keys())

                    global_feats.update(global_dipole_feats)
                    
                    if self.global_keys == None:
                        self.global_keys = list(global_feats.keys())

                    atom_feats.update(atom_feats_charge)
                except:
                    self.fail_log_dict["charge"].append(key.decode("ascii"))
                    continue

                try:
                    
                    #value_qtaim = pickle.loads(self.lmdb_dict["qtaim_lmdb"].get(key))
                    value_qtaim = self.__getitem__("qtaim_lmdb", key)
                    # parse qtaim data
                    atom_keys, bond_keys, atom_feats, bond_feats, connected_bond_paths = (
                        parse_qtaim_data(
                            value_qtaim, atom_feats, bond_feats, atom_keys, bond_keys
                        )
                    )
                    atom_keys_complete = atom_keys + self.atom_keys_grapher_no_qtaim
                except: 
                    self.fail_log_dict["qtaim"].append(key.decode("ascii"))
                    continue

                
                ############################# Molwrapper ####################################
                try: 
                    mol_wrapper_qtaim_bonds = MoleculeWrapper(
                        mol_graph,
                        functional_group=None,
                        free_energy=None,
                        id=id,
                        bonds=connected_bond_paths,
                        non_metal_bonds=connected_bond_paths,
                        atom_features=atom_feats,
                        bond_features=bond_feats,
                        global_features=global_feats,
                        original_atom_ind=None,
                        original_bond_mapping=None,
                    )

                    mol_wrapper = MoleculeWrapper(
                        mol_graph,
                        functional_group=None,
                        free_energy=None,
                        id=id,
                        bonds=bond_list,
                        non_metal_bonds=bond_list,
                        atom_features=atom_feats_no_qtaim,
                        bond_features=bond_feats,
                        global_features=global_feats,
                        original_atom_ind=None,
                        original_bond_mapping=None,
                    )
                    # print(mol_wrapper)

                    ############################# Grapher ####################################
                    if not self.grapher_qtaim:
                        self.grapher_qtaim = get_grapher(
                            element_set=self.element_set,
                            atom_keys=self.atom_keys_complete,  # todo
                            bond_keys=self.bond_keys,  # todo
                            global_keys=self.global_keys,
                            allowed_ring_size=self.config_dict["allowed_ring_size"],
                            allowed_charges=self.config_dict["allowed_charges"],
                            allowed_spins=self.config_dict["allowed_spins"],
                            self_loop=True,
                            atom_featurizer_tf=True,
                            bond_featurizer_tf=True,
                            global_featurizer_tf=True,
                        )

                    if not self.grapher_no_qtaim:
                        self.grapher_no_qtaim = get_grapher(
                            element_set=self.element_set,
                            atom_keys=self.atom_keys_grapher_no_qtaim,
                            bond_keys=[],
                            global_keys=self.global_keys,
                            allowed_ring_size=self.config_dict["allowed_ring_size"],
                            allowed_charges=self.config_dict["allowed_charges"],
                            allowed_spins=self.config_dict["allowed_spins"],
                            self_loop=True,
                            atom_featurizer_tf=True,
                            bond_featurizer_tf=True,
                            global_featurizer_tf=True,
                        )

                    # graph generation
                    graph_no_qtaim = build_and_featurize_graph(
                        self.grapher_no_qtaim, mol_wrapper
                    )
                    graph = build_and_featurize_graph(
                        self.grapher_qtaim, mol_wrapper_qtaim_bonds
                    )

                    if self.index_dict == {}:
                        key_target = {
                            "atom": self.atom_keys_grapher_no_qtaim,
                            "bond": [],
                            "global": self.global_keys,
                        }
                        # print("target keys no qtaim: ", key_target)
                        self.index_dict = get_include_exclude_indices(
                            feat_names=self.grapher_no_qtaim.feat_names, target_dict=key_target
                        )

                    if self.index_dict_qtaim == {}:
                        key_target_qtaim = {
                            "atom": self.atom_keys,
                            "bond": self.bond_keys,
                            "global": self.global_keys,
                        }
                        # print("target keys qtaim: ", key_target_qtaim)
                        self.index_dict_qtaim = get_include_exclude_indices(
                            feat_names=self.grapher_qtaim.feat_names,
                            target_dict=key_target_qtaim,
                        )

                    # split graph into features and labels
                    split_graph_labels(
                        graph_no_qtaim,
                        include_names=self.index_dict["include_names"],
                        include_locs=self.index_dict["include_locs"],
                        exclude_locs=self.index_dict["exclude_locs"],
                    )

                    split_graph_labels(
                        graph,
                        include_names=self.index_dict_qtaim["include_names"],
                        include_locs=self.index_dict_qtaim["include_locs"],
                        exclude_locs=self.index_dict_qtaim["exclude_locs"],
                    )
                except:
                    self.fail_log_dict["graph"].append(key.decode("ascii"))
                    continue
                
                try: 
                    self.feature_scaler_iterative_qtaim.update([graph])
                    self.label_scaler_iterative_qtaim.update([graph])

                    self.feature_scaler_iterative.update([graph_no_qtaim])
                    self.label_scaler_iterative.update([graph_no_qtaim])

                    txn = self.db_qtaim.begin(write=True)
                    txn.put(
                        f"{key}".encode("ascii"),
                        pickle.dumps(serialize_dgl_graph(graph, ret=True), protocol=-1),
                    )
                    txn.commit()

                    txn = self.db_non_qtaim.begin(write=True)
                    txn.put(
                        f"{key}".encode("ascii"),
                        pickle.dumps(serialize_dgl_graph(graph_no_qtaim, ret=True), protocol=-1),
                    )
                    txn.commit()         
                except:
                    self.fail_log_dict["scaler"].append(key.decode("ascii"))
                    continue
        
            self.feature_scaler_iterative.finalize()
            self.label_scaler_iterative.finalize()
            self.feature_scaler_iterative_qtaim.finalize()
            self.label_scaler_iterative_qtaim.finalize()
        
        txn = self.db_qtaim.begin(write=True)
        txn.put("scaled".encode("ascii"), pickle.dumps(False, protocol=-1))
        txn.commit()

        txn = self.db_non_qtaim.begin(write=True)
        txn.put("scaled".encode("ascii"), pickle.dumps(False, protocol=-1))
        txn.commit()

        self.db_qtaim.close()
        self.db_non_qtaim.close()

    
    def __getitem__(self, key_lmdb, idx):
        """
        Get the item from the LMDB file or folder. Idx is an index for folders but a key for single lmdbs.
        Args:
            env: The LMDB environment.
            key_lmdb: The key for the LMDB file or folder.
            idx: The index of the item to get.
        Returns:
            data_object: The data object retrieved from the LMDB.
        """

        if not self.single_lmdb:
            # Figure out which db this should be indexed from.
            db_idx = bisect.bisect(self.lmdb_dict[key_lmdb]["keylen_cumulative"], idx)
            # Extract index of element within that db.
            el_idx = idx
            if db_idx != 0:
                el_idx = idx - self.lmdb_dict[key_lmdb]["keylen_cumulative"][db_idx - 1]
            assert el_idx >= 0

            # Return features.
            datapoint_pickled = (
                self.lmdb_dict[key_lmdb]["envs"][db_idx]
                .begin()
                .get(f"{self.lmdb_dict[key_lmdb]['keys'][db_idx][el_idx]}")
            )
            data_object = pickle.loads(datapoint_pickled)
            data_object.id = f"{db_idx}_{el_idx}"

        else:
            #value_charge = pickle.loads(self.lmdb_dict["charge_lmdb"].get(key))
            print("index: ", idx)
            datapoint_pickled = self.lmdb_dict[key_lmdb]['env'].begin().get(idx)
            # throw 
            #assert False, f"LMDB file {key_lmdb} is not a folder. Use the key directly."    
            data_object = pickle.loads(datapoint_pickled)


        return data_object


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LMDB to embed")
    
    parser.add_argument(
        "--config_path",
        type=str,
        default="./lmdb_config.json",
        help="Path to the config file for converting the database",
    )

    parser.add_argument(
        "--lmdb_name",
        type=str,
        default="molecule.lmdb",
        help="name of output lmdb file",
    )

    parser.add_argument(
        "--restart",
        action="store_true",
        help="Restart the process, will overwrite the existing lmdb file",
    )

    args = parser.parse_args()
    config_path = str(args.config_path)
    
    config_dict = parse_config_gen_to_embed(args.config_path, restart=bool(args.restart))        
    scaler = QTAIMEmbedConverter(config_dict)
    scaler.main_loop()
    scaler.scale_graph_lmdb()
