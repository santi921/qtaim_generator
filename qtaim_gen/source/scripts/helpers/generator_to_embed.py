import os
import pickle
import lmdb
import json
import numpy as np
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

def parse_into_graph_and_scalers(value_structure, value_charge, value_qtaim):
    """
    Parse the LMDB files into graphs and iterate on scalers.
    Args:
        value_structure: The structure data from the LMDB file.
        value_charge: The charge data from the LMDB file.
        value_qtaim: The QTAIM data from the LMDB file.
        
    Returns:
    """
    error_list = []
    try: 
        # structure keys
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
        error_list.append("structure")

    try: 
        # parse charge data
        atom_feats_charge, global_dipole_feats = parse_charge_data(
            value_charge, n_atoms
        )
        # copy atom_feats to atom_feats_no_qtaim
        atom_feats_no_qtaim = deepcopy(atom_feats)

        if atom_keys_grapher_no_qtaim == None:
            atom_keys_grapher_no_qtaim = list(atom_feats[0].keys())

        global_feats.update(global_dipole_feats)
        atom_feats.update(atom_feats_charge)
    except:
        error_list.append("charge")


    try:
        # parse qtaim data
        atom_keys, bond_keys, atom_feats, bond_feats, connected_bond_paths = (
            parse_qtaim_data(
                value_qtaim, atom_feats, bond_feats, atom_keys, bond_keys
            )
        )

        atom_keys_complete = atom_keys + atom_keys_grapher_no_qtaim
    except: 
        error_list.append("qtaim")

    if len(error_list) > 0:
        pass



def main_loop(
    lmdb_dict: Dict[str, lmdb.Environment],
    config_dict: Dict[str, Any],
    chunk: int = -1,
) -> Dict[str, Union[HeteroGraphStandardScalerIterative, HeteroGraphLogMagnitudeScaler]]:
    """
    Main loop for processing the LMDB files and generating the graphs.
    """

    (
        atom_keys,
        bond_keys,
        atom_keys_grapher_no_qtaim
    ) = (None, None, None)

    grapher_qtaim, grapher_no_qtaim = None, None
    index_dict, index_dict_qtaim = {}, {}

    feature_scaler_iterative = HeteroGraphStandardScalerIterative(
        features_tf=True, mean={}, std={}
    )
    label_scaler_iterative = HeteroGraphStandardScalerIterative(
        features_tf=False, mean={}, std={}
    )

    feature_scaler_iterative_qtaim = HeteroGraphStandardScalerIterative(
        features_tf=True, mean={}, std={}
    )
    label_scaler_iterative_qtaim = HeteroGraphStandardScalerIterative(
        features_tf=False, mean={}, std={}
    )

    # make directory for lmdb
    if not os.path.exists(config_dict["lmdb_path_non_qtaim"]):
        os.makedirs(config_dict["lmdb_path_non_qtaim"])

    if not os.path.exists(config_dict["lmdb_path_qtaim"]):
        os.makedirs(config_dict["lmdb_path_qtaim"])

    file_qtaim = os.path.join(
        config_dict["lmdb_path_qtaim"], config_dict["lmdb_qtaim_name"]
    )
    
    file_non_qtaim = os.path.join(
        config_dict["lmdb_path_non_qtaim"], config_dict["lmdb_non_qtaim_name"]
    )

    db_qtaim = lmdb.open(
        file_qtaim,
        map_size=int(1099511627776 * 2),
        subdir=False,
        meminit=False,
        map_async=True,
    )

    db_non_qtaim = lmdb.open(
        file_non_qtaim,
        map_size=int(1099511627776 * 2),
        subdir=False,
        meminit=False,
        map_async=True,
    )

    fail_log_dict = {
        "structure": [], 
        "qtaim": [],
        "charge": [], 
        "graph": [], 
        "scaler": []
    }

    with lmdb_dict["structure_lmdb"].begin(write=False) as txn_in, lmdb_dict[
        "qtaim_lmdb"
    ].begin(write=False) as txn_qtaim, lmdb_dict["charge_lmdb"].begin(
        write=False
    ) as txn_charge:
        
        cursor = txn_in.cursor()
        for key, value in cursor:  # first loop for gathering statistics and averages
            if key.decode("ascii") != "length":
                try: 
                    value_structure = pickle.loads(value)
                    # structure keys
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
                    fail_log_dict["structure"].append(key.decode("ascii"))
                    continue

                try: 
                    value_charge = pickle.loads(txn_charge.get(key))
                    # parse charge data
                    atom_feats_charge, global_dipole_feats = parse_charge_data(
                        value_charge, n_atoms
                    )
                    # copy atom_feats to atom_feats_no_qtaim
                    atom_feats_no_qtaim = deepcopy(atom_feats)

                    if atom_keys_grapher_no_qtaim == None:
                        atom_keys_grapher_no_qtaim = list(atom_feats[0].keys())

                    global_feats.update(global_dipole_feats)
                    atom_feats.update(atom_feats_charge)
                except:
                    fail_log_dict["charge"].append(key.decode("ascii"))
                    continue

                try:
                    value_qtaim = pickle.loads(txn_qtaim.get(key))

                    # parse qtaim data
                    atom_keys, bond_keys, atom_feats, bond_feats, connected_bond_paths = (
                        parse_qtaim_data(
                            value_qtaim, atom_feats, bond_feats, atom_keys, bond_keys
                        )
                    )

                    atom_keys_complete = atom_keys + atom_keys_grapher_no_qtaim
                except: 
                    fail_log_dict["qtaim"].append(key.decode("ascii"))
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
                    if not grapher_qtaim:
                        grapher_qtaim = get_grapher(
                            element_set=element_set,
                            atom_keys=atom_keys_complete,  # todo
                            bond_keys=bond_keys,  # todo
                            global_keys=list(global_feats.keys()),
                            allowed_ring_size=config_dict["allowed_ring_size"],
                            allowed_charges=config_dict["allowed_charges"],
                            allowed_spins=config_dict["allowed_spins"],
                            self_loop=True,
                            atom_featurizer_tf=True,
                            bond_featurizer_tf=True,
                            global_featurizer_tf=True,
                        )

                    if not grapher_no_qtaim:
                        grapher_no_qtaim = get_grapher(
                            element_set=element_set,
                            atom_keys=atom_keys_grapher_no_qtaim,
                            bond_keys=[],
                            global_keys=list(global_feats.keys()),
                            allowed_ring_size=config_dict["allowed_ring_size"],
                            allowed_charges=config_dict["allowed_charges"],
                            allowed_spins=config_dict["allowed_spins"],
                            self_loop=True,
                            atom_featurizer_tf=True,
                            bond_featurizer_tf=True,
                            global_featurizer_tf=True,
                        )

                    # graph generation
                    graph_no_qtaim = build_and_featurize_graph(
                        grapher_no_qtaim, mol_wrapper
                    )
                    graph = build_and_featurize_graph(
                        grapher_qtaim, mol_wrapper_qtaim_bonds
                    )

                    if index_dict == {}:
                        key_target = {
                            "atom": atom_keys_grapher_no_qtaim,
                            "bond": [],
                            "global": list(global_dipole_feats.keys()),
                        }
                        # print("target keys no qtaim: ", key_target)
                        index_dict = get_include_exclude_indices(
                            feat_names=grapher_no_qtaim.feat_names, target_dict=key_target
                        )

                    if index_dict_qtaim == {}:
                        key_target_qtaim = {
                            "atom": atom_keys,
                            "bond": bond_keys,
                            "global": list(global_dipole_feats.keys()),
                        }
                        # print("target keys qtaim: ", key_target_qtaim)
                        index_dict_qtaim = get_include_exclude_indices(
                            feat_names=grapher_qtaim.feat_names,
                            target_dict=key_target_qtaim,
                        )

                    # split graph into features and labels
                    split_graph_labels(
                        graph_no_qtaim,
                        include_names=index_dict["include_names"],
                        include_locs=index_dict["include_locs"],
                        exclude_locs=index_dict["exclude_locs"],
                    )

                    split_graph_labels(
                        graph,
                        include_names=index_dict_qtaim["include_names"],
                        include_locs=index_dict_qtaim["include_locs"],
                        exclude_locs=index_dict_qtaim["exclude_locs"],
                    )
                except:
                    fail_log_dict["graph"].append(key.decode("ascii"))
                    continue
                try: 
                    feature_scaler_iterative_qtaim.update([graph])
                    label_scaler_iterative_qtaim.update([graph])

                    feature_scaler_iterative.update([graph_no_qtaim])
                    label_scaler_iterative.update([graph_no_qtaim])

                    txn = db_qtaim.begin(write=True)
                    txn.put(
                        f"{key}".encode("ascii"),
                        pickle.dumps(serialize_dgl_graph(graph, ret=True), protocol=-1),
                    )
                    txn.commit()

                    txn = db_non_qtaim.begin(write=True)
                    txn.put(
                        f"{key}".encode("ascii"),
                        pickle.dumps(serialize_dgl_graph(graph_no_qtaim, ret=True), protocol=-1),
                    )
                    txn.commit()         
                except:
                    fail_log_dict["scaler"].append(key.decode("ascii"))
                    continue
        
        
        feature_scaler_iterative.finalize()
        label_scaler_iterative.finalize()

        feature_scaler_iterative_qtaim.finalize()
        label_scaler_iterative_qtaim.finalize()
    
    txn = db_qtaim.begin(write=True)
    txn.put("scaled".encode("ascii"), pickle.dumps(False, protocol=-1))
    txn.commit()

    txn = db_non_qtaim.begin(write=True)
    txn.put("scaled".encode("ascii"), pickle.dumps(False, protocol=-1))
    txn.commit()

    db_qtaim.close()
    db_non_qtaim.close()

    # save scalers
    scalers = {
        "feature_no_qtaim": feature_scaler_iterative,
        "label_no_qtaim": label_scaler_iterative,
        "feature_qtaim": feature_scaler_iterative_qtaim,
        "label_qtaim": label_scaler_iterative_qtaim,
    }

    return scalers


def scale_graphs(
    lmdb_file: lmdb,
    label_scaler: Union[HeteroGraphStandardScalerIterative, HeteroGraphLogMagnitudeScaler],
    feature_scaler: Union[HeteroGraphStandardScalerIterative, HeteroGraphLogMagnitudeScaler],
    filter_list: List[str] = ["scaled", "length"],
) -> None:
    """
    Scale the graphs using the provided scaler. Read from LMDB file and apply the scaler to each graph.
    """
    with lmdb_file.begin(write=False) as txn_in:    
        cursor = txn_in.cursor()
        for key, value in cursor: 
                
            if key.decode("ascii") not in filter_list:
                #print(key.decode("ascii"))
                graph = load_dgl_graph_from_serialized(pickle.loads(value))
                #print(graph)
                graph = feature_scaler([graph])
                graph = label_scaler(graph)
                print(graph[0].ndata["feat"]['global'])
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





    with lmdb_file.begin(write=False) as txn_in:    
        cursor = txn_in.cursor()
        for key, value in cursor: 
            if key.decode("ascii") == "length":
                graph = load_dgl_graph_from_serialized(pickle.loads(value))
                #print(graph)
                print(graph.ndata["feat"]['global'])

    lmdb_file.close()
    


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
    lmdb_name = str(args.lmdb_name)
    restart = bool(args.restart)

    lmdb_dict, config_dict = parse_config_gen_to_embed(args.config_path)

    if "element_set" in config_dict.keys():
        element_set = config_dict["element_set"]
    else:
        element_set = get_elements_from_structure_lmdb(lmdb_dict["structure_lmdb"])

    element_set = sorted(list(element_set))

    if "allowed_ring_size" not in config_dict.keys():
        config_dict["allowed_ring_size"] = [4, 5, 6, 8]

    if "allowed_charges" not in config_dict.keys():
        config_dict["allowed_charges"] = None

    if "allowed_spins" not in config_dict.keys():
        config_dict["allowed_spins"] = None


    file_qtaim = os.path.join(
        config_dict["lmdb_path_qtaim"], config_dict["lmdb_qtaim_name"]
    )
    
    file_non_qtaim = os.path.join(
        config_dict["lmdb_path_non_qtaim"], config_dict["lmdb_non_qtaim_name"]
    )    
    
    if os.path.exists(file_qtaim) and restart:
        os.remove(file_qtaim)
    if os.path.exists(file_non_qtaim) and restart:
        os.remove(file_non_qtaim)
        
    scalers = main_loop(
        lmdb_dict=lmdb_dict, config_dict=config_dict
    )

    # clean
    db_qtaim = lmdb.open(
        file_qtaim,
        map_size=int(1099511627776 * 2),
        subdir=False,
        meminit=False,
        map_async=True,
    )

    db_non_qtaim = lmdb.open(
        file_non_qtaim,
        map_size=int(1099511627776 * 2),
        subdir=False,
        meminit=False,
        map_async=True,
    )

    
    scale_graphs(
        lmdb_file=db_qtaim,
        label_scaler=scalers["label_qtaim"],
        feature_scaler=scalers["feature_qtaim"],
    )
    scale_graphs(
        lmdb_file=db_non_qtaim,
        label_scaler=scalers["label_no_qtaim"],
        feature_scaler=scalers["feature_no_qtaim"],
    )
