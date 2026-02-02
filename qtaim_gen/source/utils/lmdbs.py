import os
import lmdb
import json
import pickle
from typing import Dict, List, Tuple, Any, Optional, Union
from glob import glob
from dataclasses import dataclass
import numpy as np 
from pymatgen.core import Molecule
from pymatgen.analysis.graphs import MoleculeGraph

from qtaim_gen.source.utils.io import convert_inp_to_xyz
from qtaim_gen.source.core.parse_qtaim import (
    get_spin_charge_from_orca_inp,
    orca_inp_to_dict,
)
from qtaim_gen.source.utils.bonds import get_bonds_from_rdkit


def convert_inp_to_xyz(orca_path, output_path):
    """
    Convert an ORCA input file to an XYZ file.
    Takes:
        orca_path: path to ORCA input file
        output_path: path to write XYZ file to
    Returns:
        None
    """
    charge, spin = get_spin_charge_from_orca_inp(orca_path)
    pos = orca_inp_to_dict(orca_path)

    mol_dict = {"mol": pos, "charge": charge, "spin": spin}

    n_atoms = len(mol_dict["mol"])

    xyz_str = "{}\n".format(n_atoms)
    spin_charge_line = "{} {}\n".format(mol_dict["charge"], mol_dict["spin"])
    xyz_str += spin_charge_line
    # write the atom positions
    for ind, atom in mol_dict["mol"].items():

        atom_line = "{} {} {} {}\n".format(
            atom["element"], atom["pos"][0], atom["pos"][1], atom["pos"][2]
        )
        xyz_str += atom_line

    with open(output_path, "w") as f:
        f.write(xyz_str)


def running_average(
    old_avg: float, new_value: float, n: int, n_new: Optional[int] = 1
) -> float:
    """simple running average
    Args:
        old_avg (float): old average
        new_value (float): new value
        n (int): number of samples
        n_new (Optional[int]): number of new samples
    """
    return old_avg + (new_value - old_avg) * n_new / n


def write_lmdb(
    data: dict[dict],
    lmdb_dir: str,
    lmdb_name: str,
    global_values: Optional[Dict[str, float]] = {},
):
    """General method to write data to an LMDB file.
    Args:
        data (dict[dict]): Data to write to the LMDB file.
        lmdb_dir (str): Directory to write the LMDB file.
        lmdb_name (str): Name of the LMDB file.
        global_values (Optional[Dict[str, float]], optional): Global values to write to the LMDB file. Defaults to {}.
    """

    if not os.path.exists(lmdb_dir):
        os.makedirs(lmdb_dir, exist_ok=True)

    db = lmdb.open(
        lmdb_dir + lmdb_name,
        map_size=int(1099511627776 * 2),
        subdir=False,
        meminit=False,
        map_async=True,
    )

    # write samples
    for ind, sample in data.items():
        # print(ind)
        sample_index = ind
        txn = db.begin(write=True)
        txn.put(
            f"{sample_index}".encode("ascii"),
            pickle.dumps(sample, protocol=-1),
        )

        txn.commit()

    txn = db.begin(write=True)
    txn.put("length".encode("ascii"), pickle.dumps(len(data), protocol=-1))
    txn.commit()

    if global_values != {}:
        for key, value in global_values.items():
            txn = db.begin(write=True)
            txn.put(key.encode("ascii"), pickle.dumps(value, protocol=-1))
            txn.commit()

    db.sync()
    db.close()


def merge_lmdbs(db_paths: str, out_path: str, output_file: str):
    env_out = lmdb.open(
        os.path.join(out_path, output_file),
        map_size=1099511627776 * 2,
        subdir=False,
        meminit=False,
        map_async=True,
    )

    idx = 0

    for db_path in db_paths:
        # print("merge in {}".format(db_path))
        env_in = lmdb.open(
            str(db_path),
            subdir=False,
            readonly=True,
            lock=False,
            readahead=True,
            meminit=False,
        )

        # should set indexes so that properties do not writtent down as well.
        with env_out.begin(write=True) as txn_out, env_in.begin(write=False) as txn_in:
            cursor = txn_in.cursor()
            for key, value in cursor:
                key_str = key.decode("ascii")
                if key_str != "length":
                    # key is already bytes, use it directly
                    txn_out.put(key, value)
                    idx += 1
        env_in.close()

    # update length
    txn_out = env_out.begin(write=True)
    # print("length: {}".format(idx))
    txn_out.put("length".encode("ascii"), pickle.dumps(idx, protocol=-1))
    txn_out.commit()

    env_out.sync()
    env_out.close()


def cleanup_lmdb_files(directory: str, pattern: str, dry_run: Optional[bool] = False):
    """
    Clean LMDB files in a directory that match a pattern.
    Args:
        directory (str): Directory to search for files.
        pattern (str): Pattern to match files.
        dry_run (Optional[bool], optional): If True, do not delete files. Defaults to False.
    """
    file_list = glob(os.path.join(directory, pattern))

    for file_path in file_list:
        try:
            if not dry_run:
                os.remove(file_path)
                # print(f"Deleted file: {file_path}")
            else:
                print(f"Dry run, would delete file: {file_path}")
        except OSError as e:
            print(f"Error deleting file: {file_path}. {str(e)}")


def split_list(lst: list, chunk_size: int):
    """Split without overlap"""
    # print(lst)
    if chunk_size == -1:
        yield lst

    elif chunk_size < len(lst):
        for i in range(0, len(lst), chunk_size):
            # print(lst[i:i + chunk_size])
            yield lst[i : i + chunk_size]
        # for i in range(0, len(lst), chunk_size):
        #    yield lst[i:i + chunk_size]
    else:
        yield lst


def json_2_lmdbs(
    root_dir: str,
    out_dir: str,
    data_type: str,
    out_lmdb: str,
    chunk_size: int,
    clean: Optional[bool] = True,
    merge: Optional[bool] = True,
    move_files: Optional[bool] = False,
    limit: Optional[int] = None,
):
    """Converts folders of output json files to lmdb files.
    Args:
        root_dir (str): Root directory containing the json files.
        out_dir (str): Output directory for the lmdb files.
        data_type (str): Data type to convert. Options are "charge", "bond", "other", "qtaim".
        out_lmdb (str): Output lmdb file.
        chunk_size (int): Size of the chunks to split the data into.
        clean (Optional[bool], optional): If True, delete the json files. Defaults to False.
        move_files (Optional[bool], optional): If files were moved into separate ./generator/ folders in each job
        limit (Optional[int], optional): Limit number of files to process (for debugging).
    """
    chunk_ind = 1
    if move_files:
        files_target = glob(
            root_dir + "*/generator/{}.json".format(data_type)
        )
    else:
        files_target = glob(root_dir + "*/{}.json".format(data_type))

    # Apply limit for debug mode
    if limit is not None:
        files_target = files_target[:min(limit, len(files_target))]

    for chunk in split_list(files_target, chunk_size):
        data_dict = {}
        for file in chunk:
            with open(file, "r") as f:
                data = json.load(f)
                # When move_files=True, path is root/job/generator/file.json -> use [-3]
                # When move_files=False, path is root/job/file.json -> use [-2]
                name = file.split("/")[-3] if move_files else file.split("/")[-2]
                data_dict[name] = data

        write_lmdb(data_dict, out_dir, f"{data_type}_{chunk_ind}.lmdb")
        chunk_ind += 1

    files_out = glob("{}/{}_*.lmdb".format(out_dir, data_type))
    #print("files_out: ", files_out)

    if merge:
        merge_lmdbs(files_out, out_dir, out_lmdb)

        cleanup_lmdb_files(
            directory=out_dir, pattern="{}_*.lmdb".format(data_type), dry_run=not clean
        )
        cleanup_lmdb_files(
            directory=out_dir,
            pattern="{}_*.lmdb-lock".format(data_type),
            dry_run=not clean,
        )


def inp_files_2_lmdbs(
    root_dir: str,
    out_dir: str,
    out_lmdb: str,
    chunk_size: int,
    clean: Optional[bool] = True,
    merge: Optional[bool] = True,
    limit: Optional[int] = None,
):
    """
    Converts orca inp files into lmdbs at scale.
    Args:
        root_dir (str): Root directory containing the input files.
        out_dir (str): Output directory for the lmdb files.
        out_lmdb (str): Output lmdb file.
        chunk_size (int): Size of the chunks to split the data into.
        clean (Optional[bool], optional): If True, delete the input files. Defaults to False.
        limit (Optional[int], optional): Limit number of files to process (for debugging).
    """
    files = glob(root_dir + "*/*.inp")

    # Apply limit for debug mode
    if limit is not None:
        files = files[:min(limit, len(files))]

    chunk_ind = 1

    for chunk in split_list(files, chunk_size):

        data_dict = {}

        for file in chunk:
            #print("Processing file: ", file, " chunk: ", chunk_ind)
            charge, spin = get_spin_charge_from_orca_inp(file)
            xyz_file = file.replace(".inp", ".xyz")
            convert_inp_to_xyz(file, xyz_file)
            molecule = Molecule.from_file(xyz_file)

            molecule.set_charge_and_spin(
                charge=int(charge), spin_multiplicity=int(spin)
            )
            molecule_graph = MoleculeGraph.with_empty_graph(molecule)

            identifier = file.split("/")[-2]

            try:
                bonds_rdkit = get_bonds_from_rdkit(xyz_file)
            except:
                bonds_rdkit = []
                [molecule_graph.add_edge(bond[0], bond[1]) for bond in bonds_rdkit]

            data_dict[identifier] = {
                "molecule": molecule,
                "molecule_graph": molecule_graph,
                "ids": identifier,
                "bonds": bonds_rdkit,
                "spin": int(spin),
                "charge": int(charge),
            }

        write_lmdb(data_dict, out_dir, f"geom_{chunk_ind}.lmdb")
        chunk_ind += 1

    files_out = glob("{}/geom_*.lmdb".format(out_dir))

    if merge:
        merge_lmdbs(files_out, out_dir, out_lmdb)

        cleanup_lmdb_files(
            directory=out_dir, pattern="{}_*.lmdb".format("geom"), dry_run=not clean
        )
        cleanup_lmdb_files(
            directory=out_dir,
            pattern="{}_*.lmdb-lock".format("geom"),
            dry_run=not clean,
        )


def get_elements_from_structure_lmdb(structure_lmdb: lmdb.Environment) -> set:
    """
    Get the elements from the structure lmdb
    Takes:
        structure_lmdb: lmdb file containing the structure data
    Returns:
        element_set: set of elements in the structure lmdb
    """
    element_set = set()
    with structure_lmdb.begin(write=False) as txn_in:
        cursor = txn_in.cursor()
        for key, value in cursor:  # first loop for gathering statistics and averages
            if key.decode("ascii") != "length":
                value_structure = pickle.loads(value)

                element_list = [
                    str(site.species.elements).split(" ")[-1].split("]")[0]
                    for site in value_structure["molecule"]
                ]
                element_set.update(element_list)
    return element_set


def get_elements_from_structure_lmdb_folder_list(list_lmdb: List[str]):
    """
    Get the elements from the structure lmdb
    Takes:
        list_lmdb: list of lmdb files containing the structure data
    Returns:
        element_set: set of elements in the structure lmdb
    """
    element_set = set()
    for lmdb_path in list_lmdb:
        with lmdb.open(
            lmdb_path, 
            readonly=True, 
            lock=False, 
            subdir=False
        ) as structure_lmdb:
            element_set.update(get_elements_from_structure_lmdb(structure_lmdb))

    return element_set


@dataclass
class config_converter:
    lmdb_locations: Dict[str, str]
    lmdb_paths: Dict[str, str]
    lmdb_names: Dict[str, str]
    allowed_ring_size: List[int]
    allowed_charges: List[int]
    allowed_spins: List[int]
    filter_list: List[str]
    graphs_types: List[str] = None
    element_set: List[str] = None
    chunk: int = None
    restart: bool = False
    save_scaler: bool = False


def parse_config_gen_to_embed(
    config_path: str,
    restart: Optional[bool] = False,
) -> Tuple[Dict[str, lmdb.Environment], Dict[str, Any]]:
    """
    Parse the config file for generating qtaim_embed data.

    Args:
        config_path (str): Path to the config file in JSON format.

    Returns:
        Dict[str, Any]]:
            - A dictionary containing the configuration parameters.
    """
    lmdb_dict = {}
    with open(config_path, "r") as f:
        config_dict = json.load(f)
    config_dict["restart"] = restart

    if "allowed_ring_size" not in config_dict.keys():
        config_dict["allowed_ring_size"] = [4, 5, 6, 8]

    if "allowed_charges" not in config_dict.keys():
        config_dict["allowed_charges"] = None

    if "allowed_spins" not in config_dict.keys():
        config_dict["allowed_spins"] = None

    # create config

    return config_dict


def parse_charge_data( 
    dict_charge:dict, 
    n_atoms:int,
    charge_filter: Optional[List[str]] = None
) -> Tuple[Dict[int, Dict[str, Any]], Dict[str, float]]:
    """
    Parse charge-related data and update atom features.

    Takes:
        dict_charge (dict): Dictionary containing charge-related data.
        n_atoms (int): Number of atoms in the structure.
    Returns:
        atom_feats_charge (Dict[int, Dict[str, Any]]): Dictionary containing atom features related
        to charge and spin information.
        global_dipole_feats (Dict[str, float]): Dictionary containing global dipole features.
    """
    atom_feats_charge = {i: {} for i in range(n_atoms)}
    global_dipole_feats: Dict[str, float] = {}

    for charge_type, data in dict_charge.items():
        if charge_filter is not None and charge_type not in charge_filter:
            continue

        # charges
        charge_dict = data.get("charge", {})
        for k, v in charge_dict.items():
            try:
                idx = int(k.split("_")[0]) - 1
            except Exception:
                continue
            if 0 <= idx < n_atoms:
                atom_feats_charge[idx]["charge_" + charge_type] = float(v)

        # dipole (global)
        dip = data.get("dipole")
        if isinstance(dip, dict) and "mag" in dip:
            global_dipole_feats[charge_type + "_dipole_mag"] = float(dip["mag"])

        # spin
        spin_dict = data.get("spin", {})
        for k, v in spin_dict.items():
            try:
                idx = int(k.split("_")[0]) - 1
            except Exception:
                continue
            if 0 <= idx < n_atoms:
                atom_feats_charge[idx]["spin_" + charge_type] = float(v)

    return atom_feats_charge, global_dipole_feats


def parse_fuzzy_data(dict_fuzzy: dict, n_atoms: int, fuzzy_filter: Optional[List[str]] = None) -> Tuple[Dict[int, Dict[str, Any]], Dict[str, float]]:
    """
    Parse fuzzy-related data and update atom features.
    Takes:
        dict_fuzzy: dict, dictionary containing fuzzy data
        n_atoms: int, number of atoms in the molecule
    Returns:
        atom_feats_fuzzy: dict, dictionary containing atom features related to fuzzy data
        global_fuzzy_feats: dict, dictionary containing global features related to fuzzy data
    """
    atom_feats_fuzzy = {i: {} for i in range(n_atoms)}
    global_fuzzy_feats: Dict[str, float] = {}

    for fuzzy_type, payload in dict_fuzzy.items():
        if fuzzy_filter is not None and fuzzy_type not in fuzzy_filter:
            continue

        # extract and remove global keys 'sum' and 'abs_sum' if present
        for gk in ("sum", "abs_sum"):
            if gk in payload:
                global_fuzzy_feats[f"fuzzy_{fuzzy_type}_{gk}"] = float(payload.pop(gk))

        # atom-level entries: keys like '1_xxx', '2_yyy' -> parse once and assign
        for k, v in payload.items():
            try:
                idx = int(k.split("_")[0]) - 1
            except Exception:
                continue
            if 0 <= idx < n_atoms:
                atom_feats_fuzzy[idx][f"fuzzy_{fuzzy_type}"] = float(v)

    return atom_feats_fuzzy, global_fuzzy_feats


def parse_qtaim_data(
        dict_qtaim: dict, 
        atom_feats: Dict[int, Dict[str, Any]],
        bond_feats: Dict[Tuple[int, int], Dict[str, Any]],
        atom_keys: Optional[List[str]] = None, 
        bond_keys: Optional[List[str]] = None
    ) -> Tuple[List[str], List[str], Dict[int, Dict[str, Any]], Dict[Tuple[int, int], Dict[str, Any]], List[Tuple[int, int]]]:
    """
    Parse QTAIM-related data and update atom and bond features from incoming dictionaries. If no keys are provided, 
    all keys will be extracted from the input dictionaries. We also return the list of keys used for atom and bond 
    features, as well as the list of connected bond paths.

    Takes:
        dict_qtaim (dict): Dictionary containing QTAIM-related data.
        atom_feats (Dict[int, Dict[str, Any]]): Dictionary containing atom features to be updated with QTAIM data.
        bond_feats (Dict[Tuple[int, int], Dict[str, Any]]): Dictionary containing bond features to be updated with QTAIM data.
        atom_keys (Optional[List[str]]): List of keys to extract for atom features. If None, all keys will be extracted. Defaults to None.
        bond_keys (Optional[List[str]]): List of keys to extract for bond features. If None, all keys will be extracted. Defaults to None.  

    Returns:
        atom_keys (List[str]): List of keys used for atom features.
        bond_keys (List[str]): List of keys used for bond features.
        atom_feats (Dict[int, Dict[str, Any]]): Updated dictionary containing atom features with
        QTAIM data.
        bond_feats (Dict[Tuple[int, int], Dict[str, Any]]): Updated dictionary containing bond features with QTAIM data.
        connected_bond_paths (List[Tuple[int, int]]): List of tuples representing connected bond paths.
    """

    # determine atom_keys and bond_keys robustly (avoid errors on empty inputs)
    if atom_keys is None:
        atom_keys = []
        for k, v in dict_qtaim.items():
            if "_" not in k:
                atom_keys = list(v.keys())
                break
        for rem in ("cp_num", "element", "number", "pos_ang"):
            if rem in atom_keys:
                atom_keys.remove(rem)

    if bond_keys is None:
        bond_keys = []
        for k, v in dict_qtaim.items():
            if "_" in k:
                bond_keys = list(v.keys())
                break
        for rem in ("cp_num", "connected_bond_paths", "pos_ang"):
            if rem in bond_keys:
                bond_keys.remove(rem)

    # Build integer-keyed maps to avoid repeated str/int conversions while updating
    qtaim_atoms_int: Dict[int, Dict[str, Any]] = {}
    qtaim_bonds_conv: Dict[Tuple[int, int], Dict[str, Any]] = {}

    for k, v in dict_qtaim.items():
        if "_" not in k:
            try:
                ik = int(k)
            except Exception:
                # leave non-integer atom keys out
                continue
            qtaim_atoms_int[ik] = get_several_keys(v, atom_keys)
        else:
            try:
                a, b = k.split("_")
                key_conv = tuple(sorted([int(a), int(b)]))
            except Exception:
                continue
            qtaim_bonds_conv[key_conv] = get_several_keys(v, bond_keys)

    # Update atom_feats using integer keys directly (faster than repeated str conversions)
    for key in list(atom_feats.keys()):
        vals = qtaim_atoms_int.get(key)
        if vals:
            atom_feats[key].update(vals)

    # Update bond_feats with converted tuple keys
    for key_conv, vals in qtaim_bonds_conv.items():
        bond_feats[key_conv] = vals

    # filter out degenerate self-bonds and collect connected bond paths
    bond_feats = {k: v for k, v in bond_feats.items() if k[0] != k[1]}
    connected_bond_paths = list(bond_feats.keys())

    return (
        atom_keys, 
        bond_keys, 
        atom_feats, 
        bond_feats, 
        connected_bond_paths
    )


def parse_other_data(dict_other: dict, other_filter: Optional[List[str]] = None, clean=True) -> Dict[str, Any]:
    """
    Parse other-related data and update atom features.
    Takes:
        dict_other: dict, dictionary containing other data
        other_filter: list of str, list of keys to filter in the other data
        clean: bool, whether to clean nan None, inf values from the data. Set them to 0 
    """
    global_other_feats = {}
    for k, v in dict_other.items():
        if other_filter is not None and k not in other_filter:
            continue
        if clean:
            if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
                v = 0.0
        global_other_feats["other_" + k] = float(v)
    return global_other_feats


def get_several_keys(di: Dict[str, Any], keys: List[str]) -> Dict[str, Any]:
    """
    Get several keys from a dictionary.

    Args:
        di (Dict[str, Any]): The input dictionary.
        keys (List[str]): List of keys to extract from the dictionary.

    Returns:
        Dict[str, Any]: A dictionary containing only the specified keys.
    """
    return {k: di.get(k, None) for k in keys}


def gather_structure_info(value_structure: dict) -> Tuple[MoleculeGraph, Dict[str, float]]:
    """
    Gather structure information from the input dictionary.
    Args:
        value_structure (dict): Dictionary containing the structure information.
    Returns:
        mol_graph (MoleculeGraph): The molecule graph extracted from the input dictionary.
        global_feats (Dict[str, float]): A dictionary containing global features such as charge, spin, number of atoms, and number of bonds.
    """
    mol_graph = value_structure["molecule_graph"]
    global_feats = {
        "charge": value_structure["charge"],
        "spin": value_structure["spin"],
        "n_atoms": len(mol_graph),
        "n_bonds": len(value_structure["bonds"]),
    }
    # initialize atom_feats with dictionary whos keys are atom inds and empty dict as values
    
    return mol_graph, global_feats


def parse_bond_data(
        dict_bond: dict, 
        bond_filter: Optional[List[str]] = None, 
        cutoff: Optional[float] = None, 
        bond_list_definition="fuzzy", 
        bond_feats=None, 
        clean=True, 
        as_lists=True, 
    ) -> Dict[str, Any]:
    """
    Parse bond-related data and update bond features.
    struct of inputs is {'ibsi_bond': {'1_O_to_2_C': 1.28196, '1_O_to_3_C': 0.05902, ...}, "fuzzy_bond": {'1_O_to_2_C': 0.5, '1_O_to_3_C': 0.1, ...}, ...}
    structure of bond_feats is {(0, 1): {'ibsi_bond': 1.28196, 'fuzzy_bond': 0.5}, (0, 2): {'ibsi_bond': 0.05902, 'fuzzy_bond': 0.1}, ...}
    Takes:
        dict_other: dict, dictionary containing other data
        other_filter: list of str, list of keys to filter in the other data
        clean: bool, whether to clean nan None, inf values from the data. Set them to 0 
    """
    
    # assert that bond_list_definition is in dict_bond keys
    if bond_list_definition not in dict_bond.keys() and bond_list_definition+"_bond" not in dict_bond.keys():
        raise ValueError(f"Bond list definition {bond_list_definition} not found in dict_bond keys")
    # assert that bond_list_definition is also in bond_filter if bond_filter is not None
    if bond_filter is not None and (bond_list_definition not in bond_filter and bond_list_definition+"_bond" not in bond_filter):
        raise ValueError(f"Bond list definition {bond_list_definition} cannot be in bond_filter")
    
    def clean_string_for_bond_key(s: str, as_lists: bool = False):
        # parse '1_O_to_2_C' style keys robustly and cheaply
        try:
            left, right = s.split("_to_")
            a = int(left.split("_")[0]) - 1
            b = int(right.split("_")[0]) - 1
        except Exception:
            raise ValueError(f"Malformed bond key: {s}")
        if as_lists:
            return list(sorted([a, b]))
        return tuple(sorted([a, b]))

    if bond_feats is None:
        bond_feats = {}
    bond_list: List[Tuple[int, int]] = []

    # prepare allowed bond-list keys for fast membership checks
    bond_list_keys = {bond_list_definition, bond_list_definition + "_bond"}

    for k, v in dict_bond.items():
        # filter by bond_filter if provided (allow both 'fuzzy' and 'fuzzy_bond')
        if bond_filter is not None:
            if k in bond_filter:
                pass
            elif k.endswith("_bond") and k[:-5] in bond_filter:
                pass
            else:
                continue

        # if this entry lists bonds (e.g., 'fuzzy' or 'fuzzy_bond'), capture bond_list
        if k in bond_list_keys:
            # v is a dict with keys like '1_O_to_2_C'
            bond_list = [clean_string_for_bond_key(i, as_lists=as_lists) for i in v.keys()]

        # put the bond features in the bond_feats dict with keys as tuples of atom indices
        for bond_key, bond_value in v.items():
            try:
                bond_key_tuple = clean_string_for_bond_key(bond_key)
            except ValueError:
                # skip malformed bond keys
                continue

            if clean and isinstance(bond_value, float) and (np.isnan(bond_value) or np.isinf(bond_value)):
                bond_value = 0.0

            if bond_key_tuple not in bond_feats:
                bond_feats[bond_key_tuple] = {}

            # store under the original section key (e.g., 'fuzzy' or 'ibsi_bond')
            bond_feats[bond_key_tuple][k] = float(bond_value)

    # If a cutoff is provided, or a bond_filter is provided, do a second pass
    # to filter bond_feats and bond_list based on the feature used to define
    # the bond_list (e.g., 'fuzzy' if bond_list_definition == 'fuzzy').
    # - cutoff (float): keep bonds where feature >= cutoff
    # - bond_filter (list): if cutoff is None, fall back to presence/non-zero behavior
    if cutoff is not None or bond_filter is not None:
        # determine the actual key name used in the payload (could be 'fuzzy' or 'fuzzy_bond')
        candidate_keys = [bond_list_definition, bond_list_definition + "_bond"]
        filter_key = None
        for ck in candidate_keys:
            if ck in dict_bond:
                filter_key = ck
                break
        # if still not found, try to find something from bond_filter
        if filter_key is None and bond_filter is not None:
            for ck in candidate_keys:
                if ck in bond_filter:
                    filter_key = ck
                    break
        # final fallback: if bond_filter explicitly provided, use its first element
        if filter_key is None and bond_filter is not None and len(bond_filter) > 0:
            filter_key = bond_filter[0]

        allowed = set()
        for b in bond_list:
            normalized = tuple(sorted(b))
            feats = bond_feats.get(normalized)
            if feats is None:
                continue
            if filter_key is None or filter_key not in feats:
                # if no filter key available, skip this bond for safety
                continue

            val = feats[filter_key]
            # if cutoff is provided: require numeric comparison >= cutoff
            if cutoff is not None:
                try:
                    if isinstance(val, (int, float)) and not np.isnan(val) and not np.isinf(val):
                        if val >= cutoff:
                            allowed.add(normalized)
                except Exception:
                    # if comparison fails, skip the bond
                    continue
            else:
                # fallback presence/non-zero behavior when only bond_filter provided
                try:
                    if isinstance(val, (int, float)):
                        if not np.isclose(val, 0.0):
                            allowed.add(normalized)
                    elif val is not None:
                        allowed.add(normalized)
                except Exception:
                    allowed.add(normalized)

        # filter bond_feats to only include allowed bonds
        bond_feats = {k: v for k, v in bond_feats.items() if tuple(sorted(k)) in allowed}

        # update bond_list to reflect the filtering; preserve as_lists semantics
        if as_lists:
            bond_list = [list(b) for b in allowed]
        else:
            bond_list = [tuple(b) for b in allowed]
    

    return bond_feats, bond_list


def filter_bond_feats(
        bond_feats: Dict[Tuple[int, int], Dict[str, Any]], 
        bond_list: Union[List[Tuple[int, int]], Dict[Tuple[int, int], Any]]
    ) -> Dict[Tuple[int, int], Dict[str, Any]]:
    """
    Filter bond features to only include bonds that are in the provided bond_list.
    
    Args:
        bond_feats (Dict[Tuple[int, int], Dict[str, Any]]): Dictionary of bond features 
            where keys are tuples of atom indices (e.g., (0, 1)) and values are 
            dictionaries containing bond properties.
        bond_list (List[Tuple[int, int]] | Dict[Tuple[int, int], Any]): Either a list 
            of bond tuples or a dictionary with bond tuples as keys. Bonds should be 
            represented as tuples of two atom indices.
    
    Returns:
        Dict[Tuple[int, int], Dict[str, Any]]: Filtered bond_feats dictionary containing 
            only bonds that appear in bond_list.
    
    Example:
        >>> bond_feats = {(0, 1): {'ibsi': 1.2}, (0, 2): {'ibsi': 0.5}, (1, 2): {'ibsi': 0.3}}
        >>> bond_list = [(0, 1), (1, 2)]
        >>> filtered = filter_bond_feats(bond_feats, bond_list)
        >>> filtered
        {(0, 1): {'ibsi': 1.2}, (1, 2): {'ibsi': 0.3}}
    """
    # Convert bond_list to a set of tuples for efficient lookup
    # Handle both list and dict inputs
    if isinstance(bond_list, dict):
        bond_set = set(bond_list.keys())
    else:
        bond_set = set(bond_list)
    
    # Also normalize the bonds to handle both (a, b) and (b, a) representations
    # by ensuring they are sorted tuples
    normalized_bond_set = set()
    for bond in bond_set:
        if isinstance(bond, (list, tuple)) and len(bond) == 2:
            normalized_bond_set.add(tuple(sorted(bond)))
    
    # Filter bond_feats to only include bonds in the bond_list
    filtered_bond_feats = {}
    for bond_key, bond_value in bond_feats.items():
        # Normalize the bond key for comparison
        if isinstance(bond_key, (list, tuple)) and len(bond_key) == 2:
            normalized_key = tuple(sorted(bond_key))
            if normalized_key in normalized_bond_set:
                filtered_bond_feats[bond_key] = bond_value
    
    return filtered_bond_feats