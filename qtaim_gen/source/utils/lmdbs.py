import os
import pickle as pkl
import lmdb
import json
import pickle
from typing import Dict, List, Tuple, Any, Optional
from glob import glob

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

                if key.decode("ascii") != "length":
                    # print(key.decode("ascii"))
                    try:
                        # int(key.decode("ascii"))
                        txn_out.put(
                            f"{key}".encode("ascii"),
                            value,
                        )
                        idx += 1
                        # print(idx)
                    # write properties
                    except ValueError:
                        txn_out.put(key, value)
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
):
    """Converts folders of output json files to lmdb files.
    Args:
        root_dir (str): Root directory containing the json files.
        out_dir (str): Output directory for the lmdb files.
        data_type (str): Data type to convert. Options are "charge", "bond", "other", "qtaim".
        out_lmdb (str): Output lmdb file.
        chunk_size (int): Size of the chunks to split the data into.
        clean (Optional[bool], optional): If True, delete the json files. Defaults to False.
    """
    chunk_ind = 1
    files_target = glob(root_dir + "*/{}.json".format(data_type))

    for chunk in split_list(files_target, chunk_size):
        data_dict = {}
        for file in chunk:
            with open(file, "r") as f:
                data = json.load(f)
                name = file.split("/")[-2]
                data_dict[name] = data

        write_lmdb(data_dict, out_dir, f"{data_type}_{chunk_ind}.lmdb")
        chunk_ind += 1

    files_out = glob("{}/{}_*.lmdb".format(root_dir, data_type))

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
):
    """
    Converts orca inp files into lmdbs
    Args:
        root_dir (str): Root directory containing the input files.
        out_dir (str): Output directory for the lmdb files.
        out_lmdb (str): Output lmdb file.
        chunk_size (int): Size of the chunks to split the data into.
        clean (Optional[bool], optional): If True, delete the input files. Defaults to False.
    """

    files = glob(root_dir + "*/*.inp")
    chunk_ind = 1

    for chunk in split_list(files, chunk_size):

        data_dict = {}

        for file in chunk:
            print("Processing file: ", file, " chunk: ", chunk_ind)
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

    files_out = glob("{}/geom_*.lmdb".format(root_dir))

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


def get_elements_from_structure_lmdb(structure_lmdb):
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


def get_elements_from_structure_lmdb_folder_list(list_lmdb):
    """
    Get the elements from the structure lmdb
    Takes:
        list_lmdb: list of lmdb files containing the structure data
    Returns:
        element_set: set of elements in the structure lmdb
    """
    element_set = set()
    for lmdb_path in list_lmdb:
        with lmdb.open(lmdb_path, readonly=True, lock=False) as structure_lmdb:
            element_set.update(get_elements_from_structure_lmdb(structure_lmdb))

    return element_set


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

    return config_dict


def parse_charge_data(value_charge, n_atoms):
    """
    Parse charge-related data and update atom features.
    """
    atom_feats_charge = {i: {} for i in range(n_atoms)}
    global_dipole_feats = {}
    charge_types = list(value_charge.keys())
    for charge_type in charge_types:

        # parse out into atom_feats without for loop
        {
            atom_feats_charge[int(k.split("_")[0]) - 1].update(
                {"charge_" + charge_type: v}
            )
            for k, v in value_charge[charge_type]["charge"].items()
        }

        if "dipole" in value_charge[charge_type].keys():
            global_dipole_feats.update(
                {
                    charge_type
                    + "_dipole_mag": value_charge[charge_type]["dipole"]["mag"]
                }
            )

        if "spin" in value_charge[charge_type].keys():
            {
                atom_feats_charge[int(k.split("_")[0]) - 1].update(
                    {"spin_" + charge_type: v}
                )
                for k, v in value_charge[charge_type]["spin"].items()
            }

    return atom_feats_charge, global_dipole_feats


def parse_qtaim_data(value_qtaim, atom_feats, bond_feats, atom_keys, bond_keys):
    """
    Parse QTAIM-related data and update atom and bond features.
    """

    if atom_keys is None:
        qtaim_atoms = {k: v for k, v in value_qtaim.items() if "_" not in k}
        atom_keys = list(qtaim_atoms[list(qtaim_atoms.keys())[0]].keys())
        # remove "cp_num" from atom_keys
        [atom_keys.remove(i) for i in ["cp_num", "element", "number", "pos_ang"]]

    if bond_keys is None:
        qtaim_bonds = {k: v for k, v in value_qtaim.items() if "_" in k}
        bond_keys = list(qtaim_bonds[list(qtaim_bonds.keys())[0]].keys())
        # get first k, v in qtaim_bonds
        [bond_keys.remove(i) for i in ["cp_num", "connected_bond_paths", "pos_ang"]]

    # print("*******atom keys*******: ", atom_keys)

    # only get the keys that are in the qtaim_bonds and qtaim_atoms from each dictionary in value_qtaim
    qtaim_atoms = {
        k: get_several_keys(v, atom_keys)
        for k, v in value_qtaim.items()
        if "_" not in k
    }
    qtaim_bonds = {
        k: get_several_keys(v, bond_keys) for k, v in value_qtaim.items() if "_" in k
    }

    for key, value in atom_feats.items():  # update atom_feats with qtaim_atoms
        atom_feats[key].update(qtaim_atoms[str(key)])

    for key, value in qtaim_bonds.items():  # update bond_feats with qtaim_bonds
        a, b = key.split("_")
        key_conv = tuple(sorted([int(a), int(b)]))
        bond_feats[key_conv] = qtaim_bonds[str(key)]

    bond_feats = {k: v for k, v in bond_feats.items() if k[0] != k[1]}
    connected_bond_paths = list(bond_feats.keys())

    return atom_keys, bond_keys, atom_feats, bond_feats, connected_bond_paths


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
