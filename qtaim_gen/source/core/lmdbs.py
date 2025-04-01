import os
import pickle as pkl
import lmdb
import json
import pickle
from typing import Dict, Optional
from glob import glob

from pymatgen.core import Molecule
from pymatgen.analysis.graphs import MoleculeGraph

from qtaim_gen.source.core.io import convert_inp_to_xyz
from qtaim_gen.source.core.parse_qtaim import (
    get_spin_charge_from_orca_inp,
    orca_inp_to_dict,
)
from qtaim_gen.source.core.bonds import get_bonds_from_rdkit


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
        #print("chunk size: {}".format(len(chunk)))
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
            directory=out_dir, pattern="{}_*.lmdb-lock".format(data_type), dry_run=not clean
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

    for chunk in split_list(files, chunk_size):
        chunk_ind = 1
        data_dict = {}

        for file in chunk:

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
            directory=out_dir, pattern="{}_*.lmdb-lock".format("geom"), dry_run=not clean
        )
