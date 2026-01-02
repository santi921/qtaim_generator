import os 
from tqdm import tqdm
import numpy as np 
import json 
from ase.db import connect
from ase.io.orca import read_geom_orcainp


def get_atoms_from_orca_input(test_path):
    """
    Reads an ORCA input file and returns an ASE Atoms object with charge and spin set.
    Args:
        test_path (str): Path to the ORCA input file.
    Returns:
        atoms (ase.Atoms): ASE Atoms object with charge and spin information.
    """
    atoms = read_geom_orcainp(test_path)
    with open(test_path, "r") as f:
        lines = f.readlines()
    
    for line in lines:
        if line.strip().startswith("* xyz"):
            charge = line.strip().split()[2]
            spin = line.strip().split()[3]
            break    
        if line.strip().startswith("*xyz"):
            charge = line.strip().split()[1]
            spin = line.strip().split()[2]
            break

    atoms.charge = int(charge)
    atoms.spin = int(spin)
    return atoms


def gather_dicts_from_generator_folder(dir, move_dir=True):
    # if move_dir is True, append "/generator/" to dir
    if move_dir:
        dir = os.path.join(dir, "generator")
    
    gather_list = ["timings", "bond", "fuzzy_full", "qtaim", "other", "charge"]
    gathered_dict = {}
    
    for item in gather_list:
        # get jsons 
        json_path = os.path.join(dir, f"{item}.json")
        if os.path.exists(json_path):
            with open(json_path, "r") as f:
                item_dict = json.load(f)
            gathered_dict[item] = item_dict

    return gathered_dict



def construct_lmdb_database(structures, root_lmdb, strata=None, name_list=None, reserve_ids=True):
        """
        Create a LMDB database from a list of ASE Atoms objects.
        Takes:
            structures: list of ASE Atoms objects
            root_lmdb: path to the root directory where the LMDB database will be stored
            strata: list of strata values for each structure (optional)
        
        """
        composition_set = set()
        num_atoms = []
        if strata is None:
            strata_list = []
        else: 
            strata_list = strata        
        # check and create root_lmdb if it does not exist
        if not os.path.exists(root_lmdb):
            os.makedirs(root_lmdb)

        # figure out chunks later with lmdbs
        asedb_fn = f"{root_lmdb}/asedb.aselmdb"

        with connect(asedb_fn) as database:
            id_list = []
            if reserve_ids:
                print("Reserving IDs...")
                # status bar
                for ind, name in enumerate(tqdm(name_list)):
                    id = database.reserve(name=name)
                    if id is None:
                        id_list.append(None)
                    id_list.append(id)
            
            print("Writing structures to LMDB...")
            for ind, atoms in enumerate(tqdm(structures)):
                if name_list is not None:
                    name = name_list[ind]
                else:
                    name = None

                if reserve_ids:
                    database.write(atoms, data=atoms.info, name=name, id=id_list[ind])
                else:
                    database.write(atoms, data=atoms.info, name=name)
                #write(atoms, id=None, key_value_pairs={}, data={}, **kwargs)
                num_atoms.append(len(atoms))
                composition_set.add(atoms.get_chemical_formula())
                #print(atoms.get_chemical_formula())
                
                # DUMMY VALUE FOR STRATA
                if strata is None:
                    strata_list.append(len(atoms) % 2)
                else:
                    strata_list.append(strata)
        
            database.metadata["strata"] = strata_list
            database.metadata["num_atoms"] = num_atoms

        # other metadata, one for each structure? 
        np.savez(
            f"{root_lmdb}/metadata.npz",
            natoms=num_atoms,
            strata=strata,
            compositions=list(composition_set)
        )
        
        
def construct_lmdb_otf(root_dir, root_lmdb, strata=None):
    """
    Create a LMDB database from a list of ASE Atoms objects.
    Takes:
        root_dir: path to the root directory where the ORCA input files are stored
        root_lmdb: path to the root directory where the LMDB database will be stored
        strata: list of strata values for each structure (optional)
        name_list: list of names for each structure (optional)
    
    """
    composition_set = set()

    # figure out chunks later with lmdbs
    asedb_fn = f"{root_lmdb}/asedb.aselmdb"
    num_atoms = []
    if strata is None:
        strata_list = []
    else: 
        strata_list = strata       

    with connect(asedb_fn) as database:

        list_dirs = os.listdir(root_dir)
        for dir_name in list_dirs:
            dir_path = os.path.join(root_dir, dir_name)
            orca_input_path = os.path.join(dir_path, "orca.inp")
            if os.path.exists(orca_input_path):
                atoms = get_atoms_from_orca_input(orca_input_path)
                all_properties = gather_dicts_from_generator_folder(dir_path)
                atoms.info.update(all_properties)
                database.write(atoms, data=atoms.info, name=dir_name)
                num_atoms.append(len(atoms))
                composition_set.add(atoms.get_chemical_formula())
        database.metadata["strata"] = strata_list
        database.metadata["num_atoms"] = num_atoms

    np.savez(
        f"{root_lmdb}/metadata.npz",
        natoms=num_atoms,
        strata=strata,
        compositions=list(composition_set)
    )