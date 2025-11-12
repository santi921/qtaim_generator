#!/usr/bin/env python3

"""
Helper script to convert a folder of xyz files to a json file of pmg molecules + bonds + properties
Usage:
    python create_json_from_xyz.py --xyz_folder xyz_folder --json_file json_file

"""

import argparse
import os, json
import pandas as pd
from pymatgen.core import Molecule
from pymatgen.analysis.graphs import MoleculeGraph
from qtaim_gen.source.core.bonds import get_bonds_from_rdkit
from qtaim_gen.source.core.io import convert_inp_to_xyz
from tqdm import tqdm
from qtaim_gen.source.core.parse_json import get_data


def main(argv=None):

    parser = argparse.ArgumentParser()
    parser.add_argument("-root_folder", type=str, default="./ORCA/")
    parser.add_argument("-pkl_file", type=str, default="molecules.pkl")

    parser.add_argument(
        "--gather_json_data", action="store_true", help="Gather data from json files"
    )

    args = parser.parse_args(argv)
    root_folder = args.root_folder
    pkl_file = args.pkl_file
    gather_tf = bool(args.gather_json_data)

    # get a list of all xyz files in the folder
    inp_files = []
    folder_list = []
    molecules = []
    molecule_graphs = []
    bond_list = []
    identifier = []
    xyz_files_completed = []
    spin = []
    charge = []
    extra_val_dict = {}

    # go through every folder in root and find in files in those folders
    for folder in os.listdir(root_folder):
        if os.path.isdir(root_folder + folder):
            for file in os.listdir(root_folder + folder):
                if file.endswith(".in"):
                    inp_files.append(folder + "/" + file)
                    folder_list.append(folder)

    if gather_tf:
        init_keys = True

    # create a list of pmg molecules from the xyz files
    # for ind, in_file in enumerate(inp_files):
    # rewrite for tqdm
    ind = 0
    for in_file in tqdm(inp_files):

        if gather_tf:

            try:
                data = get_data(
                    root=root_folder + folder_list[ind] + "/",
                    full_descriptors=True,
                    root_folder_tf=False,
                )
            except Exception:
                data = {}
                print("Error in reading data from json file - ", in_file)

            if init_keys:  # only need to do this once
                keys_extra_feats = list(data.keys())
                init_keys = False

            for key in keys_extra_feats:
                if key not in extra_val_dict:
                    extra_val_dict[key] = []
                if key not in data:
                    extra_val_dict[key].append(None)
                else:
                    extra_val_dict[key].append(data[key])

        xyz_file = in_file.replace(".in", ".xyz")
        convert_inp_to_xyz(root_folder + in_file, root_folder + xyz_file)
        molecule = Molecule.from_file(root_folder + xyz_file)

        with open(root_folder + xyz_file, "r") as f:
            lines = f.readlines()

        comment = lines[1].strip()
        temp_spin = int(comment.split()[1])
        spin.append(temp_spin)
        charge.append(int(comment.split()[0]))
        # add charge to molecule
        molecule.set_charge_and_spin(
            charge=int(comment.split()[0]), spin_multiplicity=temp_spin
        )

        molecule_graph = MoleculeGraph.with_empty_graph(molecule)
        molecule_graphs.append(molecule_graph)
        molecules.append(molecule)

        try:
            bonds_rdkit = get_bonds_from_rdkit(root_folder + xyz_file)
        except Exception:
            bonds_rdkit = []
        [molecule_graph.add_edge(bond[0], bond[1]) for bond in bonds_rdkit]

        bond_list.append(bonds_rdkit)
        identifier.append(ind)
        xyz_files_completed.append(xyz_file)
        ind += 1

        # append

    df = {
        "molecule": molecules,
        "molecule_graph": molecule_graphs,
        "ids": identifier,
        "names": xyz_files_completed,
        "bond": bond_list,
        "spin": spin,
        "charge": charge,
    }

    if gather_tf:
        for key in keys_extra_feats:
            df[key] = extra_val_dict[key]

    # convert to pandas dataframe and save as pickle
    df = pd.DataFrame(df)
    # print(df)
    pd.to_pickle(df, root_folder + pkl_file)


if __name__ == "__main__":
    raise SystemExit(main())
