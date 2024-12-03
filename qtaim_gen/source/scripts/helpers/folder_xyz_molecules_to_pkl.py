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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-xyz_folder", type=str, default="../../../../data/xyz/")
    parser.add_argument(
        "-pkl_file", type=str, default="../../../../data/xyz/molecules.pkl"
    )
    parser.add_argument("--determine_bonds", action="store_true")
    parser.add_argument(
        "--pull_spin",
        action="store_true",
        help="Pull spin from xyz, should be the 2nd item in the comment line",
    )
    parser.add_argument(
        "--pull_charge",
        action="store_true",
        help="Pull charge from xyz, should be the 1st item in the comment line",
    )
    args = parser.parse_args()
    xyz_folder = args.xyz_folder
    pkl_file = args.pkl_file
    pull_spin = bool(args.pull_spin)
    pull_charge = bool(args.pull_charge)
    determine_bonds = bool(args.determine_bonds)

    # get a list of all xyz files in the folder
    xyz_files = []
    molecules = []
    molecule_graphs = []
    bond_list = []
    identifier = []
    xyz_files_completed = []
    spin = []
    charge = []

    for file in os.listdir(xyz_folder):
        if file.endswith(".xyz"):
            xyz_files.append(file)

    # create a list of pmg molecules from the xyz files
    for ind, xyz_file in enumerate(xyz_files):
        # print(xyz_folder + xyz_file)
        molecule = Molecule.from_file(xyz_folder + xyz_file)

        if determine_bonds:
            try:
                bonds_rdkit = get_bonds_from_rdkit(xyz_folder + xyz_file)
            except:
                bonds_rdkit = []
            bond_list.append(bonds_rdkit)
        identifier.append(ind)

        xyz_files_completed.append(xyz_file)

        if pull_spin or pull_charge:
            with open(xyz_folder + xyz_file, "r") as f:
                lines = f.readlines()
            comment = lines[1].strip()

        if pull_spin:
            temp_spin = int(comment.split()[1])
            spin.append(temp_spin)
        else:
            temp_spin = None

        if pull_charge:
            charge.append(int(comment.split()[0]))
            # add charge to molecule
            molecule.set_charge_and_spin(
                charge=int(comment.split()[0]), spin_multiplicity=temp_spin
            )

        molecule_graph = MoleculeGraph.with_empty_graph(molecule)
        
        if determine_bonds:
            [molecule_graph.add_edge(bond[0], bond[1]) for bond in bonds_rdkit]
        
        molecule_graphs.append(molecule_graph)
        molecules.append(molecule)

    df = {
        "molecule": molecules,
        "molecule_graph": molecule_graphs,
        "ids": identifier,
        "names": xyz_files_completed,
    }
    if determine_bonds:
        df["bonds"] = bond_list
    if pull_spin:
        df["spin"] = spin
    if pull_charge:
        df["charge"] = charge
    # convert to pandas dataframe and save as pickle
    df = pd.DataFrame(df)
    pd.to_pickle(df, xyz_folder + pkl_file)
    df_pkl = pd.read_pickle(xyz_folder + pkl_file)


main()
