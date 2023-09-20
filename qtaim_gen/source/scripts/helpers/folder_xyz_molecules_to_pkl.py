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
    args = parser.parse_args()
    xyz_folder = args.xyz_folder
    pkl_file = args.pkl_file
    determine_bonds = bool(args.determine_bonds)

    # get a list of all xyz files in the folder
    xyz_files = []
    molecules = []
    molecule_graphs = []
    bond_list = []
    identifier = []
    xyz_files_completed = []

    for file in os.listdir(xyz_folder):
        if file.endswith(".xyz"):
            xyz_files.append(file)

    # create a list of pmg molecules from the xyz files
    for ind, xyz_file in enumerate(xyz_files):
        # print(xyz_folder + xyz_file)
        molecule = Molecule.from_file(xyz_folder + xyz_file)

        # bond_cutoff_bonds = connectedMatrix(
        #    molecule, {"H,H": 1.2, "Pt,Pt": 1.5, "Pt,H": 1.2}
        # )
        if determine_bonds:
            try:
                bonds_rdkit = get_bonds_from_rdkit(xyz_folder + xyz_file)
            except:
                bonds_rdkit = []

        bond_list.append(bonds_rdkit)
        molecules.append(molecule)
        identifier.append(ind)
        molecule_graph = MoleculeGraph.with_empty_graph(molecule)
        [molecule_graph.add_edge(bond[0], bond[1]) for bond in bonds_rdkit]
        molecule_graphs.append(molecule_graph)
        xyz_files_completed.append(xyz_file)

    df = {
        "molecule": molecules,
        "molecule_graph": molecule_graphs,
        "bonds": bond_list,
        "ids": identifier,
        "names": xyz_files_completed,
    }
    # convert to pandas dataframe and save as pickle
    df = pd.DataFrame(df)
    pd.to_pickle(df, xyz_folder + pkl_file)
    df_pkl = pd.read_pickle(xyz_folder + pkl_file)


main()
