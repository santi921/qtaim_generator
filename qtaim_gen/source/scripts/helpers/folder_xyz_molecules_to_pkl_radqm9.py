"""
    Helper script to convert a folder of xyz files to a json file of pmg molecules + bonds + properties
    Usage: 
        python create_json_from_xyz.py --xyz_folder xyz_folder --json_file json_file
    
"""
import argparse
import os, json, ast
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
        "--debug",
        action="store_true",
        help="debug",
    )


    args = parser.parse_args()
    xyz_folder = args.xyz_folder
    pkl_file = args.pkl_file
    debug = bool(args.debug)
    determine_bonds = bool(args.determine_bonds)

    # get a list of all xyz files in the folder
    xyz_files = []
    molecules = []
    molecule_graphs = []
    bond_list = []
    identifier = []
    xyz_files_completed = []
    position_types = []
    spin = []
    charge = []
    partial_spins = []
    partial_charges = []
    mol_ids = []
    energies = []

    for file in os.listdir(xyz_folder):
        if file.endswith(".xyz"):
            xyz_files.append(file)

    if debug:
        xyz_files = xyz_files[:10]

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

        if determine_bonds:
            [molecule_graph.add_edge(bond[0], bond[1]) for bond in bonds_rdkit]
        xyz_files_completed.append(xyz_file)

    
        with open(xyz_folder + xyz_file, "r") as f:
            lines = f.readlines()
        comment = lines[1].strip()
        #print(comment)

        temp_spin = int(comment.split()[3].split("=")[1])
        spin.append(temp_spin)

        charge_temp = int(comment.split()[2].split("=")[1])
        charge.append(charge_temp)

        
        partial_charges_temp = comment.split("partial_charge")[1].split("\"")[1][6:]
        partial_charges_temp = ast.literal_eval(partial_charges_temp)
        partial_charges.append(partial_charges_temp)
        
        partial_spins_temp = comment.split("partial_spin")[1].split("\"")[1][6:]
        partial_spins_temp = ast.literal_eval(partial_spins_temp)
        partial_spins.append(partial_spins_temp)
        
        energy = comment.split()[1].split("=")[1]
        energies.append(float(energy))
        
        mol_id_temp = comment.split()[-5].split("=")[1]
        mol_ids.append(mol_id_temp)
        
        position_type = comment.split()[-4].split("=")[1]
        position_types.append(position_type)
        
        molecule_graph = MoleculeGraph.with_empty_graph(molecule)
        molecule_graphs.append(molecule_graph)
        molecules.append(molecule)

    df = {
        "molecule": molecules,
        "molecule_graph": molecule_graphs,
        "ids": identifier,
        "names": xyz_files_completed,
        "partial_spin": partial_spins, 
        "partial_charges": partial_charges,
        "mol_id": mol_ids, 
        "position_type": position_types,
        "energy": energies,
    }
    if determine_bonds:
        df["bonds"] = bond_list
    
    df["spin"] = spin
    df["charge"] = charge
    # convert to pandas dataframe and save as pickle
    df = pd.DataFrame(df)
    pd.to_pickle(df, xyz_folder + pkl_file)
    #df_pkl = pd.read_pickle(xyz_folder + pkl_file)


main()
