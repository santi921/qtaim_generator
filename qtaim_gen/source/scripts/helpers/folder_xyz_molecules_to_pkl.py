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
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds


def connectedMatrix(struct_pmg, bond_length_dict):
    cM = {}  # connected Matrix
    for ind1, site1 in enumerate(struct_pmg):
        # print(ind1, site1.specie)
        nameStr = str(site1.specie) + str(ind1)
        cM[nameStr] = []
        for ind2, site2 in enumerate(struct_pmg):
            dict_key = str(site1.specie) + "," + str(site2.specie)
            dict_key_rev = str(site2.specie) + "," + str(site1.specie)

            if dict_key in bond_length_dict.keys():
                # print(dict_key)
                pass
            elif dict_key_rev in bond_length_dict.keys():
                # print(dict_key_rev)
                dict_key = dict_key_rev

            if (
                ind1 != ind2
                and np.linalg.norm(site1.coords - site2.coords)
                <= bond_length_dict[dict_key]
            ):
                cM[nameStr].append(ind2)

    # convert to list of lists of format [[origin, connected1], [origin, connected2]...]
    list_of_lists = []
    # print(cM)
    for key in cM.keys():
        for connected in cM[key]:
            # connected_ind = "".join([i for i in connected if i.isdigit()])
            connected_ind = int("".join([i for i in key if i.isdigit()]))
            list_of_lists.append([connected_ind, connected])

    # print(list_of_lists)
    return list_of_lists


def get_bonds_from_rdkit(xyz):
    # Create an RDKit molecule from the XYZ string
    rdkit_molecule = Chem.rdmolfiles.MolFromXYZFile(xyz)
    # Retrieve the bonds in the RDKit molecule
    # conn_mol = Chem.Mol(rdkit_molecule)
    rdDetermineBonds.DetermineConnectivity(rdkit_molecule)
    bonds = rdkit_molecule.GetBonds()
    bond_list = []
    # Iterate over the bonds and print their information
    for bond in bonds:
        atom1_idx = bond.GetBeginAtomIdx()
        atom2_idx = bond.GetEndAtomIdx()
        atom1 = rdkit_molecule.GetAtomWithIdx(atom1_idx)
        atom2 = rdkit_molecule.GetAtomWithIdx(atom2_idx)
        bond_type = bond.GetBondType()
        # print(
        #    f"Bond between atom {atom1.GetSymbol()}({atom1.GetIdx()}) and atom {atom2.GetSymbol()}({atom2.GetIdx()}): {bond_type}"
        # )
        bond_list.append([atom1_idx, atom2_idx])
    return bond_list


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-xyz_folder", type=str, default="../../../../data/xyz/")
    parser.add_argument(
        "-pkl_file", type=str, default="../../../../data/xyz/molecules.pkl"
    )
    args = parser.parse_args()
    xyz_folder = args.xyz_folder
    pkl_file = args.pkl_file

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
        bond_cutoff_bonds = connectedMatrix(
            molecule, {"H,H": 1.2, "Pt,Pt": 1.5, "Pt,H": 1.2}
        )
        bonds_rdkit = get_bonds_from_rdkit(xyz_folder + xyz_file)

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
    pd.to_pickle(df, pkl_file)
    df_pkl = pd.read_pickle(pkl_file)


main()
