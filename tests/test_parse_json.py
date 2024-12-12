import os 
import pandas as pd 
import numpy as np 
import json

from qtaim_gen.source.core.parse_json import (
    get_qtaim_data, 
    get_qtaim_data_impute,
    gather_impute, 
    get_data, 
    get_keys_qtaim, 
    parse_bond
)

from pymatgen.core import Molecule
from pymatgen.analysis.graphs import MoleculeGraph
from qtaim_gen.source.core.bonds import get_bonds_from_rdkit
from qtaim_gen.source.core.io import convert_inp_to_xyz


root_folder = "./test_files/QTAIM/"


def test_get_qtaim_data():

    file = "./test_files/QTAIM/1/qtaim.json"
    with open(file, "r") as f:
        qtaim_dict = json.load(f)
    qtaim_dict = get_qtaim_data(qtaim_dict)

    atom_count, bond_count = 0, 0
    for keys in qtaim_dict.keys():
        if "atom" in keys:
            if atom_count==0: 
                reference_atom = len(qtaim_dict[keys])
            atom_count += 1
            assert len(qtaim_dict[keys]) == reference_atom, f"atom keys are not the same length"
        if "bond" in keys:
            if bond_count==0:
                reference_bond = len(qtaim_dict[keys])
            assert len(qtaim_dict[keys]) == reference_bond, f"bond keys are not the same length"
            bond_count += 1

    assert atom_count == bond_count-1, "atom and bond keys are not the same length"
        

def test_get_data():
    data_full = get_data(
        root=root_folder,  
        full_descriptors=True
    )
    #print(data_full.keys())

    # keys to test from each category
    test_keys = [
        "extra_feat_atom_esp_nuc",
        "extra_feat_bond_esp_total",
        "bond_list_fuzzy",
        "bond_list_qtaim",
        "mbis_charge",
        "hirshfeld_dipole_mag",
        "bader_spin",
        "becke_atomic_dipole",
        "adch_dipole",
        "ibsi_data", 
        "ALIE_Positive_skewness"
    ]

    for key in test_keys:
        assert key in data_full.keys(), f"{key} not in data_full"
        assert len(data_full[key]) == 4, f"{key} not the correct length"

    assert np.isclose(data_full["ALIE_Positive_skewness"][2], -0.22218, atol=1e-5), "ALIE_Positive_skewness is not correct"
    assert np.isclose(data_full["extra_feat_atom_esp_nuc"][2][2], 1069176.567, atol=1e-5), "extra_feat_atom_esp_nuc is not correct" 
    assert len(data_full["extra_feat_bond_esp_total"][2]) == 20, "extra_feat_bond_esp_total is not the correct length"
    assert np.isclose(data_full["extra_feat_bond_esp_total"][2][-2], 0.8080011181, atol=1e-5), "extra_feat_bond_esp_total is not correct" 
    assert len(data_full["ibsi_data"][2]) == 28, "ibsi_data is not the correct length"
    assert np.isclose(data_full["ibsi_data"][2][(5, 17)], 0.05525, atol=1e-5), "ibsi_data is not correct" 
    assert len(data_full["bond_list_fuzzy"][2]) == 32, "bond_list_fuzzy is not the correct length"
    assert (0, 8) in data_full["bond_list_qtaim"][2], "bond_list_qtaim val error"
    assert len(data_full["bond_list_qtaim"][2]) == 20, "bond_list_qtaim is not the correct length"
    assert len(data_full["mbis_charge"][2]) == 18, "mbis_charge is not the correct length"
    assert np.isclose(data_full["mbis_charge"][2][-2], 0.32306682, atol=1e-5), "mbis_charge is not correct"
    assert len(data_full["bader_spin"][2]) == 18, "bader_spin is not the correct length"
    assert np.isclose(data_full["bader_spin"][2][-2], 0.0, atol=1e-5), "bader_spin is not correct"
    assert np.isclose(data_full["hirshfeld_dipole_mag"][2], 0.960007, atol=1e-5), "hirshfeld_dipole_mag is not correct"
    assert len(data_full["becke_atomic_dipole"][2]) == 18, "becke_atomic_dipole is not the correct length"
    ref_dipole = np.array(
        [
                0.486347,
                0.067275,
                0.246652
    ])
    target_dipole = np.array(data_full["becke_atomic_dipole"][2][1])
    bool_safe = np.isclose(target_dipole, ref_dipole, atol=1e-5)
    assert np.all(bool_safe), "becke_atomic_dipole is not correct"
    

def test_gather_impute():

    impute_result = gather_impute(
        root=root_folder, 
        full_descriptors=True
    )
    assert len(impute_result) == 86
    for k, v in impute_result.items():
        assert ["mean", "median"] == list(v.keys())
    

def test_get_keys_qtaim():
    file = "./test_files/QTAIM/1/qtaim.json"
    with open(file, "r") as f:
        qtaim_dict = json.load(f)
    # check that all the keys with atom are the same len
    #print(qtaim_dict.keys())
    keys = get_keys_qtaim(qtaim_dict)
    
    atom_count, bond_count = 0, 0
    for keys in keys:
        if "atom" in keys:
            atom_count += 1
        if "bond" in keys:
            bond_count += 1

    assert atom_count == bond_count, "atom and bond keys are not the same length"
        

def test_parse_bond():
    file = "./test_files/QTAIM/1/bond.json"
    with open(file, "r") as f:
        bond_dict = json.load(f)
    
    bond_res = parse_bond(bond_dict, cutoff=0.05)

    assert len(bond_res) == 4
    assert type(bond_res["bond_list_fuzzy"]) == list, "bond_list_fuzzy is not a list"
    assert type(bond_res["bond_list_ibsi"]) == list, "bond_list_qtaim is not a list"
    assert type(bond_res["ibsi_data"]) == dict, "ibsi_data is not a list"
    assert type(bond_res["fuzzy_data"]) == dict, "fuzzy_data is not a list"

    bond_res_10 = parse_bond(bond_dict, cutoff=0.1)
    assert len(bond_res["bond_list_ibsi"]) > len(bond_res_10["bond_list_ibsi"]), "ibsi data is not being filtered correctly"


def test_get_qtaim_data_impute():
    file = "./test_files/QTAIM/1/qtaim.json"
    with open(file, "r") as f:
        qtaim_dict = json.load(f)
    # check that all the keys with atom are the same len
    #print(qtaim_dict.keys())
    keys = get_keys_qtaim(qtaim_dict)
    qtaim_impute = get_qtaim_data_impute(
        qtaim_dict, 
        keys
    )

    # assert the len of all the items with 'atom' or 'bond' are the same
    atom_count, bond_count = 0, 0
    for keys in keys:
        if "atom" in keys:
            if atom_count==0: 
                reference_atom = len(qtaim_impute[keys])
            atom_count += 1
            assert len(qtaim_impute[keys]) == reference_atom, f"atom keys are not the same length"
        if "bond" in keys:
            if bond_count==0:
                reference_bond = len(qtaim_impute[keys])
            assert len(qtaim_impute[keys]) == reference_bond, f"bond keys are not the same length"
            bond_count += 1
    
    assert atom_count == bond_count, "atom and bond keys are not the same length"


def test_merge():

    pkl_file = "test.pkl"
    gather_tf = True

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
    
    # create a list of pmg molecules from the xyz files
    for ind, in_file in enumerate(inp_files):    
        if gather_tf: 
            data = get_data(
                root=root_folder + folder_list[ind] + "/",
                full_descriptors=True, 
                root_folder_tf=False
            )
            keys_extra_feats = list(data.keys())
            for key in keys_extra_feats:
                if key not in extra_val_dict:
                    extra_val_dict[key] = []
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



        try:
            bonds_rdkit = get_bonds_from_rdkit(root_folder + xyz_file)
        except:
            bonds_rdkit = []
            [molecule_graph.add_edge(bond[0], bond[1]) for bond in bonds_rdkit]
        
        molecule_graph = MoleculeGraph.with_empty_graph(molecule)
        molecule_graphs.append(molecule_graph)
        molecules.append(molecule)

        bond_list.append(bonds_rdkit)
        identifier.append(ind)
        xyz_files_completed.append(xyz_file)
        
        # append

    df = {
        "molecule": molecules,
        "molecule_graph": molecule_graphs,
        "ids": identifier,
        "names": xyz_files_completed,
    }
    
    df["bonds"] = bond_list
    df["spin"] = spin
    df["charge"] = charge

    if gather_tf:
        for key in keys_extra_feats:
            df[key] = extra_val_dict[key]
            
    
    # convert to pandas dataframe and save as pickle
    df = pd.DataFrame(df)
    pd.to_pickle(df, root_folder + pkl_file)
    
    # check that all the keys are in the dataframe
    assert "molecule" in df.keys(), "molecule not in df"
    assert "molecule_graph" in df.keys(), "molecule_graph not in df"
    assert "ids" in df.keys(), "ids not in df"
    assert "names" in df.keys(), "names not in df"
    assert "bonds" in df.keys(), "bonds not in df"
    assert "spin" in df.keys(), "spin not in df"
    assert "charge" in df.keys(), "charge not in df"
    for key in keys_extra_feats:
        assert key in df.keys(), f"{key} not in df"
    print(df)

test_merge()