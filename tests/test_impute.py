import pandas as pd 
import os
import numpy as np
from qtaim_gen.source.core.parse_qtaim import (
    gather_imputation,
)


def test_gather_imputation_reaction():
    test_root = "./test_files/reaction/"
    df = pd.read_json("./test_files/reaction/b97d3.json") 
    features_atom = [
        "Lagrangian_K",
        "eta",
    ]
    features_bond = [
        "Lagrangian_K",
        "eta",
        "connected_bond_paths"
    ]
    
    reaction = True
    define_bonds = "qtaim"
    impute_file = "./test_files/reaction/impute.json"
    if os.path.exists(impute_file):
        os.remove(impute_file)
    impute_dict = gather_imputation(
        df = df,
        features_atom = features_atom,
        features_bond = features_bond,
        root_dir=test_root,
        json_file_imputed=impute_file,
        reaction = reaction,
        define_bonds = define_bonds,
    )

    impute_dict_reup = gather_imputation(
        df = df,
        features_atom = features_atom,
        features_bond = features_bond,
        root_dir=test_root,
        json_file_imputed=impute_file,
        reaction = reaction,
        define_bonds = define_bonds,
    )
    assert impute_dict == impute_dict_reup, "not properly reloaded on impute"
    
    for k, v in impute_dict.items():
        for k_sub, v_sub in v.items():
            assert type(v_sub["mean"]) == np.float64, "impute dict buggin"
    os.remove(impute_file)

def test_gather_imputation_molecule():
    test_root = "./test_files/molecule/"
    df = pd.read_pickle("./test_files/molecule/libe_qtaim_test.pkl")
    features_atom = [
        "Lagrangian_K",
        "eta",
    ]
    features_bond = [
        "Lagrangian_K",
        "eta"
    ]
    
    reaction = False
    define_bonds = "qtaim"
    impute_file = "./test_files/molecule/impute.json"
    if os.path.exists(impute_file):
        os.remove(impute_file)
    impute_dict = gather_imputation(
        df = df,
        features_atom = features_atom,
        features_bond = features_bond,
        root_dir=test_root,
        json_file_imputed=impute_file,
        reaction = reaction,
        define_bonds = define_bonds,
    )

    impute_dict_reup = gather_imputation(
        df = df,
        features_atom = features_atom,
        features_bond = features_bond,
        root_dir=test_root,
        json_file_imputed=impute_file,
        reaction = reaction,
        define_bonds = define_bonds,
    )
    assert impute_dict == impute_dict_reup, "not properly reloaded on impute"

    for k, v in impute_dict.items():
        for k_sub, v_sub in v.items():
            #print(k_sub, v_sub)
            #print(type(v_sub["mean"]))
            assert type(v_sub["mean"]) == np.float64, "impute dict buggin"
    os.remove(impute_file)

