from pathlib import Path

import pandas as pd
import os
import numpy as np
from qtaim_gen.source.core.parse_qtaim import (
    gather_imputation,
)

TEST_FILES = Path(__file__).parent / "test_files"


def test_gather_imputation_reaction():
    test_root = str(TEST_FILES / "reaction")
    df = pd.read_json(str(TEST_FILES / "reaction" / "b97d3.json"))
    features_atom = [
        "Lagrangian_K",
        "eta",
    ]
    features_bond = ["Lagrangian_K", "eta"]

    reaction = True
    define_bonds = "qtaim"
    impute_file = str(TEST_FILES / "reaction" / "impute.json")
    if os.path.exists(impute_file):
        os.remove(impute_file)
    impute_dict = gather_imputation(
        df=df,
        features_atom=features_atom,
        features_bond=features_bond,
        root_dir=test_root,
        json_file_imputed=impute_file,
        reaction=reaction,
        define_bonds=define_bonds,
        inp_type="xyz",
    )

    impute_dict_reup = gather_imputation(
        df=df,
        features_atom=features_atom,
        features_bond=features_bond,
        root_dir=test_root,
        json_file_imputed=impute_file,
        reaction=reaction,
        define_bonds=define_bonds,
        inp_type="xyz",
    )
    assert impute_dict == impute_dict_reup, f"not properly reloaded on impute reup: {impute_dict} vs {impute_dict_reup}"

    for k, v in impute_dict.items():
        for k_sub, v_sub in v.items():
            assert type(v_sub["mean"]) == np.float64, "impute dict buggin"
    os.remove(impute_file)


def test_gather_imputation_molecule():
    test_root = str(TEST_FILES / "molecule")
    df = pd.read_json(str(TEST_FILES / "molecule" / "libe_qtaim_test.json"))
    features_atom = [
        "Lagrangian_K",
        "eta",
    ]
    features_bond = ["Lagrangian_K", "eta"]

    reaction = False
    define_bonds = "qtaim"
    impute_file = str(TEST_FILES / "molecule" / "impute.json")
    if os.path.exists(impute_file):
        os.remove(impute_file)
    impute_dict = gather_imputation(
        df=df,
        features_atom=features_atom,
        features_bond=features_bond,
        root_dir=test_root,
        json_file_imputed=impute_file,
        reaction=reaction,
        define_bonds=define_bonds,
        inp_type="xyz",
    )

    impute_dict_reup = gather_imputation(
        df=df,
        features_atom=features_atom,
        features_bond=features_bond,
        root_dir=test_root,
        json_file_imputed=impute_file,
        reaction=reaction,
        define_bonds=define_bonds,
        inp_type="xyz",
    )
    assert impute_dict == impute_dict_reup, "not properly reloaded on impute"

    for k, v in impute_dict.items():
        for k_sub, v_sub in v.items():
            # print(k_sub, v_sub)
            # print(type(v_sub["mean"]))
            assert type(v_sub["mean"]) == np.float64, "impute dict buggin"
    os.remove(impute_file)
