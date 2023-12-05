import argparse
import pandas as pd

import bson
from qtaim_gen.source.core.parse_qtaim import (
    gather_imputation,
    gather_qtaim_features,
)


def main():
    drop_list = []

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=str,
        default="../../data/rapter/",
        help="location of json file",
    )
    parser.add_argument(
        "--file_in",
        type=str,
        default="20230512_mpreact_assoc.bson",
        help="json file",
    )
    parser.add_argument(
        "--impute",
        action="store_true",
        help="impute values",
    )

    parser.add_argument(
        "--file_out",
        type=str,
        default="qtaim_nonimputed.json",
        help="output json file",
    )

    parser.add_argument(
        "--reaction",
        action="store_true",
        help="reaction or not",
    )

    parser.add_argument(
        "--update_bonds_w_qtaim",
        action="store_true",
        help="update bonds with qtaim bond path definitions",
    )

    parser.add_argument(
        "-define_bonds", choices=["distances", "qtaim"], default="qtaim"
    )

    args = parser.parse_args()
    root = args.root
    file_in = args.file_in
    impute = bool(args.impute)
    reaction = bool(args.reaction)
    file_out = args.file_out
    define_bonds = args.define_bonds
    update_bonds_w_qtaim = args.update_bonds_w_qtaim
    print("define_bonds: {}".format(define_bonds))
    print("impute: {}".format(impute))
    print("root: {}".format(root))
    print("file_in: {}".format(file_in))
    print("file_out: {}".format(file_out))
    print("reaction: {}".format(reaction))

    print("reading file from: {}".format(root + file_in))

    if file_in.endswith(".json"):
        path_json = root + file_in
        pandas_file = pd.read_json(path_json)
        print(pandas_file.shape)
    elif file_in.endswith(".pkl"):
        path_pkl = root + file_in
        pandas_file = pd.read_pickle(path_pkl)
        # if pandas file is a dict convert to dataframe
        if type(pandas_file) == dict:
            pandas_file = pd.DataFrame(pandas_file)
        print(len(pandas_file["ids"]))
    else:
        path_bson = root + file_in
        with open(path_bson, "rb") as f:
            data = bson.decode_all(f.read())
        pandas_file = pd.DataFrame(data)
        print(pandas_file.shape)

    if impute:
        imputed_file = root + "impute_vals.json"
    print("df type {}".format(type(pandas_file)))
    features_atom = [
        "Lagrangian_K",
        "Hamiltonian_K",
        "energy_density",
        "lap_e_density",
        "e_loc_func",
        "ave_loc_ion_E",
        "delta_g_promolecular",
        "delta_g_hirsh",
        "esp_nuc",
        "esp_e",
        "esp_total",
        "grad_norm",
        "lap_norm",
        "eig_hess",
        "det_hessian",
        "ellip_e_dens",
        "eta",
        "density_beta",
        "density_alpha",
        "spin_density",
        "lol",
    ]

    features_bond = [
        "Lagrangian_K",
        "Hamiltonian_K",
        "energy_density",
        "lap_e_density",
        "e_loc_func",
        "ave_loc_ion_E",
        "delta_g_promolecular",
        "delta_g_hirsh",
        "esp_nuc",
        "esp_e",
        "esp_total",
        "grad_norm",
        "lap_norm",
        "eig_hess",
        "det_hessian",
        "ellip_e_dens",
        "eta",
        "density_beta",
        "density_alpha",
        "spin_density",
        "lol",

    ]

    if impute:
        impute_dict = gather_imputation(
            df=pandas_file,
            features_atom=features_atom,
            features_bond=features_bond,
            root_dir=root,
            json_file_imputed=imputed_file,
            reaction=reaction,
            define_bonds=define_bonds,
        )
        for i in impute_dict.keys():
            print("-" * 20 + i + "-" * 20)
            for k, v in impute_dict[i].items():
                print("{} \t\t\t {}".format(k.ljust(20, " "), v["mean"]))

        print("Done gathering imputation data...")

    pandas_file, drop_list = gather_qtaim_features(
        pandas_file=pandas_file,
        root=root,
        features_atom=features_atom,
        features_bond=features_bond,
        reaction=reaction,
        define_bonds=define_bonds,
        update_bonds_w_qtaim=update_bonds_w_qtaim,
        impute=impute,
        impute_dict=impute_dict,
    )

    # if impute false then drop the rows that have -1 values
    if not impute:
        pandas_file.drop(drop_list, inplace=True)
        pandas_file.reset_index(drop=True, inplace=True)

    print("length of drop list: {}".format(len(drop_list)))
    print("done gathering and imputing features...")
    # save the pandas file
    print(pandas_file.shape)

    # pandas_file.to_json(root + file_out)

    if file_in.endswith(".json"):
        path_json = root + file_out
        pandas_file.to_json(root + file_out)

    elif file_in.endswith(".pkl"):
        path_pkl = root + file_out
        pandas_file.to_pickle(path_pkl)

    else:
        print("file format not supported")


main()
