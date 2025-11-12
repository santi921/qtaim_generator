import argparse
import pandas as pd

import bson
from qtaim_gen.source.core.parse_json import (
        get_keys_qtaim,
        get_qtaim_data_impute,
        get_qtaim_data,
        gather_impute,
        parse_bond,
        gather_dicts,
        get_data
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
    parse_charges = bool(args.charges)
    file_out = args.file_out
    define_bonds = args.define_bonds
    update_bonds_w_qtaim = args.update_bonds_w_qtaim


    print("define_bonds: {}".format(define_bonds))
    print("impute: {}".format(impute))
    print("root: {}".format(root))
    print("file_in: {}".format(file_in))
    print("file_out: {}".format(file_out))
    print("reaction: {}".format(reaction))
    print("parse_charges_spin: {}".format(parse_charges))
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


    impute_dict = {}

    

main()
