import argparse
import pandas as pd
import numpy as np
import bson
from qtaim_gen.source.core.parse_qtaim import (
    merge_qtaim_inds,
    gather_imputation,
    get_qtaim_descs,
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
        "e_density",
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
    ]

    features_bond = [
        "Lagrangian_K",
        "Hamiltonian_K",
        "e_density",
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
    ]

    if impute:
        impute_dict = gather_imputation(
            pandas_file,
            features_atom,
            features_bond,
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

    bond_list = []
    impute_count = {i: 0 for i in features_atom}
    impute_count_bond = {i: 0 for i in features_bond}

    for i in features_atom:
        str_atom = "extra_feat_atom_" + i
        pandas_file[str_atom] = ""

    for i in features_bond:
        str_atom = "extra_feat_bond_" + i
        pandas_file[str_atom] = ""

    fail_count = 0
    for ind, row in pandas_file.iterrows():
        tf_count = {i: 0 for i in features_atom}
        tf_count_bond = {i: 0 for i in features_bond}

        bonds = []
        id = row["ids"]
        if define_bonds == "distances":
            bonds = row["bonds"]
        QTAIM_loc = root + "QTAIM/" + str(id) + "/"
        cp_file = QTAIM_loc + "CPprop.txt"
        dft_inp_file = QTAIM_loc + "input.in"

        qtaim_descs = get_qtaim_descs(cp_file, verbose=False)
        mapped_descs = merge_qtaim_inds(
            qtaim_descs=qtaim_descs,
            bond_list=bonds,
            dft_inp_file=dft_inp_file,
            margin=2.0,
            define_bonds=define_bonds,
        )

        # print(mapped_descs_products)
        # fill in imputation values
        for k, v in mapped_descs.items():
            if v == {}:
                if type(k) == tuple:
                    bonds.append(list(k))
                    for i in features_bond:
                        if tf_count_bond[i] == 0:  # track if this feature is missing
                            tf_count_bond[i] = 1

                        if impute:
                            mapped_descs[k][i] = impute_dict["bond"][i]["median"]
                        else:
                            mapped_descs[k][i] = -1
                            if ind not in drop_list:
                                drop_list.append(ind)

                else:
                    for i in features_atom:
                        if tf_count[i] == 0:  # track if this feature is missing
                            tf_count[i] = 1

                        if impute:
                            mapped_descs[k][i] = impute_dict["atom"][i]["median"]
                        else:
                            mapped_descs[k][i] = -1
                            if ind not in drop_list:
                                drop_list.append(ind)

        # get all the values of a certain key for every dictionary in the dicitonary
        cps = mapped_descs.keys()
        flat = {}
        for cp in cps:
            for i in features_atom:
                if type(cp) != tuple:
                    # check if the key exists
                    name = "extra_feat_atom_" + i
                    if name not in flat.keys():
                        flat[name] = []
                    flat[name].append(mapped_descs[cp][i])

            for i in features_bond:
                if type(cp) == tuple:
                    # check if the key exists
                    name = "extra_feat_bond_" + i
                    if name not in flat.keys():
                        flat[name] = []
                    flat[name].append(mapped_descs[cp][i])

        # update the pandas file with the new values
        for k, v in flat.items():
            if "bond" in k:
                pandas_file.at[ind, k] = [v]
            else:
                pandas_file.at[ind, k] = np.array(v)

        keys = mapped_descs.keys()
        # filter keys that aren't tuples
        keys = [x for x in keys if type(x) == tuple]
        bond_list.append([keys])

        """except:
            drop_list.append(ind)
            bond_list.append([])
            # iterate over all the features and set them to -1
            for i in features_atom:
                name = "extra_feat_atom_" + i
                pandas_file.at[ind, name] = -1

            # iterate over all the features and set them to -1
            for i in features_bond:
                name = "extra_feat_bond_" + i
                pandas_file.at[ind, name] = -1

            fail_count += 1"""

    for k, v in impute_count.items():
        impute_count[k] = v + tf_count[k]
    for k, v in impute_count_bond.items():
        impute_count_bond[k] = v + tf_count_bond[k]

    with open(root + "impute_counts.txt", "w") as f:
        f.write("---------- impute_count ---------- \n")
        f.write(str(impute_count))
        f.write("\n")
        f.write("---------- impute_count_bond ---------- \n")
        f.write(str(impute_count_bond))
        f.write("\n")

    pandas_file["extra_feat_bond_indices_qtaim"] = bond_list
    if update_bonds_w_qtaim:
        if "bonds" in pandas_file.columns:
            pandas_file["bonds_original"] = pandas_file["bonds"]
        pandas_file["bonds"] = bond_list

    print(fail_count / ind)

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
