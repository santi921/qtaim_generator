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

    if reaction:
        bond_list_reactants = []
        bond_list_products = []
        impute_count_reactants = {i: 0 for i in features_atom}
        impute_count_products = {i: 0 for i in features_atom}
        impute_count_reactants_bond = {i: 0 for i in features_bond}
        impute_count_products_bond = {i: 0 for i in features_bond}

        for i in features_atom:
            str_reactant = "extra_feat_atom_reactant_" + i
            str_product = "extra_feat_atom_product_" + i
            pandas_file[str_reactant] = ""
            pandas_file[str_product] = ""

        for i in features_bond:
            str_reactant = "extra_feat_bond_reactant_" + i
            str_product = "extra_feat_bond_product_" + i
            pandas_file[str_reactant] = ""
            pandas_file[str_product] = ""

        fail_count = 0

        for ind, row in pandas_file.iterrows():
            tf_count_reactants = {i: 0 for i in features_atom}
            tf_count_products = {i: 0 for i in features_atom}
            tf_count_reactants_bond = {i: 0 for i in features_bond}
            tf_count_products_bond = {i: 0 for i in features_bond}

            try:
                reaction_id = row["reaction_id"]
                if define_bonds == "distances":
                    bonds_reactants = row["reactant_bonds"]
                    bonds_products = row["product_bonds"]

                QTAIM_loc_reactant = root + "QTAIM/" + str(reaction_id) + "/reactants/"
                cp_file_reactants = QTAIM_loc_reactant + "CPprop.txt"
                dft_inp_file_reactant = QTAIM_loc_reactant + "input.in"

                QTAIM_loc_product = root + "QTAIM/" + str(reaction_id) + "/products/"
                cp_file_products = QTAIM_loc_product + "CPprop.txt"
                dft_inp_file_product = QTAIM_loc_product + "input.in"

                qtaim_descs_reactants = get_qtaim_descs(
                    cp_file_reactants, verbose=False
                )
                qtaim_descs_products = get_qtaim_descs(cp_file_products, verbose=False)

                mapped_descs_reactants = merge_qtaim_inds(
                    qtaim_descs=qtaim_descs_reactants,
                    bond_list=bonds_reactants,
                    dft_inp_file=dft_inp_file_reactant,
                    margin=2.0,
                    define_bonds=define_bonds,
                )
                mapped_descs_products = merge_qtaim_inds(
                    qtaim_descs=qtaim_descs_products,
                    bond_list=bonds_products,
                    dft_inp_file=dft_inp_file_product,
                    margin=2.0,
                    define_bonds=define_bonds,
                )

                bonds_products, bonds_reactants = [], []

                for k, v in mapped_descs_reactants.items():
                    if v == {}:
                        if type(k) == tuple:
                            bonds_reactants.append(list(k))
                            for i in features_bond:
                                if (
                                    tf_count_reactants_bond[i] == 0
                                ):  # track if this feature is missing
                                    tf_count_reactants_bond[i] = 1

                                if impute:
                                    mapped_descs_reactants[k][i] = impute_dict["bond"][
                                        i
                                    ]["median"]
                                else:
                                    mapped_descs_reactants[k][i] = -1
                                    if ind not in drop_list:
                                        drop_list.append(ind)

                        else:
                            for i in features_atom:
                                if (
                                    tf_count_reactants[i] == 0
                                ):  # track if this feature is missing
                                    tf_count_reactants[i] = 1

                                if impute:
                                    mapped_descs_reactants[k][i] = impute_dict["atom"][
                                        i
                                    ]["median"]
                                else:
                                    mapped_descs_reactants[k][i] = -1
                                    if ind not in drop_list:
                                        drop_list.append(ind)

                for k, v in mapped_descs_products.items():
                    if v == {}:
                        if type(k) == tuple:  # bond cp
                            for i in features_bond:
                                if tf_count_products_bond[i] == 0:
                                    tf_count_products_bond[i] = 1

                                bonds_products.append(list(k))
                                if impute:
                                    mapped_descs_products[k][i] = impute_dict["bond"][
                                        i
                                    ]["median"]

                                else:
                                    mapped_descs_products[k][i] = -1
                                    if ind not in drop_list:
                                        drop_list.append(ind)
                        else:  # atom cp
                            for i in features_atom:
                                if tf_count_products[i] == 0:
                                    tf_count_products[i] = 1

                                if impute:
                                    mapped_descs_products[k][i] = impute_dict["atom"][
                                        i
                                    ]["median"]

                                else:
                                    mapped_descs_products[k][i] = -1
                                    if ind not in drop_list:
                                        drop_list.append(ind)

                cps_reactants = mapped_descs_reactants.keys()
                cps_products = mapped_descs_products.keys()
                flat_reactants, flat_products = {}, {}

                for cps_reactant in cps_reactants:
                    for i in features_atom:
                        if type(cps_reactant) != tuple:
                            # check if the key exists
                            name = "extra_feat_atom_reactant_" + i
                            if name not in flat_reactants.keys():
                                flat_reactants[name] = []
                            flat_reactants[name].append(
                                mapped_descs_reactants[cps_reactant][i]
                            )

                    for i in features_bond:
                        if type(cps_reactant) == tuple:
                            # check if the key exists
                            name = "extra_feat_bond_reactant_" + i
                            if name not in flat_reactants.keys():
                                flat_reactants[name] = []
                            flat_reactants[name].append(
                                mapped_descs_reactants[cps_reactant][i]
                            )

                for cps_product in cps_products:
                    for i in features_atom:
                        if type(cps_product) != tuple:
                            # check if the key exists
                            name = "extra_feat_atom_product_" + i
                            if name not in flat_products.keys():
                                flat_products[name] = []
                            flat_products[name].append(
                                mapped_descs_products[cps_product][i]
                            )

                    for i in features_bond:
                        if type(cps_product) == tuple:
                            # check if the key exists
                            name = "extra_feat_bond_product_" + i
                            if name not in flat_products.keys():
                                flat_products[name] = []
                            flat_products[name].append(
                                mapped_descs_products[cps_product][i]
                            )

                # update the pandas file with the new values
                for k, v in flat_reactants.items():
                    if "bond" in k:
                        pandas_file.at[ind, k] = [v]
                    else:
                        pandas_file.at[ind, k] = np.array(v)

                for k, v in flat_products.items():
                    if "bond" in k:
                        pandas_file.at[ind, k] = [v]
                    else:
                        pandas_file.at[ind, k] = np.array(v)

                keys_products = mapped_descs_products.keys()
                keys_reactants = mapped_descs_reactants.keys()
                # filter keys that aren't tuples
                keys_products = [x for x in keys_products if type(x) == tuple]
                keys_reactants = [x for x in keys_reactants if type(x) == tuple]
                bond_list_reactants.append([keys_reactants])
                bond_list_products.append([keys_products])

            except:
                drop_list.append(ind)
                bond_list_reactants.append([])
                bond_list_products.append([])
                # iterate over all the features and set them to -1
                for i in features_atom:
                    name = "extra_feat_atom_reactant_" + i
                    pandas_file.at[ind, name] = -1
                    name = "extra_feat_atom_product_" + i
                    pandas_file.at[ind, name] = -1
                # iterate over all the features and set them to -1
                for i in features_bond:
                    name = "extra_feat_bond_reactant_" + i
                    pandas_file.at[ind, name] = -1
                    name = "extra_feat_bond_product_" + i
                    pandas_file.at[ind, name] = -1

                print(reaction_id)
                fail_count += 1

            for k, v in impute_count_reactants.items():
                impute_count_reactants[k] = v + tf_count_reactants[k]
            for k, v in impute_count_products.items():
                impute_count_products[k] = v + tf_count_products[k]
            for k, v in impute_count_reactants_bond.items():
                impute_count_reactants_bond[k] = v + tf_count_reactants_bond[k]
            for k, v in impute_count_products_bond.items():
                impute_count_products_bond[k] = v + tf_count_products_bond[k]

        # save the impute counts to a file
        with open(root + "impute_counts.txt", "w") as f:
            f.write("---------- impute_count_products ---------- \n")
            f.write(str(impute_count_products))
            f.write("\n")
            f.write("---------- impute_count_reactants ---------- \n")
            f.write(str(impute_count_reactants))
            f.write("\n")
            f.write("---------- impute_count_products_bond ---------- \n")
            f.write(str(impute_count_products_bond))
            f.write("\n")
            f.write("---------- impute_count_reactants_bond ---------- \n")
            f.write(str(impute_count_reactants_bond))
            f.write("\n")

        pandas_file["extra_feat_bond_reactant_indices_qtaim"] = bond_list_reactants
        pandas_file["extra_feat_bond_product_indices_qtaim"] = bond_list_products

        if update_bonds_w_qtaim:
            if "reactant_bonds" in pandas_file.columns:
                pandas_file["reactant_bonds_original"] = pandas_file["reactant_bonds"]
            if "product_bonds" in pandas_file.columns:
                pandas_file["product_bonds_original"] = pandas_file["product_bonds"]
            pandas_file["reactant_bonds"] = bond_list_reactants
            pandas_file["product_bonds"] = bond_list_products

    else:
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

            try:
                bonds = []
                id = row["ids"]
                if define_bonds == "distances":
                    bond_dict = row["bonds"]
                QTAIM_loc = root + "QTAIM/" + str(id) + "/"
                cp_file = QTAIM_loc + "CPprop.txt"
                dft_inp_file = QTAIM_loc + "input.in"

                qtaim_descs = get_qtaim_descs(cp_file, verbose=False)
                mapped_descs = merge_qtaim_inds(
                    qtaim_descs=qtaim_descs,
                    bond_list=bond_dict,
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
                                if (
                                    tf_count_bond[i] == 0
                                ):  # track if this feature is missing
                                    tf_count_bond[i] = 1

                                if impute:
                                    mapped_descs[k][i] = impute_dict["bond"][i][
                                        "median"
                                    ]
                                else:
                                    mapped_descs[k][i] = -1
                                    if ind not in drop_list:
                                        drop_list.append(ind)

                        else:
                            for i in features_atom:
                                if tf_count[i] == 0:  # track if this feature is missing
                                    tf_count[i] = 1

                                if impute:
                                    mapped_descs[k][i] = impute_dict["atom"][i][
                                        "median"
                                    ]
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

            except:
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

                fail_count += 1

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
