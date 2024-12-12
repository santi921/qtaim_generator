import os
from glob import glob
import numpy as np
import json


def get_keys_qtaim(qtaim_dict):
    """
    Get keys from qtaim dictionary to impute missing values.
    Takes:
        qtaim_dict: dict
    Returns:
        dict_keys: list of keys
    """
    ignore_list = ["cp_num", "element", "pos_ang", "number", "connected_bond_paths"]

    bonds_tf, atoms_tf = False, False
    dict_keys = []
    for k, v in qtaim_dict.items():
        # bond keys
        if "_" in k:
            if not bonds_tf:
                for i in v.keys():
                    if i not in ignore_list:
                        dict_keys.append("extra_feat_bond_{}".format(i))
                bonds_tf = True
        else:
            if not atoms_tf:
                for i in v.keys():
                    if i not in ignore_list:
                        dict_keys.append("extra_feat_atom_{}".format(i))
                atoms_tf = True
    return dict_keys


def get_qtaim_data_impute(qtaim_dict, key_list):
    """
    Get data from qtaim dictionary to impute missing values.
    Takes:
        qtaim_dict: dict
        key_list: list of keys
    Returns:
        dict_res: dict
    """

    ignore_list = ["cp_num", "element", "pos_ang", "number", "connected_bond_paths"]

    dict_res = {}
    for k in key_list:
        dict_res[k] = []

    for k, v in qtaim_dict.items():
        # bond keys
        if "_" in k:
            for i in v.keys():
                if i not in ignore_list:
                    key = "extra_feat_bond_{}".format(i)
                    dict_res[key].append(v[i])

        else:
            for i in v.keys():
                if i not in ignore_list:
                    key = "extra_feat_atom_{}".format(i)
                    dict_res[key].append(v[i])
    return dict_res


def get_qtaim_data(qtaim_dict):
    """
    Get data from qtaim dictionary to impute missing values.
    Takes:
        qtaim_dict: dict
        key_list: list of keys
    Returns:
        dict_res: dict
    """

    ignore_list = ["cp_num", "element", "pos_ang", "number", "connected_bond_paths"]

    dict_res = {}
    bond_list = []
    # for k in key_list: dict_res[k] = []

    for k, v in qtaim_dict.items():
        # bond keys
        if "_" in k:
            bond_list.append(tuple([int(i) for i in k.split("_")]))
            for i in v.keys():
                if i not in ignore_list:
                    key = "extra_feat_bond_{}".format(i)
                    if key not in dict_res.keys():
                        dict_res[key] = []
                    dict_res[key].append(v[i])

        else:
            for i in v.keys():
                if i not in ignore_list:
                    key = "extra_feat_atom_{}".format(i)
                    if key not in dict_res.keys():
                        dict_res[key] = []
                    dict_res[key].append(v[i])

    dict_res["bond_list_qtaim"] = bond_list
    # print(dict_res)
    return dict_res


def gather_impute(root, full_descriptors=False):
    """
    Gather data from json files and impute missing values.
    Takes:
        root: str
        full_descriptors: bool
        orca: bool
    Returns:
        impute_dict: dict
    """

    if full_descriptors:
        json_list = ["qtaim", "charge", "other"]
    else:
        json_list = ["qtaim"]

    impute_dict = {}
    impute_init = False

    subfolders = [f.path for f in os.scandir(root) if f.is_dir()]
    for i in subfolders:
        for j in json_list:
            if not os.path.exists(f"{i}/{j}.json"):
                print(f"Missing {j}.json in {i}")

            else:
                if j == "qtaim":
                    with open(f"{i}/{j}.json", "r") as f:
                        qtaim_dict = json.load(f)

                    if impute_init == False:
                        impute_keys_qtaim = get_keys_qtaim(qtaim_dict)
                        for k in impute_keys_qtaim:
                            impute_dict[k] = []
                        impute_init = True

                    impute_data = get_qtaim_data_impute(qtaim_dict, impute_keys_qtaim)

                    for k, v in impute_data.items():
                        impute_dict[k].extend(v)

                if j == "charge":
                    with open(f"{i}/{j}.json", "r") as f:
                        charge_dict = json.load(f)
                        for k, v in charge_dict.items():
                            for k1, v1 in v.items():
                                if k1 == "charge":
                                    key = "{}_{}".format(k, k1)
                                    if key not in impute_dict.keys():
                                        impute_dict[key] = []
                                    impute_dict[key].extend(list(v1.values()))

                if j == "other":
                    with open(f"{i}/{j}.json", "r") as f:
                        other_dict = json.load(f)
                        for k, v in other_dict.items():
                            if k not in impute_dict.keys():
                                impute_dict[k] = []
                            impute_dict[k].append(v)

    for k, v in impute_dict.items():
        impute_dict[k] = {
            "mean": np.mean(np.array(v)),
            "median": np.median(np.array(v)),
        }

    return impute_dict


def parse_bond(bond_dict, cutoff=0.05):
    """
    Parse bond dictionary
    Takes:
        bond_dict: dict
    Returns:
        bond_dict: dict but values are bond list
    """
    bond_results = {}
    bond_results["ibsi_data"] = {}
    bond_results["fuzzy_data"] = {}

    def clean_key(key):
        return tuple([int(i.split("_")[0]) - 1 for i in key.split("to_")])

    for k, v in bond_dict["ibsi"].items():
        if v > cutoff:
            bond_results["ibsi_data"][clean_key(k)] = v

    for k, v in bond_dict["fuzzy"].items():
        if v > cutoff:
            bond_results["fuzzy_data"][clean_key(k)] = v

    bond_results["bond_list_ibsi"] = list(bond_results["ibsi_data"].keys())
    bond_results["bond_list_fuzzy"] = list(bond_results["fuzzy_data"].keys())
    return bond_results


def gather_dicts(json_list, i):
    """
    Gather data from json files and impute missing values.
    Takes:
        json_list: list of str
        i: folder with json files
    Returns:
        ret_dict: dict
    """

    ret_dict = {}

    for j in json_list:
        # try:
        if not os.path.exists(f"{i}/{j}.json"):
            print(f"Missing {j}.json in {i}")
            ret_dict[json_list] = {}

        else:
            if j == "qtaim":
                with open(f"{i}/{j}.json", "r") as f:
                    qtaim_dict = json.load(f)
                qtaim_dict = get_qtaim_data(qtaim_dict)
                ret_dict["qtaim"] = qtaim_dict

            if j == "bond":
                with open(f"{i}/{j}.json", "r") as f:
                    bond_dict = json.load(f)
                bond_dict = parse_bond(bond_dict)
                ret_dict["bond"] = bond_dict

            if j == "charge":
                with open(f"{i}/{j}.json", "r") as f:
                    charge_dict = json.load(f)

                charge_dict_ret = {}

                for k, v in charge_dict.items():
                    for k1, v1 in v.items():

                        if k1 == "charge":
                            key = "{}_{}".format(k, k1)
                            if key not in charge_dict_ret.keys():
                                charge_dict_ret[key] = []
                            charge_dict_ret[key].extend(list(v1.values()))

                        elif k1 == "dipole":
                            key_xyz = "{}_{}".format(k, k1)
                            key_mag = "{}_{}_mag".format(k, k1)
                            if key_xyz not in charge_dict_ret.keys():
                                charge_dict_ret[key_xyz] = []
                                charge_dict_ret[key_mag] = []

                            charge_dict_ret[key_xyz].extend(list(v1["xyz"]))
                            # print(v1["mag"])
                            charge_dict_ret[key_mag].append(v1["mag"])

                        elif k1 == "spin":
                            key = "{}_{}".format(k, k1)
                            if key not in charge_dict_ret.keys():
                                charge_dict_ret[key] = []
                            charge_dict_ret[key].extend(list(v1.values()))

                        elif k1 == "atomic_dipole":
                            key = "{}_{}".format(k, k1)
                            if key not in charge_dict_ret.keys():
                                charge_dict_ret[key] = []
                            charge_dict_ret[key].extend(list(v1.values()))

                ret_dict["charge"] = charge_dict_ret

            if j == "other":
                with open(f"{i}/{j}.json", "r") as f:
                    other_dict = json.load(f)
                ret_dict["other"] = other_dict

    return ret_dict


def get_data(root, full_descriptors=False, root_folder_tf=True):
    """
    Gather data
    Takes:
        root: str
        full_descriptors: bool
        root_folder: bool
    Returns:
        ret_dict: dict
    """

    if full_descriptors:
        json_list = ["qtaim", "charge", "bond", "other"]
    else:
        json_list = ["qtaim"]

    ret_dict_full = {}

    if root_folder_tf:
        subfolders = [f.path for f in os.scandir(root) if f.is_dir()]
        for i in subfolders:

            json_dict = gather_dicts(json_list, i)
            ret_dict = {}

            for k, v in json_dict["qtaim"].items():
                ret_dict[k] = v

            if full_descriptors:
                for k, v in json_dict["charge"].items():
                    if "mag" in k:
                        ret_dict[k] = v[0]
                    else:
                        ret_dict[k] = v
                for k, v in json_dict["bond"].items():
                    ret_dict[k] = v
                for k, v in json_dict["other"].items():
                    ret_dict[k] = v

            for k, v in ret_dict.items():
                if k not in ret_dict_full.keys():
                    ret_dict_full[k] = []
                ret_dict_full[k].append(v)

        return ret_dict_full

    else:

        # follow logic but for single folder
        ret_dict = {}
        json_dict = gather_dicts(json_list, root)
        for k, v in json_dict["qtaim"].items():
            ret_dict[k] = v

        if full_descriptors:
            for k, v in json_dict["charge"].items():
                if "mag" in k:
                    ret_dict[k] = v[0]
                else:
                    ret_dict[k] = v
            for k, v in json_dict["bond"].items():
                ret_dict[k] = v
            for k, v in json_dict["other"].items():
                ret_dict[k] = v

        for k, v in ret_dict.items():
            if k not in ret_dict_full.keys():
                ret_dict_full[k] = []
            ret_dict_full[k] = v

        return ret_dict_full
