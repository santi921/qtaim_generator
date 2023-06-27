import os, bson, argparse
from glob import glob
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json_file",
        type=str,
        default="20230512_mpreact_assoc.bson",
        help="json file to read from",
    )
    parser.add_argument(
        "--json_loc",
        type=str,
        default="../data/rapter/",
        help="",
    )

    args = parser.parse_args()
    json_file = args.json_file
    json_loc = args.json_loc

    print("reading file from: {}".format(json_file))
    if json_file.endswith(".json"):
        path_json = json_file
        pandas_file = pd.read_json(path_json)
    else:
        path_bson = json_file
        with open(path_bson, "rb") as f:
            data = bson.decode_all(f.read())
        pandas_file = pd.DataFrame(data)

    product_wfn_count = 0
    reactant_wfn_count = 0
    product_out_count = 0
    reactant_out_count = 0

    for ind, _ in pandas_file.iterrows():
        row = pandas_file.iloc[ind]
        reaction_id = row["reaction_id"]

        # create folder in json directory for each reaction
        QTAIM_loc = json_loc + "QTAIM/"
        QTAIM_loc_reactant = json_loc + "QTAIM/" + str(reaction_id) + "/reactants/"
        QTAIM_loc_product = json_loc + "QTAIM/" + str(reaction_id) + "/products/"

        if os.path.exists(QTAIM_loc_reactant + "input.wfn"):
            if os.path.getsize(QTAIM_loc_reactant + "input.wfn") > 0:
                reactant_wfn_count += 1
        if os.path.exists(QTAIM_loc_reactant + "CPprop.txt"):
            if os.path.getsize(QTAIM_loc_reactant + "CPprop.txt") > 0:
                reactant_out_count += 1

        if os.path.exists(QTAIM_loc_product + "input.wfn"):
            if os.path.getsize(QTAIM_loc_product + "input.wfn") > 0:
                product_wfn_count += 1
        if os.path.exists(QTAIM_loc_product + "CPprop.txt"):
            if os.path.getsize(QTAIM_loc_product + "CPprop.txt") > 0:
                product_out_count += 1

    print("Product wfn count: {}".format(product_wfn_count))
    print("Reactant wfn count: {}".format(reactant_wfn_count))
    print("Product out count: {}".format(product_out_count))
    print("Reactant out count: {}".format(reactant_out_count))
    print("total rows: {}".format(len(pandas_file)))


main()
