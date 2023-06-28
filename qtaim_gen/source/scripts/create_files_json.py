import pandas as pd
from glob import glob
import os, argparse, stat, json, bson

from qtaim_gen.source.core.io import (
    write_input_file,
    write_input_file_from_pmg_molecule,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-json_tf", type=bool, default=True)
    parser.add_argument("-json_file", type=str, default="20230512_mpreact_assoc.bson")
    parser.add_argument("-root", type=str, default="../data/rapter/")
    parser.add_argument("-options_qm_file", default="options_qm.json")
    args = parser.parse_args()

    json_tf = args.json_tf
    json_file = args.json_file
    root = args.root
    options_qm_file = args.options_qm_file
    # read json from options_qm_file
    with open(options_qm_file, "r") as f:
        options_qm = json.load(f)

    json_file = root + json_file

    if json_tf:
        print("reading file from: {}".format(json_file))
        if json_file.endswith(".json"):
            path_json = json_file
            pandas_file = pd.read_json(path_json)
        else:
            path_bson = json_file
            with open(path_bson, "rb") as f:
                data = bson.decode_all(f.read())
            pandas_file = pd.DataFrame(data)

        for ind, row in pandas_file.iterrows():
            # ind_random = random.choice(range(len(pandas_file)))

            # row = pandas_file.iloc[ind_random]
            reaction_id = row["reaction_id"]
            # print("reaction_id: {}".format(reaction_id))
            # print("bonds: {}".format(row["reactant_bonds"]))
            # create folder in json directory for each reaction
            QTAIM_loc = root + "QTAIM/"
            QTAIM_loc_reactant = root + "QTAIM/" + str(reaction_id) + "/reactants/"
            QTAIM_loc_product = root + "QTAIM/" + str(reaction_id) + "/products/"
            # create folder for each reaction + reactants + products

            if not os.path.exists(QTAIM_loc):
                os.mkdir(QTAIM_loc)
            if not os.path.exists(root + "QTAIM/" + str(reaction_id)):
                os.mkdir(root + "QTAIM/" + str(reaction_id))
            if not os.path.exists(QTAIM_loc_reactant):
                os.mkdir(QTAIM_loc_reactant)
            if not os.path.exists(QTAIM_loc_product):
                os.mkdir(QTAIM_loc_product)

            # reactants
            try:
                reactants = row["combined_reactants_graph"]
            except:
                reactants = row["reactant_molecule_graph"]

            # products
            write_input_file_from_pmg_molecule(
                folder=QTAIM_loc_reactant,
                molecule=reactants["molecule"],
                options=options_qm,
            )

            try:
                products = row["combined_products_graph"]
            except:
                products = row["product_molecule_graph"]

            write_input_file_from_pmg_molecule(
                folder=QTAIM_loc_product,
                molecule=products["molecule"],
                options=options_qm,
            )

            # run QTAIM on reactants
            with open(QTAIM_loc_reactant + "/props.sh", "w") as f:
                f.write("#!/bin/bash\n")
                f.write(
                    "./Multiwfn/Multiwfn "
                    + QTAIM_loc_reactant
                    + "/input.wfn < ./Multiwfn/data.txt | tee ./"
                    + QTAIM_loc_reactant
                    + "out \n"
                )

            # run QTAIM on products
            with open(QTAIM_loc_product + "/props.sh", "w") as f:
                f.write("#!/bin/bash\n")
                f.write(
                    "./Multiwfn/Multiwfn "
                    + QTAIM_loc_product
                    + "/input.wfn < ./Multiwfn/data.txt | tee ./"
                    + QTAIM_loc_product
                    + "out \n"
                )

            st = os.stat(QTAIM_loc_product + "/props.sh")
            os.chmod(QTAIM_loc_product + "/props.sh", st.st_mode | stat.S_IEXEC)

            st = os.stat(QTAIM_loc_reactant + "/props.sh")
            os.chmod(QTAIM_loc_reactant + "/props.sh", st.st_mode | stat.S_IEXEC)

    else:
        dir_source = "./"
        files = glob(
            dir_source + "*xyz"
        )  # xyz file names, would need to change for pandas
        for i in files:
            folder = i.split("_")[1].split(".")[0]
            try:
                os.mkdir(folder)
            except:
                pass

            with open(i) as xyz:
                lines = xyz.readlines()

            atoms = int(lines[0])

            write_input_file(options_qm_file, folder)

            with open(folder + "/props.sh", "w") as f:
                f.write("#!/bin/bash\n")
                f.write(
                    "./Multiwfn/Multiwfn "
                    + folder
                    + "/input.wfn < ./Multiwfn/data.txt | tee ./"
                    + folder
                    + "/out \n"
                )


main()
