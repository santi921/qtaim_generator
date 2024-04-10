import pandas as pd
import os, argparse, stat, json, bson
from qtaim_gen.source.core.io import write_input_file_from_pmg_molecule


def convert_graph_info(site_info):
    return {
        "name": site_info[1]["specie"],
        "species": [{"element": site_info[1]["specie"], "occu": 1}],
        "xyz": site_info[1]["coords"],
        "properties": site_info[1]["properties"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-reaction", action="store_true")
    parser.add_argument("-parser",  type=str, default="Multiwfn")
    parser.add_argument("-file", type=str, default="20230512_mpreact_assoc.bson")
    parser.add_argument("-root", type=str, default="../data/rapter/")
    parser.add_argument("-options_qm_file", default="options_qm.json")

    args = parser.parse_args()

    reaction_tf = bool(args.reaction)
    file = args.file
    root = args.root
    parser = args.parser
    options_qm_file = args.options_qm_file

    print("root: {}".format(root))
    print("file: {}".format(file))
    print("reaction_tf: {}".format(reaction_tf))
    print("options_qm_file: {}".format(options_qm_file))
    # read json from options_qm_file
    with open(options_qm_file, "r") as f:
        options_qm = json.load(f)

    if parser == "Multiwfn":
        multi_wfn_cmd = options_qm["multiwfn_cmd"]
        multi_wfn_options_file = options_qm["multiwfn_options_file"]
    else: 
        critic2_cmd = options_qm["critic2_cmd"]
    
    file = root + file

    if reaction_tf:
        print("reading file from: {}".format(file))

        if file.endswith(".json"):
            path_json = file
            pandas_file = pd.read_json(path_json)
        else:
            path_bson = file
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

            if not os.path.exists(root + "QTAIM/" + str(reaction_id)):
                os.mkdir(root + "QTAIM/" + str(reaction_id))
            if not os.path.exists(QTAIM_loc_reactant):
                os.mkdir(QTAIM_loc_reactant)
            if not os.path.exists(QTAIM_loc_product):
                os.mkdir(QTAIM_loc_product)

            # check if QTAIM_loc_reactant + "/props.sh, QTAIM_loc_product + "/props.sh" exists and are not empty
            # if so, skip
            reactant_prop_tf = (
                os.path.exists(QTAIM_loc_reactant + "/props.sh")
                and os.path.getsize(QTAIM_loc_reactant + "/props.sh") > 0
            )
            product_prop_tf = (
                os.path.exists(QTAIM_loc_product + "/props.sh")
                and os.path.getsize(QTAIM_loc_product + "/props.sh") > 0
            )

            # create folder for each reaction + reactants + products

            if not os.path.exists(QTAIM_loc):
                os.mkdir(QTAIM_loc)
            if not os.path.exists(root + "QTAIM/" + str(reaction_id)):
                os.mkdir(root + "QTAIM/" + str(reaction_id))
            if not os.path.exists(QTAIM_loc_reactant):
                os.mkdir(QTAIM_loc_reactant)
            if not os.path.exists(QTAIM_loc_product):
                os.mkdir(QTAIM_loc_product)

            if not reactant_prop_tf and not product_prop_tf:
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
                        "{} ".format(multi_wfn_cmd)
                        + QTAIM_loc_reactant
                        + "/input.wfn < {} | tee ./".format(multi_wfn_options_file)
                        + QTAIM_loc_reactant
                        + "out \n"
                    )

                # run QTAIM on products
                with open(QTAIM_loc_product + "/props.sh", "w") as f:
                    f.write("#!/bin/bash\n")
                    f.write(
                        "{} ".format(multi_wfn_cmd)
                        + QTAIM_loc_product
                        + "/input.wfn < {} | tee ./".format(multi_wfn_options_file)
                        + QTAIM_loc_product
                        + "out \n"
                    )

                st = os.stat(QTAIM_loc_product + "/props.sh")
                os.chmod(QTAIM_loc_product + "/props.sh", st.st_mode | stat.S_IEXEC)

                st = os.stat(QTAIM_loc_reactant + "/props.sh")
                os.chmod(QTAIM_loc_reactant + "/props.sh", st.st_mode | stat.S_IEXEC)

            else:
                print("skipping reaction_id: {}".format(reaction_id))

    else:
        assert file.endswith(
            ".pkl"
        ), "molecules require a .pkl file, you can process it with the script: folder_xyz_molecules_to_pkl.py"
        # pandas_file = pd.read_json(path_json)
        pkl_df = pd.read_pickle(file)
        # print(pkl_df.keys())
        molecule_graphs = pkl_df["molecule_graph"]
        molecules = pkl_df["molecule"]
        molecule_ids = pkl_df["ids"]

        folder = root + "QTAIM/"
        if not os.path.exists(folder):
            os.mkdir(folder)

        for ind, row in pkl_df.iterrows():
            # if folder already exists, skip
            folder = root + "QTAIM/" + str(molecule_ids[ind]) 
            if not os.path.exists(folder):
                os.mkdir(folder)
            completed_tf = (
                os.path.exists(folder + "props.sh")
                and os.path.getsize(folder + "props.sh") > 0
            )
            if not completed_tf:
                # graph_info = molecule.graph.nodes(data=True)
                # sites = [convert_graph_info(item) for item in graph_info]
                molecule = row["molecule"]
                molecule_graph = row["molecule_graph"]
                # print(molecule.sites)
                # print(molecule.molecule)
                write_input_file_from_pmg_molecule(
                    folder=folder, molecule=molecule_graph.molecule, options=options_qm
                )

                multi_wfn_file = folder + "props.sh"
                with open(multi_wfn_file, "w") as f:
                    f.write("#!/bin/bash\n")
                    f.write(
                        "{} ".format(multi_wfn_cmd)
                        + folder
                        + "input.wfn < {} | tee ./".format(multi_wfn_options_file)
                        + folder
                        + "out \n"
                    )
                st = os.stat(multi_wfn_file)
                os.chmod(multi_wfn_file, st.st_mode | stat.S_IEXEC)


main()
