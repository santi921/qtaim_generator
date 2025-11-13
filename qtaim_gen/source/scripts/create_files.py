#!/usr/bin/env python3

import pandas as pd
import os, argparse, stat, json, bson
from qtaim_gen.source.utils.io import write_input_file_from_pmg_molecule
from pathlib import Path


def convert_graph_info(site_info):
    return {
        "name": site_info[1]["specie"],
        "species": [{"element": site_info[1]["specie"], "occu": 1}],
        "xyz": site_info[1]["coords"],
        "properties": site_info[1]["properties"],
    }


def main(argv=None):
    """
    Entry point for create_files script.

    Args:
        argv (list[str] | None): Optional list of command-line arguments (excludes program name).
            If None, arguments are read from sys.argv.

    Returns:
        int: exit code (0 on success).
    """
    argp = argparse.ArgumentParser()
    argp.add_argument("-reaction", action="store_true")
    argp.add_argument("-parser", type=str, default="Multiwfn")
    argp.add_argument("-file", type=str, default="20230512_mpreact_assoc.bson")
    argp.add_argument("-root", type=str, default="../data/rapter/")
    argp.add_argument("-options_qm_file", default="options_qm.json")
    argp.add_argument("--molden_sub", action="store_true", help="molden subroutine")

    args = argp.parse_args(argv)

    reaction_tf = bool(args.reaction)
    molden_tf = bool(args.molden_sub)
    file = args.file
    root = args.root
    parser = args.parser
    options_qm_file = args.options_qm_file

    print("root: {}".format(root))
    print("file: {}".format(file))
    print("reaction_tf: {}".format(reaction_tf))
    print("options_qm_file: {}".format(options_qm_file))
    print("molden tf: {}".format(molden_tf))

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
            QTAIM_loc_reactant = str(
                Path.home().joinpath(QTAIM_loc, str(reaction_id), "reactants")
            )
            QTAIM_loc_product = str(
                Path.home().joinpath(QTAIM_loc, str(reaction_id), "products")
            )
            # QTAIM_loc_reactant = root + "QTAIM/" + str(reaction_id) + "/reactants/"
            # QTAIM_loc_product = root + "QTAIM/" + str(reaction_id) + "/products/"

            if not os.path.exists(root + "QTAIM/" + str(reaction_id)):
                # os.mkdir(root + "QTAIM/" + str(reaction_id))
                os.mkdir(str(Path.home().joinpath(QTAIM_loc, str(reaction_id))))
            if not os.path.exists(QTAIM_loc_reactant):
                os.mkdir(QTAIM_loc_reactant)
            if not os.path.exists(QTAIM_loc_product):
                os.mkdir(QTAIM_loc_product)

            # check if QTAIM_loc_reactant + "/props.sh, QTAIM_loc_product + "/props.sh" exists and are not empty
            # if so, skip

            # reactant path prop
            reactant_prop = str(Path.home().joinpath(QTAIM_loc_reactant, "props.sh"))
            # product path prop
            product_prop = str(Path.home().joinpath(QTAIM_loc_product, "props.sh"))
            # reaction root
            reaction_root = str(Path.home().joinpath(QTAIM_loc, str(reaction_id)))

            reactant_prop_tf = (
                os.path.exists(reactant_prop) and os.path.getsize(reactant_prop) > 0
            )
            product_prop_tf = (
                os.path.exists(product_prop) and os.path.getsize(product_prop) > 0
            )

            # create folder for each reaction + reactants + products

            if not os.path.exists(QTAIM_loc):
                os.mkdir(QTAIM_loc)
            if not os.path.exists(reaction_root):
                os.mkdir(reaction_root)
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
                if molden_tf:
                    input_file_name = "input.molden.input"
                else:
                    input_file_name = "input.wfn"

                with open(reactant_prop, "w") as f:
                    f.write("#!/bin/bash\n")

                    if molden_tf:
                        f.write(
                            "orca_2mkl "
                            + str(Path.home().joinpath(folder, "input -molden"))
                            + "\n"
                        )

                    f.write(
                        "{} ".format(multi_wfn_cmd)
                        + str(
                            Path.home().joinpath(
                                QTAIM_loc_reactant, str(reaction_id), input_file_name
                            )
                        )
                        + " < {} | tee ".format(multi_wfn_options_file)
                        # + QTAIM_loc_reactant
                        # + "out \n"
                        + str(Path.home().joinpath(QTAIM_loc_reactant, "out"))
                        + "\n"
                    )

                # run QTAIM on products
                with open(product_prop, "w") as f:
                    f.write("#!/bin/bash\n")

                    if molden_tf:
                        f.write(
                            "orca_2mkl "
                            + str(Path.home().joinpath(folder, "input -molden"))
                            + "\n"
                        )

                    f.write(
                        "{} ".format(multi_wfn_cmd)
                        + str(
                            Path.home().joinpath(
                                QTAIM_loc_product, str(reaction_id), input_file_name
                            )
                        )
                        + " < {} | tee ".format(multi_wfn_options_file)
                        # + QTAIM_loc_product
                        # + "out \n"
                        + str(Path.home().joinpath(QTAIM_loc_product, "out"))
                        + "\n"
                    )

                st = os.stat(product_prop)
                os.chmod(product_prop, st.st_mode | stat.S_IEXEC)

                st = os.stat(reactant_prop)
                os.chmod(reactant_prop, st.st_mode | stat.S_IEXEC)

            else:
                print("skipping reaction_id: {}".format(reaction_id))

    else:
        assert file.endswith(
            ".pkl"
        ), "molecules require a .pkl file, you can process it with the script: folder_xyz_molecules_to_pkl.py"
        # pandas_file = pd.read_json(path_json)
        pkl_df = pd.read_pickle(file)
        # print(pkl_df.keys())
        # molecule_graphs = pkl_df["molecule_graph"]
        # molecules = pkl_df["molecule"]
        molecule_ids = pkl_df["ids"]

        folder = root + "QTAIM/"
        if not os.path.exists(folder):
            os.mkdir(folder)

        for ind, row in pkl_df.iterrows():
            # if folder already exists, skip

            folder = str(Path.home().joinpath(root, "QTAIM", str(molecule_ids[ind])))
            props_file = str(Path.home().joinpath(folder, "props.sh"))
            if not os.path.exists(folder):
                os.mkdir(folder)
            completed_tf = (
                os.path.exists(props_file) and os.path.getsize(props_file) > 0
            )
            if not completed_tf:
                # graph_info = molecule.graph.nodes(data=True)
                # sites = [convert_graph_info(item) for item in graph_info]
                # molecule = row["molecule"]
                molecule_graph = row["molecule_graph"]
                # print(molecule.sites)
                # print(molecule.molecule)
                write_input_file_from_pmg_molecule(
                    folder=folder, molecule=molecule_graph.molecule, options=options_qm
                )

                if molden_tf:
                    input_file_name = "input.molden.input"
                else:
                    input_file_name = "input.wfn"

                # multi_wfn_file = folder + "/props.sh"
                with open(props_file, "w") as f:
                    f.write("#!/bin/bash\n")
                    if molden_tf:
                        f.write(
                            "orca_2mkl "
                            + str(Path.home().joinpath(folder, "input -molden"))
                            + "\n"
                        )

                    f.write(
                        "{} ".format(multi_wfn_cmd)
                        + str(Path.home().joinpath(folder, input_file_name))
                        # + folder
                        # + "input.wfn < {} | tee ./".format(multi_wfn_options_file)
                        + " < {} | tee ".format(multi_wfn_options_file)
                        + str(Path.home().joinpath(folder, "out"))
                        # + folder
                        + "\n"
                    )

                st = os.stat(props_file)
                os.chmod(props_file, st.st_mode | stat.S_IEXEC)

    # Completed successfully
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
