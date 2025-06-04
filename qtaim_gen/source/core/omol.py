from pathlib import Path
import os, stat, json, time

from qtaim_gen.source.data.multiwfn import (
    charge_data,
    charge_data_dict,
    bond_order_data,
    bond_order_dict,
    fuzzy_data,
    other_data,
    qtaim_data,
)

from qtaim_gen.source.core.parse_multiwfn import (
    parse_charge_doc,
    parse_charge_base,
    parse_charge_becke,
    parse_charge_doc_adch,
    parse_bond_order_doc,
    parse_bond_order_fuzzy,
    parse_bond_order_ibsi,
    parse_bond_order_laplace,
    parse_fuzzy_doc,
    parse_other_doc,
    parse_qtaim,
    parse_charge_doc_bader,
    parse_charge_chelpg,
    parse_fuzzy_real_space
)

from qtaim_gen.source.core.utils import pull_ecp_dict, overwrite_molden_w_ecp

ORDER_OF_OPERATIONS = ["fuzzy_full", "qtaim", "bond", "charge", "other"]
ORDER_OF_OPERATIONS_separate = ["fuzzy_full", "charge_separate", "bond_separate", "qtaim", "other"]
#ORDER_OF_OPERATIONS_separate = ["fuzzy_full"]
#ORDER_OF_OPERATIONS_separate = ["fuzzy_full", "charge_separate"]



def write_conversion(
    out_folder, read_file, overwrite=False, name="convert.in", orca_2mkl_cmd="orca_2mkl"
):
    """
    Function to write a bash script that runs multiwfn on a given input file.
    Args:
        out_folder(str): folder to write the bash script to
        multi_wfn_cmd(str): command to run multiwfn
        multiwfn_input_file(str): input file for multiwfn
        overwrite(bool): whether to overwrite the file if it already exists
        name(str): name of the bash script
    """

    out_file = str(Path.home().joinpath(out_folder, name))
    # print("out_file: {}".format(out_file))

    if not os.path.exists(out_folder):
        os.mkdir(out_folder)

    completed_tf = os.path.exists(out_file) and os.path.getsize(out_file) > 0

    if completed_tf and not overwrite:
        print("file exists and overwrite is true")
        return

    with open(out_file, "w") as f:
        f.write("#!/bin/bash\n")
        # remove .gbw from file name
        read_file = read_file.split(".gbw")[0]
        f.write(
            "{} ".format(orca_2mkl_cmd)
            + str(Path.home().joinpath(out_folder, read_file))
            + " -molden\n"
        )

    st = os.stat(out_file)
    os.chmod(out_file, st.st_mode | stat.S_IEXEC)


def write_multiwfn_exe(
    out_folder,
    read_file,
    multi_wfn_cmd,
    multiwfn_input_file,
    convert_gbw=False,
    overwrite=False,
    mv_cpprop=False,
    name="props.mfwn",
):
    """
    Function to write a bash script that runs multiwfn on a given input file.
    Args:
        out_folder(str): folder to write the bash script to
        read_file(str): file to read from
        multi_wfn_cmd(str): command to run multiwfn
        multiwfn_input_file(str): input file for multiwfn
        convert_gbw(bool): whether to convert the input file to a gbw file
        mv_cpprop(bool): whether to move the cpprop file to the output folder
        overwrite(bool): whether to overwrite the file if it already exists
        name(str): name of the bash script
    """

    out_file = str(Path.home().joinpath(out_folder, name))
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)

    completed_tf = os.path.exists(out_file) and os.path.getsize(out_file) > 0

    if (not completed_tf) or overwrite:
        with open(out_file, "w") as f:
            f.write("#!/bin/bash\n")
            if convert_gbw:
                f.write("orca_2mkl " + str(Path.home().joinpath(out_folder)) + "\n")

            multiwfn_input_file_root = multiwfn_input_file.split("/")[-1].split(".")[0]

            f.write(
                "{} ".format(multi_wfn_cmd)  # multiwfn command
                + str(Path.home().joinpath(out_folder, read_file))  # wfn/gbw file
                + " < {} | tee ".format(multiwfn_input_file)  # multiwfn input file
                + str(
                    Path.home().joinpath(
                        out_folder, "{}.out".format(multiwfn_input_file_root)
                    )
                )  # output file
                + "\n"
            )

            if mv_cpprop:
                f.write(
                    "mv CPprop.txt "
                    + str(Path.home().joinpath(out_folder, "CPprop.txt"))
                    + "\n"
                )

        st = os.stat(out_file)
        os.chmod(out_file, st.st_mode | stat.S_IEXEC)


def create_jobs(folder, multiwfn_cmd, orca_2mkl_cmd, separate=False, debug=True):
    """
    Create job files for multiwfn analysis
    Takes:
        folder(str): folder to create jobs in
        multiwfn_cmd(str): command to run multiwfn
        orca_2mkl_cmd(str): command to run orca_2mklß
        separate(bool): whether to separate the analysis into different files
        overwrite(bool): whether to overwrite the output files
        orca_6(bool): whether calc is from orca6
    """
    if separate:
        routine_list = ORDER_OF_OPERATIONS_separate
        # routine_list = ["charge_separate"]
    else:
        routine_list = ORDER_OF_OPERATIONS
        # routine_list = ["qtaim", "bond", "charge"]

    if debug:  # just run qtaim in debug mode
        routine_list = ["qtaim"]

    wfn_present = False
    for file in os.listdir(folder):
        # print(file)
        if file.endswith(".wfn"):
            wfn_present = True
            file_read = os.path.join(folder, file)

        if file.endswith(".gbw"):

            bool_gbw = True
            file_gbw = os.path.join(folder, file)
            file_wfn = file.replace(".gbw", ".wfn")
            file_molden = file.replace(".gbw", ".molden.input")
            file_read = os.path.join(folder, file_wfn)
            file_molden = os.path.join(folder, file_molden)

    if not wfn_present and bool_gbw:
        print("file_gbw: {}".format(file_gbw))
        print("out folder: {}".format(folder))
        # write conversion script from gbw to wfn
        print("wfn not there - writing conversion script")
        write_conversion(
            out_folder=folder,
            read_file=file_gbw,
            overwrite=True,
            name="convert.in",
            orca_2mkl_cmd=orca_2mkl_cmd,
        )
        routine_list = ["convert"] + routine_list

    job_dict = {}

    # print("folder: {}".format(folder))
    for routine in routine_list:
        if routine == "qtaim":
            job_dict["qtaim"] = os.path.join(folder, "qtaim.txt")
            # write qtaim data file
            with open(os.path.join(folder, "qtaim.txt"), "w") as f:
                data = qtaim_data()
                f.write(data)

        elif routine == "fuzzy_full":
            #job_dict["fuzzy_full"] = os.path.join(folder, "fuzzy_full.txt")
            with open(os.path.join(folder, "fuzzy_full.txt"), "w") as f:
                fuzzy_dict = fuzzy_data()
                for key, value in fuzzy_dict.items():
                    job_dict[key] = os.path.join(folder, "{}.txt".format(key))
                    with open(os.path.join(folder, "{}.txt".format(key)), "w") as f:
                        f.write(value)
                #f.write(data)

        elif routine == "bond":
            job_dict["bond"] = os.path.join(folder, "bond.txt")
            with open(os.path.join(folder, "bond.txt"), "w") as f:
                data = bond_order_data()
                f.write(data)

        elif routine == "bond_separate":
            bond_dict = bond_order_dict()
            for key, value in bond_dict.items():
                job_dict[key] = os.path.join(folder, "{}.txt".format(key))
                with open(os.path.join(folder, "{}.txt".format(key)), "w") as f:
                    f.write(value)

        elif routine == "charge":
            job_dict["charge"] = os.path.join(folder, "charge.txt")
            with open(os.path.join(folder, "charge.txt"), "w") as f:
                data = charge_data()
                f.write(data)

        elif routine == "charge_separate":
            charge_dict = charge_data_dict()
            for key, value in charge_dict.items():
                job_dict[key] = os.path.join(folder, "{}.txt".format(key))
                with open(os.path.join(folder, "{}.txt".format(key)), "w") as f:
                    f.write(value)

        elif routine == "other":
            job_dict["other"] = os.path.join(folder, "other.txt")
            with open(os.path.join(folder, "other.txt"), "w") as f:
                data = other_data()
                f.write(data)

        elif routine == "convert":
            job_dict["convert"] = os.path.join(folder, "convert.txt")
            with open(os.path.join(folder, "convert.txt"), "w") as f:
                file_wfn = file_gbw.replace(".gbw", ".wfn")
                print("file_wfn: {}".format(file_wfn))
                data = "100\n2\n5\n{}\n0\nq\n".format(file_wfn)
                f.write(data)

        else:
            print("routine not recognized")

    # print(job_dict)
    for key, value in job_dict.items():
        # print(key, value)
        if key == "qtaim":
            mv_cpprop = True
        else:
            mv_cpprop = False

        if key == "charge_separate":
            charge_dict = charge_data_dict()
            for key, value in charge_dict.items():
                write_multiwfn_exe(
                    out_folder=folder,
                    read_file=file_read,
                    multi_wfn_cmd=multiwfn_cmd,
                    multiwfn_input_file=value,
                    convert_gbw=False,
                    overwrite=True,
                    name="props_{}.mfwn".format(key),
                    mv_cpprop=mv_cpprop,
                )

        elif key == "bond_separate":
            bond_dict = bond_order_dict()
            for key, value in bond_dict.items():
                write_multiwfn_exe(
                    out_folder=folder,
                    read_file=file_read,
                    multi_wfn_cmd=multiwfn_cmd,
                    multiwfn_input_file=value,
                    convert_gbw=False,
                    overwrite=True,
                    name="props_{}.mfwn".format(key),
                    mv_cpprop=mv_cpprop,
                )
        elif key == "convert":
            write_multiwfn_exe(
                out_folder=folder,
                read_file=file_molden,
                multi_wfn_cmd=multiwfn_cmd,
                multiwfn_input_file=value,
                convert_gbw=False,
                overwrite=True,
                name="props_{}.mfwn".format(key),
                mv_cpprop=mv_cpprop,
            )

        else:
            # print("key: {}".format(key))
            write_multiwfn_exe(
                out_folder=folder,
                read_file=file_read,
                multi_wfn_cmd=multiwfn_cmd,
                multiwfn_input_file=value,
                convert_gbw=False,
                overwrite=True,
                name="props_{}.mfwn".format(key),
                mv_cpprop=mv_cpprop,
            )


def run_jobs(folder, separate=False, orca_6=True, restart=False, debug=False):
    """
    Run conversion and multiwfn jobs
    Takes:
        folder(str): folder to run jobs in
        separate(bool): whether to separate the analysis into different files
    """
    if separate:
        #order_of_operations = ORDER_OF_OPERATIONS_separate
        order_of_operations = ["qtaim"]
        charge_dict = charge_data_dict()
        [order_of_operations.append(i) for i in charge_dict.keys()]
        bond_dict = bond_order_dict()
        [order_of_operations.append(i) for i in bond_dict.keys()]
        fuzzy_dict = fuzzy_data()
        [order_of_operations.append(i) for i in fuzzy_dict.keys()]

    else:
        order_of_operations = ORDER_OF_OPERATIONS
        # order_of_operations = ["bond", "charge", "qtaim"]

    if debug:
        order_of_operations = ["qtaim"]

    wfn_present = False
    for file in os.listdir(folder):
        if file.endswith(".wfn"):
            wfn_present = True
        if file.endswith("convert.in"):
            conv_file = os.path.join(folder, file)

    # run conversion script if wfn file is not present
    if not wfn_present:
        print("running conversion script")
        # run conversion script
        os.system("{}".format(conv_file))

        for file in os.listdir(folder):
            if file.endswith(".molden.input"):
                molden_file = os.path.join(folder, file)
            if file.endswith("orca.out") or file.endswith("output.out"):
                orca_out = os.path.join(folder, file)

        if not orca_6:
            # replace ecp values in converted molden file
            dict_ecp = pull_ecp_dict(orca_out)
            overwrite_molden_w_ecp(molden_file, dict_ecp)

        order_of_operations = ["convert"] + order_of_operations
        # conversion script should read in .molden file and use multiwfn to convert to .wfn

    # create a json file to store job status

    timings = {}
    # run multiwfn scripts
    for order in order_of_operations:
        # if restart, check if timing file exists
        if restart:
            if os.path.exists(os.path.join(folder, "timings.json")):
                with open(os.path.join(folder, "timings.json"), "r") as f:
                    timings = json.load(f)
                    if order in timings.keys():
                        continue

        mfwn_file = os.path.join(folder, "props_{}.mfwn".format(order))

        try:
            start = time.time()
            os.system("{}".format(mfwn_file))
            end = time.time()
            timings[order] = end - start

        except:
            print("error running {}".format(mfwn_file))
            timings[order] = -1

        # save timings to file in folder - at each step for check pointing
        with open(os.path.join(folder, "timings.json"), "w") as f:
            json.dump(timings, f, indent=4)


def parse_multiwfn(folder, separate=False, debug=False):
    """
    Parse multiwfn output files to jsons and save them in folder
    Takes:
        folder(str): folder to parse
        separate(bool): whether to separate the analysis into different files
    """

    if separate:
        routine_list = ORDER_OF_OPERATIONS_separate
        # routine_list = ["qtaim"]
        charge_dict = charge_data_dict()
        bond_dict = bond_order_dict()
        fuzzy_dict = fuzzy_data()

        [routine_list.append(i) for i in charge_dict.keys()]
        [routine_list.append(i) for i in bond_dict.keys()]
        [routine_list.append(i) for i in fuzzy_dict.keys()]

    else:
        routine_list = ORDER_OF_OPERATIONS
        # routine_list = ["bond", "charge", "qtaim"]

    if debug:
        routine_list = ["qtaim"]

    for file in os.listdir(folder):
        if file.endswith(".out"):
            file_full_path = os.path.join(folder, file)
            for routine in routine_list:
                if routine in file:
                    # print("routine: ", routine)
                    json_file = file_full_path.replace(".out", ".json")

                    if routine == "fuzzy_full":
                        """
                        print("parsing fuzzy_full")
                        data = parse_fuzzy_doc(file_full_path)
                        with open(json_file, "w") as f:
                            json.dump(data, f, indent=4)
                        """
                        try:
                            data = parse_fuzzy_doc(file_full_path)
                            with open(json_file, "w") as f:
                                json.dump(data, f, indent=4)
                        except:
                            print("error parsing fuzzy_full")

                    elif routine == "bond":
                        try:
                            data = parse_bond_order_doc(file_full_path)
                            with open(json_file, "w") as f:
                                json.dump(data, f, indent=4)
                        except:
                            print("error parsing bond")

                    elif routine == "ibsi":
                        try:
                            data = parse_bond_order_ibsi(file_full_path)
                            with open(json_file, "w") as f:
                                json.dump(data, f, indent=4)
                        except:
                            print("error parsing ibsi")

                    elif routine == "laplace":
                        try:
                            data = parse_bond_order_laplace(file_full_path)
                            with open(json_file, "w") as f:
                                json.dump(data, f, indent=4)
                        except:
                            print("error parsing laplace")

                    elif routine == "fuzzy":
                        try:
                            data = parse_bond_order_fuzzy(file_full_path)
                            with open(json_file, "w") as f:
                                json.dump(data, f, indent=4)
                        except:
                            print("error parsing fuzzy")

                    elif routine == "other":
                        try:
                            data = parse_other_doc(file_full_path)
                            with open(json_file, "w") as f:
                                json.dump(data, f, indent=4)
                        except:
                            print("error parsing other")

                    elif routine == "charge":
                        try:
                            (
                                charge_dict_overall,
                                atomic_dipole_dict_overall,
                                dipole_info,
                            ) = parse_charge_doc(file_full_path)
                            charge_dict_overall = {
                                "charge": charge_dict_overall,
                                "dipole": dipole_info,
                                "atomic_dipole": atomic_dipole_dict_overall,
                            }
                            with open(json_file, "w") as f:
                                json.dump(charge_dict_overall, f, indent=4)
                        except:
                            print("error parsing charge")

                    elif routine == "hirshfeld":
                        try:
                            charge_dict_overall, dipole_info = parse_charge_base(
                                file_full_path, corrected=False
                            )
                            data = {
                                "charge": charge_dict_overall,
                                "dipole": dipole_info,
                            }
                            with open(json_file, "w") as f:
                                json.dump(data, f, indent=4)
                        except:
                            print("error parsing hirshfeld")

                    elif routine == "vdd":
                        try:
                            charge_dict_overall, dipole_info = parse_charge_base(
                                file_full_path, corrected=False
                            )
                            data = {
                                "charge": charge_dict_overall,
                                "dipole": dipole_info,
                            }
                            with open(json_file, "w") as f:
                                json.dump(data, f, indent=4)
                        except:
                            print("error parsing vdd")

                    elif routine == "mbis":
                        try:
                            charge_dict_overall = parse_charge_base(
                                file_full_path, corrected=False, dipole=False
                            )
                            data = {"charge": charge_dict_overall}
                            with open(json_file, "w") as f:
                                json.dump(data, f, indent=4)
                        except:
                            print("error parsing mbis")

                    elif routine == "bader":
                        try:
                            charge_dict_overall, spin_info = parse_charge_doc_bader(
                                file_full_path
                            )
                            data = {"charge": charge_dict_overall, "spin": spin_info}
                            with open(json_file, "w") as f:
                                json.dump(data, f, indent=4)
                        except:
                            print("error parsing bader")

                    elif routine == "cm5":
                        try:
                            charge_dict_overall, dipole_info = parse_charge_base(
                                file_full_path, corrected=False
                            )
                            data = {
                                "charge": charge_dict_overall,
                                "dipole": dipole_info,
                            }
                            with open(json_file, "w") as f:
                                json.dump(data, f, indent=4)
                        except:
                            print("error parsing cm5")

                    elif routine == "adch":
                        try:
                            (
                                charge_dict_overall,
                                atomic_dipole_dict_overall,
                                dipole_info,
                            ) = parse_charge_doc_adch(file_full_path)
                            data = {
                                "charge": charge_dict_overall,
                                "dipole": dipole_info,
                                "atomic_dipole": atomic_dipole_dict_overall,
                            }
                            with open(json_file, "w") as f:
                                json.dump(data, f, indent=4)
                        except:
                            print("error parsing adch")

                    elif routine == "becke":
                        try:
                            (
                                charge_dict_overall,
                                atomic_dipole_dict_overall,
                                dipole_info,
                            ) = parse_charge_becke(file_full_path)
                            data = {
                                "charge": charge_dict_overall,
                                "dipole": dipole_info,
                                "atomic_dipole": atomic_dipole_dict_overall,
                            }
                            with open(json_file, "w") as f:
                                json.dump(data, f, indent=4)
                        except:
                            print("error parsing becke")

                    elif routine == "chelpg":
                        try:
                            charge_dict_overall = parse_charge_chelpg(file_full_path)
                            data = {"charge": charge_dict_overall}
                            with open(json_file, "w") as f:
                                json.dump(data, f, indent=4)
                        except:
                            print("error parsing chelpg")

                    elif routine in list(fuzzy_dict.keys()):

                        try:
                            data = parse_fuzzy_real_space(file_full_path)
                            with open(json_file, "w") as f:
                                json.dump(data, f, indent=4)
                        except:
                            print("error parsing fuzzy")

        elif "CPprop.txt" in file and "qtaim" in routine_list:

            json_file = os.path.join(folder, "qtaim.json")
            # go through files in folder and find the one that ends in .inp
            cp_prop_path = os.path.join(folder, file)

            for file2 in os.listdir(folder):
                if file2.endswith(".inp"):
                    inp_loc = os.path.join(folder, file2)
                    inp_orca = True
                if file2.endswith("input.in"):
                    inp_loc = os.path.join(folder, file2)
                    inp_orca = False
            try:
                qtaim_dict = parse_qtaim(
                    cprop_file=cp_prop_path, inp_loc=inp_loc, orca_tf=inp_orca
                )

                with open(json_file, "w") as f:
                    json.dump(qtaim_dict, f, indent=4)
            except:
                print("error parsing qtaim")

    if separate:
        charge_routines = list(charge_dict.keys())
        bond_routines = list(bond_dict.keys())
        fuzzy_routines = list(fuzzy_data().keys())
        charge_dict_compiled = {}
        bond_dict_compiled = {}
        fuzzy_dict_compiled = {}

        for file in os.listdir(folder):
            for routine in charge_routines:
                if routine == file.split(".json")[0]:
                    with open(os.path.join(folder, file), "r") as f:
                        charge_dict_compiled[routine] = json.load(f)
                    # remove the file
                    os.remove(os.path.join(folder, file))

            for routine in bond_routines:
                if routine == file.split(".json")[0]:
                    with open(os.path.join(folder, file), "r") as f:
                        bond_dict_compiled[routine] = json.load(f)
                    # remove the file
                    os.remove(os.path.join(folder, file))

            for routine in fuzzy_routines:
                if routine == file.split(".json")[0]:
                    with open(os.path.join(folder, file), "r") as f:
                        fuzzy_dict_compiled[routine] = json.load(f)[routine]
                        #print(fuzzy_dict_compiled)
                    # remove the file
                    os.remove(os.path.join(folder, file))

        # check that charge_dict_compiled is not {}
        if charge_dict_compiled:
            with open(os.path.join(folder, "charge.json"), "w") as f:
                json.dump(charge_dict_compiled, f, indent=4)
        
        # check that bond_dict_compiled is not {}
        if bond_dict_compiled:
            with open(os.path.join(folder, "bond.json"), "w") as f:
                json.dump(bond_dict_compiled, f, indent=4)

        # check that fuzzy_dict_compiled is not {}
        if fuzzy_dict_compiled:
            #print(fuzzy_dict_compiled)
            with open(os.path.join(folder, "fuzzy_full.json"), "w") as f:
                json.dump(fuzzy_dict_compiled, f, indent=4)


def clean_jobs(folder, separate=False):
    """
    Clean up the mess of files created by the analysis
    Takes:
        folder(str): folder to clean
        separate(bool): whether to separate the analysis into different files
    """

    if separate:
        order_of_operations = ORDER_OF_OPERATIONS_separate
        charge_dict = charge_data_dict()
        [order_of_operations.append(i) for i in charge_dict.keys()]
        bond_dict = bond_order_dict()
        [order_of_operations.append(i) for i in bond_dict.keys()]
        fuzzy_dict = fuzzy_data()
        [order_of_operations.append(i) for i in fuzzy_dict.keys()]

    else:
        order_of_operations = ORDER_OF_OPERATIONS

    txt_files = []
    txt_files = [i + ".txt" for i in order_of_operations] + ["convert.txt"]
    txt_files += [i + ".out" for i in order_of_operations]

    # print all jobs ending in .mfwn
    for file in os.listdir(folder):
        if file.endswith(".mfwn"):
            os.remove(os.path.join(folder, file))

        if file.endswith(".txt"):
            if file in txt_files:
                os.remove(os.path.join(folder, file))

        if file.endswith(".out"):
            if file in txt_files:
                os.remove(os.path.join(folder, file))

        if file.endswith(".molden.input"):
            os.remove(os.path.join(folder, file))

        if file.endswith(".wfn"):
            os.remove(os.path.join(folder, file))

        if file.endswith("convert.in"):
            os.remove(os.path.join(folder, file))


def gbw_analysis(
    folder,
    multiwfn_cmd,
    orca_2mkl_cmd,
    separate=True,
    parse_only=False,
    clean=True,
    overwrite=True,
    orca_6=True,
    restart=False,
    debug=False,
):
    """
    Run a full analysis on a folder of gbw files
    Takes:
        folder(str): folder to analyze
        multiwfn_cmd(str): command to run multiwfn
        orca_2mkl_cmd(str): command to run orca_2mkl
        separate(bool): whether to separate the analysis into different files
        parse_only(bool): whether to only parse the files
        clean(bool): whether to clean
        overwrite(bool): whether to overwrite the output files
        orca_6(bool): whether calc is from orca6
        restart(bool): whether to restart from the last step using timings.json
        debug(bool): whether to run a minimal set of jobs

    """
    if not os.path.exists(folder):
        print("Folder does not exist")
        return
    
    if restart:
        print("Restarting from last step in timings.json")
        # check if the timings file exists
        if not os.path.exists(os.path.join(folder, "timings.json")):
            print("No timings file found - starting from scratch!")
            restart = False
    
    # check if output already exists
    if not overwrite:
        timings_loc = os.path.join(folder, "timings.json")
        bond_loc = os.path.join(folder, "bond.json")
        charge_loc = os.path.join(folder, "charge.json")

        tf_timings = os.path.exists(timings_loc) and os.path.getsize(timings_loc) > 0
        tf_bond = os.path.exists(bond_loc) and os.path.getsize(bond_loc) > 0
        tf_charge = os.path.exists(charge_loc) and os.path.getsize(charge_loc) > 0

        # check that all files are not empty
        if tf_timings and tf_bond and tf_charge:
            print("Output already exists")
            return

    if not parse_only:
        print("... Creating jobs")
        # create jobs for conversion to wfn and multiwfn analysis
        create_jobs(
            folder=folder,
            multiwfn_cmd=multiwfn_cmd,
            orca_2mkl_cmd=orca_2mkl_cmd,
            separate=separate,
            debug=debug,
        )
        # run jobs
        run_jobs(
            folder=folder,
            separate=separate,
            orca_6=orca_6,
            restart=restart,
            debug=debug,
        )

    print("... Parsing multiwfn output")
    # parse those jobs to jsons for 5 categories
    parse_multiwfn(folder, separate=separate, debug=debug)

    if clean:
        #    # clean some of the mess
        print("... Cleaning up")
        clean_jobs(folder, separate=separate)
