from pathlib import Path
import os, stat, json, time, logging
from typing import Optional, Dict, Any, List
import subprocess

import zipfile

from qtaim_gen.source.utils.validation import (
    validation_checks,
    get_val_breakdown_from_folder,
    get_charge_spin_n_atoms_from_folder,
)

from qtaim_gen.source.utils.io import check_results_exist
from qtaim_gen.source.utils.atomic_write import atomic_json_write

from qtaim_gen.source.data.multiwfn import (
    charge_data,
    charge_data_dict,
    bond_order_data,
    bond_order_dict,
    fuzzy_data,
    other_data,
    other_data_dict,
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
    parse_other_doc_esp, 
    parse_other_doc_geometry,
    parse_qtaim,
    parse_charge_doc_bader,
    parse_charge_chelpg,
    parse_fuzzy_real_space,
)

from qtaim_gen.source.utils.io import (
    pull_ecp_dict,
    overwrite_molden_w_ecp,
    check_spin,
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ORDER_OF_OPERATIONS = ["fuzzy_full", "qtaim", "bond", "charge", "other"]
ORDER_OF_OPERATIONS_separate = [
    "fuzzy_full",
    "charge_separate",
    "bond_separate",
    "qtaim",
    "other_separate",  # muting for meta
]


def write_settings_file(folder: str, mem: int = 400000000, n_threads: int = 3) -> None:
    # get loc of qtaim_embed folder
    qtaim_embed_loc = str(Path(__file__).parent.parent.parent)
    old_path = qtaim_embed_loc + "/source/data/settings.ini"
    # copy old path to current path
    # wwrite txt files line by line and add a line n_threads = str(n_threads) + "\n"
    # another line ompstacksize= str(mem) + "\n"
    with open(old_path, "r") as f:
        data = f.read()
    # write to new path
    # remove last line and save separately
    last_lines = data.split("\n")[-3:]
    data = "\n".join(data.split("\n")[:-3])  # remove last line

    new_path = str(Path(folder).joinpath("settings.ini"))
    # if new_path exists remove it
    if os.path.exists(new_path):
        os.remove(new_path)
    with open(new_path, "w") as f:
        f.write(data)
        f.write("  nthreads= {}\n".format(n_threads))
        f.write("  ompstacksize= {}\n".format(mem))
        f.write(last_lines[0] + "\n")  # write last line
        f.write(last_lines[1] + "\n")  # write last line without newline
        f.write(last_lines[2])  # write last line without newline
        f.write("\n")


def write_conversion(
    out_folder: str,
    read_file: str,
    overwrite: bool = False,
    name: str = "convert.in",
    orca_2mkl_cmd: str = "orca_2mkl",
) -> None:
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
            + "'" + str(Path.home().joinpath(out_folder, read_file)) + "'"
            + " -molden\n"
        )
        # also have it clean up the gbw file, .molden.input file

        if os.path.exists(str(Path.home().joinpath(out_folder, read_file + ".molden.input"))):
            f.write("rm '{}.molden.input'\n".format(str(Path.home().joinpath(out_folder, read_file))))

        # check if gbw file exists
        if os.path.exists(str(Path.home().joinpath(out_folder, read_file + ".gbw"))):
            f.write("rm '{}.gbw'\n".format(str(Path.home().joinpath(out_folder, read_file))))

    st = os.stat(out_file)
    os.chmod(out_file, st.st_mode | stat.S_IEXEC)


def write_multiwfn_exe(
    out_folder: str,
    read_file: str,
    multi_wfn_cmd: str,
    multiwfn_input_file: str,
    convert_gbw: bool = False,
    overwrite: bool = False,
    mv_cpprop: bool = False,
    gbw_override: bool = False,
    name: str = "props.mfwn",
) -> None:
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
        gbw_override(bool): whether to override the gbw file location
    """

    out_file = str(Path.home().joinpath(out_folder, name))
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)

    completed_tf = os.path.exists(out_file) and os.path.getsize(out_file) > 0

    if (not completed_tf) or overwrite:
        with open(out_file, "w") as f:
            f.write("#!/bin/bash\n")
            if convert_gbw:
                f.write("orca_2mkl '" + str(Path.home().joinpath(out_folder)) + "'\n")

            multiwfn_input_file_root = multiwfn_input_file.split("/")[-1].split(".")[0]
            bare_file = read_file.split("/")[-1]
            
            if gbw_override: 
                # 
                gbw_loc = bare_file
            else: 
                gbw_loc = str(Path.home().joinpath(out_folder, read_file))

            
            f.write(
                "{} ".format(multi_wfn_cmd)  # multiwfn command
                + "'" + gbw_loc + "'"  # wfn/gbw file
                + " < '{}' | tee ".format(multiwfn_input_file)  # multiwfn input file
                + "'" + str(
                    Path.home().joinpath(
                        out_folder, "{}.out".format(multiwfn_input_file_root)
                    )
                ) + "'"  # output file
                + "\n"
            )

            if mv_cpprop:
                f.write(
                    "mv CPprop.txt "
                    + "'" + str(Path.home().joinpath(out_folder, "CPprop.txt")) + "'"
                    + "\n"
                )

        st = os.stat(out_file)
        os.chmod(out_file, st.st_mode | stat.S_IEXEC)


def create_jobs(
    folder: str,
    multiwfn_cmd: str,
    orca_2mkl_cmd: str,
    separate: bool = False,
    debug: bool = True,
    logger: Optional[logging.Logger] = None,
    full_set: int = 0,
    patch_path: bool= False,
) -> None:
    """
    Create job files for multiwfn analysis
    Takes:
        folder(str): folder to create jobs in
        multiwfn_cmd(str): command to run multiwfn
        orca_2mkl_cmd(str): command to run orca_2mkl  
        separate(bool): whether to separate the analysis into different files
        overwrite(bool): whether to overwrite the output files
        orca_6(bool): whether calc is from orca6
        logger(logging.Logger): logger to log messages
        full_set(int): whether to use full set of analysis (1) or minimal (0)
        patch_path(bool): whether to patch the pathing in the multiwfn input files

    """
    if logger is None:
        logger = logging.getLogger("gbw_analysis")

    if separate:
        routine_list = ORDER_OF_OPERATIONS_separate.copy()
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
            file_wfn_search = os.path.join(folder, file)

    bool_gbw = False
    file_read = None

    for file in os.listdir(folder):
        if file.endswith(".gbw"):
            bool_gbw = True
            file_gbw = os.path.join(folder, file)
            file_wfn = file.replace(".gbw", ".wfn")
            # if there is a wfn change its name to the gbw prefix
            if wfn_present:
                if file_wfn not in os.listdir(folder):
                    logger.info(f"Renaming WFN file to: {file_wfn}")
                    os.rename(
                        os.path.join(folder, file_wfn_search),
                        os.path.join(folder, file_wfn),
                    )

            file_molden = file.replace(".gbw", ".molden.input")
            file_molden = os.path.join(folder, file_molden)
            file_read = os.path.join(folder, file_wfn)

        if file.endswith(".wfn"):
            file_read = os.path.join(folder, file)

    if not wfn_present and bool_gbw:
        logger.info(f"file_gbw: {file_gbw}")
        logger.info(f"out folder: {folder}")
        logger.info("wfn not there - writing conversion script")
        try:
            write_conversion(
                out_folder=folder,
                read_file=file_gbw,
                overwrite=True,
                name="convert.in",
                orca_2mkl_cmd=orca_2mkl_cmd,
            )
            routine_list = ["convert"] + routine_list

        except Exception as e:
            logger.error(f"Error writing conversion script: {e}")

    job_dict = {}

    # print("folder: {}".format(folder))
    for routine in routine_list:
        try:

            if routine == "qtaim":
                job_dict["qtaim"] = os.path.join(folder, "qtaim.txt")
                # write qtaim data file
                with open(os.path.join(folder, "qtaim.txt"), "w") as f:
                    data = qtaim_data()
                    f.write(data)

            elif routine == "fuzzy_full":
                # job_dict["fuzzy_full"] = os.path.join(folder, "fuzzy_full.txt")
                # with open(os.path.join(folder, "fuzzy_full.txt"), "w") as f:
                spin_tf = check_spin(folder)
                print("spin_tf: {}".format(spin_tf))
                fuzzy_dict = fuzzy_data(spin=spin_tf, full_set=full_set)
                for key, value in fuzzy_dict.items():
                    job_dict[key] = os.path.join(folder, "{}.txt".format(key))
                    with open(os.path.join(folder, "{}.txt".format(key)), "w") as f:
                        f.write(value)
                    # f.write(data)

            elif routine == "bond":
                job_dict["bond"] = os.path.join(folder, "bond.txt")
                with open(os.path.join(folder, "bond.txt"), "w") as f:
                    data = bond_order_data()
                    f.write(data)

            elif routine == "bond_separate":
                bond_dict = bond_order_dict(full_set=full_set)
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
                charge_dict = charge_data_dict(full_set=full_set)
                for key, value in charge_dict.items():
                    job_dict[key] = os.path.join(folder, "{}.txt".format(key))
                    with open(os.path.join(folder, "{}.txt".format(key)), "w") as f:
                        f.write(value)

            elif routine == "other":
                job_dict["other"] = os.path.join(folder, "other.txt")
                with open(os.path.join(folder, "other.txt"), "w") as f:
                    data = other_data()
                    f.write(data)

            elif routine == "other_separate":
                other_dict = other_data_dict(full_set=full_set)
                for key, value in other_dict.items():
                    job_dict[key] = os.path.join(folder, "{}.txt".format(key))
                    with open(os.path.join(folder, "{}.txt".format(key)), "w") as f:
                        f.write(value)

            elif routine == "convert":
                job_dict["convert"] = os.path.join(folder, "convert.txt")
                with open(os.path.join(folder, "convert.txt"), "w") as f:
                    file_wfn = file_gbw.replace(".gbw", ".wfn")
                    file_wfn_bare = file_wfn.split("/")[-1]
                    #print("file_wfn: {}".format(file_wfn))
                    data = "100\n2\n5\n{}\n0\nq\n".format(file_wfn_bare)
                    f.write(data)

            # else:
            #    logger.warning(f"Routine not recognized: {routine}")

        except Exception as e:
            logger.error(f"Error creating job for routine '{routine}': {e}")

    # print(job_dict)
    for key, value in job_dict.items():
        try:
            # if key == "qtaim":
            #    mv_cpprop = True
            # else:
            mv_cpprop = False

            if key == "charge_separate":
                charge_dict = charge_data_dict(full_set=full_set)
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
                        gbw_override=patch_path,

                    )

            elif key == "bond_separate":
                bond_dict = bond_order_dict(full_set=full_set)
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
                        gbw_override=patch_path,

                    )

            elif key == "other_separate":
                other_dict = other_data_dict(full_set=full_set)
                for key, value in other_dict.items():
                    write_multiwfn_exe(
                        out_folder=folder,
                        read_file=file_read,
                        multi_wfn_cmd=multiwfn_cmd,
                        multiwfn_input_file=value,
                        convert_gbw=False,
                        overwrite=True,
                        name="props_{}.mfwn".format(key),
                        mv_cpprop=mv_cpprop,
                        gbw_override=patch_path,
                    )
            
            elif key == "convert":
                # use ./ b/c multiwfn throws a fit sometimes
                write_multiwfn_exe(
                    out_folder=folder,
                    gbw_override=True,
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
                    gbw_override=patch_path,
                )

            logger.info(f"Created execution script for {key}")

        except Exception as e:
            logger.error(f"Error creating execution script for {key}: {e}")


def run_jobs(
    folder: str,
    separate: bool = False,
    orca_6: bool = True,
    restart: bool = False,
    debug: bool = False,
    logger: Optional[logging.Logger] = None,
    prof_mem: bool = False,
    full_set: int = 0,
    move_results: bool = False,
    clean_jobs_tf: bool = False,
) -> None:
    """
    Run conversion and multiwfn jobs
    Takes:
        folder(str): folder to run jobs in
        separate(bool): whether to separate the analysis into different files
        logger(logging.Logger): logger to log messages
        prof_mem(bool): whether to profile memory usage
        full_set(int): whether to use full set of analysis (1) or minimal (0)
        move_results(bool): whether to move results to a separate folder
        clean_jobs_tf(bool): whether to remove job files after running

    """
    if logger is None:
        logger = logging.getLogger("gbw_analysis")

    if separate:
        # copy the ORDER_OF_OPERATIONS_separate list
        order_of_operations = ORDER_OF_OPERATIONS_separate.copy()
        # order_of_operations = ORDER_OF_OPERATIONS_separate
        # order_of_operations = ["qtaim"]
        charge_dict = charge_data_dict(full_set=full_set)
        bond_dict = bond_order_dict(full_set=full_set)
        spin_tf = check_spin(folder)
        fuzzy_dict = fuzzy_data(spin=spin_tf, full_set=full_set)
        other_dict = other_data_dict(full_set=full_set)
        [order_of_operations.append(i) for i in charge_dict.keys()]
        [order_of_operations.append(i) for i in bond_dict.keys()]
        [order_of_operations.append(i) for i in fuzzy_dict.keys()]
        [order_of_operations.append(i) for i in other_dict.keys()]

        """# remove "charge_separate" and "bond_separate" from list
        if "charge_separate" in order_of_operations:
            order_of_operations.remove("charge_separate")
        if "bond_separate" in order_of_operations:
            order_of_operations.remove("bond_separate")
        if "fuzzy_full" in order_of_operations:
            order_of_operations.remove("fuzzy_full")"""
    else:
        order_of_operations = ORDER_OF_OPERATIONS

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
        logger.info("Running conversion script")
        if conv_file is None:
            logger.error("No conversion script (convert.in) found in folder.")
            return

        # run conversion script using subprocess with explicit cwd
        try:
            subprocess.run(["bash", conv_file], cwd=folder, check=True)
        except Exception as e:
            logger.error(f"Error running conversion script: {e}")

        for file in os.listdir(folder):
            if file.endswith(".molden.input"):
                molden_file = os.path.join(folder, file)
            if file.endswith("orca.out") or file.endswith("output.out"):
                orca_out = os.path.join(folder, file)
        if not orca_6:
            try:
                dict_ecp = pull_ecp_dict(orca_out)
                overwrite_molden_w_ecp(molden_file, dict_ecp)
            except Exception as e:
                logger.error(f"Error replacing ECP values in molden file: {e}")

        order_of_operations = ["convert"] + order_of_operations
        # conversion script should read in .molden file and use multiwfn to convert to .wfn

    # create a json file to store job status

    timings = {}
    # run multiwfn scripts
    if move_results:
        folder_check = os.path.join(folder, "generator")
    else:
        folder_check = folder

    if os.path.exists(os.path.join(folder_check, "timings.json")):

        with open(os.path.join(folder_check, "timings.json"), "r") as f:
            # check that timings isn't empty
            if os.path.getsize(os.path.join(folder_check, "timings.json")) > 0:
                timings = json.load(f)

        if restart:
            dft_dict = get_charge_spin_n_atoms_from_folder(
                folder, logger=None, verbose=False
            )
            n_atoms = len(dft_dict["mol"])
            dict_val = get_val_breakdown_from_folder(
                folder, n_atoms=n_atoms, full_set=full_set, spin_tf=spin_tf
            )

    for order in order_of_operations:
        # if restart, check if timing file exists
        if restart:
            if order in timings.keys():
                # now check if validation checks pass
                if order == "qtaim":
                    if dict_val["val_qtaim"]:
                        logger.info(
                            f"Skipping {order} in {folder}: already completed successfully."
                        )
                        continue

                elif order == "other":
                    if dict_val["val_other"]:
                        logger.info(
                            f"Skipping {order} in {folder}: already completed successfully."
                        )
                        continue

                elif order in charge_dict.keys():
                    if dict_val.get(f"val_charge", False):
                        logger.info(
                            f"Skipping {order} in {folder}: already completed successfully."
                        )
                        continue

                elif order in bond_dict.keys():
                    if dict_val.get(f"val_bond", False):
                        logger.info(
                            f"Skipping {order} in {folder}: already completed successfully."
                        )
                        continue

                elif order in other_dict.keys():
                    if dict_val.get(f"val_other", False):
                        logger.info(
                            f"Skipping {order} in {folder}: already completed successfully."
                        )
                        continue

                elif order in fuzzy_dict.keys():
                    if dict_val.get(f"val_fuzzy", False):
                        logger.info(
                            f"Skipping {order} in {folder}: already completed successfully."
                        )
                        continue

                elif order == "fuzzy_full":
                    if dict_val.get(f"val_fuzzy", False):
                        logger.info(
                            f"Skipping {order} in {folder}: already completed successfully."
                        )
                        continue

        else:
            folder_check = folder
        if prof_mem:
            memory = {}

        mfwn_file = os.path.join(folder, "props_{}.mfwn".format(order))

        try:
            logger.info(f"Running {mfwn_file}")
            start = time.time()

            # to replace
            # run the multiwfn wrapper script with explicit cwd to avoid depending on process CWD
            try:
                subprocess.run(["bash", mfwn_file], cwd=folder, check=True)
            except Exception as e:
                logger.error(f"Error running {mfwn_file} via subprocess: {e}")

            end = time.time()
            # remove the .mfwn file after running
            if os.path.exists(mfwn_file):
                logger.info(f"Removing file: {mfwn_file}")
                if clean_jobs_tf:
                    os.remove(mfwn_file)

            timings[order] = end - start
            logger.info(f"Completed {order} in {end - start:.2f} seconds")

        except Exception as e:
            logger.error(f"Error running {mfwn_file}: {e}")
            timings[order] = -1

        # save timings to file in folder - at each step for check pointing
        try:
            atomic_json_write(os.path.join(folder_check, "timings.json"), timings)
            logger.info(f"Saved timings.json in {folder}")

            if prof_mem:
                atomic_json_write(os.path.join(folder_check, "memory.json"), memory)

        except Exception as e:
            logger.error(f"Error saving timings.json: {e}")


def parse_multiwfn(
    folder: str,
    separate: bool = False,
    debug: bool = False,
    logger: Optional[logging.Logger] = None,
    full_set: int = 0,
) -> None:
    """
    Parse multiwfn output files to jsons and save them in folder
    Takes:
        folder(str): folder to parse
        separate(bool): whether to separate the analysis into different files
        debug(bool): whether to run a minimal set of jobs
        return_dicts(bool): return results as well as writing
    """
    # if return_dicts:
    #    compiled_dicts = {}

    if separate:
        routine_list = ORDER_OF_OPERATIONS_separate.copy()
        charge_dict = charge_data_dict(full_set=full_set)
        bond_dict = bond_order_dict(full_set=full_set)
        spin_tf = check_spin(folder)
        # print("spin_tf: {}".format(spin_tf))
        fuzzy_dict = fuzzy_data(spin=spin_tf, full_set=full_set)
        other_dict = other_data_dict(full_set=full_set)
        [routine_list.append(i) for i in charge_dict.keys()]
        [routine_list.append(i) for i in bond_dict.keys()]
        [routine_list.append(i) for i in fuzzy_dict.keys()]
        [routine_list.append(i) for i in other_dict.keys()]
        # print("routine_list: {}".format(routine_list))

        # if "charge_separate" in routine_list:
        #    routine_list.remove("charge_separate")
        # if "bond_separate" in routine_list:
        #    routine_list.remove("bond_separate")
        # if "fuzzy_full" in routine_list:
        #    routine_list.remove("fuzzy_full")

    else:
        routine_list = ORDER_OF_OPERATIONS

    if debug:
        routine_list = ["qtaim"]

    for file in os.listdir(folder):
        if file.endswith(".out"):
            file_full_path = os.path.join(folder, file)
            for routine in routine_list:
                if routine in file:
                    json_file = file_full_path.replace(".out", ".json")
                    try:
                        if routine == "fuzzy_full":
                            data = parse_fuzzy_doc(file_full_path)

                        elif routine == "fuzzy_bond":
                            data = parse_bond_order_fuzzy(file_full_path)

                        elif routine == "ibsi_bond":
                            data = parse_bond_order_ibsi(file_full_path)

                        elif routine == "laplacian_bond":
                            data = parse_bond_order_laplace(file_full_path)

                        elif routine == "fuzzy":
                            data = parse_bond_order_fuzzy(file_full_path)

                        elif routine == "other":
                            data = parse_other_doc(file_full_path)

                        elif routine == "other_esp":
                            data = parse_other_doc_esp(file_full_path, ind_surface_prefix="ESP")

                        elif routine == "other_alie":
                            data = parse_other_doc_esp(file_full_path, ind_surface_prefix="ALIE")

                        elif routine == "other_geometry":
                            data = parse_other_doc_geometry(file_full_path)

                        elif routine == "charge":
                            (
                                charge_dict_overall,
                                atomic_dipole_dict_overall,
                                dipole_info,
                            ) = parse_charge_doc(file_full_path)

                            data = {
                                "charge": charge_dict_overall,
                                "dipole": dipole_info,
                                "atomic_dipole": atomic_dipole_dict_overall,
                            }

                        elif routine == "hirshfeld":
                            charge_dict_overall, dipole_info = parse_charge_base(
                                file_full_path, corrected=False
                            )
                            data = {
                                "charge": charge_dict_overall,
                                "dipole": dipole_info,
                            }

                        elif routine == "vdd":
                            charge_dict_overall, dipole_info = parse_charge_base(
                                file_full_path, corrected=False
                            )
                            data = {
                                "charge": charge_dict_overall,
                                "dipole": dipole_info,
                            }

                        elif routine == "mbis":
                            charge_dict_overall = parse_charge_base(
                                file_full_path, corrected=False, dipole=False
                            )
                            data = {"charge": charge_dict_overall}

                        elif routine == "bader":
                            charge_dict_overall, spin_info = parse_charge_doc_bader(
                                file_full_path
                            )
                            data = {"charge": charge_dict_overall, "spin": spin_info}

                        elif routine == "cm5":
                            charge_dict_overall, dipole_info = parse_charge_base(
                                file_full_path, corrected=False
                            )
                            data = {
                                "charge": charge_dict_overall,
                                "dipole": dipole_info,
                            }

                        elif routine == "adch":
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

                        elif routine == "becke":
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

                        elif routine == "chelpg":
                            charge_dict_overall = parse_charge_chelpg(file_full_path)
                            data = {"charge": charge_dict_overall}

                        elif routine in list(fuzzy_dict.keys()):
                            data = parse_fuzzy_real_space(file_full_path)

                        # elif routine == "qtaim":
                        #    pass

                        else:
                            logger.warning(
                                f"Unknown routine '{routine}' in file {file_full_path}"
                            )
                            continue

                        atomic_json_write(json_file, data)
                        logger.info(f"Parsed {routine} output to {json_file}")

                    except Exception as e:
                        if routine == "qtaim":
                            pass
                        else:
                            logger.error(
                                f"Error parsing {routine} in {file_full_path}: {e}"
                            )

        elif "CPprop.txt" in file and "qtaim" in routine_list:
            json_file = os.path.join(folder, "qtaim.json")
            cp_prop_path = os.path.join(folder, file)
            inp_loc = None
            inp_orca = None
            for file2 in os.listdir(folder):
                if file2.endswith(".inp"):
                    inp_loc = os.path.join(folder, file2)
                    inp_orca = True

                if file2.endswith("input.in"):
                    inp_loc = os.path.join(folder, file2)
                    inp_orca = False

            qtaim_dict = parse_qtaim(
                cprop_file=cp_prop_path, inp_loc=inp_loc, orca_tf=inp_orca
            )

            try:
                # print(cp_prop_path)
                # print(inp_loc)
                # print(inp_orca)

                qtaim_dict = parse_qtaim(
                    cprop_file=cp_prop_path, inp_loc=inp_loc, orca_tf=inp_orca
                )
                if qtaim_dict is None or qtaim_dict == {}:
                    logger.error(
                        f"QTAIM parsing returned empty dictionary for {folder}"
                    )
                    continue

                atomic_json_write(json_file, qtaim_dict)
                # if return_dicts:
                #    compiled_dicts["qtaim"] = qtaim_dict
                logger.info(f"Parsed qtaim output to {json_file}")

            except Exception as e:
                logger.error(f"Error parsing qtaim: {e}")

    if separate:
        charge_routines = list(charge_dict.keys())
        bond_routines = list(bond_dict.keys())
        fuzzy_routines = list(fuzzy_data(spin=spin_tf, full_set=full_set).keys())
        other_routines = list(other_dict.keys())
        charge_dict_compiled = {}
        bond_dict_compiled = {}
        fuzzy_dict_compiled = {}
        other_dict_compiled = {}

        # remove double for loops by combining
        directory_files = os.listdir(folder)
        combined_routines = charge_routines + bond_routines + fuzzy_routines + other_routines
        for routine in combined_routines:
            # json name
            file = routine + ".json"
            if file in directory_files:
                if routine == file.split(".json")[0]:
                    with open(os.path.join(folder, file), "r") as f:
                        if routine in charge_routines:
                            charge_dict_compiled[routine] = json.load(f)
                        elif routine in bond_routines:
                            bond_dict_compiled[routine] = json.load(f)
                        elif routine in fuzzy_routines:
                            fuzzy_dict_compiled[routine] = json.load(f)[routine]
                        elif routine in other_routines:
                            # json.load will returna. dict with several keys, we just want to update onto compiled 
                            other_dict_compiled.update(json.load(f))
                            
                    # remove the file
                    # if os.path.exists(os.path.join(folder, file)):
                    #    logger.info(f"Removing file: {file}")
                    #    os.remove(os.path.join(folder, file))

        if charge_dict_compiled:
            atomic_json_write(os.path.join(folder, "charge.json"), charge_dict_compiled)
            # if return_dicts:
            #    compiled_dicts["charge"] = charge_dict_compiled
            logger.info("Compiled charge.json")

        if bond_dict_compiled:
            atomic_json_write(os.path.join(folder, "bond.json"), bond_dict_compiled)
            # if return_dicts:
            #    compiled_dicts["bond"] = bond_dict_compiled
            logger.info("Compiled bond.json")

        if fuzzy_dict_compiled:
            atomic_json_write(os.path.join(folder, "fuzzy_full.json"), fuzzy_dict_compiled)
            # if return_dicts:
            #    compiled_dicts["fuzzy_full"] = fuzzy_dict_compiled
            logger.info("Compiled fuzzy_full.json")

        if other_dict_compiled:
            atomic_json_write(os.path.join(folder, "other.json"), other_dict_compiled)
            # if return_dicts:
            #    compiled_dicts["other"] = other_dict_compiled
            logger.info("Compiled other.json")

    # clean individual json files if separate AFTER compiled jsons writtent to ensure
    # data isn't lost if error occurs
    if separate:
        for routine in combined_routines:
            file = routine + ".json"
            if file in directory_files:
                os.remove(os.path.join(folder, file))
                logger.info(f"Removed {file}")


def clean_jobs(
    folder: str,
    separate: bool = False,
    logger: Optional[logging.Logger] = None,
    full_set: int = 0,
    move_results: bool = True,
) -> None:
    """
    Clean up the mess of files created by the analysis
    Takes:
        folder(str): folder to clean
        separate(bool): whether to separate the analysis into different files
        logger(logging.Logger): logger to log messages
    Removes:
        - all .mfwn files
        - all .txt files that are not in the order of operations
        - all .out files that are not in the order of operations
        - all molden.input files
        - all convert.in files
    """
    if logger is None:
        logger = logging.getLogger("gbw_analysis")
    logger.info("Cleaning up jobs in folder: {}".format(folder))

    if separate:
        order_of_operations = ORDER_OF_OPERATIONS_separate.copy()
        charge_dict = charge_data_dict(full_set=full_set)
        [order_of_operations.append(i) for i in charge_dict.keys()]
        bond_dict = bond_order_dict(full_set=full_set)
        [order_of_operations.append(i) for i in bond_dict.keys()]
        spin_tf = check_spin(folder)
        fuzzy_dict = fuzzy_data(spin=spin_tf, full_set=full_set)
        other_dict = other_data_dict(full_set=full_set)
        [order_of_operations.append(i) for i in fuzzy_dict.keys()]
        [order_of_operations.append(i) for i in other_dict.keys()]
    else:
        order_of_operations = ORDER_OF_OPERATIONS

    txt_files = []
    txt_files = [i + ".txt" for i in order_of_operations] + ["convert.txt"]
    txt_files += [i + ".out" for i in order_of_operations] + ["convert.out"]
    txt_files += [i for i in ["settings.ini", "convert.out", "convert.txt"]]
    zip_file_out = os.path.join(folder, "out_files.zip")
    # print all jobs ending in .mfwn
    for file in os.listdir(folder):
        try:
            if file.endswith(".mfwn"):
                os.remove(os.path.join(folder, file))
                logger.info(f"Removed {file}")
            if file.endswith(".txt"):
                if file in txt_files:
                    os.remove(os.path.join(folder, file))
                    logger.info(f"Removed {file}")                   
            if file.endswith(".molden.input"):
                os.remove(os.path.join(folder, file))
                logger.info(f"Removed {file}")
            if file.endswith("settings.ini"):
                os.remove(os.path.join(folder, file))
                logger.info(f"Removed {file}")
            if file.endswith("convert.in"):
                os.remove(os.path.join(folder, file))
                logger.info(f"Removed {file}")
            if file.endswith("CPprop.txt"):
                os.remove(os.path.join(folder, file))
                logger.info(f"Removed {file}")
            if file.endswith("fuzzy_full.txt"):
                os.remove(os.path.join(folder, file))
                logger.info(f"Removed {file}")
            if file.endswith("wfn"):
                # if there is a gbw in the folder remove wfn
                if any(f.endswith(".gbw") for f in os.listdir(folder)):
                    os.remove(os.path.join(folder, file))
                    logger.info(f"Removed {file}")

        except Exception as e:
            logger.info(f"Couldn't rm file {file}: {e}")

    # zip all out files
    with zipfile.ZipFile(zip_file_out, "w") as zipf:
        for file in os.listdir(folder):
            # skip orca.out 
            if file.endswith(".out") and file != "orca.out":
                zipf.write(os.path.join(folder, file), arcname=file)
                os.remove(os.path.join(folder, file))
                logger.info(f"Zipped and removed {file}")
        
        if move_results:
            results_folder = os.path.join(folder, "generator")
            if not os.path.exists(results_folder):
                os.mkdir(results_folder)
            os.rename(
                zip_file_out,
                os.path.join(results_folder, "out_files.zip"),
            )
            logger.info(f"Moved zipped out files to results folder")


def setup_logger(folder: str, name: str = "gbw_analysis") -> logging.Logger:
    logger = logging.getLogger(f"{name}-{folder}")
    # Avoid duplicate handlers
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        fh = logging.FileHandler(os.path.join(folder, "gbw_analysis.log"))
        fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


def move_results_to_folder(
    folder: str, logger: logging.Logger, clean: bool = True
) -> None:
    """
    Docstring for move_results_to_folder

    :param folder: Description
    :type folder: str
    :param logger: Description
    :type logger: logging.Logger
    :param clean: Description
    :type clean: bool
    """
    results_list = [
        "timings.json",
        "charge.json",
        "bond.json",
        "fuzzy_full.json",
        "qtaim.json",
        "other.json",
        "orca.json",
    ]
    results_folder = os.path.join(folder, "generator")

    if not os.path.exists(results_folder):
        os.mkdir(results_folder)

    for file in os.listdir(folder):
        if file in results_list:
            # if file exists in results folder, merge the jsons
            if os.path.exists(os.path.join(results_folder, file)):
                try:
                    with open(os.path.join(folder, file), "r") as f:
                        data_new = json.load(f)
                    with open(os.path.join(results_folder, file), "r") as f:
                        data_existing = json.load(f)
                    # merge the two dicts
                    if isinstance(data_existing, dict) and isinstance(data_new, dict):

                        data_merged = {**data_existing, **data_new}
                    else:
                        data_merged = data_new  # if not dict, just overwrite
                    atomic_json_write(os.path.join(results_folder, file), data_merged)

                    logger.info(f"Merged {file} into results folder")
                    # remove the original file

                    if clean:
                        os.remove(os.path.join(folder, file))
                except Exception as e:
                    logger.error(f"Error merging file {file}: {e}")
            else:
                try:
                    os.rename(
                        os.path.join(folder, file),
                        os.path.join(results_folder, file),
                    )
                    logger.info(f"Moved {file} to results folder")
                except Exception as e:
                    logger.error(f"Error moving file {file}: {e}")


def _validate_parse_completeness(orca_dict: dict) -> bool:
    """Thin wrapper — canonical implementation lives in parse_orca."""
    from qtaim_gen.source.core.parse_orca import validate_parse_completeness
    return validate_parse_completeness(orca_dict)


def _extract_orca_out_from_archive(folder: str, logger: logging.Logger) -> bool:
    """Try to extract just orca.out from orca.tar.zst in *folder*.

    Returns True if orca.out was successfully extracted.
    """
    tar_zst = os.path.join(folder, "orca.tar.zst")
    if not os.path.isfile(tar_zst):
        return False

    try:
        subprocess.run(
            ["tar", "--zstd", "-xf", "orca.tar.zst", "orca.out"],
            cwd=folder,
            check=True,
            capture_output=True,
        )
        extracted = os.path.isfile(os.path.join(folder, "orca.out"))
        if extracted:
            logger.info("Extracted orca.out from orca.tar.zst")
        return extracted
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        logger.warning("Could not extract orca.out from archive: %s", e)
        return False


def _run_orca_parse(
    folder: str,
    move_results: bool,
    logger: logging.Logger,
    delete_out: bool = False,
) -> None:
    """Parse orca.out -> orca.json + merge into charge.json/bond.json.

    Handles:
    - Both 'orca.out' and 'output.out' filenames (via shared find_orca_output_file)
    - Extracts orca.out from orca.tar.zst if not already on disk
    - Missing orca.out (skip silently)
    - Writes orca_parse timing into timings.json for checkpoint support
    - Atomic writes throughout including timings.json (crash-safe)
    - Optionally deletes orca.out after successful parse (28-114MB files)
    - Validates parse completeness before allowing deletion
    - Merge is additive/idempotent -- safe to re-run
    """
    from qtaim_gen.source.core.parse_orca import (
        parse_orca_output,
        write_orca_json,
        merge_orca_into_charge_json,
        merge_orca_into_bond_json,
        find_orca_output_file,
    )

    orca_out_path = find_orca_output_file(folder)
    extracted_from_archive = False
    if orca_out_path is None:
        # orca.out may still be inside the compressed archive
        if _extract_orca_out_from_archive(folder, logger):
            orca_out_path = find_orca_output_file(folder)
            extracted_from_archive = True
    if orca_out_path is None:
        logger.info("No orca.out found in %s -- skipping ORCA parse", folder)
        return

    t_start = time.time()
    try:
        orca_dict = parse_orca_output(orca_out_path)

        # Validate completeness before committing (protects against truncated .out files)
        if not _validate_parse_completeness(orca_dict):
            logger.warning(
                "ORCA parse produced incomplete result (%d keys) for %s -- "
                "keeping orca.out for retry",
                len(orca_dict),
                orca_out_path,
            )
            # Still write partial orca.json (partial data better than none)
            write_orca_json(folder, orca_dict)
            # Clean up extracted file — archive still has it for retry
            if extracted_from_archive:
                try:
                    os.remove(orca_out_path)
                except OSError:
                    pass
            return  # Do NOT delete, do NOT merge

        write_orca_json(folder, orca_dict)
        # Resolve merge targets: prefer generator/ when move_results=True,
        # fall back to root (normal flow where parse_multiwfn just wrote them)
        charge_path = os.path.join(folder, "charge.json")
        bond_path = os.path.join(folder, "bond.json")
        if move_results:
            gen_charge = os.path.join(folder, "generator", "charge.json")
            gen_bond = os.path.join(folder, "generator", "bond.json")
            if os.path.isfile(gen_charge):
                charge_path = gen_charge
            if os.path.isfile(gen_bond):
                bond_path = gen_bond
        merge_orca_into_charge_json(orca_dict, charge_path)
        merge_orca_into_bond_json(orca_dict, bond_path)
        elapsed = round(time.time() - t_start, 2)
        logger.info("Parsed orca.out in %.2f s (%d keys)", elapsed, len(orca_dict))

        # Write orca_parse timing into timings.json (atomic write for crash safety)
        if move_results:
            timings_path = os.path.join(folder, "generator", "timings.json")
        else:
            timings_path = os.path.join(folder, "timings.json")

        if os.path.isfile(timings_path) and os.path.getsize(timings_path) > 0:
            with open(timings_path, "r") as f:
                timings = json.load(f)
            timings["orca_parse"] = elapsed
            atomic_json_write(timings_path, timings)

        # Delete orca.out AFTER successful parse + merge + timing checkpoint.
        # Always clean up if we extracted it from the archive (archive still has it).
        if delete_out or extracted_from_archive:
            try:
                os.remove(orca_out_path)
                logger.info("Deleted %s after successful parse", orca_out_path)
            except OSError as e:
                logger.warning("Could not delete %s: %s", orca_out_path, e)

    except Exception as e:
        logger.error("Error parsing orca.out in %s: %s", folder, e)
        # On error, clean up extracted file (archive still has it for retry)
        if extracted_from_archive:
            try:
                os.remove(orca_out_path)
            except OSError:
                pass


def gbw_analysis(
    folder: str,
    multiwfn_cmd: str,
    orca_2mkl_cmd: str,
    separate: bool = True,
    parse_only: bool = False,
    clean: bool = True,
    overwrite: bool = True,
    orca_6: bool = True,
    restart: bool = False,
    debug: bool = False,
    logger: Optional[logging.Logger] = None,
    mem: int = 400000000,
    n_threads: int = 4,
    prof_mem: bool = False,
    preprocess_compressed: bool = False,
    full_set: int = 0,
    move_results: bool = True,
    patch_path: bool= False,
    check_orca: bool = False,
) -> None:
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
        logger(logging.Logger): logger to log messages
        mem(int): memory to use for the analysis in bytes
        n_threads(int): number of threads to use for the analysis
        prof_mem(bool): whether to profile memory usage during the analysis
        preprocess_compressed(bool): whether to preprocess compressed files (not implemented yet)
        full_set(int): refined set of cheaper calcs or full set of analysis
        move_results(bool): whether to move results to a single results folder after analysis
    Writes:
        - settings.ini file with memory and n_threads
        - jobs for conversion to wfn and multiwfn analysis
        - timings.json file with timings for each step
        - bond.json, charge.json, fuzzy_full.json, qtaim.json, other.json files with parsed data
    Returns:
        - tf_validation(bool): whether the analysis was successful
    """

    if logger is None:
        logger = setup_logger(folder)
    logger.info("Starting gbw_analysis in folder: {}".format(folder))

    if not os.path.exists(folder):
        print("Folder does not exist")
        logger.error("Folder does not exist: {}".format(folder))
        return

    # check if there is a .wfn or .gbw file in the folder. If there is an
    # option to preprocess compressed files
    if preprocess_compressed:
        logger.info("Preprocessing compressed files in folder: {}".format(folder))
        # check if the required files are already uncompressed - .inp, .wfn
        required_files = [".inp", ".wfn"]
        uncompressed_files = [
            f for f in os.listdir(folder) if f.endswith(tuple(required_files))
        ]
        # also check these files are not empty
        uncompressed_files = [
            f
            for f in uncompressed_files
            if os.path.getsize(os.path.join(folder, f)) > 0
        ]

        if uncompressed_files:
            logger.info("Found uncompressed files: {}".format(uncompressed_files))
            logger.info("Skipping uncompression step")

        else:
            logger.warning("No uncompressed files found - will attempt to uncompress")
        # skip if uncompressed files are present

        if len(uncompressed_files) < 2:
            # run unstd and extract in the target folder so resulting files land there
            for file in os.listdir(folder):
                if file.endswith(".tar.zst") or file.endswith(".tgz"):
                    logger.info(f"Found compressed file: {file}")
                    zstd_file = file

                    # run unzstd with cwd=folder so outputs land directly in folder
                    try:
                        subprocess.run(
                            ["unzstd", "-f", zstd_file], cwd=folder, check=True
                        )
                        # remove the zstd file after successful extraction
                        os.remove(os.path.join(folder, zstd_file))
                    except Exception as e:
                        logger.error(f"Error running unzstd on {zstd_file}: {e}")

                    # untar resulting file (tar filename is zstd_file with .tar)
                    if zstd_file.endswith(".tar.zst"):
                        tar_file_name = zstd_file.replace(".tar.zst", ".tar")
                        tar_cmd = ["tar", "-xf", tar_file_name, "--directory", folder]
                        
                    else:
                        tar_file_name = zstd_file.replace(".tgz", ".tar")
                        tar_cmd = ["tar", "-xf", tar_file_name, "--directory", folder]

                    tar_file_out = tar_file_name
                    # extract tar in the folder
                    try:
                        subprocess.run(tar_cmd, cwd=folder, check=True)
                        # remove the tar file after successful extraction
                        os.remove(os.path.join(folder, tar_file_name))
                    except Exception as e:
                        logger.error(f"Error extracting tar {tar_file_name}: {e}")

                    # After extracting in-place (cwd=folder), expected files should be in folder
                    found_any = False
                    for file2 in os.listdir(folder):
                        if (
                            file2.startswith("orca.engrad")
                            or file2.startswith("orca.out")
                            or file2.startswith("orca.inp")
                            or file2.startswith("orca.property.inp")
                            or file2.startswith("orca.property.txt")
                            or file2.startswith("orca_stderr")
                        ):
                            logger.info(f"Found extracted file in folder: {file2}")
                            found_any = True
                    if not found_any:
                        logger.warning(
                            "No expected extracted files found in %s after extraction",
                            folder,
                        )

                if file.endswith(".gbw.zstd0"):
                    logger.info(f"Found compressed gbw file: {file}")
                    zstd_file = file
                    gbw_file = zstd_file.replace(".zstd0", "")
                    # run unzstd to produce gbw_file inside folder
                    try:
                        subprocess.run(
                            ["unzstd", "-o", gbw_file, "-f", zstd_file],
                            cwd=folder,
                            check=True,
                        )
                    except Exception as e:
                        logger.error(f"Error running unzstd for gbw {zstd_file}: {e}")

            for file in os.listdir(folder):
                # clean files ending in .tar, .tar.zst, .tgz, .gbw.zstd0, .zstd, .npz
                check_list = [".tar", ".tar.zst", ".tgz", ".gbw.zstd0", ".zstd", ".npz", "orca5.gbw"]
                if any(file.endswith(ext) for ext in check_list):
                    try:
                        os.remove(os.path.join(folder, file))
                        logger.info(f"Removed intermediate file: {file}")
                    except Exception as e:
                        logger.error(f"Error removing intermediate file {file}: {e}")

    if restart:
        # check if the timings file exists
        if move_results:
            timings_path = os.path.join(folder, "generator", "timings.json")
        else:
            timings_path = os.path.join(folder, "timings.json")

        if not os.path.exists(timings_path) or os.path.getsize(timings_path) == 0:
            logger.warning("No timings file found - starting from scratch!")
            restart = False
        else:
            logger.info("Timings file found - restarting from last step.")

    # check if output already exists
    if not overwrite:
        # if move_results:
        #    folder_check = os.path.join(folder, "generator")
        # else:
        folder_check = folder

        if check_results_exist(folder_check, move_results=move_results):
            print("Output already exists")
            try:
                tf_validation = validation_checks(
                    folder_check,
                    full_set=full_set,
                    verbose=False,
                    move_results=move_results,
                    logger=logger,
                    check_orca=check_orca,
                )
            except Exception as e:
                logger.error(f"Error during validation checks: {e}")
                tf_validation = False

            # we might change level-of-analysis so only return if all requested analyses are present
            if tf_validation:
                logger.info("Output already exists and is valid - skipping analysis")
                logger.info("gbw_analysis completed in folder: {}".format(folder))
                logger.info("Validation status: {}".format(tf_validation))
                return

            # attempt to reparse if output exists but validation failed
            else:
                # Check if orca.json is the ONLY missing piece — if so, just
                # run _run_orca_parse without parse_multiwfn (which would try
                # to read .txt files that may have been cleaned up already)
                try:
                    tf_without_orca = validation_checks(
                        folder_check,
                        full_set=full_set,
                        verbose=False,
                        move_results=move_results,
                        logger=logger,
                        check_orca=False,
                    )
                except Exception:
                    tf_without_orca = False

                if tf_without_orca:
                    # Everything passes except orca — orca-only reparse
                    logger.info(
                        "Validation passes without orca check - running orca-only parse"
                    )
                    _run_orca_parse(folder, move_results, logger)
                    if move_results:
                        move_results_to_folder(folder, logger=logger, clean=clean)
                    # Clean up orca.out after successful orca-only parse (28-114 MB)
                    if clean:
                        from qtaim_gen.source.core.parse_orca import find_orca_output_file
                        orca_out_path = find_orca_output_file(folder)
                        if orca_out_path is not None:
                            try:
                                os.remove(orca_out_path)
                                logger.info("Deleted %s after orca-only parse", orca_out_path)
                            except OSError as e:
                                logger.warning("Could not delete %s: %s", orca_out_path, e)
                    logger.info("gbw_analysis completed (orca-only) in folder: %s", folder)
                    return

                try:
                    logger.warning(
                        "Output exists but validation failed - attempting to reparse before re-running analysis"
                    )
                    parse_multiwfn(
                        folder,
                        separate=separate,
                        debug=debug,
                        logger=logger,
                        full_set=full_set,
                    )

                    # Also reparse ORCA output
                    _run_orca_parse(folder, move_results, logger)

                    if move_results:
                        move_results_to_folder(folder, logger=logger, clean=clean)

                    tf_validation = validation_checks(
                        folder_check,
                        full_set=full_set,
                        verbose=False,
                        move_results=move_results,
                        logger=logger,
                        check_orca=check_orca,
                    )

                    if tf_validation:
                        logger.info(
                            "Reparsing successful on 2nd try - skipping analysis"
                        )
                        logger.info(
                            "gbw_analysis completed in folder: {}".format(folder)
                        )
                        logger.info("Validation status: {}".format(tf_validation))
                        return
                except Exception as e:
                    logger.error(f"Error during reparsing attempt: {e}")
                    logger.info("Proceeding to re-run full analysis.")

    write_settings_file(mem=mem, n_threads=n_threads, folder=folder)

    if not parse_only:
        print("... Creating jobs")
        # create jobs for conversion to wfn and multiwfn analysis
        create_jobs(
            folder=folder,
            multiwfn_cmd=multiwfn_cmd,
            orca_2mkl_cmd=orca_2mkl_cmd,
            separate=separate,
            debug=debug,
            logger=logger,
            full_set=full_set,
            patch_path=patch_path,
        )
        # run jobs
        run_jobs(
            folder=folder,
            separate=separate,
            orca_6=orca_6,
            restart=restart,
            debug=debug,
            logger=logger,
            prof_mem=prof_mem,
            full_set=full_set,
            move_results=move_results,
            clean_jobs_tf=clean,
        )

    print("... Parsing multiwfn output")
    # parse those jobs to jsons for 5 categories
    parse_multiwfn(
        folder, separate=separate, debug=debug, logger=logger, full_set=full_set
    )

    # Parse ORCA output file (if present)
    _run_orca_parse(folder, move_results, logger)

    # move all results to a results folder

    if move_results:
        move_results_to_folder(folder, logger=logger, clean=clean)

    tf_validation = validation_checks(
        folder,
        full_set=full_set,
        verbose=True,
        move_results=move_results,
        logger=logger,
        check_orca=check_orca,
    )
    logger.info("gbw_analysis completed in folder: {}".format(folder))
    logger.info("Validation status: {}".format(tf_validation))
    # move log file to results folder

    # ONLY CLEAN IF VALIDATION PASSED
    if clean and tf_validation:
        logger.info("... Cleaning up")
        # Delete orca.out after validated parse (28-114 MB files)
        from qtaim_gen.source.core.parse_orca import find_orca_output_file
        orca_out_path = find_orca_output_file(folder)
        if orca_out_path is not None:
            try:
                os.remove(orca_out_path)
                logger.info("Deleted %s after validated parse", orca_out_path)
            except OSError as e:
                logger.warning("Could not delete %s: %s", orca_out_path, e)
        clean_jobs(
            folder,
            separate=separate,
            logger=logger,
            full_set=full_set,
            move_results=move_results,
        )


# /global/scratch/users/santiagovargas/gbws_cleaning_lean/ml_elytes/elytes_md_eqv2_electro_512_C3H8O_3_group_133_shell_0_0_1_1341
#!/bin/bash
# SBATCH --job-name=conj_systems
# SBATCH --partition=cm2
# SBATCH --nodes=1
# SBATCH --ntasks-per-node=6
# SBATCH --cpus-per-task=1
# SBATCH --time=40:00:00
# SBATCH -C lr6_m192
# SBATCH -p lr6
# SBATCH --account=lr_blau
# SBATCH --qos=condo_blau
