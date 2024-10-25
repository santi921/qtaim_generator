from pathlib import Path
import os, argparse, stat
from qtaim_gen.source.data.multiwfn import (
    charge_data,
    bond_order_data,
    fuzzy_data,
    other_data,
    qtaim_data,
)


def write_multiwfn_conversion(
    out_folder, read_file, overwrite=False, name="convert.in"
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
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)

    completed_tf = os.path.exists(out_file) and os.path.getsize(out_file) > 0

    if completed_tf and not overwrite:
        with open(out_file, "w") as f:
            f.write("#!/bin/bash\n")
            f.write(
                "orca_2mkl " + str(Path.home().joinpath(out_folder, read_file)) + "\n"
            )


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-root_folder", type=str, default="./")
    parser.add_argument("-full_multiwfn", type=bool, default=True)
    parser.add_argument("-routine_list", type=list, default=["qtaim"])
    parser.add_argument("-multiwfn_cmd", type=str, default="Multiwfn")

    args = parser.parse_args()
    root_folder = args.root_folder
    full_multiwfn = bool(args.full_multiwfn)
    multiwfn_cmd = args.multiwfn_cmd

    if full_multiwfn:
        routine_list = ["qtaim", "fuzzy", "bond", "charge", "other"]
    else:
        routine_list = args.routine

    print("root_folder: {}".format(root_folder))
    print("full_multiwfn: {}".format(full_multiwfn))
    print("multiwfn_cmd: {}".format(multiwfn_cmd))
    print("routine_list: {}".format(routine_list))

    job_dict = {}
    for routine in routine_list:

        if routine == "qtaim":
            job_dict["qtaim"] = os.path.join(root_folder, "qtaim.txt")
            # write qtaim data file
            with open(os.path.join(root_folder, "qtaim.txt"), "w") as f:
                data = qtaim_data()
                f.write(data)

        elif routine == "fuzzy":
            job_dict["fuzzy"] = os.path.join(root_folder, "fuzzy.txt")
            with open(os.path.join(root_folder, "fuzzy.txt"), "w") as f:
                data = fuzzy_data()
                f.write(data)

        elif routine == "bond":
            job_dict["bond"] = os.path.join(root_folder, "bond.txt")
            with open(os.path.join(root_folder, "bond.txt"), "w") as f:
                data = bond_order_data()
                f.write(data)

        elif routine == "charge":
            job_dict["charge"] = os.path.join(root_folder, "charge.txt")
            with open(os.path.join(root_folder, "charge.txt"), "w") as f:
                data = charge_data()
                f.write(data)

        elif routine == "other":
            job_dict["other"] = os.path.join(root_folder, "other.txt")
            with open(os.path.join(root_folder, "other.txt"), "w") as f:
                data = other_data()
                f.write(data)
        else:
            print("routine not recognized")

    # go through folders in root folder and if there's a *wfn or *gbw file, write job files
    for folder in os.listdir(root_folder):
        folder_full_path = os.path.join(root_folder, folder)
        if os.path.isdir(folder_full_path):
            wfn_present = False
            for file in os.listdir(folder_full_path):
                if file.endswith(".wfn"):
                    wfn_present = True
                    file_read = os.path.join(folder_full_path, file)
                if file.endswith(".gbw"):
                    bool_gbw = True
                    file_gbw = os.path.join(folder_full_path, file)

            if not wfn_present and bool_gbw:
                # write conversion script
                write_multiwfn_conversion(
                    out_folder=folder_full_path,
                    # out_file=file_gbw,
                    overwrite=True,
                    name="convert.in",
                )
            else:
                pass
                # print("wfn present")

                for key, value in job_dict.items():
                    # print("key: {}".format(key))

                    if key == "qtaim":
                        mv_cpprop = True
                    else:
                        mv_cpprop = False

                    write_multiwfn_exe(
                        out_folder=folder_full_path,
                        read_file=file_read,
                        multi_wfn_cmd=multiwfn_cmd,
                        multiwfn_input_file=value,
                        convert_gbw=False,
                        overwrite=True,
                        name="props_{}.mfwn".format(key),
                        mv_cpprop=mv_cpprop,
                    )


main()
