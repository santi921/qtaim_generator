import os
from qtaim_gen.source.core.parse_qtaim import dft_inp_to_dict
import time

def pull_ecp_dict(orca_out):
    """
    Method to pull the ecp dictionary from the orca output file.
    Takes:
        orca_out: str, path to the orca output file
    Returns:
        dict_ecp: dict, dictionary with the ecp substitutions.
    """
    trigger = "CARTESIAN COORDINATES (A.U.)"
    trigger_tf = False
    dict_ecp = {}
    with open(orca_out, "r") as f:
        for line in f:
            if trigger in line:
                trigger_tf = True

            if trigger_tf:
                if len(line) < 3:
                    break
                atom_ind_tf = bool(line.split()[0].isnumeric())

                if atom_ind_tf:
                    ind = line.split()[0]
                    element = line.split()[1]
                    number = line.split()[2]
                    if "*" in number:
                        # remove flag
                        number = float(number[:-1])
                        dict_ecp[float(ind)] = {"element": element, "number": number}
    return dict_ecp


def overwrite_molden_w_ecp(molden_file, dict_ecp):
    """
    Method to overwrite the molden file with the ecp substitutions.
    Takes:
        molden_file: str, path to the molden file
        dict_ecp: dict, dictionary with the ecp substitutions.
    """
    molden_file_temp = molden_file + "_temp"

    with open(molden_file, "r") as f:
        trigger_atoms = "[Atoms]"
        trigger_block_tf = False
        lines_replace = []

        while True:
            line = f.readline()

            if "]" in line and "[" in line and trigger_block_tf:
                break

            if trigger_block_tf:
                # print(line)
                split_line = line.split()
                element = split_line[0]
                ind_original = int(split_line[1])

                if ind_original - 1 in dict_ecp.keys():
                    dict_entry = dict_ecp[ind_original - 1]
                    # pad dict_entry["number"] to be 3 characters w/ leading blank
                    dict_entry["number"] = str(dict_entry["number"]).rjust(3)
                    # replace columns 7:10 in line with padded number
                    line = line[:7] + dict_entry["number"] + line[10:]
                    if dict_entry["element"] != element:
                        print("element mismatch!")

            if trigger_atoms in line:
                trigger_block_tf = True

            lines_replace.append(line)

    # copy molden_file to molden_file_out
    with open(molden_file, "r") as f:
        lines = f.readlines()

    with open(molden_file_temp, "w") as f:
        f.writelines(lines_replace)
        # add lines from the rest of the file
        f.writelines(lines[len(lines_replace) :])

    # overwrite the original file
    os.rename(molden_file_temp, molden_file)


def check_spin(folder):
    """
    Utility to find .inp file in folder and parse it to find if it's a doublet
    """
    # find the .inp file in the folder
    orca_inp_files = [f for f in os.listdir(folder) if f.endswith(".inp")]
    # add *.in files to list
    orca_inp_files += [f for f in os.listdir(folder) if f.endswith(".in")]
    orca_inp_files = [f for f in orca_inp_files if f != "convert.in"]

    if not orca_inp_files:
        raise FileNotFoundError("No .inp file found in the folder.")
    inp_file = os.path.join(folder, orca_inp_files[0])
    dft_dict = dft_inp_to_dict(inp_file, parse_charge_spin=True)
    spin = dft_dict.get("spin", None)
    if spin == 1:
        return False
    return True


def check_folder_writing(folder):
    """
    Utility to check how long ago any file in the folder was modified.
    Returns:
        (float) : The time in seconds since the last modification of any file in the folder.
    """
    # Get the current time
    current_time = time.time()
    # Get the modification times of all files in the folder
    mod_times = [os.path.getmtime(os.path.join(folder, f)) for f in os.listdir(folder)]
    # Return the time since the last modification
    return current_time - max(mod_times) if mod_times else 0