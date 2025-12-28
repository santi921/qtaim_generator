import os
from typing import Dict, Sequence, Any, Union, List
import time
import random

from qtaim_gen.source.core.parse_qtaim import dft_inp_to_dict
from qtaim_gen.source.utils.validation import validation_checks

def sanitize_folders(folders: List[str]) -> List[str]:
    """Sanitize folder paths by stripping whitespace and removing empty entries. Also just checks for basic validity.

    Args:
        folders: List of folder paths as strings.
    Returns:
        A sanitized list of folder paths.
    """
    sanitized = [f.strip() for f in folders if f.strip()]
    existing_folders = []
    missing = []
    for f in sanitized:
        if os.path.exists(f) and os.path.isdir(f):
            existing_folders.append(f)
        else:
            missing.append(f)
            pass
    if missing:
        print(
            f"Warning: {len(missing)} listed paths do not exist or are not directories; they will be skipped."
        )
        for m in missing[:10]:
            print("  missing:", m)
        if len(missing) > 10:
            print("  ...")
    folders = existing_folders
    
    if not folders:
        print("No valid folders to run after filtering missing entries.")
        return 0
    
    return folders
    

def get_folders_from_file(
    job_file: str, 
    num_folders: int,
    root_omol_results: str=None, 
    root_omol_inputs: str=None,
    pre_validate: bool=False, 
    move_results: bool=True, 
    full_set: int=0,
    logger: Any=None
) -> List[str]:
    
    print(f"collecting {num_folders} folders from {job_file} with pre_validate={pre_validate}")
    
    folder_file = os.path.join(job_file)
    if pre_validate:
        num_sample = num_folders * 10 # hack to just attempt to finish folders
    else:
        num_sample = num_folders

    folders = sample_lines(folder_file, num_sample)
    folders = sanitize_folders(folders)

    if not folders:
        return []
    
    #num_folders = len(folders)
    # shuffle folders
    random.shuffle(folders)
    
    if pre_validate:
        folders_run = []
        for folder in folders:
            folder_inputs = folder
            if root_omol_inputs and root_omol_results and folder_inputs.startswith(root_omol_inputs):
                folder_relative = folder_inputs[len(root_omol_inputs) :].lstrip(os.sep)
                folder_outputs = root_omol_results + os.sep + folder_relative
            else:
                folder_outputs = folder_inputs  # fallback to same folder if no roots provided
            try:
                tf_validation = validation_checks(
                    folder_outputs,
                    full_set=full_set,
                    verbose=False,
                    move_results=move_results,
                    logger=logger,
                )
                if not tf_validation:
                    folders_run.append(folder)
                    if logger:
                        logger.info(f"Adding {folder} to run list after pre-validation")
                
                else: 
                    if logger:
                        logger.info(f"Skipping {folder} due to pre-validation pass")
            
            except: 
                # add folder to run b/c it's likely json io error 
                folders_run.append(folder)
                if logger:
                    logger.info(f"Adding {folder} to run list after pre-validation exception")
            
            
            if len(folders_run) >= num_folders:
                print(f"Pre-validation collected {len(folders_run)} folders to run.")
                return folders_run
    else:
        folders_run = folders[:num_folders]
    
    return folders_run


def pull_ecp_dict(orca_out: str) -> Dict[int, Dict[str, Union[str, float]]]:
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


def overwrite_molden_w_ecp(
    molden_file: str, dict_ecp: Dict[int, Dict[str, Union[str, float]]]
) -> None:
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


def check_spin(folder: str) -> bool:
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
    spin = int(dft_dict.get("spin", None))
    # print("Spin found: {}".format(spin))
    if spin == 1:
        return False
    return True


def check_folder_writing(folder: str) -> float:
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


def write_input_file(
    folder: str, lines: Sequence[str], n_atoms: int, options: Dict[str, Any]
) -> None:
    """
    Write input file for Multiwfn.
    Takes:
        folder: folder to write input file to
        lines: lines from xyz file
        n_atoms: number of atoms in xyz file
        options: dictionary of options for input file
    Returns:
        None
    """
    with open(folder + "/input.in", "w") as f:
        f.write("!{} {} AIM\n\n".format(options["functional"], options["basis"]))
        f.write("*xyz {} {}\n".format(options["charge"], options["spin"]))
        for ind in range(n_atoms):
            f.write(
                str(lines[ind + 2].split()[0])
                + "\t"
                + str(lines[ind + 2].split()[1])
                + "\t"
                + str(lines[ind + 2].split()[1])
                + "\t"
                + str(lines[ind + 2].split()[2])
                + "\n"
            )
        f.write("*\n")


def convert_inp_to_xyz(orca_path: str, output_path: str) -> None:
    """
    Convert an ORCA input file to an XYZ file.
    Takes:
        orca_path: path to ORCA input file
        output_path: path to write XYZ file to
    Returns:
        None
    """

    mol_dict: Dict[str, Any] = dft_inp_to_dict(orca_path, parse_charge_spin=True)

    n_atoms: int = len(mol_dict["mol"])

    xyz_str: str = "{}\n".format(n_atoms)
    spin_charge_line: str = "{} {}\n".format(mol_dict["charge"], mol_dict["spin"])
    xyz_str += spin_charge_line
    # write the atom positions
    for ind, atom in mol_dict["mol"].items():

        atom_line: str = "{} {} {} {}\n".format(
            atom["element"], atom["pos"][0], atom["pos"][1], atom["pos"][2]
        )
        xyz_str += atom_line

    with open(output_path, "w") as f:
        f.write(xyz_str)


def write_input_file_from_pmg_molecule(
    folder: str, molecule: Union[Any, Dict[str, Any]], options: Dict[str, Any]
) -> None:
    try:
        sites = molecule.sites
        charge = molecule.charge
        spin = molecule.spin_multiplicity
        pmg = True
    except Exception:
        sites = molecule["sites"]
        charge = molecule["charge"]
        spin = molecule["spin_multiplicity"]
        pmg = False

    n_atoms: int = int(len(sites))

    # print(folder)
    with open(folder + "/input.in", "w") as f:
        # for relativistic set functional to "TPSS ZORA" and basis to "ZORA-def2-TZVP SARC/J"
        f.write("!{} {} AIM\n\n".format(options["functional"], options["basis"]))
        if "basis_atoms" in options:
            f.write("%basis\n")
            for atom in options["basis_atoms"]:
                f.write(
                    'NewGTO    {} "{}" end\n'.format(atom["element"], atom["basis"])
                )
            f.write("end\n")
        if "relativistic" in options:
            if options["relativistic"] == True:
                f.write("%rel\n")
                f.write("picturechange  true\n")
                f.write("end\n")
        if "parallel_procs" in options:
            f.write("{}\n".format(options["parallel_procs"]))

        f.write("%SCF\n")
        f.write("    MaxIter 1000\n")
        f.write("END\n")
        f.write(
            "* xyz {} {}\n".format(
                int(charge),
                int(spin),
            )
        )
        for ind in range(n_atoms):
            if pmg:
                xyz = sites[ind].coords
                atom = sites[ind].specie.symbol
            else:
                xyz = sites[ind]["xyz"]
                atom = sites[ind]["element"]
            f.write(
                "{}\t{: .4f}\t{: .4f}\t{: .4f}\n".format(atom, xyz[0], xyz[1], xyz[2])
            )
        f.write("*\n")


def check_results_exist(folder: str) -> bool:
    """
    Check if the results files exist in the folder.
    Takes:
        folder: folder to check
    Returns:
        bool: True if results files exist, False otherwise
    """
    required_files = [
        "timings.json",
        "qtaim.json",
        "other.json",
        "fuzzy_full.json",
        "charge.json",
    ]
    for file in required_files:
        file_check = os.path.join(folder, file)
        if not os.path.exists(file_check) or os.path.getsize(file_check) == 0:
            return False
    return True


def sample_lines(filename, n):
    """
    Uniformly sample n lines from a text file with an unknown but large number
    of lines, using a two-pass approach.

    Takes:
        filename (str): Path to the text file.
        n (int): Number of lines to sample.
    Returns:
        List[str]: List of n sampled lines.
    """
    # Pass 1: count lines
    with open(filename, "r") as f:
        total_lines = sum(1 for _ in f)

    if n > total_lines:
        # just return all lines
        with open(filename, "r") as f:
            return f.readlines()

    # Choose n distinct line indices in C and sort them
    chosen_indices = sorted(random.sample(range(total_lines), n))

    samples = []
    target_idx_pos = 0
    target = chosen_indices[target_idx_pos]

    # Pass 2: stream again and collect chosen lines
    with open(filename, "r") as f:
        for i, line in enumerate(f):
            if i == target:
                samples.append(line)
                target_idx_pos += 1
                if target_idx_pos == n:  # got all of them; stop early
                    break
                target = chosen_indices[target_idx_pos]

    return samples
