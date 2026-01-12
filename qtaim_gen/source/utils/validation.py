from asyncio.log import logger
import os
import json
from qtaim_gen.source.core.parse_qtaim import dft_inp_to_dict
import numpy as np
from datetime import datetime


def get_charge_spin_n_atoms_from_folder(
    folder: str, logger=None, verbose=False
) -> tuple:
    # check for a file ending with .inp
    inp_files = [f for f in os.listdir(folder) if f.endswith(".inp")]
    # add *.in files to list
    inp_files += [f for f in os.listdir(folder) if f.endswith(".in")]
    # remove convert.in
    inp_files = [f for f in inp_files if f != "convert.in"]

    if not inp_files:
        if logger:
            logger.error(f"No .inp file found in folder: {folder}.")
        if verbose:
            print(f"No .inp file found in folder: {folder}.")
        return False
    inp_file = inp_files[0]  # take the first .inp file found

    if logger:
        logger.info(f'Using input file "{inp_file}" for validation.')

    if verbose:
        print(f'Using input file "{inp_file}" for validation.')

    # gather n_atoms, spin, charge from the orca.inp file
    orca_inp_path = os.path.join(folder, inp_file)  # might need to change this name
    if not os.path.exists(orca_inp_path):
        if verbose:
            print(f"Missing orca.inp file at {orca_inp_path}.")
        if logger:
            logger.error(f"Missing orca.inp file at {orca_inp_path}.")
        return False
    return dft_inp_to_dict(orca_inp_path, parse_charge_spin=True)


def get_val_breakdown_from_folder(
    folder: str, full_set: int, spin_tf: bool, n_atoms: int
) -> dict:

    info = {
        "total_time": None,
        "t_qtaim": None,
        "t_charge": None,
        "t_bond": None,
        "t_fuzzy": None,
        "t_other": None,
        "val_time": None,
        "val_qtaim": None,
        "val_charge": None,
        "val_bond": None,
        "val_fuzzy": None,
        "val_other": None,
    }

    # check timings
    timings_file = os.path.join(folder, "timings.json")
    if os.path.exists(timings_file) and os.path.getsize(timings_file) > 0:
        with open(timings_file, "r") as f:
            timings = json.load(f)
        total_time = np.array(list(timings.values())).sum()
        info["total_time"] = total_time

        for col in timings.keys():
            info[f"t_{col}"] = timings[col]
        val_time = validate_timing_dict(
            timings_file, logger=None, full_set=full_set, spin_tf=spin_tf
        )
        info["val_time"] = val_time

    # check fuzzy
    fuzzy_file = os.path.join(folder, "fuzzy_full.json")
    if os.path.exists(fuzzy_file) and os.path.getsize(fuzzy_file) > 0:
        tf_fuzzy = validate_fuzzy_dict(
            fuzzy_file,
            logger=None,
            n_atoms=n_atoms,
            spin_tf=spin_tf,
            full_set=full_set,
        )
        info["val_fuzzy"] = tf_fuzzy

    # check charge
    charge_file = os.path.join(folder, "charge.json")
    if os.path.exists(charge_file) and os.path.getsize(charge_file) > 0:
        tf_charge = validate_charge_dict(charge_file, logger=None)
        info["val_charge"] = tf_charge

    # check bond
    bond_file = os.path.join(folder, "bond.json")
    if os.path.exists(bond_file) and os.path.getsize(bond_file) > 0:
        tf_bond = validate_bond_dict(bond_file, logger=None)
        info["val_bond"] = tf_bond

    # check qtaim
    qtaim_file = os.path.join(folder, "qtaim.json")
    if os.path.exists(qtaim_file) and os.path.getsize(qtaim_file) > 0:
        tf_qtaim = validate_qtaim_dict(qtaim_file, n_atoms=n_atoms, logger=None)
        info["val_qtaim"] = tf_qtaim

    # echeck other
    other_file = os.path.join(folder, "other.json")
    if os.path.exists(other_file) and os.path.getsize(other_file) > 0:
        tf_other = validate_other_dict(
            other_file,
            logger=None,
        )
        info["val_other"] = tf_other

    return info


def validate_timing_dict(
    timing_json_loc: str,
    verbose: bool = False,
    full_set: int = 0,
    spin_tf: bool = False,
    logger: any = None
):
    """
    Basic check that the timing json file has the expected structure.
    Check that it has the keys 'total', 'qtaim', 'charge', 'bond', and 'fuzzy_full'.
    """
    with open(timing_json_loc, "r") as f:
        timing_dict = json.load(f)

    expected_keys = [
        "qtaim",
        "other",
        "hirshfeld",
        "becke",
        "adch",
        "cm5",
        "fuzzy_bond",
        "becke_fuzzy_density",
        "hirsh_fuzzy_density",
    ]

    excepted_spin_keys = ["hirsh_fuzzy_spin", "becke_fuzzy_spin"]

    if full_set > 0:
        expected_keys += [
            "vdd",
            "mbis",
            "chelpg",
            "ibsi_bond",
            "elf_fuzzy",
            "mbis_fuzzy_density",
        ]

        excepted_spin_keys += ["mbis_fuzzy_spin"]

    if full_set > 1:
        expected_keys += [
            "bader",
            "laplacian_bond",
            "grad_norm_rho_fuzzy",
            "laplacian_rho_fuzzy",
        ]

    for key in expected_keys:
        if key not in timing_dict:
            if key == "other" and "other_esp" not in timing_dict: 
                if logger:
                    logger.error(f"Missing expected key '{key}' or 'other_esp' in timing json.")
                if verbose:
                    print(f"Missing expected key '{key}' or 'other_esp' in timing json.")
                return False
            elif key == "other" and "other_esp" in timing_dict:
                key = "other_esp"
            else: 
                if logger:
                    logger.error(f"Missing expected key '{key}' in timing json.")
                if verbose:
                    print(f"Missing expected key '{key}' in timing json.")
                return False

        # check that the times aren't tiny
        if timing_dict[key] < 1e-6 and key != "convert":
            if logger:
                logger.error(
                    f"Timing for '{key}' is too small: {timing_dict[key]} seconds."
                )
            print(f"Timing for '{key}' is too small: {timing_dict[key]} seconds.")
            return False

    if spin_tf:
        for key in excepted_spin_keys:
            if key not in timing_dict:
                if verbose:
                    print(f"Missing expected spin key '{key}' in timing json.")
                if logger:
                    logger.error(f"Missing expected spin key '{key}' in timing json.")
                return False

    if verbose:
        print("Timing json structure is valid.")
    return True


def validate_bond_dict(
    bond_json_loc: str, verbose: bool = False, full_set: int = 0, logger: any = None
):
    """
    Basic check that the bond json file has the expected structure.
    Check that it has the keys 'fuzzy_bond', 'ibsi_bond', and 'laplacian_bond'.
    """
    with open(bond_json_loc, "r") as f:
        bond_dict = json.load(f)

    expected_keys = ["fuzzy_bond"]

    if full_set > 0:
        expected_keys += ["ibsi_bond"]
    if full_set > 1:
        expected_keys += ["laplacian_bond"]

    for key in expected_keys:
        if key not in bond_dict:
            if verbose:
                print(f"Missing expected key '{key}' in bond json.")
            if logger:
                logger.error(f"Missing expected key '{key}' in bond json.")
            return False

    if verbose:
        print("Bond json structure is valid.")

    return True


def validate_fuzzy_dict(
    fuzzy_json_loc: str,
    n_atoms: int = None,
    spin_tf: bool = False,
    verbose: bool = False,
    full_set: int = 0,
    logger: any = None,
):
    """
    Basic check that the fuzzy json file has the expected structure.
    Check that it has the keys 'fuzzy', 'fuzzy_bonds', 'fuzzy_bcp', 'fuzzy_ncp'.
    """
    with open(fuzzy_json_loc, "r") as f:
        fuzzy_dict = json.load(f)

    expected_keys = [
        "becke_fuzzy_density",
        "hirsh_fuzzy_density",
    ]

    if full_set > 0:
        expected_keys += ["elf_fuzzy", "mbis_fuzzy_density"]

    if full_set > 1:
        expected_keys += ["grad_norm_rho_fuzzy", "laplacian_rho_fuzzy"]

    if spin_tf:
        expected_keys += ["hirsh_fuzzy_spin", "becke_fuzzy_spin"]

        if full_set > 0:
            expected_keys += ["mbis_fuzzy_spin"]

    for key in expected_keys:
        if key not in fuzzy_dict:
            if logger:
                logger.error(f"Missing expected key '{key}' in fuzzy json.")
            return False
        if n_atoms is not None:
            if len(fuzzy_dict[key]) != n_atoms + 2:
                if verbose:
                    print(
                        f"Number of fuzzy points ({len(fuzzy_dict[key])}) does not match expected ({n_atoms})."
                    )
                if logger:
                    logger.error(
                        f"Number of fuzzy points ({len(fuzzy_dict[key])}) does not match expected ({n_atoms})."
                    )
                return False
    if verbose:
        print("Fuzzy json structure is valid.")
    return True


def validate_other_dict(other_dict_loc: str, verbose: bool = False, logger: any = None):
    """
    Basic check that the other json file has the expected structure.
    Check that it has the keys 'atoms', 'bonds', 'charges', and 'fuzzy'.
    """
    with open(other_dict_loc, "r") as f:
        other_dict = json.load(f)

    expected_keys = [
        "mpp_full",
        "sdp_full",
        "mpp_heavy",
        "sdp_heavy",
        "ESP_Volume",
        "ESP_Surface_Density",
        "ESP_Minimal_value",
        "ESP_Maximal_value",
        "ESP_Overall_surface_area",
        "ESP_Positive_surface_area",
        "ESP_Negative_surface_area",
        "ESP_Overall_average_value",
        "ESP_Positive_average_value",
        "ESP_Negative_average_value",
        "ESP_Overall_variance",
        "ESP_Positive_variance",
        "ESP_Negative_variance",
        "ESP_Balance_of_charges",
        "ESP_Product_of_sigma",
        "ESP_Internal_charge_separation",
        "ESP_Molecular_polarity_index",
        "ESP_Nonpolar_surface_area",
        "ESP_Polar_surface_area",
        "ALIE_Volume",
        "ALIE_Surface_Density",
        "ALIE_Minimal_value",
        "ALIE_Maximal_value",
        "ALIE_Overall_surface_area",
        "ALIE_Positive_surface_area",
        "ALIE_Negative_surface_area",
        "ESP_Overall_skewness",
        # "ESP_Positive_skewness",
        "ALIE_Overall_skewness",
        # "ALIE_Positive_skewness",
    ]
    for key in expected_keys:
        if key not in other_dict:
            if verbose:
                print(
                    f"Warning: Missing expected key '{key}' in other json. This may not be critical."
                )
            if logger:
                logger.warning(
                    f"Missing expected key '{key}' in other json. This may not be critical."
                )
            return False

    if verbose:
        print("Other json structure is valid.")
    return True


def validate_charge_dict(
    charge_json_loc: str,
    n_atoms: int = None,
    verbose: bool = False,
    full_set: int = 0,
    logger: any = None,
):
    """
    Basic check that the charge json file has the expected structure.
    Check that it has the keys 'mbis', 'adch', 'chelpg', 'becke',  'hirshfeld', 'cm5', 'bader', 'vdd'
    Check each one of these keys has a key "charge" with n_atoms entries.
    """
    with open(charge_json_loc, "r") as f:
        charge_dict = json.load(f)

    expected_keys = ["adch", "becke", "hirshfeld", "cm5"]

    if full_set > 0:
        expected_keys += ["mbis", "vdd", "chelpg"]
    if full_set > 1:
        expected_keys += ["bader"]

    for key in expected_keys:
        if key not in charge_dict:
            if verbose:
                print(f"Warning: Missing expected key '{key}' in charge json. ")
            if logger:
                logger.warning(
                    f"Warning: Missing expected key '{key}' in charge json. "
                )
            return False

    for key in expected_keys:
        if "charge" not in charge_dict[key]:
            if verbose:
                print(f"Missing 'charge' key in '{key}' of charge json.")
            if logger:
                logger.error(f"Missing 'charge' key in '{key}' of charge json.")
            return False

        if n_atoms is not None:
            if len(charge_dict[key]["charge"]) != n_atoms:
                if verbose:
                    print(
                        f"Number of charges in '{key}' ({len(charge_dict[key]['charge'])}) does not match expected ({n_atoms})."
                    )
                if logger:
                    logger.error(
                        f"Number of charges in '{key}' ({len(charge_dict[key]['charge'])}) does not match expected ({n_atoms})."
                    )
                return False
    if verbose:
        print("Charge json structure is valid.")
    return True


def validate_qtaim_dict(
    qtaim_json_loc: str, n_atoms: int = None, verbose: bool = False, logger: any = None
):
    """
    Basic check that the qtaim json file has the expected structure
    Check that it has the keys 'atoms', 'bonds', 'charges', and 'fuzzy'.
    If n_atoms is provided, check that the number of non-bonded critical points matches n_atoms.
    If harsh_check is True, also check that the number of nuclear critical points matches n_atoms.
    """
    with open(qtaim_json_loc, "r") as f:
        qtaim_dict = json.load(f)
    # check it isn't empty
    if not qtaim_dict:
        if verbose:
            print("QTAIM json file is empty.")
        return False

    # dict_ncps = {qtaim_dict[key] for key in qtaim_dict if "_" not in key}
    dict_ncps = [qtaim_dict[key] for key in list(qtaim_dict.keys()) if "_" not in key]
    # dict_bcps = {qtaim_dict[key] for key in qtaim_dict if "_" in key}
    dict_bcps = [qtaim_dict[key] for key in list(qtaim_dict.keys()) if "_" in key]

    if n_atoms is not None:
        if len(dict_ncps) != n_atoms:
            if verbose:
                print(
                    f"Number of nuclear critical points ({len(dict_ncps)}) does not match expected ({n_atoms})."
                )
            if logger:
                logger.error(
                    f"Number of nuclear critical points ({len(dict_ncps)}) does not match expected ({n_atoms})."
                )
            return False
    if verbose:
        print(f"Number of nuclear critical points: {len(dict_ncps)}")
        print(f"Number of bond critical points: {len(dict_bcps)}")
        print("QTAIM json structure is valid.")
    return True


def validation_checks(
    folder: str,
    verbose: bool = False,
    full_set: int = 0,
    move_results: bool = True,
    logger: any = None,
):
    """
    Run all validation checks on the json files in the given folder.
    Arguments:
        folder (str): Path to the folder containing the json files.
        verbose (bool): If True, print detailed validation messages.
        full_set (int): Level of calculation detail (0-baseline, 1-baseline, 2-full).
        move_results (bool): Adjust if files have been moved during cleaning.
    Returns:
        bool: True if all validation checks pass, False otherwise.
    """
    # check that all the json files are present
    required_files = [
        "timings.json",
        "fuzzy_full.json",
        "other.json",
        "charge.json",
        "qtaim.json",
        "bond.json",
    ]
    tf = True

    if move_results:
        folder_check_res = os.path.join(folder, "generator")
    else:
        folder_check_res = folder

    for file in required_files:
        if not os.path.exists(os.path.join(folder_check_res, file)):
            if logger:
                logger.error(
                    f"Missing required file: {file} in folder: {folder_check_res}"
                )
            if verbose:
                print(f"Missing required file: {file} in folder: {folder_check_res}")
            tf = False

    if not tf:
        return False

    dft_dict = get_charge_spin_n_atoms_from_folder(
        folder, logger=logger, verbose=verbose
    )
    if not dft_dict:
        return False
    # print("log dict: ", str(dft_dict))
    n_atoms = len(dft_dict["mol"])
    spin = dft_dict.get("spin", None)
    charge = dft_dict.get("charge", None)

    if verbose:
        print(f"n_atoms: {n_atoms}, spin: {spin}, charge: {charge}")
    if logger:
        logger.info(f"n_atoms: {n_atoms}, spin: {spin}, charge: {charge}")

    if spin != 1:
        spin_tf = True
    else:
        spin_tf = False

    timing_json_loc = os.path.join(folder_check_res, "timings.json")
    fuzzy_json_loc = os.path.join(folder_check_res, "fuzzy_full.json")
    other_dict_loc = os.path.join(folder_check_res, "other.json")
    charge_json_loc = os.path.join(folder_check_res, "charge.json")
    qtaim_json_loc = os.path.join(folder_check_res, "qtaim.json")
    bond_json_loc = os.path.join(folder_check_res, "bond.json")
    # bonding_json_loc = os.path.join(folder, "bonding.json")
    tf_cond = True

    if not validate_timing_dict(
        timing_json_loc, verbose=verbose, full_set=full_set, spin_tf=spin_tf
    ):
        if logger:
            logger.error(f"Timing json validation failed in folder: {folder}")
        tf_cond = False

    if not validate_fuzzy_dict(
        fuzzy_json_loc,
        n_atoms=n_atoms,
        spin_tf=spin_tf,
        verbose=verbose,
        full_set=full_set,
        logger=logger,
    ):
        if logger:
            logger.error(f"Fuzzy json validation failed in folder: {folder}")
        tf_cond = False

    if not validate_other_dict(other_dict_loc, verbose=verbose, logger=logger):
        if logger:
            logger.error(f"Other json validation failed in folder: {folder}")
        tf_cond = False

    if not validate_charge_dict(
        charge_json_loc, n_atoms=n_atoms, verbose=verbose, logger=logger
    ):
        if logger:
            logger.error(f"Charge json validation failed in folder: {folder}")
        tf_cond = False

    if not validate_qtaim_dict(
        qtaim_json_loc, n_atoms=n_atoms, verbose=verbose, logger=logger
    ):
        if logger:
            logger.error(f"QTAIM json validation failed in folder: {folder}")
        tf_cond = False

    if not validate_bond_dict(
        bond_json_loc, verbose=verbose, full_set=full_set, logger=logger
    ):
        if logger:
            logger.error(f"Bond json validation failed in folder: {folder}")
        tf_cond = False

    if verbose:
        print("All validation checks passed.")

    return tf_cond


def get_information_from_job_folder(folder: str, full_set: int) -> dict:
    """Extracts relevant information from the job folder name."""

    # check if folder has /generator/ subdirectory, if so get timings.json in that folder

    info = {
        "validation_level_0": None,
        "validation_level_1": None,
        "validation_level_2": None,
        "total_time": None,
        "t_qtaim": None,
        "t_other": None,
        "last_edit_time": None,
        "val_time": None,
        "val_qtaim": None,
        "val_charge": None,
        "val_bond": None,
        "val_fuzzy": None,
        "val_other": None,
        "n_atoms": None,
        "spin": None,
        "charge": None,
    }

    # get .inp file in the folder for spin, charge, n_atoms
    dft_dict = get_charge_spin_n_atoms_from_folder(folder, logger=None, verbose=False)
    # print("DFT dict: ", dft_dict)
    if not dft_dict:
        return info  # return empty info if dft_dict is None or empty

    n_atoms = len(dft_dict["mol"])
    spin = dft_dict.get("spin", None)
    charge = dft_dict.get("charge", None)

    info["n_atoms"] = n_atoms
    info["spin"] = spin
    info["charge"] = charge

    if spin != 1:
        spin_tf = True
    else:
        spin_tf = False

    # check is there is a generator subfolder
    # print("Folder to analyze: ", folder)

    if "generator" in os.listdir(folder):
        # print("Found generator subfolder.")
        gen_folder = folder + "/generator/"
        timings_file = os.path.join(gen_folder, "timings.json")

        if os.path.exists(timings_file) and os.path.getsize(timings_file) > 0:
            with open(timings_file, "r") as f:
                timings = json.load(f)
            total_time = float(np.array(list(timings.values())).sum())
            info["total_time"] = total_time

            for col in timings.keys():
                info[f"t_{col}"] = timings[col]
        else:
            return info  # return empty info if timings file is missing or empty

        tf_validation_level_0 = validation_checks(
            folder, full_set=0, verbose=False, move_results=True, logger=None
        )

        tf_validation_level_1 = validation_checks(
            folder, full_set=1, verbose=False, move_results=True, logger=None
        )

        tf_validation_level_2 = validation_checks(
            folder, full_set=2, verbose=False, move_results=True, logger=None
        )

        # set val_qtaim, val_charge, val_bond, val_fuzzy, val_other to True for corresponding level
        if full_set == 0:
            status_val = tf_validation_level_0
        elif full_set == 1:
            status_val = tf_validation_level_1
        elif full_set == 2:
            status_val = tf_validation_level_2

        if status_val:
            info["val_qtaim"] = True
            info["val_charge"] = True
            info["val_bond"] = True
            info["val_fuzzy"] = True
            info["val_other"] = True
            info["val_time"] = True
        else:
            dict_val = get_val_breakdown_from_folder(
                gen_folder, n_atoms=n_atoms, full_set=full_set, spin_tf=spin_tf
            )
            info.update(dict_val)

        # check edit date of timings.json
        mtime_timestamp = os.path.getmtime(timings_file)
        # Convert the timestamp to a datetime object
        mtime_datetime = datetime.fromtimestamp(mtime_timestamp)
        # Format the datetime object into a human-readable string
        # Example format: YYYY-MM-DD HH:MM:SS
        human_readable_mtime = mtime_datetime.strftime("%Y-%m-%d %H:%M:%S")
        info["last_edit_time"] = human_readable_mtime

        info.update(
            {
                "validation_level_0": tf_validation_level_0,
                "validation_level_1": tf_validation_level_1,
                "validation_level_2": tf_validation_level_2,
            }
        )

    else:
        timings_file = os.path.join(folder, "timings.json")

        if os.path.exists(timings_file):
            with open(timings_file, "r") as f:
                timings = json.load(f)
            total_time = float(np.array(list(timings.values())).sum())
            info["total_time"] = total_time

            for col in timings.keys():
                info[f"t_{col}"] = timings[col]

            edit_time = os.path.getmtime(timings_file)
            # Convert the timestamp to a datetime object
            mtime_datetime = datetime.fromtimestamp(edit_time)
            # Format the datetime object into a human-readable string
            human_readable_mtime = mtime_datetime.strftime("%Y-%m-%d %H:%M:%S")
            info["last_edit_time"] = human_readable_mtime

        dict_val = get_val_breakdown_from_folder(
            folder, n_atoms=n_atoms, full_set=full_set, spin_tf=spin_tf
        )
        info.update(dict_val)

    return info
