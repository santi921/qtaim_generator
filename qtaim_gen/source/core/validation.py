import os
import json 
from qtaim_gen.source.core.parse_qtaim import dft_inp_to_dict

def validate_timing_dict(timing_json_loc, verbose=False): 
    """
    Basic check that the timing json file has the expected structure.
    Check that it has the keys 'total', 'qtaim', 'charge', 'bond', and 'fuzzy_full'.
    """
    with open(timing_json_loc, "r") as f:
        timing_dict = json.load(f)
    
    expected_keys = ['convert',
        'qtaim',
        'other',
        'bader',
        'hirshfeld',
        'vdd',
        'becke',
        'adch',
        'mbis',
        'cm5',
        'chelpg',
        'fuzzy_bond',
        'ibsi_bond',
        'becke_fuzzy_density',
        'mbis_fuzzy_density',
        'hirsh_fuzzy_density',
        'grad_norm_rho_fuzzy',
        'laplacian_rho_fuzzy',
        'elf_fuzzy']

    excepted_spin_keys = [
        'mbis_fuzzy_spin',
        'hirsh_fuzzy_spin',
        'becke_fuzzy_spin',
    ]
    for key in expected_keys:
        if key not in timing_dict:
            if verbose: 
                print(f"Missing expected key '{key}' in timing json.")
            return False
        
        # check that the times aren't tiny 
        if timing_dict[key] < 1e-6 and key != 'convert':
            print(f"Timing for '{key}' is too small: {timing_dict[key]} seconds.")
            return 
    
    for key in excepted_spin_keys:
        if key not in timing_dict:
            if verbose:
                print(f"Missing expected spin key '{key}' in timing json.")
            return False
    
    if verbose:
        print("Timing json structure is valid.")
    return True
    

def validate_fuzzy_dict(fuzzy_json_loc, n_atoms=None, spin=True, verbose=False):
    """
    Basic check that the fuzzy json file has the expected structure.
    Check that it has the keys 'fuzzy', 'fuzzy_bonds', 'fuzzy_bcp', 'fuzzy_ncp'.
    """
    with open(fuzzy_json_loc, "r") as f:
        fuzzy_dict = json.load(f)

    expected_keys = ['grad_norm_rho_fuzzy', 'elf_fuzzy', 'becke_fuzzy_density', 'hirsh_fuzzy_density', 'mbis_fuzzy_density', 'laplacian_rho_fuzzy']
    if spin:
        expected_keys += ['hirsh_fuzzy_spin', 'becke_fuzzy_spin', 'mbis_fuzzy_spin']
    for key in expected_keys:
        if key not in fuzzy_dict:
            raise ValueError(f"Missing expected key '{key}' in fuzzy json.")
        if n_atoms is not None:
            if len(fuzzy_dict[key]) != n_atoms + 2:
                if verbose:
                    print(f"Number of fuzzy points ({len(fuzzy_dict[key])}) does not match expected ({n_atoms}).")
                return False
    if verbose: 
        print("Fuzzy json structure is valid.")
    return True


def validate_other_dict(other_dict_loc, verbose=False):
    """
    Basic check that the other json file has the expected structure.
    Check that it has the keys 'atoms', 'bonds', 'charges', and 'fuzzy'.
    """
    with open(other_dict_loc, "r") as f:
        other_dict = json.load(f)
    expected_keys = ['mpp_full', 'sdp_full', 'mpp_heavy', 'sdp_heavy', 'ESP_Volume', 'ESP_Surface_Density', 'ESP_Minimal_value', 'ESP_Maximal_value', 'ESP_Overall_surface_area', 'ESP_Positive_surface_area', 'ESP_Negative_surface_area', 'ESP_Overall_average_value', 'ESP_Positive_average_value', 'ESP_Negative_average_value', 'ESP_Overall_variance', 'ESP_Positive_variance', 'ESP_Negative_variance', 'ESP_Balance_of_charges', 'ESP_Product_of_sigma', 'ESP_Internal_charge_separation', 'ESP_Molecular_polarity_index', 'ESP_Nonpolar_surface_area', 'ESP_Polar_surface_area', 'ESP_Overall_skewness', 'ESP_Positive_skewness', 'ALIE_Volume', 'ALIE_Surface_Density', 'ALIE_Minimal_value', 'ALIE_Maximal_value', 'ALIE_Overall_surface_area', 'ALIE_Positive_surface_area', 'ALIE_Negative_surface_area', 'ALIE_Overall_skewness', 'ALIE_Positive_skewness']
    for key in expected_keys:
        if key not in other_dict:
            if verbose:
                print(f"Warning: Missing expected key '{key}' in other json. This may not be critical.")
            return False
        
    if verbose:
        print("Other json structure is valid.")   
    return True


def validate_charge_dict(charge_json_loc, n_atoms=None, verbose=False):
    """
    Basic check that the charge json file has the expected structure.
    Check that it has the keys 'mbis', 'adch', 'chelpg', 'becke',  'hirshfeld', 'cm5', 'bader', 'vdd'
    Check each one of these keys has a key "charge" with n_atoms entries.
    """
    with open(charge_json_loc, "r") as f:
        charge_dict = json.load(f)

    expected_keys = ['mbis', 'adch', 'chelpg', 'becke', 'hirshfeld', 'cm5', 'bader', 'vdd']
    for key in expected_keys:
        if key not in charge_dict:
            if verbose:
                print(f"Warning: Missing expected key '{key}' in charge json. ")
            return False

    for key in expected_keys:
        if 'charge' not in charge_dict[key]:
            if verbose:
                print(f"Missing 'charge' key in '{key}' of charge json.")
            return False
        if n_atoms is not None:
            if len(charge_dict[key]['charge']) != n_atoms:
                if verbose:
                    print(f"Number of charges in '{key}' ({len(charge_dict[key]['charge'])}) does not match expected ({n_atoms}).")
                return False
    if verbose:
        print("Charge json structure is valid.")
    return True


def validate_qtaim_dict(qtaim_json_loc, n_atoms=None, verbose=False):
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
    
    #dict_ncps = {qtaim_dict[key] for key in qtaim_dict if "_" not in key}
    dict_ncps = [qtaim_dict[key] for key in list(qtaim_dict.keys()) if "_" not in key]
    #dict_bcps = {qtaim_dict[key] for key in qtaim_dict if "_" in key}
    dict_bcps = [qtaim_dict[key] for key in list(qtaim_dict.keys()) if "_" in key]

    if n_atoms is not None:
        if len(dict_ncps) != n_atoms:
            if verbose:
                print(f"Number of nuclear critical points ({len(dict_ncps)}) does not match expected ({n_atoms}).")
            return False
    if verbose:
        print(f"Number of nuclear critical points: {len(dict_ncps)}")
        print(f"Number of bond critical points: {len(dict_bcps)}")
        print("QTAIM json structure is valid.")
    return True
    

def validation_checks(folder, verbose=False): 
    """
    Run all validation checks on the json files in the given folder.
    """
    # check that all the json files are present
    required_files = ["timings.json", "fuzzy_full.json", "other.json", "charge.json", "qtaim.json"]
    for file in required_files:
        if not os.path.exists(os.path.join(folder, file)):
            print(f"Missing required file: {file}")
            return False

    # check for a file ending with .inp
    inp_files = [f for f in os.listdir(folder) if f.endswith('.inp')]
    if not inp_files:
        if verbose:
            print("No .inp file found in the folder.")
        return False
    inp_file = inp_files[0]  # take the first .inp file found
    if verbose:
        print(f"Using input file \"{inp_file}\" for validation.")

    # gather n_atoms, spin, charge from the orca.inp file
    orca_inp_path = os.path.join(folder, inp_file) # might need to change this name 
    if not os.path.exists(orca_inp_path):
        if verbose:
            print(f"Missing orca.inp file at {orca_inp_path}.")
        return False
    dft_dict = dft_inp_to_dict(orca_inp_path, parse_charge_spin=True)
    #print(dft_dict)
    n_atoms = len(dft_dict["mol"])
    spin = dft_dict.get("spin", None)
    charge = dft_dict.get("charge", None)
    
    if verbose:
        print(f"n_atoms: {n_atoms}, spin: {spin}, charge: {charge}")
    
    if spin != 2:
        spin_tf = True
    else:
        spin_tf = False
        
    timing_json_loc = os.path.join(folder, "timings.json")
    fuzzy_json_loc = os.path.join(folder, "fuzzy_full.json")
    other_dict_loc = os.path.join(folder, "other.json")
    charge_json_loc = os.path.join(folder, "charge.json")
    qtaim_json_loc = os.path.join(folder, "qtaim.json")
    #bonding_json_loc = os.path.join(folder, "bonding.json")

    if not validate_timing_dict(timing_json_loc, verbose=verbose):
        return False
    if not validate_fuzzy_dict(fuzzy_json_loc, n_atoms=n_atoms, spin=spin_tf, verbose=verbose):
        return False
    if not validate_other_dict(other_dict_loc, verbose=verbose):
        return False
    if not validate_charge_dict(charge_json_loc, n_atoms=n_atoms, verbose=verbose):
        return False
    if not validate_qtaim_dict(qtaim_json_loc, n_atoms=n_atoms, verbose=verbose):
        return False

    if verbose:
        print("All validation checks passed.")
    return True
    