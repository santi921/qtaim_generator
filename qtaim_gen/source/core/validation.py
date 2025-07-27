import os
import json 
from qtaim_gen.source.core.parse_qtaim import dft_inp_to_dict

def validate_timing_dict(timing_json_loc): 
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
        'becke_fuzzy_spin',
        'mbis_fuzzy_density',
        'mbis_fuzzy_spin',
        'hirsh_fuzzy_density',
        'hirsh_fuzzy_spin',
        'grad_norm_rho_fuzzy',
        'laplacian_rho_fuzzy',
        'elf_fuzzy']
    for key in expected_keys:
        if key not in timing_dict:
            raise ValueError(f"Missing expected key '{key}' in timing json.")
        # check that the times aren't tiny 
        
        if timing_dict[key] < 1e-6 and key != 'convert':
            print(f"Timing for '{key}' is too small: {timing_dict[key]} seconds.")
            return False
    print("Timing json structure is valid.")
    return True
    

def validate_fuzzy_dict(fuzzy_json_loc, n_atoms=None, spin=True):
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
                print(f"Number of fuzzy points ({len(fuzzy_dict[key])}) does not match expected ({n_atoms}).")
                return False
    print("Fuzzy json structure is valid.")
    return True

def validate_other_dict(other_dict_loc):
    """
    Basic check that the other json file has the expected structure.
    Check that it has the keys 'atoms', 'bonds', 'charges', and 'fuzzy'.
    """
    with open(other_dict_loc, "r") as f:
        other_dict = json.load(f)
    expected_keys = ['mpp_full', 'sdp_full', 'mpp_heavy', 'sdp_heavy', 'ESP_Volume', 'ESP_Surface_Density', 'ESP_Minimal_value', 'ESP_Maximal_value', 'ESP_Overall_surface_area', 'ESP_Positive_surface_area', 'ESP_Negative_surface_area', 'ESP_Overall_average_value', 'ESP_Positive_average_value', 'ESP_Negative_average_value', 'ESP_Overall_variance', 'ESP_Positive_variance', 'ESP_Negative_variance', 'ESP_Balance_of_charges', 'ESP_Product_of_sigma', 'ESP_Internal_charge_separation', 'ESP_Molecular_polarity_index', 'ESP_Nonpolar_surface_area', 'ESP_Polar_surface_area', 'ESP_Overall_skewness', 'ESP_Positive_skewness', 'ALIE_Volume', 'ALIE_Surface_Density', 'ALIE_Minimal_value', 'ALIE_Maximal_value', 'ALIE_Overall_surface_area', 'ALIE_Positive_surface_area', 'ALIE_Negative_surface_area', 'ALIE_Overall_skewness', 'ALIE_Positive_skewness']
    for key in expected_keys:
        if key not in other_dict:
            print(f"Warning: Missing expected key '{key}' in other json. This may not be critical.")
            return False
        
            
    print("Other json structure is valid.")
    return True


def validate_charge_dict(charge_json_loc, n_atoms=None):
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
            print(f"Missing expected key '{key}' in charge json.")
            return False

    for key in expected_keys:
        if 'charge' not in charge_dict[key]:
            print(f"Missing 'charge' key in '{key}' of charge json.")
            return False
        if n_atoms is not None:
            if len(charge_dict[key]['charge']) != n_atoms:
                print(f"Number of charges in '{key}' ({len(charge_dict[key]['charge'])}) does not match expected ({n_atoms}).")
                return False
    print("Charge json structure is valid.")
    return True


def validate_qtaim_dict(qtaim_json_loc, n_atoms=None, harsh_check=False):
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
        print("QTAIM json file is empty.")
        return False
    
    #dict_ncps = {qtaim_dict[key] for key in qtaim_dict if "_" not in key}
    dict_ncps = [qtaim_dict[key] for key in list(qtaim_dict.keys()) if "_" not in key]
    #dict_bcps = {qtaim_dict[key] for key in qtaim_dict if "_" in key}
    dict_bcps = [qtaim_dict[key] for key in list(qtaim_dict.keys()) if "_" in key]

    if n_atoms is not None:
        if len(dict_ncps) != n_atoms:
            print(f"Number of non-bonded critical points ({len(dict_ncps)}) does not match expected ({n_atoms}).")
            if harsh_check:
                return False
    print("QTAIM json structure is valid.")
    return True
    


def validation_checks(folder): 
    """
    Run all validation checks on the json files in the given folder.
    """
    # check that all the json files are present
    required_files = ["timings.json", "fuzzy_full.json", "other.json", "charge.json", "qtaim.json", "orca.inp"]
    for file in required_files:
        if not os.path.exists(os.path.join(folder, file)):
            print(f"Missing required file: {file}")
            return False

    # gather n_atoms, spin, charge from the orca.inp file
    orca_inp_path = os.path.join(folder, "orca.inp")
    if not os.path.exists(orca_inp_path):
        print("Missing orca.inp file.")
        return False
    dft_dict = dft_inp_to_dict(orca_inp_path, parse_charge_spin=True)
    #print(dft_dict)
    n_atoms = len(dft_dict["mol"])
    spin = dft_dict.get("spin", None)
    charge = dft_dict.get("charge", None)
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

    if not validate_timing_dict(timing_json_loc):
        return False
    if not validate_fuzzy_dict(fuzzy_json_loc, n_atoms=n_atoms, spin=spin_tf):
        return False
    if not validate_other_dict(other_dict_loc):
        return False
    if not validate_charge_dict(charge_json_loc, n_atoms=n_atoms):
        return False
    if not validate_qtaim_dict(qtaim_json_loc, n_atoms=n_atoms, harsh_check=True):
        return False
    #if not validate_qtaim_dict(bonding_json_loc, n_atoms=n_atoms, harsh_check=False):
    #    return False
    
    print("All validation checks passed.")
    return True
    