from qtaim_gen.source.core.parse_qtaim import get_qtaim_descs, merge_qtaim_inds


def parse_charge_doc(charge_out_txt):
    """
    Method to parse the charge out from multiwfn
    Takes:
        charge_out_txt: str, path to the charge out file
    Returns:
        charge_dict_overall: dict, dictionary with the charges from different methods
        atomic_dipole_dict_overall: dict, dictionary with the atomic dipoles from different methods
        dipole_info: dict, dictionary with the total dipoles from different methods
    """

    charge_key_1 = "Atom        Charge"
    charge_key_2 = "Center       Charge"
    charge_key_3 = "Final atomic charges:"
    atomic_dipole_key = "Atomic dipole moments (a.u.):"

    dipole_cm5_xyz_key = "X/Y/Z of dipole moment from CM5 charges"
    dipole_adch_xyz_key = "X/Y/Z of dipole moment from the charge (a.u.)"
    dipole_cm5_key = "Total dipole moment from CM5 charges"
    dipole_adch_key = "Total dipole from ADC charges (a.u.)"
    dipole_key = "Total dipole moment from atomic charges:"
    dipole_xyz_key = "X/Y/Z of dipole moment vector:"

    # charge_ordering = ["hirshfeld", "vdd", "becke", "adch", "chelpg", "mk", "cm5", "resp", "peoe", "mbis"]
    charge_ordering = ["hirshfeld", "vdd", "becke", "adch", "cm5", "mbis"]
    dipole_order = ["hirshfeld", "vdd", "becke", "hirshfeld"]
    atomic_dipole_order = ["becke", "adch"]

    charge_dict_overall = {}
    atomic_dipole_dict_overall = {}
    dipole_info = {"adch": {}, "hirshfeld": {}, "vdd": {}, "becke": {}}

    # iterate over lines of the file
    with open(charge_out_txt, "r") as f:

        charge_dict_index = 0
        dipole_index = 0
        atomic_dipole_index = 0
        trigger, trigger2 = False, False
        trigger_dipole = False
        for line in f:

            if line == "\n" or len(line) < 3:
                if trigger or trigger2:
                    trigger, trigger2 = False, False
                    charge_dict_overall[
                        charge_ordering[charge_dict_index]
                    ] = charge_dict
                    charge_dict_index += 1

                elif trigger_dipole:
                    trigger_dipole = False
                    atomic_dipole_dict_overall[
                        atomic_dipole_order[atomic_dipole_index]
                    ] = atomic_dipole_dict
                    atomic_dipole_index += 1

            if trigger or trigger2:
                if line.split()[0] == "Atom":
                    ind, element = line.split()[1].split("(")
                    if "):" in element:
                        element = element[:-2]
                    # print(ind, element)
                elif line.strip()[0].isnumeric():
                    ind, element = line.split()[0].split("(")
                    if "):" in element:
                        element = element[:-2]
                    # print(ind, element)
                value = float(line.split()[-1])
                charge_dict[ind + "_" + element] = value

            if trigger_dipole:
                # print(len(line) < 3)
                if line.strip().startswith("Atom"):
                    ind, element = line.split()[1].split("(")
                    shift = 0
                    if ")" in element:
                        element = element[:-1]
                        shift = -1
                    dipole = line.split()[5 + shift : 8 + shift]
                    # print(x, y, z)
                    # print(line.split())
                    atomic_dipole_dict[ind + "_" + element] = [float(i) for i in dipole]

            if dipole_cm5_key in line:
                float_dipole_cm5 = float(line.split()[-2])
                dipole_info["cm5"]["mag"] = float_dipole_cm5

            if dipole_cm5_xyz_key in line:
                str_nums = line.split()[7:-1]
                float_cm5_xyz = [float(num) for num in str_nums]
                dipole_info["cm5"]["xyz"] = float_cm5_xyz

            if dipole_adch_key in line:
                float_dipole_adch = float(line.split()[-3])
                dipole_info["adch"]["mag"] = float_dipole_adch

            if dipole_adch_xyz_key in line:
                str_nums = line.split()[8:]
                float_adch_xyz = [float(num) for num in str_nums]
                dipole_info["adch"]["xyz"] = float_adch_xyz

            if dipole_index < 3:
                if dipole_key in line:
                    float_dipole_temp = float(line.split()[-2])
                    dipole_info[dipole_order[dipole_index]]["mag"] = float_dipole_temp
                if dipole_xyz_key in line:
                    str_nums = line.split()[5:-1]
                    float_diple_xyz_temp = [float(num) for num in str_nums]
                    dipole_info[dipole_order[dipole_index]][
                        "xyz"
                    ] = float_diple_xyz_temp
                    dipole_index += 1

            if charge_key_1 in line:
                trigger = True
                charge_dict = {}

            if charge_key_2 in line:
                trigger2 = True
                charge_dict = {}

            if charge_key_3 in line:
                trigger2 = True
                charge_dict = {}

            if atomic_dipole_key in line:
                trigger_dipole = True
                atomic_dipole_dict = {}

    return charge_dict_overall, atomic_dipole_dict_overall, dipole_info


def parse_charge_base(charge_out_txt, corrected=False, dipole=True):
    """
    Method to parse the charge out from multiwfn
    Takes:
        charge_out_txt: str, path to the charge out file
        corrected: bool, if the charges are corrected
        dipole: bool, if the dipole is to be parsed
    Returns:
        charge_dict_overall: dict, dictionary with the charges from different methods
        atomic_dipole_dict_overall: dict, dictionary with the atomic dipoles from different methods
        dipole_info: dict, dictionary with the total dipoles from different methods
    """

    charge_key_3 = "Final atomic charges:"
    if corrected:
        dipole_xyz_key = "X/Y/Z of dipole moment from the charge (a.u.)"
        dipole_key = "Total dipole from ADC charges (a.u.)"

    else:
        dipole_key = "Total dipole moment from atomic charges:"
        dipole_xyz_key = "X/Y/Z of dipole moment vector:"

    charge_dict_overall = {}
    dipole_info = {}

    # iterate over lines of the file
    with open(charge_out_txt, "r") as f:

        trigger2 = False

        for line in f:
            if line == "\n" or len(line) < 3:
                if trigger2:
                    trigger2 = False
                    charge_dict_overall = charge_dict

            if trigger2:
                if line.split()[0] == "Atom":
                    ind, element = line.split()[1].split("(")
                    if "):" in element:
                        element = element[:-2]
                    # print(ind, element)
                elif line.strip()[0].isnumeric():
                    ind, element = line.split()[0].split("(")
                    if "):" in element:
                        element = element[:-2]
                    # print(ind, element)
                value = float(line.split()[-1])
                charge_dict[ind + "_" + element] = value

            if dipole_key in line:
                if corrected:
                    float_dipole_temp = float(line.split()[-3])
                else:
                    float_dipole_temp = float(line.split()[-2])
                dipole_info["mag"] = float_dipole_temp

            if dipole_xyz_key in line:
                if corrected:
                    str_nums = line.split()[-3:]
                else:
                    str_nums = line.split()[5:-1]
                float_diple_xyz_temp = [float(num) for num in str_nums]
                dipole_info["xyz"] = float_diple_xyz_temp

            if charge_key_3 in line:
                trigger2 = True
                charge_dict = {}

    # print("charge dict overall keys: ", charge_dict_overall.keys())
    # print("dipole info keys: ", dipole_info.keys())
    if dipole:
        return charge_dict_overall, dipole_info
    else:
        return charge_dict_overall


def parse_charge_doc_bader(charge_out_txt):
    """
    Method to parse the charge out from multiwfn bader charges.
    Takes: 
        charge_out_txt: str, path to the charge out file
    Returns: 
        charge_dict_overall: dict, dictionary with the charges from different methods
        spin_dict_overall: dict, dictionary with the spin from different methods
    """

    charge_key_3 = "The atomic charges"
    spin_key = "Atom       Basin"
    
    charge_dict = {}
    spin_dict = {}
    charge_dict_overall = {}
    spin_dict_overall = {}
    # iterate over lines of the file 
    with open(charge_out_txt, 'r') as f:
        trigger=False
        trigger_spin = False

        for line in f:
            if line == "\n" or len(line)<3:
                if trigger:    
                        trigger=False
                        charge_dict_overall = charge_dict

                elif trigger_spin:                 
                        trigger_spin = False
                        spin_dict_overall = spin_dict

            if trigger:
                if line.split()[0].isnumeric():
                    element = line.split()[1].split("(")[1]
                    ind = line.split()[0]

                    shift=0
                    if ")" in element:
                        element = element[:-1]
                        shift = -1                
                    charge_dict[ind + "_" + element] = float(line.split()[4+shift])
            
            if trigger_spin: 
                if line.split()[0].isnumeric():
                    
                    element = line.split()[1].split("(")[1]
                    ind = line.split()[0]

                    shift=0
                    if ")" in element:
                        element = element[:-1]
                        shift = -1
                    spin_dict[ind + "_" + element] = float(line.split()[4+shift])

            if charge_key_3 in line:
                trigger=True
                charge_dict = {}

            if spin_key in line:
                trigger_spin = True
                spin_dict = {}

    return charge_dict_overall, spin_dict_overall


def parse_charge_becke(charge_out_txt):
    """
    Method to parse the charge out from multiwfn
    Takes:
        charge_out_txt: str, path to the charge out file
        corrected: bool, if the charges are corrected
        dipole: bool, if the dipole is to be parsed
    Returns:
        charge_dict_overall: dict, dictionary with the charges from different methods
        atomic_dipole_dict_overall: dict, dictionary with the atomic dipoles from different methods
        dipole_info: dict, dictionary with the total dipoles from different methods
    """

    charge_key_3 = "Final atomic charges:"
    atomic_dipole_key = "Atomic dipole moments (a.u.):"
    dipole_xyz_key = "X/Y/Z of dipole moment from the charge (a.u.)"
    dipole_key = "Total dipole from ADC charges (a.u.)"

    charge_dict_overall = {}
    dipole_info = {}
    atomic_dipole_dict_overall = {}

    # iterate over lines of the file
    with open(charge_out_txt, "r") as f:

        trigger2 = False
        trigger_dipole = False
        atomic_dipole_dict = {}

        for line in f:
            if line == "\n" or len(line) < 3:
                if trigger2:
                    trigger2 = False
                    charge_dict_overall = charge_dict

                if trigger_dipole:
                    trigger_dipole = False
                    atomic_dipole_dict_overall = atomic_dipole_dict

            if trigger2:
                if line.split()[0] == "Atom":
                    ind, element = line.split()[1].split("(")
                    if "):" in element:
                        element = element[:-2]
                    # print(ind, element)
                elif line.strip()[0].isnumeric():
                    ind, element = line.split()[0].split("(")
                    if "):" in element:
                        element = element[:-2]
                    # print(ind, element)
                value = float(line.split()[-1])
                charge_dict[ind + "_" + element] = value

            if trigger_dipole:
                if line.strip().startswith("Atom"):
                    ind, element = line.split()[1].split("(")
                    shift = 0
                    if ")" in element:
                        element = element[:-1]
                        shift = -1
                    dipole = line.split()[5 + shift : 8 + shift]
                    atomic_dipole_dict[ind + "_" + element] = [float(i) for i in dipole]

            if dipole_key in line:
                float_dipole_temp = float(line.split()[-3])
                dipole_info["mag"] = float_dipole_temp

            if dipole_xyz_key in line:
                str_nums = line.split()[-3:]
                float_diple_xyz_temp = [float(num) for num in str_nums]
                dipole_info["xyz"] = float_diple_xyz_temp

            if charge_key_3 in line:
                trigger2 = True
                charge_dict = {}

            if atomic_dipole_key in line:
                trigger_dipole = True
                atomic_dipole_dict = {}

    # print("charge dict overall keys: ", charge_dict_overall.keys())
    # print("dipole info keys: ", dipole_info.keys())
    # print("atomic_dipole_dict_overall keys: ", atomic_dipole_dict_overall.keys())

    return charge_dict_overall, atomic_dipole_dict_overall, dipole_info


def parse_charge_doc_adch(charge_out_txt):
    """
    Method to parse the charge out from multiwfn on acdh charges.
    Takes:
        charge_out_txt: str, path to the charge out file
    Returns:
        charge_dict_overall: dict, dictionary with the charges from different methods
        atomic_dipole_dict_overall: dict, dictionary with the atomic dipoles from different methods
        dipole_info: dict, dictionary with the total dipoles from different methods
    """

    charge_key_3 = "Final atomic charges:"
    atomic_dipole_key = "Atomic dipole moments (a.u.):"
    dipole_adch_xyz_key = "X/Y/Z of dipole moment from the charge (a.u.)"
    dipole_adch_key = "Total dipole from ADC charges (a.u.)"

    charge_dict_overall = {}
    atomic_dipole_dict_overall = {}
    dipole_info = {}

    # iterate over lines of the file
    with open(charge_out_txt, "r") as f:
        trigger = False
        trigger_dipole = False

        for line in f:
            if line == "\n" or len(line) < 3:
                if trigger:
                    trigger = False
                    charge_dict_overall = charge_dict

                elif trigger_dipole:
                    trigger_dipole = False
                    atomic_dipole_dict_overall = atomic_dipole_dict

            if trigger:
                if line.split()[0] == "Atom":
                    ind, element = line.split()[1].split("(")
                    if "):" in element:
                        element = element[:-2]
                    # print(ind, element)
                elif line.strip()[0].isnumeric():
                    ind, element = line.split()[0].split("(")
                    if "):" in element:
                        element = element[:-2]
                    # print(ind, element)
                value = float(line.split()[-1])
                charge_dict[ind + "_" + element] = value

            if trigger_dipole:
                if line.strip().startswith("Atom"):
                    ind, element = line.split()[1].split("(")
                    shift = 0
                    if ")" in element:
                        element = element[:-1]
                        shift = -1
                    dipole = line.split()[5 + shift : 8 + shift]

                    atomic_dipole_dict[ind + "_" + element] = [float(i) for i in dipole]

            if dipole_adch_key in line:
                float_dipole_adch = float(line.split()[-3])
                dipole_info["mag"] = float_dipole_adch

            if dipole_adch_xyz_key in line:
                str_nums = line.split()[8:]
                float_adch_xyz = [float(num) for num in str_nums]
                dipole_info["xyz"] = float_adch_xyz

            if charge_key_3 in line:
                trigger = True
                charge_dict = {}

            if atomic_dipole_key in line:
                trigger_dipole = True
                atomic_dipole_dict = {}

    return charge_dict_overall, atomic_dipole_dict_overall, dipole_info


def parse_bond_order_doc(bond_order_txt):
    """
    Method to parse the bond order out from multiwfn
    Takes:
        bond_out_txt: str, path to the bond out file
    Returns:
        bond_dict: dict, dictionary with the bond orders
    """

    fuzzy_trigger = "The total bond order >=  0.050000"
    laplace_trigger = "The bond orders >=  0.050000"
    ibsi_trigger = 'Note: "Dist"'
    ibsi_detrigger = "---------- Intrinsic bond strength index (IBSI) ----------"
    fuzzy_bool, laplace_bool, ibsi_bool = False, False, False
    fuzzy_bond_dict, laplace_bond_dict, ibsi_bond_dict = {}, {}, {}

    with open(bond_order_txt, "r") as f:
        for line in f:
            if fuzzy_bool:
                if line.strip() == "":
                    fuzzy_bool = False
                else:
                    split_list = line.split()
                    a, b, order = (
                        split_list[2].replace("(", "_"),
                        split_list[4].replace("(", "_"),
                        float(split_list[-1]),
                    )
                    # fuzzy_bond_dict.append((a, b, order))
                    fuzzy_bond_dict["{}_to_{}".format(a, b)] = order

            if laplace_bool:
                if line.strip() == "":
                    laplace_bool = False
                else:
                    split_list = line.split()
                    a, b, order = (
                        split_list[2].replace("(", "_"),
                        split_list[4].replace("(", "_"),
                        float(split_list[-1]),
                    )
                    laplace_bond_dict["{}_to_{}".format(a, b)] = order

            if ibsi_bool:
                if ibsi_detrigger in line:
                    ibsi_bool = False
                else:
                    if line.strip() == "":
                        continue
                    split_list = line.split()
                    a, b, ibsi = (
                        split_list[0].replace("(", "_"),
                        split_list[2].replace("(", "_"),
                        split_list[-1],
                    )
                    ibsi_bond_dict["{}_to_{}".format(a, b)] = float(ibsi)

            if fuzzy_trigger in line:
                fuzzy_bool = True

            if laplace_trigger in line:
                laplace_bool = True

            if ibsi_trigger in line:
                if ibsi_bool:
                    # stop parsing the file
                    break
                ibsi_bool = True

    bond_dict = {
        "fuzzy": fuzzy_bond_dict,
        "laplace": laplace_bond_dict,
        "ibsi": ibsi_bond_dict,
    }

    return bond_dict


def parse_bond_order_laplace(bond_order_txt):
    """
    Method to parse the laplace bond order out from multiwfn
    Takes:
        bond_out_txt: str, path to the bond out file
    Returns:
        bond_dict: dict, dictionary with the bond orders
    """

    laplace_trigger = "The bond orders >=  0.050000"
    laplace_bool = False
    laplace_bond_dict = {}

    with open(bond_order_txt, "r") as f:
        for line in f:

            if laplace_bool:
                if line.strip() == "":
                    laplace_bool = False
                else:
                    split_list = line.split()
                    a, b, order = (
                        split_list[2].replace("(", "_"),
                        split_list[4].replace("(", "_"),
                        float(split_list[-1]),
                    )
                    laplace_bond_dict["{}_to_{}".format(a, b)] = order

            if laplace_trigger in line:
                laplace_bool = True

    return laplace_bond_dict


def parse_bond_order_ibsi(bond_order_txt):
    """
    Method to parse the ibsi bond order out from multiwfn
    Takes:
        bond_out_txt: str, path to the bond out file
    Returns:
        bond_dict: dict, dictionary with the bond orders
    """

    ibsi_trigger = 'Note: "Dist"'
    ibsi_detrigger = "---------- Intrinsic bond strength index (IBSI) ----------"
    ibsi_bool = False
    ibsi_bond_dict = {}

    with open(bond_order_txt, "r") as f:
        for line in f:

            if ibsi_bool:
                if ibsi_detrigger in line:
                    ibsi_bool = False
                else:
                    if line.strip() == "":
                        continue
                    split_list = line.split()
                    a, b, ibsi = (
                        split_list[0].replace("(", "_"),
                        split_list[2].replace("(", "_"),
                        split_list[-1],
                    )
                    ibsi_bond_dict["{}_to_{}".format(a, b)] = float(ibsi)

            if ibsi_trigger in line:
                if ibsi_bool:
                    # stop parsing the file
                    break
                ibsi_bool = True

    return ibsi_bond_dict


def parse_bond_order_fuzzy(bond_order_txt):
    """
    Method to parse the fuzzy bond order out from multiwfn
    Takes:
        bond_out_txt: str, path to the bond out file
    Returns:
        bond_dict: dict, dictionary with the bond orders
    """

    fuzzy_trigger = "The total bond order >=  0.050000"
    fuzzy_bool = False
    fuzzy_bond_dict = {}

    with open(bond_order_txt, "r") as f:
        for line in f:
            if fuzzy_bool:
                if line.strip() == "":
                    fuzzy_bool = False
                else:
                    split_list = line.split()
                    a, b, order = (
                        split_list[2].replace("(", "_"),
                        split_list[4].replace("(", "_"),
                        float(split_list[-1]),
                    )
                    # fuzzy_bond_dict.append((a, b, order))
                    fuzzy_bond_dict["{}_to_{}".format(a, b)] = order

            if fuzzy_trigger in line:
                fuzzy_bool = True

    return fuzzy_bond_dict


def parse_other_doc(other_txt):
    """
    Method to parse other info from multiwfn
    Takes:
        other_info_txt: str, path to the bond out file
    Returns:
        other_info_dict: dict, dictionary with the bond orders
    """
    dict_other = {}
    bool_trigger_surface_summary = False
    ind_planarity = 0
    ind_surface_prefix = "ESP"

    trigger_mpp = "Molecular planarity parameter (MPP) is"
    trigger_sdp = "Span of deviation from plane (SDP) is"
    trigger_surface_summary = (
        "================= Summary of surface analysis ================="
    )
    trigger_surface_summary_end = "Surface analysis finished!"

    trigger_dict_surface = {
        "Volume:": {"name": "Volume", "data_ind": 1},
        "Estimated density according to mass and volume (M/V):": {
            "name": "Surface_Density",
            "data_ind": -2,
        },
        "Minimal value:": {
            "name": ["Minimal_value", "Maximal_value"],
            "data_ind": [2, -2],
        },
        "Overall surface area:": {"name": "Overall_surface_area", "data_ind": 3},
        "Positive surface area:": {"name": "Positive_surface_area", "data_ind": 3},
        "Negative surface area:": {"name": "Negative_surface_area", "data_ind": 3},
        "Overall average value:": {"name": "Overall_average_value", "data_ind": 3},
        "Positive average value:": {"name": "Positive_average_value", "data_ind": 3},
        "Negative average value:": {"name": "Negative_average_value", "data_ind": 3},
        "Overall variance (sigma^2_tot):": {"name": "Overall_variance", "data_ind": 3},
        "Positive variance:": {"name": "Positive_variance", "data_ind": 2},
        "Negative variance:": {"name": "Negative_variance", "data_ind": 2},
        "Balance of charges (nu):": {"name": "Balance_of_charges", "data_ind": 4},
        "Product of sigma^2_tot and nu:": {"name": "Product_of_sigma", "data_ind": 5},
        "Internal charge separation (Pi):": {
            "name": "Internal_charge_separation",
            "data_ind": 4,
        },
        "Molecular polarity index (MPI):": {
            "name": "Molecular_polarity_index",
            "data_ind": 4,
        },
        "Nonpolar surface area (|ESP| <= 10 kcal/mol):": {
            "name": "Nonpolar_surface_area",
            "data_ind": 7,
        },
        "Polar surface area (|ESP| > 10 kcal/mol):": {
            "name": "Polar_surface_area",
            "data_ind": 7,
        },
        "Overall skewness:": {"name": "Overall_skewness", "data_ind": -1},
        "Positive skewness:": {"name": "Positive_skewness", "data_ind": -1},
        "Negative skewness:": {"name": "Negative_skewness", "data_ind": -1},
    }

    surface_trigger_keys = trigger_dict_surface.keys()

    with open(other_txt, "r") as f:
        for line in f:
            if trigger_mpp in line:
                if ind_planarity == 0:
                    dict_other["mpp_full"] = float(line.split()[-2])
                else:
                    dict_other["mpp_heavy"] = float(line.split()[-2])
            if trigger_sdp in line:
                if ind_planarity == 0:
                    dict_other["sdp_full"] = float(line.split()[-2])
                    ind_planarity += 1
                else:
                    dict_other["sdp_heavy"] = float(line.split()[-2])

            if trigger_surface_summary in line:
                bool_trigger_surface_summary = True

            if bool_trigger_surface_summary:
                if trigger_surface_summary_end in line:
                    bool_trigger_surface_summary = False
                    ind_surface_prefix = "ALIE"

            if bool_trigger_surface_summary:
                for key in surface_trigger_keys:
                    if key in line:
                        name = trigger_dict_surface[key]["name"]
                        data_ind = trigger_dict_surface[key]["data_ind"]
                        if isinstance(name, list):
                            for i, n in enumerate(name):
                                dict_other[ind_surface_prefix + "_" + n] = float(
                                    line.split()[data_ind[i]]
                                )
                        else:
                            dict_other[ind_surface_prefix + "_" + name] = float(
                                line.split()[data_ind]
                            )

    return dict_other


def parse_fuzzy_doc(fuzzy_loc):
    """
    Function to parse fuzzy doc.
    Takes:
        fuzzy_loc(str): location of fuzzy analysis output from multiwfn
    Returns:
        ret_dict(dictionary): dictionary with real-space values and localization indices
    """

    ret_dict = {"real_space": {}}
    data_order = ["density", "grad_norm", "laplacian", "elf"]
    trigger_data_block = "Atomic space"
    trigger_local_block = " Localization index:"

    data_ind = 0
    trigger_bool_real = False
    trigger_bool_local_block = False

    # iterate over lines of the file and print line if trigger is found
    with open(fuzzy_loc, "r") as f:
        for line in f:

            if trigger_bool_real:
                if line == "\n" or len(line) < 3:
                    trigger_bool_real = False
                    ret_dict["real_space"][data_order[data_ind]] = dict_data_temp
                    data_ind += 1
                else:
                    line_split = line.split()
                    if line_split[0] == "Summing":
                        if line_split[2] == "absolute":
                            dict_data_temp["abs_sum"] = float(line_split[-1])
                        else:
                            dict_data_temp["sum"] = float(line_split[-1])
                    else:
                        name = line_split[0].replace("(", "_")
                        dict_data_temp[name] = float(line_split[2])

            if trigger_bool_local_block:
                if line == "\n" or len(line) < 3:
                    trigger_bool_local_block = False
                    ret_dict["localization"] = local_dict_temp
                else:
                    line_split = line.split()
                    # filter items that have a colon in the strings
                    line_split = [i for i in line_split if ":" not in i]
                    atoms_inds = [i.replace("(", "_") for i in line_split if "(" in i]
                    values = [float(i) for i in line_split if "(" not in i]
                    for i, atom in enumerate(atoms_inds):
                        local_dict_temp[atom] = values[i]

            if trigger_data_block in line:
                trigger_bool_real = True
                dict_data_temp = {}

            if trigger_local_block in line:
                trigger_bool_local_block = True
                local_dict_temp = {}

    return ret_dict


def parse_qtaim(cprop_file, inp_loc):
    """
    Function to parse qtaim output from multiwfn
    Takes:
        cprop_file(str): location of qtaim output from multiwfn
        inp_loc(str): location of the input file
    Returns:
        qtaim_dict(dictionary): dictionary with qtaim properties
    """
    qtaim_descs = get_qtaim_descs(cprop_file)

    ret_dict = merge_qtaim_inds(
        dft_inp_file=inp_loc,
        qtaim_descs=qtaim_descs,
        bond_list=None,
        define_bonds="qtaim",
        margin=1.0,
    )

    # convert tuple keys to strings
    clean_dict = {}
    for key in ret_dict.keys():
        if isinstance(key, tuple):
            new_key = str(key[0]) + "_" + str(key[1])
            clean_dict[new_key] = ret_dict[key]
        else:
            clean_dict[key] = ret_dict[key]

    return clean_dict
