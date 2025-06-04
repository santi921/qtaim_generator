"""
Helpers to generate the data strings for the multiwfn program
"""

def charge_data():
    # old string with every charge method possible with a gbw
    # string_ret = "7\n1\n1\nn\n2\n1\nn\n10\n0\nn\n11\n1\nn\n12\n1\nn\n0\n13\n1\nn\n0\n16\n1\nn\n18\n1\nn\n0\n19\nn\n20\n1\nn\n0\nq\n"
    string_ret = (
        "7\n1\n1\nn\n2\n1\nn\n10\n0\nn\n11\n1\nn\n13\n1\nn\n0\n20\n1\nn\n0\nq\n"
    )
    return string_ret


def charge_data_dict():
    # cut resp and chelpg and peoe
    string_dict = {}
    string_dict["bader"] = "17\n1\n1\n2\n7\n1\n1\n7\n1\n5\n-10\nq\n"
    string_dict["hirshfeld"] = "7\n1\n1\nn\n0\nq\n"
    string_dict["vdd"] = "7\n2\n1\nn\n0\nq\n"
    string_dict["becke"] = "7\n10\n0\nn\n0\nq\n"
    string_dict["adch"] = "7\n11\n1\nn\n0\nq\n"
    string_dict["mbis"] = "7\n20\n1\nn\n0\nq\n"
    string_dict["cm5"] = "7\n16\n1\nn\n0\nq\n"  # might cut later
    string_dict["chelpg"] = "7\n12\n1\nn\n0\n0\nq\n"  # might cut later

    return string_dict


def bond_order_data():  # separate out into dictionary
    string_ret = "9\n7\nn\n8\nn\n10\n1\n1\n0\n0\nq\n"
    return string_ret


def bond_order_dict():
    string_dict = {}
    string_dict["fuzzy_bond"] = "9\n7\nn\n0\nq\n"
    # string_dict["laplacian"] = "9\n8\nn\n0\nq\n" expensive
    string_dict["ibsi_bond"] = "9\n10\n1\n1\n0\n0\nq\n"
    return string_dict


def fuzzy_data():  # not being used so don't need to move to dictionary
    string_ret = "15\n1\n1\n1\n2\n1\n3\n1\n9\n4\nn\n0\nq\n"
    #string_ret = "15\n1\n1\nn1\n2\n1\n3\n1\n9\n4\nn\n0\nq\n"
    string_dict = {}
    string_dict["becke_fuzzy_density"] = "15\n1\n1\n0\nq\n"
    string_dict["grad_norm_rho_fuzzy"] = "15\n1\n2\n0\nq\n"
    string_dict["laplacian_rho_fuzzy"] = "15\n1\n3\n0\nq\n"
    string_dict["elf_fuzzy"] = "15\n1\n9\n0\nq\n"
    string_dict["becke_fuzzy_spin"] = "15\n1\n5\n0\nq\n"
    string_dict["mbis_fuzzy_spin"] = "15\n-1\n5\n1\n1\n5\n0\nq\n" 
    #string_dict["becke_fuzzy_density"] = "15\n1\n1\n0\nq\n"
    string_dict["mbis_fuzzy_density"] = "15\n-1\n5\n1\n1\n1\n0\nq\n" 
    #return string_ret
    return string_dict 


def other_data():  # potentially rework?
    string_ret = "26\n3\na\nn\n3\nh\nn\n8\n0\n0\n12\n0\n-1\n2\n2\n0\n-1\n-1\nq\n"
    return string_ret


def qtaim_data(exhaustive=False):  # can work in one go
    if exhaustive:
        string_ret = "2\n2\n3\n4\n5\n6\n-1\n-9\n8\n7\n0\n-10\nq\n"
    else:  # skips spherical search around atoms
        string_ret = "2\n2\n3\n4\n5\n8\n7\n0\n-10\nq\n"
    return string_ret
