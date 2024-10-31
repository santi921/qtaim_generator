from qtaim_gen.source.core.parse_qtaim import (
    parse_cp,
    get_qtaim_descs,
    dft_inp_to_dict,
    only_atom_cps,
    find_cp_map,
    merge_qtaim_inds,
    orca_inp_to_dict
)

from qtaim_gen.source.core.parse_critic import parse_critic2


def get_full_qtaim():
    # read full qtaim file from multiwfn out
    qtaim_file = "./test_files/test_prop_full.txt"
    with open(qtaim_file, "r") as f:
        qtaim = f.readlines()
    return qtaim


def get_critical_point_atom():
    # read in atom critical point from file
    cp_file = "./test_files/test_atom_cp.txt"
    with open(cp_file, "r") as f:
        cp = f.readlines()
    return cp


def get_critical_point_bond():
    # read in bond critical point from file
    cp_file = "./test_files/test_bond_cp.txt"
    with open(cp_file, "r") as f:
        cp = f.readlines()
    return cp


def get_critical_point_ring():
    # read in ring critical point from file
    cp_file = "./test_files/test_ring_cp.txt"
    with open(cp_file, "r") as f:
        cp = f.readlines()
    return cp


def test_parse_cp_test():
    # check that the critical point is parsed correctly from txt

    # load atom critical point
    cp = get_critical_point_atom()
    cp_dict = parse_cp(cp, verbose=True)
    # assert 0 is an integer
    assert cp_dict[0].split("_")[0].isdigit(), "cp index not an integer"
    assert cp_dict[1] != {}, "bond cp wrongly parsed"

    cp_bond = get_critical_point_bond()
    # print(cp_bond)
    cp_bond_dict = parse_cp(cp_bond, verbose=True)
    assert cp_bond_dict[0].split("_")[1] == "bond", "bond cp wrongly parsed"
    # assert cp_bond_dict[1] isnt an empty dict
    assert cp_bond_dict[1] != {}, "bond cp wrongly parsed"

    cp_ring = get_critical_point_ring()
    cp_ring_dict = parse_cp(cp_ring, verbose=True)
    assert cp_ring_dict[0] == "ring", "ring cp wrongly parsed"
    assert cp_ring_dict[1] == {}, "ring cp wrongly parsed"


def test_parse_full_cp():
    dict_qtaim = get_qtaim_descs("./test_files/test_prop_full.txt")
    # print(dict_qtaim)
    count_atoms = 0
    count_bonds = 0
    list_keys = list(dict_qtaim.keys())

    for key in list_keys:
        if "bond" in key:
            count_bonds += 1
        else:
            string_int = key.split("_")[0]
            if string_int.isdigit():
                count_atoms += 1
            assert string_int.isdigit(), "atom index not an integer"

    print("number of atoms: ", count_atoms)
    print("number of bonds: ", count_bonds)
    atom_cp_test = dict_qtaim["14_H"]
    bond_cp_test = dict_qtaim["23_bond"]

    atom_cp_keys_check = [
        "cp_num",
        "element",
        "number",
        "pos_ang",
        "density_alpha",
        "density_beta",
        "spin_density",
        "lol",
        "energy_density",
        "Lagrangian_K",
        "Hamiltonian_K",
        "lap_e_density",
        "e_loc_func",
        "ave_loc_ion_E",
        "delta_g_promolecular",
        "delta_g_hirsh",
        "esp_nuc",
        "esp_e",
        "esp_total",
        "grad_norm",
        "lap_norm",
        "eig_hess",
        "det_hessian",
        "ellip_e_dens",
        "eta",
    ]
    bond_cp_keys_check = [
        "cp_num",
        "pos_ang",
        "density_alpha",
        "density_beta",
        "spin_density",
        "lol",
        "energy_density",
        "Lagrangian_K",
        "Hamiltonian_K",
        "lap_e_density",
        "e_loc_func",
        "ave_loc_ion_E",
        "delta_g_promolecular",
        "delta_g_hirsh",
        "esp_nuc",
        "esp_e",
        "esp_total",
        "grad_norm",
        "lap_norm",
        "eig_hess",
        "det_hessian",
        "ellip_e_dens",
        "eta",
    ]
    for i in atom_cp_keys_check:
        assert i in atom_cp_test.keys(), "key not in atom CP: {}".format(i)
    for i in bond_cp_keys_check:
        assert i in bond_cp_test.keys(), "key not in bond CP: {}".format(i)
    assert count_atoms == 20, "wrong number of atom CP"
    assert count_bonds == 21, "wrong number of bond CP"


def test_parse_dft_inp():
    dict_dft = dft_inp_to_dict("./test_files/test_dft.in")
    # assert number of keys is 19
    assert len(dict_dft.keys()) == 19, "wrong number of keys"
    for i in dict_dft.values():
        assert "element" in i.keys(), "element not in dict"
        assert "pos" in i.keys(), "xyz not in dict"
        assert len(i["pos"]) == 3, "xyz not length 3"

def test_parse_dft_inp_orca():
    dict_dft = orca_inp_to_dict("./test_files/orca/orca.inp")
    assert len(dict_dft.keys()) == 87, "wrong number of keys"
    for i in dict_dft.values():
        assert "element" in i.keys(), "element not in dict"
        assert "pos" in i.keys(), "xyz not in dict"
        assert len(i["pos"]) == 3, "xyz not length 3"

    dict_dft = orca_inp_to_dict("./test_files/orca/orca5.inp")
    assert len(dict_dft.keys()) == 118, "wrong number of keys"
    
    for i in dict_dft.values():
        assert "element" in i.keys(), "element not in dict"
        assert "pos" in i.keys(), "xyz not in dict"
        assert len(i["pos"]) == 3, "xyz not length 3"


test_parse_dft_inp_orca()

def test_only_atom_cps():
    qtaim_dict = get_qtaim_descs("./test_files/CPprop.txt")
    ret_dict, ret_dict_bonds = only_atom_cps(qtaim_dict)
    assert len(ret_dict.keys()) == 22, "wrong number of atom CP"
    assert len(ret_dict_bonds.keys()) == 23, "wrong number of bond CP"
    for i in ret_dict.values():
        assert "element" in i.keys(), "element not in dict"
        assert "pos_ang" in i.keys(), "xyz not in dict"
    # print(ret_dict_bonds.keys())


def test_find_cp_map():
    qtaim_dict = get_qtaim_descs("./test_files/CPprop_w_bond_paths.txt")
    atom_dict, atom_dict_bonds = only_atom_cps(qtaim_dict)
    dict_dft = dft_inp_to_dict("./test_files/input_bond_paths.in")
    ret_dict, qtaim_to_dft_map, missing_atoms = find_cp_map(
        dict_dft, atom_dict, margin=0.5
    )

    key_of_keys = list([i["key"] for i in qtaim_to_dft_map.values()])
    assert len(key_of_keys) == 13, "should have 13 keys"
    # assert -1 in key_of_keys, "should have -1 in keys"
    # assert len(missing_atoms) == 1, "should have one atom missing"


def test_merge_qtaim_inds():
    bonds = [
        [0, 9],
        [0, 1],
        [0, 10],
        [1, 2],
        [1, 3],
        [3, 4],
        [3, 11],
        [4, 5],
        [4, 8],
        [5, 12],
        [5, 6],
        [6, 13],
        [6, 7],
        [7, 8],
        [14, 15],
        [14, 16],
    ]
    dict_qtaim = get_qtaim_descs("./test_files/CPprop_bond_map.txt")
    cp_dict = merge_qtaim_inds(
        qtaim_descs=dict_qtaim,
        bond_list=bonds,
        define_bonds="distance",
        dft_inp_file="./test_files/input_bond_map.in",
        inp_type="xyz",
    )
    # count number of cp_dict keys that are integers
    cp_dict_keys = list(cp_dict.keys())
    count_tuple, count_int = 0, 0
    for i in cp_dict_keys:
        if type(i) is int:
            count_int += 1
        elif type(i) == tuple:
            count_tuple += 1
    print("number of atoms: ", count_int)
    print("number of bonds: ", count_tuple)
    assert count_int == 17, "wrong number of atom CP"
    assert count_tuple == 16, "wrong number of bond CP"


def test_bond_cp_via_qtaim():
    qtaim_dict = get_qtaim_descs("./test_files/CPprop_w_bond_paths.txt")
    # assert that any key with "bond" in it has a path under the key connected_bond_paths

    for key in qtaim_dict.keys():
        if "bond" in key:
            assert (
                "connected_bond_paths" in qtaim_dict[key].keys()
            ), "no path in bond CP"


def test_bond_cp_via_qtaim_bond_defns():
    qtaim_dict = get_qtaim_descs("./test_files/CPprop_w_bond_paths.txt")
    dft_dict = dft_inp_to_dict("./test_files/input_bond_paths.in")
    atom_cp, bond_cps = only_atom_cps(qtaim_dict)
    # [print(k, v["cp_num"]) for k, v in atom_cp.items()]
    atom_cp, qtaim_to_dft, missing_atoms = find_cp_map(dft_dict, atom_cp, margin=0.5)
    bond_cps_qtaim = {}
    # print(qtaim_to_dft)
    for k, v in bond_cps.items():
        bond_list_unsorted = v["connected_bond_paths"]
        # print(bond_list_unsorted)

        bond_list_unsorted = [
            int(qtaim_to_dft[i - 1]["key"].split("_")[0]) - 1
            for i in bond_list_unsorted
        ]
        bond_list_unsorted = sorted(bond_list_unsorted)
        # print(bond_list_unsorted)
        bond_cps_qtaim[tuple(bond_list_unsorted)] = v

    count_tuple = 0
    for i in bond_cps_qtaim:
        if type(i) == tuple:
            count_tuple += 1

    bond_list_correct = [
        (6, 7),
        (5, 6),
        (6, 12),
        (2, 6),
        (3, 5),
        (1, 2),
        (1, 3),
        (4, 11),
        (3, 4),
        (0, 1),
        (0, 8),
        (0, 10),
        (0, 9),
    ]
    for i in bond_cps_qtaim.keys():
        assert i in bond_list_correct, "{}, bond not in correct list".format(i)
    assert count_tuple == 13, "wrong number of bond CP"
    # print(bond_cps_qtaim[(3, 4)])


def test_parse_critic2():
    cro = "./test_files/critic2/molecule/phenol_phenol.json"
    features = {
        "atom": [
            "field",
            "gradient_norm",
            "laplacian",
            "hessian_eigenvalues",
            "ellipticity",
        ],
        "bond": [
            "field",
            "gradient_norm",
            "laplacian",
            "hessian_eigenvalues",
            "ellipticity",
        ],
    }
    ret_dict = parse_critic2(cro, features=features)
    probe_bond_cp = ret_dict["bond_cps"]["55_bond"]
    probe_atom_cp = ret_dict["atom_cps"]["16_H"]

    assert len(probe_atom_cp.keys()) == 5, "wrong number of atoms"
    assert len(probe_bond_cp.keys()) == 8, "wrong number of bonds"
