from qtaim_gen.source.core.parse_qtaim import (
    parse_cp,
    get_qtaim_descs,
    dft_inp_to_dict,
    only_atom_cps,
    find_cp_map,
    merge_qtaim_inds,
)


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


def test_only_atom_cps():
    qtaim_dict = get_qtaim_descs("./test_files/CPprop.txt")
    ret_dict, ret_dict_bonds = only_atom_cps(qtaim_dict)
    assert len(ret_dict.keys()) == 22, "wrong number of atom CP"
    assert len(ret_dict_bonds.keys()) == 23, "wrong number of bond CP"
    for i in ret_dict.values():
        assert "element" in i.keys(), "element not in dict"
        assert "pos_ang" in i.keys(), "xyz not in dict"


def test_find_cp_map():
    qtaim_dict = get_qtaim_descs("./test_files/CPprop.txt")
    atom_dict, atom_dict_bonds = only_atom_cps(qtaim_dict)
    dict_dft = dft_inp_to_dict("./test_files/input.in")
    ret_dict, qtaim_to_dft_map, missing_atoms = find_cp_map(
        dict_dft, atom_dict, margin=0.5
    )

    key_of_keys = list([i["key"] for i in qtaim_to_dft_map.values()])
    assert len(key_of_keys) == 17, "should have 22 keys"
    assert -1 in key_of_keys, "should have -1 in keys"
    assert len(missing_atoms) == 1, "should have one atom missing"


def test_add_closest_atoms_to_bond():
    pass


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
    cp_dict = merge_qtaim_inds(dict_qtaim, bonds, "./test_files/input_bond_map.in")
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
