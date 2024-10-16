import numpy as np 

from qtaim_gen.source.core.parse_multiwfn import (
    parse_bond_order_doc,
    parse_other_doc,
    parse_charge_doc, 
    parse_fuzzy_doc
)


def test_bond_info():
    file_bond_info = "./test_files/multiwfn/bond_out.txt"
    bond_dict = parse_bond_order_doc(file_bond_info)
    
    assert len(bond_dict.keys()) == 3, "incorrect number of keys in the dictionary"
    assert len(bond_dict["laplace"]) == 24, "wrong number of laplace bonds"
    assert len(bond_dict["fuzzy"]) == 39, "wrong number of fuzzy bonds"
    assert len(bond_dict["ibsi"]) == 133, "wrong number of ibsi bonds"
    assert bond_dict["fuzzy"][-2] == ('8_C', '20_H', 0.87186477)
    assert np.isclose(bond_dict["laplace"][('9_C', '21_H')], 0.994489, atol=1e-3), "laplace bond order is not right"
    assert np.isclose(bond_dict["ibsi"][('19_H', '20_H')], 0.00291, atol=1e-3), "ibsi bond order is not right"

def test_other_info():
    file_other_info = "./test_files/multiwfn/other_out.txt"
    other_info_dict = parse_other_doc(file_other_info)
    assert len(other_info_dict.keys()) == 35, "incorrect number of keys in the dictionary"
    assert np.isclose(other_info_dict["ALIE_Maximal_value"], 12.40965, atol=1e-3), "ALIE_Maximal_value is not right"
    assert np.isclose(other_info_dict["ESP_Maximal_value"], 10.01633, atol=1e-3), "ESP_Maximal_value is not right"
    assert np.isclose(other_info_dict["mpp_heavy"], 0.605341, atol=1e-3), "mpp_heavy is not right"
    assert np.isclose(other_info_dict["sdp_heavy"], 2.073794, atol=1e-3), "sdp_heavy is not right"
    assert np.isclose(other_info_dict["sdp_full"], 4.146655, atol=1e-3), "sdp_full is not right"
    assert np.isclose(other_info_dict["mpp_full"], 0.942400, atol=1e-3), "mpp_full is not right"

def test_parse_charge(): 

    file_charge_info = "./test_files/multiwfn/charge_out.txt"
    
    (
        charge_dict_overall, 
        atomic_dipole_dict_overall, 
        dipole_info 
    ) = parse_charge_doc(file_charge_info)

    # overall
    assert len(dipole_info.keys()) == 5, "incorrect number of keys in dipole_info"
    assert len(charge_dict_overall.keys()) == 10, "incorrect number of keys in charge_dict_overall"
    assert len(atomic_dipole_dict_overall.keys()) == 2, "incorrect number of keys in atomic_dipole_dict_overall"
    
    # some charge tests
    # for each charge type make sure there are 21 atoms 
    assert len(charge_dict_overall["mbis"]) == 21, "wrong number of atoms in mbis charge"
    assert len(charge_dict_overall["cm5"]) == 21, "wrong number of atoms in cm5 charge"
    assert len(charge_dict_overall["resp"]) == 21, "wrong number of atoms in resp charge"
    assert len(charge_dict_overall["vdd"]) == 21, "wrong number of atoms in vdd charge"
    assert len(charge_dict_overall["hirshfeld"]) == 21, "wrong number of atoms in hirshfeld charge"
    assert len(charge_dict_overall["adch"]) == 21, "wrong number of atoms in adch charge"
    assert len(charge_dict_overall["mk"]) == 21, "wrong number of atoms in mk charge"
    assert len(charge_dict_overall["peoe"]) == 21, "wrong number of atoms in peoe charge"
    assert len(charge_dict_overall["becke"]) == 21, "wrong number of atoms in becke charge"
    # check some values
    assert np.isclose(charge_dict_overall["mbis"]["9_C"], -0.07241245, atol=1e-3), "mbis charge is not right"
    assert np.isclose(charge_dict_overall["cm5"]["9_C"], -0.07136386, atol=1e-3), "cm5 charge is not right"
    assert np.isclose(charge_dict_overall["resp"]["9_C"], -0.1170913892, atol=1e-3), "resp charge is not right"
    assert np.isclose(charge_dict_overall["vdd"]["9_C"], -0.01951725, atol=1e-3), "vdd charge is not right"
    assert np.isclose(charge_dict_overall["hirshfeld"]["9_C"], -0.01592160, atol=1e-3), "hirshfeld charge is not right"
    assert np.isclose(charge_dict_overall["adch"]["9_C"], -0.08840616, atol=1e-3), "adch charge is not right"
    assert np.isclose(charge_dict_overall["mk"]["9_C"], -0.1234529776, atol=1e-3), "mk charge is not right"
    assert np.isclose(charge_dict_overall["peoe"]["9_C"], -0.02939796, atol=1e-3), "peoe charge is not right"
    assert np.isclose(charge_dict_overall["becke"]["9_C"], 0.00267819, atol=1e-3), "becke charge is not right"
    

    # atom dipoles
    assert len(atomic_dipole_dict_overall["adch"]) == 21, "wrong number of atoms in adch dipole"
    assert len(atomic_dipole_dict_overall["becke"]) == 21, "wrong number of atoms in becke dipole"
    assert atomic_dipole_dict_overall["becke"]["12_H"] == [0.056434, -0.14593, -0.112629], "wrong becke dipole"
    assert atomic_dipole_dict_overall["adch"]["12_H"] == [-0.047288, 0.079529, 0.096176], "wrong adch dipole"
    
    # overall dipoles
    assert np.isclose(dipole_info['cm5']['mag'], 0.0755761, atol=1e-3), "cm5 dipole is not right"
    assert np.isclose(dipole_info['cm5']['xyz'][1], 0.05797, atol=1e-3), "cm5 dipole is not right"
    assert np.isclose(dipole_info['adch']['mag'], 0.0268691, atol=1e-3), "adch dipole is not right"
    assert np.isclose(dipole_info['hirshfeld']['mag'], 0.064283, atol=1e-3), "hirshfeld dipole is not right"
    assert np.isclose(dipole_info['hirshfeld']['xyz'][1], 0.057821, atol=1e-3), "hirshfeld dipole is not right"
    assert np.isclose(dipole_info['becke']['mag'], 0.301398, atol=1e-3), "becke dipole is not right"
    assert np.isclose(dipole_info['vdd']['mag'], 0.123382, atol=1e-3), "vdd dipole is not right"
    assert np.isclose(dipole_info['vdd']['xyz'][1], 0.123263, atol=1e-3), "vdd dipole is not right"

def test_fuzzy_parse(): 
    file_fuzzy_info = "./test_files/multiwfn/fuzzy_out.txt"
    dict_fuzzy = parse_fuzzy_doc(file_fuzzy_info)
    probe_atom = "5_C"
    probe_values = []
    ground_values = np.array([6.20602588, 27.71346755, 0.31558365, 24.10015000, 4.015])
    for key in dict_fuzzy["real_space"].keys():
        for key2 in dict_fuzzy["real_space"][key].keys():
            if key2 == probe_atom:
                probe_values.append(dict_fuzzy["real_space"][key][key2])
    probe_values.append(dict_fuzzy["localization"][probe_atom])
    assert len(dict_fuzzy.keys()) == 2, "incorrect number of keys in the dictionary"
    assert len(dict_fuzzy["real_space"].keys()) == 4, "incorrect number of atoms in real space"
    assert len(dict_fuzzy["localization"].keys()) == 21, "incorrect number of atoms in localization"
    assert len(dict_fuzzy["real_space"]["density"].keys()) == 23, "incorrect number of atoms in localization"
    
    probe_values = np.array(probe_values)
    assert np.allclose(probe_values, ground_values, atol=1e-3), "fuzzy values are not right"
