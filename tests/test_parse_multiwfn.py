import numpy as np

from qtaim_gen.source.core.parse_multiwfn import (
    parse_bond_order_doc,
    parse_other_doc,
    parse_charge_doc,
    parse_fuzzy_doc,
    parse_bond_order_ibsi,
    parse_bond_order_laplace,
    parse_bond_order_fuzzy,
    parse_charge_doc_adch,
    parse_charge_becke,
    parse_charge_base,
)

from qtaim_gen.source.core.omol import gbw_analysis


class TestMultiwfnParser:

    orca_6 = "./test_files/multiwfn/"

    gbw_analysis(
        folder=orca_6,
        orca_2mkl_cmd="orca2_mkl",
        multiwfn_cmd="Multiwfn",
        parse_only=True,
        separate=True,
        clean=False,
    )  # works!

    def test_bond_info(self):
        file_bond_info = "./test_files/multiwfn/bond.out"
        bond_dict = parse_bond_order_doc(file_bond_info)
        # print(bond_dict["fuzzy"]['81_H_to_82_N'], bond_dict["laplace"]['81_H_to_82_N'], bond_dict["ibsi"]['81_H_to_82_N'])
        assert len(bond_dict.keys()) == 3, "incorrect number of keys in the dictionary"
        assert len(bond_dict["laplace"]) == 88, "wrong number of laplace bonds"
        assert len(bond_dict["fuzzy"]) == 158, "wrong number of fuzzy bonds"
        assert len(bond_dict["ibsi"]) == 568, "wrong number of ibsi bonds"
        assert np.isclose(
            bond_dict["fuzzy"]["81_H_to_82_N"], 0.82812969, atol=1e-3
        ), "fuzzy bond order is not right"
        assert np.isclose(
            bond_dict["laplace"]["81_H_to_82_N"], 0.690022, atol=1e-3
        ), "laplace bond order is not right"
        assert np.isclose(
            bond_dict["ibsi"]["81_H_to_82_N"], 0.83809, atol=1e-3
        ), "ibsi bond order is not right"

    def test_other_info(self):
        file_other_info = "./test_files/multiwfn/other.out"
        other_info_dict = parse_other_doc(file_other_info)
        assert (
            len(other_info_dict.keys()) == 35
        ), "incorrect number of keys in the dictionary"
        assert np.isclose(
            other_info_dict["mpp_heavy"], 0.60534, atol=1e-3
        ), "mpp_heavy is not right"
        assert np.isclose(
            other_info_dict["sdp_heavy"],  2.073794, atol=1e-3
        ), "sdp_heavy is not right"
        assert np.isclose(
            other_info_dict["sdp_full"], 4.146655, atol=1e-3
        ), "sdp_full is not right"
        assert np.isclose(
            other_info_dict["mpp_full"], 0.9424, atol=1e-3
        ), "mpp_full is not right"

    def test_parse_charge(self):

        file_charge_info = "./test_files/multiwfn/charge.out"

        (
            charge_dict_overall,
            atomic_dipole_dict_overall,
            dipole_info,
        ) = parse_charge_doc(file_charge_info)

        # overall
        assert len(dipole_info.keys()) == 4, "incorrect number of keys in dipole_info"
        assert (
            len(charge_dict_overall.keys()) == 6
        ), "incorrect number of keys in charge_dict_overall"
        assert (
            len(atomic_dipole_dict_overall.keys()) == 2
        ), "incorrect number of keys in atomic_dipole_dict_overall"

        # some charge tests
        # for each charge type make sure there are 21 atoms
        assert (
            len(charge_dict_overall["mbis"]) == 87
        ), "wrong number of atoms in mbis charge"
        assert (
            len(charge_dict_overall["cm5"]) == 87
        ), "wrong number of atoms in cm5 charge"
        assert (
            len(charge_dict_overall["vdd"]) == 87
        ), "wrong number of atoms in vdd charge"
        assert (
            len(charge_dict_overall["hirshfeld"]) == 87
        ), "wrong number of atoms in hirshfeld charge"
        assert (
            len(charge_dict_overall["adch"]) == 87
        ), "wrong number of atoms in adch charge"
        assert (
            len(charge_dict_overall["becke"]) == 87
        ), "wrong number of atoms in becke charge"
        # check some values

        assert np.isclose(
            charge_dict_overall["mbis"]["9_C"], 0.76835115, atol=1e-3
        ), "mbis charge is not right"
        assert np.isclose(
            charge_dict_overall["cm5"]["9_C"], 0.6247437354, atol=1e-3
        ), "cm5 charge is not right"
        assert np.isclose(
            charge_dict_overall["vdd"]["9_C"], 0.2250167, atol=1e-3
        ), "vdd charge is not right"
        assert np.isclose(
            charge_dict_overall["hirshfeld"]["9_C"], 0.20358532, atol=1e-3
        ), "hirshfeld charge is not right"
        assert np.isclose(
            charge_dict_overall["adch"]["9_C"], 0.15440985, atol=1e-3
        ), "adch charge is not right"
        assert np.isclose(
            charge_dict_overall["becke"]["9_C"], 0.66429787, atol=1e-3
        ), "becke charge is not right"

        # atom dipoles
        assert (
            len(atomic_dipole_dict_overall["adch"]) == 87
        ), "wrong number of atoms in adch dipole"
        assert (
            len(atomic_dipole_dict_overall["becke"]) == 87
        ), "wrong number of atoms in becke dipole"
        assert atomic_dipole_dict_overall["becke"]["9_C"] == [
            0.149466,
            -0.080294,
            -0.034142,
        ], "wrong becke dipole"
        assert atomic_dipole_dict_overall["adch"]["9_C"] == [
            -0.031129,
            -0.01329,
            -0.018004,
        ], "wrong adch dipole"

        # overall dipoles
        assert np.isclose(
            dipole_info["adch"]["mag"], 1.4609484, atol=1e-3
        ), "adch dipole is not right"
        assert np.isclose(
            dipole_info["hirshfeld"]["mag"], 1.825152, atol=1e-3
        ), "hirshfeld dipole is not right"
        assert np.isclose(
            dipole_info["vdd"]["mag"], 1.708353, atol=1e-3
        ), "vdd dipole is not right"
        assert np.isclose(
            dipole_info["becke"]["mag"], 1.99846, atol=1e-3
        ), "becke dipole is not right"

        assert np.isclose(
            dipole_info["adch"]["xyz"][-1], -1.424021, atol=1e-3
        ), "adch dipole is not right"
        assert np.isclose(
            dipole_info["hirshfeld"]["xyz"][-1], -1.77656, atol=1e-3
        ), "hirshfeld dipole is not right"
        assert np.isclose(
            dipole_info["vdd"]["xyz"][-1], -1.63161, atol=1e-3
        ), "vdd dipole is not right"
        assert np.isclose(
            dipole_info["becke"]["xyz"][-1], -0.855972, atol=1e-3
        ), "becke dipole is not right"

    def test_fuzzy_parse(self):
        file_fuzzy_info = "./test_files/multiwfn/fuzzy_full.out"
        dict_fuzzy = parse_fuzzy_doc(file_fuzzy_info)
        probe_atom = "5_C"
        probe_values = []
        ground_values = np.array(
            [6.20602588, 27.71346755, 0.31558365, 24.10015000, 4.015]
        )
        for key in dict_fuzzy["real_space"].keys():
            for key2 in dict_fuzzy["real_space"][key].keys():
                if key2 == probe_atom:
                    probe_values.append(dict_fuzzy["real_space"][key][key2])
        probe_values.append(dict_fuzzy["localization"][probe_atom])
        assert len(dict_fuzzy.keys()) == 2, "incorrect number of keys in the dictionary"
        assert (
            len(dict_fuzzy["real_space"].keys()) == 4
        ), "incorrect number of atoms in real space"
        assert (
            len(dict_fuzzy["localization"].keys()) == 21
        ), "incorrect number of atoms in localization"
        assert (
            len(dict_fuzzy["real_space"]["density"].keys()) == 23
        ), "incorrect number of atoms in localization"

        probe_values = np.array(probe_values)
        assert np.allclose(
            probe_values, ground_values, atol=1e-3
        ), "fuzzy values are not right"

    def test_charge_info_separate(self):

        file_mbis_info = "./test_files/multiwfn/mbis.out"
        charge_dict_overall = parse_charge_base(
            file_mbis_info, corrected=False, dipole=False
        )
        
        assert len(charge_dict_overall) == 21, "wrong number of atoms in mbis charge"
        assert np.isclose(
            charge_dict_overall["9_C"], -0.07241207, atol=1e-3
        ), "mbis charge is not right"

        file_hirshfeld_info = "./test_files/multiwfn/hirshfeld.out"
        charge_dict_overall, dipole_info = parse_charge_base(
            file_hirshfeld_info, corrected=False
        )
        assert (
            len(charge_dict_overall) == 21
        ), "wrong number of atoms in hirshfeld charge"
        assert np.isclose(
            charge_dict_overall["9_C"], -0.0159214, atol=1e-3
        ), "hirshfeld charge is not right"
        assert np.isclose(
            dipole_info["mag"], 0.064279, atol=1e-3
        ), "hirshfeld dipole is not right"


        file_vdd_info = "./test_files/multiwfn/vdd.out"
        charge_dict_overall, dipole_info = parse_charge_base(
            file_vdd_info, corrected=False, dipole=True
        )
        
        assert len(charge_dict_overall) == 21, "wrong number of atoms in vdd charge"
        assert np.isclose(
            charge_dict_overall["9_C"], -0.01951698, atol=1e-3
        ), "vdd charge is not right"
        assert np.isclose(
            dipole_info["mag"], 0.123379, atol=1e-3
        ), "vdd dipole is not right"

        file_cm5_info = "./test_files/multiwfn/cm5.out"
        charge_dict_overall, dipole_info = parse_charge_base(
            file_cm5_info, corrected=False
        )
        
        assert len(charge_dict_overall) == 21, "wrong number of atoms in cm5 charge"
        assert np.isclose(
            charge_dict_overall["9_C"], -0.07136367, atol=1e-3
        ), "cm5 charge is not right"
        assert np.isclose(
            dipole_info["mag"], 0.064279, atol=1e-3
        ), "cm5 dipole is not right"

        file_adch_info = "./test_files/multiwfn/adch.out"
        (
            charge_dict_overall,
            atomic_dipole_dict_overall,
            dipole_info,
        ) = parse_charge_doc_adch(file_adch_info)
        
        assert len(charge_dict_overall) == 21, "wrong number of atoms in adch charge"
        assert np.isclose(
            charge_dict_overall["9_C"], -0.08840594, atol=1e-3
        ), "adch charge is not right"
        assert np.isclose(
            dipole_info["mag"], 0.0262041, atol=1e-3
        ), "adch dipole is not right"
        assert np.isclose(
            dipole_info["xyz"][-1], 0.0150, atol=1e-3
        ), "adch dipole is not right"
        assert np.isclose(
            atomic_dipole_dict_overall["9_C"][-1],  0.001915, atol=1e-3
        ), "adch dipole is not right"

        file_becke_info = "./test_files/multiwfn/becke.out"
        (
            charge_dict_overall,
            atomic_dipole_dict_overall,
            dipole_info,
        ) = parse_charge_becke(file_becke_info)
        #print(charge_dict_overall["9_C"], dipole_info["mag"], dipole_info["xyz"][-1], atomic_dipole_dict_overall["9_C"][-1])
        assert len(charge_dict_overall) == 21, "wrong number of atoms in becke charge"
        assert np.isclose(
            charge_dict_overall["9_C"], 0.00267841, atol=1e-3
        ), "becke charge is not right"
        assert np.isclose(
            dipole_info["mag"],  0.0268685, atol=1e-3
        ), "becke dipole is not right"
        assert np.isclose(
            dipole_info["xyz"][-1], 0.015906, atol=1e-3
        ), "becke dipole is not right"
        assert np.isclose(
            atomic_dipole_dict_overall["9_C"][-1], 0.230093, atol=1e-3
        ), "becke dipole is not right"

    def test_bond_info_separate(self):
        file_bond_info = "./test_files/multiwfn/ibsi.out"
        ibsi_dict = parse_bond_order_ibsi(file_bond_info)
        file_bond_info = "./test_files/multiwfn/laplacian.out"
        laplace_dict = parse_bond_order_laplace(file_bond_info)
        file_bond_info = "./test_files/multiwfn/fuzzy.out"
        fuzzy_dict = parse_bond_order_fuzzy(file_bond_info)
        print(ibsi_dict["6_C_to_9_C"])
        assert len(laplace_dict) == 88, "wrong number of laplace bonds"
        assert len(fuzzy_dict) == 39, "wrong number of fuzzy bonds"
        assert len(ibsi_dict) == 133, "wrong number of ibsi bonds"
        assert np.isclose(
            fuzzy_dict["6_C_to_9_C"],  0.07468526, atol=1e-3
        ), "fuzzy bond order is not right"
        assert np.isclose(
            laplace_dict["68_C_to_76_H"], 0.819832, atol=1e-3
        ), "laplace bond order is not right"
        assert np.isclose(
            ibsi_dict["6_C_to_9_C"], 0.07658, atol=1e-3
        ), "ibsi bond order is not right"



