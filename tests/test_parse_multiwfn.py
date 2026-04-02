from pathlib import Path

import numpy as np

from qtaim_gen.source.core.parse_multiwfn import (
    _extract_au_float,
    _split_fortran_floats,
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
    parse_fuzzy_real_space,
)

from qtaim_gen.source.core.omol import gbw_analysis

TEST_FILES = Path(__file__).parent / "test_files"


class TestMultiwfnParser:

    orca_6 = str(TEST_FILES / "multiwfn")
    gbw_analysis(
        folder=orca_6,
        orca_2mkl_cmd="orca2_mkl",
        multiwfn_cmd="Multiwfn",
        parse_only=True,
        separate=True,
        clean=False,
    )  # works!

    def test_bond_info(self):
        file_bond_info = str(TEST_FILES / "multiwfn" / "bond.out")
        bond_dict = parse_bond_order_doc(file_bond_info)
        # print(bond_dict["fuzzy"]['81_H_to_82_N'], bond_dict["laplace"]['81_H_to_82_N'], bond_dict["ibsi"]['81_H_to_82_N'])
        # assert len(bond_dict.keys()) == 3, "incorrect number of keys in the dictionary"
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
        file_other_info = str(TEST_FILES / "multiwfn" / "other.out")
        other_info_dict = parse_other_doc(file_other_info)
        assert (
            len(other_info_dict.keys()) == 32
        ), "incorrect number of keys in the dictionary"
        assert np.isclose(
            other_info_dict["mpp_heavy"], 0.60534, atol=1e-3
        ), "mpp_heavy is not right"
        assert np.isclose(
            other_info_dict["sdp_heavy"], 2.073794, atol=1e-3
        ), "sdp_heavy is not right"
        assert np.isclose(
            other_info_dict["sdp_full"], 4.146655, atol=1e-3
        ), "sdp_full is not right"
        assert np.isclose(
            other_info_dict["mpp_full"], 0.9424, atol=1e-3
        ), "mpp_full is not right"

    def test_parse_charge(self):

        file_charge_info = str(TEST_FILES / "multiwfn" / "charge.out")

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
        file_fuzzy_info = str(TEST_FILES / "multiwfn" / "fuzzy_full.out")
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

        file_mbis_info = str(TEST_FILES / "multiwfn" / "mbis.out")
        charge_dict_overall = parse_charge_base(
            file_mbis_info, corrected=False, dipole=False
        )

        assert len(charge_dict_overall) == 21, "wrong number of atoms in mbis charge"
        assert np.isclose(
            charge_dict_overall["9_C"], -0.07241207, atol=1e-3
        ), "mbis charge is not right"

        file_hirshfeld_info = str(TEST_FILES / "multiwfn" / "hirshfeld.out")
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

        file_vdd_info = str(TEST_FILES / "multiwfn" / "vdd.out")
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

        file_cm5_info = str(TEST_FILES / "multiwfn" / "cm5.out")
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

        file_adch_info = str(TEST_FILES / "multiwfn" / "adch.out")
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
            atomic_dipole_dict_overall["9_C"][-1], 0.001915, atol=1e-3
        ), "adch dipole is not right"

        file_becke_info = str(TEST_FILES / "multiwfn" / "becke.out")
        (
            charge_dict_overall,
            atomic_dipole_dict_overall,
            dipole_info,
        ) = parse_charge_becke(file_becke_info)
        # print(charge_dict_overall["9_C"], dipole_info["mag"], dipole_info["xyz"][-1], atomic_dipole_dict_overall["9_C"][-1])
        assert len(charge_dict_overall) == 21, "wrong number of atoms in becke charge"
        assert np.isclose(
            charge_dict_overall["9_C"], 0.00267841, atol=1e-3
        ), "becke charge is not right"
        assert np.isclose(
            dipole_info["mag"], 0.0268685, atol=1e-3
        ), "becke dipole is not right"
        assert np.isclose(
            dipole_info["xyz"][-1], 0.015906, atol=1e-3
        ), "becke dipole is not right"
        assert np.isclose(
            atomic_dipole_dict_overall["9_C"][-1], 0.230093, atol=1e-3
        ), "becke dipole is not right"

    def test_bond_info_separate(self):
        file_bond_info = str(TEST_FILES / "multiwfn" / "ibsi.out")
        ibsi_dict = parse_bond_order_ibsi(file_bond_info)
        file_bond_info = str(TEST_FILES / "multiwfn" / "laplacian.out")
        laplace_dict = parse_bond_order_laplace(file_bond_info)
        file_bond_info = str(TEST_FILES / "multiwfn" / "fuzzy.out")
        fuzzy_dict = parse_bond_order_fuzzy(file_bond_info)
        print(ibsi_dict["6_C_to_9_C"])
        assert len(laplace_dict) == 88, "wrong number of laplace bonds"
        assert len(fuzzy_dict) == 39, "wrong number of fuzzy bonds"
        assert len(ibsi_dict) == 133, "wrong number of ibsi bonds"
        assert np.isclose(
            fuzzy_dict["6_C_to_9_C"], 0.07468526, atol=1e-3
        ), "fuzzy bond order is not right"
        assert np.isclose(
            laplace_dict["68_C_to_76_H"], 0.819832, atol=1e-3
        ), "laplace bond order is not right"
        assert np.isclose(
            ibsi_dict["6_C_to_9_C"], 0.07658, atol=1e-3
        ), "ibsi bond order is not right"

    def test_fuzzy_info_separate(self):
        spin_dict = parse_fuzzy_real_space(
            str(TEST_FILES / "multiwfn" / "becke_fuzzy_spin.out")
        )["becke_fuzzy_spin"]
        density_dict = parse_fuzzy_real_space(
            str(TEST_FILES / "multiwfn" / "becke_fuzzy_density.out")
        )["becke_fuzzy_density"]

        assert len(spin_dict) == 60, "wrong number of atoms in becke fuzzy spin"
        assert len(density_dict) == 23, "wrong number of atoms in becke fuzzy density"
        assert np.isclose(
            spin_dict["45_Cl"], 0.0, atol=1e-3
        ), "wrong value for becke fuzzy spin for 13_H"
        assert np.isclose(
            density_dict["13_H"], 0.81110108, atol=1e-3
        ), "wrong value for becke fuzzy density for 13_H"
        assert np.isclose(
            density_dict["sum"], 65.99996761, atol=1e-3
        ), "wrong value for becke fuzzy density for 13_H"


EDGE_CASES = Path(__file__).parent / "test_files" / "edge_cases"


class TestExtractAuFloat:
    """Unit tests for _extract_au_float helper."""

    def test_normal_token(self):
        assert np.isclose(_extract_au_float("1048.617"), 1048.617, atol=1e-3)

    def test_overflow_no_colon(self):
        assert np.isclose(
            _extract_au_float("(a.u.)1048.6171082"), 1048.6171082, atol=1e-6
        )

    def test_overflow_with_colon(self):
        assert np.isclose(
            _extract_au_float("(a.u.):1048.6171082"), 1048.6171082, atol=1e-6
        )

    def test_negative_overflow(self):
        assert np.isclose(
            _extract_au_float("(a.u.)-1048.617"), -1048.617, atol=1e-3
        )

    def test_small_normal_value(self):
        assert np.isclose(_extract_au_float("0.0262041"), 0.0262041, atol=1e-6)


class TestLargeMoleculeEdgeCases:
    """Tests for parsing large-molecule Multiwfn output with field overflows."""

    def test_parse_adch_large_molecule(self):
        file_path = str(EDGE_CASES / "adch.out")
        charges, atomic_dipoles, dipole_info = parse_charge_doc_adch(file_path)
        assert len(charges) == 123, f"expected 123 atoms, got {len(charges)}"
        assert len(atomic_dipoles) == 123, f"expected 123 atomic dipoles, got {len(atomic_dipoles)}"
        assert np.isclose(
            dipole_info["mag"], 1048.6171, atol=0.1
        ), f"adch dipole magnitude wrong: {dipole_info['mag']}"
        assert len(dipole_info["xyz"]) == 3, "expected 3 XYZ dipole components"
        assert np.isclose(dipole_info["xyz"][0], 431.44, atol=0.1)

    def test_parse_becke_large_molecule(self):
        file_path = str(EDGE_CASES / "becke.out")
        charges, atomic_dipoles, dipole_info = parse_charge_becke(file_path)
        assert len(charges) == 136, f"expected 136 atoms, got {len(charges)}"
        assert len(atomic_dipoles) == 136, f"expected 136 atomic dipoles, got {len(atomic_dipoles)}"
        assert np.isclose(
            dipole_info["mag"], 1346.2753, atol=0.1
        ), f"becke dipole magnitude wrong: {dipole_info['mag']}"
        assert len(dipole_info["xyz"]) == 3, "expected 3 XYZ dipole components"
        assert np.isclose(dipole_info["xyz"][0], -952.46, atol=0.1)


class TestSplitFortranFloats:
    """Unit tests for _split_fortran_floats helper that handles column overflow."""

    def test_normal_tokens_pass_through(self):
        assert _split_fortran_floats(["1.0", "-2.5", "3.14"]) == ["1.0", "-2.5", "3.14"]

    def test_concatenated_negative_pair(self):
        result = _split_fortran_floats(["-990.879181-1009.749968"])
        assert len(result) == 2
        assert np.isclose(float(result[0]), -990.879181, atol=1e-6)
        assert np.isclose(float(result[1]), -1009.749968, atol=1e-6)

    def test_mixed_concat_and_normal(self):
        result = _split_fortran_floats(["-990.879181-1009.749968", "-653.968178"])
        assert len(result) == 3
        assert np.isclose(float(result[0]), -990.879181, atol=1e-6)
        assert np.isclose(float(result[1]), -1009.749968, atol=1e-6)
        assert np.isclose(float(result[2]), -653.968178, atol=1e-6)

    def test_positive_value_no_split(self):
        result = _split_fortran_floats(["431.441513"])
        assert result == ["431.441513"]

    def test_empty_list(self):
        assert _split_fortran_floats([]) == []

    def test_au_token_filtered(self):
        """Non-numeric '(a.u.)' tokens should be filtered out."""
        result = _split_fortran_floats(["(a.u.)", "-990.879181-1009.749968", "-653.968178"])
        assert len(result) == 3
        floats = [float(x) for x in result]
        assert np.isclose(floats[0], -990.879181, atol=1e-6)
        assert np.isclose(floats[1], -1009.749968, atol=1e-6)
        assert np.isclose(floats[2], -653.968178, atol=1e-6)


class TestFortranOverflowEdgeCases3:
    """Tests for _3 edge case files (139-atom protein fragment):
    - adch_3/becke_3: concatenated dipole XYZ in both Hirshfeld and ADC sections
    - hirshfeld_3: concatenated dipole XYZ
    - cm5_3: asterisk overflow (********) in CM5 dipole Z component
    """

    def test_parse_adch_3_fortran_overflow(self):
        file_path = str(EDGE_CASES / "adch_3.out")
        charges, atomic_dipoles, dipole_info = parse_charge_doc_adch(file_path)
        assert len(charges) == 139, f"expected 139 atoms, got {len(charges)}"
        assert len(atomic_dipoles) == 139
        assert np.isclose(dipole_info["mag"], 1536.6853156, atol=0.1)
        assert len(dipole_info["xyz"]) == 3
        assert np.isclose(dipole_info["xyz"][0], -920.70713, atol=0.01)
        assert np.isclose(dipole_info["xyz"][1], -657.50725, atol=0.01)
        assert np.isclose(dipole_info["xyz"][2], -1039.89632, atol=0.01)
        assert np.isclose(charges["1_C"], -0.3010114, atol=1e-3)
        assert np.isclose(charges["139_H"], 0.06504135, atol=1e-3)

    def test_parse_becke_3_fortran_overflow(self):
        file_path = str(EDGE_CASES / "becke_3.out")
        charges, atomic_dipoles, dipole_info = parse_charge_becke(file_path)
        assert len(charges) == 139, f"expected 139 atoms, got {len(charges)}"
        assert len(atomic_dipoles) == 139
        assert np.isclose(dipole_info["mag"], 1537.7637936, atol=0.1)
        assert len(dipole_info["xyz"]) == 3
        assert np.isclose(dipole_info["xyz"][0], -921.361692, atol=0.01)
        assert np.isclose(dipole_info["xyz"][1], -657.908738, atol=0.01)
        assert np.isclose(dipole_info["xyz"][2], -1040.656624, atol=0.01)
        assert np.isclose(charges["1_C"], -0.2683871, atol=1e-3)
        assert np.isclose(charges["139_H"], 0.0222125, atol=1e-3)

    def test_parse_hirshfeld_3_fortran_overflow(self):
        file_path = str(EDGE_CASES / "hirshfeld_3.out")
        charges, dipole_info = parse_charge_base(
            file_path, corrected=False, dipole=True
        )
        assert len(charges) == 139, f"expected 139 atoms, got {len(charges)}"
        assert np.isclose(dipole_info["mag"], 1537.101194, atol=0.1)
        assert len(dipole_info["xyz"]) == 3
        assert np.isclose(dipole_info["xyz"][0], -922.036924, atol=0.01)
        assert np.isclose(dipole_info["xyz"][1], -656.383404, atol=0.01)
        assert np.isclose(dipole_info["xyz"][2], -1040.0427, atol=0.01)
        assert np.isclose(charges["1_C"], -0.08210226, atol=1e-3)
        assert np.isclose(charges["139_H"], 0.01669355, atol=1e-3)

    def test_parse_cm5_3_asterisk_overflow(self):
        """cm5_3.out has '**********' for the Z dipole component (Fortran field overflow)
        and the CM5 magnitude label concatenated with the value.
        parse_charge_base returns Hirshfeld dipole (first match) and CM5 charges (last
        'Final atomic charges:' section). The asterisk does not cause a crash here
        because parse_charge_base uses the Hirshfeld dipole key, not the CM5-specific one.
        """
        file_path = str(EDGE_CASES / "cm5_3.out")
        charges, dipole_info = parse_charge_base(
            file_path, corrected=False, dipole=True
        )
        assert len(charges) == 139, f"expected 139 atoms, got {len(charges)}"
        assert np.isclose(charges["1_C"], -0.21245274, atol=1e-3)
        assert np.isclose(charges["139_H"], 0.07437931, atol=1e-3)
        # dipole_info reflects the Hirshfeld dipole (first match in file)
        assert np.isclose(dipole_info["mag"], 1537.101194, atol=0.1)

    def test_parse_charge_doc_cm5_3_magnitude_concat(self):
        """parse_charge_doc on cm5_3.out: CM5 magnitude glued to key
        ('Total dipole moment from CM5 charges1537.0019201') and Z component
        overflowed to '**********' must not crash; Z should be NaN.
        """
        from qtaim_gen.source.core.parse_multiwfn import parse_charge_doc
        import math

        file_path = str(EDGE_CASES / "cm5_3.out")
        _, _, dipole_info = parse_charge_doc(file_path)
        assert np.isclose(dipole_info["cm5"]["mag"], 1537.0019201, atol=0.1)
        assert len(dipole_info["cm5"]["xyz"]) == 3
        assert np.isclose(dipole_info["cm5"]["xyz"][0], -921.45822, atol=0.01)
        assert np.isclose(dipole_info["cm5"]["xyz"][1], -657.02420, atol=0.01)
        assert math.isnan(dipole_info["cm5"]["xyz"][2])


class TestFortranOverflowEdgeCases:
    """Tests for parsing Multiwfn output files with Fortran column overflow
    in dipole XYZ vectors (e.g. '-990.879181-1009.749968')."""

    def test_parse_adch_2_fortran_overflow(self):
        """adch_2.out has concatenated dipole XYZ values."""
        file_path = str(EDGE_CASES / "adch_2.out")
        charges, atomic_dipoles, dipole_info = parse_charge_doc_adch(file_path)
        assert len(charges) == 119, f"expected 119 atoms, got {len(charges)}"
        assert len(atomic_dipoles) == 119
        assert np.isclose(dipole_info["mag"], 1558.5605, atol=0.1)
        assert len(dipole_info["xyz"]) == 3, "should have 3 XYZ components after split"
        assert np.isclose(dipole_info["xyz"][0], -990.879181, atol=0.01)
        assert np.isclose(dipole_info["xyz"][1], -1009.749968, atol=0.01)
        assert np.isclose(dipole_info["xyz"][2], -653.968178, atol=0.01)

    def test_parse_becke_2_fortran_overflow(self):
        """becke_2.out has concatenated dipole XYZ values."""
        file_path = str(EDGE_CASES / "becke_2.out")
        charges, atomic_dipoles, dipole_info = parse_charge_becke(file_path)
        assert len(charges) == 119, f"expected 119 atoms, got {len(charges)}"
        assert len(atomic_dipoles) == 119
        assert np.isclose(dipole_info["mag"], 1563.4249, atol=0.1)
        assert len(dipole_info["xyz"]) == 3
        assert np.isclose(dipole_info["xyz"][0], -993.865061, atol=0.01)
        assert np.isclose(dipole_info["xyz"][1], -1012.952241, atol=0.01)
        assert np.isclose(dipole_info["xyz"][2], -656.092732, atol=0.01)

    def test_parse_hirshfeld_2_fortran_overflow(self):
        """hirshfeld_2.out has concatenated dipole XYZ in the vector line."""
        file_path = str(EDGE_CASES / "hirshfeld_2.out")
        charges, dipole_info = parse_charge_base(
            file_path, corrected=False, dipole=True
        )
        assert len(charges) == 119, f"expected 119 atoms, got {len(charges)}"
        assert np.isclose(dipole_info["mag"], 1558.325406, atol=0.1)
        assert len(dipole_info["xyz"]) == 3
        assert np.isclose(dipole_info["xyz"][0], -990.754330, atol=0.01)
        assert np.isclose(dipole_info["xyz"][1], -1009.896429, atol=0.01)
        assert np.isclose(dipole_info["xyz"][2], -653.370593, atol=0.01)


class TestFortranOverflowEdgeCases4:
    """Tests for _4 edge case files (78-atom peptide, H39 C25 N6 O8, ~552 Da):
    - adch_4: dipole magnitude concatenated to key, all 3 XYZ concatenated
    - becke_4: same pattern — previously crashed with float('charge')
    - hirshfeld_4: all 3 XYZ values concatenated (same pattern as _2/_3)
    - cm5_4: CM5 XYZ has X valid + Y and Z *both* overflowed to asterisks,
             producing a single 20-char '*' block that must yield 3 values.
    """

    def test_parse_adch_4_fortran_overflow(self):
        file_path = str(EDGE_CASES / "adch_4.out")
        charges, atomic_dipoles, dipole_info = parse_charge_doc_adch(file_path)
        assert len(charges) == 78, f"expected 78 atoms, got {len(charges)}"
        assert len(atomic_dipoles) == 78
        assert np.isclose(dipole_info["mag"], 1902.8705203, atol=0.1)
        assert len(dipole_info["xyz"]) == 3
        assert np.isclose(dipole_info["xyz"][0], -862.341510, atol=0.01)
        assert np.isclose(dipole_info["xyz"][1], -1363.724694, atol=0.01)
        assert np.isclose(dipole_info["xyz"][2], -1008.731033, atol=0.01)
        assert np.isclose(charges["1_C"], -0.26732153, atol=1e-3)
        assert np.isclose(charges["78_H"], 0.05003116, atol=1e-3)

    def test_parse_becke_4_fortran_overflow(self):
        """becke_4.out: all 3 XYZ concatenated into one token — previously
        crashed because line.split()[-3:] grabbed 'charge' and '(a.u.)'."""
        file_path = str(EDGE_CASES / "becke_4.out")
        charges, atomic_dipoles, dipole_info = parse_charge_becke(file_path)
        assert len(charges) == 78, f"expected 78 atoms, got {len(charges)}"
        assert len(atomic_dipoles) == 78
        assert np.isclose(dipole_info["mag"], 1903.6441316, atol=0.1)
        assert len(dipole_info["xyz"]) == 3
        assert np.isclose(dipole_info["xyz"][0], -862.742686, atol=0.01)
        assert np.isclose(dipole_info["xyz"][1], -1364.253702, atol=0.01)
        assert np.isclose(dipole_info["xyz"][2], -1009.132238, atol=0.01)
        assert np.isclose(charges["1_C"], -0.34556096, atol=1e-3)
        assert np.isclose(charges["78_H"], -0.02819415, atol=1e-3)

    def test_parse_hirshfeld_4_fortran_overflow(self):
        file_path = str(EDGE_CASES / "hirshfeld_4.out")
        charges, dipole_info = parse_charge_base(
            file_path, corrected=False, dipole=True
        )
        assert len(charges) == 78, f"expected 78 atoms, got {len(charges)}"
        assert np.isclose(dipole_info["mag"], 1902.468492, atol=0.1)
        assert len(dipole_info["xyz"]) == 3
        assert np.isclose(dipole_info["xyz"][0], -862.561502, atol=0.01)
        assert np.isclose(dipole_info["xyz"][1], -1363.291334, atol=0.01)
        assert np.isclose(dipole_info["xyz"][2], -1008.370347, atol=0.01)
        assert np.isclose(charges["1_C"], -0.08524946, atol=1e-3)
        assert np.isclose(charges["78_H"], 0.01351075, atol=1e-3)

    def test_parse_cm5_4_double_asterisk_overflow(self):
        """cm5_4.out: CM5 XYZ line has X valid + 20 asterisks representing
        Y and Z both overflowed.  Must produce 3 values with Y=nan, Z=nan."""
        import math
        file_path = str(EDGE_CASES / "cm5_4.out")
        _, _, dipole_info = parse_charge_doc(file_path)
        assert np.isclose(dipole_info["cm5"]["mag"], 1902.9145878, atol=0.1)
        assert len(dipole_info["cm5"]["xyz"]) == 3
        assert np.isclose(dipole_info["cm5"]["xyz"][0], -862.57631, atol=0.01)
        assert math.isnan(dipole_info["cm5"]["xyz"][1])
        assert math.isnan(dipole_info["cm5"]["xyz"][2])
