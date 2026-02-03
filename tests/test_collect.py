from pathlib import Path

import pandas as pd
from qtaim_gen.source.core.parse_qtaim import (
    gather_imputation,
    gather_qtaim_features,
)

TEST_FILES = Path(__file__).parent / "test_files"


class TestParser:
    features_atom = [
        "Lagrangian_K",
        "energy_density",
        "e_loc_func",
        "esp_total",
        "ellip_e_dens",
        "eta",
        "lol",
        "density_alpha",
        "density_beta",
    ]

    features_bond = [
        "Lagrangian_K",
        "energy_density",
        "e_loc_func",
        "esp_total",
        "ellip_e_dens",
        "eta",
        "lol",
        "density_alpha",
        "density_beta",
    ]

    def test_reaction_parsing(self):
        reaction = True
        define_bonds = "qtaim"
        impute_file = str(TEST_FILES / "reaction" / "impute.json")
        test_root = str(TEST_FILES / "reaction") + "/"
        df = pd.read_json(str(TEST_FILES / "reaction" / "b97d3.json"))

        impute_dict = gather_imputation(
            df,
            self.features_atom,
            self.features_bond,
            root_dir=test_root,
            json_file_imputed=impute_file,
            reaction=reaction,
            define_bonds=define_bonds,
            inp_type="xyz",
        )
        print(impute_dict)

        pandas_file, drop_list = gather_qtaim_features(
            df,
            test_root,
            self.features_atom,
            self.features_bond,
            reaction,
            define_bonds=define_bonds,
            update_bonds_w_qtaim=True,
            impute=True,
            impute_dict=impute_dict,
            inp_type="xyz",
        )
        print(drop_list)
        # print(pandas_file.iloc[0]["reactant_bonds_original"])
        # print(pandas_file.iloc[0]["reactant_bonds"])
        # print(pandas_file.iloc[0]["extra_feat_reactant_bond_indices_qtaim"])
        # print(pandas_file.columns)
        # for ind, row in pandas_file.iterrows():
        #    print(row["reactant_bonds"])
        #    print(row["extra_feat_bond_reactant_indices_qtaim"])
        #    print()
        #    print(row["product_bonds"])
        #    print(row["extra_feat_bond_product_indices_qtaim"])
        #    #assert len(row["bonds_original"]) == len(row["bonds"]), "bonds not same length"
        #    #assert len(row["bonds_original"]) == len(row["extra_feat_bond_indices_qtaim"]), "bonds not same length"

        assert len(drop_list) == 0, "drop list not empty"

    def test_molecule_parsing(self):
        test_root = str(TEST_FILES / "molecule") + "/"
        pkl_path = str(TEST_FILES / "molecule" / "libe_qtaim_test.pkl")
        try:
            df = pd.read_pickle(pkl_path)
        except:
            try:
                df = pd.read_pickle(pkl_path, encoding="latin1")
            except:
                raise Exception("Could not read test pickle file.")

        reaction = False
        define_bonds = "qtaim"
        impute_file = str(TEST_FILES / "molecule" / "impute.json")

        impute_dict = gather_imputation(
            df=df,
            features_atom=self.features_atom,
            features_bond=self.features_bond,
            root_dir=test_root,
            json_file_imputed=impute_file,
            reaction=reaction,
            define_bonds=define_bonds,
            inp_type="xyz",
        )
        print(impute_dict)

        pandas_file, drop_list = gather_qtaim_features(
            df,
            test_root,
            self.features_atom,
            self.features_bond,
            reaction,
            define_bonds=define_bonds,
            update_bonds_w_qtaim=True,
            impute=True,
            impute_dict=impute_dict,
            inp_type="xyz",
        )

        for ind, row in pandas_file.iterrows():
            # print(row["bonds"])
            # print(row["extra_feat_bond_indices_qtaim"])
            # assert len(row["bonds_original"]) == len(row["bonds"]), "bonds not same length"
            assert len(row["bonds"][0]) == len(
                row["extra_feat_bond_indices_qtaim"]
            ), "bonds not same length"

        assert len(drop_list) == 0, "drop list not empty"
        # print("---"*20)


# tester=TestParser()
# tester.test_reaction_parsing()
# tester.test_molecule_parsing()
