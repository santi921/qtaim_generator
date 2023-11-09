import pandas as pd 
from qtaim_gen.source.core.parse_qtaim import (
    gather_imputation,
    gather_qtaim_features,
)
class TestParser:
    features_atom = [
        "Lagrangian_K",
        "e_density",
        "e_loc_func",
        "esp_total",
        "ellip_e_dens",
        "eta",
    ]

    features_bond = [
        "Lagrangian_K",
        "e_density",
        "e_loc_func",
        "esp_total",
        "ellip_e_dens",
        "eta",
    ]

    def test_reaction_parsing(self):
        reaction = True
        define_bonds = "qtaim"
        impute_file = "./test_files/reaction/impute.json"
        test_root = "./test_files/reaction/"
        df = pd.read_json("./test_files/reaction/b97d3.json") 

        
        impute_dict = gather_imputation(
            df,
            self.features_atom,
            self.features_bond,
            root_dir=test_root,
            json_file_imputed=impute_file,
            reaction=reaction,
            define_bonds=define_bonds,
        )


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
        )

        assert len(drop_list) == 0, "drop list not empty"   
        


    def test_molecule_parsing(self):
        test_root = "./test_files/molecule/"
        df = pd.read_pickle("./test_files/molecule/libe_qtaim_test.pkl")
        reaction = False
        define_bonds = "qtaim"
        impute_file = "./test_files/molecule/impute.json"

        impute_dict = gather_imputation(
            df = df,
            features_atom = self.features_atom,
            features_bond = self.features_bond,
            root_dir=test_root,
            json_file_imputed=impute_file,
            reaction = reaction,
            define_bonds = define_bonds,
        )

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
        )
        assert len(drop_list) == 0, "drop list not empty"   

