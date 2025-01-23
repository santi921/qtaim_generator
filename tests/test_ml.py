import pandas as pd
from qtaim_gen.source.core.parse_qtaim import (
    gather_imputation,
    gather_qtaim_features,
)


class TestML:
    features_atom = [
        "Lagrangian_K",
        "energy_density",
        "e_loc_func",
        "esp_total",
        "ellip_e_dens",
        "eta",
    ]

    features_bond = [
        "Lagrangian_K",
        "energy_density",
        "e_loc_func",
        "esp_total",
        "ellip_e_dens",
        "eta",
    ]

    reaction = True
    define_bonds = "qtaim"
    impute_file = "./test_files/reaction/impute.json"
    test_root = "./test_files/reaction/"
    df = pd.read_json("./test_files/reaction/b97d3.json")

    impute_dict = gather_imputation(
        df,
        features_atom,
        features_bond,
        root_dir=test_root,
        json_file_imputed=impute_file,
        reaction=reaction,
        define_bonds=define_bonds,
        inp_type="xyz",
    )

    pandas_file, drop_list = gather_qtaim_features(
        df,
        test_root,
        features_atom,
        features_bond,
        reaction,
        define_bonds=define_bonds,
        update_bonds_w_qtaim=True,
        impute=True,
        impute_dict=impute_dict,
        inp_type="xyz",
    )

    rxn_df = pandas_file
    # save
    rxn_df.to_pickle("./test_files/reaction/libe_qtaim_test.pkl")

    test_root = "./test_files/molecule/"
    df = pd.read_pickle("./test_files/molecule/libe_qtaim_test.pkl")
    reaction = False
    define_bonds = "qtaim"
    impute_file = "./test_files/molecule/impute.json"

    impute_dict = gather_imputation(
        df=df,
        features_atom=features_atom,
        features_bond=features_bond,
        root_dir=test_root,
        json_file_imputed=impute_file,
        reaction=reaction,
        define_bonds=define_bonds,
        inp_type="xyz",
    )

    pandas_file, drop_list = gather_qtaim_features(
        df,
        test_root,
        features_atom,
        features_bond,
        reaction,
        define_bonds=define_bonds,
        update_bonds_w_qtaim=True,
        impute=True,
        impute_dict=impute_dict,
        inp_type="xyz",
    )
    print("length drop list: ", len(drop_list))

    mol_df = pandas_file
    # save
    mol_df.to_pickle("./test_files/molecule/libe_qtaim_test.pkl")

    def test_bondnet(self):
        from bondnet.data.dataset import ReactionDatasetGraphs
        from bondnet.model.training_utils import get_grapher

        extra_features = {
            "bond": ["bond_length", "esp_total"],
            "atom": ["esp_total"],
            "mappings": ["indices_qtaim"],
        }

        dataset_bondnet = ReactionDatasetGraphs(
            grapher=get_grapher(extra_features),
            file="./test_files/reaction/libe_qtaim_test.pkl",
            feature_filter=False,
            target="ea",
            filter_species=[4, 5],
            filter_outliers=False,
            filter_sparse_rxns=False,
            classifier=False,
            debug=False,
            classif_categories=None,
            extra_keys=extra_features,
            extra_info=None,
        )
        assert len(dataset_bondnet) == 3, "bondnet dataset not parsed correctly"

    def test_qtaim_embed(self):
        from qtaim_embed.core.dataset import HeteroGraphGraphLabelDataset

        qtaim_embed_dataset = HeteroGraphGraphLabelDataset(
            file="./test_files/molecule/libe_qtaim_test.pkl",
            allowed_ring_size=[5, 6, 7],
            allowed_charges=None,
            allowed_spins=None,
            self_loop=True,
            extra_keys={
                "atom": ["extra_feat_atom_esp_total"],
                "bond": [
                    "extra_feat_bond_esp_total",
                    "bond_length",
                ],
                "global": ["shifted_rrho_ev_free_energy"],
            },
            target_list=["shifted_rrho_ev_free_energy"],
            extra_dataset_info=None,
            debug=False,
            log_scale_features=False,
            log_scale_targets=False,
            standard_scale_features=False,
            standard_scale_targets=False,
            bond_key="extra_feat_bond_indices_qtaim",
            map_key="extra_feat_bond_indices_qtaim",
        )

        assert (
            "extra_feat_atom_esp_total" in qtaim_embed_dataset.exclude_names["atom"]
        ), "extra atom feature not parsed correctly"
        assert (
            "extra_feat_bond_esp_total" in qtaim_embed_dataset.exclude_names["bond"]
        ), "extra bond feature not parsed correctly"
        assert qtaim_embed_dataset.include_names == {
            "global": ["shifted_rrho_ev_free_energy"]
        }, "targets not parsed correctly"


tester = TestML()
tester.test_qtaim_embed()
