"""
Convert qtaim_generator LMDB files to qtaim_embed graph format.

Supports three converter types:
- BaseConverter: Structure-only graphs (geometry + bonds)
- QTAIMConverter: Structure + QTAIM features (bond critical points)
- GeneralConverter: Full pipeline with structure, QTAIM, charges, fuzzy, bonds, other data

Example config (lmdb_config.json):
{
    "restart": false,
    "lmdb_path": "/eagle/projects/generator/qtaim_embed/test_parse/",
    "lmdb_name": "base_graphs.lmdb",
    "lmdb_locations": {
        "geom_lmdb": "/eagle/projects/generator/lmdbs/rmechdb/structure.lmdb",
        "qtaim_lmdb": "/eagle/projects/generator/lmdbs/rmechdb/qtaim.lmdb",
        "charge_lmdb": "/eagle/projects/generator/lmdbs/rmechdb/charge.lmdb",
        "fuzzy_full_lmdb": "/eagle/projects/generator/lmdbs/rmechdb/fuzzy.lmdb",
        "bonds_lmdb": "/eagle/projects/generator/lmdbs/rmechdb/bond.lmdb",
        "other_lmdb": "/eagle/projects/generator/lmdbs/rmechdb/other.lmdb"
    },
    "keys_data": {
        "atom": ["eta", "lol"],
        "bond": ["eta", "lol"],
        "global": ["n_atoms", "spin", "charge", "n_bonds"]
    },
    "keys_target": {
        "atom": [],
        "bond": [],
        "global": ["n_bonds"]
    },
    "allowed_ring_size": [3, 4, 5, 6, 7, 8],
    "allowed_charges": [-2, -1, 0, 1, 2],
    "allowed_spins": [1, 2, 3],
    "n_workers": 8,
    "batch_size": 500,
    "charge_filter": ["adch", "hirshfeld", "cm5", "becke"],
    "bonding_scheme": "structural",
    "bond_filter": ["fuzzy"],
    "fuzzy_filter": ["becke_fuzzy_density", "hirsh_fuzzy_density"],
    "bond_cutoff": 0.3,
    "bond_list_definition": "fuzzy",
    "missing_data_strategy": "skip",
    "sentinel_value": null,
    "charge_filter": null,
    "fuzzy_filter": null,
    "other_filter": null,
    "save_scaler": true, 
    "filter_list": ["length", "scaled"]
}

charge: 'hirshfeld', 'adch', 'cm5', 'becke'
fuzzy_full: 'becke_fuzzy_density', 'hirsh_fuzzy_density', 'hirsh_fuzzy_spin', 'becke_fuzzy_spin'
qtaim: "eta", "lol",
other: 'mpp_full', 'sdp_full', 'mpp_heavy', 'sdp_heavy', 'ESP_Volume', 'ESP_Surface_Density', 'ESP_Minimal_value', 'ESP_Maximal_value', 'ESP_Overall_surface_area', 'ESP_Positive_surface_area

Usage:
    python generator_to_embed.py --config_path ./lmdb_config.json --converter general
    python generator_to_embed.py --config_path ./lmdb_config.json --converter qtaim --restart
    python generator_to_embed.py --config_path ./lmdb_config.json --converter base
"""

import argparse

from qtaim_gen.source.utils.lmdbs import parse_config_gen_to_embed
from qtaim_gen.source.core.converter import (
    BaseConverter,
    QTAIMConverter,
    GeneralConverter,
)


CONVERTER_MAP = {
    "base": BaseConverter,
    "qtaim": QTAIMConverter,
    "general": GeneralConverter,
}


def main():
    parser = argparse.ArgumentParser(
        description="Convert qtaim_generator LMDB files to qtaim_embed graph format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Converter types:
  base      Structure-only graphs (geometry + bonds)
  qtaim     Structure + QTAIM features (bond critical points)
  general   Full pipeline with all data sources (structure, QTAIM, charges, fuzzy, bonds, other)
        """
    )

    parser.add_argument(
        "--config_path",
        type=str,
        default="./lmdb_config.json",
        help="Path to the config file for converting the database",
    )

    parser.add_argument(
        "--converter",
        type=str,
        default="general",
        choices=["base", "qtaim", "general"],
        help="Converter type to use (default: general)",
    )

    parser.add_argument(
        "--restart",
        action="store_true",
        help="Restart mode: skip already processed keys instead of overwriting",
    )

    parser.add_argument(
        "--skip_scaling",
        action="store_true",
        help="Skip the scaling step after graph generation",
    )

    args = parser.parse_args()
    config_path = str(args.config_path)

    # Parse config
    config_dict = parse_config_gen_to_embed(
        args.config_path, restart=bool(args.restart)
    )

    # Select converter class
    ConverterClass = CONVERTER_MAP[args.converter]
    print(f"Using {ConverterClass.__name__} with {config_dict.get('n_workers', 8)} workers")

    # Initialize and run converter
    converter = ConverterClass(config_dict, config_path=config_path)
    converter.process()

    # Scale graphs unless skipped
    if not args.skip_scaling:
        converter.scale_graph_lmdb()
    else:
        print("Skipping scaling step")

    print("Done!")


if __name__ == "__main__":
    main()
