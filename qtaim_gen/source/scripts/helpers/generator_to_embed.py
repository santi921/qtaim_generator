"""
Convert qtaim_generator LMDB files to qtaim_embed graph format.

Supports three converter types:
- BaseConverter: Structure-only graphs (geometry + bonds)
- QTAIMConverter: Structure + QTAIM features (bond critical points)
- GeneralConverter: Full pipeline with structure, QTAIM, charges, fuzzy, bonds, other data

Example config (lmdb_config.json):
{
    "restart": false,
    "lmdb_path": "/path/to/output/",
    "lmdb_name": "graphs.lmdb",

    "lmdb_locations": {
        "geom_lmdb": "/path/to/structure.lmdb",
        "qtaim_lmdb": "/path/to/qtaim.lmdb",
        "charge_lmdb": "/path/to/charge.lmdb",
        "fuzzy_full_lmdb": "/path/to/fuzzy.lmdb",
        "bonds_lmdb": "/path/to/bonds.lmdb",
        "other_lmdb": "/path/to/other.lmdb"
    },

    "keys_data": {
        "atom": [],
        "bond": [],
        "global": []
    },
    "keys_target": {
        "atom": [],
        "bond": [],
        "global": ["extra_energy"]
    },

    "allowed_ring_size": [3, 4, 5, 6, 7, 8],
    "allowed_charges": [-2, -1, 0, 1, 2],
    "allowed_spins": [1, 2, 3],

    "filter_list": [],

    "n_workers": 8,
    "batch_size": 500,

    "bonding_scheme": "structural",
    "bond_filter": ["fuzzy", "ibsi"],
    "bond_cutoff": 0.3,
    "bond_list_definition": "fuzzy",

    "missing_data_strategy": "skip",
    "sentinel_value": null,

    "charge_filter": null,
    "fuzzy_filter": null,
    "other_filter": null,

    "save_scaler": true
}

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
