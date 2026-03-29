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
    "filter_list": ["length", "scaled"],
    "split_method": "random",
    "split_ratios": [0.8, 0.1, 0.1],
    "split_seed": 42
}

charge: 'hirshfeld', 'adch', 'cm5', 'becke'
fuzzy_full: 'becke_fuzzy_density', 'hirsh_fuzzy_density', 'hirsh_fuzzy_spin', 'becke_fuzzy_spin'
qtaim: "eta", "lol",
other: 'mpp_full', 'sdp_full', 'mpp_heavy', 'sdp_heavy', 'ESP_Volume', 'ESP_Surface_Density', 'ESP_Minimal_value', 'ESP_Maximal_value', 'ESP_Overall_surface_area', 'ESP_Positive_surface_area

Usage:
    python generator_to_embed.py --config_path ./lmdb_config.json --converter general
    python generator_to_embed.py --config_path ./lmdb_config.json --converter qtaim --restart
    python generator_to_embed.py --config_path ./lmdb_config.json --converter base
    python generator_to_embed.py --config_path ./lmdb_config.json --converter general --split
"""

import argparse

from qtaim_gen.source.utils.lmdbs import parse_config_gen_to_embed
from qtaim_gen.source.core.converter import (
    BaseConverter,
    QTAIMConverter,
    GeneralConverter,
)
from qtaim_gen.source.utils.splits import (
    SplitConfig,
    build_formula_map_from_structure_lmdb,
    partition_lmdb_into_splits,
)
from qtaim_gen.source.utils.scaling import (
    fit_scalers_on_lmdbs,
    apply_scalers_to_lmdb_inplace,
    save_scalers,
)


CONVERTER_MAP = {
    "base": BaseConverter,
    "qtaim": QTAIMConverter,
    "general": GeneralConverter,
}


def scale_split_lmdbs(converter, split_paths: dict[str, str]) -> None:
    """Fit scalers on the train split only, then apply to all splits.

    Delegates to utils/scaling.py for the reusable fit/apply/save logic.
    """
    lmdb_path = converter.config_dict["lmdb_path"]
    skip_keys = (
        set(converter.skip_keys)
        | {"split_name"}
        | set(converter.config_dict.get("filter_list", ["length", "scaled"]))
    )

    # Fit on train only
    feature_scaler, label_scaler = fit_scalers_on_lmdbs(
        [split_paths["train"]], skip_keys
    )

    # Save scalers
    save_scalers(feature_scaler, label_scaler, lmdb_path)

    # Apply to all splits
    for split_name, split_path in split_paths.items():
        print(f"Applying scalers to {split_name} split...")
        count = apply_scalers_to_lmdb_inplace(
            split_path, feature_scaler, label_scaler, skip_keys
        )
        print(f"  Scaled {count} graphs in {split_name}")


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

    parser.add_argument(
        "--split",
        action="store_true",
        help="Enable train/val/test splitting of the output LMDB",
    )

    args = parser.parse_args()
    config_path = str(args.config_path)

    # Parse config (passes split flag for sharding validation)
    config_dict = parse_config_gen_to_embed(
        args.config_path, restart=bool(args.restart)
    )
    # Inject split flag for sharding mutual-exclusivity check
    if args.split:
        config_dict["_split_enabled"] = True
        total_shards = config_dict.get("total_shards", 1)
        if total_shards > 1:
            raise ValueError(
                "Splitting and sharding are mutually exclusive. "
                f"Got total_shards={total_shards} with --split enabled."
            )

    # Select converter class
    ConverterClass = CONVERTER_MAP[args.converter]
    print(f"Using {ConverterClass.__name__} with {config_dict.get('n_workers', 8)} workers")

    # Initialize and run converter
    converter = ConverterClass(config_dict, config_path=config_path)
    converter.process()

    if args.split:
        # Build formula map if composition-based splitting
        formula_map = None
        if config_dict.get("split_method") == "composition":
            geom_lmdb_path = config_dict["lmdb_locations"]["geom_lmdb"]
            print(f"Building formula map from {geom_lmdb_path}...")
            formula_map = build_formula_map_from_structure_lmdb(geom_lmdb_path)
            print(f"Found {len(formula_map)} entries with {len(set(formula_map.values()))} unique formulas")

        # Build split config
        split_config = SplitConfig(
            method=config_dict.get("split_method", "random"),
            ratios=tuple(config_dict.get("split_ratios", [0.8, 0.1, 0.1])),
            seed=config_dict.get("split_seed", 42),
        )

        # Derive base name from lmdb_name
        lmdb_name = config_dict["lmdb_name"]
        base_name = lmdb_name.replace(".lmdb", "") if lmdb_name.endswith(".lmdb") else lmdb_name

        # Partition the intermediate LMDB into splits
        print(f"Splitting with method='{split_config.method}', ratios={split_config.ratios}, seed={split_config.seed}")
        split_paths = partition_lmdb_into_splits(
            src_path=converter.file,
            out_dir=config_dict["lmdb_path"],
            base_name=base_name,
            config=split_config,
            formula_map=formula_map,
        )

        # Scale using train-only fitting
        if not args.skip_scaling:
            scale_split_lmdbs(converter, split_paths)
        else:
            print("Skipping scaling step")

    else:
        # Original behavior: scale the single output LMDB
        if not args.skip_scaling:
            converter.scale_graph_lmdb()
        else:
            print("Skipping scaling step")

    print("Done!")


if __name__ == "__main__":
    main()
