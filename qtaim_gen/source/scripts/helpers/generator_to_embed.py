import argparse

from qtaim_gen.source.utils.lmdbs import parse_config_gen_to_embed
from qtaim_gen.source.core.converter import QTAIMEmbedConverter


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LMDB to embed")

    parser.add_argument(
        "--config_path",
        type=str,
        default="./lmdb_config.json",
        help="Path to the config file for converting the database",
    )

    parser.add_argument(
        "--lmdb_name",
        type=str,
        default="molecule.lmdb",
        help="name of output lmdb file",
    )

    parser.add_argument(
        "--restart",
        action="store_true",
        help="Restart the process, will overwrite the existing lmdb file",
    )

    args = parser.parse_args()
    config_path = str(args.config_path)

    config_dict = parse_config_gen_to_embed(
        args.config_path, restart=bool(args.restart)
    )
    scaler = QTAIMEmbedConverter(config_dict, config_path=config_path)
    scaler.main_loop()
    scaler.scale_graph_lmdb()
