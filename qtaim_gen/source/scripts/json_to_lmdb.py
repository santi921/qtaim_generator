"""
Convert generator JSON output folders to LMDB files for ML training.

This script converts the JSON output from qtaim_generator analysis into
separate LMDB files for each data type (structure, charge, qtaim, bonds, fuzzy, other).

Example directory structure expected:
    root_dir/
        job_001/
            *.inp           # ORCA input file (structure)
            charge.json     # Charge analysis
            qtaim.json      # QTAIM analysis
            bond.json       # Bond analysis
            fuzzy_full.json # Fuzzy analysis
            other.json      # Other properties

    OR with move_files=True:
    root_dir/
        job_001/
            generator/
                *.inp
                charge.json
                ...

Usage:
    # Convert all data types
    json-to-lmdb --root_dir ./jobs/ --out_dir ./lmdbs/ --all

    # Convert specific data types
    json-to-lmdb --root_dir ./jobs/ --out_dir ./lmdbs/ --types charge qtaim structure

    # With generator subfolder structure
    json-to-lmdb --root_dir ./jobs/ --out_dir ./lmdbs/ --all --move_files

    # Custom chunk size and no merge
    json-to-lmdb --root_dir ./jobs/ --out_dir ./lmdbs/ --all --chunk_size 1000 --no_merge
"""

import argparse
import os
import sys

from qtaim_gen.source.utils.lmdbs import json_2_lmdbs, inp_files_2_lmdbs


# Valid data types for JSON conversion
JSON_DATA_TYPES = ["charge", "qtaim", "bond", "fuzzy_full", "other"]

# Default output LMDB names
DEFAULT_LMDB_NAMES = {
    "structure": "structure.lmdb",
    "charge": "charge.lmdb",
    "qtaim": "qtaim.lmdb",
    "bond": "bond.lmdb",
    "fuzzy_full": "fuzzy.lmdb",
    "other": "other.lmdb",
}


def convert_structure(
    root_dir: str,
    out_dir: str,
    out_lmdb: str,
    chunk_size: int,
    clean: bool,
    merge: bool,
    move_files: bool,
) -> None:
    """Convert ORCA .inp files to structure LMDB."""
    print(f"Converting structure from .inp files...")
    inp_files_2_lmdbs(
        root_dir=root_dir,
        out_dir=out_dir,
        out_lmdb=out_lmdb,
        chunk_size=chunk_size,
        clean=clean,
        merge=merge,
        move_files=move_files,
    )
    print(f"  -> {os.path.join(out_dir, out_lmdb)}")


def convert_json_type(
    root_dir: str,
    out_dir: str,
    data_type: str,
    out_lmdb: str,
    chunk_size: int,
    clean: bool,
    merge: bool,
    move_files: bool,
) -> None:
    """Convert JSON files of a specific type to LMDB."""
    print(f"Converting {data_type} from JSON files...")
    json_2_lmdbs(
        root_dir=root_dir,
        out_dir=out_dir,
        data_type=data_type,
        out_lmdb=out_lmdb,
        chunk_size=chunk_size,
        clean=clean,
        merge=merge,
        move_files=move_files,
    )
    print(f"  -> {os.path.join(out_dir, out_lmdb)}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert generator JSON/inp files to LMDB databases",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Data types available:
  structure   Convert ORCA .inp files to geometry LMDB
  charge      Convert charge.json files
  qtaim       Convert qtaim.json files
  bond        Convert bond.json files
  fuzzy_full  Convert fuzzy_full.json files
  other       Convert other.json files

Examples:
  # Convert all data types with defaults
  json-to-lmdb --root_dir ./jobs/ --out_dir ./lmdbs/ --all

  # Convert only structure and charge
  json-to-lmdb --root_dir ./jobs/ --out_dir ./lmdbs/ --types structure charge

  # Use generator subfolder structure
  json-to-lmdb --root_dir ./jobs/ --out_dir ./lmdbs/ --all --move_files

  # Large dataset with custom chunk size
  json-to-lmdb --root_dir ./jobs/ --out_dir ./lmdbs/ --all --chunk_size 5000
        """,
    )

    parser.add_argument(
        "--root_dir",
        type=str,
        required=True,
        help="Root directory containing job folders with JSON/inp files",
    )

    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Output directory for LMDB files",
    )

    parser.add_argument(
        "--types",
        type=str,
        nargs="+",
        choices=["structure", "charge", "qtaim", "bond", "fuzzy_full", "other"],
        help="Data types to convert (use --all for all types)",
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Convert all data types",
    )

    parser.add_argument(
        "--chunk_size",
        type=int,
        default=1000,
        help="Number of files to process per chunk (default: 1000)",
    )

    parser.add_argument(
        "--no_merge",
        action="store_true",
        help="Don't merge chunk LMDBs into single file (keeps separate chunk files)",
    )

    parser.add_argument(
        "--no_clean",
        action="store_true",
        help="Don't delete intermediate chunk LMDB files after merge",
    )

    parser.add_argument(
        "--move_files",
        action="store_true",
        help="Look for files in job/generator/ subfolders instead of job/",
    )

    parser.add_argument(
        "--prefix",
        type=str,
        default="",
        help="Prefix for output LMDB filenames (e.g., 'dataset_' -> 'dataset_charge.lmdb')",
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.all and not args.types:
        parser.error("Must specify either --all or --types")

    # Ensure root_dir ends with separator
    root_dir = args.root_dir
    if not root_dir.endswith(os.sep):
        root_dir += os.sep

    # Create output directory
    out_dir = args.out_dir
    if not out_dir.endswith(os.sep):
        out_dir += os.sep
    os.makedirs(out_dir, exist_ok=True)

    # Determine which types to convert
    if args.all:
        types_to_convert = ["structure"] + JSON_DATA_TYPES
    else:
        types_to_convert = args.types

    # Settings
    merge = not args.no_merge
    clean = not args.no_clean
    chunk_size = args.chunk_size
    move_files = args.move_files
    prefix = args.prefix

    print(f"Converting data from: {root_dir}")
    print(f"Output directory: {out_dir}")
    print(f"Data types: {', '.join(types_to_convert)}")
    print(f"Chunk size: {chunk_size}")
    print(f"Merge chunks: {merge}")
    print(f"Clean intermediate: {clean}")
    print(f"Generator subfolders: {move_files}")
    print()

    # Convert each type
    converted = []
    failed = []

    for data_type in types_to_convert:
        out_lmdb = f"{prefix}{DEFAULT_LMDB_NAMES[data_type]}"

        try:
            if data_type == "structure":
                convert_structure(
                    root_dir=root_dir,
                    out_dir=out_dir,
                    out_lmdb=out_lmdb,
                    chunk_size=chunk_size,
                    clean=clean,
                    merge=merge,
                    move_files=move_files,
                )
            else:
                convert_json_type(
                    root_dir=root_dir,
                    out_dir=out_dir,
                    data_type=data_type,
                    out_lmdb=out_lmdb,
                    chunk_size=chunk_size,
                    clean=clean,
                    merge=merge,
                    move_files=move_files,
                )
            converted.append(data_type)
        except Exception as e:
            print(f"  ERROR: Failed to convert {data_type}: {e}")
            failed.append(data_type)

    # Summary
    print()
    print("=" * 50)
    print("Conversion Summary")
    print("=" * 50)
    print(f"Converted: {', '.join(converted) if converted else 'None'}")
    if failed:
        print(f"Failed: {', '.join(failed)}")
        sys.exit(1)
    else:
        print("All conversions completed successfully!")


if __name__ == "__main__":
    main()
