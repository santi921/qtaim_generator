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
            *.inp
            generator/
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

    # With verbose logging
    json-to-lmdb --root_dir ./jobs/ --out_dir ./lmdbs/ --all --verbose

    # Log to file
    json-to-lmdb --root_dir ./jobs/ --out_dir ./lmdbs/ --all --log_file conversion.log
"""

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from glob import glob
from typing import Dict, List, Optional, Set

from qtaim_gen.source.utils.lmdbs import json_2_lmdbs, inp_files_2_lmdbs


@dataclass
class ConversionStats:
    """Statistics for a single data type conversion."""
    data_type: str
    files_found: int = 0
    files_converted: int = 0
    files_skipped: int = 0
    files_failed: int = 0
    empty_files: int = 0
    missing_keys: Dict[str, int] = field(default_factory=dict)
    failed_files: List[str] = field(default_factory=list)
    skipped_files: List[str] = field(default_factory=list)

    def add_missing_key(self, key: str):
        """Track a missing key."""
        self.missing_keys[key] = self.missing_keys.get(key, 0) + 1

    def log_summary(self, logger: logging.Logger):
        """Log a summary of the conversion statistics."""
        logger.info(f"--- {self.data_type.upper()} Conversion Summary ---")
        logger.info(f"  Files found:     {self.files_found}")
        logger.info(f"  Files converted: {self.files_converted}")
        logger.info(f"  Files skipped:   {self.files_skipped}")
        logger.info(f"  Files failed:    {self.files_failed}")
        logger.info(f"  Empty files:     {self.empty_files}")

        if self.missing_keys:
            logger.warning(f"  Missing keys (top 10):")
            sorted_keys = sorted(self.missing_keys.items(), key=lambda x: -x[1])[:10]
            for key, count in sorted_keys:
                logger.warning(f"    - '{key}': {count} occurrences")

        if self.failed_files and logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"  Failed files:")
            for f in self.failed_files[:20]:  # Limit to first 20
                logger.debug(f"    - {f}")
            if len(self.failed_files) > 20:
                logger.debug(f"    ... and {len(self.failed_files) - 20} more")


def setup_logging(verbose: bool = False, log_file: Optional[str] = None) -> logging.Logger:
    """Configure logging for the conversion script."""
    logger = logging.getLogger("json_to_lmdb")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    console_format = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # File handler (if specified)
    if log_file:
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setLevel(logging.DEBUG)  # Always verbose in file
        file_format = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
        logger.info(f"Logging to file: {log_file}")

    return logger


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


def get_expected_json_keys(data_type: str, full_set: int = 0) -> Set[str]:
    """
    Return expected keys for each JSON data type based on validation.py.

    Args:
        data_type: Type of data (charge, bond, fuzzy_full, other, qtaim)
        full_set: Level of calculation detail (0=baseline, 1=extended, 2=full)

    Returns:
        Set of expected top-level keys for the JSON file.
    """
    if data_type == "charge":
        # Base keys
        keys = {"adch", "becke", "hirshfeld", "cm5"}
        if full_set > 0:
            keys.update({"mbis", "vdd", "chelpg"})
        if full_set > 1:
            keys.add("bader")
        return keys

    elif data_type == "bond":
        keys = {"fuzzy_bond"}
        if full_set > 0:
            keys.add("ibsi_bond")
        if full_set > 1:
            keys.add("laplacian_bond")
        return keys

    elif data_type == "fuzzy_full":
        # Base keys (density)
        keys = {"becke_fuzzy_density", "hirsh_fuzzy_density"}
        if full_set > 0:
            keys.update({"elf_fuzzy", "mbis_fuzzy_density"})
        if full_set > 1:
            keys.update({"grad_norm_rho_fuzzy", "laplacian_rho_fuzzy"})
        # Note: spin keys (hirsh_fuzzy_spin, becke_fuzzy_spin, mbis_fuzzy_spin)
        # depend on molecule spin state, so we don't require them by default
        return keys

    elif data_type == "other":
        keys = {
            "mpp_full", "sdp_full", "mpp_heavy", "sdp_heavy",
            "ALIE_Volume", "ALIE_Surface_Density",
            "ALIE_Minimal_value", "ALIE_Maximal_value",
            "ALIE_Overall_surface_area", "ALIE_Positive_surface_area",
            "ALIE_Negative_surface_area", "ALIE_Overall_skewness",
        }
        if full_set > 1:
            keys.update({
                "ESP_Volume", "ESP_Surface_Density",
                "ESP_Minimal_value", "ESP_Maximal_value",
                "ESP_Overall_surface_area", "ESP_Positive_surface_area",
                "ESP_Negative_surface_area", "ESP_Overall_skewness",
            })
        return keys

    elif data_type == "qtaim":
        # QTAIM keys are dynamic (atom indices and bond pairs)
        # Validation just checks structure, not specific keys
        return set()

    return set()


def inspect_lmdb_first_entry(lmdb_path: str, logger: logging.Logger) -> Optional[Dict]:
    """Read and return the first entry from an LMDB file."""
    import lmdb
    import pickle

    try:
        env = lmdb.open(
            lmdb_path,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        with env.begin(write=False) as txn:
            cursor = txn.cursor()
            for key, value in cursor:
                key_str = key.decode("ascii")
                if key_str != "length":
                    data = pickle.loads(value)
                    env.close()
                    return {"key": key_str, "data": data}
        env.close()
    except Exception as e:
        logger.warning(f"Could not inspect LMDB {lmdb_path}: {e}")
    return None


def log_first_entry(lmdb_path: str, data_type: str, logger: logging.Logger):
    """Log information about the first entry in an LMDB."""
    entry = inspect_lmdb_first_entry(lmdb_path, logger)
    if entry:
        logger.info(f"  First entry key: '{entry['key']}'")
        data = entry["data"]
        if isinstance(data, dict):
            keys = list(data.keys())[:10]
            logger.info(f"  First entry keys: {keys}")
            if len(data.keys()) > 10:
                logger.info(f"  ... and {len(data.keys()) - 10} more keys")
        elif hasattr(data, "__dict__"):
            logger.info(f"  First entry type: {type(data).__name__}")


def convert_json_with_stats(
    root_dir: str,
    out_dir: str,
    data_type: str,
    out_lmdb: str,
    chunk_size: int,
    clean: bool,
    merge: bool,
    move_files: bool,
    logger: logging.Logger,
    full_set: int = 0,
) -> ConversionStats:
    """Convert JSON files with detailed statistics tracking."""
    stats = ConversionStats(data_type=data_type)
    expected_keys = get_expected_json_keys(data_type, full_set=full_set)

    # Find files
    if move_files:
        pattern = os.path.join(root_dir, "*/generator/", f"{data_type}.json")
    else:
        pattern = os.path.join(root_dir, "*/", f"{data_type}.json")

    files = glob(pattern)
    stats.files_found = len(files)
    logger.info(f"Found {stats.files_found} {data_type}.json files")

    if stats.files_found == 0:
        logger.warning(f"No {data_type}.json files found matching pattern: {pattern}")
        return stats

    # Process files and track statistics
    for file_path in files:
        try:
            with open(file_path, "r") as f:
                data = json.load(f)

            if not data:
                stats.empty_files += 1
                stats.skipped_files.append(file_path)
                logger.debug(f"Empty file: {file_path}")
                continue

            # Check for expected keys
            if expected_keys and isinstance(data, dict):
                present_keys = set(data.keys())
                missing = expected_keys - present_keys
                for key in missing:
                    stats.add_missing_key(key)

            stats.files_converted += 1

        except json.JSONDecodeError as e:
            stats.files_failed += 1
            stats.failed_files.append(file_path)
            logger.warning(f"JSON decode error in {file_path}: {e}")
        except Exception as e:
            stats.files_failed += 1
            stats.failed_files.append(file_path)
            logger.warning(f"Error processing {file_path}: {e}")

    # Actually perform the conversion using the existing function
    logger.info(f"Writing LMDB for {data_type}...")
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

    # Log first entry from the resulting LMDB
    lmdb_path = os.path.join(out_dir, out_lmdb)
    if os.path.exists(lmdb_path):
        log_first_entry(lmdb_path, data_type, logger)

    return stats


def convert_structure_with_stats(
    root_dir: str,
    out_dir: str,
    out_lmdb: str,
    chunk_size: int,
    clean: bool,
    merge: bool,
    logger: logging.Logger,
) -> ConversionStats:
    """Convert ORCA .inp files to structure LMDB with statistics."""
    stats = ConversionStats(data_type="structure")

    pattern = os.path.join(root_dir, "*/*.inp")

    files = glob(pattern)
    stats.files_found = len(files)
    logger.info(f"Found {stats.files_found} .inp files for structure conversion")

    if stats.files_found == 0:
        logger.warning(f"No .inp files found matching pattern: {pattern}")
        return stats

    # Track processing (simplified - the actual conversion happens in inp_files_2_lmdbs)
    for file_path in files:
        try:
            if os.path.getsize(file_path) == 0:
                stats.empty_files += 1
                stats.skipped_files.append(file_path)
                logger.debug(f"Empty file: {file_path}")
            else:
                stats.files_converted += 1
        except Exception as e:
            stats.files_failed += 1
            stats.failed_files.append(file_path)
            logger.warning(f"Error checking {file_path}: {e}")

    # Perform the actual conversion
    logger.info("Writing structure LMDB...")
    inp_files_2_lmdbs(
        root_dir=root_dir,
        out_dir=out_dir,
        out_lmdb=out_lmdb,
        chunk_size=chunk_size,
        clean=clean,
        merge=merge,
    )

    # Log first entry from the resulting LMDB
    lmdb_path = os.path.join(out_dir, out_lmdb)
    if os.path.exists(lmdb_path):
        log_first_entry(lmdb_path, "structure", logger)

    return stats


def convert_structure(
    root_dir: str,
    out_dir: str,
    out_lmdb: str,
    chunk_size: int,
    clean: bool,
    merge: bool,
    move_files: bool,
) -> None:
    """Convert ORCA .inp files to structure LMDB (legacy interface)."""
    print(f"Converting structure from .inp files...")
    inp_files_2_lmdbs(
        root_dir=root_dir,
        out_dir=out_dir,
        out_lmdb=out_lmdb,
        chunk_size=chunk_size,
        clean=clean,
        merge=merge,
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

    parser.add_argument(
        "--full_set",
        type=int,
        default=0,
        choices=[0, 1, 2],
        help="Calculation detail level for key validation (0=baseline, 1=extended, 2=full)",
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging output",
    )

    parser.add_argument(
        "--log_file",
        type=str,
        default=None,
        help="Path to log file (logs are always verbose in file)",
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
    full_set = args.full_set

    # Setup logging
    logger = setup_logging(verbose=args.verbose, log_file=args.log_file)

    logger.info("=" * 60)
    logger.info("JSON to LMDB Conversion")
    logger.info("=" * 60)
    logger.info(f"Root directory:      {root_dir}")
    logger.info(f"Output directory:    {out_dir}")
    logger.info(f"Data types:          {', '.join(types_to_convert)}")
    logger.info(f"Chunk size:          {chunk_size}")
    logger.info(f"Merge chunks:        {merge}")
    logger.info(f"Clean intermediate:  {clean}")
    logger.info(f"Generator subfolders: {move_files}")
    logger.info(f"Validation level:    {full_set} ({'baseline' if full_set == 0 else 'extended' if full_set == 1 else 'full'})")
    logger.info("")

    # Convert each type and collect statistics
    converted = []
    failed = []
    all_stats: List[ConversionStats] = []

    for data_type in types_to_convert:
        out_lmdb = f"{prefix}{DEFAULT_LMDB_NAMES[data_type]}"
        logger.info(f"Processing {data_type}...")

        try:
            if data_type == "structure":
                stats = convert_structure_with_stats(
                    root_dir=root_dir,
                    out_dir=out_dir,
                    out_lmdb=out_lmdb,
                    chunk_size=chunk_size,
                    clean=clean,
                    merge=merge,
                    logger=logger,
                )
            else:
                stats = convert_json_with_stats(
                    root_dir=root_dir,
                    out_dir=out_dir,
                    data_type=data_type,
                    out_lmdb=out_lmdb,
                    chunk_size=chunk_size,
                    clean=clean,
                    merge=merge,
                    move_files=move_files,
                    logger=logger,
                    full_set=full_set,
                )

            all_stats.append(stats)
            converted.append(data_type)
            logger.info(f"  -> {os.path.join(out_dir, out_lmdb)}")

        except Exception as e:
            logger.error(f"Failed to convert {data_type}: {e}")
            failed.append(data_type)
            import traceback
            logger.debug(traceback.format_exc())

    # Final Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("CONVERSION SUMMARY")
    logger.info("=" * 60)

    total_found = sum(s.files_found for s in all_stats)
    total_converted = sum(s.files_converted for s in all_stats)
    total_failed = sum(s.files_failed for s in all_stats)
    total_empty = sum(s.empty_files for s in all_stats)

    logger.info(f"Total files found:     {total_found}")
    logger.info(f"Total files converted: {total_converted}")
    logger.info(f"Total files failed:    {total_failed}")
    logger.info(f"Total empty files:     {total_empty}")
    logger.info("")

    # Per-type summaries
    for stats in all_stats:
        stats.log_summary(logger)
        logger.info("")

    logger.info(f"Data types converted: {', '.join(converted) if converted else 'None'}")
    if failed:
        logger.error(f"Data types failed: {', '.join(failed)}")
        sys.exit(1)
    else:
        logger.info("All conversions completed successfully!")


if __name__ == "__main__":
    main()
