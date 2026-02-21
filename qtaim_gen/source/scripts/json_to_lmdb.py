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
import time
from dataclasses import dataclass, field
from glob import glob
from typing import Dict, List, Optional, Set, Tuple

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
    # Timing stats
    scan_time_sec: float = 0.0
    lmdb_write_time_sec: float = 0.0
    total_bytes_read: int = 0

    def add_missing_key(self, key: str):
        """Track a missing key."""
        self.missing_keys[key] = self.missing_keys.get(key, 0) + 1

    @property
    def read_rate_files_per_sec(self) -> float:
        """Calculate file read rate."""
        if self.scan_time_sec > 0:
            return self.files_converted / self.scan_time_sec
        return 0.0

    @property
    def write_rate_files_per_sec(self) -> float:
        """Calculate LMDB write rate (files written per second)."""
        if self.lmdb_write_time_sec > 0:
            return self.files_converted / self.lmdb_write_time_sec
        return 0.0

    @property
    def read_rate_mb_per_sec(self) -> float:
        """Calculate read throughput in MB/sec."""
        if self.scan_time_sec > 0 and self.total_bytes_read > 0:
            return (self.total_bytes_read / (1024 * 1024)) / self.scan_time_sec
        return 0.0

    def log_summary(self, logger: logging.Logger):
        """Log a summary of the conversion statistics."""
        logger.info(f"--- {self.data_type.upper()} Conversion Summary ---")
        logger.info(f"  Files found:     {self.files_found}")
        logger.info(f"  Files converted: {self.files_converted}")
        logger.info(f"  Files skipped:   {self.files_skipped}")
        logger.info(f"  Files failed:    {self.files_failed}")
        logger.info(f"  Empty files:     {self.empty_files}")

        # Timing stats
        if self.scan_time_sec > 0:
            logger.info(f"  Scan/read time:  {self.scan_time_sec:.2f}s ({self.read_rate_files_per_sec:.1f} files/sec)")
        if self.total_bytes_read > 0:
            mb_read = self.total_bytes_read / (1024 * 1024)
            logger.info(f"  Data read:       {mb_read:.2f} MB ({self.read_rate_mb_per_sec:.2f} MB/sec)")
        if self.lmdb_write_time_sec > 0:
            logger.info(f"  LMDB write time: {self.lmdb_write_time_sec:.2f}s ({self.write_rate_files_per_sec:.1f} files/sec)")

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


def partition_folders_by_shard(root_dir: str, shard_index: int, total_shards: int, logger: logging.Logger) -> List[str]:
    """
    Partition job folders across shards based on folder name.

    This ensures all JSON files for a given job go to the same shard,
    maintaining consistency across different data types.

    Args:
        root_dir: Root directory containing job folders
        shard_index: Current shard index (0-based)
        total_shards: Total number of shards
        logger: Logger instance

    Returns:
        List of folder names assigned to this shard
    """
    if total_shards <= 1:
        return []  # No sharding, process all folders

    # Get all job folders (one level deep)
    pattern = os.path.join(root_dir, "*/")
    all_folders = sorted(glob(pattern))

    # Extract folder names (without trailing slash)
    folder_names = [os.path.basename(os.path.normpath(f)) for f in all_folders]

    # Partition based on index modulo
    partitioned = [name for i, name in enumerate(folder_names) if i % total_shards == shard_index]

    logger.info(f"Shard {shard_index}: assigned {len(partitioned)} folders (of {len(folder_names)} total)")
    logger.debug(f"First 10 folders in shard {shard_index}: {partitioned[:10]}")

    return partitioned


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
        result = None
        with env.begin(write=False) as txn:
            cursor = txn.cursor()
            for key, value in cursor:
                key_str = key.decode("ascii")
                if key_str != "length":
                    data = pickle.loads(value)
                    result = {"key": key_str, "data": data}
                    break
        env.close()
        return result
    except Exception as e:
        logger.warning(f"Could not inspect LMDB {lmdb_path}: {e}")
    return None


def merge_shards(
    shard_lmdbs: List[str],
    output_path: str,
    logger: logging.Logger,
) -> Tuple[str, int]:
    """
    Merge multiple shard LMDB files into a single LMDB.

    Args:
        shard_lmdbs: List of paths to shard LMDB files
        output_path: Path for the merged output LMDB
        logger: Logger instance

    Returns:
        Tuple of (path to the merged LMDB file, number of entries merged)
    """
    import lmdb
    import pickle

    if not shard_lmdbs:
        raise ValueError("No shard LMDBs provided for merging")

    logger.info(f"Merging {len(shard_lmdbs)} shards into: {output_path}")

    # Calculate total entries across all shards
    total_entries = 0
    for shard_path in shard_lmdbs:
        if not os.path.exists(shard_path):
            logger.warning(f"Shard LMDB not found: {shard_path}, skipping")
            continue
        try:
            env = lmdb.open(shard_path, subdir=False, readonly=True, lock=False)
            with env.begin() as txn:
                length_bytes = txn.get("length".encode())
                if length_bytes:
                    shard_length = pickle.loads(length_bytes)
                    total_entries += shard_length
                    logger.debug(f"Shard {shard_path}: {shard_length} entries")
            env.close()
        except Exception as e:
            logger.warning(f"Error reading shard {shard_path}: {e}")

    logger.info(f"Total entries to merge: {total_entries}")

    # Create output LMDB with appropriate map_size
    map_size = max(10 * 1024**3, total_entries * 1024 * 100)  # 10GB minimum or ~100KB per entry
    out_env = lmdb.open(
        output_path,
        subdir=False,
        map_size=map_size,
        lock=False,
    )

    # Merge all shards
    merged_count = 0
    with out_env.begin(write=True) as out_txn:
        for shard_path in shard_lmdbs:
            if not os.path.exists(shard_path):
                continue

            logger.info(f"Merging shard: {shard_path}")
            try:
                in_env = lmdb.open(shard_path, subdir=False, readonly=True, lock=False)
                with in_env.begin() as in_txn:
                    cursor = in_txn.cursor()
                    for key, value in cursor:
                        key_str = key.decode("ascii")
                        if key_str == "length":
                            continue  # Skip length key, we'll set it at the end
                        out_txn.put(key, value)
                        merged_count += 1

                        if merged_count % 10000 == 0:
                            logger.debug(f"Merged {merged_count} entries...")
                in_env.close()
            except Exception as e:
                logger.error(f"Error merging shard {shard_path}: {e}")
                raise

        # Set final length
        out_txn.put("length".encode(), pickle.dumps(merged_count))

    out_env.close()
    logger.info(f"Merge complete: {merged_count} entries written to {output_path}")

    return output_path, merged_count


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
    limit: Optional[int] = None,
    shard_index: int = 0,
    total_shards: int = 1,
    shard_folders: Optional[List[str]] = None,
) -> ConversionStats:
    """Convert JSON files with detailed statistics tracking."""
    stats = ConversionStats(data_type=data_type)
    expected_keys = get_expected_json_keys(data_type, full_set=full_set)

    # Update output LMDB name for sharding
    if total_shards > 1:
        base_name = out_lmdb.replace(".lmdb", "")
        out_lmdb = f"{base_name}_shard_{shard_index}.lmdb"
        logger.info(f"Sharding enabled: output will be {out_lmdb}")

    # Find files - filter by shard assignment if sharding is enabled
    if total_shards > 1 and shard_folders:
        # Build patterns for assigned folders only
        files = []
        for folder in shard_folders:
            if move_files:
                pattern = os.path.join(root_dir, folder, "generator", f"{data_type}.json")
            else:
                pattern = os.path.join(root_dir, folder, f"{data_type}.json")
            folder_files = glob(pattern)
            files.extend(folder_files)
        logger.debug(f"Shard {shard_index}: found {len(files)} {data_type}.json files in {len(shard_folders)} assigned folders")
    else:
        # No sharding - process all files
        if move_files:
            pattern = os.path.join(root_dir, "*/generator/", f"{data_type}.json")
        else:
            pattern = os.path.join(root_dir, "*/", f"{data_type}.json")
        files = glob(pattern)

    # Apply debug limit
    if limit is not None:
        files = files[:min(limit, len(files))]
        logger.debug(f"Debug mode: limiting to {len(files)} files")

    stats.files_found = len(files)
    logger.info(f"Found {stats.files_found} {data_type}.json files")

    if stats.files_found == 0:
        logger.warning(f"No {data_type}.json files found matching pattern: {pattern}")
        return stats

    # Process files and track statistics with timing
    scan_start = time.time()
    progress_interval = max(1, stats.files_found // 20)  # Update ~20 times during scan

    for i, file_path in enumerate(files):
        try:
            # Track file size for throughput calculation
            file_size = os.path.getsize(file_path)
            stats.total_bytes_read += file_size

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

            # Dynamic progress update
            if (i + 1) % progress_interval == 0 or (i + 1) == stats.files_found:
                elapsed = time.time() - scan_start
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                print(f"\r  Scanning {data_type}: {i + 1}/{stats.files_found} files ({rate:.1f} files/sec)", end="", flush=True)

        except json.JSONDecodeError as e:
            stats.files_failed += 1
            stats.failed_files.append(file_path)
            logger.warning(f"JSON decode error in {file_path}: {e}")
        except Exception as e:
            stats.files_failed += 1
            stats.failed_files.append(file_path)
            logger.warning(f"Error processing {file_path}: {e}")

    stats.scan_time_sec = time.time() - scan_start
    print()  # Newline after progress

    # Actually perform the conversion using the existing function
    logger.info(f"Writing LMDB for {data_type}...")
    write_start = time.time()
    json_2_lmdbs(
        root_dir=root_dir,
        out_dir=out_dir,
        data_type=data_type,
        out_lmdb=out_lmdb,
        chunk_size=chunk_size,
        clean=clean,
        merge=merge,
        move_files=move_files,
        limit=limit,
        shard_folders=shard_folders,
    )
    stats.lmdb_write_time_sec = time.time() - write_start

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
    limit: Optional[int] = None,
    shard_index: int = 0,
    total_shards: int = 1,
    shard_folders: Optional[List[str]] = None,
) -> ConversionStats:
    """Convert ORCA .inp files to structure LMDB with statistics."""
    stats = ConversionStats(data_type="structure")

    # Update output LMDB name for sharding
    if total_shards > 1:
        base_name = out_lmdb.replace(".lmdb", "")
        out_lmdb = f"{base_name}_shard_{shard_index}.lmdb"
        logger.info(f"Sharding enabled: output will be {out_lmdb}")

    # Find files - filter by shard assignment if sharding is enabled
    if total_shards > 1 and shard_folders:
        # Build patterns for assigned folders only
        files = []
        for folder in shard_folders:
            pattern = os.path.join(root_dir, folder, "*.inp")
            folder_files = glob(pattern)
            files.extend(folder_files)
        total_found = len(files)
        logger.debug(f"Shard {shard_index}: found {total_found} .inp files in {len(shard_folders)} assigned folders")
    else:
        # No sharding - process all files
        pattern = os.path.join(root_dir, "*/*.inp")
        files = glob(pattern)
        total_found = len(files)

    # Apply debug limit
    if limit is not None:
        files = files[:min(limit, len(files))]
        logger.debug(f"Debug mode: limiting to {len(files)} of {total_found} files")

    stats.files_found = len(files)
    logger.info(f"Found {stats.files_found} .inp files for structure conversion" +
                (f" (limited from {total_found})" if limit else ""))

    if stats.files_found == 0:
        logger.warning(f"No .inp files found matching pattern: {pattern}")
        return stats

    # Track processing with timing
    scan_start = time.time()
    progress_interval = max(1, stats.files_found // 20)  # Update ~20 times during scan

    for i, file_path in enumerate(files):
        try:
            file_size = os.path.getsize(file_path)
            stats.total_bytes_read += file_size

            if file_size == 0:
                stats.empty_files += 1
                stats.skipped_files.append(file_path)
                logger.debug(f"Empty file: {file_path}")
            else:
                stats.files_converted += 1

            # Dynamic progress update
            if (i + 1) % progress_interval == 0 or (i + 1) == stats.files_found:
                elapsed = time.time() - scan_start
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                print(f"\r  Scanning structure: {i + 1}/{stats.files_found} files ({rate:.1f} files/sec)", end="", flush=True)

        except Exception as e:
            stats.files_failed += 1
            stats.failed_files.append(file_path)
            logger.warning(f"Error checking {file_path}: {e}")

    stats.scan_time_sec = time.time() - scan_start
    print()  # Newline after progress

    # Perform the actual conversion
    logger.info("Writing structure LMDB...")
    write_start = time.time()
    inp_files_2_lmdbs(
        root_dir=root_dir,
        out_dir=out_dir,
        out_lmdb=out_lmdb,
        chunk_size=chunk_size,
        clean=clean,
        merge=merge,
        limit=limit,
        shard_folders=shard_folders,
    )
    stats.lmdb_write_time_sec = time.time() - write_start

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

            # Parallel processing with sharding (run these in separate jobs)
            # Note: Each shard auto-creates its own subdirectory (shard_0/, shard_1/, etc.)
            json-to-lmdb --root_dir ./jobs/ --out_dir ./lmdbs/ --all --shard_index 0 --total_shards 4
            json-to-lmdb --root_dir ./jobs/ --out_dir ./lmdbs/ --all --shard_index 1 --total_shards 4
            json-to-lmdb --root_dir ./jobs/ --out_dir ./lmdbs/ --all --shard_index 2 --total_shards 4
            json-to-lmdb --root_dir ./jobs/ --out_dir ./lmdbs/ --all --shard_index 3 --total_shards 4 --auto_merge
            # Results: lmdbs/shard_0/, lmdbs/shard_1/, ..., lmdbs/merged/
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
        "--debug",
        type=int,
        default=None,
        metavar="N",
        help="Debug mode: only process first N folders (default: 100 if flag used without value)",
        nargs="?",
        const=100,
    )

    parser.add_argument(
        "--clean_output",
        action="store_true",
        help="Delete existing LMDB files in output directory before conversion",
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

    parser.add_argument(
        "--shard_index",
        type=int,
        default=0,
        help="Shard index for parallel processing (0-based, default: 0)",
    )

    parser.add_argument(
        "--total_shards",
        type=int,
        default=1,
        help="Total number of shards for parallel processing (default: 1 = no sharding)",
    )

    parser.add_argument(
        "--auto_merge",
        action="store_true",
        default=False,
        help="Automatically merge shards after last shard completes (only applies to last shard)",
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.all and not args.types:
        parser.error("Must specify either --all or --types")

    # Validate sharding arguments
    if args.shard_index < 0:
        parser.error("--shard_index must be >= 0")
    if args.total_shards < 1:
        parser.error("--total_shards must be >= 1")
    if args.shard_index >= args.total_shards:
        parser.error(f"--shard_index ({args.shard_index}) must be less than --total_shards ({args.total_shards})")

    # Ensure root_dir ends with separator
    root_dir = args.root_dir
    if not root_dir.endswith(os.sep):
        root_dir += os.sep

    # Create output directory
    out_dir = args.out_dir
    if not out_dir.endswith(os.sep):
        out_dir += os.sep
    os.makedirs(out_dir, exist_ok=True)

    # Clean existing LMDB files if requested
    if args.clean_output:
        existing_lmdbs = glob(os.path.join(out_dir, "*.lmdb"))
        existing_locks = glob(os.path.join(out_dir, "*.lmdb-lock"))
        if existing_lmdbs or existing_locks:
            print(f"Cleaning {len(existing_lmdbs)} LMDB files and {len(existing_locks)} lock files from {out_dir}")
            for f in existing_lmdbs + existing_locks:
                try:
                    os.remove(f)
                except OSError as e:
                    print(f"Warning: Could not remove {f}: {e}")

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
    debug_limit = args.debug
    shard_index = args.shard_index
    total_shards = args.total_shards
    auto_merge = args.auto_merge

    # Setup logging
    logger = setup_logging(verbose=args.verbose, log_file=args.log_file)

    # Auto-create shard subdirectories to avoid chunk naming conflicts
    base_out_dir = out_dir  # Save original output directory for auto-merge
    if total_shards > 1:
        out_dir = os.path.join(out_dir, f"shard_{shard_index}")
        os.makedirs(out_dir, exist_ok=True)
        logger.info(f"Sharding: Creating output subdirectory {out_dir}")

    # Compute shard folder assignments
    shard_folders = None
    if total_shards > 1:
        shard_folders = partition_folders_by_shard(root_dir, shard_index, total_shards, logger)

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
    if total_shards > 1:
        logger.info(f"Sharding:            Shard {shard_index} of {total_shards}")
        if auto_merge and shard_index == total_shards - 1:
            logger.info(f"Auto-merge:          Enabled (last shard will merge)")
        logger.info(f"Assigned folders:    {len(shard_folders) if shard_folders else 0}")
    if args.clean_output:
        logger.info(f"Clean output:        Yes (removed existing LMDBs)")
    if debug_limit:
        logger.info(f"DEBUG MODE:          Limited to {debug_limit} folders")
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
                    limit=debug_limit,
                    shard_index=shard_index,
                    total_shards=total_shards,
                    shard_folders=shard_folders,
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
                    limit=debug_limit,
                    shard_index=shard_index,
                    total_shards=total_shards,
                    shard_folders=shard_folders,
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
    total_scan_time = sum(s.scan_time_sec for s in all_stats)
    total_write_time = sum(s.lmdb_write_time_sec for s in all_stats)
    total_bytes = sum(s.total_bytes_read for s in all_stats)

    logger.info(f"Total files found:     {total_found}")
    logger.info(f"Total files converted: {total_converted}")
    logger.info(f"Total files failed:    {total_failed}")
    logger.info(f"Total empty files:     {total_empty}")
    logger.info("")

    # Aggregate timing stats
    logger.info("--- Aggregate I/O Statistics ---")
    if total_scan_time > 0:
        avg_read_rate = total_converted / total_scan_time
        logger.info(f"Total scan/read time:  {total_scan_time:.2f}s")
        logger.info(f"Avg read rate:         {avg_read_rate:.1f} files/sec")
    if total_bytes > 0:
        total_mb = total_bytes / (1024 * 1024)
        mb_per_sec = total_mb / total_scan_time if total_scan_time > 0 else 0
        logger.info(f"Total data read:       {total_mb:.2f} MB ({mb_per_sec:.2f} MB/sec)")
    if total_write_time > 0:
        avg_write_rate = total_converted / total_write_time
        logger.info(f"Total LMDB write time: {total_write_time:.2f}s")
        logger.info(f"Avg write rate:        {avg_write_rate:.1f} files/sec")
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

    # Write sentinel file to signal this shard is fully done (chunks merged, all types complete)
    if total_shards > 1:
        shard_dir = os.path.join(base_out_dir, f"shard_{shard_index}")
        sentinel_path = os.path.join(shard_dir, f"shard_{shard_index}.done")
        with open(sentinel_path, "w") as f:
            f.write(f"shard {shard_index} completed at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"data types: {', '.join(converted)}\n")
        logger.info(f"Wrote sentinel file: {sentinel_path}")

    # Auto-merge shards if this is the last shard
    if auto_merge and total_shards > 1 and shard_index == total_shards - 1:
        logger.info("")
        logger.info("=" * 60)
        logger.info("AUTO-MERGE: Waiting for all shards to complete...")
        logger.info("=" * 60)

        # Poll for sentinel files from all shards before merging
        max_wait = 3600 * 6  # 6 hours
        poll_interval = 30  # seconds
        waited = 0

        expected_sentinels = [
            os.path.join(base_out_dir, f"shard_{i}", f"shard_{i}.done")
            for i in range(total_shards)
        ]

        while waited < max_wait:
            missing = [p for p in expected_sentinels if not os.path.exists(p)]
            if not missing:
                logger.info("All shard sentinel files detected. Proceeding with merge.")
                break
            logger.info(
                f"Waiting for {len(missing)}/{total_shards} shard(s) to finish... "
                f"({waited}s elapsed, polling every {poll_interval}s)"
            )
            for p in missing[:5]:
                logger.info(f"  waiting on: {os.path.basename(os.path.dirname(p))}")
            if len(missing) > 5:
                logger.info(f"  ... and {len(missing) - 5} more")
            time.sleep(poll_interval)
            waited += poll_interval
        else:
            missing = [p for p in expected_sentinels if not os.path.exists(p)]
            logger.error(
                f"Timed out after {max_wait}s waiting for {len(missing)} shard(s). "
                f"Proceeding with available shards only."
            )

        logger.info("=" * 60)
        logger.info("AUTO-MERGE: Starting automatic shard merging...")
        logger.info("=" * 60)

        merged_dir = os.path.join(base_out_dir, "merged")
        os.makedirs(merged_dir, exist_ok=True)

        # Build lookup of this shard's entry counts per data type for sanity checking
        last_shard_counts = {s.data_type: s.files_converted for s in all_stats}

        for data_type in converted:
            base_lmdb_name = f"{prefix}{DEFAULT_LMDB_NAMES[data_type]}"
            base_name = base_lmdb_name.replace(".lmdb", "")

            # Find all shard LMDBs for this data type (in shard subdirectories)
            shard_lmdbs = []
            for i in range(total_shards):
                # Look in each shard subdirectory
                shard_dir = os.path.join(base_out_dir, f"shard_{i}")
                shard_path = os.path.join(shard_dir, f"{base_name}_shard_{i}.lmdb")
                if os.path.exists(shard_path):
                    shard_lmdbs.append(shard_path)
                else:
                    logger.warning(f"Shard LMDB not found: {shard_path}")

            if len(shard_lmdbs) != total_shards:
                logger.warning(
                    f"Expected {total_shards} shards for {data_type} but found {len(shard_lmdbs)}. "
                    f"Skipping merge for {data_type}."
                )
                continue

            logger.info(f"Merging {data_type}: {len(shard_lmdbs)} shards")

            try:
                merged_path, merged_count = merge_shards(
                    shard_lmdbs=shard_lmdbs,
                    output_path=os.path.join(merged_dir, base_lmdb_name),
                    logger=logger,
                )
                logger.info(f"  -> Merged: {merged_path}")

                # Sanity check: merged total should be roughly
                # last_shard_count * total_shards (shards differ by at most 1 folder)
                my_count = last_shard_counts.get(data_type, 0)
                if my_count > 0:
                    expected = my_count * total_shards
                    deviation = abs(merged_count - expected) / expected
                    if deviation > 0.25:
                        logger.warning(
                            f"SANITY CHECK: {data_type} merged {merged_count} entries but "
                            f"expected ~{expected} (this shard had {my_count} x {total_shards} shards). "
                            f"Deviation: {deviation:.0%}. Some shards may have failed or processed different data."
                        )
                    else:
                        logger.info(
                            f"  Sanity check OK: {merged_count} merged vs ~{expected} expected "
                            f"({deviation:.1%} deviation)"
                        )

            except Exception as e:
                logger.error(f"Failed to merge {data_type} shards: {e}")
                import traceback
                logger.debug(traceback.format_exc())

        logger.info("=" * 60)
        logger.info(f"AUTO-MERGE COMPLETE: Merged LMDBs in {merged_dir}")
        logger.info("=" * 60)


if __name__ == "__main__":
    main()
