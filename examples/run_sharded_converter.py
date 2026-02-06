"""
Example script for running sharded converter processing.

This demonstrates how to:
1. Run multiple shards in parallel (manually or with job scheduler)
2. Merge the shard outputs
3. Verify the results

For HPC clusters, you can use SLURM array jobs or similar to run shards in parallel.
"""

import json
import os
from pathlib import Path
from qtaim_gen.source.core.converter import BaseConverter


def run_single_shard(
    config_path: str,
    shard_index: int,
    total_shards: int,
    output_base_dir: str
):
    """
    Run a single shard of the conversion.

    Args:
        config_path: Path to base config JSON
        shard_index: Index of this shard (0 to total_shards-1)
        total_shards: Total number of shards
        output_base_dir: Base directory for shard outputs
    """
    # Load base config
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Update config for this shard
    shard_dir = os.path.join(output_base_dir, f"shard_{shard_index}")
    os.makedirs(shard_dir, exist_ok=True)

    config["shard_index"] = shard_index
    config["total_shards"] = total_shards
    config["skip_scaling"] = True  # Scale after merging
    config["save_unfinalized_scaler"] = True  # Save for merging
    config["lmdb_path"] = shard_dir

    # Create and run converter
    print(f"Starting shard {shard_index}/{total_shards}...")
    converter = BaseConverter(
        config,
        config_path=os.path.join(shard_dir, "config.json")
    )
    converter.process(return_info=True)
    print(f"Shard {shard_index} complete!")


def merge_shards(output_base_dir: str, total_shards: int):
    """
    Merge all shard outputs into a single LMDB.

    Args:
        output_base_dir: Base directory containing shard subdirectories
        total_shards: Number of shards to merge
    """
    shard_dirs = [
        os.path.join(output_base_dir, f"shard_{i}")
        for i in range(total_shards)
    ]

    # Verify all shards exist
    for shard_dir in shard_dirs:
        if not os.path.exists(shard_dir):
            raise ValueError(f"Shard directory not found: {shard_dir}")

    merged_dir = os.path.join(output_base_dir, "merged")
    os.makedirs(merged_dir, exist_ok=True)

    print(f"Merging {total_shards} shards...")
    merged_path = BaseConverter.merge_shards(
        shard_dirs=shard_dirs,
        output_dir=merged_dir,
        output_name="merged.lmdb",
        skip_scaling=False  # Apply scalers during merge
    )
    print(f"Merge complete: {merged_path}")


def run_sharded_pipeline(config_path: str, total_shards: int, output_base_dir: str):
    """
    Run complete sharded pipeline: process shards, then merge.

    For production use on HPC, you would typically:
    1. Submit shard jobs as array job (e.g., SLURM --array=0-7)
    2. Wait for all shards to complete
    3. Run merge as a separate job with dependencies

    This example runs sequentially for demonstration.
    """
    print(f"=== Running {total_shards}-shard converter pipeline ===")
    print(f"Config: {config_path}")
    print(f"Output: {output_base_dir}")

    # Step 1: Process shards
    print("\n=== Step 1: Processing shards ===")
    for shard_idx in range(total_shards):
        run_single_shard(config_path, shard_idx, total_shards, output_base_dir)

    # Step 2: Merge shards
    print("\n=== Step 2: Merging shards ===")
    merge_shards(output_base_dir, total_shards)

    print("\n=== Pipeline complete! ===")


# Example SLURM script (save as run_shard_array.sh):
SLURM_EXAMPLE = """
#!/bin/bash
#SBATCH --array=0-7           # 8 shards (0-7)
#SBATCH --cpus-per-task=1     # Single-threaded per shard
#SBATCH --time=02:00:00
#SBATCH --mem=8GB

# Set threading to avoid conflicts
export MKL_THREADING_LAYER=GNU
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1

# Activate environment
source ~/.conda/envs/generator/bin/activate

# Run shard
python -c "
from qtaim_gen.source.core.converter import BaseConverter
import json
import os

# Load config
with open('config.json', 'r') as f:
    config = json.load(f)

# Set shard parameters
shard_idx = int(os.environ['SLURM_ARRAY_TASK_ID'])
config['shard_index'] = shard_idx
config['total_shards'] = 8
config['skip_scaling'] = True
config['save_unfinalized_scaler'] = True
config['lmdb_path'] = f'output/shard_{shard_idx}'

# Run
converter = BaseConverter(config, config_path=f'output/shard_{shard_idx}/config.json')
converter.process()
"

# After all shards complete, run merge:
# python -c "from qtaim_gen.source.core.converter import BaseConverter; \\
#           BaseConverter.merge_shards(['output/shard_0', ...], 'output/merged')"
"""


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 4:
        print("Usage: python run_sharded_converter.py <config.json> <num_shards> <output_dir>")
        print("\nExample:")
        print("  python run_sharded_converter.py base_config.json 4 output/")
        print("\nFor SLURM array job example, see SLURM_EXAMPLE in this script")
        sys.exit(1)

    config_path = sys.argv[1]
    total_shards = int(sys.argv[2])
    output_dir = sys.argv[3]

    run_sharded_pipeline(config_path, total_shards, output_dir)
