#!/usr/bin/env python3
"""
Generate sharded PBS job scripts from a template.

Takes a single json-to-lmdb PBS script and produces N shard scripts
with --shard_index, --total_shards, and --auto_merge on the last shard.

Usage:
    python shard_pbs_scripts.py --template geom_orca6.sh --num_shards 4
    python shard_pbs_scripts.py --template geom_orca6.sh --num_shards 8 --out_dir ./sharded/

Output:
    geom_orca6_0.sh, geom_orca6_1.sh, ..., geom_orca6_3.sh
"""

import argparse
import re
import os
from pathlib import Path


def generate_shard_scripts(template_path: str, num_shards: int, out_dir: str = None):
    template = Path(template_path)
    if not template.exists():
        raise FileNotFoundError(f"Template not found: {template}")

    text = template.read_text()
    stem = template.stem  # e.g. "geom_orca6"
    out_path = Path(out_dir) if out_dir else template.parent

    # Find the json-to-lmdb command line
    pattern = re.compile(r"^(\s*)(json-to-lmdb\s+.+)$", re.MULTILINE)
    match = pattern.search(text)
    if not match:
        raise ValueError("No 'json-to-lmdb' command found in template")

    indent = match.group(1)
    cmd = match.group(2)

    # Strip any existing shard args (will be re-added per shard)
    cmd_clean = re.sub(r"\s+--shard_index\s+\d+", "", cmd)
    cmd_clean = re.sub(r"\s+--total_shards\s+\d+", "", cmd_clean)
    cmd_clean = re.sub(r"\s+--auto_merge", "", cmd_clean)
    cmd_clean = cmd_clean.rstrip()

    os.makedirs(out_path, exist_ok=True)
    generated = []

    for i in range(num_shards):
        shard_cmd = f"{cmd_clean} --shard_index {i} --total_shards {num_shards}"
        if i == num_shards - 1:
            shard_cmd += " --auto_merge"

        shard_text = text.replace(match.group(0), f"{indent}{shard_cmd}")

        out_file = out_path / f"{stem}_{i}.sh"
        out_file.write_text(shard_text)
        out_file.chmod(0o755)
        generated.append(out_file)
        print(f"  {out_file}")

    return generated


def main():
    parser = argparse.ArgumentParser(description="Generate sharded PBS scripts from a template")
    parser.add_argument("--template", required=True, help="Path to the template PBS script")
    parser.add_argument("--num_shards", type=int, required=True, help="Number of shards to generate")
    parser.add_argument("--out_dir", default=None, help="Output directory (default: same as template)")
    args = parser.parse_args()

    print(f"Generating {args.num_shards} shard scripts from {args.template}:")
    generate_shard_scripts(args.template, args.num_shards, args.out_dir)
    print("Done.")


if __name__ == "__main__":
    main()
