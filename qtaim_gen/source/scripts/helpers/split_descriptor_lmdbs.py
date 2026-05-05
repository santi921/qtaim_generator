"""Partition per-vertical descriptor LMDBs into train/val/test splits.

Composition-ordered split via blake2b(formula_hill) % buckets, with
optional holdout exclusion. Writes per-vertical, per-descriptor LMDB
files preserving the original record format (no graph conversion). Use
when you want descriptor-level train/val/test for downstream tools that
operate on raw records (Multiwfn-style workflows, custom feature
extraction, anything not built on top of qtaim_embed graphs).

Pipeline per vertical:
  1. read structure.lmdb to build {lmdb_key: formula_hill}
  2. drop keys present in --holdout_parquet (typically
     manifest_holdout.parquet from pull_holdout_records)
  3. assign every remaining key to train/val/test by hashing its
     formula via assign_formula_to_split (deterministic, seeded)
  4. for each descriptor family, open the source LMDB read-only and
     three destination LMDBs (train / val / test) write-only; copy
     each record into the LMDB matching its assigned split

Output layout (one block per vertical):
  <output_dir>/<vertical>/<descriptor>_train.lmdb
  <output_dir>/<vertical>/<descriptor>_val.lmdb
  <output_dir>/<vertical>/<descriptor>_test.lmdb
  <output_dir>/<vertical>/split_assignment.json   # per-key split decision
  <output_dir>/split_plan.json                    # global summary

Sharded usage (mirror of pull_holdout_records):
  printf '%s\\n' "${VERTICALS[@]}" | xargs -P 8 -I {} \\
      python -m qtaim_gen.source.scripts.helpers.split_descriptor_lmdbs \\
          --lmdb_root /path/to/per_vertical_lmdbs \\
          --holdout_parquet /path/to/manifest_holdout.parquet \\
          --output_dir /path/to/splits \\
          --include_verticals {}

Each shard produces a per-vertical split_assignment.json plus its
descriptor LMDBs. A subsequent collation step (or a single un-sharded
final invocation) writes the global split_plan.json.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set

import lmdb

from qtaim_gen.source.utils.multi_vertical import load_exclusion_set
from qtaim_gen.source.utils.splits import (
    SPLIT_NAMES,
    assign_formula_to_split,
    build_formula_map_from_structure_lmdb,
)


DEFAULT_LMDB_ROOT = Path(
    "/home/santiagovargas/dev/qtaim_generator/data/OMol4M_lmdbs"
)
DEFAULT_HOLDOUT_PARQUET = Path(
    "/home/santiagovargas/dev/qtaim_generator/data/holdouts/manifest_holdout.parquet"
)
DEFAULT_OUT_DIR = Path(
    "/home/santiagovargas/dev/qtaim_generator/data/OMol4M_lmdbs/splits"
)
DEFAULT_DESCRIPTORS = (
    "structure", "charge", "bond", "qtaim", "fuzzy",
    "other", "orca", "timings",
)

# Subdirs of lmdb_root that are not per-vertical LMDB folders
NON_VERTICAL_DIRS = {
    "tm_bond_lists",
    "ln_bond_lists",
    "filter_csv_for_holdouts",
    "holdout_lmdbs",
    "holdouts",
    "splits",
}

LMDB_MAP_SIZE = 1099511627776 * 2  # 2 TiB sparse map (matches utils.lmdbs)


def discover_verticals(lmdb_root: Path) -> List[str]:
    out = []
    for child in sorted(lmdb_root.iterdir()):
        if not child.is_dir():
            continue
        if child.name in NON_VERTICAL_DIRS:
            continue
        out.append(child.name)
    return out


def find_lmdb(lmdb_root: Path, vertical: str, descriptor: str) -> Optional[Path]:
    direct = lmdb_root / vertical / f"{descriptor}.lmdb"
    if direct.exists():
        return direct
    merged = lmdb_root / vertical / "merged" / f"{descriptor}.lmdb"
    if merged.exists():
        return merged
    return None


def split_one_vertical(
    vertical: str,
    lmdb_root: Path,
    output_dir: Path,
    descriptors: Iterable[str],
    ratios: tuple,
    seed: int,
    excl_keys: Set[str],
) -> dict:
    """Process one vertical. Returns per-vertical summary dict."""
    out_v = output_dir / vertical
    out_v.mkdir(parents=True, exist_ok=True)

    structure_path = find_lmdb(lmdb_root, vertical, "structure")
    if structure_path is None:
        return {
            "vertical": vertical,
            "status": "no_structure_lmdb",
            "n_keys": 0,
        }

    print(f"[{vertical}] reading structure.lmdb ({structure_path})...")
    formula_map = build_formula_map_from_structure_lmdb(str(structure_path))
    n_total = len(formula_map)
    print(f"[{vertical}]   {n_total:,} keys, "
          f"{len(set(formula_map.values()))} unique formulas")

    # Compute split assignment per key (skip excluded keys entirely).
    n_excluded = 0
    assignment: Dict[str, str] = {}  # key -> split_name (excluded keys absent)
    per_split_keys: Dict[str, List[str]] = {s: [] for s in SPLIT_NAMES}

    for key, formula in formula_map.items():
        if key in excl_keys:
            n_excluded += 1
            continue
        split = assign_formula_to_split(formula, ratios, seed)
        assignment[key] = split
        per_split_keys[split].append(key)

    n_assigned = len(assignment)
    print(f"[{vertical}]   excluded={n_excluded:,}  assigned={n_assigned:,}  "
          + " ".join(f"{s}={len(per_split_keys[s]):,}" for s in SPLIT_NAMES))

    # Persist per-key assignment for audit.
    assignment_path = out_v / "split_assignment.json"
    with open(assignment_path, "w") as f:
        json.dump({
            "vertical": vertical,
            "ratios": list(ratios),
            "seed": seed,
            "n_total": n_total,
            "n_excluded": n_excluded,
            "per_split_counts": {s: len(per_split_keys[s]) for s in SPLIT_NAMES},
            "assignment": assignment,
        }, f)

    # Copy records per descriptor.
    desc_summary: List[dict] = []
    for descriptor in descriptors:
        src = find_lmdb(lmdb_root, vertical, descriptor)
        if src is None:
            print(f"[{vertical}]   {descriptor}: source LMDB missing, skipping")
            desc_summary.append({
                "descriptor": descriptor,
                "status": "no_source_lmdb",
            })
            continue

        # Open source read-only and three destination LMDBs write-mode.
        env_in = lmdb.open(
            str(src), subdir=False, readonly=True, lock=False,
            readahead=True, meminit=False,
        )
        envs_out = {}
        for s in SPLIT_NAMES:
            out_path = out_v / f"{descriptor}_{s}.lmdb"
            for ext in ("", "-lock"):
                p = Path(str(out_path) + ext)
                if p.exists():
                    p.unlink()
            envs_out[s] = lmdb.open(
                str(out_path),
                map_size=LMDB_MAP_SIZE,
                subdir=False,
                meminit=False,
                map_async=True,
            )

        per_split_written = {s: 0 for s in SPLIT_NAMES}
        per_split_missing = {s: 0 for s in SPLIT_NAMES}

        with env_in.begin() as txn_in:
            for s in SPLIT_NAMES:
                with envs_out[s].begin(write=True) as txn_out:
                    for k in per_split_keys[s]:
                        val = txn_in.get(k.encode("ascii"))
                        if val is None:
                            per_split_missing[s] += 1
                            continue
                        txn_out.put(k.encode("ascii"), val)
                        per_split_written[s] += 1
                    # length sentinel matches utils.lmdbs convention
                    txn_out.put(
                        b"length",
                        pickle.dumps(per_split_written[s], protocol=-1),
                    )
                envs_out[s].sync()
                envs_out[s].close()

        env_in.close()

        total_missing = sum(per_split_missing.values())
        print(f"[{vertical}]   {descriptor:<10} "
              + " ".join(
                  f"{s}={per_split_written[s]:,}" for s in SPLIT_NAMES
              )
              + (f"  missing={total_missing}" if total_missing else ""))
        desc_summary.append({
            "descriptor": descriptor,
            "status": "ok",
            "per_split_written": per_split_written,
            "per_split_missing": per_split_missing,
        })

    return {
        "vertical": vertical,
        "status": "ok",
        "n_total": n_total,
        "n_excluded": n_excluded,
        "n_assigned": n_assigned,
        "per_split_counts": {s: len(per_split_keys[s]) for s in SPLIT_NAMES},
        "descriptors": desc_summary,
    }


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--lmdb_root", type=Path, default=DEFAULT_LMDB_ROOT,
                   help=f"Per-vertical descriptor LMDB root. "
                        f"Default: {DEFAULT_LMDB_ROOT}")
    p.add_argument("--holdout_parquet", type=Path,
                   default=DEFAULT_HOLDOUT_PARQUET,
                   help=f"manifest_holdout.parquet from pull_holdout_records. "
                        f"Pass empty string to disable exclusion. "
                        f"Default: {DEFAULT_HOLDOUT_PARQUET}")
    p.add_argument("--output_dir", type=Path, default=DEFAULT_OUT_DIR,
                   help=f"Where to write <vertical>/<descriptor>_{{train,val,test}}.lmdb. "
                        f"Default: {DEFAULT_OUT_DIR}")
    p.add_argument("--descriptors", nargs="+", default=list(DEFAULT_DESCRIPTORS),
                   help="Descriptor families to split. "
                        f"Default: {' '.join(DEFAULT_DESCRIPTORS)}")
    p.add_argument("--ratios", nargs=3, type=float, default=[0.8, 0.1, 0.1],
                   metavar=("TRAIN", "VAL", "TEST"),
                   help="Composition split ratios (train val test). Default: 0.8 0.1 0.1")
    p.add_argument("--seed", type=int, default=42,
                   help="blake2b salt for composition hashing. Default: 42")
    p.add_argument("--include_verticals", nargs="*", default=None, metavar="V",
                   help="Process only these verticals (use to shard across "
                        "parallel jobs).")
    args = p.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"lmdb_root:        {args.lmdb_root}")
    print(f"holdout_parquet:  {args.holdout_parquet}")
    print(f"output_dir:       {args.output_dir}")
    print(f"descriptors:      {args.descriptors}")
    print(f"ratios:           {args.ratios}")
    print(f"seed:             {args.seed}")
    if args.include_verticals:
        print(f"include_verticals: {args.include_verticals}")

    ratios = tuple(args.ratios)
    if abs(sum(ratios) - 1.0) > 1e-6:
        print(f"error: ratios must sum to 1.0, got {ratios} (sum={sum(ratios)})",
              file=sys.stderr)
        return 2

    # Load exclusion set if provided.
    exclusion: Dict[str, Set[str]] = {}
    if args.holdout_parquet and str(args.holdout_parquet) != "" and args.holdout_parquet.exists():
        print(f"\nloading exclusion set from {args.holdout_parquet}...")
        exclusion = load_exclusion_set(str(args.holdout_parquet))
        total_excl = sum(len(s) for s in exclusion.values())
        print(f"  exclusion set: {total_excl:,} keys across "
              f"{len(exclusion)} verticals")
    elif args.holdout_parquet and str(args.holdout_parquet) != "":
        print(f"\nwarning: --holdout_parquet={args.holdout_parquet} not found, "
              f"running with no exclusion", file=sys.stderr)

    # Discover verticals to process.
    available = discover_verticals(args.lmdb_root)
    if args.include_verticals:
        keep = set(args.include_verticals)
        unknown = sorted(keep - set(available))
        if unknown:
            print(f"warning: --include_verticals contains {len(unknown)} "
                  f"unknown verticals: {unknown}", file=sys.stderr)
        available = [v for v in available if v in keep]

    print(f"\nverticals to process: {len(available)}")
    for v in available:
        print(f"  {v}")

    summaries: List[dict] = []
    for v in available:
        summary = split_one_vertical(
            v, args.lmdb_root, args.output_dir, args.descriptors,
            ratios, args.seed, exclusion.get(v, set()),
        )
        summaries.append(summary)

    # Global summary.
    plan_path = args.output_dir / "split_plan.json"
    # Merge into existing plan if present (sharded runs).
    existing = {}
    if plan_path.exists():
        try:
            existing = json.loads(plan_path.read_text())
        except Exception:
            existing = {}
    plan_data = {
        "lmdb_root": str(args.lmdb_root),
        "holdout_parquet": str(args.holdout_parquet),
        "ratios": list(ratios),
        "seed": args.seed,
        "descriptors": list(args.descriptors),
        "verticals": existing.get("verticals", {}),
    }
    for s in summaries:
        plan_data["verticals"][s["vertical"]] = s
    with open(plan_path, "w") as f:
        json.dump(plan_data, f, indent=2)
    print(f"\nwrote split_plan.json -> {plan_path}")

    # Aggregate counts across this run only.
    run_total = sum(s.get("n_total", 0) for s in summaries)
    run_excl = sum(s.get("n_excluded", 0) for s in summaries)
    run_per_split = {sn: 0 for sn in SPLIT_NAMES}
    for s in summaries:
        for sn in SPLIT_NAMES:
            run_per_split[sn] += s.get("per_split_counts", {}).get(sn, 0)
    print(f"\nthis run: total={run_total:,}  excluded={run_excl:,}  "
          + " ".join(f"{sn}={run_per_split[sn]:,}" for sn in SPLIT_NAMES))

    return 0


if __name__ == "__main__":
    sys.exit(main())
