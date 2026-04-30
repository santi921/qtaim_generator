"""Compute transition-metal neighbor lists per vertical from bond.lmdb files.

Walks one or more vertical directories that each contain a `bond.lmdb`,
extracts every bond involving a transition metal atom (Sc-Zn, Y-Cd, Hf-Hg),
and writes per-row neighbor records plus aggregated TM-partner pair counts.
Verticals with no TM-bearing rows are skipped.

Output layout under --output_dir:
    {vertical}/tm_neighbors.parquet   one row per (rel_path, TM atom, partner atom)
    {vertical}/tm_pair_counts.csv     (tm_symbol, partner_symbol) -> count, n_rows
    summary.json                      {vertical: {n_rows_total, n_rows_with_tm, n_bond_records}}
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from collections import Counter, defaultdict
from typing import Iterable, List, Optional, Tuple

import lmdb
import numpy as np
import pandas as pd

from qtaim_gen.source.utils.element_classes import (
    SYMBOL_TO_Z,
    TRANSITION_METALS,
)


def find_bond_lmdbs(root: str) -> List[Tuple[str, str]]:
    """Locate bond.lmdb files under root.

    Returns list of (vertical_name, lmdb_path). vertical_name is the
    directory holding bond.lmdb expressed as a path relative to root,
    with separators replaced by underscores for use as an output dirname.
    """
    pairs: List[Tuple[str, str]] = []
    for dirpath, _dirnames, filenames in os.walk(root):
        if "bond.lmdb" in filenames:
            rel = os.path.relpath(dirpath, root)
            vertical = "root" if rel == "." else rel.replace(os.sep, "__")
            pairs.append((vertical, os.path.join(dirpath, "bond.lmdb")))
    pairs.sort()
    return pairs


def parse_bond_key(s: str) -> Optional[Tuple[int, str, int, str]]:
    """Parse '1_O_to_2_C' style key into (idx_a, sym_a, idx_b, sym_b).

    Returns None when the key is malformed.
    """
    try:
        left, right = s.split("_to_")
        lp = left.split("_", 1)
        rp = right.split("_", 1)
        idx_a = int(lp[0]) - 1
        idx_b = int(rp[0]) - 1
        sym_a = lp[1]
        sym_b = rp[1]
    except (ValueError, IndexError):
        return None
    return idx_a, sym_a, idx_b, sym_b


def select_bond_field(bond_dict: dict, candidates: Iterable[str]) -> Optional[str]:
    """Pick the first present field name from candidates.

    Each candidate is matched both as-is and with a `_bond` suffix variant.
    """
    keys = set(bond_dict.keys())
    for cand in candidates:
        if cand in keys:
            return cand
        alt = cand if cand.endswith("_bond") else cand + "_bond"
        if alt in keys:
            return alt
    return None


def process_vertical(
    lmdb_path: str,
    bond_keys: List[str],
    cutoff: float,
) -> Tuple[List[dict], dict]:
    """Walk one bond.lmdb and emit per-bond TM records.

    Returns (records, stats). stats: n_rows_total, n_rows_with_tm,
    n_bond_records, n_rows_no_bond_field, field_usage Counter (as dict).
    """
    records: List[dict] = []
    n_rows_total = 0
    n_rows_with_tm = 0
    n_rows_no_bond_field = 0
    field_usage: Counter = Counter()

    env = lmdb.open(lmdb_path, readonly=True, lock=False, subdir=False, max_readers=1024)
    try:
        with env.begin() as txn:
            for raw_key, raw_val in txn.cursor():
                key_str = raw_key.decode("utf-8", errors="replace")
                if key_str == "length":
                    continue
                n_rows_total += 1
                try:
                    bond_dict = pickle.loads(raw_val)
                except Exception:
                    continue
                if not isinstance(bond_dict, dict):
                    continue

                field = select_bond_field(bond_dict, bond_keys)
                if field is None:
                    n_rows_no_bond_field += 1
                    continue
                field_usage[field] += 1

                row_has_tm = False
                inner = bond_dict[field]
                if not isinstance(inner, dict):
                    continue
                for bk, bv in inner.items():
                    parsed = parse_bond_key(bk)
                    if parsed is None:
                        continue
                    idx_a, sym_a, idx_b, sym_b = parsed
                    z_a = SYMBOL_TO_Z.get(sym_a)
                    z_b = SYMBOL_TO_Z.get(sym_b)
                    if z_a is None or z_b is None:
                        continue
                    a_is_tm = z_a in TRANSITION_METALS
                    b_is_tm = z_b in TRANSITION_METALS
                    if not (a_is_tm or b_is_tm):
                        continue
                    try:
                        val = float(bv)
                    except (TypeError, ValueError):
                        continue
                    if not np.isfinite(val) or val < cutoff:
                        continue

                    # Emit one row per (TM_atom, partner_atom). When both atoms
                    # are TM (rare; metal-metal bond), emit twice so each TM
                    # sees the partner.
                    if a_is_tm:
                        records.append({
                            "rel_path": key_str,
                            "tm_atom_idx": idx_a,
                            "tm_symbol": sym_a,
                            "tm_z": z_a,
                            "partner_atom_idx": idx_b,
                            "partner_symbol": sym_b,
                            "partner_z": z_b,
                            "bond_order": val,
                            "bond_field": field,
                        })
                        row_has_tm = True
                    if b_is_tm:
                        records.append({
                            "rel_path": key_str,
                            "tm_atom_idx": idx_b,
                            "tm_symbol": sym_b,
                            "tm_z": z_b,
                            "partner_atom_idx": idx_a,
                            "partner_symbol": sym_a,
                            "partner_z": z_a,
                            "bond_order": val,
                            "bond_field": field,
                        })
                        row_has_tm = True
                if row_has_tm:
                    n_rows_with_tm += 1
    finally:
        env.close()

    stats = {
        "n_rows_total": n_rows_total,
        "n_rows_with_tm": n_rows_with_tm,
        "n_bond_records": len(records),
        "n_rows_no_bond_field": n_rows_no_bond_field,
        "field_usage": dict(field_usage),
    }
    return records, stats


def write_outputs(out_dir: str, records: List[dict]) -> None:
    os.makedirs(out_dir, exist_ok=True)
    df = pd.DataFrame.from_records(records)
    df.to_parquet(os.path.join(out_dir, "tm_neighbors.parquet"), index=False)

    # Pair counts: total bond records and unique rows per (tm_symbol, partner_symbol)
    pair_total = (
        df.groupby(["tm_symbol", "partner_symbol"], sort=True)
          .size()
          .rename("count")
          .reset_index()
    )
    pair_rows = (
        df.drop_duplicates(["rel_path", "tm_symbol", "partner_symbol"])
          .groupby(["tm_symbol", "partner_symbol"], sort=True)
          .size()
          .rename("n_rows")
          .reset_index()
    )
    summary = pair_total.merge(pair_rows, on=["tm_symbol", "partner_symbol"], how="left")
    summary.to_csv(os.path.join(out_dir, "tm_pair_counts.csv"), index=False)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True, help="Directory to scan for bond.lmdb files (one per vertical).")
    p.add_argument("--output_dir", required=True, help="Where per-vertical outputs and summary.json are written.")
    p.add_argument(
        "--bond_keys",
        nargs="+",
        default=["mayer_orca", "fuzzy_bond"],
        help="Bond-order fields to try in order. The first present in a row is used. "
             "Each is also matched with a '_bond' suffix variant.",
    )
    p.add_argument("--cutoff", type=float, default=0.3, help="Minimum bond order to keep a bond. Default 0.3.")
    p.add_argument(
        "--include_empty",
        action="store_true",
        help="Also write outputs for verticals that produced zero TM records (default skips).",
    )
    args = p.parse_args(argv)

    if not os.path.isdir(args.root):
        print(f"error: --root {args.root!r} is not a directory", file=sys.stderr)
        return 2
    os.makedirs(args.output_dir, exist_ok=True)

    pairs = find_bond_lmdbs(args.root)
    if not pairs:
        print(f"no bond.lmdb files found under {args.root}", file=sys.stderr)
        return 1
    print(f"found {len(pairs)} bond.lmdb file(s) under {args.root}")

    summary: dict = {}
    for vertical, lmdb_path in pairs:
        print(f"  processing {vertical}: {lmdb_path}")
        records, stats = process_vertical(lmdb_path, args.bond_keys, args.cutoff)
        stats["lmdb_path"] = lmdb_path
        summary[vertical] = stats

        if not records and not args.include_empty:
            print(
                f"    skipped (no TM records). rows_total={stats['n_rows_total']} "
                f"no_bond_field={stats['n_rows_no_bond_field']}"
            )
            stats["skipped"] = True
            continue
        stats["skipped"] = False

        out_dir = os.path.join(args.output_dir, vertical)
        write_outputs(out_dir, records)
        print(
            f"    wrote {len(records)} bond records covering "
            f"{stats['n_rows_with_tm']} / {stats['n_rows_total']} rows -> {out_dir}"
        )

    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print(f"summary written to {os.path.join(args.output_dir, 'summary.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
