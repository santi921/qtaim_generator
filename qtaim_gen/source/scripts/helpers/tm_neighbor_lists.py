"""Compute metal-class neighbor lists per vertical from bond.lmdb files.

Walks one or more vertical directories that each contain a `bond.lmdb`,
extracts every bond involving an atom in the chosen element class
(transition metals, lanthanides, or actinides), and writes per-row
neighbor records plus aggregated metal-partner pair counts. Verticals
with no qualifying rows are skipped.

The element class is selected via --element_class {tm,ln,an} (default tm).
Output filenames and CSV column names use the class prefix (e.g.
`tm_neighbors.csv` with columns `tm_atom_idx, tm_symbol, tm_z` for tm,
or `ln_neighbors.csv` with `ln_atom_idx, ln_symbol, ln_z` for ln). This
keeps an Ln extraction's outputs distinct from a prior TM extraction
written to the same parent directory.

No pandas/numpy dependency - uses stdlib csv/json only.

Output layout under --output_dir:
    {vertical}/{prefix}_neighbors.csv    one row per (rel_path, metal atom, partner atom)
    {vertical}/{prefix}_pair_counts.csv  ({prefix}_symbol, partner_symbol) -> count, n_rows
    summary.json                         {vertical: {n_rows_total, n_rows_with_{prefix}, n_bond_records, ...}}
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pickle
import sys
from collections import Counter, defaultdict
from typing import FrozenSet, Iterable, List, Optional, Tuple

import lmdb

from qtaim_gen.source.utils.element_classes import (
    ACTINIDES,
    LANTHANIDES,
    SYMBOL_TO_Z,
    TRANSITION_METALS,
)


# Element-class registry: cli flag -> (prefix used in filenames/columns,
# atomic-number set, human label for messages).
ELEMENT_CLASSES = {
    "tm": ("tm", TRANSITION_METALS, "transition metal"),
    "ln": ("ln", LANTHANIDES,       "lanthanide"),
    "an": ("an", ACTINIDES,         "actinide"),
}


def find_bond_lmdbs(root: str) -> List[Tuple[str, str]]:
    """Locate bond.lmdb files under root.

    Returns list of (vertical_name, lmdb_path). vertical_name is the
    directory holding bond.lmdb expressed relative to root, with os.sep
    replaced by '__'.
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
    """Return the first field from candidates present in bond_dict.

    Each candidate is matched both as-is and with a '_bond' suffix variant.
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
    out_dir: str,
    prefix: str,
    metal_z_set: FrozenSet[int],
) -> dict:
    """Walk one bond.lmdb and write per-bond metal records as CSV.

    `prefix` is used for output filenames and metal-side column names
    (e.g. 'tm' -> tm_neighbors.csv with tm_atom_idx, tm_symbol, tm_z).
    `metal_z_set` is the set of atomic numbers that count as 'metal'
    for filtering; partner atoms are anything not in this set OR another
    metal of the same class (metal-metal bonds are emitted twice).

    Returns stats dict; the metal-row counter is keyed n_rows_with_{prefix}.
    """
    neighbors_fields = [
        "rel_path",
        f"{prefix}_atom_idx", f"{prefix}_symbol", f"{prefix}_z",
        "partner_atom_idx", "partner_symbol", "partner_z",
        "bond_order", "bond_field",
    ]
    pair_fields = [f"{prefix}_symbol", "partner_symbol", "count", "n_rows"]

    n_rows_total = 0
    n_rows_with_metal = 0
    n_rows_no_bond_field = 0
    n_bond_records = 0
    field_usage: Counter = Counter()

    # pair -> (total_bond_count, set_of_rel_paths)
    pair_counts: dict = defaultdict(lambda: [0, set()])

    os.makedirs(out_dir, exist_ok=True)
    neighbors_path = os.path.join(out_dir, f"{prefix}_neighbors.csv")

    with open(neighbors_path, "w", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=neighbors_fields)
        writer.writeheader()

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

                    row_has_metal = False
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
                        a_is_metal = z_a in metal_z_set
                        b_is_metal = z_b in metal_z_set
                        if not (a_is_metal or b_is_metal):
                            continue
                        try:
                            val = float(bv)
                        except (TypeError, ValueError):
                            continue
                        if not math.isfinite(val) or val < cutoff:
                            continue

                        if a_is_metal:
                            writer.writerow({
                                "rel_path": key_str,
                                f"{prefix}_atom_idx": idx_a,
                                f"{prefix}_symbol": sym_a,
                                f"{prefix}_z": z_a,
                                "partner_atom_idx": idx_b,
                                "partner_symbol": sym_b,
                                "partner_z": z_b,
                                "bond_order": val,
                                "bond_field": field,
                            })
                            pair = (sym_a, sym_b)
                            pair_counts[pair][0] += 1
                            pair_counts[pair][1].add(key_str)
                            n_bond_records += 1
                            row_has_metal = True
                        if b_is_metal:
                            writer.writerow({
                                "rel_path": key_str,
                                f"{prefix}_atom_idx": idx_b,
                                f"{prefix}_symbol": sym_b,
                                f"{prefix}_z": z_b,
                                "partner_atom_idx": idx_a,
                                "partner_symbol": sym_a,
                                "partner_z": z_a,
                                "bond_order": val,
                                "bond_field": field,
                            })
                            pair = (sym_b, sym_a)
                            pair_counts[pair][0] += 1
                            pair_counts[pair][1].add(key_str)
                            n_bond_records += 1
                            row_has_metal = True
                    if row_has_metal:
                        n_rows_with_metal += 1
        finally:
            env.close()

    if n_bond_records == 0:
        os.remove(neighbors_path)

    # write pair counts
    if pair_counts:
        pair_path = os.path.join(out_dir, f"{prefix}_pair_counts.csv")
        rows = sorted(
            [(metal_sym, partner, cnt, len(relpaths))
             for (metal_sym, partner), (cnt, relpaths) in pair_counts.items()],
            key=lambda r: (-r[2], r[0], r[1]),
        )
        with open(pair_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(pair_fields)
            writer.writerows(rows)

    return {
        "n_rows_total": n_rows_total,
        f"n_rows_with_{prefix}": n_rows_with_metal,
        "n_bond_records": n_bond_records,
        "n_rows_no_bond_field": n_rows_no_bond_field,
        "field_usage": dict(field_usage),
    }


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True, help="Directory to scan for bond.lmdb files (one per vertical).")
    p.add_argument("--output_dir", required=True, help="Where per-vertical outputs and summary.json are written.")
    p.add_argument(
        "--bond_keys",
        nargs="+",
        default=["mayer_orca", "fuzzy_bond"],
        help="Bond-order fields to try in order. First present in a row is used. "
             "Each is also matched with a '_bond' suffix variant.",
    )
    p.add_argument("--cutoff", type=float, default=0.3, help="Minimum bond order to keep. Default 0.3.")
    p.add_argument(
        "--element_class",
        choices=sorted(ELEMENT_CLASSES),
        default="tm",
        help="Which element class to extract metal-side bonds for: "
             "tm=transition metals (Sc-Zn, Y-Cd, Hf-Hg), "
             "ln=lanthanides (La-Lu), "
             "an=actinides (Ac-Lr). Default: tm.",
    )
    p.add_argument(
        "--include_verticals",
        nargs="+",
        default=None,
        metavar="V",
        help="If set, process only these vertical names (matched against the discovered vertical name).",
    )
    p.add_argument(
        "--include_empty",
        action="store_true",
        help="Write outputs even for verticals with zero metal-bearing records.",
    )
    args = p.parse_args(argv)

    prefix, metal_z_set, label = ELEMENT_CLASSES[args.element_class]

    if not os.path.isdir(args.root):
        print(f"error: --root {args.root!r} is not a directory", file=sys.stderr)
        return 2
    os.makedirs(args.output_dir, exist_ok=True)

    pairs = find_bond_lmdbs(args.root)
    if not pairs:
        print(f"no bond.lmdb files found under {args.root}", file=sys.stderr)
        return 1

    if args.include_verticals:
        allowed = set(args.include_verticals)
        pairs = [(v, p) for v, p in pairs if v in allowed]
        if not pairs:
            print(f"error: none of the requested verticals found under {args.root}", file=sys.stderr)
            return 1

    print(f"found {len(pairs)} bond.lmdb file(s) to process  ({label} class, prefix '{prefix}')")

    summary: dict = {}
    for vertical, lmdb_path in pairs:
        print(f"  {vertical}: {lmdb_path}")
        out_dir = os.path.join(args.output_dir, vertical)
        stats = process_vertical(
            lmdb_path, args.bond_keys, args.cutoff, out_dir, prefix, metal_z_set
        )
        stats["lmdb_path"] = lmdb_path
        stats["element_class"] = args.element_class

        metal_rows = stats[f"n_rows_with_{prefix}"]
        if stats["n_bond_records"] == 0 and not args.include_empty:
            stats["skipped"] = True
            print(
                f"    skipped (no {label} records)  rows_total={stats['n_rows_total']} "
                f"no_bond_field={stats['n_rows_no_bond_field']}"
            )
            try:
                os.rmdir(out_dir)
            except OSError:
                pass
        else:
            stats["skipped"] = False
            print(
                f"    {stats['n_bond_records']} bond records / "
                f"{metal_rows} {label} rows / "
                f"{stats['n_rows_total']} total"
            )
        summary[vertical] = stats

    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print(f"summary -> {os.path.join(args.output_dir, 'summary.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
