"""Rebuild the H1/H3/H7/H8 evaluation holdout filter CSVs.

Reference script for the OMol4M descriptor-dataset paper. Each invocation
writes per-holdout CSVs to `--output_dir` (default
`data/OMol4M_lmdbs/filter_csv_for_holdouts`). Schema is the same across
all holdouts: one row per (vertical, rel_path), suitable as a join key
into any of the per-vertical descriptor LMDBs.

Holdout definitions (see paper outline section 8.3.1 for rationale):

  H1  Metal-ligand pairs.
      17-pair stratified sample across log-spaced (TM, partner) bond
      frequency bands. Bond data is read from the per-vertical
      `tm_neighbors.{csv,parquet}` files produced by
      `qtaim_gen.source.scripts.helpers.tm_neighbor_lists`.
      Default shape (3, 2, 2, 4, 0) with seed 87 lands at ~15k structures
      (2.31% of TM-bonded, 0.38% of dataset).

  H3  Reactivity.
      Composition-stratified subsample of three reactivity verticals
      (`tm_react`, `electrolytes_reactivity`, `pmechdb`). Per-vertical
      target sizes are hashed by `formula_hill` to keep the sample
      deterministic and composition-coherent (no formula split between
      held-out and training). Default targets sum to ~12.5k structures.

  H7  Large systems.
      Structures with `n_atoms > 250`, ~18k structures (0.46% of dataset).

  H8  Weird charges.
      Structures with strict `|net_charge_abs| > 4` (i.e. |q| >= 5),
      ~12k structures (0.31% of dataset).

H2 (PDB-TM) is documented as a known limitation rather than a holdout:
the public OMol25 4M release excludes metal-containing protein
structures from the main training split, so the PDB-TM hold-out is
empty in this slice and only becomes evaluable on the full OMol25
release.

Usage:
    python -m qtaim_gen.source.analysis.build_holdout_csvs \\
        --manifest_dir data/omol_manifest \\
        --bond_root data/OMol4M_lmdbs/tm_bond_lists \\
        --output_dir data/OMol4M_lmdbs/filter_csv_for_holdouts
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


# --------------------------------------------------------------------------- defaults
DEFAULT_MANIFEST_DIR = Path(
    "/home/santiagovargas/dev/qtaim_generator/data/omol_manifest"
)
DEFAULT_BOND_ROOT = Path(
    "/home/santiagovargas/dev/qtaim_generator/data/OMol4M_lmdbs/tm_bond_lists"
)
DEFAULT_OUT_DIR = Path(
    "/home/santiagovargas/dev/qtaim_generator/data/OMol4M_lmdbs/filter_csv_for_holdouts"
)

# H1 stratified sampling
H1_BAND_EDGES = (1, 10, 100, 1000, 10000, 100000)
H1_SHAPE = (3, 2, 2, 4, 0)
H1_SEED = 87

# H3 per-vertical composition-stratified subsample targets
H3_TARGETS = {
    "tm_react": 5000,
    "electrolytes_reactivity": 5000,
    "pmechdb": 2500,
}
H3_SEED = 87

# H7 / H8 thresholds (strict)
H7_NATOMS_THRESHOLD = 250
H8_CHARGE_THRESHOLD = 4


# --------------------------------------------------------------------------- helpers
def load_manifest(manifest_dir: Path) -> pd.DataFrame:
    parts = sorted(p for p in manifest_dir.glob("manifest_*.parquet")
                   if p.name != "manifest.parquet")
    if not parts:
        raise FileNotFoundError(
            f"no per-vertical manifest_*.parquet files under {manifest_dir}"
        )
    df = pd.concat([pq.read_table(p).to_pandas() for p in parts],
                   ignore_index=True)
    return df[df["read_status"] == "ok"].copy()


def load_tm_neighbors(bond_root: Path) -> pd.DataFrame:
    cols = ["rel_path", "tm_symbol", "partner_symbol"]
    frames: List[pd.DataFrame] = []
    for v in sorted(d.name for d in bond_root.iterdir() if d.is_dir()):
        found = None
        for sub in ("merged", "root"):
            for ext in ("csv", "parquet"):
                cand = bond_root / v / sub / f"tm_neighbors.{ext}"
                if cand.exists():
                    found = cand
                    break
            if found is not None:
                break
        if found is None:
            continue
        sd = (pd.read_csv(found, usecols=cols) if found.suffix == ".csv"
              else pd.read_parquet(found, columns=cols))
        sd["vertical"] = v
        frames.append(sd)
    if not frames:
        raise FileNotFoundError(f"no tm_neighbors.* files found under {bond_root}")
    df = pd.concat(frames, ignore_index=True)
    df["pair"] = df["tm_symbol"] + "-" + df["partner_symbol"]
    return df


def formula_rank(formula: str, salt: int) -> int:
    """Deterministic per-formula hash for composition-stratified sampling."""
    h = hashlib.blake2b(
        f"{salt}|{formula}".encode("utf-8"), digest_size=8
    ).digest()
    return int.from_bytes(h, "big")


# --------------------------------------------------------------------------- H1
def build_h1(
    bond_df: pd.DataFrame,
    out_dir: Path,
    shape: Iterable[int] = H1_SHAPE,
    seed: int = H1_SEED,
    edges: Iterable[int] = H1_BAND_EDGES,
) -> pd.DataFrame:
    sp = bond_df[["vertical", "rel_path", "pair"]].drop_duplicates()
    n_tm_struct = sp[["vertical", "rel_path"]].drop_duplicates().shape[0]
    pair_struct = sp.groupby("pair").apply(
        lambda g: set(zip(g["vertical"], g["rel_path"])), include_groups=False
    )
    pair_size = pair_struct.map(len)

    edges = list(edges)
    band_pairs = [
        sorted(pair_size[(pair_size >= lo) & (pair_size < hi)].index.tolist())
        for lo, hi in zip(edges[:-1], edges[1:])
    ]
    band_labels = [f"B{i+1}_[{edges[i]},{edges[i+1]})"
                   for i in range(len(edges) - 1)]

    rng = np.random.default_rng(seed)
    selected: List[dict] = []
    for lbl, ps, n in zip(band_labels, band_pairs, shape):
        k = min(n, len(ps))
        if k == 0:
            continue
        for pair in rng.choice(ps, size=k, replace=False).tolist():
            tm_sym, partner_sym = pair.split("-", 1)
            selected.append({
                "holdout_id": "H1", "pair": pair,
                "tm_symbol": tm_sym, "partner_symbol": partner_sym,
                "n_structures_in_pair": int(pair_size[pair]),
                "band": lbl, "shape": str(list(shape)), "seed": seed,
            })

    defs_df = pd.DataFrame(selected).sort_values(
        ["band", "n_structures_in_pair"]
    ).reset_index(drop=True)
    defs_df.to_csv(out_dir / "h1_metal_ligand_pair_definitions.csv", index=False)

    membership = sp[sp["pair"].isin(set(defs_df["pair"]))].copy()
    agg = (
        membership.groupby(["vertical", "rel_path"])["pair"]
        .apply(lambda s: ",".join(sorted(set(s))))
        .reset_index().rename(columns={"pair": "matched_pairs"})
    )
    agg["holdout_id"] = "H1"
    agg["n_matched_pairs"] = agg["matched_pairs"].str.count(",") + 1
    agg = (agg[["holdout_id", "vertical", "rel_path",
                 "matched_pairs", "n_matched_pairs"]]
           .sort_values(["vertical", "rel_path"]).reset_index(drop=True))
    agg.to_csv(out_dir / "h1_metal_ligand_pairs.csv", index=False)

    print(f"  shape={tuple(shape)} seed={seed} -> {len(agg):,} structures "
          f"({100*len(agg)/n_tm_struct:.2f}% TM-bonded)")
    for band in band_labels:
        sub = defs_df[defs_df["band"] == band]
        if len(sub):
            print(f"    {band}: {sub['pair'].tolist()}")
    return agg


# --------------------------------------------------------------------------- H3
def build_h3(
    ok: pd.DataFrame,
    out_dir: Path,
    targets: dict = H3_TARGETS,
    seed: int = H3_SEED,
) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    for vert, target in targets.items():
        sub = ok[ok["vertical"] == vert].copy()
        if sub.empty:
            print(f"  WARNING: vertical {vert!r} not in manifest, skipping")
            continue
        formula_counts = sub.groupby("formula_hill").size()
        ranked = sorted(
            ((f, n, formula_rank(f, salt=seed)) for f, n in formula_counts.items()),
            key=lambda r: r[2],
        )
        accumulated = 0
        chosen_formulas: List[str] = []
        for f, n, _r in ranked:
            if accumulated >= target:
                break
            chosen_formulas.append(f)
            accumulated += n
        sel = sub[sub["formula_hill"].isin(chosen_formulas)].copy()
        sel = sel[["vertical", "rel_path", "formula_hill", "n_atoms",
                    "charge", "net_charge_abs"]].copy()
        sel.insert(0, "holdout_id", "H3")
        rows.append(sel)
        print(f"  {vert}: target={target:,} formulas={len(chosen_formulas):,} "
              f"realized={len(sel):,}")

    df = pd.concat(rows, ignore_index=True).sort_values(
        ["vertical", "rel_path"]
    ).reset_index(drop=True)
    df.to_csv(out_dir / "h3_reactivity.csv", index=False)
    print(f"  total H3: {len(df):,}")
    return df


# --------------------------------------------------------------------------- H7 / H8
def build_h7(ok: pd.DataFrame, out_dir: Path,
             threshold: int = H7_NATOMS_THRESHOLD) -> pd.DataFrame:
    mask = ok["n_atoms"] > threshold
    df = ok.loc[mask, ["vertical", "rel_path", "n_atoms"]].copy()
    df.insert(0, "holdout_id", "H7")
    df = df.sort_values(["vertical", "rel_path"]).reset_index(drop=True)
    df.to_csv(out_dir / "h7_large_systems.csv", index=False)
    print(f"  cutoff > {threshold}: {len(df):,}")
    return df


def build_h8(ok: pd.DataFrame, out_dir: Path,
             threshold: int = H8_CHARGE_THRESHOLD) -> pd.DataFrame:
    mask = ok["net_charge_abs"] > threshold
    df = ok.loc[mask, ["vertical", "rel_path", "charge", "net_charge_abs"]].copy()
    df.insert(0, "holdout_id", "H8")
    df = df.sort_values(["vertical", "rel_path"]).reset_index(drop=True)
    df.to_csv(out_dir / "h8_weird_charges.csv", index=False)
    print(f"  cutoff > {threshold}: {len(df):,}")
    return df


# --------------------------------------------------------------------------- main
def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--manifest_dir", type=Path, default=DEFAULT_MANIFEST_DIR,
                   help=f"Per-vertical manifest parquet root. Default: {DEFAULT_MANIFEST_DIR}")
    p.add_argument("--bond_root", type=Path, default=DEFAULT_BOND_ROOT,
                   help=f"tm_neighbors extraction root. Default: {DEFAULT_BOND_ROOT}")
    p.add_argument("--output_dir", type=Path, default=DEFAULT_OUT_DIR,
                   help=f"Where to write the holdout CSVs. Default: {DEFAULT_OUT_DIR}")
    p.add_argument("--skip", nargs="*", choices=["h1", "h3", "h7", "h8"],
                   default=[], help="Skip these holdouts (e.g. --skip h1)")
    args = p.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"manifest_dir: {args.manifest_dir}")
    print(f"bond_root:    {args.bond_root}")
    print(f"output_dir:   {args.output_dir}")
    print(f"skipping:     {args.skip or '(none)'}")

    ok = load_manifest(args.manifest_dir)
    n_dataset = ok.shape[0]  # equals total ok manifest rows
    print(f"\nmanifest ok rows: {n_dataset:,}")

    summary_rows = []

    if "h1" not in args.skip:
        print("\n=== H1: TM bond pairs (stratified) ===")
        bond_df = load_tm_neighbors(args.bond_root)
        agg_h1 = build_h1(bond_df, args.output_dir)
        summary_rows.append(("H1", "h1_metal_ligand_pairs.csv", len(agg_h1),
                             f"Metal-ligand pairs (stratified shape "
                             f"{list(H1_SHAPE)}, seed {H1_SEED}; see 8.3.1)"))

    if "h3" not in args.skip:
        print("\n=== H3: reactivity (composition-stratified per vertical) ===")
        h3_df = build_h3(ok, args.output_dir)
        summary_rows.append(("H3", "h3_reactivity.csv", len(h3_df),
                             "Reactivity, composition-stratified subsample "
                             "(tm_react=5k + electrolytes_reactivity=5k + pmechdb=2.5k)"))

    if "h7" not in args.skip:
        print(f"\n=== H7: n_atoms > {H7_NATOMS_THRESHOLD} ===")
        h7_df = build_h7(ok, args.output_dir)
        summary_rows.append(("H7", "h7_large_systems.csv", len(h7_df),
                             f"Large systems (n_atoms > {H7_NATOMS_THRESHOLD})"))

    if "h8" not in args.skip:
        print(f"\n=== H8: |net_charge_abs| > {H8_CHARGE_THRESHOLD} (strict) ===")
        h8_df = build_h8(ok, args.output_dir)
        summary_rows.append(("H8", "h8_weird_charges.csv", len(h8_df),
                             f"Weird charges (|net_charge_abs| > {H8_CHARGE_THRESHOLD})"))

    # H2 is structurally empty in the public OMol25 4M release
    h2_path = args.output_dir / "h2_pdb_tm.csv"
    if h2_path.exists():
        os.remove(h2_path)
        print(f"\n=== H2: removed empty {h2_path.name} ===")

    if summary_rows:
        idx_df = pd.DataFrame(
            summary_rows,
            columns=["holdout_id", "filename", "n_structures", "description"],
        )
        idx_df.to_csv(args.output_dir / "INDEX.csv", index=False)
        print(f"\n=== INDEX ===")
        print(idx_df.to_string(index=False))

    return 0


if __name__ == "__main__":
    sys.exit(main())
