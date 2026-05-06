"""CLI for Stream F: Cross-method noise floors.

Single-vertical mode (root contains charge.lmdb):

  analysis-noise-floors --root data/OMol4M_lmdbs/mo_hydrides --output /tmp/nf.parquet

Corpus mode (root contains per-vertical subdirs):

  analysis-noise-floors --root data/OMol4M_lmdbs --output /tmp/nf_out/ --no-progress

Output layout per vertical:
  <output>                         noise-floor table (B4), one row per (descriptor, element-or-pair)
  <stem>_exemplars.parquet         high-disagreement exemplars (B5)
  <stem>_charge_atoms.parquet      intermediate per-atom charge rows (B1 raw)
  <stem>_bond_pairs.parquet        intermediate per-pair bond-order rows (B2 raw)
  <stem>_qtaim_bcps.parquet        intermediate per-BCP descriptor rows (B3 raw)

In corpus mode --output is treated as a directory; per-vertical files are
written as <output>/<vertical>_nf.parquet etc. plus combined_nf.parquet.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from qtaim_gen.source.analysis.census import discover_verticals
from qtaim_gen.source.analysis.noise_floors import (
    aggregate_bond_noise_floor,
    aggregate_charge_noise_floor,
    aggregate_qtaim_redundancy,
    build_noise_floor_table,
    emit_exemplars,
    per_record_bond_pairs,
    per_record_charge_atoms,
    per_record_qtaim_bcps,
)
from qtaim_gen.source.analysis.streaming_aggregator import stream_to_parquet

logger = logging.getLogger(__name__)


def _run_vertical(
    vertical: str,
    root: Path,
    nf_path: Path,
    topk: int,
    progress: bool,
) -> pd.DataFrame:
    """Run B1-B5 for one vertical. Returns the noise-floor DataFrame.

    nf_path is the exact output path for the noise-floor parquet.
    Intermediate and exemplar files share the same directory and stem.
    """
    stem = nf_path.with_suffix("").name
    out_dir = nf_path.parent
    charge_atoms_path = out_dir / f"{stem}_charge_atoms.parquet"
    bond_pairs_path = out_dir / f"{stem}_bond_pairs.parquet"
    qtaim_bcps_path = out_dir / f"{stem}_qtaim_bcps.parquet"
    exemplars_path = out_dir / f"{stem}_exemplars.parquet"

    # B1: stream charge.lmdb -> per-atom charge rows
    def _charge_fn(key, charge_rec):
        return per_record_charge_atoms(key, charge_rec, vertical=vertical)

    stream_to_parquet(
        root=root,
        lmdb_types=["charge"],
        per_record_fn=_charge_fn,
        output_path=charge_atoms_path,
        progress=progress,
    )

    # B2: stream bond.lmdb -> per-pair bond-order rows
    def _bond_fn(key, bond_rec):
        return per_record_bond_pairs(key, bond_rec, vertical=vertical)

    stream_to_parquet(
        root=root,
        lmdb_types=["bond"],
        per_record_fn=_bond_fn,
        output_path=bond_pairs_path,
        progress=progress,
    )

    # B3: stream qtaim.lmdb + structure.lmdb -> per-BCP descriptor rows
    def _qtaim_fn(key, qtaim_rec, structure_rec):
        return per_record_qtaim_bcps(key, qtaim_rec, structure_rec, vertical=vertical)

    stream_to_parquet(
        root=root,
        lmdb_types=["qtaim", "structure"],
        per_record_fn=_qtaim_fn,
        output_path=qtaim_bcps_path,
        progress=progress,
    )

    charge_df = pd.read_parquet(charge_atoms_path)
    bond_df = pd.read_parquet(bond_pairs_path)
    qtaim_df = pd.read_parquet(qtaim_bcps_path)

    logger.info(
        "%s: %d charge-atom rows, %d bond-pair rows, %d bcp rows",
        vertical, len(charge_df), len(bond_df), len(qtaim_df),
    )

    charge_nf = aggregate_charge_noise_floor(charge_df, vertical, topk=topk)
    bond_nf = aggregate_bond_noise_floor(bond_df, vertical, topk=topk)
    qtaim_redund = aggregate_qtaim_redundancy(qtaim_df, vertical)
    nf = build_noise_floor_table(charge_nf, bond_nf, qtaim_redund)
    nf.to_parquet(nf_path, index=False)
    logger.info("%s: noise-floor table -> %s (%d rows)", vertical, nf_path, len(nf))

    exemplars = emit_exemplars(charge_df, bond_df, topk=topk, vertical=vertical)
    exemplars.to_parquet(exemplars_path, index=False)
    logger.info("%s: exemplars -> %s (%d rows)", vertical, exemplars_path, len(exemplars))

    return nf


def _write_combined(nfs: list[pd.DataFrame], out_dir: Path) -> None:
    combined = pd.concat(nfs, ignore_index=True)
    path = out_dir / "combined_nf.parquet"
    combined.to_parquet(path, index=False)
    logger.info("combined noise-floor table -> %s (%d rows)", path, len(combined))

    print("\n--- combined noise-floor summary (global rows only) ---")
    summary = combined[combined["element"].isna() & combined["element_pair"].isna()]
    if not summary.empty:
        print(summary[["vertical", "analysis", "descriptor", "mar", "n_obs"]].to_string(index=False))


def main():
    parser = argparse.ArgumentParser(
        description="Stream F: cross-method noise floors (B1-B5).",
    )
    parser.add_argument(
        "--root", type=Path, required=True,
        help="Vertical root (has charge.lmdb) or corpus root (subdirs are verticals).",
    )
    parser.add_argument(
        "--output", type=Path, required=True,
        help="Single-vertical: noise-floor parquet path. Corpus: output directory.",
    )
    parser.add_argument(
        "--topk", type=int, default=50,
        help="Max exemplars emitted per (descriptor, element_or_pair). Default: 50.",
    )
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s",
                        stream=sys.stderr)

    verticals = discover_verticals(args.root)
    if not verticals:
        logger.error("no verticals found under %s", args.root)
        sys.exit(1)

    corpus_mode = len(verticals) > 1 or (
        len(verticals) == 1 and verticals[0][1] != args.root
    )

    if corpus_mode:
        out_dir = args.output
        out_dir.mkdir(parents=True, exist_ok=True)
        nfs: list[pd.DataFrame] = []
        for name, lmdb_root in verticals:
            nf = _run_vertical(
                vertical=name,
                root=lmdb_root,
                nf_path=out_dir / f"{name}_nf.parquet",
                topk=args.topk,
                progress=not args.no_progress,
            )
            if not nf.empty:
                nfs.append(nf)
        if nfs:
            _write_combined(nfs, out_dir)
    else:
        name, lmdb_root = verticals[0]
        # In single-vertical mode --output IS the nf parquet path.
        args.output.parent.mkdir(parents=True, exist_ok=True)
        nf = _run_vertical(
            vertical=name,
            root=lmdb_root,
            nf_path=args.output,
            topk=args.topk,
            progress=not args.no_progress,
        )
        if not nf.empty:
            print(nf[nf["element"].isna() & nf["element_pair"].isna()][
                ["analysis", "descriptor", "mar", "n_obs"]
            ].to_string(index=False))


if __name__ == "__main__":
    main()
