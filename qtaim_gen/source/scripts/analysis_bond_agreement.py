"""CLI for Stream D: Bond agreement analysis.

Single-vertical mode (root contains structure.lmdb):

  analysis-bond-agreement --root data/OMol4M_lmdbs/droplet --output out/droplet_ba.parquet

Corpus mode (root contains per-vertical subdirs):

  analysis-bond-agreement --root data/OMol4M_lmdbs --output out/ --no-progress

  Writes out/<vertical>_ba.parquet per vertical, plus:
    out/combined_ba.parquet           all per-pair rows concatenated
    out/combined_ba_agg.parquet       per-(vertical, scheme) F1
    out/combined_ba_agg_element_pair.parquet

Output layout per run:
  <output>                         per-pair rows
  <stem>_agg.parquet               per-scheme binary F1 vs geom_bonded
  <stem>_agg_element_pair.parquet  per-(scheme, element_pair) F1
  --emit-disagreements <path>      rows where any scheme disagrees with geom_bonded
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from qtaim_gen.source.analysis.bond_agreement import (
    aggregate,
    aggregate_by_element_pair,
    per_pair_classification,
)
from qtaim_gen.source.analysis.census import discover_verticals
from qtaim_gen.source.analysis.streaming_aggregator import stream_to_parquet

logger = logging.getLogger(__name__)

_SCHEME_COLS = ["qtaim_bonded", "mayer_bonded", "loewdin_bonded", "fuzzy_bonded"]


def _run_vertical(
    vertical: str,
    root: Path,
    output_path: Path,
    geom_k: float,
    bo_threshold: float,
    pool_multiplier: float,
    progress: bool,
    emit_disagreements: Path | None,
) -> pd.DataFrame:
    """Run one vertical; return the per-pair DataFrame."""
    def fn(key, structure, qtaim, bond):
        return per_pair_classification(
            key, structure, qtaim, bond,
            pool_multiplier=pool_multiplier,
            geom_k=geom_k,
            bo_threshold=bo_threshold,
            vertical=vertical,
        )

    stream_to_parquet(
        root=root,
        lmdb_types=["structure", "qtaim", "bond"],
        per_record_fn=fn,
        output_path=output_path,
        progress=progress,
    )

    df = pd.read_parquet(output_path)
    if df.empty:
        logger.warning("%s: no candidate pairs produced", vertical)
        return df

    stem = output_path.stem
    agg = aggregate(df, vertical)
    agg.to_parquet(output_path.with_name(f"{stem}_agg.parquet"), index=False)
    agg_ep = aggregate_by_element_pair(df, vertical)
    agg_ep.to_parquet(output_path.with_name(f"{stem}_agg_element_pair.parquet"), index=False)
    logger.info("%s: %d pairs, wrote agg parquets", vertical, len(df))

    if emit_disagreements is not None:
        present = [c for c in _SCHEME_COLS if c in df.columns]
        if present:
            mask = df[present].apply(
                lambda col: col.notna() & (col.astype(bool) != df["geom_bonded"])
            ).any(axis=1)
            dis = df[mask]
            dis.to_parquet(emit_disagreements, index=False)
            logger.info("%s: %d disagreement rows -> %s", vertical, len(dis), emit_disagreements)

    return df


def _write_combined(dfs: list[pd.DataFrame], out_dir: Path) -> None:
    """Concat all per-vertical DataFrames and write combined agg parquets."""
    combined = pd.concat(dfs, ignore_index=True)
    combined_path = out_dir / "combined_ba.parquet"
    combined.to_parquet(combined_path, index=False)
    logger.info("combined: %d total pairs -> %s", len(combined), combined_path)

    agg_rows = []
    for vertical, grp in combined.groupby("vertical"):
        agg_rows.append(aggregate(grp, vertical))
    agg_combined = aggregate(combined, "combined")
    agg_all = pd.concat(agg_rows + [agg_combined], ignore_index=True)
    agg_path = out_dir / "combined_ba_agg.parquet"
    agg_all.to_parquet(agg_path, index=False)

    ep_rows = []
    for vertical, grp in combined.groupby("vertical"):
        ep_rows.append(aggregate_by_element_pair(grp, vertical))
    ep_combined = aggregate_by_element_pair(combined, "combined")
    ep_all = pd.concat(ep_rows + [ep_combined], ignore_index=True)
    ep_path = out_dir / "combined_ba_agg_element_pair.parquet"
    ep_all.to_parquet(ep_path, index=False)

    logger.info("wrote %s", agg_path)
    logger.info("wrote %s", ep_path)

    print("\n--- combined F1 ---")
    print(agg_combined.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(
        description="Stream D: bond agreement analysis (single vertical or corpus).",
    )
    parser.add_argument("--root", type=Path, required=True,
                        help="Vertical root (has structure.lmdb) or corpus root "
                             "(subdirs are verticals). Auto-detected.")
    parser.add_argument("--output", type=Path, required=True,
                        help="Single-vertical: per-pair parquet path. "
                             "Corpus: output directory.")
    parser.add_argument("--geom-k", type=float, default=1.3)
    parser.add_argument("--bo-threshold", type=float, default=0.5)
    parser.add_argument("--pool-multiplier", type=float, default=1.4)
    parser.add_argument("--emit-disagreements", type=Path, default=None,
                        help="Write disagreement rows here. In corpus mode, "
                             "one file per vertical: <output>/<vertical>_dis.parquet.")
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
        dfs: list[pd.DataFrame] = []
        for name, lmdb_root in verticals:
            out_path = out_dir / f"{name}_ba.parquet"
            dis_path = (
                out_dir / f"{name}_dis.parquet"
                if args.emit_disagreements is not None
                else None
            )
            df = _run_vertical(
                vertical=name,
                root=lmdb_root,
                output_path=out_path,
                geom_k=args.geom_k,
                bo_threshold=args.bo_threshold,
                pool_multiplier=args.pool_multiplier,
                progress=not args.no_progress,
                emit_disagreements=dis_path,
            )
            if not df.empty:
                dfs.append(df)
        if dfs:
            _write_combined(dfs, out_dir)
    else:
        name, lmdb_root = verticals[0]
        _run_vertical(
            vertical=name,
            root=lmdb_root,
            output_path=args.output,
            geom_k=args.geom_k,
            bo_threshold=args.bo_threshold,
            pool_multiplier=args.pool_multiplier,
            progress=not args.no_progress,
            emit_disagreements=args.emit_disagreements,
        )
        df = pd.read_parquet(args.output)
        if not df.empty:
            agg = aggregate(df, name)
            print(agg.to_string(index=False))


if __name__ == "__main__":
    main()
