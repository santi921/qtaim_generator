"""CLI for Stream D: Bond agreement analysis.

Single-vertical mode (root contains structure.lmdb):

  analysis-bond-agreement --root data/OMol4M_lmdbs/droplet --output out/droplet_ba.parquet

Corpus mode (root contains per-vertical subdirs):

  analysis-bond-agreement --root data/OMol4M_lmdbs --output out/ --no-progress

  Writes out/<vertical>_ba.parquet per vertical plus combined aggregation parquets.
  Memory-safe: combined aggregation sums TP/FP/FN/TN from the small per-vertical
  _agg.parquet files rather than concat-ing all raw pair DataFrames.

Output layout per vertical:
  <stem>.parquet                   per-pair rows (streamed, constant memory)
  <stem>_agg.parquet               per-scheme binary F1 vs geom_bonded
  <stem>_agg_element_pair.parquet  per-(scheme, element_pair) F1

Corpus-mode additional outputs in <output>/:
  combined_ba_agg.parquet              all verticals + combined row
  combined_ba_agg_element_pair.parquet all verticals + combined row
"""
from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path

import pandas as pd

from qtaim_gen.source.analysis.bond_agreement import (
    SCHEMES,
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
) -> None:
    """Stream one vertical to parquet and write aggregation files."""
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

    # Load one vertical at a time for aggregation, then release.
    df = pd.read_parquet(output_path)
    if df.empty:
        logger.warning("%s: no candidate pairs produced", vertical)
        return

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


def _f1_from_counts(tp: int, fp: int, fn: int) -> float:
    prec = tp / (tp + fp) if (tp + fp) > 0 else math.nan
    rec = tp / (tp + fn) if (tp + fn) > 0 else math.nan
    if math.isnan(prec) or math.isnan(rec) or (prec + rec) == 0:
        return math.nan
    return 2 * prec * rec / (prec + rec)


def _write_combined(vertical_outputs: list[tuple[str, Path]], out_dir: Path) -> None:
    """Write combined aggregation parquets from the small per-vertical agg files.

    Reads only <stem>_agg.parquet and <stem>_agg_element_pair.parquet (tiny),
    never the raw per-pair parquets. Combined F1 is derived by summing
    TP/FP/FN/TN counts, which are additive across verticals.
    """
    agg_parts: list[pd.DataFrame] = []
    ep_parts: list[pd.DataFrame] = []

    for name, ba_path in vertical_outputs:
        stem = ba_path.stem
        p = ba_path.with_name(f"{stem}_agg.parquet")
        if p.exists():
            agg_parts.append(pd.read_parquet(p))
        p = ba_path.with_name(f"{stem}_agg_element_pair.parquet")
        if p.exists():
            ep_parts.append(pd.read_parquet(p))

    if not agg_parts:
        logger.warning("no per-vertical agg files found; skipping combined output")
        return

    agg_df = pd.concat(agg_parts, ignore_index=True)

    # Combined F1: sum counts across all verticals per scheme.
    combined_rows = []
    for scheme in SCHEMES:
        sub = agg_df[agg_df["scheme"] == scheme]
        if sub.empty:
            continue
        tp = int(sub["tp"].sum())
        fp = int(sub["fp"].sum())
        fn = int(sub["fn"].sum())
        tn = int(sub["tn"].sum())
        n = int(sub["n_pairs"].sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else math.nan
        rec = tp / (tp + fn) if (tp + fn) > 0 else math.nan
        combined_rows.append({
            "vertical": "combined", "scheme": scheme, "n_pairs": n,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": prec, "recall": rec, "f1": _f1_from_counts(tp, fp, fn),
        })

    agg_combined = pd.DataFrame(combined_rows)
    agg_all = pd.concat([agg_df, agg_combined], ignore_index=True)
    agg_path = out_dir / "combined_ba_agg.parquet"
    agg_all.to_parquet(agg_path, index=False)
    logger.info("wrote %s", agg_path)

    if ep_parts:
        ep_df = pd.concat(ep_parts, ignore_index=True)
        ep_combined_rows = []
        for (scheme, ep), grp in ep_df.groupby(["scheme", "element_pair"]):
            tp = int(grp["tp"].sum())
            fp = int(grp["fp"].sum())
            fn = int(grp["fn"].sum())
            tn = int(grp["tn"].sum())
            n = int(grp["n_pairs"].sum())
            prec = tp / (tp + fp) if (tp + fp) > 0 else math.nan
            rec = tp / (tp + fn) if (tp + fn) > 0 else math.nan
            ep_combined_rows.append({
                "vertical": "combined", "scheme": scheme, "element_pair": ep,
                "n_pairs": n, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                "precision": prec, "recall": rec, "f1": _f1_from_counts(tp, fp, fn),
            })
        ep_combined = pd.DataFrame(ep_combined_rows)
        ep_all = pd.concat([ep_df, ep_combined], ignore_index=True)
        ep_path = out_dir / "combined_ba_agg_element_pair.parquet"
        ep_all.to_parquet(ep_path, index=False)
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
                        help="Write disagreement rows here. Corpus mode: "
                             "<output>/<vertical>_dis.parquet per vertical.")
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
        vertical_outputs: list[tuple[str, Path]] = []
        for name, lmdb_root in verticals:
            out_path = out_dir / f"{name}_ba.parquet"
            dis_path = (
                out_dir / f"{name}_dis.parquet"
                if args.emit_disagreements is not None
                else None
            )
            _run_vertical(
                vertical=name,
                root=lmdb_root,
                output_path=out_path,
                geom_k=args.geom_k,
                bo_threshold=args.bo_threshold,
                pool_multiplier=args.pool_multiplier,
                progress=not args.no_progress,
                emit_disagreements=dis_path,
            )
            vertical_outputs.append((name, out_path))
        _write_combined(vertical_outputs, out_dir)
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
