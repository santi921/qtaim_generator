"""CLI for Stream D: Bond agreement analysis.

Runs per-pair bond classification for one vertical and writes three parquet
outputs:

  <output>                     per-pair rows
  <stem>_agg.parquet           per-scheme binary F1 vs geom_bonded
  <stem>_agg_element_pair.parquet  per-(scheme, element_pair) F1

Examples:

    analysis-bond-agreement --root data/OMol4M_lmdbs/droplet \\
        --output /tmp/droplet_ba.parquet

    analysis-bond-agreement --root data/OMol4M_lmdbs/mo_hydrides \\
        --output /tmp/mo_hydrides_ba.parquet \\
        --emit-disagreements /tmp/mo_hydrides_dis.parquet
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
from qtaim_gen.source.analysis.streaming_aggregator import stream_to_parquet


def main():
    parser = argparse.ArgumentParser(
        description="Stream D: per-vertical bond agreement analysis.",
    )
    parser.add_argument("--root", type=Path, required=True,
                        help="Vertical root containing structure.lmdb, qtaim.lmdb, bond.lmdb.")
    parser.add_argument("--output", type=Path, required=True,
                        help="Per-pair output parquet path.")
    parser.add_argument("--geom-k", type=float, default=1.3,
                        help="Geometric bond threshold multiplier (default 1.3).")
    parser.add_argument("--bo-threshold", type=float, default=0.5,
                        help="Bond-order threshold for mayer/loewdin/fuzzy (default 0.5).")
    parser.add_argument("--pool-multiplier", type=float, default=1.4,
                        help="Candidate pair pool multiplier (default 1.4).")
    parser.add_argument("--emit-disagreements", type=Path, default=None,
                        help="If set, write disagreement rows to this parquet path.")
    parser.add_argument("--no-progress", action="store_true",
                        help="Suppress tqdm progress bar.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s",
                        stream=sys.stderr)

    vertical = args.root.name
    geom_k = args.geom_k
    bo_threshold = args.bo_threshold
    pool_multiplier = args.pool_multiplier

    def fn(key, structure, qtaim, bond):
        return per_pair_classification(
            key, structure, qtaim, bond,
            pool_multiplier=pool_multiplier,
            geom_k=geom_k,
            bo_threshold=bo_threshold,
            vertical=vertical,
        )

    stream_to_parquet(
        root=args.root,
        lmdb_types=["structure", "qtaim", "bond"],
        per_record_fn=fn,
        output_path=args.output,
        progress=not args.no_progress,
    )

    df = pd.read_parquet(args.output)
    if df.empty:
        logging.warning("no candidate pairs produced")
        return

    stem = args.output.stem
    agg = aggregate(df, vertical)
    agg_path = args.output.with_name(f"{stem}_agg.parquet")
    agg.to_parquet(agg_path, index=False)
    logging.info("wrote %s", agg_path)

    agg_ep = aggregate_by_element_pair(df, vertical)
    agg_ep_path = args.output.with_name(f"{stem}_agg_element_pair.parquet")
    agg_ep.to_parquet(agg_ep_path, index=False)
    logging.info("wrote %s", agg_ep_path)

    if args.emit_disagreements is not None:
        scheme_cols = [c for c in ["qtaim_bonded", "mayer_bonded", "loewdin_bonded", "fuzzy_bonded"]
                       if c in df.columns]
        if scheme_cols:
            mask = df[scheme_cols].apply(
                lambda col: col.notna() & (col.astype(bool) != df["geom_bonded"])
            ).any(axis=1)
            dis = df[mask]
            dis.to_parquet(args.emit_disagreements, index=False)
            logging.info("wrote %d disagreement rows to %s", len(dis), args.emit_disagreements)

    print(agg.to_string(index=False))


if __name__ == "__main__":
    main()
