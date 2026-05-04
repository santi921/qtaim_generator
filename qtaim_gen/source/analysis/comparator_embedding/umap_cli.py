"""CLI entry: ``analysis-soap-umap``.

Fit UMAP on SOAP parquets and save embedding + plot. Modes auto-discover
parquets under ``--input-dir`` (matches the layout produced by
scripts/run_soap_featurization.sh).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from qtaim_gen.source.analysis.comparator_embedding.umap_plot import (
    balanced_downsample,
    discover_parquets,
    fit_umap,
    load_and_stack,
    plot_2d,
    plot_3d_html,
    save_reducer,
    write_embedding_parquet,
)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Root containing comparator soap.parquets and omol/soap_<vertical>.parquet.",
    )
    p.add_argument(
        "--mode",
        choices=["all", "omol-by-vertical", "comparators-only"],
        default="all",
        help="all = OMol + 4 comparators colored by source; omol-by-vertical = OMol parquets only colored by vertical.",
    )
    p.add_argument(
        "--output-prefix",
        type=Path,
        required=True,
        help="Path prefix; .embedding.parquet, .png, .umap.joblib appended.",
    )
    p.add_argument("--n-per-class", type=int, default=5000)
    p.add_argument("--n-components", type=int, default=2, choices=[2, 3])
    p.add_argument("--n-neighbors", type=int, default=30)
    p.add_argument("--min-dist", type=float, default=0.1)
    p.add_argument("--metric", default="cosine")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--label-by",
        choices=["source", "vertical"],
        default=None,
        help="Color/label key for downsample + plot. Defaults: 'source' for mode=all, 'vertical' for mode=omol-by-vertical.",
    )
    p.add_argument(
        "--plot-3d-html",
        action="store_true",
        help="With --n-components 3, also write an interactive plotly HTML.",
    )
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)

    logging.basicConfig(level=args.log_level, format="[%(levelname)s] %(message)s")

    label_by = args.label_by or ("vertical" if args.mode == "omol-by-vertical" else "source")

    specs = discover_parquets(args.input_dir, args.mode)
    if not specs:
        print(f"no parquets found under {args.input_dir} (mode={args.mode})")
        return 2
    print(f"discovered {len(specs)} parquets:")
    for s in specs:
        print(f"  {s.path}  (source={s.source}, vertical={s.vertical})")

    # Sample at load time so peak memory is bounded by batch_size * dim, not
    # by the largest parquet's full SOAP column.
    X_ds, ids_ds, sources_ds, verticals_ds = load_and_stack(
        specs,
        n_per_class=args.n_per_class,
        label_by=label_by,
        seed=args.seed,
    )
    labels_ds = verticals_ds if label_by == "vertical" else sources_ds

    # Final balanced trim in case quota * n_parquets > n_per_class.
    keep = balanced_downsample(len(ids_ds), labels_ds, args.n_per_class, args.seed)
    X_ds = X_ds[keep]
    ids_ds = [ids_ds[i] for i in keep]
    sources_ds = [sources_ds[i] for i in keep]
    verticals_ds = [verticals_ds[i] for i in keep]
    labels_ds = [labels_ds[i] for i in keep]
    print(
        f"loaded + downsampled to {len(ids_ds):,} records "
        f"({args.n_per_class}/class, {len(set(labels_ds))} classes, dim={X_ds.shape[1]})"
    )

    reducer, Y = fit_umap(
        X_ds,
        n_components=args.n_components,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        metric=args.metric,
        seed=args.seed,
    )

    # Recover species_z from any one parquet metadata (already validated identical).
    import pyarrow.parquet as pq

    species_z = (pq.read_schema(str(specs[0].path)).metadata or {}).get(
        b"species_z", b""
    ).decode()

    out = args.output_prefix
    write_embedding_parquet(
        Path(f"{out}.embedding.parquet"),
        Y,
        ids_ds,
        sources_ds,
        verticals_ds,
        species_z,
        args.n_components,
    )
    plot_2d(
        Y if args.n_components == 2 else Y[:, :2],
        labels_ds,
        Path(f"{out}.png"),
        title=f"SOAP UMAP ({args.mode}, label_by={label_by})",
    )
    if args.n_components == 3 and args.plot_3d_html:
        plot_3d_html(
            Y,
            labels_ds,
            ids_ds,
            Path(f"{out}.3d.html"),
            title=f"SOAP UMAP 3D ({args.mode}, label_by={label_by})",
        )
    save_reducer(reducer, Path(f"{out}.umap.joblib"))

    print(f"done. outputs: {out}.embedding.parquet, {out}.png, {out}.umap.joblib")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
