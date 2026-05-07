"""CLI entry: ``analysis-soap-umap``.

Fit UMAP on SOAP parquets and save embedding + plot. Modes auto-discover
parquets under ``--input-dir`` (matches the layout produced by
scripts/run_soap_featurization.sh).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from qtaim_gen.source.analysis.comparator_embedding.umap_plot import (
    balanced_downsample,
    discover_parquets,
    fit_umap,
    load_and_stack,
    plot_2d,
    plot_3d_html,
    save_reducer,
    transform_rest,
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
    p.add_argument(
        "--no-save-reducer",
        action="store_true",
        help=(
            "Skip writing the fitted reducer to <prefix>.umap.joblib. The "
            "embedding parquet is enough for plotting; the joblib only matters "
            "if you want reducer.transform() of new vectors later. Reducers "
            "for big fits can be 6+ GB so you usually don't want them."
        ),
    )
    p.add_argument(
        "--transform-rest",
        action="store_true",
        help=(
            "After fitting on the n_per_class subset, stream-load every "
            "remaining row from each parquet and project it via "
            "reducer.transform(). Output embedding contains all rows with "
            "an in_fit_set bool column. Memory bounded by batch_size."
        ),
    )
    p.add_argument(
        "--transform-batch-size",
        type=int,
        default=2048,
        help="Batch size for the transform-rest streaming pass.",
    )
    p.add_argument(
        "--transform-fraction",
        type=float,
        default=1.0,
        help=(
            "Fraction of the non-fit rows to transform (0 < x <= 1). "
            "Lower means a sparser overlay but proportionally faster wall time. "
            "Default 1.0 = transform everything."
        ),
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
    X_ds, ids_ds, sources_ds, verticals_ds, picked_idx = load_and_stack(
        specs,
        n_per_class=args.n_per_class,
        label_by=label_by,
        seed=args.seed,
        return_indices=True,
    )
    labels_ds = verticals_ds if label_by == "vertical" else sources_ds

    # Final balanced trim in case quota * n_parquets > n_per_class. Note: the
    # trim happens *after* picked_idx is recorded, so transform-rest uses the
    # untrimmed picked set as the "in fit" definition. That's correct: any
    # row we read but didn't fit on still gets projected via .transform().
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

    reducer, Y_fit = fit_umap(
        X_ds,
        n_components=args.n_components,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        metric=args.metric,
        seed=args.seed,
    )

    # Recover species_z + dim from any one parquet metadata (already validated identical).
    import pyarrow.parquet as pq

    schema0 = pq.read_schema(str(specs[0].path))
    species_z = (schema0.metadata or {}).get(b"species_z", b"").decode()
    dim = int(schema0.field("soap").type.list_size)

    if args.transform_rest:
        print(
            f"transform-rest: streaming complement of fit set "
            f"(fraction={args.transform_fraction})..."
        )
        Y_rest, ids_rest, sources_rest, verticals_rest = transform_rest(
            reducer,
            specs,
            picked_idx,
            dim=dim,
            batch_size=args.transform_batch_size,
            fraction=args.transform_fraction,
            seed=args.seed,
        )
        print(f"  transformed {len(ids_rest):,} additional records")
        Y_all = np.vstack([Y_fit, Y_rest]) if len(ids_rest) else Y_fit
        ids_all = ids_ds + ids_rest
        sources_all = sources_ds + sources_rest
        verticals_all = verticals_ds + verticals_rest
        labels_all = (
            verticals_all if label_by == "vertical" else sources_all
        )
        in_fit = [True] * len(ids_ds) + [False] * len(ids_rest)
    else:
        Y_all, ids_all, sources_all, verticals_all = (
            Y_fit,
            ids_ds,
            sources_ds,
            verticals_ds,
        )
        labels_all = labels_ds
        in_fit = [True] * len(ids_ds)

    out = args.output_prefix
    write_embedding_parquet(
        Path(f"{out}.embedding.parquet"),
        np.asarray(Y_all),
        ids_all,
        sources_all,
        verticals_all,
        species_z,
        args.n_components,
        in_fit=in_fit,
    )
    plot_2d(
        Y_all if args.n_components == 2 else Y_all[:, :2],
        labels_all,
        Path(f"{out}.png"),
        title=f"SOAP UMAP ({args.mode}, label_by={label_by}, n={len(ids_all):,})",
    )
    if args.n_components == 3 and args.plot_3d_html:
        plot_3d_html(
            Y_all,
            labels_all,
            ids_all,
            Path(f"{out}.3d.html"),
            title=f"SOAP UMAP 3D ({args.mode}, label_by={label_by})",
        )
    if not args.no_save_reducer:
        save_reducer(reducer, Path(f"{out}.umap.joblib"))

    written = [f"{out}.embedding.parquet ({len(ids_all):,} rows)", f"{out}.png"]
    if not args.no_save_reducer:
        written.append(f"{out}.umap.joblib")
    print("done. outputs: " + ", ".join(written))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
