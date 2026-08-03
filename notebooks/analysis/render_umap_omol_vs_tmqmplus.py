"""Render the OMol-vs-tmQM+ UMAP panel for the NeurIPS paper.

Reads ``data/comparators/umap_tmqm/umap_omol_vs_tmqmplus.embedding.parquet``
produced by ``analysis-soap-umap`` and emits paper-quality PNG + PDF in
``docs/neurips/figures/umap/``.

Run after ``analysis-soap-umap`` has produced the embedding parquet:
    conda run -n generator python notebooks/render_umap_omol_vs_tmqmplus.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import pyarrow.parquet as pq

EMBED = Path("data/comparators/umap_tmqm/umap_omol_vs_tmqmplus.embedding.parquet")
OUT_DIR = Path("docs/neurips/figures/umap")

SOURCE_LABEL_MAP = {
    "omol": "OMol-Descriptors-4M",
    "tmqmplus": "tmQM+",
}
SOURCE_COLOR_MAP = {
    "OMol-Descriptors-4M": "#9e9e9e",
    "tmQM+": "#ff7f0e",
}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tbl = pq.read_table(str(EMBED))
    df = tbl.to_pandas()
    print(f"loaded {len(df):,} rows from {EMBED}")
    print(df.groupby("source").size().to_string())

    df = df.assign(
        source_pretty=df["source"].map(SOURCE_LABEL_MAP).fillna(df["source"])
    )

    with mpl.rc_context({
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "legend.fontsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.18,
        "grid.linestyle": "--",
        "savefig.bbox": "tight",
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }):
        fig, ax = plt.subplots(figsize=(8.5, 6.5))
        labels = sorted(df["source_pretty"].unique())
        backdrop = "OMol-Descriptors-4M"
        draw_order = (
            ([backdrop] if backdrop in labels else [])
            + [l for l in labels if l != backdrop]
        )
        for lab in draw_order:
            sub = df[df["source_pretty"] == lab]
            is_backdrop = lab == backdrop
            ax.scatter(
                sub["umap_x"],
                sub["umap_y"],
                s=6 if is_backdrop else 3,
                alpha=0.45 if is_backdrop else 0.30,
                color=SOURCE_COLOR_MAP.get(lab, None),
                linewidths=0,
                label=f"{lab} (n={len(sub):,})",
                rasterized=True,
            )
        ax.set_xlabel("UMAP-1")
        ax.set_ylabel("UMAP-2")
        ax.set_title("SOAP UMAP: OMol-Descriptors-4M vs tmQM+ (TM-inclusive basis)")
        leg = ax.legend(loc="best", framealpha=0.9, markerscale=2.5)
        for handle in leg.legend_handles:
            handle.set_alpha(1.0)
        fig.tight_layout()
        png = OUT_DIR / "umap_omol_vs_tmqmplus_paper.png"
        pdf = OUT_DIR / "umap_omol_vs_tmqmplus_paper.pdf"
        fig.savefig(png, dpi=300)
        fig.savefig(pdf)
        print(f"saved {png}")
        print(f"saved {pdf}")


if __name__ == "__main__":
    main()
