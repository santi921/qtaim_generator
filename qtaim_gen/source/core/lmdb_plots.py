"""
Plotting functions for LMDB descriptor analysis.

Separated from lmdb_analysis.py so that module can be imported
on headless HPC systems without requiring matplotlib.

All plot functions accept output_path and output_format parameters.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Lazy import matplotlib/seaborn to avoid import cost when not plotting
_MPL_IMPORTED = False


def _ensure_mpl():
    """Lazy-import matplotlib and seaborn, configure style."""
    global _MPL_IMPORTED
    if _MPL_IMPORTED:
        return
    import matplotlib

    matplotlib.use("Agg")  # Non-interactive backend for headless
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid", palette="colorblind", font_scale=1.2)
    plt.rcParams.update(
        {
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "figure.figsize": (8, 5),
            "axes.titlesize": 14,
            "axes.labelsize": 12,
        }
    )
    _MPL_IMPORTED = True


def _savefig(fig, output_path: str | Path, output_format: str = "png"):
    """Save figure and close it."""
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        str(output_path),
        format=output_format,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)
    logger.info(f"Saved plot: {output_path}")


def plot_histogram(
    values: np.ndarray,
    name: str,
    output_path: str | Path,
    bins: int = 100,
    log_scale: bool = False,
    output_format: str = "png",
    dpi: int = 300,
    clip_percentile: tuple[float, float] | None = (1, 99),
    signed_log_x: bool = False,
) -> None:
    """Histogram with NaN/inf annotation.

    clip_percentile: clips to this percentile range before plotting.
    signed_log_x: applies sign(x)*log10(|x|+1) transform to x values — useful
        for QTAIM properties that span many orders of magnitude with mixed sign.
    log_scale: log y-axis (count axis).
    """
    _ensure_mpl()
    import matplotlib.pyplot as plt

    values = np.asarray(values, dtype=np.float64)
    nan_count = int(np.isnan(values).sum())
    inf_count = int(np.isinf(values).sum())
    clean = values[np.isfinite(values)]

    if signed_log_x and len(clean) > 0:
        clean = np.sign(clean) * np.log10(np.abs(clean) + 1)
        name = f"{name} [signed log₁₀(|x|+1)]"

    if clip_percentile is not None and len(clean) > 1:
        lo, hi = np.percentile(clean, [clip_percentile[0], clip_percentile[1]])
        clean = clean[(clean >= lo) & (clean <= hi)]

    fig, ax = plt.subplots(figsize=(8, 5), dpi=dpi)

    if len(clean) > 0:
        ax.hist(clean, bins=bins, edgecolor="white", linewidth=0.5, alpha=0.85)
    else:
        ax.text(
            0.5,
            0.5,
            "No finite values",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=14,
        )

    if log_scale and len(clean) > 0:
        ax.set_yscale("log")

    ax.set_xlabel(name)
    ax.set_ylabel("Count")
    ax.set_title(name)

    # Annotate NaN/inf counts
    annotation_parts = []
    if nan_count > 0:
        annotation_parts.append(f"NaN: {nan_count}")
    if inf_count > 0:
        annotation_parts.append(f"Inf: {inf_count}")
    annotation_parts.append(f"N={len(values):,}")
    annotation = " | ".join(annotation_parts)
    ax.annotate(
        annotation,
        xy=(0.98, 0.95),
        xycoords="axes fraction",
        ha="right",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.7),
    )

    _savefig(fig, output_path, output_format)


def plot_nan_summary(
    nan_fractions: dict[str, float],
    output_path: str | Path,
    output_format: str = "png",
) -> None:
    """Bar chart of NaN fraction per descriptor key, sorted descending."""
    _ensure_mpl()
    import matplotlib.pyplot as plt

    if not nan_fractions:
        logger.info("No NaN data to plot.")
        return

    # Sort by fraction descending, take top 30
    sorted_items = sorted(nan_fractions.items(), key=lambda x: -x[1])[:30]
    names = [item[0] for item in sorted_items]
    fractions = [item[1] for item in sorted_items]

    fig, ax = plt.subplots(figsize=(10, max(4, len(names) * 0.3)))
    bars = ax.barh(range(len(names)), fractions, color="salmon", edgecolor="white")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("NaN Fraction")
    ax.set_title("NaN Fraction by Descriptor")
    ax.invert_yaxis()
    ax.set_xlim(0, min(1.0, max(fractions) * 1.2) if fractions else 1.0)

    # Add value labels on bars
    for bar, frac in zip(bars, fractions):
        if frac > 0:
            ax.text(
                bar.get_width() + 0.005,
                bar.get_y() + bar.get_height() / 2,
                f"{frac:.3f}",
                va="center",
                fontsize=8,
            )

    _savefig(fig, output_path, output_format)


def plot_correlation_heatmap(
    values_by_key: dict[str, np.ndarray],
    output_path: str | Path,
    title: str = "Correlation",
    output_format: str = "png",
) -> None:
    """Pairwise correlation heatmap.

    Used for cross-charge-method comparison, cross-QTAIM-property comparison, etc.
    Computes correlations on finite values only.
    """
    _ensure_mpl()
    import matplotlib.pyplot as plt
    import seaborn as sns

    if len(values_by_key) < 2:
        logger.info("Need at least 2 keys for correlation heatmap.")
        return

    keys = list(values_by_key.keys())
    n = len(keys)

    # Build correlation matrix
    corr_matrix = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(n):
            vi = values_by_key[keys[i]]
            vj = values_by_key[keys[j]]
            # Use only mutually finite values
            mask = np.isfinite(vi) & np.isfinite(vj)
            if mask.sum() > 1:
                corr_matrix[i, j] = np.corrcoef(vi[mask], vj[mask])[0, 1]

    fig, ax = plt.subplots(figsize=(max(6, n * 0.8), max(5, n * 0.7)))
    sns.heatmap(
        corr_matrix,
        xticklabels=keys,
        yticklabels=keys,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        ax=ax,
    )
    ax.set_title(title)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(fontsize=9)

    _savefig(fig, output_path, output_format)


def plot_per_element_boxplots(
    values_by_element: dict[str, np.ndarray],
    descriptor_name: str,
    output_path: str | Path,
    top_n: int = 10,
    output_format: str = "png",
) -> None:
    """Box/violin plots for a descriptor broken down by element. Top N most frequent."""
    _ensure_mpl()
    import matplotlib.pyplot as plt

    if not values_by_element:
        logger.info("No per-element data to plot.")
        return

    # Sort by count, take top N
    sorted_elements = sorted(
        values_by_element.items(), key=lambda x: -len(x[1])
    )[:top_n]

    labels = []
    data = []
    for element, vals in sorted_elements:
        clean = vals[np.isfinite(vals)]
        if len(clean) > 0:
            labels.append(f"{element} (n={len(clean):,})")
            data.append(clean)

    if not data:
        logger.info("No finite per-element data to plot.")
        return

    fig, ax = plt.subplots(figsize=(max(6, len(data) * 0.8), 5))
    parts = ax.violinplot(data, showmedians=True, showextrema=True)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel(descriptor_name)
    ax.set_title(f"{descriptor_name} by Element")

    _savefig(fig, output_path, output_format)


def plot_value_range_summary(
    stats_by_descriptor: dict[str, dict],
    output_path: str | Path,
    output_format: str = "png",
) -> None:
    """Bar chart showing p5/p95/mean per descriptor with error bars.

    Uses 5th/95th percentile bounds instead of raw min/max to avoid
    extreme outliers (e.g., det_hessian at 1e44) collapsing the axis.
    Applies symlog x-scale to handle mixed-sign values across many orders
    of magnitude.
    """
    _ensure_mpl()
    import matplotlib.pyplot as plt

    if not stats_by_descriptor:
        return

    names = list(stats_by_descriptor.keys())
    means = []
    stds = []
    p5s = []
    p95s = []

    for name in names:
        s = stats_by_descriptor[name]
        means.append(s.get("mean") or 0)
        stds.append(s.get("std") or 0)
        pcts = s.get("percentiles") or {}
        p5s.append(pcts.get(5) or s.get("min") or 0)
        p95s.append(pcts.get(95) or s.get("max") or 0)

    y_pos = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(10, max(4, len(names) * 0.35)))

    ax.barh(y_pos, means, xerr=stds, color="steelblue", alpha=0.8, label="Mean +/- Std")
    ax.scatter(p5s, y_pos, color="red", marker="|", s=100, zorder=5, label="p5")
    ax.scatter(p95s, y_pos, color="green", marker="|", s=100, zorder=5, label="p95")

    # symlog handles mixed-sign values spanning orders of magnitude
    all_vals = [v for v in means + p5s + p95s if v is not None and np.isfinite(v) and v != 0]
    if all_vals:
        linthresh = max(1e-10, np.percentile(np.abs(all_vals), 10))
        ax.set_xscale("symlog", linthresh=linthresh)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Value (symlog scale)")
    ax.set_title("Descriptor Value Ranges (p5 – p95, symlog)")
    ax.legend(loc="lower right", fontsize=9)
    ax.invert_yaxis()

    _savefig(fig, output_path, output_format)


def plot_overlay_distributions(
    values_by_key: dict[str, np.ndarray],
    output_path: str | Path,
    title: str = "",
    output_format: str = "png",
    clip_percentile: tuple[float, float] | None = (1, 99),
) -> None:
    """Overlaid KDE/histograms for comparing methods.

    Useful for comparing charge types, bond types, etc.
    clip_percentile: if set, clips each series to this percentile range before KDE.
    """
    _ensure_mpl()
    import matplotlib.pyplot as plt
    import seaborn as sns

    if not values_by_key:
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    for label, values in values_by_key.items():
        clean = values[np.isfinite(values)]
        if clip_percentile is not None and len(clean) > 1:
            lo, hi = np.percentile(clean, [clip_percentile[0], clip_percentile[1]])
            clean = clean[(clean >= lo) & (clean <= hi)]
        if len(clean) > 1:
            sns.kdeplot(clean, ax=ax, label=label, fill=True, alpha=0.3)

    ax.set_title(title or "Distribution Comparison")
    ax.set_ylabel("Density")
    ax.legend(fontsize=9)

    _savefig(fig, output_path, output_format)


def plot_cp_comparison(
    atom_stats: dict[str, dict],
    bond_stats: dict[str, dict],
    output_path: str | Path,
    output_format: str = "png",
) -> None:
    """Side-by-side comparison of atom CPs vs bond CPs for QTAIM properties.

    Shows mean values with error bars for shared properties.
    """
    _ensure_mpl()
    import matplotlib.pyplot as plt

    # Find shared property names
    shared = sorted(set(atom_stats.keys()) & set(bond_stats.keys()))
    if not shared:
        logger.info("No shared properties between atom and bond CPs.")
        return

    atom_means = [atom_stats[k].get("mean") or 0 for k in shared]
    bond_means = [bond_stats[k].get("mean") or 0 for k in shared]
    atom_stds = [atom_stats[k].get("std") or 0 for k in shared]
    bond_stds = [bond_stats[k].get("std") or 0 for k in shared]

    y_pos = np.arange(len(shared))
    height = 0.35

    fig, ax = plt.subplots(figsize=(10, max(5, len(shared) * 0.4)))
    ax.barh(y_pos - height / 2, atom_means, height, xerr=atom_stds,
            label="Atom CP", color="steelblue", alpha=0.8)
    ax.barh(y_pos + height / 2, bond_means, height, xerr=bond_stds,
            label="Bond CP", color="coral", alpha=0.8)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(shared, fontsize=8)
    ax.set_xlabel("Value")
    ax.set_title("Atom CP vs Bond CP Properties")
    ax.legend(fontsize=10)
    ax.invert_yaxis()

    _savefig(fig, output_path, output_format)


def plot_outlier_scatter(
    values: np.ndarray,
    name: str,
    output_path: str | Path,
    threshold_std: float = 5.0,
    output_format: str = "png",
) -> None:
    """Scatter plot highlighting values beyond threshold_std standard deviations."""
    _ensure_mpl()
    import matplotlib.pyplot as plt

    values = np.asarray(values, dtype=np.float64)
    clean = values[np.isfinite(values)]

    if len(clean) < 2:
        logger.info(f"Not enough finite values for outlier scatter: {name}")
        return

    mean = np.mean(clean)
    std = np.std(clean)

    if std == 0:
        logger.info(f"Zero std for {name}, skipping outlier scatter.")
        return

    z_scores = np.abs((clean - mean) / std)
    is_outlier = z_scores > threshold_std

    fig, ax = plt.subplots(figsize=(10, 5))
    indices = np.arange(len(clean))

    # Plot normal points
    normal_mask = ~is_outlier
    ax.scatter(
        indices[normal_mask],
        clean[normal_mask],
        s=2,
        alpha=0.3,
        color="steelblue",
        label="Normal",
    )

    # Plot outliers
    if is_outlier.any():
        ax.scatter(
            indices[is_outlier],
            clean[is_outlier],
            s=20,
            color="red",
            marker="x",
            label=f"Outlier (>{threshold_std}σ)",
            zorder=5,
        )

    # Reference lines
    ax.axhline(mean, color="gray", linestyle="--", linewidth=0.8, label="Mean")
    ax.axhline(
        mean + threshold_std * std,
        color="red",
        linestyle=":",
        linewidth=0.8,
        alpha=0.5,
    )
    ax.axhline(
        mean - threshold_std * std,
        color="red",
        linestyle=":",
        linewidth=0.8,
        alpha=0.5,
    )

    outlier_count = int(is_outlier.sum())
    ax.set_xlabel("Entry Index")
    ax.set_ylabel(name)
    ax.set_title(f"{name} — {outlier_count} outliers (>{threshold_std}σ)")
    ax.legend(fontsize=9)

    _savefig(fig, output_path, output_format)
