"""Load SOAP parquets, fit UMAP, plot, save embedding + reducer.

Workflow:
1. discover_parquets(input_dir, mode) -> list of (path, source, vertical)
2. load_and_stack(parquets) -> (X float32, ids, labels)
3. balanced_downsample(X, labels, n_per_class, seed) -> subset
4. fit_umap(X, n_components, n_neighbors, min_dist, metric, seed) -> reducer + Y
5. write parquet (id/source/vertical/umap_x/umap_y[/umap_z])
6. write png (matplotlib, alpha-blended scatter)
7. dump reducer.joblib for later .transform() of new sources

The reducer is fit on the downsampled stack but can transform any new SOAP
vector with the matching species set; this is the iterative-UMAP path
documented in docs/plans/2026-05-04-soap-featurization-plan.md.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

OMOL_FILE_RE = re.compile(r"^soap_(?P<vertical>.+)\.parquet$")


@dataclass(frozen=True)
class ParquetSpec:
    path: Path
    source: str
    vertical: str  # equals source for comparators; vertical name for omol


def discover_parquets(input_dir: Path, mode: str) -> list[ParquetSpec]:
    """Auto-discover SOAP parquets under ``input_dir``.

    Layout expected (matches scripts/run_soap_featurization.sh output):
        <input_dir>/{schnet4aim,qm7x,pcqm4mv2,qmugs}/soap.parquet
        <input_dir>/omol/soap_<vertical>.parquet
    """
    found: list[ParquetSpec] = []
    omol_dir = input_dir / "omol"
    if omol_dir.is_dir():
        for p in sorted(omol_dir.glob("soap_*.parquet")):
            m = OMOL_FILE_RE.match(p.name)
            if not m:
                continue
            found.append(ParquetSpec(path=p, source="omol", vertical=m.group("vertical")))
    for src in ("schnet4aim", "qm7x", "pcqm4mv2", "qmugs"):
        p = input_dir / src / "soap.parquet"
        if p.is_file():
            found.append(ParquetSpec(path=p, source=src, vertical=src))

    if mode == "all":
        return found
    if mode == "omol-by-vertical":
        return [s for s in found if s.source == "omol"]
    if mode == "comparators-only":
        return [s for s in found if s.source != "omol"]
    raise ValueError(f"unknown mode {mode!r}")


def _read_metadata_species(p: Path) -> tuple[str, int]:
    schema = pq.read_schema(str(p))
    kv = schema.metadata or {}
    species = kv.get(b"species_z", b"").decode()
    soap_field = schema.field("soap")
    dim = int(soap_field.type.list_size)
    return species, dim


def _sample_indices(n_total: int, n_keep: int, rng: np.random.Generator) -> np.ndarray:
    if n_keep >= n_total:
        return np.arange(n_total, dtype=np.int64)
    return np.sort(rng.choice(n_total, size=n_keep, replace=False)).astype(np.int64)


def _read_parquet_sampled(
    path: Path,
    keep_indices: np.ndarray,
    dim: int,
    batch_size: int = 2048,
) -> tuple[np.ndarray, list[str]]:
    """Stream a parquet in row-batches and pull rows whose original index is
    in ``keep_indices``. Peak memory ~ ``batch_size * dim * 4`` bytes.
    """
    out_X = np.empty((len(keep_indices), dim), dtype=np.float32)
    out_ids: list[str] = [""] * len(keep_indices)
    keep_arr = np.asarray(keep_indices, dtype=np.int64)
    keep_pos = 0
    row_offset = 0

    pf = pq.ParquetFile(str(path))
    for batch in pf.iter_batches(batch_size=batch_size, columns=["structure_id", "soap"]):
        n = batch.num_rows
        if keep_pos >= len(keep_arr):
            break
        # local indices into this batch
        upper = row_offset + n
        local_mask_start = keep_pos
        while keep_pos < len(keep_arr) and keep_arr[keep_pos] < upper:
            keep_pos += 1
        if keep_pos > local_mask_start:
            wanted = keep_arr[local_mask_start:keep_pos] - row_offset
            soap_arr = np.asarray(batch.column("soap").values).reshape(n, dim)
            ids_arr = batch.column("structure_id").to_pylist()
            for offset, li in enumerate(wanted):
                out_X[local_mask_start + offset] = soap_arr[li]
                out_ids[local_mask_start + offset] = ids_arr[li]
        row_offset = upper
    return out_X, out_ids


def load_and_stack(
    specs: list[ParquetSpec],
    n_per_class: Optional[int] = None,
    label_by: str = "source",
    seed: int = 42,
    batch_size: int = 2048,
) -> tuple[np.ndarray, list[str], list[str], list[str]]:
    """Load all parquets, with optional per-class sampling at read time.

    When ``n_per_class`` is given, the per-parquet quota is
    ``ceil(n_per_class / parquets_in_same_class)``. Sampling happens *before*
    SOAP arrays land in memory (row-batch streaming via pyarrow), so peak
    memory is independent of the largest parquet's size.

    Returns (X, ids, sources, verticals). All parquets must share the same
    species_z and dim; we abort otherwise.
    """
    if not specs:
        raise RuntimeError("no parquets to load")

    rng = np.random.default_rng(seed)

    # validate schema first, decide per-parquet quotas
    ref_species: Optional[str] = None
    ref_dim: Optional[int] = None
    parquets_in_class: dict[str, int] = {}
    for s in specs:
        species, dim = _read_metadata_species(s.path)
        if ref_species is None:
            ref_species, ref_dim = species, dim
        elif species != ref_species or dim != ref_dim:
            raise ValueError(
                f"parquet schema mismatch: {s.path} has species={species!r} dim={dim} "
                f"but reference is species={ref_species!r} dim={ref_dim}"
            )
        key = s.vertical if label_by == "vertical" else s.source
        parquets_in_class[key] = parquets_in_class.get(key, 0) + 1

    Xs: list[np.ndarray] = []
    ids: list[str] = []
    sources: list[str] = []
    verticals: list[str] = []

    for s in specs:
        n_total = pq.read_metadata(str(s.path)).num_rows
        if n_total == 0:
            logger.warning("empty parquet, skipping: %s", s.path)
            continue

        if n_per_class is None:
            quota = n_total
        else:
            key = s.vertical if label_by == "vertical" else s.source
            n_par = parquets_in_class[key]
            # ceil-divide so the class total is at least n_per_class when each
            # parquet has enough rows; oversample is trimmed at the end.
            quota = (n_per_class + n_par - 1) // n_par

        keep_idx = _sample_indices(n_total, quota, rng)
        if len(keep_idx) == 0:
            continue

        soap_np, batch_ids = _read_parquet_sampled(
            s.path, keep_idx, ref_dim, batch_size=batch_size
        )
        Xs.append(soap_np)
        ids.extend(batch_ids)
        sources.extend([s.source] * len(batch_ids))
        verticals.extend([s.vertical] * len(batch_ids))
        logger.info(
            "loaded %d/%d records from %s (%s/%s)",
            len(batch_ids),
            n_total,
            s.path.name,
            s.source,
            s.vertical,
        )

    if not Xs:
        raise RuntimeError("no SOAP records loaded")

    X = np.vstack(Xs).astype(np.float32, copy=False)
    return X, ids, sources, verticals


def balanced_downsample(
    n_total: int,
    labels: list[str],
    n_per_class: int,
    seed: int = 42,
) -> np.ndarray:
    """Return indices that keep at most ``n_per_class`` points per label.

    Uniform random within each label. Deterministic via ``seed``.
    """
    rng = np.random.default_rng(seed)
    by_label: dict[str, list[int]] = {}
    for i, lab in enumerate(labels):
        by_label.setdefault(lab, []).append(i)
    keep: list[int] = []
    for lab, idx_list in by_label.items():
        idx = np.asarray(idx_list)
        if len(idx) > n_per_class:
            idx = rng.choice(idx, size=n_per_class, replace=False)
        keep.extend(idx.tolist())
        logger.info("downsample %s: %d -> %d", lab, len(idx_list), len(idx))
    keep_arr = np.asarray(sorted(keep), dtype=np.int64)
    return keep_arr


def fit_umap(
    X: np.ndarray,
    n_components: int = 2,
    n_neighbors: int = 30,
    min_dist: float = 0.1,
    metric: str = "cosine",
    seed: int = 42,
):
    import umap

    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=seed,
        verbose=True,
    )
    Y = reducer.fit_transform(X)
    return reducer, np.asarray(Y, dtype=np.float32)


def write_embedding_parquet(
    output_path: Path,
    Y: np.ndarray,
    ids: list[str],
    sources: list[str],
    verticals: list[str],
    species_z: str,
    n_components: int,
) -> None:
    import pyarrow as pa

    cols: dict[str, list] = {
        "structure_id": ids,
        "source": sources,
        "vertical": verticals,
        "umap_x": Y[:, 0].tolist(),
        "umap_y": Y[:, 1].tolist(),
    }
    if n_components >= 3:
        cols["umap_z"] = Y[:, 2].tolist()
    schema_metadata = {
        b"species_z": species_z.encode("ascii"),
        b"n_components": str(n_components).encode("ascii"),
    }
    tbl = pa.Table.from_pydict(cols).replace_schema_metadata(schema_metadata)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(tbl, str(output_path))
    logger.info("wrote %s (%d rows)", output_path, len(ids))


def plot_2d(
    Y: np.ndarray,
    labels: list[str],
    output_path: Path,
    title: str = "SOAP UMAP",
    point_size: float = 4.0,
    alpha: float = 0.4,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    unique_labels = sorted(set(labels))
    n_lab = len(unique_labels)
    # Wider canvas + side legend when many classes (e.g. 33 OMol verticals).
    if n_lab > 20:
        fig, ax = plt.subplots(figsize=(13, 8), dpi=150)
    else:
        fig, ax = plt.subplots(figsize=(10, 8), dpi=150)

    if n_lab <= 10:
        colors = [plt.get_cmap("tab10")(i) for i in range(n_lab)]
    elif n_lab <= 20:
        colors = [plt.get_cmap("tab20")(i) for i in range(n_lab)]
    else:
        # Concat tab20 + tab20b + tab20c for up to 60 distinct hues.
        seq = []
        for cm_name in ("tab20", "tab20b", "tab20c"):
            cm = plt.get_cmap(cm_name)
            seq.extend([cm(i) for i in range(cm.N)])
        colors = [seq[i % len(seq)] for i in range(n_lab)]

    for i, lab in enumerate(unique_labels):
        mask = np.asarray([l_ == lab for l_ in labels])
        ax.scatter(
            Y[mask, 0],
            Y[mask, 1],
            s=point_size,
            alpha=alpha,
            color=colors[i],
            label=f"{lab} (n={int(mask.sum())})",
            linewidths=0,
        )
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.set_title(title)
    if n_lab > 20:
        ax.legend(markerscale=3, loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=7, framealpha=0.7, ncol=2)
    else:
        ax.legend(markerscale=3, loc="best", fontsize=8, framealpha=0.7)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    logger.info("wrote %s", output_path)


def plot_3d_html(
    Y: np.ndarray,
    labels: list[str],
    ids: list[str],
    output_path: Path,
    title: str = "SOAP UMAP (3D)",
) -> None:
    import plotly.express as px

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig = px.scatter_3d(
        x=Y[:, 0],
        y=Y[:, 1],
        z=Y[:, 2],
        color=labels,
        hover_name=ids,
        title=title,
        opacity=0.6,
    )
    fig.update_traces(marker=dict(size=2))
    fig.write_html(str(output_path))
    logger.info("wrote %s", output_path)


def save_reducer(reducer, path: Path) -> None:
    import joblib

    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(reducer, str(path))
    logger.info("wrote %s", path)
