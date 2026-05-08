"""Random-projection -> UMAP pipeline that projects ALL rows under
data/comparators/umap_tmqm/ onto 2D.

Two-stage dim reduction:
  1) GaussianRandomProjection: 230,272 -> 256 dim. Random Gaussian matrix
     multiply. O(N * d_in * d_out), no SVD. Streamable, deterministic via SEED.
     Preserves pairwise distances within (1+/-eps) per Johnson-Lindenstrauss
     for cosine geometry.
  2) UMAP fit on a balanced anchor sample of the random-projected vectors,
     then UMAP transform of the remaining (sampled at SAMPLE_FRACTION) rows.

This avoids the IncrementalPCA SVD bottleneck and the PyNNDescent stall on
230k-dim cosine that the stock analysis-soap-umap --transform-rest hits.

Outputs a parquet drop-in compatible with notebooks/render_umap_omol_vs_tmqmplus.py:
    data/comparators/umap_tmqm/umap_omol_vs_tmqmplus.embedding.parquet
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.random_projection import GaussianRandomProjection

# Force unbuffered logging so we see progress when piped through tee.
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s",
                    stream=sys.stdout, force=True)
log = logging.getLogger("rp_umap")

INPUT_DIR = Path("data/comparators/umap_tmqm")
OUT_PARQUET = Path("data/comparators/umap_tmqm/umap_omol_vs_tmqmplus.embedding.parquet")

N_PER_CLASS = 5000           # anchors used to fit UMAP on RP-reduced features
SAMPLE_FRACTION = 0.5        # fraction of every parquet to keep on the plot
RP_COMPONENTS = 256
UMAP_N_NEIGHBORS = 30
UMAP_MIN_DIST = 0.1
SEED = 42
BATCH = 1024


def discover_parquets():
    found = []
    omol_dir = INPUT_DIR / "omol"
    for p in sorted(omol_dir.glob("soap_*.parquet")):
        v = p.stem.replace("soap_", "")
        found.append((p, "omol", v))
    tmqm = INPUT_DIR / "tmqmplus" / "soap.parquet"
    if tmqm.is_file():
        found.append((tmqm, "tmqmplus", "tmqmplus"))
    return found


def stream_soap(path: Path):
    pf = pq.ParquetFile(str(path))
    dim = int(pf.schema_arrow.field("soap").type.list_size)
    for batch in pf.iter_batches(batch_size=BATCH, columns=["structure_id", "soap"]):
        n = batch.num_rows
        ids = batch.column("structure_id").to_pylist()
        X = np.asarray(batch.column("soap").values).reshape(n, dim).astype(np.float32, copy=False)
        yield ids, X


def main():
    rng = np.random.default_rng(SEED)
    specs = discover_parquets()
    log.info(f"discovered {len(specs)} parquets")

    # 1) Pick which rows go to the UMAP fit set vs. the projected overlay.
    n_per_source = {"omol": 0, "tmqmplus": 0}
    for _, src, _ in specs:
        n_per_source[src] += 1
    quota_per_parquet = {
        src: max(1, (N_PER_CLASS + n - 1) // n) for src, n in n_per_source.items()
    }
    log.info(f"quota_per_parquet={quota_per_parquet}")

    # Per-parquet: anchor row indices + overlay (rest) row indices, sampled at
    # SAMPLE_FRACTION of the non-anchor rows.
    plan = []  # (path, src, vert, anchor_idx_set, rest_idx_list)
    for path, src, vert in specs:
        n_total = pq.read_metadata(str(path)).num_rows
        q = min(quota_per_parquet[src], n_total)
        anchor_idx = set(int(i) for i in rng.choice(n_total, size=q, replace=False))
        rest_pool = [i for i in range(n_total) if i not in anchor_idx]
        if SAMPLE_FRACTION < 1.0 and rest_pool:
            n_keep = max(1, int(round(len(rest_pool) * SAMPLE_FRACTION)))
            picked_rest = rng.choice(len(rest_pool), size=n_keep, replace=False)
            rest_idx = sorted(rest_pool[i] for i in picked_rest)
        else:
            rest_idx = rest_pool
        plan.append((path, src, vert, anchor_idx, set(rest_idx)))
    log.info(f"plan: {sum(len(p[3]) for p in plan)} anchors, "
             f"{sum(len(p[4]) for p in plan)} overlay rows")

    # 2) Build random projection matrix once. dscribe SOAP dim is fixed across
    #    parquets at 230,272.
    sch0 = pq.read_schema(str(specs[0][0]))
    in_dim = int(sch0.field("soap").type.list_size)
    log.info(f"GaussianRandomProjection {in_dim} -> {RP_COMPONENTS} (seed={SEED})")
    rp = GaussianRandomProjection(n_components=RP_COMPONENTS, random_state=SEED)
    # sklearn requires fit() with at least one vector to set the matrix shape.
    rp.fit(np.zeros((1, in_dim), dtype=np.float32))

    # 3) Single streaming pass: project EVERY row that we plan to keep
    #    (anchor or overlay). Keep results split by class so UMAP fit uses
    #    only the anchor projection.
    Z_anchor: list[np.ndarray] = []
    Z_rest: list[np.ndarray] = []
    ids_anchor: list[str] = []
    ids_rest: list[str] = []
    src_anchor: list[str] = []
    src_rest: list[str] = []
    vert_anchor: list[str] = []
    vert_rest: list[str] = []

    t0 = time.time()
    for path, src, vert, anchor_idx, rest_idx in plan:
        offset = 0
        n_a = 0
        n_r = 0
        for ids, X in stream_soap(path):
            n = len(ids)
            mask_a = []
            mask_r = []
            for i in range(n):
                gi = offset + i
                if gi in anchor_idx:
                    mask_a.append(i)
                elif gi in rest_idx:
                    mask_r.append(i)
            if mask_a:
                Xa = X[mask_a]
                Za = rp.transform(Xa).astype(np.float32, copy=False)
                Z_anchor.append(Za)
                ids_anchor.extend([ids[i] for i in mask_a])
                src_anchor.extend([src] * len(mask_a))
                vert_anchor.extend([vert] * len(mask_a))
                n_a += len(mask_a)
            if mask_r:
                Xr = X[mask_r]
                Zr = rp.transform(Xr).astype(np.float32, copy=False)
                Z_rest.append(Zr)
                ids_rest.extend([ids[i] for i in mask_r])
                src_rest.extend([src] * len(mask_r))
                vert_rest.extend([vert] * len(mask_r))
                n_r += len(mask_r)
            offset += n
        log.info(
            f"  {path.name}: anchor={n_a} overlay={n_r} elapsed={time.time()-t0:.1f}s"
        )

    Za = np.vstack(Z_anchor) if Z_anchor else np.empty((0, RP_COMPONENTS), dtype=np.float32)
    Zr = np.vstack(Z_rest) if Z_rest else np.empty((0, RP_COMPONENTS), dtype=np.float32)
    log.info(f"random-projection complete: anchors={Za.shape}, overlay={Zr.shape}")

    # 4) UMAP fit on RP-reduced anchors (cosine on 256 dim is fast).
    import umap
    log.info("UMAP fit on RP-reduced anchors...")
    t0 = time.time()
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        metric="cosine",
        random_state=SEED,
        verbose=True,
    )
    Ya = reducer.fit_transform(Za).astype(np.float32, copy=False)
    log.info(f"UMAP fit done in {time.time()-t0:.1f}s")

    # 5) UMAP transform on RP-reduced overlay rows.
    log.info(f"UMAP transform on {len(Zr):,} overlay rows...")
    t0 = time.time()
    Yr = reducer.transform(Zr).astype(np.float32, copy=False) if len(Zr) else np.empty((0, 2), dtype=np.float32)
    log.info(f"UMAP transform done in {time.time()-t0:.1f}s")

    # Re-bind for the writer block.
    rest_ids = ids_rest
    rest_sources = src_rest
    rest_verticals = vert_rest
    Yrest = Yr
    anchor_ids = ids_anchor
    anchor_sources = src_anchor
    anchor_verticals = vert_anchor

    # 6) Concatenate anchor + rest, dump parquet matching analysis-soap-umap schema.
    Y_all = np.vstack([Ya, Yrest]) if len(Yrest) else Ya
    ids_all = anchor_ids + rest_ids
    src_all = anchor_sources + rest_sources
    vert_all = anchor_verticals + rest_verticals
    in_fit = [True] * len(anchor_ids) + [False] * len(rest_ids)

    schema_meta = {
        b"species_z": pq.read_schema(str(specs[0][0])).metadata.get(b"species_z", b""),
        b"n_components": b"2",
        b"rp_components": str(RP_COMPONENTS).encode(),
        b"n_per_class_anchor": str(N_PER_CLASS).encode(),
        b"sample_fraction": str(SAMPLE_FRACTION).encode(),
    }
    table = pa.Table.from_pydict({
        "structure_id": ids_all,
        "source": src_all,
        "vertical": vert_all,
        "umap_x": Y_all[:, 0].tolist(),
        "umap_y": Y_all[:, 1].tolist(),
        "in_fit_set": in_fit,
    }).replace_schema_metadata(schema_meta)
    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, str(OUT_PARQUET))
    log.info(f"wrote {OUT_PARQUET} (rows={len(ids_all)})")


if __name__ == "__main__":
    main()
