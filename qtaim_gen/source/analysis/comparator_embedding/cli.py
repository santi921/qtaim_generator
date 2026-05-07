"""CLI entry: ``analysis-soap-featurize``.

Compute SOAP descriptors for one source (omol vertical or comparator) and
write to parquet. See docs/plans/2026-05-04-soap-featurization-plan.md.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from qtaim_gen.source.analysis.comparator_embedding.compute_soap import (
    SoapConfig,
    featurize_to_parquet,
)
from qtaim_gen.source.analysis.comparator_embedding.loaders import LOADERS


# Global species lock (2026-05-04): one shared SOAP basis across OMol + all
# four comparators so they can land in a single UMAP plot. Set is the union
# of comparator-side >= 0.01% elements (12 Z values). Atoms outside this set
# are silently dropped by dscribe; OMol structures with heavier elements
# (e.g. transition metals) lose those atoms in their fingerprint.
GLOBAL_SPECIES: list[int] = [1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53]

DEFAULT_SPECIES_BY_SOURCE: dict[str, list[int]] = {
    "omol": GLOBAL_SPECIES,
    "pcqm4mv2": GLOBAL_SPECIES,
    "qmugs": GLOBAL_SPECIES,
    "qm7x": GLOBAL_SPECIES,
    "schnet4aim": GLOBAL_SPECIES,
}


def _parse_species(arg: str) -> list[int]:
    return sorted({int(x) for x in arg.split(",") if x.strip()})


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--source",
        required=True,
        choices=sorted(LOADERS),
        help="Which loader to invoke.",
    )
    p.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Source root directory. Defaults to the loader's default_root.",
    )
    p.add_argument(
        "--species",
        type=_parse_species,
        default=None,
        help=(
            "Comma-separated atomic numbers. Defaults to the locked species "
            "list for the source (omol has no default - pass explicitly to "
            "match the comparator you are pairing with)."
        ),
    )
    p.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output parquet path.",
    )
    p.add_argument("--n-sample", type=int, default=None, help="Cap on records.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--r-cut", type=float, default=5.0)
    p.add_argument("--n-max", type=int, default=8)
    p.add_argument("--l-max", type=int, default=6)
    p.add_argument("--sigma", type=float, default=0.5)
    p.add_argument("--n-jobs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument(
        "--no-l2-normalize",
        action="store_true",
        help="Skip per-vector l2 normalization (default: normalize).",
    )
    p.add_argument("--no-progress", action="store_true")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)

    logging.basicConfig(level=args.log_level)

    spec = LOADERS[args.source]
    root = args.root or spec.default_root
    species = args.species or DEFAULT_SPECIES_BY_SOURCE[args.source]

    cfg = SoapConfig(
        species=species,
        r_cut=args.r_cut,
        n_max=args.n_max,
        l_max=args.l_max,
        sigma=args.sigma,
        l2_normalize=not args.no_l2_normalize,
    )

    iterator = spec.iterate(root, max_n=args.n_sample, seed=args.seed)
    result = featurize_to_parquet(
        iterator=iterator,
        cfg=cfg,
        source=args.source,
        output_path=args.output,
        n_jobs=args.n_jobs,
        batch_size=args.batch_size,
        progress=not args.no_progress,
    )

    print(
        f"[soap] source={args.source} root={root} -> {result.output_path}",
        f"records={result.n_records:,} features={result.n_features:,}",
        f"species={result.species}",
        sep="\n  ",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
