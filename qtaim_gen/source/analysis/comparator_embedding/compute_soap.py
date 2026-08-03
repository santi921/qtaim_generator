"""Drive a loader through dscribe SOAP and write per-structure descriptors.

Locked hyperparameters (see docs/plans/2026-05-04-soap-featurization-plan.md):
- r_cut = 5.0 Å
- n_max = 8, l_max = 6, sigma = 0.5
- rbf = "gto", periodic = False, average = "inner", sparse = False
- l2-normalize each output vector before storing.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from ase import Atoms
from ase.data import chemical_symbols as ASE_SYMBOLS
from dscribe.descriptors import SOAP

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SoapConfig:
    species: list[int]
    r_cut: float = 5.0
    n_max: int = 8
    l_max: int = 6
    sigma: float = 0.5
    rbf: str = "gto"
    periodic: bool = False
    average: str = "inner"
    l2_normalize: bool = True

    @property
    def species_symbols(self) -> list[str]:
        return [ASE_SYMBOLS[z] for z in self.species]


@dataclass
class WriteResult:
    n_records: int = 0
    n_features: int = 0
    species: list[int] = field(default_factory=list)
    output_path: Optional[Path] = None


def build_soap(cfg: SoapConfig) -> SOAP:
    return SOAP(
        species=cfg.species_symbols,
        r_cut=cfg.r_cut,
        n_max=cfg.n_max,
        l_max=cfg.l_max,
        sigma=cfg.sigma,
        rbf=cfg.rbf,
        periodic=cfg.periodic,
        average=cfg.average,
        sparse=False,
    )


def _filter_atoms_to_species(atoms: Atoms, species_zs: set[int]) -> Atoms:
    """Drop atoms whose Z is outside ``species``. dscribe also drops them
    silently, but doing it explicitly avoids a confusing zero descriptor for
    structures with zero allowed atoms."""
    keep = [i for i, z in enumerate(atoms.get_atomic_numbers()) if int(z) in species_zs]
    if len(keep) == len(atoms):
        return atoms
    return atoms[keep]


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(vec))
    if not math.isfinite(n) or n == 0.0:
        return vec
    return vec / n


def featurize_to_parquet(
    iterator: Iterator[tuple[str, Atoms]],
    cfg: SoapConfig,
    source: str,
    output_path: Path,
    n_jobs: int = 1,
    batch_size: int = 256,
    progress: bool = True,
) -> WriteResult:
    """Stream (id, Atoms) -> SOAP -> parquet rows.

    Records that have no atoms inside ``cfg.species`` are skipped with a log
    message rather than written as zero vectors.
    """
    soap = build_soap(cfg)
    n_features = soap.get_number_of_features()
    species_zs = set(cfg.species)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    schema = pa.schema(
        [
            pa.field("structure_id", pa.string()),
            pa.field("source", pa.string()),
            pa.field("n_atoms", pa.int32()),
            pa.field("n_atoms_kept", pa.int32()),
            pa.field("soap", pa.list_(pa.float32(), list_size=n_features)),
        ],
        metadata={
            b"species_z": ",".join(str(z) for z in cfg.species).encode("ascii"),
            b"r_cut": f"{cfg.r_cut}".encode("ascii"),
            b"n_max": f"{cfg.n_max}".encode("ascii"),
            b"l_max": f"{cfg.l_max}".encode("ascii"),
            b"sigma": f"{cfg.sigma}".encode("ascii"),
            b"rbf": cfg.rbf.encode("ascii"),
            b"average": cfg.average.encode("ascii"),
            b"l2_normalized": str(cfg.l2_normalize).encode("ascii"),
            b"source": source.encode("ascii"),
        },
    )

    writer: Optional[pq.ParquetWriter] = None
    pending_ids: list[str] = []
    pending_atoms: list[Atoms] = []
    pending_n_atoms: list[int] = []
    pending_n_kept: list[int] = []
    n_skipped = 0
    n_records = 0

    iterable = iterator
    if progress:
        try:
            from tqdm import tqdm

            iterable = tqdm(iterator, desc=f"soap[{source}]")
        except ImportError:
            pass

    def _flush() -> None:
        nonlocal writer, n_records
        if not pending_atoms:
            return
        descs = soap.create(pending_atoms, n_jobs=n_jobs)
        # `average="inner"` returns one vector per structure (shape (N, D));
        # if for some reason a structure had no atoms in species, dscribe will
        # return zeros — we already filter those upstream.
        if cfg.l2_normalize:
            descs = np.vstack([_l2_normalize(d) for d in descs])
        descs = descs.astype(np.float32, copy=False)

        rows = pa.Table.from_pydict(
            {
                "structure_id": pending_ids,
                "source": [source] * len(pending_ids),
                "n_atoms": pa.array(pending_n_atoms, type=pa.int32()),
                "n_atoms_kept": pa.array(pending_n_kept, type=pa.int32()),
                "soap": pa.array(
                    [list(d) for d in descs],
                    type=pa.list_(pa.float32(), list_size=n_features),
                ),
            },
            schema=schema,
        )
        if writer is None:
            writer = pq.ParquetWriter(str(output_path), schema)
        writer.write_table(rows)
        n_records += len(pending_ids)
        pending_ids.clear()
        pending_atoms.clear()
        pending_n_atoms.clear()
        pending_n_kept.clear()

    try:
        for structure_id, atoms in iterable:
            n_total = len(atoms)
            atoms_filtered = _filter_atoms_to_species(atoms, species_zs)
            if len(atoms_filtered) == 0:
                n_skipped += 1
                continue
            pending_ids.append(structure_id)
            pending_atoms.append(atoms_filtered)
            pending_n_atoms.append(n_total)
            pending_n_kept.append(len(atoms_filtered))
            if len(pending_atoms) >= batch_size:
                _flush()
        _flush()
    finally:
        if writer is not None:
            writer.close()

    if n_skipped:
        logger.info(
            "soap[%s]: skipped %d records with zero atoms in species %s",
            source,
            n_skipped,
            cfg.species,
        )

    return WriteResult(
        n_records=n_records,
        n_features=n_features,
        species=list(cfg.species),
        output_path=output_path,
    )
