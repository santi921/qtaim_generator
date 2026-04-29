"""Per-vertical manifest builder for the dataset root.

Walks each vertical, parses every `orca.inp`, and streams rows to a Parquet
file via ParquetWriter. Failures never abort the run; they are recorded with
`read_status` populated and dumped to a per-vertical CSV.
"""

from __future__ import annotations

import csv
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Tuple

import pyarrow as pa
import pyarrow.parquet as pq

try:
    from tqdm.auto import tqdm
except ImportError:  # tqdm is optional
    def tqdm(it, **_kwargs):
        return it

from qtaim_gen.source.core.parse_qtaim import dft_inp_to_dict
from qtaim_gen.source.utils.element_classes import (
    ACTINIDES,
    LANTHANIDES,
    SYMBOL_TO_Z,
    TRANSITION_METALS,
)


def _parse_inp(orca_path: str):
    """Lightweight wrapper around dft_inp_to_dict. Returns (species, charge, mult).

    Avoids `qtaim_gen.source.utils.io` so the manifest builder does not need
    RDKit importable. Some HPC nodes' system libstdc++ shadows conda-forge's,
    causing rdkit's CXXABI symbols to fail to resolve at import time.
    """
    mol_dict = dft_inp_to_dict(orca_path, parse_charge_spin=True)
    species = [atom["element"] for atom in mol_dict["mol"].values()]
    for atom in mol_dict["mol"].values():
        if len(atom["pos"]) != 3:
            raise ValueError(f"atom row missing coordinates: {atom}")
    return species, int(mol_dict["charge"]), int(mol_dict["spin"])


INP_NAME = "orca.inp"

SCHEMA = pa.schema(
    [
        ("vertical", pa.string()),
        ("rel_path", pa.string()),
        ("job_id", pa.string()),
        ("folder_depth", pa.int32()),
        ("charge", pa.int32()),
        ("mult", pa.int32()),
        ("spin", pa.float32()),
        ("n_atoms", pa.int32()),
        ("element_set", pa.list_(pa.string())),
        ("n_unique_elements", pa.int32()),
        ("formula_hill", pa.string()),
        ("heaviest_z", pa.int32()),
        ("has_tm", pa.bool_()),
        ("has_lanthanide", pa.bool_()),
        ("has_actinide", pa.bool_()),
        ("net_charge_abs", pa.int32()),
        ("read_status", pa.string()),
    ]
)


@dataclass
class JobTarget:
    root: str
    vertical: str
    job_dir: str  # absolute path
    rel_path: str  # relative to root


def hill_formula(species: List[str]) -> str:
    counts = Counter(species)
    parts: List[str] = []
    if "C" in counts:
        c = counts.pop("C")
        parts.append(f"C{c}" if c > 1 else "C")
        if "H" in counts:
            h = counts.pop("H")
            parts.append(f"H{h}" if h > 1 else "H")
    for el in sorted(counts):
        n = counts[el]
        parts.append(f"{el}{n}" if n > 1 else el)
    return "".join(parts)


def _empty_row(target: JobTarget, status: str) -> Dict:
    return {
        "vertical": target.vertical,
        "rel_path": target.rel_path,
        "job_id": os.path.basename(target.job_dir),
        "folder_depth": target.rel_path.count(os.sep) + 1,
        "charge": None,
        "mult": None,
        "spin": None,
        "n_atoms": None,
        "element_set": None,
        "n_unique_elements": None,
        "formula_hill": None,
        "heaviest_z": None,
        "has_tm": None,
        "has_lanthanide": None,
        "has_actinide": None,
        "net_charge_abs": None,
        "read_status": status,
    }


def process_job(target: JobTarget) -> Tuple[Dict, Optional[str]]:
    """Parse one job folder. Returns (row_dict, error_message_or_None)."""
    inp_path = os.path.join(target.job_dir, INP_NAME)
    if not os.path.isfile(inp_path):
        return _empty_row(target, "missing_inp"), "missing_inp"

    try:
        species, charge, mult = _parse_inp(inp_path)
    except (KeyError, ValueError, IndexError, UnboundLocalError) as e:
        return _empty_row(target, "corrupt_inp"), f"corrupt_inp: {e!r}"
    except Exception as e:
        return _empty_row(target, "parse_error"), f"parse_error: {e!r}"

    if not species:
        return _empty_row(target, "corrupt_inp"), "corrupt_inp: no atoms"

    element_set = sorted(set(species))
    z_values = [SYMBOL_TO_Z.get(el, 0) for el in element_set]
    z_set = set(z_values)
    heaviest_z = max(z_values) if z_values else 0

    row = {
        "vertical": target.vertical,
        "rel_path": target.rel_path,
        "job_id": os.path.basename(target.job_dir),
        "folder_depth": target.rel_path.count(os.sep) + 1,
        "charge": int(charge),
        "mult": int(mult),
        "spin": (int(mult) - 1) / 2.0,
        "n_atoms": len(species),
        "element_set": element_set,
        "n_unique_elements": len(element_set),
        "formula_hill": hill_formula(species),
        "heaviest_z": heaviest_z,
        "has_tm": bool(z_set & TRANSITION_METALS),
        "has_lanthanide": bool(z_set & LANTHANIDES),
        "has_actinide": bool(z_set & ACTINIDES),
        "net_charge_abs": abs(int(charge)),
        "read_status": "ok",
    }
    return row, None


def walk_vertical(root: str, vertical: str) -> Iterator[JobTarget]:
    """Yield every directory under root/vertical that contains an orca.inp."""
    base = os.path.join(root, vertical)
    if not os.path.isdir(base):
        return
    seen: set = set()
    for dirpath, dirs, files in os.walk(base, followlinks=True):
        real = os.path.realpath(dirpath)
        if real in seen:
            dirs[:] = []
            continue
        seen.add(real)
        if INP_NAME in files:
            rel = os.path.relpath(dirpath, root)
            yield JobTarget(
                root=root, vertical=vertical, job_dir=dirpath, rel_path=rel
            )
            # Job dirs are terminal: do not recurse into nested subfolders.
            dirs[:] = []


def list_verticals(root: str) -> List[str]:
    return sorted(
        d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))
    )


def _rows_to_table(rows: List[Dict]) -> pa.Table:
    return pa.Table.from_pylist(rows, schema=SCHEMA)


def build_vertical(
    root: str,
    vertical: str,
    out_dir: str,
    workers: int = 1,
    limit: Optional[int] = None,
    chunk_size: int = 5000,
    overwrite: bool = False,
    progress: bool = False,
) -> Dict[str, int]:
    """Process one vertical and write manifest_<vertical>.parquet + failures CSV.

    Returns a dict of read_status counts.
    """
    out_path = os.path.join(out_dir, f"manifest_{vertical}.parquet")
    fail_path = os.path.join(out_dir, f"failures_{vertical}.csv")

    if os.path.exists(out_path) and not overwrite:
        print(f"[{vertical}] exists, skipping (use --overwrite to rerun)")
        return {"skipped": 1}

    os.makedirs(out_dir, exist_ok=True)

    targets = list(walk_vertical(root, vertical))
    if limit is not None:
        targets = targets[:limit]

    print(f"[{vertical}] {len(targets)} job folders found")
    if not targets:
        # Write empty parquet so downstream dataset() still works.
        pq.write_table(pa.table({c.name: [] for c in SCHEMA}, schema=SCHEMA), out_path)
        return {"ok": 0, "missing_inp": 0, "corrupt_inp": 0, "parse_error": 0}

    counts: Counter = Counter()
    buffer: List[Dict] = []
    failures: List[Tuple[str, str]] = []

    bar = (
        tqdm(total=len(targets), desc=vertical, unit="job", smoothing=0.05)
        if progress
        else None
    )

    def _record(row, err):
        counts[row["read_status"]] += 1
        buffer.append(row)
        if err and row["read_status"] != "ok":
            failures.append((row["rel_path"], err))
        if bar is not None:
            bar.update(1)
            if counts.get("ok") is not None:
                bar.set_postfix(
                    ok=counts["ok"],
                    bad=counts["corrupt_inp"] + counts["missing_inp"] + counts["parse_error"],
                )

    writer = pq.ParquetWriter(out_path, SCHEMA, compression="snappy")
    try:
        if workers <= 1:
            for t in targets:
                row, err = process_job(t)
                _record(row, err)
                if len(buffer) >= chunk_size:
                    writer.write_table(_rows_to_table(buffer))
                    buffer.clear()
        else:
            with ProcessPoolExecutor(max_workers=workers) as ex:
                futures = [ex.submit(process_job, t) for t in targets]
                for fut in as_completed(futures):
                    row, err = fut.result()
                    _record(row, err)
                    if len(buffer) >= chunk_size:
                        writer.write_table(_rows_to_table(buffer))
                        buffer.clear()
        if buffer:
            writer.write_table(_rows_to_table(buffer))
    finally:
        writer.close()

    if failures:
        with open(fail_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["rel_path", "error"])
            w.writerows(failures)

    summary = {k: counts[k] for k in ("ok", "missing_inp", "corrupt_inp", "parse_error")}
    print(
        f"[{vertical}] ok={summary['ok']} corrupt={summary['corrupt_inp']} "
        f"missing={summary['missing_inp']} parse_error={summary['parse_error']}"
    )
    return summary


def merge_manifests(out_dir: str) -> str:
    """Concatenate all manifest_*.parquet in out_dir into manifest.parquet."""
    parts = sorted(
        os.path.join(out_dir, f)
        for f in os.listdir(out_dir)
        if f.startswith("manifest_") and f.endswith(".parquet")
    )
    if not parts:
        raise FileNotFoundError(f"No manifest_*.parquet found in {out_dir}")

    merged_path = os.path.join(out_dir, "manifest.parquet")
    writer: Optional[pq.ParquetWriter] = None
    try:
        for p in parts:
            t = pq.read_table(p)
            if t.schema != SCHEMA:
                # Diff columns for a useful error.
                want = {f.name for f in SCHEMA}
                got = {f.name for f in t.schema}
                raise ValueError(
                    f"Schema mismatch in {p}: missing={want - got} extra={got - want}"
                )
            if writer is None:
                writer = pq.ParquetWriter(merged_path, SCHEMA, compression="snappy")
            writer.write_table(t)
    finally:
        if writer is not None:
            writer.close()
    print(f"merged {len(parts)} files -> {merged_path}")
    return merged_path
