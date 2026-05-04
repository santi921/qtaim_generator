"""Streaming aggregator: foundation for analysis Streams C-G.

Open one or more LMDBs in a vertical, iterate keys, run a per-record
function, accumulate into a DataFrame and (optionally) a parquet file.

Multi-vertical loops are the caller's responsibility.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any, Callable, Iterable

import lmdb
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

LMDB_FILENAMES = {
    "structure": "structure.lmdb",
    "charge": "charge.lmdb",
    "qtaim": "qtaim.lmdb",
    "bond": "bond.lmdb",
    "fuzzy": "fuzzy.lmdb",
    "other": "other.lmdb",
    "orca": "orca.lmdb",
    "timings": "timings.lmdb",
}

RESERVED_KEYS = {"length"}


def _open(root: Path, lmdb_type: str) -> lmdb.Environment:
    if lmdb_type not in LMDB_FILENAMES:
        raise ValueError(
            f"unknown lmdb_type {lmdb_type!r}; expected one of {sorted(LMDB_FILENAMES)}"
        )
    path = root / LMDB_FILENAMES[lmdb_type]
    if not path.exists():
        raise FileNotFoundError(f"missing {path}")
    return lmdb.open(
        str(path),
        subdir=False,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
    )


def _iter_keys(env: lmdb.Environment) -> Iterable[str]:
    with env.begin(write=False) as txn:
        cur = txn.cursor()
        for raw, _ in cur:
            try:
                key = raw.decode("ascii")
            except UnicodeDecodeError:
                continue
            if key in RESERVED_KEYS:
                continue
            yield key


def _key_set(env: lmdb.Environment) -> set[str]:
    return set(_iter_keys(env))


def _fetch(env: lmdb.Environment, key: str) -> Any:
    with env.begin(write=False) as txn:
        raw = txn.get(key.encode("ascii"))
    if raw is None:
        return None
    return pickle.loads(raw)


def _normalize_result(key: str, result: Any) -> list[dict]:
    if result is None:
        return []
    if isinstance(result, dict):
        if "key" not in result:
            return [{"key": key, **result}]
        return [result]
    if isinstance(result, list):
        out = []
        for r in result:
            if not isinstance(r, dict):
                raise TypeError(
                    f"per_record_fn returned list element of type {type(r).__name__}; expected dict"
                )
            out.append(r if "key" in r else {"key": key, **r})
        return out
    raise TypeError(
        f"per_record_fn returned {type(result).__name__}; expected dict or list[dict]"
    )


def _maybe_progress(it, total, enabled: bool, desc: str | None = None):
    if not enabled:
        return it
    try:
        from tqdm import tqdm
    except ImportError:
        return it
    return tqdm(it, total=total, desc=desc)


def _resolve_keys(
    envs: dict[str, lmdb.Environment],
    lmdb_types: list[str],
    keys: Iterable[str] | None,
) -> tuple[list[str], dict[str, set[str]]]:
    key_sets = {t: _key_set(env) for t, env in envs.items()}
    if keys is None:
        primary = lmdb_types[0]
        cand_keys = list(_iter_keys(envs[primary]))
    else:
        cand_keys = list(keys)
    return cand_keys, key_sets


def stream(
    root: Path | str,
    lmdb_types: list[str],
    per_record_fn: Callable[..., dict | list[dict] | None],
    keys: Iterable[str] | None = None,
    progress: bool = True,
) -> pd.DataFrame:
    """Stream over one or more LMDBs in a vertical, applying per_record_fn.

    Inner-join semantics across `lmdb_types`: if a candidate key is missing
    from any required LMDB, it is dropped. The list of dropped keys is
    attached to the returned DataFrame as ``df.attrs["dropped_keys"]``.

    `per_record_fn(key, *values)` returns one of:
      - a dict (one row, "key" column injected if absent)
      - a list of dicts (multiple rows, "key" injected per row if absent)
      - None or empty list (zero rows for that key)
    """
    if not lmdb_types:
        raise ValueError("lmdb_types must contain at least one entry")
    root = Path(root)
    envs = {t: _open(root, t) for t in lmdb_types}
    try:
        cand_keys, key_sets = _resolve_keys(envs, lmdb_types, keys)
        rows: list[dict] = []
        dropped: list[str] = []
        iterator = _maybe_progress(cand_keys, len(cand_keys), progress, desc="stream")
        for k in iterator:
            if not all(k in key_sets[t] for t in lmdb_types):
                dropped.append(k)
                continue
            values = [_fetch(envs[t], k) for t in lmdb_types]
            rows.extend(_normalize_result(k, per_record_fn(k, *values)))
        df = pd.DataFrame(rows)
        df.attrs["dropped_keys"] = dropped
        df.attrs["lmdb_types"] = list(lmdb_types)
        if dropped:
            logger.info(
                "stream: dropped %d / %d keys missing from required LMDBs",
                len(dropped),
                len(cand_keys),
            )
        return df
    finally:
        for env in envs.values():
            env.close()


def stream_to_parquet(
    root: Path | str,
    lmdb_types: list[str],
    per_record_fn: Callable[..., dict | list[dict] | None],
    output_path: Path | str,
    keys: Iterable[str] | None = None,
    progress: bool = True,
    chunk_size: int = 10_000,
) -> Path:
    """Same as `stream()` but writes results incrementally to parquet.

    Rows are flushed to disk every `chunk_size` rows so very large verticals
    do not require holding the full DataFrame in memory.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not lmdb_types:
        raise ValueError("lmdb_types must contain at least one entry")
    root = Path(root)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    envs = {t: _open(root, t) for t in lmdb_types}
    writer: pq.ParquetWriter | None = None
    buffer: list[dict] = []
    dropped: list[str] = []
    try:
        cand_keys, key_sets = _resolve_keys(envs, lmdb_types, keys)
        iterator = _maybe_progress(cand_keys, len(cand_keys), progress, desc="stream_to_parquet")

        def _flush():
            nonlocal writer
            if not buffer:
                return
            table = pa.Table.from_pylist(buffer)
            if writer is None:
                writer = pq.ParquetWriter(str(output_path), table.schema)
            writer.write_table(table)
            buffer.clear()

        for k in iterator:
            if not all(k in key_sets[t] for t in lmdb_types):
                dropped.append(k)
                continue
            values = [_fetch(envs[t], k) for t in lmdb_types]
            buffer.extend(_normalize_result(k, per_record_fn(k, *values)))
            if len(buffer) >= chunk_size:
                _flush()
        _flush()
        if writer is None:
            # No rows produced; emit an empty parquet with a single null-typed column.
            empty = pa.table({"key": pa.array([], type=pa.string())})
            pq.write_table(empty, str(output_path))
        if dropped:
            logger.info(
                "stream_to_parquet: dropped %d keys missing from required LMDBs",
                len(dropped),
            )
        return output_path
    finally:
        if writer is not None:
            writer.close()
        for env in envs.values():
            env.close()
