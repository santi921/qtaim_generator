"""Analysis subpackage. Submodules are imported lazily so that, e.g., the
SOAP comparator-embedding CLI does not pay the pandas import cost (and its
libstdc++ version dependency on some HPC systems).

Public re-exports are provided via ``__getattr__`` so existing call sites
``from qtaim_gen.source.analysis import stream`` keep working.
"""

from __future__ import annotations

from typing import Any

__all__ = ["stream", "stream_to_parquet", "LMDB_FILENAMES"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from qtaim_gen.source.analysis import streaming_aggregator

        return getattr(streaming_aggregator, name)
    raise AttributeError(name)
