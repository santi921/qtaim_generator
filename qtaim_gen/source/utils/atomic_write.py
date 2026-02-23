"""Atomic JSON write utility — no heavy dependencies (RDKit, numpy, etc.)."""

import json
import os
import tempfile
from typing import Optional


def atomic_json_write(path: str, data: dict, indent: Optional[int] = 4) -> None:
    """Write JSON atomically using temp-file-then-rename.

    Safe for HPC environments where jobs may be killed mid-write.
    The temp file is created in the same directory as the target to
    ensure os.replace() is an atomic rename (same filesystem).
    """
    dir_name = os.path.dirname(path) or "."
    with tempfile.NamedTemporaryFile(
        mode="w", dir=dir_name, suffix=".tmp", delete=False
    ) as tmp:
        json.dump(data, tmp, indent=indent)
        tmp_path = tmp.name
    os.replace(tmp_path, path)
