"""
provenance.json: what produced each routine of a job folder (docs/engine_roadmap/provenance_plan.md).

    {"schema": 1, "engine_version": 4,
     "converter": {"by": "native", "mode": "fixed", "engine_version": 4},
     "routines": {routine: {"by": ..., "engine_version": ..., "spec": {...}, ...}},
     "skipped": {routine: {"reason": ..., "by": ...}},
     "engines": {engine: {its run record, e.g. the charge engine's _meta}}}

Writers update the job root's file, which move_results_to_folder merges into generator/
(merge); readers take generator/ and then the root on top (load), as timings.json is read.
"""

import json
import os
import re
from typing import Optional

from qtaim_gen.source.data.engine_versions import ENGINE_VERSION, ENGINES, is_current
from qtaim_gen.source.utils.atomic_write import atomic_json_write

PROVENANCE_FILE = "provenance.json"
SCHEMA = 1
_SECTIONS = ("routines", "skipped", "engines")
_MULTIWFN_BANNER = re.compile(r"Version\s+(\S+?),\s+(?:update|release) date:\s*(\S+)")


def empty() -> dict:
    return {"schema": SCHEMA, "engine_version": ENGINE_VERSION, "routines": {}, "skipped": {}, "engines": {}}


def _read(path: str) -> Optional[dict]:
    try:
        with open(path) as f:
            d = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    return d if isinstance(d, dict) else None


def merge(old: Optional[dict], new: Optional[dict]) -> dict:
    """new on top of old, entry by entry in each section (a routine of old that new does
    not mention is kept); top-level fields from new. A routine new records as produced
    is no longer skipped, and one new records as skipped is no longer produced."""
    out = empty()
    for d in (old or {}, new or {}):
        for r in d.get("routines") or {}:
            out["skipped"].pop(r, None)
        for r in d.get("skipped") or {}:
            out["routines"].pop(r, None)
        for k, v in d.items():
            if k in _SECTIONS and isinstance(v, dict):
                out[k].update(v)
            else:
                out[k] = v
    return out


def load(folder: str) -> dict:
    """The folder's provenance: generator/provenance.json with the root's on top."""
    return merge(_read(os.path.join(folder, "generator", PROVENANCE_FILE)),
                 _read(os.path.join(folder, PROVENANCE_FILE)))


def _update(folder: str, fn) -> None:
    path = os.path.join(folder, PROVENANCE_FILE)
    prov = merge(_read(path), None)
    fn(prov)
    prov["engine_version"] = ENGINE_VERSION
    atomic_json_write(path, prov)


def record_routines(folder: str, routines, by: str, engine_version: Optional[int] = None,
                    spec: Optional[dict] = None, **extra) -> None:
    """One entry per routine. Engines pass their ENGINE_VERSION; Multiwfn and ORCA entries
    carry their own build or parser version in extra."""
    entry = {"by": by, **({"engine_version": engine_version} if engine_version is not None else {}),
             **({"spec": spec} if spec else {}), **extra}

    def fn(prov):
        for r in routines:
            prov["routines"][r] = dict(entry)
            prov["skipped"].pop(r, None)
    _update(folder, fn)


def record_skips(folder: str, records: dict) -> None:
    """{routine: {"reason": ..., "by": ...}}: routines that can never be computed here."""
    def fn(prov):
        for r, rec in records.items():
            prov["skipped"][r] = rec
            prov["routines"].pop(r, None)
    _update(folder, fn)


def record_engine(folder: str, engine: str, record: dict) -> None:
    def fn(prov):
        prov["engines"][engine] = record
    _update(folder, fn)


def record_converter(folder: str, by: str, mode: Optional[str] = None, **extra) -> None:
    """The wavefunction converter: "native" with its mode, or "multiwfn" (its build in extra)."""
    def fn(prov):
        prov["converter"] = {"by": by, **({"mode": mode, "engine_version": ENGINE_VERSION} if mode else {}), **extra}
    _update(folder, fn)


def converter_tag(prov: dict) -> Optional[str]:
    """"native:<mode>" or "multiwfn" (the values timings.json's _converter used to hold)."""
    c = prov.get("converter") or {}
    if c.get("by") == "native":
        return f"native:{c.get('mode')}"
    return c.get("by")


def drop(folder: str, routines=(), converter: bool = False, engines: bool = False) -> None:
    """Remove entries from both the root and the generator/ file (a partial engine run,
    or outputs discarded with their wavefunction: then also the converter and the
    engines' run records, e.g. failure counts)."""
    routines = set(routines)
    for path in (os.path.join(folder, PROVENANCE_FILE), os.path.join(folder, "generator", PROVENANCE_FILE)):
        prov = _read(path)
        if prov is None:
            continue
        for sec in ("routines", "skipped"):
            for r in routines & set(prov.get(sec) or {}):
                prov[sec].pop(r)
        if converter:
            prov.pop("converter", None)
        if engines:
            prov["engines"] = {}
        atomic_json_write(path, prov)


def skipped(prov: dict) -> set:
    return set(prov.get("skipped") or {})


def produced_by(prov: dict, routine: str) -> Optional[str]:
    return ((prov.get("routines") or {}).get(routine) or {}).get("by")


def engine_current(prov: dict, routine: str) -> bool:
    """True when an engine produced the routine at a version that needs no recompute."""
    entry = (prov.get("routines") or {}).get(routine) or {}
    return entry.get("by") in ENGINES and is_current(routine, entry.get("engine_version"))


def engine_routines(prov: dict) -> frozenset:
    """The folder's routines that are current engine output."""
    return frozenset(r for r in prov.get("routines") or {} if engine_current(prov, r))


def engine_produced(prov: dict) -> frozenset:
    """The folder's routines an engine produced, at any version (values that are the
    engine's, current or not)."""
    return frozenset(r for r, e in (prov.get("routines") or {}).items() if (e or {}).get("by") in ENGINES)


def engine_versions(prov: dict) -> list:
    """The engine versions the folder's engine routines were computed at, sorted."""
    return sorted({e["engine_version"] for e in (prov.get("routines") or {}).values()
                   if (e or {}).get("by") in ENGINES and isinstance(e.get("engine_version"), int)})


def multiwfn_build(out_path: str, multiwfn_cmd: Optional[str] = None) -> dict:
    """{"version", "date", "binary"} from a Multiwfn output's banner
    ("Version 3.8(dev), update date: 2024-Oct-6"); fields it cannot find are left out."""
    build = {"binary": multiwfn_cmd} if multiwfn_cmd else {}
    try:
        with open(out_path, errors="replace") as f:
            for _, line in zip(range(20), f):
                m = _MULTIWFN_BANNER.search(line)
                if m:
                    build.update(version=m.group(1), date=m.group(2))
                    break
    except OSError:
        pass
    return build
