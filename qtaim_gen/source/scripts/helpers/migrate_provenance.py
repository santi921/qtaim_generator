#!/usr/bin/env python3
"""One-shot migration of engine job folders to provenance.json (docs/engine_roadmap/provenance_plan.md).

Engine runs before provenance.json kept their records in timings.json: "_engine_versions",
"_surface_engine_routines", "_converter" and "_skipped_routines". This writes the same
information as provenance.json and removes those keys from timings.json (generator/ and the
job root). "_timings_patched" stays: it describes the timing values.

A folder is migrated when it has one of those keys or a positive charge_engine,
surface_engine or qtaim_engine timing. Every entry it writes carries "legacy": true. Entries
follow what the pre-provenance readers concluded:
  - charge engine: the routines its version computed (no stamp: version 1), where the
    compiled JSONs hold them;
  - surface engine: its routine record, else the routines its version computed, where
    other.json holds their marker field;
  - QTAIM engine: version 1;
  - every other routine with a positive step timing: Multiwfn;
  - orca.json: ORCA, with its parser version;
  - the converter only when "_converter" recorded it (a folder without it is reconverted
    under --native_convert, as before).
An existing provenance.json wins over the legacy record for every entry it has.

Dry run unless --apply. Folders with a .processing.lock are left alone.

    migrate-provenance --root /pscratch/sd/s/santiago/actinide_test/prod_l1 --report prod_l1_migration.tsv
    migrate-provenance --root /pscratch/sd/s/santiago/actinide_test/prod_l1 --apply
"""

import argparse
import csv
import json
import os
import sys
from typing import Iterator, Optional

from qtaim_gen.source.data.engine_versions import routines_at
from qtaim_gen.source.data.multiwfn import ENGINE_ROUTINES, SURFACE_ENGINE_ROUTINES
from qtaim_gen.source.utils import provenance
from qtaim_gen.source.utils.atomic_write import atomic_json_write
from qtaim_gen.source.utils.validation import get_expected_timing_keys

LEGACY_KEYS = ("_engine_versions", "_surface_engine_routines", "_converter", "_skipped_routines")
ENGINE_TIMINGS = ("charge_engine", "surface_engine", "qtaim_engine")
OTHER_MARKERS = {"other_geometry": "mpp_full", "other_alie": "ALIE_Volume", "other_esp": "ESP_Volume"}
CHARGE_ROUTINES = ENGINE_ROUTINES - SURFACE_ENGINE_ROUTINES
_keys, _spin_keys = get_expected_timing_keys(full_set=2, spin_tf=True)
STEP_ROUTINES = (set(_keys) | set(_spin_keys) | set(OTHER_MARKERS)) - {"other", "convert"}


def _read(path: str) -> Optional[dict]:
    try:
        with open(path) as f:
            d = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    return d if isinstance(d, dict) else None


def _positive(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0


def job_folders(root: str) -> Iterator[str]:
    """Folders under root holding a generator/timings.json, or a timings.json and no generator/."""
    for dirpath, dirnames, files in os.walk(root):
        if os.path.basename(dirpath) == "generator":
            dirnames[:] = []
            if "timings.json" in files:
                yield os.path.dirname(dirpath)
        elif "timings.json" in files and "generator" not in dirnames:
            yield dirpath


def legacy_provenance(folder: str):
    """(provenance dict, legacy keys found, migrate?) from the folder's timings.json records."""
    gen = os.path.join(folder, "generator")
    timings = {}
    for base in (gen, folder):
        timings.update(_read(os.path.join(base, "timings.json")) or {})
    found = [k for k in LEGACY_KEYS if k in timings]
    if not found and not any(_positive(timings.get(k)) for k in ENGINE_TIMINGS):
        return None, found, False

    def data(name):
        return _read(os.path.join(gen, name)) or _read(os.path.join(folder, name)) or {}

    charge, fuzzy, bond, other, orca = (data(n) for n in ("charge.json", "fuzzy_full.json", "bond.json",
                                                          "other.json", "orca.json"))
    present = set(charge) | set(fuzzy) | ({"fuzzy_bond"} if bond.get("fuzzy_bond") else set())
    stamps = timings.get("_engine_versions") if isinstance(timings.get("_engine_versions"), dict) else {}
    prov = provenance.empty()

    def add(routine, entry):
        prov["routines"][routine] = {**entry, "legacy": True}

    if _positive(timings.get("charge_engine")):
        v = stamps.get("charge_engine", 1)
        for r in sorted(routines_at(v) & CHARGE_ROUTINES & present):
            add(r, {"by": "charge_engine", "engine_version": v})
    if _positive(timings.get("surface_engine")):
        v = stamps.get("surface_engine", 1)
        rec = timings.get("_surface_engine_routines")
        surface = set(rec) if isinstance(rec, list) else routines_at(v)
        for r in sorted(surface & SURFACE_ENGINE_ROUTINES):
            if other.get(OTHER_MARKERS[r]) is not None:
                add(r, {"by": "surface_engine", "engine_version": v})
    if _positive(timings.get("qtaim_engine")):
        add("qtaim", {"by": "qtaim_engine", "engine_version": 1})
    for r in sorted(STEP_ROUTINES - set(prov["routines"])):
        if _positive(timings.get(r)):
            add(r, {"by": "multiwfn"})
    if orca:
        add("orca", {"by": "orca", "parser_version": orca.get("orca_parser_version")})
    conv = timings.get("_converter")
    if isinstance(conv, str) and conv.startswith("native:"):
        prov["converter"] = {"by": "native", "mode": conv.split(":", 1)[1], "legacy": True}
    elif isinstance(conv, str):
        prov["converter"] = {"by": conv, "legacy": True}
    skipped = timings.get("_skipped_routines")
    if isinstance(skipped, dict):
        prov["skipped"] = skipped
    return prov, found, True


def migrate(folder: str, apply: bool) -> dict:
    row = {"folder": folder, "status": "", "engine_routines": "", "multiwfn_routines": "", "skipped": "",
           "converter": "", "removed_keys": ""}
    if os.path.exists(os.path.join(folder, ".processing.lock")):
        row["status"] = "locked"
        return row
    legacy, found, needed = legacy_provenance(folder)
    if not needed:
        row["status"] = "nothing"
        return row
    gen = os.path.join(folder, "generator")
    target = os.path.join(gen if os.path.isdir(gen) else folder, provenance.PROVENANCE_FILE)
    existing = _read(target)
    if existing and not found:
        row["status"] = "already"
        return row
    prov = provenance.merge(legacy, existing)
    prov["engine_version"] = (existing or {}).get("engine_version", legacy["engine_version"])
    routines = prov["routines"]
    row.update(
        engine_routines=",".join(sorted(r for r, e in routines.items() if e.get("by") not in ("multiwfn", "orca"))),
        multiwfn_routines=",".join(sorted(r for r, e in routines.items() if e.get("by") == "multiwfn")),
        skipped=",".join(sorted(prov["skipped"])), converter=provenance.converter_tag(prov) or "",
        removed_keys=",".join(found), status="migrated" if apply else "would_migrate")
    if not apply:
        return row
    atomic_json_write(target, prov)
    for base in (gen, folder):
        path = os.path.join(base, "timings.json")
        t = _read(path)
        if t is not None and any(k in t for k in LEGACY_KEYS):
            atomic_json_write(path, {k: v for k, v in t.items() if k not in LEGACY_KEYS})
    return row


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--root", help="tree to walk for job folders")
    src.add_argument("--folder_list", help="file with one job folder per line")
    p.add_argument("--apply", action="store_true", help="write provenance.json and strip timings.json (default: dry run)")
    p.add_argument("--report", help="TSV with one row per migrated or skipped folder")
    args = p.parse_args(argv)

    if args.root:
        folders = job_folders(args.root)
    else:
        with open(args.folder_list) as f:
            folders = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
    counts, rows = {}, []
    for folder in folders:
        row = migrate(folder, args.apply)
        counts[row["status"]] = counts.get(row["status"], 0) + 1
        if row["status"] != "nothing":
            rows.append(row)
    if args.report and rows:
        with open(args.report, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t")
            w.writeheader()
            w.writerows(rows)
    print(" ".join(f"{k}: {v}" for k, v in sorted(counts.items())) or "no job folders found")
    return 0


if __name__ == "__main__":
    sys.exit(main())
