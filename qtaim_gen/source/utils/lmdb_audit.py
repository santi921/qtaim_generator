"""Shared LMDB audit logic.

Used by:
  - scripts/helpers/lmdb_status_audit.py (cross-vertical CLI audit)
  - scripts/json_to_lmdb.py (opt-in post-write validation)

Per-record validation is shallow: each LMDB type has a `validate_record`
classifier that returns one of
{"ok", "empty", "malformed", "missing_critical", "no_bonds"}
plus the list of expected methods that are populated. "ok" means at least
one expected method is present and non-empty (so the validator stays correct
across mixed full_set tiers). "no_bonds" is bond-only (see validate_record).
"""

import json
import logging
import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

import lmdb

logger = logging.getLogger(__name__)


DEFAULT_DATA_TYPES: List[str] = [
    "structure",
    "charge",
    "qtaim",
    "bond",
    "fuzzy",
    "other",
    "orca",
    "timings",
]

# Methods we expect to find inside each LMDB record. Aligned with
# get_expected_json_keys() in scripts/json_to_lmdb.py. Tier 0 / 1 / 2 keys
# are all listed; validate_record returns "ok" if at least one is populated.
EXPECTED_METHODS: Dict[str, List[str]] = {
    # Multiwfn-derived charge methods (tier 0 / 1 / 2 from get_expected_json_keys)
    # plus the five ORCA-derived charge tables that get copied into charge.json
    # by merge_orca_into_charge_json. The _orca-suffixed entries are stored as
    # {"charge": {...}, optional "spin": {...}, optional "population": {...}};
    # validate_record's inner-charge check picks them up the same way it does
    # the Multiwfn entries.
    "charge": [
        "hirshfeld", "adch", "cm5", "becke", "mbis", "chelpg", "vdd", "bader",
        "mulliken_orca", "loewdin_orca", "hirshfeld_orca", "mayer_orca", "mbis_orca",
    ],
    # Tier 0 / 1 / 2 Multiwfn-derived bond schemes plus the two ORCA-derived
    # bond-order tables (Mayer + Loewdin) that get copied into bond.json from
    # orca.out. ORCA bond orders are valid bond evidence and count toward "ok".
    "bond": [
        "fuzzy_bond",
        "ibsi_bond",
        "laplacian_bond",
        "mayer_orca",
        "loewdin_orca",
    ],
    "fuzzy": [
        "becke_fuzzy_density",
        "hirsh_fuzzy_density",
        "elf_fuzzy",
        "mbis_fuzzy_density",
        "grad_norm_rho_fuzzy",
        "laplacian_rho_fuzzy",
        "hirsh_fuzzy_spin",
        "becke_fuzzy_spin",
        "mbis_fuzzy_spin",
    ],
}

ORCA_CRITICAL_KEYS: Tuple[str, ...] = ("final_energy_eh",)
QTAIM_ATOM_FEATURE_KEYS: Tuple[str, ...] = (
    "density_all",
    "lap_e_density",
    "energy_density",
)


def _is_empty(obj: Any) -> bool:
    if obj is None:
        return True
    if isinstance(obj, (dict, list, tuple, set, str, bytes)):
        return len(obj) == 0
    return False


def validate_record(data_type: str, value: Any) -> Tuple[str, List[str]]:
    """Return (status, present_methods).

    status: "ok" | "empty" | "malformed" | "missing_critical" | "no_bonds".

    "no_bonds" is bond-only and means the record is well-formed (dict with the
    expected method keys present) but every bond scheme is empty. This is
    sometimes legitimate (single atoms, noble gases) and sometimes an upstream
    Multiwfn / ORCA miss; either way it is not the same severity as a
    malformed or missing-critical-key record.
    """
    if not isinstance(value, dict):
        return "malformed", []

    if _is_empty(value):
        return "empty", []

    if data_type == "structure":
        required = ("molecule", "bonds", "charge", "spin")
        if any(k not in value for k in required):
            return "missing_critical", []
        return "ok", []

    if data_type == "orca":
        if any(k not in value or value.get(k) is None for k in ORCA_CRITICAL_KEYS):
            return "missing_critical", []
        present = []
        if value.get("scf_converged") is True:
            present.append("scf_converged")
        if value.get("gradient"):
            present.append("gradient")
        if value.get("dipole_au"):
            present.append("dipole")
        return "ok", present

    if data_type == "qtaim":
        for k, v in value.items():
            if "_" in str(k):
                continue
            if isinstance(v, dict) and any(fk in v for fk in QTAIM_ATOM_FEATURE_KEYS):
                return "ok", []
        return "missing_critical", []

    if data_type == "other":
        return "ok", []

    if data_type == "timings":
        return "ok", []

    if data_type in EXPECTED_METHODS:
        present = []
        for method in EXPECTED_METHODS[data_type]:
            # bond fixtures use both legacy ('fuzzy', 'ibsi') and current
            # ('fuzzy_bond', 'ibsi_bond') naming. parse_bond_data accepts both.
            candidates = [method]
            if data_type == "bond" and method.endswith("_bond"):
                candidates.append(method[: -len("_bond")])
            for candidate in candidates:
                if candidate not in value:
                    continue
                sub = value[candidate]
                if not isinstance(sub, dict) or _is_empty(sub):
                    continue
                if data_type == "charge":
                    # Stricter charge check: a method counts as populated only
                    # if its inner per-atom 'charge' (or 'spin') dict is non-empty.
                    # A method that wrote only metadata (e.g. dipole) without
                    # any per-atom charges should NOT pass — that is a real
                    # silent failure mode worth surfacing.
                    inner = sub.get("charge") or sub.get("spin")
                    if isinstance(inner, dict) and not _is_empty(inner):
                        present.append(method)
                        break
                    # Outer dict non-empty but no per-atom payload: skip.
                    continue
                # Non-charge types: outer dict non-empty is sufficient.
                present.append(method)
                break
        if not present:
            # bond is the only type where "no methods populated" is plausibly
            # legitimate (atomic systems, noble gases, upstream fuzzy_bond
            # miss). Carve it out so it can be triaged separately.
            if data_type == "bond":
                return "no_bonds", []
            return "missing_critical", []
        return "ok", present

    return "ok", []


def scan_lmdb(
    lmdb_path: str,
    data_type: str,
    sample_failed: int = 200,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    """Single-pass scan of one LMDB file. Returns counts + key set + samples."""
    result: Dict[str, Any] = {
        "data_type": data_type,
        "path": lmdb_path,
        "exists": os.path.exists(lmdb_path),
        "entries": 0,
        "length_field": None,
        "ok": 0,
        "empty": 0,
        "malformed": 0,
        "missing_critical": 0,
        "no_bonds": 0,
        "unpickle_error": 0,
        "method_counts": {},
        "failed_samples": {
            "empty": [],
            "malformed": [],
            "missing_critical": [],
            "no_bonds": [],
            "unpickle_error": [],
        },
        "all_keys": set(),
    }

    if not result["exists"]:
        return result

    try:
        env = lmdb.open(
            lmdb_path,
            readonly=True,
            lock=False,
            subdir=False,
            readahead=False,
            meminit=False,
        )
    except lmdb.Error as e:
        result["open_error"] = str(e)
        return result

    method_counts: Dict[str, int] = {m: 0 for m in EXPECTED_METHODS.get(data_type, [])}

    with env.begin() as txn:
        cursor = txn.cursor()
        seen_data = 0
        for key, value in cursor:
            try:
                key_str = key.decode("ascii", errors="replace")
            except Exception:
                key_str = repr(key)

            if key_str == "length":
                try:
                    result["length_field"] = pickle.loads(value)
                except Exception:
                    pass
                continue

            result["entries"] += 1
            result["all_keys"].add(key_str)
            seen_data += 1

            try:
                obj = pickle.loads(value)
            except Exception as e:
                result["unpickle_error"] += 1
                if len(result["failed_samples"]["unpickle_error"]) < sample_failed:
                    result["failed_samples"]["unpickle_error"].append(
                        f"{key_str}\t{type(e).__name__}: {str(e)[:80]}"
                    )
                if limit is not None and seen_data >= limit:
                    break
                continue

            status, present_methods = validate_record(data_type, obj)
            result[status] = result.get(status, 0) + 1
            for m in present_methods:
                method_counts[m] = method_counts.get(m, 0) + 1

            if status != "ok" and len(result["failed_samples"][status]) < sample_failed:
                result["failed_samples"][status].append(key_str)

            if limit is not None and seen_data >= limit:
                break

    env.close()
    result["method_counts"] = method_counts
    return result


#: Default statuses treated as failures for filtering purposes.
DEFAULT_EXCLUDE_STATUSES: FrozenSet[str] = frozenset(
    {"unpickle_error", "missing_critical", "empty", "malformed"}
)


def collect_bad_keys(
    lmdb_path: str,
    data_type: str,
    exclude_statuses: Optional[Set[str]] = None,
    limit: Optional[int] = None,
) -> Dict[str, str]:
    """Scan lmdb_path and return {key: status} for every key whose status is in exclude_statuses.

    Unlike scan_lmdb, no sampling: every failing key is returned.
    exclude_statuses defaults to DEFAULT_EXCLUDE_STATUSES.
    """
    if exclude_statuses is None:
        exclude_statuses = set(DEFAULT_EXCLUDE_STATUSES)

    bad: Dict[str, str] = {}
    if not os.path.exists(lmdb_path):
        return bad

    try:
        env = lmdb.open(
            lmdb_path,
            readonly=True,
            lock=False,
            subdir=False,
            readahead=False,
            meminit=False,
        )
    except lmdb.Error as e:
        logger.warning("collect_bad_keys: cannot open %s: %s", lmdb_path, e)
        return bad

    with env.begin() as txn:
        cursor = txn.cursor()
        seen = 0
        for key_bytes, value in cursor:
            try:
                key_str = key_bytes.decode("ascii", errors="replace")
            except Exception:
                key_str = repr(key_bytes)

            if key_str == "length":
                continue

            seen += 1

            try:
                obj = pickle.loads(value)
            except Exception:
                if "unpickle_error" in exclude_statuses:
                    bad[key_str] = "unpickle_error"
                if limit is not None and seen >= limit:
                    break
                continue

            status, _ = validate_record(data_type, obj)
            if status in exclude_statuses:
                bad[key_str] = status

            if limit is not None and seen >= limit:
                break

    env.close()
    return bad


def audit_lmdb_paths(
    lmdb_paths: Dict[str, str],
    sample_failed: int = 200,
    limit: Optional[int] = None,
    workers: int = 1,
) -> Dict[str, Any]:
    """Scan a mapping {data_type: lmdb_path}. Optionally parallel across types.

    Returns a single-set audit report:
      {
        "elapsed_sec": float,
        "per_type":     {data_type: scan_lmdb-result with all_keys collapsed to int},
        "cross_drop_vs_structure": {data_type: {missing_from_lmdb, ...}}
      }
    """
    t0 = time.time()
    per_type: Dict[str, Dict[str, Any]] = {}

    if workers <= 1 or len(lmdb_paths) <= 1:
        for dt, path in lmdb_paths.items():
            per_type[dt] = scan_lmdb(path, dt, sample_failed, limit)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futs = {
                pool.submit(scan_lmdb, path, dt, sample_failed, limit): dt
                for dt, path in lmdb_paths.items()
            }
            for fut in as_completed(futs):
                dt = futs[fut]
                per_type[dt] = fut.result()

    cross_drop: Dict[str, Dict[str, Any]] = {}
    structure_info = per_type.get("structure", {})
    structure_keys = structure_info.get("all_keys", set())
    descriptor_types = [dt for dt in lmdb_paths if dt != "structure"]
    if not structure_keys and descriptor_types:
        # Without a structure.lmdb (or with an empty one) we cannot compute
        # cross-LMDB drop. Warn rather than fail silently — a missing
        # structure.lmdb is itself a strong signal that the conversion is
        # broken or that the wrong target_dir was passed in.
        if "structure" in lmdb_paths and not structure_info.get("exists", False):
            logger.warning(
                "structure.lmdb missing at %s; cross-LMDB drop check skipped.",
                lmdb_paths["structure"],
            )
        elif "structure" in lmdb_paths and structure_info.get("entries", 0) == 0:
            logger.warning(
                "structure.lmdb at %s contains no records; "
                "cross-LMDB drop check skipped.",
                lmdb_paths["structure"],
            )
        elif "structure" not in lmdb_paths:
            logger.info(
                "No structure.lmdb in audit set; cross-LMDB drop check skipped."
            )
    if structure_keys:
        for dt in lmdb_paths:
            if dt == "structure":
                continue
            other_keys = per_type[dt].get("all_keys", set())
            missing = structure_keys - other_keys
            extra = other_keys - structure_keys
            cross_drop[dt] = {
                "structure_count": len(structure_keys),
                "lmdb_count": len(other_keys),
                "missing_from_lmdb": len(missing),
                "extra_in_lmdb": len(extra),
                "missing_sample": sorted(missing)[:sample_failed],
                "extra_sample": sorted(extra)[:sample_failed],
            }

    for dt in per_type:
        per_type[dt]["all_keys"] = len(per_type[dt]["all_keys"])

    return {
        "elapsed_sec": time.time() - t0,
        "per_type": per_type,
        "cross_drop_vs_structure": cross_drop,
    }


def render_audit_md(
    report: Dict[str, Any],
    data_types: List[str],
    label: str = "audit",
) -> str:
    """Render a single audit report (one set of LMDBs) as markdown."""
    lines: List[str] = []
    lines.append(f"# LMDB validation: {label}")
    lines.append("")
    lines.append(f"Elapsed: {report.get('elapsed_sec', 0):.1f}s")
    lines.append("")

    descriptor_types = [dt for dt in data_types if dt != "structure"]
    cross = report.get("cross_drop_vs_structure", {})
    structure_n = report["per_type"].get("structure", {}).get("entries", 0)

    if descriptor_types:
        lines.append("## Cross-LMDB drop vs structure.lmdb")
        lines.append("")
        header = ["data_type", "structure_count", "lmdb_count", "missing_from_lmdb", "extra_in_lmdb"]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "|".join(["---"] * len(header)) + "|")
        for dt in descriptor_types:
            d = cross.get(dt, {})
            lines.append(
                "| "
                + " | ".join(
                    [
                        dt,
                        str(d.get("structure_count", structure_n)),
                        str(d.get("lmdb_count", "n/a")),
                        str(d.get("missing_from_lmdb", "n/a")),
                        str(d.get("extra_in_lmdb", "n/a")),
                    ]
                )
                + " |"
            )
        lines.append("")

    lines.append("## Per-record validity")
    lines.append("")
    val_header = [
        "data_type", "entries", "ok", "empty", "no_bonds",
        "missing_critical", "malformed", "unpickle_err",
    ]
    lines.append("| " + " | ".join(val_header) + " |")
    lines.append("|" + "|".join(["---"] * len(val_header)) + "|")
    for dt in data_types:
        info = report["per_type"].get(dt, {})
        if not info.get("exists", False):
            lines.append(f"| {dt} | MISSING | - | - | - | - | - | - |")
            continue
        lines.append(
            "| " + " | ".join([
                dt,
                str(info["entries"]),
                str(info["ok"]),
                str(info["empty"]),
                str(info.get("no_bonds", 0)),
                str(info["missing_critical"]),
                str(info["malformed"]),
                str(info["unpickle_error"]),
            ]) + " |"
        )
    lines.append("")

    method_types = [dt for dt in data_types if dt in EXPECTED_METHODS]
    if method_types:
        lines.append("## Per-method coverage")
        lines.append("")
        for dt in method_types:
            info = report["per_type"].get(dt, {})
            if not info.get("exists", False):
                continue
            methods = EXPECTED_METHODS[dt]
            mh = ["entries"] + methods
            lines.append(f"### {dt}")
            lines.append("")
            lines.append("| " + " | ".join(mh) + " |")
            lines.append("|" + "|".join(["---"] * len(mh)) + "|")
            row = [str(info["entries"])]
            for m in methods:
                row.append(str(info["method_counts"].get(m, 0)))
            lines.append("| " + " | ".join(row) + " |")
            lines.append("")

    return "\n".join(lines)


def write_audit_report(
    report: Dict[str, Any],
    out_dir: str,
    data_types: List[str],
    label: str = "audit",
) -> Dict[str, str]:
    """Write summary.md / summary.json / failed_keys/ for one audit report.

    Returns paths written.
    """
    os.makedirs(out_dir, exist_ok=True)
    paths_written: Dict[str, str] = {}

    json_path = os.path.join(out_dir, f"{label}.json")
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    paths_written["json"] = json_path

    md_path = os.path.join(out_dir, f"{label}.md")
    with open(md_path, "w") as f:
        f.write(render_audit_md(report, data_types, label=label))
    paths_written["md"] = md_path

    failed_dir = os.path.join(out_dir, f"{label}_failed_keys")
    failed_written = False

    for dt, drop in report.get("cross_drop_vs_structure", {}).items():
        if drop.get("missing_from_lmdb", 0) > 0:
            os.makedirs(failed_dir, exist_ok=True)
            p = os.path.join(failed_dir, f"{dt}__missing_from_lmdb.txt")
            with open(p, "w") as f:
                for k in drop["missing_sample"]:
                    f.write(k + "\n")
            failed_written = True

    for dt, info in report["per_type"].items():
        for status in ("empty", "malformed", "missing_critical", "no_bonds", "unpickle_error"):
            samples = info.get("failed_samples", {}).get(status, [])
            if not samples:
                continue
            os.makedirs(failed_dir, exist_ok=True)
            p = os.path.join(failed_dir, f"{dt}__{status}.txt")
            with open(p, "w") as f:
                for s in samples:
                    f.write(str(s) + "\n")
            failed_written = True

    if failed_written:
        paths_written["failed_keys"] = failed_dir

    return paths_written


def render_summary_line(report: Dict[str, Any], data_types: List[str]) -> str:
    """One-line summary for log output."""
    parts = []
    for dt in data_types:
        info = report["per_type"].get(dt, {})
        if not info.get("exists", False):
            parts.append(f"{dt}=MISSING")
            continue
        n = info["entries"]
        bad = info["empty"] + info["malformed"] + info["missing_critical"] + info["unpickle_error"]
        no_bonds = info.get("no_bonds", 0)
        if bad or no_bonds:
            tags = []
            if bad:
                tags.append(f"{bad} bad")
            if no_bonds:
                tags.append(f"{no_bonds} no_bonds")
            parts.append(f"{dt}={info['ok']}/{n} (" + ", ".join(tags) + ")")
        else:
            parts.append(f"{dt}={n} ok")
    return "  ".join(parts)
