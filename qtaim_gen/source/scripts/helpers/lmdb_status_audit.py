#!/usr/bin/env python3
"""Audit JSON-to-LMDB conversion status across verticals.

For each vertical (subdir of --root) and each LMDB type, report:
  - presence of the LMDB file
  - total record count
  - corrupt records (unpickle errors)
  - empty records (dict is {} or missing all critical sub-keys)
  - per-method coverage (e.g. fraction of charge records with hirshfeld present)
  - cross-LMDB drop: keys in structure.lmdb missing from each descriptor LMDB

Outputs:
  <output>/by_vertical/<vertical>.json      detailed counts + failed-key samples
  <output>/failed_keys/<vertical>__<dt>.txt one failed key per line (for reruns)
  <output>/summary.md                        aggregated cross-vertical table
  <output>/summary.json                      machine-readable aggregate

Shared validation logic lives in qtaim_gen.source.utils.lmdb_audit so that
json_to_lmdb.py can reuse it for opt-in post-write validation.
"""

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from glob import glob
from typing import Any, Dict, List, Optional

from qtaim_gen.source.utils.lmdb_audit import (
    DEFAULT_DATA_TYPES,
    EXPECTED_METHODS,
    audit_lmdb_paths,
)


def audit_vertical(
    vertical_dir: str,
    data_types: List[str],
    sample_failed: int,
    limit: Optional[int],
) -> Dict[str, Any]:
    vertical = os.path.basename(os.path.normpath(vertical_dir))
    paths = {dt: os.path.join(vertical_dir, f"{dt}.lmdb") for dt in data_types}
    report = audit_lmdb_paths(
        paths, sample_failed=sample_failed, limit=limit, workers=1
    )
    report["vertical"] = vertical
    report["vertical_dir"] = vertical_dir
    return report


def write_failed_keys(report: Dict[str, Any], out_dir: str) -> None:
    failed_dir = os.path.join(out_dir, "failed_keys")
    os.makedirs(failed_dir, exist_ok=True)
    vertical = report["vertical"]
    for dt, drop in report["cross_drop_vs_structure"].items():
        if drop["missing_from_lmdb"] == 0:
            continue
        path = os.path.join(failed_dir, f"{vertical}__{dt}__missing.txt")
        with open(path, "w") as f:
            for k in drop["missing_sample"]:
                f.write(k + "\n")
    for dt, info in report["per_type"].items():
        for status in (
            "empty", "malformed", "missing_critical", "no_bonds", "unpickle_error",
        ):
            samples = info["failed_samples"].get(status, [])
            if not samples:
                continue
            path = os.path.join(failed_dir, f"{vertical}__{dt}__{status}.txt")
            with open(path, "w") as f:
                for k in samples:
                    f.write(str(k) + "\n")


def render_markdown_summary(reports: List[Dict[str, Any]], data_types: List[str]) -> str:
    lines: List[str] = []
    lines.append("# LMDB conversion status audit")
    lines.append("")
    lines.append(f"Verticals scanned: {len(reports)} | data types: {', '.join(data_types)}")
    lines.append("")

    descriptor_types = [dt for dt in data_types if dt != "structure"]
    header = ["vertical", "structure"] + [f"{dt}_drop" for dt in descriptor_types]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for r in sorted(reports, key=lambda x: x["vertical"]):
        struct_n = r["per_type"].get("structure", {}).get("entries", 0)
        row = [r["vertical"], str(struct_n)]
        for dt in descriptor_types:
            drop = r["cross_drop_vs_structure"].get(dt, {}).get("missing_from_lmdb", "n/a")
            row.append(str(drop))
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    lines.append("## Per-record validity (counts)")
    lines.append("")
    val_header = [
        "vertical", "data_type", "entries", "ok", "empty", "no_bonds",
        "missing_critical", "malformed", "unpickle_err",
    ]
    lines.append("| " + " | ".join(val_header) + " |")
    lines.append("|" + "|".join(["---"] * len(val_header)) + "|")
    for r in sorted(reports, key=lambda x: x["vertical"]):
        for dt in data_types:
            info = r["per_type"][dt]
            if not info.get("exists", False):
                lines.append(f"| {r['vertical']} | {dt} | MISSING | - | - | - | - | - | - |")
                continue
            lines.append(
                "| " + " | ".join([
                    r["vertical"], dt, str(info["entries"]),
                    str(info["ok"]), str(info["empty"]),
                    str(info.get("no_bonds", 0)),
                    str(info["missing_critical"]),
                    str(info["malformed"]), str(info["unpickle_error"]),
                ]) + " |"
            )
    lines.append("")

    method_types = [dt for dt in data_types if dt in EXPECTED_METHODS]
    if method_types:
        lines.append("## Per-method coverage (records with method present)")
        lines.append("")
        for dt in method_types:
            methods = EXPECTED_METHODS[dt]
            mh = ["vertical", "entries"] + methods
            lines.append(f"### {dt}")
            lines.append("")
            lines.append("| " + " | ".join(mh) + " |")
            lines.append("|" + "|".join(["---"] * len(mh)) + "|")
            for r in sorted(reports, key=lambda x: x["vertical"]):
                info = r["per_type"][dt]
                if not info.get("exists", False):
                    continue
                row = [r["vertical"], str(info["entries"])]
                for m in methods:
                    row.append(str(info["method_counts"].get(m, 0)))
                lines.append("| " + " | ".join(row) + " |")
            lines.append("")

    return "\n".join(lines)


def _aggregate_totals(reports: List[Dict[str, Any]], data_types: List[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for dt in data_types:
        agg = {
            "entries": 0, "ok": 0, "empty": 0, "malformed": 0,
            "missing_critical": 0, "no_bonds": 0, "unpickle_error": 0,
            "missing_from_lmdb_vs_structure": 0,
            "method_counts": {m: 0 for m in EXPECTED_METHODS.get(dt, [])},
        }
        for r in reports:
            info = r["per_type"].get(dt, {})
            for k in (
                "entries", "ok", "empty", "malformed", "missing_critical",
                "no_bonds", "unpickle_error",
            ):
                agg[k] += int(info.get(k, 0) or 0)
            for m, c in info.get("method_counts", {}).items():
                agg["method_counts"][m] = agg["method_counts"].get(m, 0) + int(c)
            if dt != "structure":
                cross = r["cross_drop_vs_structure"].get(dt, {})
                agg["missing_from_lmdb_vs_structure"] += int(cross.get("missing_from_lmdb", 0) or 0)
        out[dt] = agg
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--root", required=True,
        help="Directory containing one subdir per vertical, each holding "
             "<data_type>.lmdb files (e.g. data/OMol4M_lmdbs/).")
    p.add_argument("--output", required=True, help="Directory to write the audit report.")
    p.add_argument("--verticals", nargs="+", default=None,
        help="Restrict to these vertical subdir names (default: all).")
    p.add_argument("--data-types", nargs="+", default=DEFAULT_DATA_TYPES,
        help="LMDB data types to check (default: %(default)s).")
    p.add_argument("--workers", type=int, default=0,
        help="Parallel workers (default: min(cpu_count, n_verticals); 1 disables).")
    p.add_argument("--sample-failed", type=int, default=200,
        help="Max sample keys to retain per failure category (default: 200).")
    p.add_argument("--limit", type=int, default=None,
        help="Stop after this many records per LMDB (debug only).")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    root = os.path.abspath(args.root)
    out_dir = os.path.abspath(args.output)

    if not os.path.isdir(root):
        print(f"ERROR: --root is not a directory: {root}", file=sys.stderr)
        return 2

    if args.verticals:
        vertical_dirs = [os.path.join(root, v) for v in args.verticals]
    else:
        vertical_dirs = sorted(
            d for d in glob(os.path.join(root, "*"))
            if os.path.isdir(d) and not os.path.basename(d).startswith(".")
        )
    vertical_dirs = [v for v in vertical_dirs if os.path.isdir(v)]
    if not vertical_dirs:
        print(f"ERROR: no verticals found under {root}", file=sys.stderr)
        return 2

    os.makedirs(out_dir, exist_ok=True)
    detail_dir = os.path.join(out_dir, "by_vertical")
    os.makedirs(detail_dir, exist_ok=True)

    n_verticals = len(vertical_dirs)
    workers = args.workers if args.workers > 0 else min(os.cpu_count() or 1, n_verticals)

    print(
        f"Auditing {n_verticals} verticals under {root} "
        f"with {workers} worker(s); types={args.data_types}",
        file=sys.stderr,
    )

    reports: List[Dict[str, Any]] = []
    if workers <= 1:
        for vd in vertical_dirs:
            print(f"  scanning {os.path.basename(vd)}...", file=sys.stderr)
            reports.append(audit_vertical(vd, args.data_types, args.sample_failed, args.limit))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(audit_vertical, vd, args.data_types, args.sample_failed, args.limit): vd
                for vd in vertical_dirs
            }
            for fut in as_completed(futures):
                vd = futures[fut]
                try:
                    rep = fut.result()
                except Exception as e:
                    print(f"  ERROR scanning {os.path.basename(vd)}: {e}", file=sys.stderr)
                    continue
                reports.append(rep)
                print(f"  done {rep['vertical']} in {rep['elapsed_sec']:.1f}s", file=sys.stderr)

    for r in reports:
        with open(os.path.join(detail_dir, f"{r['vertical']}.json"), "w") as f:
            json.dump(r, f, indent=2, default=str)
        write_failed_keys(r, out_dir)

    md = render_markdown_summary(reports, args.data_types)
    with open(os.path.join(out_dir, "summary.md"), "w") as f:
        f.write(md)

    aggregate = {
        "root": root,
        "verticals": [r["vertical"] for r in reports],
        "data_types": args.data_types,
        "totals": _aggregate_totals(reports, args.data_types),
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(aggregate, f, indent=2, default=str)

    print(f"\nReport written to {out_dir}", file=sys.stderr)
    print(f"  summary.md, summary.json", file=sys.stderr)
    print(f"  by_vertical/<vertical>.json", file=sys.stderr)
    print(f"  failed_keys/<vertical>__<dt>__<status>.txt", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
