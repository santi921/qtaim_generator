"""CLI tool to compute cumulative file sizes by filename and trailing-path suffix.

This script is designed for large datasets: it partitions the root by its
immediate children and scans each partition in a worker thread. Results are
aggregated and printed in human-readable form.

Example:
    python -m qtaim_gen.source.scripts.size_partition --root /path/to/root -w 12 -n 30
"""

from __future__ import annotations

import argparse
import os
import math
import concurrent.futures
from tqdm import tqdm
from collections import Counter
from typing import Dict, Tuple, Optional
import sqlite3
import json
from datetime import datetime


def _bytes_to_human(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    magnitude = int(math.log(n, 1024))
    val = n / (1024 ** magnitude)
    suf = ["B", "KB", "MB", "GB", "TB", "PB"][magnitude]
    return f"{val:.2f} {suf}"


def _scan_subtree(subroot: str) -> Tuple[Counter, Counter, int, int]:
    """Walk a subtree and return (sizes_by_name, sizes_by_suffix).

    sizes_by_name maps basename -> cumulative bytes.
    sizes_by_suffix maps trailing two-component path (e.g. "generator/time.json") -> bytes.
    """
    sizes_by_name = Counter()
    sizes_by_suffix = Counter()

    for dirpath, dirnames, filenames in os.walk(subroot, followlinks=False):
        for fname in filenames:
            try:
                fpath = os.path.join(dirpath, fname)
                stat = os.stat(fpath, follow_symlinks=False)
                size = stat.st_size
            except (FileNotFoundError, PermissionError, OSError):
                continue

            sizes_by_name[fname] += size

            # build suffix: last two path components relative to subroot
            rel = os.path.relpath(fpath, subroot)
            parts = rel.split(os.sep)
            if len(parts) >= 2:
                suffix = os.path.join(parts[-2], parts[-1])
            else:
                suffix = parts[-1]
            sizes_by_suffix[suffix] += size

    total_bytes = sum(sizes_by_name.values())
    total_files = sum(1 for _dirpath, _dirnames, filenames in os.walk(subroot, followlinks=False) for _ in (filenames) for __ in ([None] if filenames else []))
    # total_files computed conservatively: count of all filenames
    total_files = sum(len(filenames) for _dirpath, _dirnames, filenames in os.walk(subroot, followlinks=False))
    return sizes_by_name, sizes_by_suffix, total_bytes, total_files


def _init_tracking_db(db_path: str) -> None:
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS size_scan (
            subroot TEXT PRIMARY KEY,
            scanned_at TEXT,
            total_bytes INTEGER,
            total_files INTEGER,
            status TEXT,
            error TEXT
        )
        """
    )
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS size_summary (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            scanned_at TEXT,
            total_bytes INTEGER,
            note TEXT
        )
        """
    )
    conn.commit()
    conn.close()


def _write_subroot_record(db_path: str, subroot: str, total_bytes: int, total_files: int, status: str, error: Optional[str] = None) -> None:
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute(
        "INSERT OR REPLACE INTO size_scan (subroot, scanned_at, total_bytes, total_files, status, error) VALUES (?, ?, ?, ?, ?, ?)",
        (subroot, datetime.utcnow().isoformat(), total_bytes, total_files, status, error or ""),
    )
    conn.commit()
    conn.close()


def _write_summary(db_path: str, total_bytes: int, note: str = "") -> None:
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("INSERT INTO size_summary (scanned_at, total_bytes, note) VALUES (?, ?, ?)", (datetime.utcnow().isoformat(), total_bytes, note))
    conn.commit()
    conn.close()


def scan_sizes(root: str, max_workers: int = 8, track_db: Optional[str] = None) -> Tuple[Dict[str, int], Dict[str, int]]:
    """Scan `root` directory in parallel and return two dicts: (sizes_by_name, sizes_by_suffix).

    The root is partitioned by its immediate children (top-level entries)
    and each partition is scanned in a separate thread.
    """
    root = os.path.abspath(root)
    try:
        children = [os.path.join(root, d) for d in os.listdir(root)]
    except (FileNotFoundError, PermissionError):
        return {}, {}

    subroots = [p for p in children if os.path.isdir(p)]
    if not subroots and os.path.isdir(root):
        subroots = [root]

    total_name = Counter()
    total_suffix = Counter()
    grand_total_bytes = 0

    if track_db:
        _init_tracking_db(track_db)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_scan_subtree, s): s for s in subroots}
        # Show progress as partitions complete
        for fut in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Scanning partitions"):
            subroot = futures[fut]
            try:
                s_name, s_suffix, total_bytes, total_files = fut.result()
                total_name.update(s_name)
                total_suffix.update(s_suffix)
                grand_total_bytes += total_bytes
                if track_db:
                    _write_subroot_record(track_db, subroot, total_bytes, total_files, "ok")
            except Exception as e:
                if track_db:
                    _write_subroot_record(track_db, subroot, 0, 0, "error", str(e))
                continue

    if track_db:
        _write_summary(track_db, grand_total_bytes, note=f"scanned root {root}")

    return dict(total_name), dict(total_suffix)


def print_top_sizes(sizes: Dict[str, int], top_n: int = 20, label: str = "files") -> None:
    items = sorted(sizes.items(), key=lambda t: t[1], reverse=True)[:top_n]
    for name, sz in items:
        print(f"Total size of {label} '{name}': {_bytes_to_human(sz)}")


def main() -> None:
    p = argparse.ArgumentParser(description="Compute cumulative sizes by filename and trailing-path suffix")
    p.add_argument("--root", "-r", required=True, help="Root directory to scan")
    p.add_argument("--workers", "-w", type=int, default=8, help="Number of threads")
    p.add_argument("--top", "-n", type=int, default=20, help="Top N items to print")
    p.add_argument("--track-db", help="Path to SQLite DB to record per-subroot progress and summary")
    args = p.parse_args()

    sizes_by_name, sizes_by_suffix = scan_sizes(args.root, max_workers=args.workers, track_db=args.track_db)

    print("\nTop filenames by cumulative size:")
    print_top_sizes(sizes_by_name, top_n=args.top, label="files")

    print("\nTop trailing-path suffixes by cumulative size (e.g. 'generator/time.json'):")
    print_top_sizes(sizes_by_suffix, top_n=args.top, label="suffixes")


if __name__ == "__main__":
    main()
