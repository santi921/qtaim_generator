"""CLI for atom-level partial-charge agreement analysis.

Streams charge.lmdb per vertical, writes:
  - <vertical>_charge_atoms.parquet     (long; one row per (record, atom))
  - <vertical>_charge_summary.parquet   (per-scheme + pair stats with sufficient stats)

Layout follows analysis-dipole-alignment:
  --root contains charge.lmdb -> single-vertical mode.
  Otherwise -> corpus mode: one pair of parquets per immediate subdir <v>
  for which <v>/charge.lmdb exists.

Examples:
    analysis-charge-alignment --root data/OMol4M_lmdbs/5A_elytes \\
        --output_dir analysis_outputs/charge_alignment/
    analysis-charge-alignment --root data/OMol4M_lmdbs \\
        --output_dir analysis_outputs/charge_alignment/ \\
        --include_verticals 5A_elytes droplet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from qtaim_gen.source.analysis.charge_alignment import run_vertical
from qtaim_gen.source.scripts.helpers.pull_holdout_records import NON_VERTICAL_DIRS


def _resolve_vertical(path: Path) -> Path | None:
    if (path / "charge.lmdb").exists():
        return path
    return None


def _discover_verticals(root: Path) -> list[tuple[str, Path]]:
    direct = _resolve_vertical(root)
    if direct is not None:
        return [(root.name, direct)]
    out: list[tuple[str, Path]] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        if child.name in NON_VERTICAL_DIRS:
            continue
        v = _resolve_vertical(child)
        if v is not None:
            out.append((child.name, v))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Atom-level partial-charge agreement (paper section 6.2 / B1).",
    )
    parser.add_argument("--root", type=Path, required=True,
                        help="Vertical root or corpus root.")
    parser.add_argument("--output_dir", type=Path,
                        default=Path("analysis_outputs/charge_alignment"),
                        help="Directory to write per-vertical parquets into.")
    parser.add_argument("--include_verticals", nargs="*", default=None, metavar="V",
                        help="Restrict corpus mode to these vertical names.")
    parser.add_argument("--no_progress", action="store_true",
                        help="Disable tqdm progress bar.")
    args = parser.parse_args()

    if not args.root.exists():
        print(f"error: --root {args.root} does not exist", file=sys.stderr)
        return 2

    verticals = _discover_verticals(args.root)
    if args.include_verticals is not None:
        requested = set(args.include_verticals)
        unknown = sorted(requested - {n for n, _ in verticals})
        if unknown:
            print(f"warn: --include_verticals contains {len(unknown)} not under "
                  f"--root: {unknown}", file=sys.stderr)
        verticals = [(n, p) for n, p in verticals if n in requested]

    if not verticals:
        print(f"error: no verticals (with charge.lmdb) under {args.root}",
              file=sys.stderr)
        return 2

    for name, path in verticals:
        atoms_path, sum_path = run_vertical(
            root=path, output_dir=args.output_dir,
            vertical_name=name, progress=not args.no_progress,
        )
        print(f"{name}: {atoms_path.name}, {sum_path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
