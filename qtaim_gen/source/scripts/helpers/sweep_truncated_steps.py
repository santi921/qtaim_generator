#!/usr/bin/env python3
"""Dry-run of the per-sub-job restart skip logic over a job list.

For each folder, replays the exact skip decisions gbw_analysis(restart=True)
would make (_has_usable_step_output / _compiled_data_present) plus the
folder-level validation gate, and classifies:

  complete         validation passes; nothing to do
  needs_rerun      validation fails and >=1 step would re-run
                   (self-heals on the next --restart pass)
  validation_loop  validation fails but every step would be skipped --
                   the folder would requeue forever; needs attention
  no_outputs       results folder missing or never started (no gbw_analysis.log)
  error            classification raised an exception

Writes one JSON line per folder to --report_file, prints a summary, and
optionally writes non-complete folders to --requeue_file.

Example (OMol4M):
    python -m qtaim_gen.source.scripts.helpers.sweep_truncated_steps \
        --job_file job_lists/ml_elytes_refined.txt \
        --root_omol_inputs /p/lustre5/bennion1/Omol2025-4M-DiversitySet/ \
        --root_omol_results /p/lustre5/vargas58/OMol4M/ \
        --full_set 1 --n_workers 32 \
        --report_file sweep_ml_elytes.jsonl
"""
import os
import json
import argparse
import concurrent.futures
from collections import Counter
from typing import Optional, List

from tqdm import tqdm

from qtaim_gen.source.core.omol import (
    _compiled_data_present,
    _has_usable_step_output,
    _is_substantive_step_out,
    _wavefunction_present,
)
from qtaim_gen.source.data.multiwfn import (
    bond_order_dict,
    charge_data_dict,
    fuzzy_data,
    other_data_dict,
)
from qtaim_gen.source.utils.validation import (
    get_charge_spin_n_atoms_from_folder,
    validation_checks,
)


def resolve_results_folder(
    folder_inputs: str,
    root_omol_inputs: Optional[str],
    root_omol_results: Optional[str],
) -> str:
    """Map an input folder path to its results location (mirrors get_folders_from_file)."""
    if (
        root_omol_inputs
        and root_omol_results
        and folder_inputs.startswith(root_omol_inputs)
    ):
        rel = folder_inputs[len(root_omol_inputs):].lstrip(os.sep)
        return os.path.join(root_omol_results, rel)
    return folder_inputs


def routine_sets(full_set: int, spin_tf: bool):
    """Routine order + compiled_map exactly as run_jobs builds them."""
    charge = charge_data_dict(full_set)
    bond = bond_order_dict(full_set)
    fuzzy = fuzzy_data(spin=spin_tf, full_set=full_set)
    other = other_data_dict(full_set)
    compiled_map = {}
    for op in charge:
        compiled_map[op] = ("charge.json", op, "charge")
    for op in bond:
        compiled_map[op] = ("bond.json", op)
    for op in fuzzy:
        compiled_map[op] = ("fuzzy_full.json", op)
    for op in other:
        compiled_map[op] = ("other.json", None)
    order = list(charge) + list(bond) + list(fuzzy) + list(other) + ["qtaim"]
    return order, compiled_map, set(fuzzy)


def classify_folder(
    folder_inputs: str,
    root_omol_inputs: Optional[str],
    root_omol_results: Optional[str],
    full_set: int,
    move_results: bool,
) -> dict:
    folder = resolve_results_folder(folder_inputs, root_omol_inputs, root_omol_results)
    rec = {"folder": folder_inputs, "results_folder": folder}

    if not os.path.isdir(folder) or not os.path.exists(
        os.path.join(folder, "gbw_analysis.log")
    ):
        rec["class"] = "no_outputs"
        return rec

    n_atoms = None
    spin_tf = False
    try:
        dft_dict = get_charge_spin_n_atoms_from_folder(folder)
        if dft_dict and dft_dict.get("mol"):
            n_atoms = len(dft_dict["mol"])
            spin_tf = dft_dict.get("spin", 1) != 1
    except Exception:
        pass
    rec["n_atoms"] = n_atoms

    # workload done so far (restart resumes from here)
    for base in (os.path.join(folder, "generator"), folder):
        timings_path = os.path.join(base, "timings.json")
        try:
            with open(timings_path, "r") as f:
                timings = json.load(f)
            rec["timings_sum_s"] = round(
                sum(v for v in timings.values() if isinstance(v, (int, float)) and v > 0),
                1,
            )
            break
        except (OSError, json.JSONDecodeError, AttributeError):
            continue

    try:
        valid = bool(
            validation_checks(
                folder,
                full_set=full_set,
                move_results=move_results,
                verbose=False,
                logger=None,
            )
        )
    except Exception:
        valid = False
    if valid:
        rec["class"] = "complete"
        return rec

    order, compiled_map, fuzzy_routines = routine_sets(full_set, spin_tf)
    rerun_steps: List[str] = []
    truncated_steps: List[str] = []
    for op in order:
        will_skip = _has_usable_step_output(folder, op) or _compiled_data_present(
            folder,
            op,
            compiled_map,
            n_atoms=n_atoms,
            fuzzy_routines=fuzzy_routines,
        )
        if will_skip:
            continue
        rerun_steps.append(op)
        for base in (folder, os.path.join(folder, "generator")):
            out_path = os.path.join(base, f"{op}.out")
            if os.path.isfile(out_path) and not _is_substantive_step_out(
                out_path, order=op
            ):
                truncated_steps.append(op)
                break

    rec["rerun_steps"] = rerun_steps
    rec["truncated_steps"] = truncated_steps
    rec["wavefunction_present"] = _wavefunction_present(folder)
    rec["class"] = "needs_rerun" if rerun_steps else "validation_loop"
    return rec


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="sweep-truncated-steps",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--job_file", type=str, required=True,
                        help="file with one job folder path per line")
    parser.add_argument("--num_folders", type=int, default=-1,
                        help="max folders to sweep (-1 = all)")
    parser.add_argument("--full_set", type=int, default=0,
                        help="calculation detail level (0/1/2), must match the run")
    parser.add_argument("--root_omol_inputs", type=str, default=None,
                        help="input root to strip when mapping to results")
    parser.add_argument("--root_omol_results", type=str, default=None,
                        help="results root where job outputs live")
    parser.add_argument("--move_results", action="store_true",
                        help="validate against generator/ subfolder (post-move layout)")
    parser.add_argument("--n_workers", type=int, default=8,
                        help="parallel workers (I/O bound; default 8)")
    parser.add_argument("--report_file", type=str, default="sweep_report.jsonl",
                        help="JSONL output, one record per folder")
    parser.add_argument("--requeue_file", type=str, default=None,
                        help="if set, write non-complete folder paths here")
    args = parser.parse_args(argv)

    with open(args.job_file, "r") as f:
        folders = [
            line.strip() for line in f
            if line.strip() and not line.strip().startswith("#")
        ]
    if args.num_folders > 0:
        folders = folders[: args.num_folders]
    if not folders:
        print(f"No folders found in {args.job_file}")
        return 2

    records = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.n_workers) as ex, \
            open(args.report_file, "w") as report:
        futures = {
            ex.submit(
                classify_folder,
                folder,
                args.root_omol_inputs,
                args.root_omol_results,
                args.full_set,
                args.move_results,
            ): folder
            for folder in folders
        }
        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures), desc="Sweeping folders", unit="folder",
        ):
            try:
                rec = future.result()
            except Exception as e:
                rec = {
                    "folder": futures[future],
                    "class": "error",
                    "error": f"{type(e).__name__}: {e}",
                }
            records.append(rec)
            report.write(json.dumps(rec) + "\n")

    class_counts = Counter(rec["class"] for rec in records)
    rerun_counts = Counter(
        step for rec in records for step in rec.get("rerun_steps", [])
    )
    truncated_counts = Counter(
        step for rec in records for step in rec.get("truncated_steps", [])
    )

    print("\nFolder classes:")
    for cls in ("complete", "needs_rerun", "validation_loop", "no_outputs", "error"):
        if class_counts.get(cls):
            print(f"  {cls:16s} {class_counts[cls]}")
    if rerun_counts:
        print("\nSteps needing rerun (folder counts):")
        for step, n in rerun_counts.most_common():
            trunc = truncated_counts.get(step, 0)
            print(f"  {step:22s} {n:6d}  (truncated .out: {trunc})")
    big = sorted(
        (r for r in records if r["class"] == "needs_rerun" and r.get("n_atoms")),
        key=lambda r: r["n_atoms"], reverse=True,
    )[:5]
    if big:
        print("\nLargest needs_rerun folders (n_atoms):")
        for r in big:
            print(f"  {r['n_atoms']:5d} atoms  {r['folder']}")
    print(f"\nReport: {args.report_file}")

    if args.requeue_file:
        requeue = [r["folder"] for r in records if r["class"] != "complete"]
        with open(args.requeue_file, "w") as f:
            for folder in requeue:
                f.write(folder + "\n")
        print(f"Requeue list ({len(requeue)} folders): {args.requeue_file}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
