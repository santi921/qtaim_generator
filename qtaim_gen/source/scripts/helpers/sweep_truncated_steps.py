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

With --recheck_fuzzy (see qtaim_gen.source.utils.fuzzy_recheck), also:

  reparse_only           wrong fuzzy values fixable from archived output, no rerun
  no_wavefunction_source open-shell rerun needed but only a .wfn exists

With --check_orca, a folder whose orca.json is missing, malformed or older
than the current parser (orca_parser_version) is not complete, and every
record carries "orca_stale". When that is the only problem:

  orca_reparse           runner takes the orca-only path (no Multiwfn)
  orca_no_source         no orca.out / orca.tar.zst in the results or inputs
                         folder, so the reparse cannot happen

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
    _gbw_source_present,
    _has_ecp_atoms,
    _has_usable_step_output,
    _is_substantive_step_out,
    _wavefunction_present,
)
from qtaim_gen.source.utils.fuzzy_recheck import recheck_fuzzy as _recheck_fuzzy
from qtaim_gen.source.core.parse_orca import ORCA_PARSER_VERSION, find_orca_output_file
from qtaim_gen.source.data.multiwfn import (
    bond_order_dict,
    charge_data_dict,
    fuzzy_data,
    other_data_dict,
)
from qtaim_gen.source.utils.validation import (
    get_charge_spin_n_atoms_from_folder,
    validate_orca_dict,
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


def orca_stale(folder: str, move_results: bool) -> bool:
    """What validation_checks(check_orca=True) rejects on orca.json alone."""
    base = os.path.join(folder, "generator") if move_results else folder
    path = os.path.join(base, "orca.json")
    if not os.path.isfile(path):
        return True
    return not validate_orca_dict(path, min_parser_version=ORCA_PARSER_VERSION)


def orca_source_present(folder: str, folder_inputs: str) -> bool:
    """orca.out or orca.tar.zst in the results folder, or in the inputs folder
    (the ALCF runner copies it in at run time)."""
    for base in {folder, folder_inputs}:
        if os.path.isdir(base) and (
            find_orca_output_file(base) or os.path.isfile(os.path.join(base, "orca.tar.zst"))
        ):
            return True
    return False


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
    recheck_fuzzy: bool = False,
    preprocess_compressed: bool = False,
    check_orca: bool = False,
) -> dict:
    folder = resolve_results_folder(folder_inputs, root_omol_inputs, root_omol_results)
    rec = {"folder": folder_inputs, "results_folder": folder}

    if not os.path.isdir(folder) or not os.path.exists(
        os.path.join(folder, "gbw_analysis.log")
    ):
        rec["class"] = "no_outputs"
        return rec

    n_atoms = None
    charge = None
    spin_tf = False
    mult = None
    try:
        dft_dict = get_charge_spin_n_atoms_from_folder(folder)
        if dft_dict and dft_dict.get("mol"):
            n_atoms = len(dft_dict["mol"])
            spin_tf = dft_dict.get("spin", 1) != 1
            mult = dft_dict.get("spin")
            if dft_dict.get("charge") is not None and not _has_ecp_atoms(dft_dict):
                charge = int(dft_dict["charge"])
    except Exception:
        pass
    rec["n_atoms"] = n_atoms

    # what gbw_analysis(recheck_fuzzy=True) would reparse / invalidate
    recheck = None
    if recheck_fuzzy and mult is not None:
        recheck = _recheck_fuzzy(
            folder, int(mult), dry_run=True, preprocess_compressed=preprocess_compressed
        )
        # the ALCF runner copies the gbw source in from the inputs tree at run time
        if (
            not recheck["ok"]
            and os.path.abspath(folder_inputs) != os.path.abspath(folder)
            and os.path.isdir(folder_inputs)
            and _gbw_source_present(folder_inputs, preprocess_compressed)
        ):
            recheck["ok"] = True
            recheck["wavefunction"] = "gbw source in the inputs folder"
        rec["recheck"] = recheck

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

    stale = check_orca and orca_stale(folder, move_results)
    if check_orca:
        rec["orca_stale"] = stale

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
        if not recheck or not (recheck["reparse"] or recheck["rerun"]):
            if not stale:
                rec["class"] = "complete"
            elif orca_source_present(folder, folder_inputs):
                rec["class"] = "orca_reparse"
            else:
                rec["class"] = "orca_no_source"
        elif not recheck["ok"]:
            rec["class"] = "no_wavefunction_source"
        elif not recheck["rerun"]:
            rec["class"] = "reparse_only"
        else:
            rec["rerun_steps"] = list(recheck["rerun"])
            rec["wavefunction_present"] = _wavefunction_present(folder)
            rec["class"] = "needs_rerun"
        return rec

    order, compiled_map, fuzzy_routines = routine_sets(full_set, spin_tf)
    rerun_steps: List[str] = []
    truncated_steps: List[str] = []
    for op in order:
        # same operand order as run_jobs: cheap compiled-JSON check first
        will_skip = _compiled_data_present(
            folder,
            op,
            compiled_map,
            n_atoms=n_atoms,
            fuzzy_routines=fuzzy_routines,
            charge=charge,
        ) or _has_usable_step_output(
            folder, op, n_atoms=n_atoms, charge=charge, fuzzy_routines=fuzzy_routines
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

    if recheck:
        rerun_steps += [s for s in recheck["rerun"] if s not in rerun_steps]
    rec["rerun_steps"] = rerun_steps
    rec["truncated_steps"] = truncated_steps
    rec["wavefunction_present"] = _wavefunction_present(folder)
    if recheck and not recheck["ok"]:
        rec["class"] = "no_wavefunction_source"
    else:
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
    parser.add_argument("--recheck_fuzzy", action="store_true",
                        help="also classify physically wrong fuzzy integrations / open-shell "
                             "fuzzy bonds as gbw_analysis(recheck_fuzzy=True) would")
    parser.add_argument("--preprocess_compressed", action="store_true",
                        help="with --recheck_fuzzy: count .gbw.zstd0 as a wavefunction source, "
                             "as the runner does under --preprocess_compressed")
    parser.add_argument("--check_orca", action="store_true",
                        help="require a current orca.json (orca_parser_version), as the "
                             "runner does under --check_orca")
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
                args.recheck_fuzzy,
                args.preprocess_compressed,
                args.check_orca,
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
    for cls in ("complete", "orca_reparse", "orca_no_source", "reparse_only", "needs_rerun", "validation_loop",
                "no_wavefunction_source", "no_outputs", "error"):
        if class_counts.get(cls):
            print(f"  {cls:16s} {class_counts[cls]}")
    if args.check_orca:
        n_stale = sum(1 for rec in records if rec.get("orca_stale"))
        print(f"\norca.json stale (any class, reparsed on the runner pass): {n_stale}")
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
        # no_wavefunction_source folders are always refused; requeueing them wastes a slot
        requeue = [r["folder"] for r in records
                   if r["class"] not in ("complete", "no_wavefunction_source", "orca_no_source")]
        with open(args.requeue_file, "w") as f:
            for folder in requeue:
                f.write(folder + "\n")
        print(f"Requeue list ({len(requeue)} folders): {args.requeue_file}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
