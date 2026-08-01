"""select-qtaim-rerun: turn audit CSVs into a rerun job list and a pull manifest.

Bridges audit-qtaim-connectivity to the runner. Two path trees are involved and
they are not the same:

- the **results** tree (--root_omol_results, e.g. OMol25_postprocessing/) holds
  qtaim.json and is what the audit walked, so it is what the CSV records;
- the **inputs** tree (--root_omol_inputs, e.g. OMol25/) holds
  orca.gbw.zstd0 / orca.tar.zst and is what --job_file must contain, because
  the runner maps inputs -> results itself.

Outputs:
  <out_prefix>_jobs.txt      input-tree paths, one per line -> runner --job_file
  <out_prefix>_pull.tsv      both paths per job, for staging to another machine
  <out_prefix>_before.csv    the pre-rerun state, so a rerun can be verified
                             rather than assumed to have worked

A stratified selection is the default: reruns are cheap to get wrong quietly, so
the sample deliberately spans failure modes *and* includes known-good controls.
If a control's qtaim.json changes after a rerun, the remedy is unsafe and that
matters more than whether the defects got fixed.
"""

import argparse
import collections
import csv
import os
import random
import sys
from typing import List, Optional

from qtaim_gen.source.utils.validation import (
    DEFAULT_BCP_TOLERANCE,
    as_tristate,
)


def classify(row: dict, bcp_tolerance: int = DEFAULT_BCP_TOLERANCE) -> str:
    """Failure mode, or 'control' for a record with no detected defect.

    bcp_tolerance must match what the runner and refine_list_of_jobs use, or
    this keeps selecting jobs those two consider acceptable -- they would rerun,
    come back identical, and be selected again on the next pass.
    """

    def as_int(key):
        try:
            return int(row.get(key) or 0)
        except ValueError:
            return 0

    # Absence first: these are not "clean", they are unexamined. A folder with
    # no qtaim.json never ran QTAIM (or lost the output); one with no qtaim.out
    # may hold a silently truncated CP set that nothing on disk can rule out.
    # Both have to rerun for the dataset to be uniform, so they are selectable
    # modes rather than controls -- which is what they used to fall through to,
    # because search_done/export_done are empty (not "False") without qtaim.out.
    if as_tristate(row.get("have_qtaim_json")) is False:
        return "no_qtaim_json"
    if row.get("error"):
        return "error"
    if (
        as_tristate(row.get("search_done")) is False
        or as_tristate(row.get("export_done")) is False
    ):
        return "incomplete_run"
    if as_tristate(row.get("have_qtaim_out")) is False:
        return "no_provenance"
    if as_int("n_bcp") == 0 and as_int("n_cov_bonds") > 0:
        return "empty_bcp"
    if as_int("bcp_shortfall") > bcp_tolerance:
        return "shortfall"
    n_atoms = as_int("n_atoms")
    if n_atoms and as_int("n_isolated_bonded") / n_atoms > 0.10:
        return "severe_isolated"
    if as_int("n_isolated_bonded") > 0:
        return "single_isolated"
    return "control"


def size_bin(row: dict) -> str:
    try:
        n = int(row.get("n_atoms") or 0)
    except ValueError:
        return "unknown"
    for lo, hi in ((0, 50), (50, 100), (100, 200), (200, 400)):
        if lo <= n < hi:
            return f"{lo}-{hi}"
    return "400+"


def results_to_inputs(folder: str, root_results: str, root_inputs: str) -> Optional[str]:
    folder = os.path.normpath(folder)
    root_results = os.path.normpath(root_results)
    if not folder.startswith(root_results):
        return None
    rel = os.path.relpath(folder, root_results)
    return os.path.join(root_inputs, rel)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit_csv", nargs="+", required=True)
    parser.add_argument(
        "--root_omol_results",
        required=True,
        help="prefix the audit walked (results tree, holds qtaim.json)",
    )
    parser.add_argument(
        "--root_omol_inputs",
        required=True,
        help="matching inputs tree (holds orca.gbw.zstd0); what --job_file needs",
    )
    parser.add_argument("--out_prefix", required=True)
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=100,
        help="total jobs to select (0 = all defects). Not --n: conda run treats "
        "that as an ambiguous abbreviation of its own --name.",
    )
    parser.add_argument(
        "--n_controls",
        type=int,
        default=10,
        help="known-good jobs to include, to prove a rerun does not disturb them",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--bcp_tolerance",
        type=int,
        default=DEFAULT_BCP_TOLERANCE,
        help=(
            "bond CPs allowed to be missing before 'shortfall' is selectable "
            f"(default {DEFAULT_BCP_TOLERANCE}). Keep this equal to the runner's "
            "--bcp_tolerance; a smaller value here selects jobs the runner will "
            "accept unchanged, and they never leave the queue."
        ),
    )
    parser.add_argument(
        "--modes",
        default=(
            "shortfall,incomplete_run,empty_bcp,severe_isolated,"
            "no_qtaim_json,no_provenance"
        ),
        help=(
            "failure modes eligible for selection. no_qtaim_json and "
            "no_provenance are the absence modes: nothing on disk proves those "
            "records complete, so they rerun for uniformity. They dominate the "
            "count on verticals that were cleaned, so drop them from this list "
            "to target proven defects only."
        ),
    )
    args = parser.parse_args(argv)

    rows = []
    for path in args.audit_csv:
        with open(path) as f:
            for row in csv.DictReader(f):
                if not row.get("folder"):
                    print(
                        f"{path}: no 'folder' column -- this needs a folder-mode "
                        "audit CSV, not an LMDB-mode one",
                        file=sys.stderr,
                    )
                    return 2
                row["_mode"] = classify(row, args.bcp_tolerance)
                row["_bin"] = size_bin(row)
                row["_source"] = os.path.basename(path)
                rows.append(row)
    if not rows:
        print("no rows in audit CSVs", file=sys.stderr)
        return 2

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    defects = [r for r in rows if r["_mode"] in modes]
    controls = [r for r in rows if r["_mode"] == "control"]

    print(f"{len(rows)} audited rows; mode breakdown:")
    for mode, n in collections.Counter(r["_mode"] for r in rows).most_common():
        print(f"  {mode:<18}{n:>7}")

    rng = random.Random(args.seed)
    n_defect = max(args.n_jobs - args.n_controls, 0) if args.n_jobs else len(defects)

    # spread across mode x size bin so a single easy stratum cannot dominate and
    # make the test look better than it is
    strata = collections.defaultdict(list)
    for r in defects:
        strata[(r["_mode"], r["_bin"])].append(r)
    for v in strata.values():
        rng.shuffle(v)

    picked, keys = [], sorted(strata)
    while len(picked) < n_defect and any(strata[k] for k in keys):
        for k in keys:
            if strata[k] and len(picked) < n_defect:
                picked.append(strata[k].pop())

    rng.shuffle(controls)
    picked_controls = controls[: args.n_controls]
    selection = picked + picked_controls

    unmapped = 0
    jobs_path = f"{args.out_prefix}_jobs.txt"
    pull_path = f"{args.out_prefix}_pull.tsv"
    before_path = f"{args.out_prefix}_before.csv"
    with open(jobs_path, "w") as jf, open(pull_path, "w") as pf:
        pf.write("mode\tsize_bin\tkey\tinputs_path\tresults_path\n")
        for r in selection:
            inp = results_to_inputs(
                r["folder"], args.root_omol_results, args.root_omol_inputs
            )
            if inp is None:
                unmapped += 1
                continue
            jf.write(inp + "\n")
            pf.write(
                f"{r['_mode']}\t{r['_bin']}\t{r.get('key','')}\t{inp}\t{r['folder']}\n"
            )

    fields = [
        "_mode", "_bin", "_source", "vertical", "key", "folder", "n_atoms", "n_ncp",
        "n_bcp", "reported_bcp", "bcp_shortfall", "have_qtaim_json",
        "have_qtaim_out", "search_done", "export_done",
        "n_cov_bonds", "n_isolated_bonded", "n_components", "qtaim_time_s",
    ]
    with open(before_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(selection)

    print(f"\nselected {len(selection)} jobs "
          f"({len(picked)} defects + {len(picked_controls)} controls)")
    for mode, n in collections.Counter(r["_mode"] for r in selection).most_common():
        print(f"  {mode:<18}{n:>5}")
    print("  by size bin:", dict(collections.Counter(r["_bin"] for r in selection)))
    if unmapped:
        print(
            f"\n  !! {unmapped} rows did not sit under --root_omol_results and were "
            "skipped; check the prefix matches the audit's --folder_root"
        )
    print(f"\n  {jobs_path}    -> runner --job_file (input-tree paths)")
    print(f"  {pull_path}    -> both paths per job, for staging")
    print(f"  {before_path}  -> pre-rerun state; diff against a re-audit to verify")
    print(
        "\nVerify a rerun with, at minimum: bcp_shortfall -> 0 and export_done -> True\n"
        "on the defects, AND every control's qtaim.json unchanged. Also confirm\n"
        "charge/bond/fuzzy/other/orca json are byte-identical, since only the QTAIM\n"
        "step is supposed to move."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
