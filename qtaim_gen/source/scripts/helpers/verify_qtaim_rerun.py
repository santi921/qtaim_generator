"""verify-qtaim-rerun: did the rerun fix the defects without disturbing anything?

Re-audits the folders named in a select-qtaim-rerun before-state CSV and reports
each job's transition. Answers three questions that a plain re-run of
refine_list_of_jobs cannot:

1. did each defect actually get FIXED, or is it merely broken differently now?
2. did anything become NEWLY broken?
3. were the CONTROLS left alone? A control whose bond-CP set moved means the
   rerun is not a repair, it is a perturbation -- and applying it dataset-wide
   would make repaired records inconsistent with untouched ones. That is the
   finding that would otherwise go unnoticed.

It also reports whether charge/bond/fuzzy/other/orca json mtimes advanced, but
only as a note: move_results_to_folder rewrites those on every run, so the
signal fires even when the step was skipped and the content is identical. The
unchanged controls are the real evidence that nothing was disturbed.

Example:
    verify-qtaim-rerun --before_csv qtaim_rerun_test_before.csv \
        --out_csv qtaim_rerun_test_verified.csv
"""

import argparse
import collections
import csv
import os
import sys
from typing import List, Optional

from qtaim_gen.source.utils.validation import DEFAULT_BCP_TOLERANCE

SIBLING_JSONS = ("charge.json", "bond.json", "fuzzy_full.json", "other.json", "orca.json")


def as_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def newer_siblings(folder: str) -> List[str]:
    """Sibling JSONs whose mtime advanced with qtaim.json's.

    WEAK SIGNAL, and it over-reports: move_results_to_folder re-merges these
    files into generator/ on every run, so their mtimes advance even when the
    step was skipped and the content is byte-identical. Observed firing on
    98/100 jobs in a run whose logs showed every charge step skipped as "data
    verified". Read it as "the runner rewrote the file", not "the values
    changed"; a content diff needs hashes captured before the rerun.
    """
    qtaim_mtime = None
    for base in (folder, os.path.join(folder, "generator")):
        p = os.path.join(base, "qtaim.json")
        if os.path.isfile(p):
            qtaim_mtime = os.path.getmtime(p)
            break
    if qtaim_mtime is None:
        return []
    touched = []
    for name in SIBLING_JSONS:
        for base in (folder, os.path.join(folder, "generator")):
            p = os.path.join(base, name)
            if os.path.isfile(p) and os.path.getmtime(p) >= qtaim_mtime - 1.0:
                touched.append(name)
                break
    return touched


def classify_state(row: dict, bcp_tolerance: int = DEFAULT_BCP_TOLERANCE) -> str:
    """'ok' or the reason the record is still considered defective.

    Uses the runner's tolerance so a job left one unmappable CP short reads as
    repaired rather than as a permanent failure.
    """
    if row.get("error"):
        return "error"
    if row.get("have_qtaim_json") is False:
        return "no_qtaim_json"
    if row.get("search_done") is False or row.get("export_done") is False:
        return "incomplete_run"
    # No qtaim.out means the rerun left nothing to check the CP count against,
    # so it cannot be called fixed -- the shortfall column is null, not zero.
    if row.get("have_qtaim_out") is False:
        return "no_provenance"
    n_bcp = as_int(row.get("n_bcp"))
    if n_bcp == 0 and as_int(row.get("n_cov_bonds")) > 0:
        return "empty_bcp"
    if as_int(row.get("bcp_shortfall")) > bcp_tolerance:
        return "shortfall"
    return "ok"


def main(argv: Optional[List[str]] = None) -> int:
    from qtaim_gen.source.scripts.helpers.audit_qtaim_connectivity import audit_folder

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before_csv", required=True)
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--covalent_factor", type=float, default=1.3)
    parser.add_argument(
        "--bcp_tolerance",
        type=int,
        default=DEFAULT_BCP_TOLERANCE,
        help="match the runner's --bcp_tolerance so a tolerated shortfall "
        "counts as fixed",
    )
    args = parser.parse_args(argv)

    with open(args.before_csv) as f:
        before = list(csv.DictReader(f))
    if not before:
        print("before CSV is empty", file=sys.stderr)
        return 2

    rows, verdicts = [], collections.Counter()
    for b in before:
        folder = b["folder"]
        mode = b.get("_mode", "")
        was_control = mode == "control"
        rec = {
            "folder": folder,
            "key": b.get("key", ""),
            "mode_before": mode,
            "n_bcp_before": as_int(b.get("n_bcp")),
            "reported_before": b.get("reported_bcp", ""),
            "shortfall_before": b.get("bcp_shortfall", ""),
        }
        try:
            after = audit_folder(folder, args.covalent_factor)
        except Exception as e:
            rec["verdict"] = "audit_failed"
            rec["detail"] = f"{type(e).__name__}: {e}"[:120]
            verdicts["audit_failed"] += 1
            rows.append(rec)
            continue

        rec.update(
            {
                "n_bcp_after": after.get("n_bcp"),
                "reported_after": after.get("reported_bcp"),
                "shortfall_after": after.get("bcp_shortfall"),
                "export_done_after": after.get("export_done"),
                "state_after": classify_state(after, args.bcp_tolerance),
                "siblings_touched": " ".join(newer_siblings(folder)),
            }
        )

        changed = rec["n_bcp_after"] != rec["n_bcp_before"]
        if was_control:
            verdict = "control_unchanged" if not changed else "CONTROL_PERTURBED"
        elif rec["state_after"] == "ok":
            verdict = "fixed"
        elif changed:
            verdict = "changed_still_broken"
        else:
            verdict = "unchanged_still_broken"
        rec["verdict"] = verdict
        verdicts[verdict] += 1
        rows.append(rec)

    fields = [
        "verdict", "mode_before", "key", "n_bcp_before", "n_bcp_after",
        "reported_before", "reported_after", "shortfall_before", "shortfall_after",
        "export_done_after", "state_after", "siblings_touched", "detail", "folder",
    ]
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    print(f"{len(rows)} jobs verified -> {args.out_csv}\n")
    for verdict, n in verdicts.most_common():
        print(f"  {verdict:<24}{n:>5}")

    defects = [r for r in rows if r["mode_before"] != "control"]
    fixed = [r for r in defects if r["verdict"] == "fixed"]
    if defects:
        print(
            f"\n  repair rate: {len(fixed)}/{len(defects)} "
            f"({100 * len(fixed) / len(defects):.1f}%) of defects now pass"
        )

    perturbed = [r for r in rows if r["verdict"] == "CONTROL_PERTURBED"]
    touched = [r for r in rows if r.get("siblings_touched")]
    problems = []
    if perturbed:
        problems.append(
            f"{len(perturbed)} CONTROL(S) PERTURBED -- the rerun changed records that "
            "were already complete, so it is not a safe dataset-wide repair"
        )
        for r in perturbed[:5]:
            print(
                f"    perturbed: {r['key']} bcp {r['n_bcp_before']} -> {r['n_bcp_after']}"
            )
    if touched:
        # deliberately not a hard problem: see newer_siblings' docstring
        print(
            f"\n  note: {len(touched)} job(s) had non-QTAIM json mtimes advance. "
            "move_results_to_folder\n  rewrites those every run, so this is "
            "expected and does not by itself mean the\n  values changed -- "
            "unchanged controls are the stronger evidence of that."
        )
    still = [r for r in defects if r["verdict"].endswith("still_broken")]
    if still:
        problems.append(
            f"{len(still)} defect(s) still broken; if unchanged_still_broken "
            "dominates, the loss is deterministic and rerunning cannot fix it"
        )

    if problems:
        print("\n  PROBLEMS:")
        for p in problems:
            print(f"    - {p}")
        return 1
    print("\n  All defects repaired, controls untouched, only QTAIM moved.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
