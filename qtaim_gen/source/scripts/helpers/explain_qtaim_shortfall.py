"""explain-qtaim-shortfall: attribute missing bond CPs to a specific cause.

A shortfall between Multiwfn's reported (3,-1) count and the bond CPs stored in
qtaim.json is not automatically data loss. `merge_qtaim_inds` has three
legitimate drop paths, and only the first is a genuine loss:

  truncated            CPprop.txt holds fewer (3,-1) blocks than qtaim.out
                       reported -- critical points really were lost
  no_bond_path         a CP block carries no "Connected atoms:" line, so the
                       merge filters it (parse_qtaim.py, define_bonds="qtaim")
  duplicate_pair       two CPs resolve to the same atom pair; the merge keys a
                       plain dict on that pair, so the second silently
                       overwrites the first. This yields a shortfall of exactly
                       one per collision with nothing lost from the search
  unmatched_attractor  a CP's attractor has no nuclear-CP match, e.g. a
                       non-nuclear attractor; the merge skips it with a warning

Only `truncated` is repairable by rerunning. The other three are deterministic:
the same CPprop.txt yields the same qtaim.json, so a rerun reproduces them
exactly -- which is what an `unchanged_still_broken` verdict from
verify-qtaim-rerun looks like.

Needs CPprop.txt, which lives in the job folder or in generator/out_files.zip
(archived only by runs after the fix that stopped deleting it pre-zip).

Example:
    explain-qtaim-shortfall --verified_csv qtaim_rerun_test_verified.csv \
        --verdicts unchanged_still_broken changed_still_broken
"""

import argparse
import collections
import csv
import json
import os
import sys
import tempfile
import zipfile
from typing import List, Optional


def find_cpprop(folder: str) -> Optional[str]:
    """Path to a readable CPprop.txt, extracting from the zip if needed.

    Returns a path the caller should treat as read-only; when extracted from the
    archive it lands in a temp dir the caller need not clean up eagerly.
    """
    for rel in ("CPprop.txt", os.path.join("generator", "CPprop.txt")):
        path = os.path.join(folder, rel)
        if os.path.isfile(path) and os.path.getsize(path) > 0:
            return path
    zip_path = os.path.join(folder, "generator", "out_files.zip")
    if os.path.isfile(zip_path):
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                if "CPprop.txt" in zf.namelist():
                    tmp = tempfile.mkdtemp(prefix="cpprop_")
                    zf.extract("CPprop.txt", tmp)
                    return os.path.join(tmp, "CPprop.txt")
        except (zipfile.BadZipFile, OSError, KeyError):
            pass
    return None


def explain(folder: str) -> dict:
    """Attribute a folder's bond-CP shortfall to the causes above."""
    from qtaim_gen.source.core.parse_qtaim import get_qtaim_descs, only_atom_cps
    from qtaim_gen.source.utils.validation import qtaim_run_status

    out = {"folder": folder, "cause": "", "detail": ""}

    qpath = None
    for base in (folder, os.path.join(folder, "generator")):
        cand = os.path.join(base, "qtaim.json")
        if os.path.isfile(cand):
            qpath = cand
            break
    if qpath is None:
        out["cause"] = "no_qtaim_json"
        return out
    with open(qpath) as f:
        stored = json.load(f)
    n_stored = sum(1 for k in stored if k != "_meta" and "_" in k)
    out["n_bcp_stored"] = n_stored

    status = qtaim_run_status(folder)
    out["reported_bcp"] = status["reported_bcp"]
    out["export_done"] = status["export_done"]

    cpprop = find_cpprop(folder)
    if cpprop is None:
        out["cause"] = "no_cpprop"
        out["detail"] = "CPprop.txt not retained; cannot attribute"
        return out

    descs = get_qtaim_descs(cpprop)
    _atoms, bonds = only_atom_cps(descs)
    out["n_bcp_blocks"] = len(bonds)

    with_paths = {k: v for k, v in bonds.items() if v.get("connected_bond_paths")}
    out["n_no_bond_path"] = len(bonds) - len(with_paths)

    pairs = [tuple(sorted(v["connected_bond_paths"])) for v in with_paths.values()]
    counts = collections.Counter(pairs)
    dups = {p: c for p, c in counts.items() if c > 1}
    out["n_duplicate_pairs"] = sum(c - 1 for c in dups.values())
    out["duplicate_pairs"] = " ".join(f"{a}-{b}" for a, b in list(dups)[:6])
    out["n_unique_pairs"] = len(counts)

    reported = status["reported_bcp"]
    if reported is not None and len(bonds) < reported:
        out["cause"] = "truncated"
        out["detail"] = (
            f"CPprop.txt holds {len(bonds)} (3,-1) blocks vs {reported} reported"
        )
        return out

    # CPprop.txt is complete; the shortfall came from the merge
    if out["n_duplicate_pairs"] and out["n_no_bond_path"]:
        out["cause"] = "duplicate_pair+no_bond_path"
    elif out["n_duplicate_pairs"]:
        out["cause"] = "duplicate_pair"
    elif out["n_no_bond_path"]:
        out["cause"] = "no_bond_path"
    elif n_stored < len(bonds):
        out["cause"] = "unmatched_attractor"
    else:
        out["cause"] = "explained_none"
    out["detail"] = (
        f"{len(bonds)} blocks -> {out['n_unique_pairs']} unique pairs -> "
        f"{n_stored} stored"
    )
    return out


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--verified_csv", help="output of verify-qtaim-rerun")
    src.add_argument("--folders", nargs="+", help="job folders to explain directly")
    parser.add_argument(
        "--verdicts",
        nargs="+",
        default=["unchanged_still_broken", "changed_still_broken"],
        help="which verify-qtaim-rerun verdicts to explain",
    )
    parser.add_argument("--out_csv", default=None)
    args = parser.parse_args(argv)

    if args.folders:
        folders = args.folders
    else:
        with open(args.verified_csv) as f:
            folders = [
                r["folder"] for r in csv.DictReader(f) if r.get("verdict") in args.verdicts
            ]
    if not folders:
        print("no folders matched", file=sys.stderr)
        return 2
    print(f"explaining {len(folders)} folders\n")

    rows = [explain(f) for f in folders]
    causes = collections.Counter(r["cause"] for r in rows)

    fields = [
        "cause", "detail", "n_bcp_stored", "reported_bcp", "n_bcp_blocks",
        "n_unique_pairs", "n_no_bond_path", "n_duplicate_pairs", "duplicate_pairs",
        "export_done", "folder",
    ]
    out_csv = args.out_csv or "qtaim_shortfall_causes.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    for cause, n in causes.most_common():
        print(f"  {cause:<28}{n:>5}")
    repairable = causes["truncated"]
    deterministic = sum(
        n for c, n in causes.items()
        if c in ("duplicate_pair", "no_bond_path", "duplicate_pair+no_bond_path",
                 "unmatched_attractor")
    )
    print(f"\n  repairable by rerunning (truncated): {repairable}")
    print(f"  deterministic merge behaviour:        {deterministic}")
    if causes["no_cpprop"]:
        print(
            f"  unattributable (no CPprop.txt):       {causes['no_cpprop']}"
            "  <- pre-dates the archiving fix"
        )
    if deterministic and not repairable:
        print(
            "\n  These are not lost data: Multiwfn found the critical points and the\n"
            "  merge collapsed or filtered them. Rerunning cannot change that, so the\n"
            "  fix belongs in the merge/schema, not in more compute."
        )
    print(f"\n  -> {out_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
