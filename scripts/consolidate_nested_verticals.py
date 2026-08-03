#!/usr/bin/env python3
"""Decide which copy of a duplicated job folder to keep, for verticals whose
tree nests a directory of its own name.

Some verticals under the OMol4M results root contain a second copy of
themselves one level deeper:

    OMol4M/tm_react/MR_123_0_1/            <- top level
    OMol4M/tm_react/tm_react/MR_123_0_1/   <- nested copy, same key

Two very different situations produce that, and they need opposite remedies:

- **stale duplicate** (observed in rgd_uks): every nested key also exists at the
  top level and the top copy is never worse. The nested tree is discardable.
- **older, larger body of work with a partial re-run on top** (observed in
  tm_react): the nested tree is the bigger one, holds tens of thousands of keys
  that exist nowhere else, and is the *better* copy for many shared keys.
  Discarding it would destroy data.

So the decision has to be made per key, from evidence, before anything moves.
This script produces that decision and a proposed move list. **It never writes
into the tree** -- applying the plan is a separate deliberate step, because the
losing copy is sometimes the only holder of a given descriptor family and a
wrong move is unrecoverable.

Why it does not reuse get_information_from_job_folder (validation.py:1078),
which is otherwise the natural per-folder scorecard:

- it returns every field as None when timings.json is missing or empty in the
  generator/ layout, which scores a copy holding real descriptors as empty and
  would bias every verdict toward whichever copy happens to have timings;
- it validates QTAIM without check_bcp_count, so the bond-CP shortfall this
  whole campaign is about would not influence the verdict.

The per-family validators it wraps are called directly here instead.

Usage:
    python scripts/consolidate_nested_verticals.py \
        --folder_root /p/vast1/vargas58/OMol4M \
        --verticals tm_react scaled_separations_exp \
        --quarantine_dir /p/lustre5/vargas58/generator_working/quarantine \
        --out_csv ./audits/consolidate.csv --workers 20

Read the verdict tally, confirm MIXED and GEOMETRY_MISMATCH are zero (or handle
them), then apply <out_csv>_moves.tsv -- quarantine lines before promote lines
for any given key.
"""

import argparse
import collections
import concurrent.futures
import csv
import os
import sys
from typing import List, Optional

from qtaim_gen.source.core.parse_qtaim import dft_inp_to_dict
from qtaim_gen.source.utils.validation import (
    DEFAULT_BCP_TOLERANCE,
    validate_bond_dict,
    validate_charge_dict,
    validate_fuzzy_dict,
    validate_orca_dict,
    validate_other_dict,
    validate_qtaim_dict,
)

JOB_INPUT_NAMES = ("orca.inp", "input.in", "orca.in", "input.inp")

# Families a verdict weighs. QTAIM alone is not enough: a copy can hold a
# complete CP set and a truncated charge table.
FAMILIES = ("qtaim", "charge", "bond", "fuzzy", "other", "orca")
FAMILY_FILES = {
    "qtaim": "qtaim.json",
    "charge": "charge.json",
    "bond": "bond.json",
    "fuzzy": "fuzzy_full.json",
    "other": "other.json",
    "orca": "orca.json",
}


def find_job_input(folder: str) -> Optional[str]:
    """The job's ORCA input, checking the folder root then generator/."""
    for cand in JOB_INPUT_NAMES:
        for base in (folder, os.path.join(folder, "generator")):
            p = os.path.join(base, cand)
            if os.path.isfile(p):
                return p
    return None


def resolve_family_file(folder: str, name: str) -> Optional[str]:
    """Locate one descriptor JSON, root first then generator/.

    Resolved per file rather than by picking a single base directory for the
    whole folder. Choosing one base means detecting the layout from some
    representative file, and then any copy missing *that* file is scored as
    holding nothing -- including copies whose other five families are present
    and valid. Real folders are also genuinely split (qtaim.json left at the
    root by a partial re-run while the rest sits in generator/ from an earlier
    move_results_to_folder), which no single base can represent.
    """
    for base in (folder, os.path.join(folder, "generator")):
        p = os.path.join(base, name)
        if os.path.isfile(p) and os.path.getsize(p) > 0:
            return p
    return None


def find_duplicate_keys(root: str, vertical: str):
    """(shared, nested_only, top_only) keys for one vertical.

    Both asymmetric lists are reported rather than summarised, because
    nested_only being large means the nested tree is not a duplicate at all and
    the whole remedy changes.
    """
    top_dir = os.path.join(root, vertical)
    nest_dir = os.path.join(top_dir, vertical)
    if not os.path.isdir(nest_dir):
        return [], [], []
    top = {e.name for e in os.scandir(top_dir) if e.is_dir() and e.name != vertical}
    nested = {e.name for e in os.scandir(nest_dir) if e.is_dir()}
    return sorted(top & nested), sorted(nested - top), sorted(top - nested)


def geometry_matches(folder_a: str, folder_b: str, tol: float = 1e-3):
    """Whether two copies are the same calculation. Returns (bool, reason).

    Gate for everything downstream: if the geometries differ, the folders are
    not interchangeable and no descriptor comparison between them means
    anything. Atom ORDER is part of the check, not just composition -- every
    descriptor is keyed by atom index, so a reordered input makes the two
    records non-substitutable even at identical coordinates.

    Coordinates compare with a tolerance because inputs regenerated at
    different times can differ in printed precision; an exact compare would
    flag identical jobs as mismatched.
    """
    pa, pb = find_job_input(folder_a), find_job_input(folder_b)
    if pa is None or pb is None:
        return False, "missing_input"
    try:
        da = dft_inp_to_dict(pa, parse_charge_spin=True)
        db = dft_inp_to_dict(pb, parse_charge_spin=True)
    except Exception as e:
        return False, f"unparseable_input:{type(e).__name__}"
    if (da.get("charge"), da.get("spin")) != (db.get("charge"), db.get("spin")):
        return False, "charge_spin_differs"
    ma, mb = da["mol"], db["mol"]
    if len(ma) != len(mb):
        return False, "n_atoms_differs"
    for i in sorted(ma):
        if ma[i]["element"] != mb[i]["element"]:
            return False, f"element_order_differs_at_{i}"
        for x, y in zip(ma[i]["pos"], mb[i]["pos"]):
            if abs(x - y) > tol:
                return False, f"coords_differ_at_{i}"
    return True, "match"


def score_folder(
    folder: str, full_set: int = 1, bcp_tolerance: int = DEFAULT_BCP_TOLERANCE
) -> dict:
    """Per-family validity for one copy: {family: True | False | None}.

    None means the file is absent, False that it is present but invalid. A plain
    bool would lose that distinction, and it matters: absent can mean cleaned,
    while invalid means the step produced garbage.
    """
    out = {f: None for f in FAMILIES}
    out.update({"n_atoms": None, "spin": None, "layout": None, "last_edit": None,
                "error": ""})

    inp = find_job_input(folder)
    if inp is None:
        out["error"] = "no_input"
        return out
    try:
        dft = dft_inp_to_dict(inp, parse_charge_spin=True)
    except Exception as e:
        out["error"] = f"unparseable_input:{type(e).__name__}"
        return out
    n_atoms = len(dft["mol"])
    spin = dft.get("spin")
    spin_tf = spin is not None and spin != 1
    out["n_atoms"], out["spin"] = n_atoms, spin

    checks = {
        "qtaim": lambda p: validate_qtaim_dict(
            p, n_atoms=n_atoms, folder=folder,
            check_bcp_count=True, bcp_tolerance=bcp_tolerance,
        ),
        "charge": lambda p: validate_charge_dict(
            p, n_atoms=n_atoms, full_set=full_set
        ),
        "bond": lambda p: validate_bond_dict(p, full_set=full_set, n_atoms=n_atoms),
        "fuzzy": lambda p: validate_fuzzy_dict(
            p, n_atoms=n_atoms, spin_tf=spin_tf, full_set=full_set
        ),
        "other": lambda p: validate_other_dict(p, full_set=full_set),
        "orca": lambda p: validate_orca_dict(p, n_atoms=n_atoms),
    }
    found, where = {}, set()
    for fam, name in FAMILY_FILES.items():
        p = resolve_family_file(folder, name)
        if p is None:
            continue
        found[fam] = p
        where.add("generator" if os.path.dirname(p).endswith("generator") else "root")
        try:
            out[fam] = bool(checks[fam](p))
        except Exception:
            out[fam] = False

    # "split" is worth surfacing: it means a partial re-run left files in both
    # places, which is how the single-base version misread these folders
    out["layout"] = (
        "split" if len(where) > 1 else (where.pop() if where else "none")
    )
    if found:
        out["last_edit"] = int(max(os.path.getmtime(p) for p in found.values()))
    return out


def consolidate_verdict(top: dict, nested: dict) -> str:
    """Which copy to keep, by set dominance over the valid families.

    Deliberately not a count or a weighted score. If each copy is valid for a
    family the other is not, neither substitutes for the other and the answer is
    MIXED: taking charge.json from one and qtaim.json from the other would pair
    descriptors computed from different wavefunctions, which is worse than
    either copy alone. Those get rerun, not merged.
    """
    st = {f for f in FAMILIES if top.get(f)}
    sn = {f for f in FAMILIES if nested.get(f)}
    if st == sn:
        return "equal"
    if st > sn:
        return "top"
    if sn > st:
        return "nested"
    return "MIXED"


def plan_moves(rows, root: str, quarantine: str):
    """(action, src, dst, verdict, key) tuples. Nothing is executed."""
    moves = []
    for r in rows:
        verdict, key, vert = r["verdict"], r["key"], r["vertical"]
        dest_top = os.path.join(root, vert, key)
        if verdict in ("top", "equal") and quarantine:
            moves.append(("quarantine", r["nested"],
                          os.path.join(quarantine, vert, "nested", key), verdict, key))
        elif verdict == "nested" and quarantine:
            # order matters on apply: vacate the top slot before promoting
            moves.append(("quarantine", r["top"],
                          os.path.join(quarantine, vert, "top", key), verdict, key))
            moves.append(("promote", r["nested"], dest_top, verdict, key))
        elif verdict == "nested_only":
            moves.append(("promote", r["nested"], dest_top, verdict, key))
    return moves


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--folder_root", required=True, help="results tree root")
    parser.add_argument(
        "--verticals", nargs="+", default=None,
        help="default: every vertical that nests a directory of its own name",
    )
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--full_set", type=int, default=1,
        help="analysis level the per-family validators expect; match the "
        "runner's --full_set (default 1)",
    )
    parser.add_argument(
        "--bcp_tolerance", type=int, default=DEFAULT_BCP_TOLERANCE,
        help="bond CPs allowed missing before the QTAIM family counts invalid "
        f"(default {DEFAULT_BCP_TOLERANCE}); keep equal to the runner's value so "
        "a copy the runner accepts is not scored as broken here",
    )
    parser.add_argument(
        "--geom_tol", type=float, default=1e-3,
        help="per-coordinate tolerance in Angstrom when deciding whether two "
        "copies are the same calculation (default 1e-3)",
    )
    parser.add_argument(
        "--quarantine_dir", default=None,
        help="where the plan should send losing copies. Must sit OUTSIDE "
        "--folder_root, or the next audit walks them again. Omit to plan "
        "promotions only.",
    )
    args = parser.parse_args(argv)

    root = os.path.abspath(args.folder_root)
    verticals = args.verticals or sorted(
        d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d, d))
    )
    if not verticals:
        print("no vertical under --folder_root nests a dir of its own name",
              file=sys.stderr)
        return 1

    rows = []
    for vert in verticals:
        shared, nested_only, top_only = find_duplicate_keys(root, vert)
        if not shared and not nested_only:
            print(f"{vert:<26} no nested tree", flush=True)
            continue
        top_dir = os.path.join(root, vert)
        nest_dir = os.path.join(top_dir, vert)
        print(f"{vert:<26} shared {len(shared)}  nested-only {len(nested_only)}  "
              f"top-only {len(top_only)}", flush=True)

        def one(key, top_dir=top_dir, nest_dir=nest_dir, vert=vert):
            tp, np_ = os.path.join(top_dir, key), os.path.join(nest_dir, key)
            row = {"vertical": vert, "key": key, "top": tp, "nested": np_}
            same, why = geometry_matches(tp, np_, tol=args.geom_tol)
            row["geometry"] = why
            if not same:
                row["verdict"] = "GEOMETRY_MISMATCH"
                return row
            st = score_folder(tp, args.full_set, args.bcp_tolerance)
            sn = score_folder(np_, args.full_set, args.bcp_tolerance)
            for fam in FAMILIES:
                row[f"top_{fam}"] = st.get(fam)
                row[f"nested_{fam}"] = sn.get(fam)
            row.update({
                "top_layout": st.get("layout"), "nested_layout": sn.get("layout"),
                "top_last_edit": st.get("last_edit"),
                "nested_last_edit": sn.get("last_edit"),
                "n_atoms": st.get("n_atoms"),
                "error": st.get("error") or sn.get("error") or "",
                "verdict": consolidate_verdict(st, sn),
            })
            return row

        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
            for i, row in enumerate(ex.map(one, shared)):
                rows.append(row)
                if (i + 1) % 5000 == 0:
                    print(f"  {i + 1}/{len(shared)} pairs scored", flush=True)

        for key in nested_only:
            rows.append({
                "vertical": vert, "key": key, "top": "",
                "nested": os.path.join(nest_dir, key),
                "verdict": "nested_only", "geometry": "n/a",
            })

    quarantine = os.path.abspath(args.quarantine_dir) if args.quarantine_dir else ""
    if quarantine and quarantine.startswith(root + os.sep):
        print(f"--quarantine_dir sits inside --folder_root ({quarantine}); pick a "
              "path outside it or the next audit walks the quarantined copies",
              file=sys.stderr)
        return 2
    moves = plan_moves(rows, root, quarantine)

    fields = (
        ["vertical", "key", "verdict", "geometry", "n_atoms", "top_layout",
         "nested_layout", "top_last_edit", "nested_last_edit"]
        + [f"top_{f}" for f in FAMILIES]
        + [f"nested_{f}" for f in FAMILIES]
        + ["error", "top", "nested"]
    )
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    moves_path = os.path.splitext(args.out_csv)[0] + "_moves.tsv"
    with open(moves_path, "w") as f:
        f.write("action\tsrc\tdst\tverdict\tkey\n")
        for m in moves:
            f.write("\t".join(str(x) for x in m) + "\n")

    tally = collections.Counter(r["verdict"] for r in rows)
    print(f"\n{len(rows)} folder pairs -> {args.out_csv}")
    for v, n in tally.most_common():
        print(f"  {v:<20}{n:>8}")

    mixed = [r for r in rows if r["verdict"] == "MIXED"]
    bad = [r for r in rows if r["verdict"] == "GEOMETRY_MISMATCH"]
    if mixed:
        print(f"\n  {len(mixed)} MIXED: each copy is valid for a family the other "
              "is not, so neither\n  substitutes for the other. NOT in the moves "
              "plan -- rerun these rather than\n  merging, or descriptors from two "
              "different wavefunctions get paired.")
        for r in mixed[:5]:
            t = [f for f in FAMILIES if r.get(f"top_{f}")]
            n_ = [f for f in FAMILIES if r.get(f"nested_{f}")]
            print(f"    {r['key']}: top={t} nested={n_}")
    if bad:
        print(f"\n  {len(bad)} GEOMETRY_MISMATCH: the two copies are not the same "
              "calculation.\n  Excluded from the plan; investigate before touching "
              "either copy.")
        for r in bad[:5]:
            print(f"    {r['key']}: {r['geometry']}")
    if not quarantine:
        print("\n  no --quarantine_dir, so only nested_only promotions were "
              "planned. Pass one\n  to plan the quarantine moves too.")

    print(f"\n  {moves_path}  <- {len(moves)} proposed moves; NOTHING was moved.")
    print("  Review, then apply deliberately: quarantine lines before promote "
          "lines for\n  the same key, and mv rather than rm so a wrong verdict "
          "stays recoverable.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
