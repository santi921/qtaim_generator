"""Sync loose qtaim.out/CPprop.txt for jobs merged before 250b5e9.

usage: python resync_qtaim_loose.py shadow_mismatch.txt PLAN.tsv [PLAN.tsv ...]
Run from the directory holding merge_l1_into_l0.py (250b5e9 or later).
"""
import csv
import json
import sys

import merge_l1_into_l0 as m
from qtaim_gen.source.utils.validation import read_qtaim_out

assert hasattr(m, "_sync_qtaim_loose"), "merge_l1_into_l0.py is older than 250b5e9; copy the new one first"

wanted = set(open(sys.argv[1]).read().split())
dirs = {}
for plan in sys.argv[2:]:
    for row in csv.DictReader(open(plan), delimiter="\t"):
        if row["rel"] in wanted:
            dirs[row["rel"]] = (row["l0_dir"], row["l1_dir"])

fixed = rolled = 0
for rel in sorted(wanted):
    if rel not in dirs:
        print(f"NOT IN PLANS  {rel}")
        continue
    dst, src = dirs[rel]
    act = json.load(open(f"{dst}/generator/{m.MARKER}"))["action"]
    # PATCH kept dst's generator/, so a stale generator/qtaim.out can shadow too;
    # REPLACE installed src's whole generator/, so only the job root is at risk
    locs = ("", "generator") if act == "PATCH_QTAIM" else ("",)
    level = 0 if act == "REPLACE_L0" else 1
    flags = m.LOOSE if act == "REPLACE_LOOSE" else m.STRICT_DEFAULT
    copied, over = [], []
    m._sync_qtaim_loose(src, dst, locs, copied, over)
    ok, why = m._validate(dst, level, flags)
    if ok and read_qtaim_out(dst) == read_qtaim_out(src):
        m._commit_overwrites(dst, over)
        fixed += 1
        print(f"fixed         {act:12s} {rel}")
    else:
        m._undo_root_copies(dst, copied, over)
        rolled += 1
        print(f"ROLLED BACK   {act:12s} {rel}  {why[:100]}")
print(f"\n{fixed} fixed, {rolled} rolled back")
