"""End-to-end check of gbw_analysis(recheck_fuzzy=True) on a real open-shell job (tracker #28).

Usage: python scripts/restart_fuzzy_rerun_test.py SRC_DIR WORK_DIR
  SRC_DIR holds orca.inp, orca.gbw, orca.out of a small open-shell job
  (validated on an IO radical: doublet, I with a def2 ECP). Needs local
  Multiwfn and orca_2mkl (paths below).

1. Legacy .wfn run (fuzzy spin read as all-alpha, fuzzy_bond inflated), then
   zero hirsh_fuzzy_density to mimic the old two-block input bug.
2a. recheck_fuzzy with no gbw left in the folder: reparses hirsh density from the
    archive (needs no wavefunction) but refuses the spin/fuzzy_bond rerun and
    leaves those values and orca.wfn untouched.
2b. gbw restored: recheck_fuzzy (no --restart) must rerun only the spin steps +
    fuzzy_bond on a .wfx and leave every other output byte-identical.
3. recheck_fuzzy again: nothing left to do.
"""
import hashlib
import json
import os
import re
import shutil
import sys

from qtaim_gen.source.core.omol import gbw_analysis

SRC, WORK = (os.path.abspath(p) for p in sys.argv[1:3])
MW = "/home/santiagovargas/dev/Multiwfn_3_8/Multiwfn_noGUI"
O2M = "/home/santiagovargas/dev/orca_6_0_1_linux_x86-64_shared_openmpi416/orca_2mkl"
os.environ["Multiwfnpath"] = os.path.dirname(MW)
KW = dict(multiwfn_cmd=MW, orca_2mkl_cmd=O2M, separate=True, clean=True, overwrite=False,
          orca_6=True, full_set=0, move_results=True, n_threads=4, preprocess_compressed=False)
GEN = os.path.join(WORK, "generator")
LOG = os.path.join(WORK, "gbw_analysis.log")


def log_since(n):
    lines = open(LOG).read().splitlines()
    new = lines[n:]
    ran = [re.search(r"Completed (\S+) in", l).group(1) for l in new if "Completed " in l and " seconds" in l]
    return ran, new, len(lines)


def jload(name):
    return json.load(open(os.path.join(GEN, name)))


def digest(name):
    p = os.path.join(GEN, name)
    return hashlib.sha256(open(p, "rb").read()).hexdigest() if os.path.exists(p) else None


def spin_sum(d):
    return round(sum(v for k, v in d.items() if k not in ("sum", "abs_sum")), 4)


shutil.rmtree(WORK, ignore_errors=True)
os.makedirs(WORK)
for f in ("orca.inp", "orca.gbw", "orca.out"):
    shutil.copy(os.path.join(SRC, f), WORK)

# 1. legacy run + old hirsh bug
gbw_analysis(WORK, restart=False, wfx=False, **KW)
fz = jload("fuzzy_full.json")
fz["hirsh_fuzzy_density"] = {k: 0.0 for k in fz["hirsh_fuzzy_density"]}
json.dump(fz, open(os.path.join(GEN, "fuzzy_full.json"), "w"))
ran, _, n = log_since(0)
print("1  ran:", ran)
print("1  becke/hirsh spin sums:", spin_sum(fz["becke_fuzzy_spin"]), spin_sum(fz["hirsh_fuzzy_spin"]),
      "fuzzy_bond:", jload("bond.json")["fuzzy_bond"], "root:", sorted(os.listdir(WORK)))
bond_before = jload("bond.json")
before = {f: digest(f) for f in ("charge.json", "qtaim.json", "other.json", "orca.json", "fuzzy_full.json", "bond.json")}

# 2a. no wavefunction source: refuse
res = gbw_analysis(WORK, restart=False, wfx=True, recheck_fuzzy=True, **KW)
ran, new, n = log_since(n)
fz_a = jload("fuzzy_full.json")
print("2a returned:", res, "ran:", ran,
      "non-fuzzy outputs unchanged:", all(digest(f) == before[f] for f in before if f != "fuzzy_full.json"),
      "spin keys untouched:", all(fz_a[k] == fz[k] for k in ("becke_fuzzy_spin", "hirsh_fuzzy_spin")),
      "hirsh density reparsed:", spin_sum(fz_a["hirsh_fuzzy_density"]),
      "orca.wfn kept:", os.path.exists(os.path.join(WORK, "orca.wfn")))

# 2b. source restored: reparse + targeted rerun
shutil.copy(os.path.join(SRC, "orca.gbw"), WORK)
res = gbw_analysis(WORK, restart=False, wfx=True, recheck_fuzzy=True, **KW)
ran, new, n = log_since(n)
print("2b returned:", res, "ran:", ran)
print("2b recheck:", [l.split("recheck_fuzzy: ", 1)[1] for l in new if "recheck_fuzzy: reparsed" in l])
fz2, bd2 = jload("fuzzy_full.json"), jload("bond.json")
print("2b becke/hirsh spin sums:", spin_sum(fz2["becke_fuzzy_spin"]), spin_sum(fz2["hirsh_fuzzy_spin"]),
      "hirsh density sum:", spin_sum(fz2["hirsh_fuzzy_density"]), "fuzzy_bond:", bd2["fuzzy_bond"])
print("2b identical:", {f: digest(f) == before[f] for f in ("charge.json", "qtaim.json", "other.json", "orca.json")},
      "becke density identical:", fz2["becke_fuzzy_density"] == fz["becke_fuzzy_density"],
      "mayer identical:", bd2["mayer_orca"] == bond_before["mayer_orca"])

# 3. idempotent
res = gbw_analysis(WORK, restart=False, wfx=True, recheck_fuzzy=True, **KW)
ran, new, n = log_since(n)
print("3  returned:", res, "ran:", ran,
      "recheck:", [l.split("recheck_fuzzy: ", 1)[1] for l in new if "recheck_fuzzy: reparsed" in l])
