#!/usr/bin/env python3
"""Fix all-alpha qtaim.json records of UKS singlets in place, without a QTAIM rerun.

Multiwfn read some unrestricted .wfn files as all-alpha (density_beta == 0 at every CP). For a
singlet whose alpha and beta densities agree (<S**2> ~ 0), the total-density fields are right and
the wrong ones follow from them exactly:
  density_alpha = density_beta = density_all / 2,  spin_density = 0
  ELF and LOL: the uniform-gas term D0 = 3/10 (6 pi^2)^(2/3) (rho_a^5/3 + rho_b^5/3) came out
  c = 2^(2/3) too big, while the kinetic terms were right, so
    e_loc_func = 1 / (1 + c^2 (1/ELF - 1))    (ELF = 1/(1+chi^2), chi = D/D0)
    lol        = t / (1 + t),  t = (LOL / (1 - LOL)) / c    (LOL = t/(1+t), t = D0/tau)
Checked against fresh .wfx reruns (ani1xbb, trans1x, tm_react; 2026-10-06): at <S**2> < 0.05 the
max error was 1.8e-6 (ELF), 2.6e-7 (LOL), 1.5e-3 e/bohr^3 (spin fields). Above that it is not exact.

Per folder, under the runners' .processing.lock (no lock with --dry_run):
  1. qtaim.json must be all-alpha: density_beta == 0 at every CP with density_all > 1e-6
  2. multiplicity 1 (geometry input, results folder first, then the input folder)
  3. ORCA <S**2> (orca.json s_squared) below --max_s2
  4. rewrite the five fields at every CP, and record what was done in qtaim_allalpha_fix.json
     next to qtaim.json (the archived CPprop.txt keeps the all-alpha values)
A fixed record is no longer all-alpha, so a second pass reports not_allalpha.

Statuses: fixed, would_fix (--dry_run), not_allalpha (nothing to do), open_shell, high_s2,
no_s2, no_inp (multiplicity unknown), no_qtaim_json, missing (no folder), locked, failed.
--list_remaining writes the entries (as given) that need a QTAIM rerun or a look: everything except
fixed, would_fix and not_allalpha.

Example:
    fix-allalpha-qtaim --folder_list scan/ani1xbb_allalpha.txt --workers 32 --dry_run \\
        --report ani1xbb_allalpha_plan.json --list_remaining ani1xbb_allalpha_rerun.txt
"""

import argparse
import json
import logging
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm

STATUS_FIXED = "fixed"
STATUS_WOULD_FIX = "would_fix"
STATUS_NOT_ALLALPHA = "not_allalpha"
STATUS_OPEN_SHELL = "open_shell"
STATUS_HIGH_S2 = "high_s2"
STATUS_NO_S2 = "no_s2"
STATUS_NO_INP = "no_inp"
STATUS_NO_QTAIM_JSON = "no_qtaim_json"
STATUS_MISSING = "missing"
STATUS_LOCKED = "locked"
STATUS_FAILED = "failed"
STATUSES = (STATUS_FIXED, STATUS_WOULD_FIX, STATUS_NOT_ALLALPHA, STATUS_OPEN_SHELL, STATUS_HIGH_S2,
            STATUS_NO_S2, STATUS_NO_INP, STATUS_NO_QTAIM_JSON, STATUS_MISSING, STATUS_LOCKED, STATUS_FAILED)
DONE = (STATUS_FIXED, STATUS_WOULD_FIX, STATUS_NOT_ALLALPHA)

C = 2 ** (2 / 3)
SIDECAR = "qtaim_allalpha_fix.json"
FIELDS = ("density_alpha", "density_beta", "spin_density", "e_loc_func", "lol")


def _num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _paths(entry: str, root_inputs: Optional[str], root_results: Optional[str]) -> Tuple[str, Optional[str]]:
    """(results folder, input folder or None) for an entry given as either path."""
    if root_inputs and root_results:
        if entry.startswith(root_inputs):
            return os.path.join(root_results, entry[len(root_inputs):].lstrip(os.sep)), entry
        if entry.startswith(root_results):
            return entry, os.path.join(root_inputs, entry[len(root_results):].lstrip(os.sep))
    return entry, None


def _first_file(folder: str, name: str) -> Optional[str]:
    for rel in (os.path.join("generator", name), name):
        p = os.path.join(folder, rel)
        if os.path.isfile(p):
            return p
    return None


def is_all_alpha(record: dict) -> bool:
    cps = [v for v in record.values() if isinstance(v, dict) and _num(v.get("density_all"))
           and v["density_all"] > 1e-6]
    if not cps or not any(_num(v.get("density_beta")) for v in cps):
        return False
    return all(abs(v.get("density_beta") or 0.0) < 1e-12 for v in cps)


def elf_fix(elf: float) -> float:
    if elf <= 0:
        return elf
    return 1 / (1 + C * C * (1 / elf - 1))


def lol_fix(lol: float) -> float:
    if lol <= 0 or lol >= 1:
        return lol
    t = lol / (1 - lol) / C
    return t / (1 + t)


def fix_record(record: dict) -> dict:
    """The record with the five spin-dependent fields recomputed at every CP."""
    out = {}
    for key, cp in record.items():
        if not (isinstance(cp, dict) and _num(cp.get("density_all"))):
            out[key] = cp
            continue
        cp = dict(cp)
        half = cp["density_all"] / 2
        cp["density_alpha"] = half
        cp["density_beta"] = half
        cp["spin_density"] = 0.0
        if _num(cp.get("e_loc_func")):
            cp["e_loc_func"] = elf_fix(cp["e_loc_func"])
        if _num(cp.get("lol")):
            cp["lol"] = lol_fix(cp["lol"])
        out[key] = cp
    return out


def _multiplicity(bases: List[Optional[str]]) -> Optional[int]:
    from qtaim_gen.source.utils.validation import get_charge_spin_n_atoms_from_folder

    for base in bases:
        if base and os.path.isdir(base):
            parsed = get_charge_spin_n_atoms_from_folder(base)
            if parsed:
                return int(parsed["spin"])
    return None


def _s_squared(bases: List[Optional[str]]) -> Optional[float]:
    for base in bases:
        if not (base and os.path.isdir(base)):
            continue
        path = _first_file(base, "orca.json")
        if path:
            try:
                with open(path) as f:
                    s2 = json.load(f).get("s_squared")
            except (OSError, ValueError):
                continue
            if _num(s2):
                return float(s2)
    return None


def _plan(folder: str, inputs: Optional[str], max_s2: float) -> Tuple[str, Optional[dict], str, Optional[float]]:
    """(status, fixed record to write or None, qtaim.json path, s_squared)."""
    path = _first_file(folder, "qtaim.json")
    if path is None:
        return STATUS_NO_QTAIM_JSON, None, "", None
    with open(path) as f:
        record = json.load(f)
    if not is_all_alpha(record):
        return STATUS_NOT_ALLALPHA, None, path, None
    mult = _multiplicity([folder, inputs])
    if mult is None:
        return STATUS_NO_INP, None, path, None
    if mult != 1:
        return STATUS_OPEN_SHELL, None, path, None
    s2 = _s_squared([folder, inputs])
    if s2 is None:
        return STATUS_NO_S2, None, path, None
    if s2 >= max_s2:
        return STATUS_HIGH_S2, None, path, s2
    return STATUS_FIXED, fix_record(record), path, s2


def process_folder(entry: str, root_inputs: Optional[str], root_results: Optional[str],
                   dry_run: bool, max_s2: float) -> Dict[str, object]:
    from qtaim_gen.source.core.workflow import acquire_lock, release_lock
    from qtaim_gen.source.utils.atomic_write import atomic_json_write

    folder, inputs = _paths(entry, root_inputs, root_results)
    result: Dict[str, object] = {"entry": entry, "folder": folder, "status": "", "s_squared": None,
                                 "error": ""}
    if not os.path.isdir(folder):
        result["status"] = STATUS_MISSING
        return result
    if not dry_run and not acquire_lock(folder):
        result["status"] = STATUS_LOCKED
        return result
    try:
        status, fixed, path, s2 = _plan(folder, inputs, max_s2)
        result["s_squared"] = s2
        if fixed is not None:
            if dry_run:
                status = STATUS_WOULD_FIX
            else:
                atomic_json_write(os.path.join(os.path.dirname(path), SIDECAR), {
                    "tool": "fix-allalpha-qtaim", "fixed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "s_squared": s2, "max_s2": max_s2, "fields": list(FIELDS),
                    "n_cps": sum(1 for v in fixed.values() if isinstance(v, dict))})
                atomic_json_write(path, fixed)
        result["status"] = status
    except Exception as e:
        result["status"] = STATUS_FAILED
        result["error"] = f"{type(e).__name__}: {e}"
    finally:
        if not dry_run:
            release_lock(folder)
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--folder_list", required=True,
                    help="One job folder per line (input-tree paths with the two roots, or results paths).")
    ap.add_argument("--root_omol_inputs", default=None)
    ap.add_argument("--root_omol_results", default=None)
    ap.add_argument("--max_s2", type=float, default=0.05,
                    help="Fix only below this ORCA <S**2> (default 0.05, the validated range).")
    ap.add_argument("--workers", type=int, default=0, help="Parallel workers (default: cpu_count, 1 disables).")
    ap.add_argument("--ordered", action="store_true", help="List order instead of a random order.")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None, help="Process at most N folders.")
    ap.add_argument("--dry_run", action="store_true", help="Classify only, write nothing, take no lock.")
    ap.add_argument("--report", default=None, help="Write a JSON report of per-folder results.")
    ap.add_argument("--list_remaining", default=None,
                    help="Write the entries that need a QTAIM rerun or a look (every status except "
                         + ", ".join(DONE) + ").")
    args = ap.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    with open(args.folder_list) as f:
        entries = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    if not args.ordered:
        random.Random(args.seed).shuffle(entries)
    if args.limit is not None:
        entries = entries[: args.limit]
    n = len(entries)
    if n == 0:
        print("No folders listed.", file=sys.stderr)
        return 1
    workers = min(args.workers if args.workers > 0 else (os.cpu_count() or 1), n)
    print(f"Fixing all-alpha qtaim.json in {n} folders with {workers} worker(s); max_s2={args.max_s2}, "
          f"dry_run={args.dry_run}", file=sys.stderr)

    kwargs = dict(root_inputs=args.root_omol_inputs, root_results=args.root_omol_results,
                  dry_run=args.dry_run, max_s2=args.max_s2)
    results: List[Dict[str, object]] = []
    t0 = time.time()
    with tqdm(total=n, desc="Fixing", unit="folder", file=sys.stderr, mininterval=2.0) as bar:
        if workers <= 1:
            for e in entries:
                results.append(process_folder(e, **kwargs))
                bar.update(1)
        else:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futs = [pool.submit(process_folder, e, **kwargs) for e in entries]
                for fut in as_completed(futs):
                    results.append(fut.result())
                    bar.update(1)

    agg: Dict[str, object] = {"folders_total": n}
    for s in STATUSES:
        agg[s] = sum(1 for r in results if r["status"] == s)
    agg["elapsed_sec"] = round(time.time() - t0, 2)
    print("\nAll-alpha fix summary:", file=sys.stderr)
    for k, v in agg.items():
        print(f"  {k:24s} {v}", file=sys.stderr)
    for r in [r for r in results if r["status"] == STATUS_FAILED][:10]:
        print(f"  FAILED {r['folder']}: {r['error']}", file=sys.stderr)

    if args.report:
        os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
        with open(args.report, "w") as f:
            json.dump({"aggregate": agg, "per_folder": results}, f, indent=2, default=str)
        print(f"\nReport written: {args.report}", file=sys.stderr)
    if args.list_remaining:
        os.makedirs(os.path.dirname(args.list_remaining) or ".", exist_ok=True)
        remaining = [r["entry"] for r in results if r["status"] not in DONE]
        with open(args.list_remaining, "w") as f:
            f.write("\n".join(remaining) + ("\n" if remaining else ""))
        print(f"{len(remaining)} folders need a QTAIM rerun or a look: {args.list_remaining}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
