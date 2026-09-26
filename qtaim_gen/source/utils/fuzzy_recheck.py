"""Value-level recheck of fuzzy integrations and fuzzy bond orders (tracker #28).

The restart gate only checks that a step's data is present and complete, so
three classes of wrong-but-complete data are never rerun:

- `hirsh_fuzzy_density` all zeros (a Multiwfn input integrated rho twice and
  the parser kept the second, empty block);
- fuzzy spin integrations that do not sum to (multiplicity - 1): open-shell
  jobs run from a .wfn, which Multiwfn reads as all-alpha, and every
  `hirsh_fuzzy_spin` (its input integrated rho, not spin density);
- open-shell `fuzzy_bond` from a .wfn (all-alpha) or parsed before the parser
  read the Total column (alpha only).

`recheck_fuzzy` finds these, re-parses the archived Multiwfn output where that
output is trustworthy (no rerun), and otherwise invalidates the step (compiled
key plus per-step .out/.json in the job root and generator/) so a restart
reruns exactly that step. Open-shell spin steps only rerun on a .wfx: an
existing one, or one rebuilt from a gbw source (any .wfn is then removed).
When a rerun could not actually happen (no source, or --wfn for spin steps)
the steps are left untouched and the folder is refused; reparses, which need
no wavefunction, are applied either way.
"""

import glob
import json
import math
import os
import tempfile
from typing import Dict, List, Optional, Tuple

from qtaim_gen.source.core.omol import _gbw_source_present, _wavefunction_path
from qtaim_gen.source.core.parse_multiwfn import (
    parse_bond_order_fuzzy,
    parse_fuzzy_real_space,
)
from qtaim_gen.source.utils.atomic_write import atomic_json_write
from qtaim_gen.source.utils.validation import read_multiwfn_out

DENSITY_KEYS = ("becke_fuzzy_density", "hirsh_fuzzy_density", "mbis_fuzzy_density")
SPIN_KEYS = ("becke_fuzzy_spin", "hirsh_fuzzy_spin", "mbis_fuzzy_spin")
SPIN_TOL = 0.1  # integrated spin density = N_alpha - N_beta exactly; grid error is ~1e-4
_SUMMARY = ("sum", "abs_sum")
SPIN_SENSITIVE = frozenset(SPIN_KEYS) | {"fuzzy_bond"}


def _atom_values(entry) -> Optional[List[float]]:
    if not isinstance(entry, dict):
        return None
    try:
        vals = [float(v) for k, v in entry.items() if k not in _SUMMARY]
    except (TypeError, ValueError):
        return None
    if not vals or not all(math.isfinite(v) for v in vals):
        return None
    return vals


def fuzzy_value_failures(fuzzy_dict: dict, mult: int) -> List[str]:
    """Steps present in `fuzzy_dict` whose values are physically impossible."""
    bad = []
    for key in DENSITY_KEYS:
        if key in fuzzy_dict:
            vals = _atom_values(fuzzy_dict[key])
            if vals is None or all(abs(v) < 1e-12 for v in vals):
                bad.append(key)
    if mult > 1:
        for key in SPIN_KEYS:
            if key in fuzzy_dict:
                vals = _atom_values(fuzzy_dict[key])
                if vals is None or abs(sum(vals) - (mult - 1)) > SPIN_TOL:
                    bad.append(key)
    return bad


def out_electron_counts(text: str) -> Optional[Tuple[float, float, float]]:
    """(total, alpha, beta) from a Multiwfn banner line, or None."""
    for line in text.splitlines():
        if "Total/Alpha/Beta electrons:" in line:
            parts = line.split(":", 1)[1].split()
            try:
                return float(parts[0]), float(parts[1]), float(parts[2])
            except (IndexError, ValueError):
                return None
    return None


def _parse_text(step: str, text: str, parser):
    # the parsers take a path and name fuzzy results after the file stem
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, f"{step}.out")
        with open(p, "w") as f:
            f.write(text)
        return parser(p)


def _load_json(path: str) -> Optional[dict]:
    try:
        with open(path) as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def _compiled(folder: str, name: str) -> Dict[str, dict]:
    """{path: data} for each existing copy of a compiled JSON (root and generator/)."""
    out = {}
    for base in (folder, os.path.join(folder, "generator")):
        p = os.path.join(base, name)
        if os.path.isfile(p):
            data = _load_json(p)
            if data is not None:
                out[p] = data
    return out


def _preferred(folder: str, copies: Dict[str, dict]) -> dict:
    # generator/ holds the merged result of a completed run
    gen_dir = os.path.join(folder, "generator")
    gen = [d for p, d in copies.items() if os.path.dirname(p) == gen_dir]
    return gen[0] if gen else next(iter(copies.values()), {})


def plan_recheck(folder: str, mult: int) -> Dict[str, object]:
    """Decide, without writing anything, what each suspect step needs.

    Returns {"reparse": {step: payload}, "rerun": [step, ...]}.
    """
    reparse: Dict[str, object] = {}
    rerun: List[str] = []

    fuzzy = _preferred(folder, _compiled(folder, "fuzzy_full.json"))
    for step in fuzzy_value_failures(fuzzy, mult):
        text = read_multiwfn_out(folder, f"{step}.out")
        if text is not None:
            try:
                payload = _parse_text(step, text, parse_fuzzy_real_space).get(step)
            except Exception:
                payload = None
            if payload is not None and not fuzzy_value_failures({step: payload}, mult):
                reparse[step] = payload
                continue
        rerun.append(step)

    if mult > 1:
        bond = _preferred(folder, _compiled(folder, "bond.json"))
        if "fuzzy_bond" in bond:
            text = read_multiwfn_out(folder, "fuzzy_bond.out")
            counts = out_electron_counts(text) if text is not None else None
            # beta > 0: alpha/beta resolved (.wfx era), only the parser column
            # can be wrong. beta == 0 is also genuine when every electron is
            # unpaired (H atom, H2+): total == mult - 1.
            if counts is not None and (counts[2] > 0 or abs(counts[0] - (mult - 1)) < 0.5):
                try:
                    payload = _parse_text("fuzzy_bond", text, parse_bond_order_fuzzy)
                except Exception:
                    payload = None
                if payload:
                    if payload != bond["fuzzy_bond"]:
                        reparse["fuzzy_bond"] = payload
                else:
                    rerun.append("fuzzy_bond")
            else:
                # read as all-alpha, or no output to tell which era it came from
                rerun.append("fuzzy_bond")
    return {"reparse": reparse, "rerun": rerun}


def _remove_step_files(folder: str, step: str, exts, logger=None) -> None:
    for base in (folder, os.path.join(folder, "generator")):
        for ext in exts:
            p = os.path.join(base, step + ext)
            if os.path.isfile(p):
                os.remove(p)
                if logger:
                    logger.info("recheck_fuzzy: removed stale %s", p)


def _wavefunction_plan(
    folder: str, rerun: List[str], mult: int, wfx: bool, preprocess_compressed: bool
) -> Tuple[bool, str, List[str]]:
    """Whether the rerun steps can actually run, and which .wfn files to remove.

    Returns (ok, note, wfn_files_to_remove). Spin-sensitive open-shell steps
    need a .wfx: an existing one, or a gbw source to convert from (any .wfn is
    then removed so conversion runs). Other steps need any wavefunction or
    source. Without them the rerun could not happen and must not be set up.
    """
    if not rerun:
        return True, "not needed", []
    spin_rerun = mult > 1 and bool(set(rerun) & SPIN_SENSITIVE)
    source = _gbw_source_present(folder, preprocess_compressed)
    wfns = sorted(
        p for b in (folder, os.path.join(folder, "generator")) for p in glob.glob(os.path.join(b, "*.wfn"))
    )
    if spin_rerun:
        if not wfx:
            return False, "open-shell spin steps need .wfx; refusing to rerun them with --wfn", []
        existing = _wavefunction_path(folder)
        if existing is not None and existing.endswith(".wfx"):
            return True, "orca.wfx present", []
        if source:
            return True, "rebuilding .wfx from the gbw source", wfns
        return False, "open-shell rerun needs a .wfx and no gbw source is present", []
    if _wavefunction_path(folder) is not None or wfns or source:
        return True, "wavefunction or gbw source present", []
    return False, "no wavefunction or gbw source to rerun from", []


def recheck_fuzzy(
    folder: str,
    mult: int,
    logger=None,
    dry_run: bool = False,
    wfx: bool = True,
    preprocess_compressed: bool = False,
) -> Dict[str, object]:
    """Recheck, repair by reparse, and invalidate for rerun. See module docstring.

    Returns {"reparse": [...], "rerun": [...], "wavefunction": str, "ok": bool}.
    Reparses are always applied (they need no wavefunction). Rerun steps are
    only invalidated when `ok`, i.e. they can really run; otherwise their
    current values are left in place and `ok` is False.
    """
    plan = plan_recheck(folder, mult)
    reparse, rerun = plan["reparse"], plan["rerun"]
    ok, wf_note, wfns = _wavefunction_plan(folder, rerun, mult, wfx, preprocess_compressed)
    report = {"reparse": sorted(reparse), "rerun": sorted(rerun), "wavefunction": wf_note, "ok": ok}
    if dry_run:
        return report
    invalidate = rerun if ok else []

    for name in ("fuzzy_full.json", "bond.json"):
        for path, data in _compiled(folder, name).items():
            changed = False
            for step, payload in reparse.items():
                if (name == "bond.json") == (step == "fuzzy_bond") and step in data:
                    data[step] = payload
                    changed = True
            for step in invalidate:
                if (name == "bond.json") == (step == "fuzzy_bond") and step in data:
                    del data[step]
                    changed = True
            if changed:
                atomic_json_write(path, data)
                if logger:
                    logger.info("recheck_fuzzy: rewrote %s", path)
    # A stale per-step .json would be recompiled over the fix or mark a rerun
    # step as done. A reparsed step keeps its .out (it may not be zipped yet).
    for step in reparse:
        _remove_step_files(folder, step, (".json",), logger=logger)
    for step in invalidate:
        _remove_step_files(folder, step, (".out", ".json"), logger=logger)
    if ok:
        for p in wfns:
            os.remove(p)
            if logger:
                logger.info("recheck_fuzzy: removed %s so the rerun converts to .wfx", p)
    if logger:
        logger.info("recheck_fuzzy: reparsed %s, rerun %s (%s)", report["reparse"], report["rerun"], wf_note)
    return report
