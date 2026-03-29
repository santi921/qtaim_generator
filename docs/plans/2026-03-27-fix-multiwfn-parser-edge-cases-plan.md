---
title: "fix: Multiwfn parser edge cases for large molecules"
type: fix
date: 2026-03-27
---

# fix: Multiwfn parser edge cases for large molecules

## Overview

Two parsing failures occur with large-molecule Multiwfn output files (123+ atoms):
1. **Dipole magnitude overflow** — Large dipole values lose whitespace between `(a.u.)` and the number
2. **Missing atom CP in bond path resolution** — Unmatched atoms crash bond path remapping

Edge case files: `data/buggy_qtaim/edge_cases/` (adch.out: 123-atom protein core, becke.out: 136-atom protein core, qtaim_0: 87-atom Ir complex)

## Problem Statement

### Bug 1: Dipole magnitude field overflow (`parse_multiwfn.py`)

Multiwfn uses fixed-width formatting. For large molecules with high dipole moments, the number overflows its field and concatenates directly to `(a.u.)`:

```
# Normal (small molecule):
Total dipole from ADC charges (a.u.):   0.0262041  Error:   0.0001234

# Edge case (large molecule, 123 atoms):
Total dipole from ADC charges (a.u.)1048.6171082  Error:   0.0182899
```

The parser does `float(line.split()[-3])`, which yields `'(a.u.)1048.6171082'` — a `ValueError`.

**Affected functions (4):**
- `parse_charge_doc_adch` (line 471) — `float(line.split()[-3])`
- `parse_charge_becke` (line 387) — `float(line.split()[-3])`
- `parse_charge_doc` (line 96) — `float(line.split()[-3])`
- `parse_charge_base` (line 186, `corrected=True` path) — `float(line.split()[-3])`

### Bug 2: Missing atom CP in bond path resolution (`parse_qtaim.py`)

When `find_cp_map` cannot match a DFT atom to a QTAIM critical point (atom 59 in the 87-atom Ir complex), it stores `{"key": -1, "pos": []}`. Bond CPs that reference the missing atom then crash at `merge_qtaim_inds` line 529:

```python
# line 529: -1 is an int, .split("_") fails
int(qtaim_to_dft[i - 1]["key"].split("_")[0]) - 1
#                         ^^^^ AttributeError: 'int' object has no attribute 'split'
```

**Known limitation (out of scope):** The QTAIM-to-DFT index mapping assumes QTAIM atom CP indices map 1:1 to DFT atom order. This is fragile if QTAIM reorders atoms. Not addressed here — only the crash is fixed.

## Proposed Solution

### Fix 1: Robust `(a.u.)` number extraction

Add a helper to extract floats from potentially concatenated `(a.u.)` tokens:

```python
# qtaim_gen/source/core/parse_multiwfn.py

def _extract_au_float(token: str) -> float:
    """Extract float from a token that may have '(a.u.)' or '(a.u.):' prefix glued to it."""
    if token.startswith("(a.u.)"):
        remainder = token[6:]
        if remainder.startswith(":"):
            remainder = remainder[1:]
        return float(remainder)
    return float(token)
```

Apply in all 4 dipole magnitude parsing locations:

```python
# Before (all 4 functions):
float_dipole = float(line.split()[-3])

# After:
float_dipole = _extract_au_float(line.split()[-3])
```

### Fix 2: Skip bonds referencing missing atoms

In `merge_qtaim_inds` at line 525-538, add a compact guard with logging:

```python
# qtaim_gen/source/core/parse_qtaim.py, merge_qtaim_inds()

for k, v in bond_cps.items():
    bond_list_unsorted = v["connected_bond_paths"]

    # Skip bonds where any connected atom has no CP match
    if any(not isinstance(qtaim_to_dft.get(i - 1, {}).get("key"), str)
           for i in bond_list_unsorted):
        print(f"Warning: skipping bond CP {k}, connected atom has no QTAIM match")
        continue

    bond_list_unsorted = [
        int(qtaim_to_dft[i - 1]["key"].split("_")[0]) - 1
        for i in bond_list_unsorted
    ]
    ...
```

## Acceptance Criteria

- [ ] `parse_charge_doc_adch('data/buggy_qtaim/edge_cases/adch.out')` returns valid charges (123 atoms), dipole magnitude, and atomic dipoles without error
- [ ] `parse_charge_becke('data/buggy_qtaim/edge_cases/becke.out')` returns valid charges (136 atoms), dipole info, and atomic dipoles without error
- [ ] `parse_qtaim('data/buggy_qtaim/edge_cases/qtaim_0/CPprop.txt', 'data/buggy_qtaim/edge_cases/qtaim_0/orca.inp')` completes without error, skipping 1 unmatchable bond with a warning
- [ ] All existing tests in `test_parse_multiwfn.py` and `test_parse.py` still pass
- [ ] New tests for large-molecule edge cases using `data/buggy_qtaim/edge_cases/` fixtures
- [ ] New test for `merge_qtaim_inds` with missing-atom scenario

## Implementation

**Files modified:**
- `qtaim_gen/source/core/parse_multiwfn.py` — add `_extract_au_float`, update 4 call sites
- `qtaim_gen/source/core/parse_qtaim.py` — add bond-skip guard in `merge_qtaim_inds`
- `tests/test_parse_multiwfn.py` — add `test_parse_adch_large_molecule`, `test_parse_becke_large_molecule`
- `tests/test_parse.py` — add `test_merge_qtaim_inds_missing_atom`

## Reviewer Feedback (incorporated)

- **Dropped Bug 2 (XYZ dipole overflow)** — speculative, YAGNI; fix when/if it manifests
- **Simplified bond guard** — `any(not isinstance(..., str))` replaces 8-line flag-and-loop
- **Added warning log** — silently dropping bonds in a scientific pipeline is unacceptable
- **Handle `(a.u.):` colon variant** — `_extract_au_float` strips both 6-char and 7-char prefixes

## References

- Edge case fixtures: `data/buggy_qtaim/edge_cases/`
  - `adch.out` — 123-atom protein core (H65 C37 N12 O9), dipole ~1048 a.u.
  - `becke.out` — 136-atom protein core (H71 C39 N11 O15), dipole ~1346 a.u.
  - `qtaim_0/` — 87-atom Ir complex (charge +2, doublet), 1 unmatchable H atom (DFT index 59)
- Parsers: [parse_multiwfn.py](qtaim_gen/source/core/parse_multiwfn.py), [parse_qtaim.py](qtaim_gen/source/core/parse_qtaim.py)
- Existing tests: [test_parse_multiwfn.py](tests/test_parse_multiwfn.py), [test_parse.py](tests/test_parse.py)
