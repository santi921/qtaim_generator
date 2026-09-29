# Handoff: ORBITAL ENERGIES parsing in `parse_orca.py`

Three defects in `qtaim_gen/source/core/parse_orca.py` around the orbital-energy
state machine. All three were found while building a quality-filter readout in
`oact_utilities`, which consumes `parse_orca_output()` output via
`analysis.parse_generator_data`.

## Summary

- The `SPIN DOWN ORBITALS` block of an unrestricted calculation is silently
  dropped. Every orbital-derived key (`homo_*`, `lumo_*`, `homo_lumo_gap_eh`,
  `n_electrons`, `n_orbitals`) describes the alpha channel only.
- HOMO and LUMO are picked positionally (last occupied line seen, first
  zero-occupancy line seen), which assumes the block is sorted by energy. ROHF
  and ROKS blocks are not sorted, and the reported gap comes out 3.3x too large
  on the test case below.
- `Number of Electrons NEL` is not parsed. It is the only exact electron count
  in the file, and it cannot be reconstructed from the geometry because ECPs
  remove core electrons.

---

## Issue 1: SPIN DOWN ORBITALS block is never read

**Location**

- IDLE trigger: `parse_orca.py:259`, `elif stripped == "ORBITAL ENERGIES":`
- State handler: `parse_orca.py:428-457`
- Finalizer: `_finalize_orbitals`, `parse_orca.py:142`

**Mechanism**

The state machine enters `ORBITAL_ENERGIES` only on a line whose stripped value
is exactly `ORBITAL ENERGIES`. In an unrestricted run ORCA prints that header
once, then two sub-blocks:

```
----------------
ORBITAL ENERGIES
----------------
                 SPIN UP ORBITALS
  NO   OCC          E(Eh)            E(eV)
   0   1.0000     -24.847557      -676.1364
   ...
  (blank line)
                 SPIN DOWN ORBITALS
  NO   OCC          E(Eh)            E(eV)
   ...
```

The handler exits on `stripped == "" or stripped.startswith("----")` once
`section_line_count > 3` (`parse_orca.py:431`), so the blank line between the
two sub-blocks finalizes the result and returns to IDLE. The header
`SPIN DOWN ORBITALS` does not match the IDLE trigger, so the beta block is
skipped entirely. The `SPIN UP ORBITALS` header is also harmless today only by
accident: it splits into 3 tokens and the orbital-line branch requires exactly 4.

**Evidence** (NpF3, multiplicity 5, 60 electrons, UKS)

| key | reported | correct |
| --- | --- | --- |
| `n_electrons` | 32.0 | 60.0 (32 alpha + 28 beta) |
| `homo_eh` | -0.316865 | alpha value; beta HOMO not reported |
| `homo_lumo_gap_eh` | 0.278931 | alpha 0.278931, beta 0.428205 |
| `n_orbitals` | 223 | 223 per spin |

**Suggested shape**

Read both sub-blocks and emit per-spin keys, keeping the existing flat keys
populated for back-compat (see contract below). Something like
`homo_eh_alpha` / `homo_eh_beta` / `homo_lumo_gap_eh_alpha` / `..._beta`, with
the existing `homo_eh`, `lumo_eh`, `homo_lumo_gap_eh` continuing to carry the
alpha values, and `n_electrons` becoming the true total (alpha + beta).

Note the accumulators are reset in the IDLE trigger (`parse_orca.py:260-266`),
not on block entry, so a two-block read needs its own per-block reset. Also the
"last occurrence wins" behavior for repeated `ORBITAL ENERGIES` sections
(geometry optimizations print one per step) must be preserved.

---

## Issue 2: HOMO/LUMO chosen positionally, not by energy

**Location** `parse_orca.py:445-457` and `_finalize_orbitals` at `parse_orca.py:142`

**Mechanism**

`last_occupied_energy` is overwritten on every line with `occ > 0`, so it ends
up holding the last occupied orbital *in file order*. `first_virtual_energy` is
set once on the first `occ == 0` line. That is correct only when the block is
sorted ascending by energy.

**Evidence** (AmO, ROHF/ROKS, single block, 32 orbitals listed)

The singly-occupied orbitals are printed out of energy order:

```
  13   2.0000      -0.359291
  14   1.0000      -0.262394
  15   1.0000      -0.130460   <- true HOMO
  16   1.0000      -0.441338
  ...
  20   1.0000      -0.434274   <- last occupied in file order
  21   0.0000       0.002276   <- LUMO
```

| quantity | value |
| --- | --- |
| positional gap (current) | +0.436550 Eh |
| true gap (max occupied to min virtual) | +0.132736 Eh |

3.3x overstated. The UKS blocks in the NpF3 file are sorted, so the two methods
agree there, which is why this has not surfaced before.

**Suggested fix**

Track `max(energy)` over occupied lines and `min(energy)` over virtual lines
instead of first/last. Keep the eV value paired with the same orbital the Eh
value came from (track a tuple, do not compute the max over Eh and the min over
eV independently).

Consumers currently test only `gap > 0`, but an unsorted block can flip that
sign in either direction, so this is a correctness issue and not only a
magnitude one.

---

## Issue 3: `Number of Electrons NEL` is not parsed

ORCA prints the exact SCF electron count in the settings block:

```
 Number of Electrons    NEL             ....   60
```

This is not recoverable from the molecule. ECPs remove core electrons:

| system | sum of Z minus charge | `NEL` |
| --- | --- | --- |
| NpF3 (def-ECP on Np) | 120 | 60 |
| AmO (Am ECP, 60 core electrons) | 103 | 35 |

Downstream this is `num_electrons_scf`, compared against the grid-integrated
`n_alpha + n_beta` to catch DFT grid integration error (NpF3: 59.999630 vs 60).
`n_alpha`/`n_beta` alone cannot support that check because the sum is compared
against nothing exact.

Once Issue 1 is fixed, `n_electrons` (sum of occupations) equals `NEL` for both
restricted and unrestricted runs, so `NEL` becomes a cheap cross-check rather
than a necessity. It is a one-line addition to the IDLE trigger either way, and
worth having as an independent value.

---

## Consumer contract (read before renaming keys)

`oact_utilities/utils/analysis.py:77` (`parse_generator_data`) calls
`parse_orca_output()`, JSON-serializes the dict, and caches it as
`generator_metrics.json` in each job directory. That JSON is also stored in the
workflow database in a `generator_data` TEXT column. Existing corpora have
hundreds of thousands of these caches.

Two consequences:

1. **Add keys, do not rename them.** Anything reading `homo_lumo_gap_eh` today
   should keep working.
2. **The cache has no staleness check.** `parse_generator_data` returns the
   cached file whenever it exists (`if cache_file.exists() and not recompute`),
   with no mtime or schema comparison. Existing job directories will not pick up
   these fixes without an explicit `recompute=True` pass. If you add a version
   marker to the result dict, say so in the PR and the oact_utilities side can
   gate on it.

---

## Reproduction

Both real ORCA outputs live in the oact_utilities repo:

- `/Users/santiagovargas/dev/oact_utils/tests/files/orca_direct_example/orca.out`
  (AmO, ROHF/ROKS, unsorted occupied orbitals, no `s_squared`, no `n_alpha`)
- `/Users/santiagovargas/dev/oact_utils/tests/files/quacc_example/orca.out.gz`
  (NpF3, UKS multiplicity 5, two spin blocks; gunzip to a temp file first)

```python
from qtaim_gen.source.core.parse_orca import parse_orca_output

r = parse_orca_output(path)
print(r.get("n_electrons"), r.get("homo_lumo_gap_eh"), r.get("n_orbitals"))
```

## Testing

Existing suite: `tests/test_parse_orca.py`, fixtures in
`tests/test_files/orca_outs/` are hand-written minimal `.out` files
(`minimal_rks.out`, `minimal_truncated.out`, `minimal_duplicate_energy.out`).
Follow that style rather than committing the multi-megabyte real outputs.

Fixtures worth adding:

- `minimal_uks.out`: `ORBITAL ENERGIES` header, `SPIN UP ORBITALS` block, blank
  line, `SPIN DOWN ORBITALS` block, with different alpha and beta gaps. Assert
  both per-spin gaps, and `n_electrons == alpha + beta`.
- `minimal_roks_unsorted.out`: single block with occupied orbitals out of energy
  order, as in the AmO case. Assert the gap uses max-occupied and min-virtual.
- `minimal_nel.out` or an addition to an existing fixture: assert `NEL` parses
  and does not equal a naive sum of Z (use an ECP system).
- A regression assert that a two-step geometry optimization (two
  `ORBITAL ENERGIES` sections) still yields the last section's values.

Also confirm the truncated-file fixture still finalizes partial orbital data
(`parse_orca.py:795` finalizes at EOF).

## Out of scope

Restricted open-shell outputs legitimately carry no `UHF SPIN CONTAMINATION`
block and no DFT components block, so `s_squared`, `n_alpha` and `n_beta` are
absent for ROKS and HF runs. That is ORCA behavior, not a parser gap, and should
not be worked around.
