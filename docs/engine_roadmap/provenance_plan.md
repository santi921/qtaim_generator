# Engine provenance: one ENGINE_VERSION, provenance.json and provenance.lmdb

Date: 2026-11-11. Status: plan, no code changed. Branch `feat/engine-provenance` off
`feat/charge-engine` (8b4e8c1). Labels: [V] verified in code, [I] inference, [E] estimate.

## Summary

- One integer, `ENGINE_VERSION`, for the whole engine: charge, surface, QTAIM, native
  converter, and DDEC6 when it merges. An `ENGINE_CHANGES` table says per version which
  routines were added or changed, by which sub-engine, and why.
- Each job folder gets `provenance.json`: per routine, what produced it (an engine or
  Multiwfn or ORCA), at which engine version, with which settings, plus permanent skips,
  the wavefunction converter and the engines' own run records (the charge engine's `_meta`,
  including MBIS selection). `json-to-lmdb` converts it to `provenance.lmdb`.
- `timings.json` goes back to timings only. The engine-only underscore keys move to
  provenance; a one-shot migration script rewrites the few engine trees that carry them.
  No dual-read code stays in the engine.
- A stored routine is current when its recorded version is at least the last version that
  added or changed it. That one rule replaces `engine_routines_at`, the surface routine
  record and both stamp checks, and closes the gap that a value-changing release could not
  force a recompute.

## 1. What exists today [V]

| record | where it lives | written by | read by |
|---|---|---|---|
| `_engine_versions` `{charge_engine: N, surface_engine: N}` | timings.json | `omol._stamp_engine_version` (omol.py:1748), called at 1881 and 2009; popped on a partial run at 1877 | `charge_engine_version`, `engine_routines_of`, `surface_engine_routines_of` (validation.py:47-82) |
| `CHARGE_ENGINE_VERSION = 3`, `CHARGE_ENGINE_ROUTINES_ADDED`, `engine_routines_at` | data/multiwfn.py:24-37 | - | validation.py:65, 82 |
| `_surface_engine_routines` | timings.json | omol.py:2010 | `surface_engine_routines_of` (validation.py:68), restart at omol.py:1969, compare_engine_merge.py:273 |
| `_converter` | timings.json | run_jobs (omol.py:858) | `validate_timing_dict` (validation.py:356), the wrong-converter discard (omol.py:2567-2605) |
| `_skipped_routines` | timings.json | `_record_skipped` (omol.py:1760), from the charge engine (1852) and Multiwfn limits (3660) | `skipped_routines` (validation.py:33) at validation.py:389, 1785 and omol.py:1438, 1732 |
| `_timings_patched` | timings.json | `patch_timings_from_log` | validation. **On `main` since 2026-04 (0d3b9e5); describes the timing values. Stays in timings.json.** |
| `QTAIM_ENGINE_VERSION = 1` | qtaim.json `_meta.qtaim_engine.version` | cp_search.py:501, 681 | qtaim.out banner (697) |
| `ORCA_PARSER_VERSION` | orca.json | parse_orca.py | converter.py gate, lmdbs.py, reparse helpers |
| charge engine `_meta` (grid, nelec, MBIS cycles/selection/degenerate, skips, stage timings) | not written on feat/charge-engine; `charge_engine_meta.json` on `perf/mbis-converged` (0a12ec9, unmerged) | charge_engine.py:1307 | audit_prod on that branch |

Consumers outside the engine: recheck_fuzzy_folders.py:71-109, compare_engine_merge.py:194,
273, scripts/nersc/audit_prod.py:192. `timings.json` enters `timings.lmdb` whole, underscore
keys included (lmdbs.py JSON_DATA_TYPES).

Trees that carry the engine keys [I, from the runs on record]: NERSC openac `prod_l1` (4235
jobs) and `prod_l1_santi` (179), the current NERSC level 1/2 eval, the LRC `engine_test`
trees. `main` never wrote them.

## 2. Design

### 2a. Versions (`qtaim_gen/source/data/engine_versions.py`, new)

```python
ENGINE_VERSION = 4
ENGINE_CHANGES = {
    1: {"added": {...charge and surface routines...}, "changed": set(), "engines": {"charge_engine", "surface_engine"}, "note": "..."},
    2: {"added": {"elf_fuzzy", "grad_norm_rho_fuzzy", "laplacian_rho_fuzzy"}, ...},
    3: {"added": {"other_esp"}, ...},
    4: {"added": set(), "changed": set(), "note": "single engine version, provenance.json"},
}
def last_change(routine) -> int      # max version whose added|changed holds the routine
def is_current(routine, version) -> bool   # version >= last_change(routine)
```

- Versions 1-3 are the old charge/surface numbers, which always moved together. The QTAIM
  engine (version 1, shipped 2026-10-16) and the native converter (2026-11-03) predate the
  first stamp (version 2, 77b1565), so they enter under version 1 [V].
- `ROUTINE_ENGINE` maps each engine routine to its sub-engine (charge_engine,
  surface_engine, qtaim_engine, converter); that is the `by` of its provenance entry.
- `ENGINE_ROUTINES`, the level sets and `DERIV_FUZZY_FAMILY` stay in data/multiwfn.py;
  `CHARGE_ENGINE_VERSION`, `CHARGE_ENGINE_ROUTINES_ADDED` and `engine_routines_at` go.
- Rule for the future: a change that alters stored values of a routine adds it to
  `changed` under a new version; a new routine goes in `added`. MBIS (`perf/mbis-kernel`)
  and DDEC6 add their own entries when they rebase.

### 2b. provenance.json

```json
{
  "schema": 1,
  "engine_version": 4,
  "converter": {"by": "native", "mode": "fixed", "engine_version": 4},
  "routines": {
    "hirshfeld": {"by": "charge_engine", "engine_version": 4,
                  "spec": {"grid": [75, 434, 10.0], "proatoms": "multiwfn_atmraddens", "edf": "multiwfn_edf"}},
    "mbis":      {"by": "charge_engine", "engine_version": 4, "spec": {"mode": "converged", "guess": "fitted"}},
    "other_esp": {"by": "surface_engine", "engine_version": 4, "spec": {"spacing_bohr": 0.25, "boys": "exact"}},
    "qtaim":     {"by": "qtaim_engine", "engine_version": 4, "spec": {"search": "engine"}},
    "chelpg":    {"by": "multiwfn"},
    "orca":      {"by": "orca", "parser_version": 2}
  },
  "skipped": {"mbis": {"reason": "...", "by": "charge_engine"}},
  "engines": {"charge_engine": {"...": "the _meta dict: grid, nelec, mbis_selection, ..."},
              "surface_engine": {"...": "grid and V/E/F per surface"}}
}
```

- `routines` holds an entry for every routine the folder has, including Multiwfn's (no
  version) and ORCA's (parser version), so one file answers "who produced this value".
- `spec` holds the settings that change values and are not implied by the version (grid,
  MBIS mode, Boys mode, converter mode). Small, explicit dicts, not a hash, so the LMDB
  stays readable.
- `engines` holds the engines' run records (decision: the charge engine `_meta` folds in
  here, so `charge_engine_meta.json` is not created).
- QTAIM keeps its detailed `_meta` in qtaim.json (CP counts, Poincare-Hopf, reseeds); the
  provenance entry only records who and which version.

### 2c. Library (`qtaim_gen/source/utils/provenance.py`, new)

- `load(folder)` (generator/ first, then root, as timings are read today),
  `record_routines(folder, by, routines, spec, meta=None)`, `record_skip(...)`,
  `record_converter(...)`, `drop(folder, routines)` (partial run, discard).
- `engine_current(folder_or_prov, routine)`: by an engine and `is_current`.
- Writes are atomic (atomic_json_write) and merge per routine.

### 2d. Pipeline changes

| place | change |
|---|---|
| run_jobs (omol.py:858) | converter into provenance instead of `_converter`; Multiwfn routines recorded `by: multiwfn` as they finish |
| `_run_charge_engine` (omol.py:1809-1885) | `record_routines` + `_meta` into `engines.charge_engine`; partial run drops its entries instead of popping the stamp |
| `_run_surface_engine` (omol.py:1945) | per-routine records replace `_surface_engine_routines` and the stamp; restart computes the routines not `engine_current`; **review #2**: ALIE and planarity are written before ESP runs in its own try, an ESP failure is a permanent skip with the exception text |
| QTAIM engine path | record `qtaim` by `qtaim_engine`; `QTAIM_ENGINE_VERSION` becomes ENGINE_VERSION in the `_meta` too |
| `_run_orca_parse` | record `orca` with ORCA_PARSER_VERSION |
| `_record_skipped` | into provenance `skipped` |
| move_results_to_folder (omol.py:1498) | provenance.json joins the moved files, merged **per routine** (the current merge is shallow `{**old, **new}`, which would drop entries) |
| wrong-converter discard (omol.py:2567-2605) | drop the wavefunction-derived routine entries and the converter, keep ORCA's |
| `_engine_outputs_present` (omol.py:1715) | `engine_current` per wanted routine |
| validate_timing_dict (validation.py:340-400) | expected Multiwfn timings only for routines provenance says Multiwfn produced; engine routines must be `engine_current`; converter check from provenance; skips from provenance |
| patch_timings (omol.py:1433-1438) | engine routines and skips from provenance |
| recheck_fuzzy_folders, compare_engine_merge, audit_prod | read provenance |
| json-to-lmdb | new data type `provenance` -> provenance.lmdb (same key formula, so it joins) |

**Item 5** (reuse the density of the derivative pass at full_set >= 1): measured first. If
the charges and fuzzy integrals stay bit-identical it ships under version 4 with no
`changed` entry; if not, the affected routines go in `changed`.

### 2e. Migration (`qtaim_gen/source/scripts/helpers/migrate_provenance.py`, new)

- Per folder: read `_engine_versions`, `_surface_engine_routines`, `_converter`,
  `_skipped_routines` and the qtaim `_meta`; write provenance.json with the same meaning the
  current readers give them; strip those keys from timings.json (generator/ and root).
- Dry run by default, a TSV report, idempotent, never touches `_timings_patched` or data
  JSONs. Run on the engine trees listed in section 1 after the merge and before their next
  restart.

## 3. Phases [E]

| phase | content | effort |
|---|---|---|
| P1 | engine_versions.py, provenance.py, unit tests | 0.5 day |
| P2 | writers (converter, charge, surface incl. review #2, QTAIM, ORCA, Multiwfn, skips), move/discard merge | 1 day |
| P3 | readers (restart, validation, patch_timings, recheck, compare, audit) | 1 day |
| P4 | provenance.lmdb in json-to-lmdb | 0.5 day |
| P5 | migration script, run on local test trees and data/engine_test_2026-11 | 0.5 day |
| P6 | item 5 measurement | 0.5 day |
| P7 | docs: README engine-version section, ledger "code" column, CLAUDE.md job-folder layout | 0.5 day |

About 4-5 days, plus the full suite and a local end-to-end run (full runner on a few
fixtures, restart, validation). Cluster checks: none needed beyond running the migration
on the engine trees and one restart there, since no values change.

## 4. Decisions (2026-11-11)

1. Multiwfn-produced routines get entries (`by: multiwfn`, no engine version) with the
   Multiwfn build: the binary path and the version/date line of its banner.
2. Legacy history is imprecise [I]: the stamp only exists since 2026-11-03 (77b1565), so
   earlier value changes have no version, e.g. converged MBIS as the default (2026-10-15)
   and planarity in the engine (2026-11-02). The migration keeps the stamps as found, marks
   the entries `"legacy": true` and does not reconstruct history.
3. Review #2: an ESP failure is retried once on the next pass (counted in provenance), then
   recorded as a permanent skip.
4. Handoff notes for `perf/mbis-kernel` (drop `charge_engine_meta.json`, write into
   `engines.charge_engine`, add a `changed: mbis` entry) and `feat/engine-ddec6` (records +
   its own `added` entry) are written when this branch merges.

## Caveats

- Effort figures are judgment.
- The list of trees carrying engine keys comes from the runs recorded in the ledger; a scan
  of each tree with the migration's dry run is the check.
