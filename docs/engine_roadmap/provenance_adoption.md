# Rebasing onto provenance: notes for the MBIS and DDEC6 branches

Date: 2026-11-11. For the agents working on `perf/mbis-kernel` and `feat/engine-ddec6`.
Read `provenance_plan.md` (design and decisions) and the "Engine versions and provenance"
section of `README.md` first.

## What changed (branch `feat/engine-provenance`, PR into `feat/charge-engine`)

- One `ENGINE_VERSION` (now 4) in `qtaim_gen/source/data/engine_versions.py`, with
  `ENGINE_CHANGES = {version: {"added": frozenset, "changed": frozenset, "note": str}}` and
  `ROUTINE_ENGINE = {routine: sub-engine}`. A stored routine is current when its recorded
  version is at least the last version that added or changed it (`is_current`).
- Per job folder `provenance.json` (`qtaim_gen/source/utils/provenance.py`): `routines`
  (who produced each one, engine version, settings), `converter`, `skipped`, `engines` (each
  engine's run record). Moved into `generator/` and merged entry by entry by
  `move_results_to_folder`; `json-to-lmdb` makes `provenance.lmdb`.
- `timings.json` holds timings only (plus `_timings_patched`).

Removed, with their replacements:

| removed | use instead |
|---|---|
| `CHARGE_ENGINE_VERSION`, `CHARGE_ENGINE_ROUTINES_ADDED`, `engine_routines_at` (data/multiwfn.py) | `ENGINE_VERSION`, `ENGINE_CHANGES`, `routines_at`, `is_current` (data/engine_versions.py) |
| `TIMINGS_ENGINE_VERSIONS_KEY` / timings `_engine_versions` | entries in provenance `routines` with `engine_version` |
| `TIMINGS_SKIPPED_KEY` / timings `_skipped_routines`, `skipped_routines(timings)` | `provenance.record_skips(folder, {r: {"reason", "by"}})`, `provenance.skipped(provenance.load(folder))`; `omol._record_skipped` still exists and writes there |
| `CONVERTER_KEY` / timings `_converter` | `provenance.record_converter`, `provenance.converter_tag(prov)` |
| `TIMINGS_SURFACE_ROUTINES_KEY`, `surface_engine_routines_of` | per-routine provenance entries |
| `charge_engine_version`, `engine_routines_of` (utils/validation.py) | `provenance.engine_routines(prov)`, `provenance.engine_current(prov, routine)` |
| `omol._stamp_engine_version` | `provenance.record_routines(folder, routines, by, ENGINE_VERSION, spec=...)` |
| `cp_search.QTAIM_ENGINE_VERSION` | `ENGINE_VERSION` |

Rule for a version: a new routine goes in `added` under a new version; a change to stored
values of a routine goes in `changed`. Both branches will add a version; whichever merges
second renumbers to the next free version.

## perf/mbis-kernel

- Drop `charge_engine_meta.json`. `_run_charge_engine` on the provenance branch already writes
  the charge engine's `_meta` to `provenance.json` `engines.charge_engine` (MBIS selection,
  cycles, merged shells, `mbis_unique`, `mbis_degenerate`, nelec, grid, stage timings).
  `_COMPILED_JSONS` no longer needs it.
- `scripts/nersc/audit_prod.py`: read the MBIS record from
  `provenance.load(folder)["engines"]["charge_engine"]` and skips from
  `provenance.load(folder)["skipped"]` (the provenance branch already reads skips and the
  engine version that way; your MBIS table is the part to port).
- The MBIS selection, objective and shell-merge changes alter stored values: add a version
  with `"changed": frozenset(MBIS_FAMILY)` and a note. Older folders then recompute MBIS on
  restart. If the density-reuse change ("item 5", reuse `density_derivatives` rho at full_set
  >= 1, not bit-identical: up to 1.7e-15 rel) is wanted, this is the version to ship it in,
  with the charge routines it touches in `changed`.
- Tests that wrote or read `charge_engine_meta.json` (`test_audit_prod.py`,
  `test_charge_engine.py`, `test_engine_subprocess.py`) move to `provenance.json`. The
  provenance branch already strips `timings_s` from both sides when it compares child and
  in-process `provenance.json` in `test_engine_subprocess.py`.
- `docs/engine_roadmap/validation_ledger.md`: your "7b runs in progress" section and the
  provenance branch's "Runs in progress" section overlap; keep one.

## feat/engine-ddec6

- `_run_ddec6`: replace the `_engine_versions.ddec6` write with
  `provenance.record_routines(folder, ["ddec6"], "ddec6", ENGINE_VERSION, spec=...)` and the
  run record with `provenance.record_engine(folder, "ddec6", out["_meta"])` if you want it
  in provenance. Read skips with `provenance.skipped(provenance.load(folder))` instead of
  `skipped_routines(timings)`; `_record_skipped` already writes to provenance.
- Restart check: `provenance.engine_current(provenance.load(folder), "ddec6")` together with
  the result file, so a later version that changes DDEC6 recomputes it.
- `data/engine_versions.py`: `ROUTINE_ENGINE["ddec6"] = "ddec6"` and a new version with
  `"added": frozenset({"ddec6"})`. `DDEC6_VERSION` can go (one number) or stay as an
  internal setting recorded in the `spec`.
- `validate_ddec6` reads the skip from provenance; the timing check can stay.
- Tests asserting `t["_engine_versions"]["ddec6"]` or `TIMINGS_SKIPPED_KEY` move to
  provenance.

## Trees and running work

- Old code (any branch without provenance) and new code read different records. Old code on
  a migrated folder, or new code on an unmigrated engine folder, does not see the engine
  record: under `--native_convert` (the default) it discards `generator/` and recomputes the
  folder from the gbw; without it, it recomputes the engine routines it no longer sees.
- Migrated so far (do not run old code there): local `data/actinide_dev/*` (173 folders), LRC
  `engine_test/fz_l1` and `fz_l2`, and LRC `engine_test/merge`, `scratch`, `scratch_l1*`,
  `ref_l2` once their apply finishes. Not migrated: LRC `nc_*` (converter validation, running
  on old code), all of NERSC.
- Before running rebased code on an existing engine tree, migrate it:
  `python -m qtaim_gen.source.scripts.helpers.migrate_provenance --root <tree>` (dry run),
  then `--apply`. It skips locked folders, so stop or let finish any job on the tree first.
- Do not rebase a branch while a campaign on that branch is still running on old code; finish
  or stop it, rebase, migrate its trees, then resume.
- Direct calls (`charge_engine.run`, `surface_engine`, `ddec6.analyze` on a wfx, the
  Chargemol harness) do not touch these records and are unaffected.
