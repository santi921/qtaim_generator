---
title: "fix: timings.json path mismatch when move_results=True"
type: fix
date: 2026-04-01
---

# fix: timings.json path mismatch when move_results=True

## Problem Statement

When `move_results=True`, the code in `run_jobs()` sets `folder_check = folder/generator` at initialization (line 573-576). This path is used to read and write `timings.json` for checkpoint/restart logic. However, the `generator/` subfolder **does not exist** until `move_results_to_folder()` runs at the very end of `gbw_analysis()` (line 1622).

### Why it appears to work on first runs

On first run (`restart=False`), the `else` branch at line 648 overwrites `folder_check = folder`, so timings are written to the **job root**. Then `move_results_to_folder()` moves them into `generator/`. So first-run completes fine.

### Where it breaks: crash-then-restart

1. Job crashes mid-`run_jobs()` — timings are in **job root** (written via the `else` branch)
2. `move_results_to_folder()` never ran (crashed before reaching it)
3. On restart: line 573 sets `folder_check = generator/`, line 578 checks there — **file not found**
4. `timings` dict stays empty, restart check at `gbw_analysis()` line 1466 fails → `restart = False`
5. All Multiwfn steps rerun from scratch

### Additional risks

- `atomic_json_write()` uses `tempfile.NamedTemporaryFile(dir=dir_name)` which raises `FileNotFoundError` if `generator/` doesn't exist. The try/except at line 688 catches this silently.
- `_run_orca_parse()` line 1266 also writes to `generator/timings.json` which may not exist yet.
- `memory.json` at line 686 has the same bug pattern.

## Affected Locations

All in `qtaim_gen/source/core/omol.py`:

| Location | Line(s) | Operation | Issue |
|----------|---------|-----------|-------|
| `run_jobs()` init | 573-576 | Sets `folder_check` | Points to non-existent `generator/` |
| `run_jobs()` read | 578-583 | READ timings | Reads from `generator/` — misses job root |
| `run_jobs()` write | 682 | WRITE timings | Writes to `folder_check` (wrong on restart path) |
| `run_jobs()` write | 686 | WRITE memory.json | Same `folder_check` bug |
| `run_jobs()` else | 648 | Overwrite `folder_check` | Masks the bug on first run |
| `gbw_analysis()` restart | 1459-1468 | READ timings | Only checks `generator/`, not job root |
| `_run_orca_parse()` | 1265-1268 | WRITE timings | Writes to `generator/` which may not exist |

**Not affected** (already safe):
- `validation.py:776` (`get_information_from_job_folder`) — checks for `generator/` existence first
- `validation.py:640` (`validation_checks`) — called after `move_results_to_folder()`
- `io.py:563` (`check_results_exist`) — proper conditional path
- `workflow.py:117,336` — uses job root directly

## Proposed Solution

**Principle:** Always write timings to **job root** during processing. On restart, check **both** locations (`generator/` first for completed runs, job root for interrupted runs). Let `move_results_to_folder()` handle relocation as it already does.

### Change 1: `run_jobs()` — timings read path (lines 571-576)

Replace the `folder_check` initialization with dual-location lookup:

```python
timings = {}
# During processing, always write timings to job root.
# move_results_to_folder() relocates to generator/ at the end.
# On restart, check generator/ first (previous completed run),
# then job root (interrupted run).
timings_read_path = os.path.join(folder, "generator", "timings.json")
if not os.path.exists(timings_read_path):
    timings_read_path = os.path.join(folder, "timings.json")
```

Update lines 578-583 to use `timings_read_path` instead of `os.path.join(folder_check, "timings.json")`.

### Change 2: `run_jobs()` — validation folder for restart (line 590-591)

`get_val_breakdown_from_folder` needs the folder where result JSONs live. On restart after a crash, they're in job root; after a completed run, in `generator/`. Use the same dual-check:

```python
val_folder = os.path.join(folder, "generator")
if not os.path.isdir(val_folder):
    val_folder = folder
dict_val = get_val_breakdown_from_folder(
    val_folder, n_atoms=n_atoms, full_set=full_set, spin_tf=spin_tf
)
```

### Change 3: `run_jobs()` — timings write path (line 682, 686)

Always write to job root:

```python
atomic_json_write(os.path.join(folder, "timings.json"), timings)
# ...
atomic_json_write(os.path.join(folder, "memory.json"), memory)
```

### Change 4: `run_jobs()` — remove the `else: folder_check = folder` (line 648)

This line was a workaround that masked the bug on first runs. With the write path always targeting job root, it's no longer needed. Remove it.

### Change 5: `gbw_analysis()` — restart check (lines 1459-1468)

Check both locations:

```python
if restart:
    gen_timings = os.path.join(folder, "generator", "timings.json")
    root_timings = os.path.join(folder, "timings.json")

    if os.path.exists(gen_timings) and os.path.getsize(gen_timings) > 0:
        timings_path = gen_timings
    elif os.path.exists(root_timings) and os.path.getsize(root_timings) > 0:
        timings_path = root_timings
    else:
        timings_path = None

    if timings_path is None:
        logger.warning("No timings file found - starting from scratch!")
        restart = False
    else:
        logger.info("Timings file found at %s - restarting from last step.", timings_path)
```

### Change 6: `_run_orca_parse()` — timings write (lines 1265-1274)

Always write to job root, read from either location:

```python
# Read from wherever timings currently lives
gen_timings = os.path.join(folder, "generator", "timings.json")
root_timings = os.path.join(folder, "timings.json")
if os.path.isfile(gen_timings) and os.path.getsize(gen_timings) > 0:
    timings_path = gen_timings
elif os.path.isfile(root_timings) and os.path.getsize(root_timings) > 0:
    timings_path = root_timings
else:
    timings_path = None

if timings_path is not None:
    with open(timings_path, "r") as f:
        timings = json.load(f)
    timings["orca_parse"] = elapsed
    # Always write back to job root
    atomic_json_write(root_timings, timings)
```

## Acceptance Criteria

- [ ] On first run with `move_results=True`, `timings.json` is written to job root during processing
- [ ] `move_results_to_folder()` moves `timings.json` from root to `generator/` at the end (already works)
- [ ] On restart after crash (timings in root, no `generator/` dir), restart logic finds timings and skips completed steps
- [ ] On restart after completed run (timings in `generator/`), restart logic still works
- [ ] `_run_orca_parse()` can write timing regardless of whether `generator/` exists
- [ ] `memory.json` follows the same fix pattern
- [ ] All existing tests pass
- [ ] No changes needed outside `omol.py`

## References

- `omol.py` `run_jobs()`: lines 565-690
- `omol.py` `gbw_analysis()`: lines 1380-1640
- `omol.py` `_run_orca_parse()`: lines 1195-1284
- `omol.py` `move_results_to_folder()`: lines 1095-1155
- `atomic_write.py` `atomic_json_write()`: lines 9-22
