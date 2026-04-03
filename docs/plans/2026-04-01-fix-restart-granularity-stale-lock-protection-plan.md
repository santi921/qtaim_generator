---
title: "fix: Per-Sub-Job Restart Granularity + Stale Lock File Protection"
type: fix
date: 2026-04-01
revised: 2026-04-01
---

# Per-Sub-Job Restart Granularity + Stale Lock File Protection

## Overview

Two related improvements to the Parsl job runner pipeline:

1. **Per-sub-job restart granularity**: Change `run_jobs()` skip logic from category-level (`val_charge`) to per-sub-job (`timings["adch"] > 0`), so only failed/missing sub-jobs re-run on restart.
2. **Stale lock file protection**: Wire `acquire_lock`/`release_lock` into `process_folder`/`process_folder_alcf` with mtime-based stale detection and heartbeat, so crashed workers' locks are automatically broken.

## Problem Statement

### Restart Granularity

In `run_jobs()` ([omol.py:600-652](qtaim_gen/source/core/omol.py#L600-L652)), the restart skip logic checks category-level validation:

```python
elif order in charge_dict.keys():
    if dict_val.get("val_charge", False):  # checks ALL of charge.json
        continue  # skips ALL charge sub-jobs
```

If `adch` fails but `hirshfeld`, `cm5`, `becke` all succeed, `val_charge=False` because `charge.json` is incomplete. **All 4 charge sub-jobs re-run**, wasting 75% of the work.

`timings.json` already tracks per-sub-job completion (`{"hirshfeld": 3.2, "adch": -1, "cm5": 2.8, ...}`), but the skip logic doesn't use this granularity.

Additionally, the restart path calls `get_val_breakdown_from_folder()` which opens and parses **6 JSON files** just to make skip decisions. With per-sub-job timing checks, this becomes unnecessary -- a single `timings.json` read suffices.

### Stale Lock Files

`acquire_lock`/`release_lock` exist in [workflow.py:28-44](qtaim_gen/source/core/workflow.py#L28-L44) but are **never called**. When a Parsl worker crashes or times out, there is no protection against concurrent processing and no stale detection.

## Proposed Solution

### Per-Sub-Job Restart

Replace the 50+ lines of category-level validation checks with a single per-sub-job timing check:

```python
# BEFORE (category-level -- 50+ lines)
elif order in charge_dict.keys():
    if dict_val.get("val_charge", False):
        continue

# AFTER (per-sub-job -- 3 lines)
if restart and order in timings and timings[order] > 0:
    logger.info(f"Skipping {order}: timing={timings[order]:.2f}s")
    continue
```

**Skip criterion**: `timings[order] > 0` (positive = success, `-1` = error, absent = never ran).

Remove the `get_val_breakdown_from_folder()` call from the restart path (saves 6 file reads per folder). Category-level `validation_checks()` remains as the **final sanity check** in `gbw_analysis()`.

### Stale Lock Protection

**Design**: mtime-based staleness with `O_EXCL` atomicity and heartbeat.

- Lock file contains bare PID (for debugging via `cat`), no JSON
- Staleness determined by filesystem `mtime` (age > 8 hours = stale)
- `O_CREAT | O_EXCL` preserved for race-free acquisition
- Stale lock broken via `os.remove()` then atomic re-create
- **Heartbeat**: `os.utime(lockfile)` after each sub-job completes in `run_jobs()`, keeping mtime fresh for long-running folders

**Why mtime, not JSON with PID/hostname**: On multi-node HPC (ALCF Polaris), PIDs are node-local so `os.kill(pid, 0)` only works on the originating node. Cross-node falls back to age-based detection anyway. Using filesystem mtime is simpler (15 lines vs 60+), works identically across nodes, and avoids dropping `O_EXCL` atomicity.

**Why 8-hour timeout**: Some individual Multiwfn sub-jobs (QTAIM on large molecules) can take up to 8 hours. The heartbeat after each sub-job keeps the lock fresh, so the timeout only triggers if the process truly died without completing any sub-job for 8+ hours.

## Technical Considerations

### `clean_first` + Lock Interaction

Lock acquired **before** cleanup. Cleanup excludes `.processing.lock`:

```python
if item not in ("gbw_analysis.log", ".processing.lock"):
    # ... delete item
```

### Sub-Job Skipped But Category Validation Fails

If all charge sub-jobs have positive timings but `charge.json` is corrupted:
1. Sub-jobs skipped (positive timings)
2. `parse_multiwfn()` re-parses from `.out` files if they exist
3. Final `validation_checks()` fails → logged as warning
4. User re-runs with `--overwrite` or `--clean_first`

### Phantom Keys at Runtime

`charge_separate`, `bond_separate`, `other_separate`, `fuzzy_full` have no `.mfwn` files and must not appear in the `run_jobs()` execution list. However, `create_jobs()` depends on them to expand the individual sub-job `.mfwn` files, so they cannot be removed from `ORDER_OF_OPERATIONS_separate`. Fix: filter them at runtime inside `run_jobs()` only.

### `separate=False` + `restart=True` NameError

`charge_dict`, `bond_dict`, `fuzzy_dict`, `other_dict`, `spin_tf` only defined in `if separate:` block. Fix: call `check_spin(folder)` unconditionally, initialize empty dicts in `else` branch. With per-sub-job timing check, these dicts are no longer consulted for skip decisions anyway, but they must exist to avoid NameError.

### Corrupted `timings.json`

Wrap `json.load()` with `JSONDecodeError` handling. Log warning, treat as empty dict, proceed as fresh start. No backup needed -- `atomic_json_write` already prevents corruption under normal conditions, and timing data is re-derived from execution.

## Acceptance Criteria

### Per-Sub-Job Restart

- [ ] Sub-jobs with `timings[order] > 0` are skipped on restart
- [ ] Sub-jobs with `timings[order] == -1` (error) are re-run
- [ ] Sub-jobs absent from `timings.json` are run
- [ ] `get_val_breakdown_from_folder()` no longer called on restart path
- [ ] Phantom keys (`charge_separate`, `bond_separate`, `other_separate`, `fuzzy_full`) filtered at runtime in `run_jobs()` — constant left intact for `create_jobs()`
- [ ] `separate=False, restart=True` does not raise `NameError`
- [ ] Corrupted `timings.json` handled gracefully (log + fresh start)
- [ ] Category-level `validation_checks()` still runs as final sanity check

### Stale Lock Protection

- [ ] `acquire_lock` uses `O_CREAT | O_EXCL` for atomic creation
- [ ] Lock file contains bare PID for debugging
- [ ] Stale detection via mtime age > 8 hours
- [ ] Stale lock broken via `os.remove()` then atomic re-create
- [ ] Heartbeat: `os.utime(lockfile)` after each sub-job in `run_jobs()`
- [ ] `release_lock` in `finally` block of `process_folder`/`process_folder_alcf`
- [ ] Non-stale lock → folder skipped with `status="skipped"`
- [ ] `clean_first=True` does not delete `.processing.lock`

### Tests

- [ ] Per-sub-job skip with partial timings (only missing/errored re-run)
- [ ] Timing value -1 triggers re-run
- [ ] Absent timing key triggers run
- [ ] Stale lock detection (mtime > threshold)
- [ ] Non-stale lock prevents processing
- [ ] Lock acquire/release lifecycle
- [ ] `clean_first` preserves `.processing.lock`
- [ ] Corrupted `timings.json` → fresh start
- [ ] `separate=False, restart=True` no NameError
- [ ] Phantom keys absent from the operations list passed to `run_jobs()` (filtered, not removed from constant)
- [ ] Lock heartbeat updates mtime

## Dependencies & Risks

**Risk: Parsl `SIGKILL` bypasses `finally` block.** The `finally` block won't run, leaving stale locks. **Mitigation**: 8-hour mtime-based stale detection handles this.

**Risk: Single sub-job takes >8 hours.** The heartbeat only fires between sub-jobs. A single sub-job running >8 hours would appear stale. **Mitigation**: 8 hours is the confirmed upper bound for any single sub-job. If this changes, increase the constant.

**Risk: Shared filesystem mtime granularity.** Lustre/GPFS mtime has second-level granularity, which is more than sufficient for an 8-hour threshold.

## MVP

### Phase A: Per-sub-job restart + cleanup

#### [omol.py](qtaim_gen/source/core/omol.py) - Phantom key filter in `run_jobs()`

`ORDER_OF_OPERATIONS_separate` is left intact — `create_jobs()` uses `charge_separate`, `bond_separate`, `other_separate`, `fuzzy_full` to expand sub-job `.mfwn` files. Filter only at runtime in `run_jobs()`:

```python
_phantom_keys = {"charge_separate", "bond_separate", "other_separate", "fuzzy_full"}
order_of_operations = [o for o in order_of_operations if o not in _phantom_keys]
```

#### [omol.py](qtaim_gen/source/core/omol.py) - `run_jobs()` restart refactor

```python
# Lines 502-528: Fix separate=False, extract spin_tf unconditionally
spin_tf = check_spin(folder)

if separate:
    order_of_operations = ORDER_OF_OPERATIONS_separate.copy()
    charge_dict = charge_data_dict(full_set=full_set)
    bond_dict = bond_order_dict(full_set=full_set)
    fuzzy_dict = fuzzy_data(spin=spin_tf, full_set=full_set)
    other_dict = other_data_dict(full_set=full_set)
    [order_of_operations.append(i) for i in charge_dict.keys()]
    [order_of_operations.append(i) for i in bond_dict.keys()]
    [order_of_operations.append(i) for i in fuzzy_dict.keys()]
    [order_of_operations.append(i) for i in other_dict.keys()]
else:
    order_of_operations = ORDER_OF_OPERATIONS
    charge_dict = {}
    bond_dict = {}
    fuzzy_dict = {}
    other_dict = {}
```

```python
# Lines 582-587: Corrupted timings.json handling
if os.path.exists(timings_read_path):
    if os.path.getsize(timings_read_path) > 0:
        try:
            with open(timings_read_path, "r") as f:
                timings = json.load(f)
        except json.JSONDecodeError:
            logger.warning(
                "Corrupted timings.json at %s -- starting fresh",
                timings_read_path,
            )
            timings = {}
```

```python
# Lines 589-652: Replace entire category-level skip block with per-sub-job
# REMOVE: the get_charge_spin_n_atoms_from_folder + get_val_breakdown_from_folder calls
# REMOVE: the 50+ lines of category-level elif chains

# REPLACE with:
for order in order_of_operations:
    if restart and order in timings and timings[order] > 0:
        logger.info(f"Skipping {order} in {folder}: timing={timings[order]:.2f}s")
        continue

    # ... existing sub-job execution code (mfwn_file, subprocess.run, etc.) ...

    # After sub-job completes and timing is recorded:
    timings[order] = end - start
    # Heartbeat: touch lock file to keep mtime fresh
    lockfile = os.path.join(folder, ".processing.lock")
    if os.path.exists(lockfile):
        try:
            os.utime(lockfile, None)
        except OSError:
            pass
```

### Phase B: Lock wiring

#### [workflow.py](qtaim_gen/source/core/workflow.py) - Lock functions

```python
import time

_LOCK_MAX_AGE_S = 28800  # 8 hours

def acquire_lock(folder: str, max_age_s: float = _LOCK_MAX_AGE_S) -> bool:
    """Acquire a processing lock on a folder.

    Uses mtime-based stale detection. If an existing lock's mtime
    is older than max_age_s, it is considered stale and broken.
    Atomic O_CREAT|O_EXCL prevents races between concurrent workers.
    """
    lockfile = os.path.join(folder, ".processing.lock")

    if os.path.exists(lockfile):
        try:
            age = time.time() - os.path.getmtime(lockfile)
        except OSError:
            age = float("inf")  # Can't stat → treat as stale

        if age < max_age_s:
            return False  # Not stale, genuinely locked

        # Stale → break it
        logging.getLogger("lock").warning(
            "Breaking stale lock in %s (age=%.0fs, threshold=%.0fs)",
            folder, age, max_age_s,
        )
        try:
            os.remove(lockfile)
        except FileNotFoundError:
            pass  # Another worker already broke it

    # Atomic create-or-fail
    try:
        fd = os.open(lockfile, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode())  # PID for debugging only
        os.close(fd)
        return True
    except FileExistsError:
        return False  # Lost race to another worker


def release_lock(folder: str) -> None:
    """Release processing lock. Safe to call even if lock doesn't exist."""
    lockfile = os.path.join(folder, ".processing.lock")
    try:
        os.remove(lockfile)
    except FileNotFoundError:
        pass
```

#### [workflow.py](qtaim_gen/source/core/workflow.py) - Wire into `process_folder`

```python
def process_folder(folder, ...):
    result = {"folder": folder, "status": "unknown", "elapsed": None, "error": None}
    folder = os.path.abspath(folder)
    logger = setup_logger_for_folder(folder)

    if not acquire_lock(folder):
        logger.info("Skipping %s: folder locked by active process", folder)
        result["status"] = "skipped"
        result["error"] = "folder locked by active process"
        return result

    orig_cwd = os.getcwd()
    try:
        os.chdir(folder)
        if clean_first:
            for item in os.listdir(folder):
                if item not in ("gbw_analysis.log", ".processing.lock"):
                    # ... existing cleanup ...
        # ... rest of existing logic ...
    except Exception as exc:
        # ... existing error handling ...
    finally:
        release_lock(folder)
        try:
            os.chdir(orig_cwd)
        except Exception:
            print(f"Warning: failed to restore cwd to {orig_cwd}")
```

#### [workflow.py](qtaim_gen/source/core/workflow.py) - Wire into `process_folder_alcf`

Same pattern on `folder_outputs`: acquire before any work (including `clean_first`), release in `finally`. Exclude `.processing.lock` from cleanup.

### Tests (inline with each phase)

#### Per-sub-job restart tests

```python
class TestPerSubJobRestart:
    def test_positive_timing_skips_subjob(self):
        """Sub-job with timing > 0 is skipped on restart."""

    def test_error_timing_reruns_subjob(self):
        """Sub-job with timing == -1 is re-run on restart."""

    def test_missing_timing_runs_subjob(self):
        """Sub-job absent from timings is run on restart."""

    def test_partial_charge_only_reruns_failed(self):
        """If adch=-1 but hirshfeld/cm5/becke >0, only adch re-runs."""

    def test_phantom_keys_removed_from_operations(self):
        """ORDER_OF_OPERATIONS_separate has no phantom keys."""

    def test_separate_false_restart_no_nameerror(self):
        """separate=False + restart=True doesn't raise NameError."""

    def test_corrupted_timings_fresh_start(self):
        """Corrupted timings.json → log warning, start fresh."""

    def test_heartbeat_updates_lock_mtime(self):
        """Lock file mtime advances after sub-job completion."""
```

#### Lock tests

```python
class TestLockProtection:
    def test_acquire_release_lifecycle(self):
        """Lock can be acquired and released."""

    def test_lock_contains_pid(self):
        """Lock file contains current PID."""

    def test_double_acquire_fails(self):
        """Second acquire_lock returns False when lock is active."""

    def test_stale_lock_broken_by_age(self):
        """Lock with mtime > threshold is broken and reacquired."""

    def test_fresh_lock_not_broken(self):
        """Lock within age threshold is respected."""

    def test_concurrent_stale_break_race(self):
        """If two workers break the same stale lock, only one wins."""

    def test_clean_first_preserves_lock(self):
        """clean_first=True does not delete .processing.lock."""

    def test_locked_folder_returns_skipped(self):
        """process_folder returns status=skipped for locked folder."""
```

## Deferred (YAGNI)

These were in the original plan but cut based on review:

- **Pre-validation lock check in `get_folders_from_file()`**: Parsl already dispatches one folder per worker. Double-submission is an operational error.
- **CLI `--lock_timeout_hr` flag**: Hardcode 8 hours. Add flag if someone needs it.
- **`convert` sub-job special case**: Existing `wf_present` check at [omol.py:533-537](qtaim_gen/source/core/omol.py#L533-L537) already handles this.
- **Corrupted timings.json backup**: Not worth preserving; timing data is re-derived from execution.
- **Lock utilities in `utils/lock.py`**: With the simplified mtime approach, the lock code is ~25 lines total in `workflow.py`. Extraction adds complexity for no benefit. Revisit if lock logic grows.

## References

### Internal
- Recent timings.json path fix: commit `3a5a500`
- Existing restart tests: [test_restart_partial_folders.py](tests/test_restart_partial_folders.py) (48 tests)
- Restart bug fixes: commit `655285b`
- `atomic_json_write`: [atomic_write.py](qtaim_gen/source/utils/atomic_write.py)

### Architecture
- `run_jobs()` restart logic: [omol.py:600-652](qtaim_gen/source/core/omol.py#L600-L652)
- `ORDER_OF_OPERATIONS_separate`: [omol.py:57-63](qtaim_gen/source/core/omol.py#L57-L63)
- Lock functions: [workflow.py:28-44](qtaim_gen/source/core/workflow.py#L28-L44)
- Category validation: [validation.py:44-124](qtaim_gen/source/utils/validation.py#L44-L124)
- Data dicts: [multiwfn.py](qtaim_gen/source/data/multiwfn.py)
