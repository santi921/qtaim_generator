---
title: "feat: Add --wfx flag for WFX wavefunction format support"
type: feat
date: 2026-02-24
---

# feat: Add --wfx flag for WFX wavefunction format support

## Overview

Add an opt-in `--wfx` CLI flag that makes the `.gbw` -> wavefunction conversion step produce `.wfx` files instead of `.wfn`. This is critical for Multiwfn stability when processing systems with atoms heavier than Kr (Krypton). After conversion, all downstream code auto-detects whichever format (`.wfn` or `.wfx`) is present, avoiding flag-threading through the entire pipeline.

## Problem Statement / Motivation

- `.wfn` is a legacy wavefunction format that can be numerically unstable for systems with heavy atoms (Z > 36) and ECPs
- `.wfx` is a more robust extended format that Multiwfn supports natively
- The codebase currently hardcodes `.wfn` in ~25 locations across the pipeline
- There is also a **latent bug**: `wfn_tf` at `omol.py:371` is referenced but never defined in `create_jobs()` scope, meaning the "convert" routine silently fails via the `except Exception` catch at line 380. This was likely the original attempt at this feature

## Proposed Solution

**Two-part approach:**

1. **`--wfx` flag at CLI / conversion layer** — Controls which Multiwfn menu option is used:
   - Default (no flag): `100\n2\n4\n<stem>.wfn\n0\nq\n` (produces `.wfn`)
   - With `--wfx`: `100\n2\n5\n<stem>.wfx\n0\nq\n` (produces `.wfx`)

2. **Auto-detection everywhere else** — A centralized `find_wavefunction_file(folder)` utility replaces all hardcoded `.wfn` checks downstream of conversion

## Technical Approach

### Phase 1: Centralized utility + bug fix

**1a. Create `find_wavefunction_file()` in `utils/io.py`**

```python
# qtaim_gen/source/utils/io.py

from typing import Optional
import os
from glob import glob

WFN_EXTENSIONS = (".wfx", ".wfn")  # .wfx preferred when both exist

def find_wavefunction_file(folder: str) -> Optional[str]:
    """Find a wavefunction file (.wfx or .wfn) in folder.

    Returns absolute path to the file, preferring .wfx over .wfn
    when both exist. Returns None if neither is found.
    """
    for ext in WFN_EXTENSIONS:
        matches = glob(os.path.join(folder, f"*{ext}"))
        if matches:
            return matches[0]
    return None


def has_wavefunction_file(folder: str) -> bool:
    """Check if any wavefunction file (.wfn or .wfx) exists in folder."""
    return find_wavefunction_file(folder) is not None
```

**1b. Fix `wfn_tf` bug at `omol.py:371`**

Replace the undefined `wfn_tf` variable with the new `wfx` parameter:

```python
# omol.py create_jobs() — line 365-375, convert routine
elif routine == "convert":
    job_dict["convert"] = os.path.join(folder, "convert.txt")
    with open(os.path.join(folder, "convert.txt"), "w") as f:
        if wfx:
            file_wf = file_gbw.replace(".gbw", ".wfx")
            file_wf_bare = file_wf.split("/")[-1]
            data = "100\n2\n5\n{}\n0\nq\n".format(file_wf_bare)
        else:
            file_wf = file_gbw.replace(".gbw", ".wfn")
            file_wf_bare = file_wf.split("/")[-1]
            data = "100\n2\n4\n{}\n0\nq\n".format(file_wf_bare)
        f.write(data)
```

### Phase 2: Update `create_jobs()` to use auto-detection + wfx param

**Key insight**: `create_jobs()` serves two purposes:
1. Write the conversion command (needs the `wfx` flag to know the target format)
2. Build `file_read` path for downstream Multiwfn scripts (can auto-detect IF file exists, but must predict IF conversion hasn't run yet)

**Changes to `create_jobs()` in `omol.py:215`:**

- Add `wfx: bool = False` parameter
- Replace hardcoded `.wfn` extension logic with format-aware construction:

```python
# omol.py create_jobs() — lines 253-282

wf_ext = ".wfx" if wfx else ".wfn"
wfn_present = False
wfx_present = False

for file in os.listdir(folder):
    if file.endswith(".wfn"):
        wfn_present = True
        file_wfn_search = os.path.join(folder, file)
    if file.endswith(".wfx"):
        wfx_present = True
        file_wfn_search = os.path.join(folder, file)

wf_present = wfx_present or wfn_present

# ... later, when building file_read from .gbw:
for file in os.listdir(folder):
    if file.endswith(".gbw"):
        bool_gbw = True
        file_gbw = os.path.join(folder, file)
        file_wf = file.replace(".gbw", wf_ext)
        # if there is a wfn/wfx, rename to match gbw prefix
        if wf_present:
            if file_wf not in os.listdir(folder):
                logger.info(f"Renaming wavefunction file to: {file_wf}")
                os.rename(
                    os.path.join(folder, file_wfn_search),
                    os.path.join(folder, file_wf),
                )
        file_molden = file.replace(".gbw", ".molden.input")
        file_molden = os.path.join(folder, file_molden)
        file_read = os.path.join(folder, file_wf)

    if file.endswith((".wfn", ".wfx")):
        file_read = os.path.join(folder, file)
```

### Phase 3: Update `run_jobs()` auto-detection

**Changes to `run_jobs()` in `omol.py:472`:**

Replace lines 527-530:

```python
# Before (hardcoded .wfn):
wfn_present = False
for file in os.listdir(folder):
    if file.endswith(".wfn"):
        wfn_present = True

# After (auto-detect):
from qtaim_gen.source.utils.io import has_wavefunction_file
wf_present = has_wavefunction_file(folder)
```

### Phase 4: Update `clean_jobs()`

**Changes to `clean_jobs()` in `omol.py:975`:**

Replace line 1043:

```python
# Before:
if file.endswith("wfn"):

# After:
if file.endswith((".wfn", ".wfx")):
```

### Phase 5: Update `gbw_analysis()` preprocess_compressed check

**Changes to `gbw_analysis()` in `omol.py:1288`:**

- Add `wfx: bool = False` parameter
- Pass `wfx` to `create_jobs()`
- Update lines 1352-1353:

```python
# Before:
required_files = [".inp", ".wfn"]

# After:
required_files = [".inp", ".wfn", ".wfx"]
```

Note: Use a list with all three so either format satisfies the check. The logic checks `if f.endswith(tuple(required_files))` so this works.

### Phase 6: Thread `wfx` through workflow.py

**Changes to `workflow.py`:**

- `process_folder()` (line 47): Add `wfx: bool = False` parameter, pass to `gbw_analysis()`
- `process_folder_alcf()` (line 219): Add `wfx: bool = False` parameter, pass to `gbw_analysis()`
- Update `files_to_remove` at line 270:

```python
# Before:
"orca.wfn",

# After:
"orca.wfn",
"orca.wfx",
```

- `run_folder_task()` and `run_folder_task_alcf()` use `**kwargs`, so they pass `wfx` through automatically

### Phase 7: Add `--wfx` to CLI entry points

**Files to update:**

1. `full_runner_parsl_alcf.py` — Add argparse argument + extract + pass to `run_folder_task_alcf()`
2. `full_runner_parsl.py` — Add argparse argument + extract + pass to `run_folder_task()`
3. `full_runner.py` — Add argparse argument + extract + pass to `gbw_analysis()`

```python
parser.add_argument(
    "--wfx",
    action="store_true",
    help="Use .wfx wavefunction format instead of .wfn (more stable for heavy atoms, Z > 36)",
)
```

### Phase 8: Update `refine_list_of_jobs.py`

**Changes to `refine_list_of_jobs.py:158-163`:**

Replace hardcoded `orca.wfn` check with auto-detection:

```python
# Before:
os.path.join(folder_outputs, "orca.wfn")

# After:
from qtaim_gen.source.utils.io import has_wavefunction_file
has_wavefunction_file(folder_outputs)
```

## Acceptance Criteria

### Functional Requirements

- [x] `--wfx` flag accepted by `full-runner`, `full-runner-parsl`, `full-runner-parsl-alcf`
- [x] Without `--wfx`: pipeline produces `.wfn` (current behavior unchanged)
- [x] With `--wfx`: `convert.txt` uses Multiwfn option `5` and targets `<stem>.wfx`
- [x] All Multiwfn analysis scripts (`props_*.mfwn`) reference the correct wavefunction file
- [x] Auto-detection works: folders with `.wfx` are processed without errors
- [x] Auto-detection prefers `.wfx` over `.wfn` when both exist
- [x] `clean_jobs()` removes both `.wfn` and `.wfx` files (when `.gbw` exists)
- [x] `--wfx --restart` on a folder with existing `.wfn` uses the existing `.wfn` (no re-conversion)
- [x] HPC cleanup (`workflow.py`) removes both `orca.wfn` and `orca.wfx`

### Bug Fixes (Pre-existing)

- [x] `wfn_tf` NameError at `omol.py:371` is fixed
- [x] `endswith("wfn")` at `omol.py:1043` corrected to `endswith((".wfn", ".wfx"))`

### Quality Gates

- [x] Unit test for `find_wavefunction_file()` with .wfn-only, .wfx-only, both, neither
- [x] Unit test for `create_jobs()` producing correct `convert.txt` content for both modes
- [x] Existing tests continue to pass (no regression)

## Files Changed (Summary)

| File | Change |
|------|--------|
| `qtaim_gen/source/utils/io.py` | Add `find_wavefunction_file()`, `has_wavefunction_file()`, `WFN_EXTENSIONS` |
| `qtaim_gen/source/core/omol.py` | Fix `wfn_tf` bug; add `wfx` param to `create_jobs()` + `gbw_analysis()`; auto-detect in `run_jobs()`, `clean_jobs()`, preprocess check |
| `qtaim_gen/source/core/workflow.py` | Add `wfx` param to `process_folder()` + `process_folder_alcf()`; add `orca.wfx` to cleanup |
| `qtaim_gen/source/scripts/full_runner_parsl_alcf.py` | Add `--wfx` argparse flag + thread through |
| `qtaim_gen/source/scripts/full_runner_parsl.py` | Add `--wfx` argparse flag + thread through |
| `qtaim_gen/source/scripts/full_runner.py` | Add `--wfx` argparse flag + thread through |
| `qtaim_gen/source/scripts/helpers/refine_list_of_jobs.py` | Use `has_wavefunction_file()` instead of hardcoded `orca.wfn` |
| `tests/test_wfx_support.py` (new) | Tests for `find_wavefunction_file()` and `create_jobs()` wfx mode |

## Out of Scope (Known Limitations)

- **Legacy scripts**: `controller.py`, `create_files.py`, `check_res_wfn.py`, `check_res_rxn_json.py` retain hardcoded `.wfn` references. These are used by older workflows (`run-qtaim-gen`, `create-files`) and are documented as not supporting `--wfx`
- **Auto-detection by atomic number**: Automatically switching to `.wfx` based on heavy atom presence (Z > 36) is a potential follow-up but out of scope for this change
- **Per-folder format selection**: The `--wfx` flag is global per batch run. Mixed datasets require two separate batch invocations

## Dependencies & Risks

- **Multiwfn version compatibility**: Menu option `5` for `.wfx` export must be available in the deployed Multiwfn version. Tested against Multiwfn 3.8+
- **No post-conversion validation exists**: If Multiwfn silently fails to produce the `.wfx` file, downstream scripts will fail. Consider adding a post-conversion file existence check in `run_jobs()` as a follow-up
- **HPC storage**: `.wfx` files may be slightly larger than `.wfn`. Not expected to be significant given cleanup removes them after analysis

## References

- [Multiwfn Manual](http://sobereva.com/multiwfn/Multiwfn_manual.html) — Main function 100, subfunction 2
- [Multiwfn Forum: WFX conversion](https://www.umsyar.com/wfnbbs/viewtopic.php?id=803)
- [ResearchGate: ORCA to WFN conversion](https://www.researchgate.net/post/How_do_I_convert_a_DFT_calculated_ORCA_output_file_to_wfn_file)
- Existing pattern: `--exhaustive_qtaim` flag (added in commit `5eab882`) follows the same CLI → `gbw_analysis()` → `create_jobs()` threading pattern
- Existing pattern: `find_orca_output_file()` in `parse_orca.py` is the model for the centralized file-finding utility
