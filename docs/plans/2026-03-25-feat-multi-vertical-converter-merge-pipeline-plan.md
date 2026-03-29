---
title: "feat: Multi-Vertical Converter-Merge Pipeline"
type: feat
date: 2026-03-25
---

# Multi-Vertical Converter-Merge Pipeline

## Overview

A 3-phase pipeline that processes multiple dataset verticals (e.g. SPICE, QM9, RMechDB) into per-vertical train/val/test graph LMDBs with composition-stratified splitting and train-only scaler fitting. Each vertical has its own source LMDBs and converter config, but shares a global split assignment so the same molecular formula always lands in the same split across all verticals.

## Problem Statement / Motivation

Training molecular property prediction models on diverse chemical spaces requires combining multiple datasets. Currently, `generator-to-embed --split` supports only a single vertical. To combine SPICE + QM9 + RMechDB, users must:

1. Run `generator-to-embed --split` independently per vertical (no global composition consistency)
2. Manually merge per-vertical outputs (key collision risk)
3. Manually fit a cross-vertical scaler (error-prone, data leakage risk)

This pipeline automates the workflow while guaranteeing composition-consistent splits and train-only scaler fitting across all verticals.

## Proposed Solution

A new CLI command `multi-vertical-merge` driven by a JSON pipeline config. Three phases: Plan, Build, Scale. The existing single-vertical `generator-to-embed --split` is unchanged.

---

## Design Principles

1. **Converter stays unaware of splits** — per [train-test-split-architecture-decisions.md](docs/solutions/logic-errors/train-test-split-architecture-decisions.md). Split logic lives in `utils/splits.py`; orchestration lives in the scripts layer.
2. **Backward compatible** — single-vertical `--split` is untouched. Multi-vertical is a separate CLI command.
3. **Resumable via output existence** — re-running skips completed work by checking whether output LMDBs exist with valid metadata. No sentinel files for phase boundaries.
4. **Minimal converter changes** — two small additions to the base `Converter` class: `include_keys` (via existing `_partition_keys()`) and element set injection (via existing `element_set` config key).

## Architecture Decisions

### Feature Schema Compatibility

**All verticals must produce graph-compatible feature schemas.** Same `bonding_scheme`, `keys_data`, `keys_target`, and filter lists across all verticals. Enforced at pipeline startup. Verticals may differ in: source LMDB paths, `allowed_ring_size`, `allowed_charges`, `allowed_spins`, `missing_data_strategy`, `bond_cutoff`.

### Element Set Unification

The global element set (union across all verticals) is computed during the Plan phase and injected into each converter via the **existing** `element_set` config key — zero converter changes needed for this.

### Formula Canonicalization

Formulas come from `pymatgen.core.Composition.formula` (Hill system ordering). The existing `build_formula_map_from_structure_lmdb()` already uses this — no changes needed.

### Cross-Vertical Merge (Deferred)

Merging per-vertical split LMDBs into unified per-split LMDBs (one `train.lmdb` containing all verticals) is deferred to a future PR. Per-vertical per-split LMDBs are valid standalone outputs — training code can concatenate dataset paths. This avoids key re-indexing complexity and provenance tracking.

---

## Pipeline Config

```json
{
  "output_dir": "/data/merged",
  "verticals": [
    {"name": "spice", "converter_config": "/data/spice/config.json"},
    {"name": "qm9", "converter_config": "/data/qm9/config.json"}
  ],
  "split_method": "composition",
  "split_ratios": [0.8, 0.1, 0.1],
  "split_seed": 42,
  "n_shards_per_split": 1
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `output_dir` | str | Yes | Root output directory |
| `verticals` | list | Yes | One entry per dataset vertical |
| `split_method` | str | Yes | `"composition"` or `"random"` |
| `split_ratios` | list[float] | Yes | 3 floats summing to 1.0 |
| `split_seed` | int | Yes | Seed for deterministic splits |
| `n_shards_per_split` | int | No | Sharding within each (vertical, split) pair. Default `1` (no sharding). When >1, uses existing `total_shards`/`shard_index` + `merge_shards()` machinery. |

Per-vertical: `name` (str, unique, filesystem-safe) and `converter_config` (path to existing converter JSON config). Converter type is always `GeneralConverter` — it is a superset of the others, and mixed converter types would violate schema compatibility.

**In-code representation** — frozen dataclasses, not raw dicts:

```python
@dataclass(frozen=True)
class VerticalConfig:
    name: str
    converter_config: str  # path to existing converter JSON

@dataclass(frozen=True)
class MultiVerticalPipelineConfig:
    output_dir: str
    verticals: list[VerticalConfig]
    split_config: SplitConfig  # reuse existing dataclass
    n_shards_per_split: int = 1
```

---

## Phase 1 — Plan (Sequential, Fast)

**Purpose:** Validate configs, read structure LMDBs, compute global element set, assign formulas to splits, write `split_plan.json`.

### Process

1. **Validate** — all converter configs parse, all source LMDBs exist, schema compatibility check (`keys_data`, `keys_target`, `bonding_scheme`, filter lists identical across verticals), vertical names unique and filesystem-safe, `SplitConfig` valid.

2. **Census** — for each vertical, call `build_formula_map_from_structure_lmdb(geom_lmdb_path)` to get `{key: formula}`. Also collect element sets via `get_elements_from_structure_lmdb()`. Compute global element set (union).

3. **Split assignment** — collect all unique formulas across all verticals. For each formula, call `assign_formula_to_split(formula, ratios, seed)` to deterministically assign it to a split. Build the assignment: `{vertical: {split: [keys]}}`.

4. **Write artifacts:**
   - `{output_dir}/split_plan.json` — the split assignment plus summary stats (per-split counts, actual ratios, deviation from requested, global element set)
   - Print a warning to stdout if any split deviates >5 percentage points from requested ratio

### Inputs / Outputs

| Input | Source |
|-------|--------|
| Pipeline config JSON | CLI argument |
| Per-vertical `structure.lmdb` | From each vertical's converter config `lmdb_locations.geom_lmdb` |

| Output | Path |
|--------|------|
| Split plan (assignment + summary) | `{output_dir}/split_plan.json` |

### Error Handling

- Validation failure: abort immediately with descriptive error (which config, which field)
- Missing/corrupt structure LMDB: abort naming the vertical and path
- Key missing `molecule_graph` or `composition`: skip with warning, count in summary
- All keys in one split (degenerate): warn loudly, continue

### New/Changed Code

**New in `utils/splits.py`:**

```python
def assign_formula_to_split(
    formula: str,
    ratios: tuple[float, float, float],
    seed: int,
    split_names: tuple[str, ...] = SPLIT_NAMES,
) -> str:
    """Hash a formula to a split. Extracted from partition_keys_by_composition."""
```

Refactor `partition_keys_by_composition()` to delegate to this function (no behavior change).

**New in `scripts/helpers/multi_vertical_merge.py`:**

```python
def plan_phase(config: MultiVerticalPipelineConfig) -> SplitPlan:
    """Validate, census, assign splits. Returns typed result AND writes split_plan.json."""
```

Returns a typed `SplitPlan` object for direct use by Phase 2. The JSON write is a side effect for inspection.

---

## Phase 2 — Build (Parallel: Vertical x Split x Shard)

**Purpose:** Build unscaled graph LMDBs. Each job processes a specific subset of keys from one vertical for one split.

### Process

For each `(vertical, split)` pair:

1. Load the vertical's converter config
2. Create a **copy** with injected overrides:
   - `lmdb_path` → `{output_dir}/{vertical_name}/`
   - `lmdb_name` → `{split_name}.lmdb`
   - `skip_scaling` → `true`
   - `save_scaler` → `false`
   - `element_set` → global element set (uses existing config key, zero converter changes)
   - `include_keys` → keys assigned to this (vertical, split) from Phase 1
3. Instantiate `GeneralConverter(config_copy)`
4. Run `converter.process()` — builds graphs only for assigned keys
5. `converter.finalize()` — writes metadata, closes DB

### Sharding (`n_shards_per_split > 1`)

When sharding is enabled, each `(vertical, split)` is further divided into `n_shards_per_split` jobs. Each shard job receives a slice of the split's key list and additionally injects `shard_index` and `total_shards` into the config copy. The existing `_partition_keys()` method (with the new `include_keys` guard) handles the key slicing.

After all shards for a `(vertical, split)` complete, call `Converter.merge_shards(skip_scaling=True)` to merge shard LMDBs. This reuses the existing shard sentinel pattern (`shard_N.done`) inside `merge_shards()`.

### Resumption

Check if `{output_dir}/{vertical}/train.lmdb` exists with a valid `length` metadata key. If so, skip that (vertical, split) job on re-run. For sharded jobs, rely on existing `merge_shards()` sentinel logic.

### Parallelism

- Each `(vertical, split)` job is independent — separate source LMDBs (read-only), separate output LMDB
- Launch via `ProcessPoolExecutor` or `subprocess`
- Max parallelism: `n_verticals * 3 * n_shards_per_split`
- No LMDB write contention — each job writes to its own file

### Inputs / Outputs

| Input | Source |
|-------|--------|
| Split plan | Phase 1 return value (or `split_plan.json` on resume) |
| Per-vertical converter configs | From pipeline config |

| Output | Path |
|--------|------|
| Per-vertical per-split unscaled graph LMDB | `{output_dir}/{vertical}/{split}.lmdb` |

### Error Handling

- Converter job failure: log error, the output LMDB won't have valid `length` metadata, so re-run will retry
- Individual key failures within a job: converter's existing skip logic handles this (logged in converter output)

### Converter Changes — `include_keys` via `_partition_keys()`

One change in the base `Converter` class. No subclass modifications needed:

```python
# In Converter.__init__:
self.include_keys: set[str] | None = (
    set(config_dict["include_keys"]) if "include_keys" in config_dict else None
)

# In Converter._partition_keys():
def _partition_keys(self, keys: list) -> list:
    if self.include_keys is not None:
        include_set = self.include_keys
        keys = [k for k in keys if (
            k.decode("ascii") if isinstance(k, bytes) else str(k)
        ) in include_set]
    if self.total_shards <= 1:
        return keys
    return [k for i, k in enumerate(keys) if i % self.total_shards == self.shard_index]
```

This is ~10 lines total in one file. Existing behavior is unchanged when `include_keys` is absent from config (default `None`).

---

## Phase 3 — Scale (Sequential Fit, Parallel Apply)

**Purpose:** Fit a single global scaler on all train data across all verticals, then apply to every split LMDB.

### Process

**Step 3a — Fit (Sequential):**

1. Create fresh `HeteroGraphStandardScalerIterative` instances (feature + label)
2. For each vertical's `train.lmdb`:
   a. Open read-only
   b. Iterate all non-metadata keys (skip `length`, `scaled`, `split_name`)
   c. Deserialize: `graph = load_dgl_graph_from_serialized(pickle.loads(value))`
   d. `feature_scaler.update([graph])`
   e. `label_scaler.update([graph])`
3. Finalize scalers
4. Save to `{output_dir}/feature_scaler.pt` and `{output_dir}/label_scaler.pt`

**Step 3b — Apply (Parallel):**

For each `(vertical, split)` LMDB:
1. Open read-write
2. Iterate all non-metadata keys
3. Deserialize graph (two-step: `pickle.loads` then `load_dgl_graph_from_serialized`)
4. Apply: `scaled_graphs = feature_scaler([graph]); label_scaler(scaled_graphs[0])`
5. Serialize: `pickle.dumps(serialize_dgl_graph(scaled_graphs[0], ret=True))`
6. Write back
7. Set `scaled=True` metadata

### Resumption

- If scaler `.pt` files exist, skip fitting (Step 3a)
- For each LMDB, check `scaled` metadata — if `True`, skip application for that LMDB

### Parallelism

- Step 3a: **Sequential** — iterative scaler accumulates running statistics
- Step 3b: **Parallel** — each LMDB is independent, each job modifies its own file

### Inputs / Outputs

| Input | Source |
|-------|--------|
| Per-vertical train LMDBs | Phase 2 output |
| All per-vertical per-split LMDBs | Phase 2 output |

| Output | Path |
|--------|------|
| Feature scaler | `{output_dir}/feature_scaler.pt` |
| Label scaler | `{output_dir}/label_scaler.pt` |
| Scaled LMDBs (in-place) | `{output_dir}/{vertical}/{split}.lmdb` (now with `scaled=True`) |

### Error Handling

- Missing train LMDB: abort — cannot produce valid scalers without complete train data
- Scaler application failure for one LMDB: that LMDB retains `scaled=False`, re-run retries it
- Scaler `.pt` files are saved before application begins, so partial application failure does not require re-fitting

### Extracted Reusable Functions

Extract from `scale_split_lmdbs()` in [generator_to_embed.py:93](qtaim_gen/source/scripts/helpers/generator_to_embed.py#L93) into `utils/scaling.py`:

```python
def fit_scalers_on_lmdbs(
    train_lmdb_paths: list[str],
    skip_keys: set[str],
) -> tuple[HeteroGraphStandardScalerIterative, HeteroGraphStandardScalerIterative]:
    """Create fresh scalers and fit on train LMDBs only. Streaming, no full-dataset load."""

def apply_scalers_to_lmdb_inplace(
    lmdb_path: str,
    feature_scaler: HeteroGraphStandardScalerIterative,
    label_scaler: HeteroGraphStandardScalerIterative,
    skip_keys: set[str],
) -> int:
    """Apply scalers to all graphs in an LMDB in-place. Returns count of scaled graphs."""

def save_scalers(
    feature_scaler: HeteroGraphStandardScalerIterative,
    label_scaler: HeteroGraphStandardScalerIterative,
    output_dir: str,
) -> None:
    """Save scaler .pt files."""
```

Then refactor `scale_split_lmdbs()` in `generator_to_embed.py` to delegate to these three functions (no behavior change, reduces duplication).

---

## Output Layout

```
{output_dir}/
  split_plan.json           # Split assignment + summary stats + global element set
  feature_scaler.pt         # Fitted on train data only
  label_scaler.pt
  spice/
    train.lmdb              # Scaled graph LMDB
    val.lmdb
    test.lmdb
  qm9/
    train.lmdb
    val.lmdb
    test.lmdb
```

---

## Implementation Plan

### Step 1: Foundation (New Files + Refactors)

**New files:**
- `qtaim_gen/source/scripts/helpers/multi_vertical_merge.py` — CLI entry point + phase orchestration
- `qtaim_gen/source/utils/scaling.py` — extracted `fit_scalers_on_lmdbs`, `apply_scalers_to_lmdb_inplace`, `save_scalers`
- `qtaim_gen/source/scripts/helpers/configs_converter/multi_vertical_example.json` — example pipeline config
- `tests/test_multi_vertical.py` — tests

**Changes to existing files:**
- `qtaim_gen/source/utils/splits.py` — extract `assign_formula_to_split()` from `partition_keys_by_composition()` (refactor, no behavior change)
- `qtaim_gen/source/scripts/helpers/generator_to_embed.py` — refactor `scale_split_lmdbs()` to delegate to `utils/scaling.py` functions (no behavior change)
- `pyproject.toml` — add `multi-vertical-merge` CLI entry point

**Deliverables:**
- `MultiVerticalPipelineConfig` and `VerticalConfig` dataclasses with validation
- Phase 1 (Plan): config validation, census, split assignment, `split_plan.json`
- Unit tests: config validation, `assign_formula_to_split`, element set unification

### Step 2: Graph Construction (Converter Change + Phase 2)

**Changes to existing files:**
- `qtaim_gen/source/core/converter.py` — add `include_keys` support in `__init__` + `_partition_keys()` (~10 lines total in base class only)

**New code in `multi_vertical_merge.py`:**
- Phase 2 orchestration: build config copies with overrides, launch parallel converter jobs, handle sharding via existing `merge_shards()`

**Deliverables:**
- `include_keys` filtering in base `Converter._partition_keys()`
- Phase 2 parallel orchestration
- Integration test: 2 synthetic verticals through Phases 1-2

### Step 3: Scaler Fitting + Application (Phase 3)

**New code in `multi_vertical_merge.py`:**
- Phase 3 orchestration using extracted `utils/scaling.py` functions

**Deliverables:**
- Cross-vertical scaler fitting on train only
- Parallel scaler application
- End-to-end integration test: 2 synthetic verticals, 3 formulas each, verify composition consistency, scaler correctness, no data leakage
- Documentation update to `CLAUDE.md`

---

## Acceptance Criteria

### Functional

- [x] Pipeline config validated at startup with clear errors for schema mismatches
- [x] Same formula -> same split across all verticals (composition consistency)
- [x] Split assignment uses existing SHA-256 hash from `partition_keys_by_composition()`
- [x] Each (vertical, split) converter job only processes assigned keys
- [x] Scaler fitted on train data only across all verticals (no data leakage)
- [x] Same scaler applied to all splits across all verticals
- [x] Single-vertical `generator-to-embed --split` unchanged
- [x] Pipeline resumable via output LMDB existence + `scaled` metadata
- [x] `split_plan.json` reports actual ratios and warns on >5pp deviation
- [ ] Sharding within (vertical, split) works via existing `total_shards`/`merge_shards()` (orchestration wired, needs end-to-end test with real converter)

### Non-Functional

- [x] Phase 2 jobs run in parallel (no LMDB write contention)
- [x] Scaler fitting streams through train LMDBs (no full-dataset memory load)
- [x] Two-step serialization used consistently (`pickle` + `load_dgl_graph_from_serialized`)
- [x] Glob patterns use `[0-9]*` character classes where applicable

### Quality Gates

- [x] Unit tests: config validation, `assign_formula_to_split`, element set union
- [x] Integration test: multi-vertical full pipeline, composition consistency verified
- [ ] Data leakage test: scaler stats from train only (requires qtaim_embed DGL graphs)
- [x] Backward compat test: `generator-to-embed --split` output unchanged after converter changes
- [x] Regression test: `include_keys=None` (default) produces identical output to before

---

## Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| `include_keys` in `_partition_keys` breaks existing behavior | High | Guard with `if self.include_keys is not None`; default `None`. Regression test. |
| Large datasets OOM during scaler fitting | Medium | Streaming iteration (already how iterative scaler works). |
| Formula canonicalization inconsistency across pymatgen versions | Low | Unit test with known formulas. |
| LMDB file size limits (default 10GB) | Medium | Document `map_size` configuration. |

---

## Critical Gotchas (from documented learnings)

1. **Two-step deserialization** — always `pickle.loads()` then `load_graph_from_serialized()`. Missing the second step gives raw bytes, not PyG HeteroData graphs. ([merge-scaling-deserialization-bug.md](docs/solutions/performance-issues/merge-scaling-deserialization-bug.md))
2. **Scaler API** — `scaler.update([graph])` for fitting, `scaler([graph])` returns a list for application, index `[0]`. ([train-test-split-architecture-decisions.md](docs/solutions/logic-errors/train-test-split-architecture-decisions.md))
3. **Glob patterns** — use `[0-9]*` to distinguish chunks from shards. ([lmdb-sharding-glob-and-folder-filtering.md](docs/solutions/logic-errors/lmdb-sharding-glob-and-folder-filtering.md))
4. **`merge_shards()` return type** — `Tuple[str, int]` (output_path, merged_count). Always unpack both. ([pre-existing-test-failures docs](docs/solutions/logic-errors/pre-existing-test-failures-merge-shards-validation-converter-phase-asymmetry.md))
5. **Metadata keys to skip** — `{"length", "scaled", "split_name"}` during LMDB iteration.

---

## References

- Architecture decisions: [train-test-split-architecture-decisions.md](docs/solutions/logic-errors/train-test-split-architecture-decisions.md)
- Split logic: [splits.py](qtaim_gen/source/utils/splits.py) — `SplitConfig`, `partition_keys_by_composition`, `build_formula_map_from_structure_lmdb`
- Converter: [converter.py](qtaim_gen/source/core/converter.py) — `Converter.__init__`, `_partition_keys()`, `process()`, `merge_shards()`
- CLI orchestration: [generator_to_embed.py](qtaim_gen/source/scripts/helpers/generator_to_embed.py) — `scale_split_lmdbs()`
- LMDB utilities: [lmdbs.py](qtaim_gen/source/utils/lmdbs.py) — `merge_lmdbs()`, `parse_config_gen_to_embed()`
- Existing split tests: [test_train_test_split.py](tests/test_train_test_split.py)
- Existing split plan: [2026-03-23-feat-train-test-split-generator-to-embed-plan.md](docs/plans/2026-03-23-feat-train-test-split-generator-to-embed-plan.md)
