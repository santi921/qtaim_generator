---
title: "feat: Migrate qtaim_generator to qtaim_embed PyG API"
type: feat
date: 2026-03-10
---

# Migrate qtaim_generator to qtaim_embed PyG API

## Overview

`qtaim_embed` completed a full migration from **DGL to PyTorch Geometric (PyG)** as of commit `e00fcc8` (March 5, 2026). All 21 source files were migrated — graphs are now `torch_geometric.data.HeteroData`, serialization uses `torch.save`/`torch.load`, and DGL is no longer a dependency.

`qtaim_generator` still references the old DGL-era API throughout its converter pipeline, tests, and examples. This plan covers updating all integration points to the new PyG-based API.

## Problem Statement / Motivation

- **Broken imports**: `serialize_dgl_graph` and `load_dgl_graph_from_serialized` no longer exist in `qtaim_embed.data.lmdb`
- **Graph format mismatch**: Converter code accesses graph data via DGL patterns (`.ndata["feat"]`) but qtaim_embed now produces PyG `HeteroData` objects
- **Blocked pipeline**: The full JSON → LMDB → Graph LMDB pipeline cannot run until these are reconciled
- 15 files across core, tests, and examples need updates

## Proposed Solution

Systematically update all `qtaim_embed` integration points in `qtaim_generator` to use the new PyG API. The changes are mostly mechanical renames and accessor pattern updates.

## Technical Considerations

### Breaking API Changes (qtaim_embed)

| Old (DGL era) | New (PyG era) | Location |
|---|---|---|
| `serialize_dgl_graph(graph, ret=True)` | `serialize_graph(graph, ret=True)` | `qtaim_embed.data.lmdb` |
| `load_dgl_graph_from_serialized(bytes)` | `load_graph_from_serialized(bytes)` | `qtaim_embed.data.lmdb` |
| `graph.ndata["feat"]` | `data[node_type].feat` or `_get_ndata(data, "feat")` | Graph access pattern |
| DGL heterograph return types | PyG `HeteroData` return types | `grapher.build_graph()`, `grapher.featurize()` |

### Unchanged API (confirmed still compatible)

| API | Status |
|---|---|
| `HeteroGraphStandardScalerIterative` class name | ✅ Unchanged |
| `.update()`, `.finalize()`, `.save_scaler()`, `.dict_node_sizes` | ✅ Unchanged |
| `merge_scalers()` function and signature | ✅ Unchanged |
| `HeteroGraphLogMagnitudeScaler` | ✅ Unchanged |
| `get_grapher()` function and signature | ✅ Unchanged |
| `MoleculeWrapper` constructor signature | ✅ Unchanged |
| `HeteroCompleteGraphFromMolWrapper.build_graph()` / `.featurize()` | ✅ Signatures unchanged (return type changed to HeteroData) |

### Graph Data Access Migration

DGL pattern:
```python
# Read
feats = graph.ndata["feat"]  # returns dict {ntype: tensor}
# Write
graph.nodes["atom"].data["feat"] = tensor
```

PyG pattern:
```python
# Read (via helper)
from qtaim_embed.data.processing import _get_ndata
feats = _get_ndata(data, "feat")  # returns {ntype: tensor}
# Read (direct)
atom_feat = data["atom"].feat
# Write
data["atom"].feat = tensor
```

### Scaler Interaction

The scalers now operate on PyG `HeteroData` instead of DGL graphs. The `__call__` interface is the same — pass a list of graphs and get scaled graphs back. Internal access uses the `_get_ndata`/`_set_ndata` helpers. No changes needed in how `converter.py` calls the scalers, but the graphs passed in must be PyG format (which they will be after the grapher migration).

## Acceptance Criteria

- [x] All imports from `qtaim_embed.data.lmdb` updated to new function names
- [x] All graph data access patterns updated from DGL to PyG
- [x] `converter.py` produces valid PyG `HeteroData` graphs in output LMDBs
- [x] `generator_to_embed.py` runs end-to-end with updated API
- [x] All existing tests pass (with updated assertions for PyG graph format)
- [x] No remaining references to `dgl` in qtaim_generator codebase (grep clean)
- [x] Example scripts updated and functional

## Files to Modify

### Tier 1: Core (must change — pipeline broken without these)

#### `converter.py` — Heavy edits
**Path:** `qtaim_gen/source/core/converter.py`

Changes:
- [ ] **Line 33**: `from qtaim_embed.data.lmdb import serialize_dgl_graph, load_dgl_graph_from_serialized` → `from qtaim_embed.data.lmdb import serialize_graph, load_graph_from_serialized`
- [ ] **Line 34**: `serial_func = serialize_dgl_graph` → `serial_func = serialize_graph`
- [ ] **~10 call sites**: Replace `serialize_dgl_graph(graph, ret=True)` → `serialize_graph(graph, ret=True)` (lines ~501, 930, 1117, 1156, 1373, 1412, 1968, 2015)
- [ ] **~3 call sites**: Replace `load_dgl_graph_from_serialized(...)` → `load_graph_from_serialized(...)` (lines ~489, scale_graphs_single)
- [ ] **Graph data access**: Audit all `graph.ndata[...]` and `graph.nodes[...].data[...]` patterns and convert to PyG accessor pattern (`data[ntype].key` or `_get_ndata`/`_set_ndata`)
- [ ] **split_graph_labels()**: This function moves features from ndata to labels — must use PyG accessors
- [ ] **Type hints/comments**: Update any references to "DGL" or "dgl" in docstrings

#### `qtaim_embed.py` — Light edits
**Path:** `qtaim_gen/source/core/qtaim_embed.py`

Changes:
- [ ] Verify `get_grapher` import still works (should be fine)
- [ ] Update `build_and_featurize_graph()` if it accesses graph data directly
- [ ] Update any DGL-specific graph inspection code

#### `generator_to_embed.py` — Medium edits
**Path:** `qtaim_gen/source/scripts/helpers/generator_to_embed.py`

Changes:
- [ ] Update any direct imports from qtaim_embed
- [ ] Verify the config-driven converter instantiation still works
- [ ] Test end-to-end with a sample config

### Tier 2: Tests (must change — tests will fail)

| File | Changes Needed |
|---|---|
| `tests/utils_lmdb.py` | Update `load_dgl_graph_from_serialized` → `load_graph_from_serialized`; update graph inspection to PyG |
| `tests/test_converter_molwrapper_integration.py` | Update imports, graph assertions (`.ndata` → PyG accessors) |
| `tests/test_sharded_converter.py` | Update serialization imports, graph loading, scaler assertions |
| `tests/test_serialize_roundtrip.py` | Update serialization function calls, graph format assertions |
| `tests/test_ml.py` | Update `HeteroGraphGraphLabelDataset` usage if API changed; verify graph format |
| `tests/test_key_normalization.py` | Update graph loading/inspection if it touches serialized graphs |
| `tests/test_scaler_merge.py` | Likely minimal — scaler API unchanged, but verify graph inputs |

### Tier 3: Examples (nice to have — not blocking)

| File | Changes Needed |
|---|---|
| `examples/test_all_converters.py` | Update imports and graph inspection |
| `examples/test_fuzzy_debug.py` | Update imports and graph inspection |

### Tier 4: Ancillary

| File | Changes Needed |
|---|---|
| `scripts/helpers/debug_lmdb_contents.py` | Update deserialization call |
| `docs/` references | Update any docs mentioning DGL graph format |

## Implementation Strategy

### Phase 1: Core serialization rename (mechanical, low risk)
1. Global find-replace of import names and function calls
2. `serialize_dgl_graph` → `serialize_graph`
3. `load_dgl_graph_from_serialized` → `load_graph_from_serialized`
4. Run grep to confirm no remaining references

### Phase 2: Graph data access patterns (requires careful audit)
1. Identify all `.ndata[...]`, `.nodes[...].data[...]`, `.edata[...]` patterns in `converter.py`
2. Determine if `_get_ndata`/`_set_ndata` helpers should be imported or if direct PyG access is cleaner
3. Update `split_graph_labels()` and any graph inspection code
4. Update `scale_graphs_single()` — the scaler `__call__` should work transparently but verify

### Phase 3: Test updates
1. Update imports in all test files
2. Update graph format assertions (check for PyG `HeteroData` type, use `.feat` instead of `.ndata["feat"]`)
3. Run `pytest -q` and fix failures iteratively

### Phase 4: Examples and docs
1. Update example scripts
2. Update debug utilities
3. Grep for any remaining "dgl" references

## Dependencies & Risks

**Dependencies:**
- `qtaim_embed` must be installed from the updated local repo (`pip install -e /home/santiagovargas/dev/qtaim_embed`)
- PyTorch Geometric must be installed (replaces DGL in environment)
- `torch_geometric` package and its dependencies (torch-scatter, torch-sparse, etc.)

**Risks:**
- **Environment conflict**: DGL and PyG may have conflicting CUDA/torch requirements. The conda env may need updating.
- **Existing LMDB files**: Any previously generated graph LMDBs contain DGL-serialized graphs and are **incompatible** with the new format. They must be regenerated.
- **Hidden DGL usage**: There may be indirect DGL usage through qtaim_embed utilities not immediately obvious from imports. A thorough grep is essential.
- **`split_graph_labels()` complexity**: This function manipulates graph internals heavily. It's the highest-risk change area in `converter.py`.

**Mitigations:**
- Run `conda run -n generator pip install -e /home/santiagovargas/dev/qtaim_embed` first to verify clean install
- Run `conda run -n generator python -c "import torch_geometric; print(torch_geometric.__version__)"` to verify PyG is available
- Keep DGL-era LMDB test fixtures alongside new PyG ones during transition (or regenerate test fixtures)

## Success Metrics

- `pytest -q` passes with 0 failures
- `generator-to-embed` CLI runs successfully on a test config
- No `dgl` imports remain anywhere in qtaim_generator (verified by grep)
- Output LMDB graphs are valid PyG `HeteroData` objects loadable by qtaim_embed

## References & Research

### Internal References
- Converter pipeline: [converter.py](qtaim_gen/source/core/converter.py)
- qtaim_embed bridge: [qtaim_embed.py](qtaim_gen/source/core/qtaim_embed.py)
- Generator CLI: [generator_to_embed.py](qtaim_gen/source/scripts/helpers/generator_to_embed.py)
- Past serialization bug: [merge-scaling-deserialization-bug.md](docs/solutions/performance-issues/merge-scaling-deserialization-bug.md)
- GeneralConverter API discovery: [generalconverter-api-discovery-testing.md](docs/solutions/test-improvements/generalconverter-api-discovery-testing.md)

### External References
- qtaim_embed repo: `/home/santiagovargas/dev/qtaim_embed`
- qtaim_embed PyG migration commit: `e00fcc8` (March 5, 2026)
- PyG HeteroData docs: https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.data.HeteroData.html
