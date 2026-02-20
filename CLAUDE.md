# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**qtaim_generator** is a high-throughput post-processing package for quantum chemistry calculations. It wraps Multiwfn/ORCA to generate ML-ready descriptors (QTAIM, partial charges, bond orders) from DFT outputs, and converts them to graph-based LMDB datasets for training with `qtaim_embed`.

## Development Commands

```bash
# Install (editable mode)
conda env create -f environment.yml
pip install -e .

# Run tests
pytest -q                              # All tests
pytest tests/test_lmdb.py -v           # Specific file
pytest tests/test_parse.py::test_name  # Specific test

# Lint (dev dependency)
ruff check .
```

## Architecture

```
qtaim_gen/source/
├── core/                # Core analysis, parsing & conversion
│   ├── converter.py     # LMDB→graph conversion (BaseConverter, QTAIMConverter, GeneralConverter, ASELMDBConverter)
│   ├── qtaim_embed.py   # Utilities bridging qtaim_embed (feature indexing, scaling)
│   ├── omol.py          # gbw_analysis() - main job folder processing
│   ├── controller.py    # Thread-based DFT/QTAIM job controller
│   ├── workflow.py      # Parsl task orchestration
│   ├── parse_qtaim.py   # Critical point parsing from Multiwfn
│   ├── parse_multiwfn.py# Charge/bond/fuzzy output parsing
│   ├── parse_orca.py    # ORCA .out file parser (enum state machine, WIP on feat/orca-out-parser)
│   └── parse_json.py    # JSON output merging/validation
├── data/                # Multiwfn command helpers (charge_data, bond_order_data, etc.)
├── scripts/             # CLI entry points (defined in pyproject.toml)
│   ├── create_files.py  # create-files: generate DFT/QTAIM inputs
│   ├── run.py           # run-qtaim-gen: execute jobs
│   ├── parse_data.py    # parse-data: parse outputs to JSON
│   ├── json_to_lmdb.py  # json-to-lmdb: JSON→LMDB conversion (supports sharding)
│   ├── full_runner*.py  # full-runner*: orchestrated workflows
│   ├── helpers/
│   │   ├── tracking_db.py       # SQLite monitoring + W&B integration
│   │   ├── clean_omol.py        # Bulk cleanup of job folders (removes intermediate files)
│   │   ├── debug_lmdb_contents.py # LMDB inspection/debugging utility
│   │   ├── generator_to_embed.py  # generator-to-embed: run converter from JSON config
│   │   ├── refine_list_of_jobs.py # Filter/refine job lists for reprocessing
│   │   ├── configs_converter/     # Validated JSON configs for different converter types
│   │   └── ...                    # Other helpers (check_res_*, folder_*_to_pkl, etc.)
│   └── old/             # Deprecated/archived scripts
└── utils/               # Utilities
    ├── validation.py    # Job completeness validation
    ├── lmdbs.py         # LMDB read/write utilities (json_2_lmdbs, sharded writes, merge)
    ├── bonds.py         # Bond detection (RDKit-based and coordinate-based)
    ├── io.py            # Input file generation, format conversion, bond detection
    ├── aselmdb.py       # ASE LMDB format helpers
    └── parsl_configs.py # Parsl executor configurations (ALCF, NERSC)
```

## Key Workflows

**QTAIM-only (3-step):**
1. `create-files` → Generate ORCA inputs + Multiwfn scripts
2. `run-qtaim-gen` → Execute DFT + QTAIM analysis
3. `parse-data` → Parse outputs into unified JSON

**Full analysis:** `full-runner`, `full-runner-parsl`, `full-runner-parsl-alcf`

**JSON → LMDB → Graphs (ML pipeline):**
1. `json-to-lmdb` → Convert parsed JSON outputs to typed LMDB files (structure, charge, qtaim, bond, fuzzy)
2. `generator-to-embed` → Run a converter (Base/QTAIM/General) to build DGL graph LMDBs for `qtaim_embed`

## Converter System

The converter classes in `core/converter.py` transform raw LMDB data into DGL heterographs:

- **BaseConverter**: Structural info only (positions, elements, connectivity)
- **QTAIMConverter**: QTAIM bond paths + critical point properties
- **GeneralConverter**: Flexible — supports fuzzy/IBSI/QTAIM bonding schemes, charge/bond/fuzzy filters
- **ASELMDBConverter**: For ASE-formatted LMDB inputs

Converters are driven by JSON config files (see `scripts/helpers/configs_converter/`). Key config params: `bonding_scheme`, `bond_list_definition`, `bond_cutoff`, `charge_filter`, `fuzzy_filter`.

Sharding is supported for large datasets — process in chunks, then merge. See `docs/SHARDING_GUIDE.md`.

## Job Folder Layout

Scripts expect a two-level hierarchy: `root_dir/category/subset/job/`. Each job folder contains:
- `input.inp` (ORCA input)
- `*.wfn`/`*.gbw` (wavefunctions)
- `charge.json`, `bond.json`, `qtaim.json` (outputs)
- `timings.json`, `.processing.lock`

## External Dependencies

- **ORCA**: DFT calculations (v5 & v6 supported)
- **Multiwfn**: QTAIM analysis (preferred over Critic2)
- **orca_2mkl**: For ECP/Molden conversions
- **RDKit**: Install from conda-forge for full functionality
- **qtaim_embed**: Required for converter classes (MoleculeWrapper, grapher, DGL serialization)

## CLI Entry Points

All commands defined in `pyproject.toml [project.scripts]`. Main ones:
- `create-files`, `run-qtaim-gen`, `parse-data` (QTAIM workflow)
- `json-to-lmdb` (JSON → LMDB conversion, supports sharding via `--sharded`)
- `generator-to-embed` (LMDB → DGL graph LMDB via converter config)
- `full-runner`, `full-runner-parsl`, `full-runner-parsl-alcf` (orchestrated full analysis)
- `check-res-wfn`, `check-res-rxn-json` (validation helpers)
- `folder-xyz-molecules-to-pkl`, `folder-orca-inp-to-pkl`, `outcar-seek-and-convert-xyz` (format conversion)

## Code Conventions

- Input datasets: JSON, PKL, or BSON formats accepted
- Use CLI entrypoints rather than invoking modules directly
- When modifying parsing logic, add tests and fixtures under `tests/test_files/`
- Preserve `pyproject.toml` entrypoints when renaming scripts
- Many scripts use `--file` for input path and `--root` for output directory
- Converter configs go in `scripts/helpers/configs_converter/` with tests
- Deprecated scripts go in `scripts/old/`, not deleted

## Test Suite

Run tests with `pytest -q` or specific files with `pytest tests/<file>.py -v`.

| Test File | Coverage |
|-----------|----------|
| `test_parse.py` | QTAIM/DFT input parsing, critical point extraction |
| `test_parse_multiwfn.py` | Multiwfn output parsing (charges, bonds, fuzzy) |
| `test_parse_orca.py` | ORCA `.out` file parser (WIP) |
| `test_bond_detection.py` | Bond detection from coordinates, ORCA input parsing |
| `test_lmdb.py` | LMDB read/write, key normalization, serialization roundtrips |
| `test_parse_json.py` | JSON output parsing and validation |
| `test_tracking_db.py` | SQLite tracking database operations |
| `test_ml.py` | ML-related data transformations |
| `test_general_converter_filters.py` | GeneralConverter with different bonding schemes/filters |
| `test_converter_molwrapper_integration.py` | Converter + MoleculeWrapper integration |
| `test_sharded_converter.py` | Sharded converter processing and merge |
| `test_json_to_lmdb_sharding.py` | json-to-lmdb sharding pipeline |
| `test_scaler_merge.py` | Graph scaler merge behavior |

Test fixtures are in `tests/test_files/` with subdirectories for ORCA inputs, Multiwfn outputs, etc.

## Documentation

Additional docs in `docs/`:
- `SHARDING_GUIDE.md` — How to use sharded conversion for large datasets
- `JSON_TO_LMDB_SHARDING.md` — json-to-lmdb sharding details
- `WANDB_INTEGRATION.md` — W&B tracking setup
- `solutions/` — Documented solutions to past bugs (merge/scaling, file I/O, test improvements)
- `brainstorms/` and `plans/` — Feature design documents
