# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**qtaim_generator** is a high-throughput post-processing package for quantum chemistry calculations. It wraps Multiwfn/ORCA to generate ML-ready descriptors (QTAIM, partial charges, bond orders) from DFT outputs.

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
├── core/           # Core analysis & parsing
│   ├── converter.py     # LMDB conversion (BaseConverter, QTAIMConverter)
│   ├── omol.py          # gbw_analysis() - main job folder processing
│   ├── workflow.py      # Parsl task orchestration
│   ├── parse_qtaim.py   # Critical point parsing from Multiwfn
│   └── parse_multiwfn.py# Charge/bond/fuzzy output parsing
├── data/           # Multiwfn command helpers (charge_data, bond_order_data, etc.)
├── scripts/        # CLI entry points (defined in pyproject.toml)
│   ├── create_files.py  # create-files: generate DFT/QTAIM inputs
│   ├── run.py           # run-qtaim-gen: execute jobs
│   ├── parse_data.py    # parse-data: parse outputs to JSON
│   ├── full_runner*.py  # full-runner*: orchestrated workflows
│   └── tracking_db.py   # SQLite monitoring + W&B integration
└── utils/          # Utilities
    ├── validation.py    # Job completeness validation
    ├── lmdbs.py         # LMDB read/write utilities
    └── io.py            # Input file generation, format conversion
```

## Key Workflows

**QTAIM-only (3-step):**
1. `create-files` → Generate ORCA inputs + Multiwfn scripts
2. `run-qtaim-gen` → Execute DFT + QTAIM analysis
3. `parse-data` → Parse outputs into unified JSON

**Full analysis:** `full-runner`, `full-runner-parsl`, `full-runner-parsl-alcf`, `full-runner-parsl-nersc`

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

## CLI Entry Points

All commands defined in `pyproject.toml [project.scripts]`. Main ones:
- `create-files`, `run-qtaim-gen`, `parse-data` (QTAIM workflow)
- `full-runner*` (orchestrated full analysis)
- `check-res-wfn`, `check-res-rxn-json` (validation helpers)

## Code Conventions

- Input datasets: JSON, PKL, or BSON formats accepted
- Use CLI entrypoints rather than invoking modules directly
- When modifying parsing logic, add tests and fixtures under `tests/test_files/`
- Preserve `pyproject.toml` entrypoints when renaming scripts
- Many scripts use `--file` for input path and `--root` for output directory
