# Copilot / AI Agent Instructions — qtaim_generator

Keep guidance short and actionable. Focus on where core logic lives, how data flows, and common developer workflows.

Overview
- Purpose: post-process quantum chemistry outputs (Multiwfn/Orca) into ML-ready descriptors (QTAIM, charges, bonding).
- Language: Python 3.9+; package name `qtaim_generator` (editable install used in development).

Big picture (quick)
- Input: datasets as JSON/PKL/BSON under `data/` (see `data/inputs` and many dataset folders).
- Processing: CLI scripts live in `qtaim_gen/source/scripts/` (entrypoints declared in `pyproject.toml`).
- Core utilities: `qtaim_gen/source/utils/` contains validation and parsing helpers (e.g. `validation.py`).
- Output: parsed QTAIM JSONs and LMDBs under `data/lmdb_conv/` for downstream ML.

Key entry points (examples)
- `create-files` — generate DFT & QTAIM job inputs. (Script: `qtaim_gen/source/scripts/create_files.py`)
- `run-qtaim-gen` — run DFT/QTAIM jobs in a folder. (Script: `qtaim_gen/source/scripts/run.py`)
- `parse-data` — parse outputs into unified JSON. (Script: `qtaim_gen/source/scripts/parse_data.py`)
- See `pyproject.toml` for CLI names and module targets.

Developer workflows
- Install for development:
  - Prefer conda for RDKit: `conda env create -f environment.yml` then `pip install -e .`.
  - Alternatively: `pip install -e .` (RDKit on pip may be incomplete).
- Run tests: `pytest -q` (dev deps in `[project.optional-dependencies].dev`).
- Lint: `ruff` is used in dev dependencies; run locally if configured.

Project-specific conventions
- Datasets: input files may be JSON, PKL, or BSON; many scripts accept `--file` for input path.
- CLI args are the primary surface — prefer using the packaged CLI entrypoints (see `pyproject.toml`).
- High-throughput runs: two styles supported — local Python multithreading and `parsl`-based runners (look for `full-runner-parsl*` entrypoints).
- Monitoring: `qtaim_gen/source/scripts/tracking_db.py` shows patterns for sqlite monitoring and parallel scanning (ThreadPoolExecutor + sqlite upserts).

Monitoring (concrete)
- Purpose: `qtaim_gen/source/scripts/tracking_db.py` builds/queries a SQLite "validation" DB to track QTAIM/DFT job completeness and timing.
- Key functions:
  - `scan_and_store_parallel(root_dir, db_path, full_set=0, max_workers=8, sub_dirs_to_sweep=None, debug=False)` — walks a two-level folder layout (category/subset/job), extracts job info via `get_information_from_job_folder`, and upserts rows into `validation` table.
  - `print_summary(db_path, path_to_overall_counts_db)` — prints aggregate counts, recent activity, and per-subset averages from the DB.
- Expected layout: top-level `root_dir` with category folders, each containing subset folders, each containing job folders (the code uses `os.listdir` twice to build jobs).
- Example: run `scan_and_store_parallel` from a Python one-liner (after `pip install -e .`):

```bash
python -c "from qtaim_gen.source.scripts.tracking_db import scan_and_store_parallel; scan_and_store_parallel('/path/to/root', 'validation.sqlite', max_workers=12, debug=True)"
```

- Example: print a summary from the DB:

```bash
python -c "from qtaim_gen.source.scripts.tracking_db import print_summary; print_summary('validation.sqlite')"
```

- Notes:
  - Dependencies: `pandas`, `tqdm` (and stdlib `sqlite3`, `concurrent.futures`).
  - Performance: uses `ThreadPoolExecutor` to parse folders in parallel; increase `max_workers` on machines with many I/O threads.
  - The `scan_and_store_parallel` function infers DB columns from the first found job; ensure `root_dir` contains representative jobs when running.

Integration & external deps
- Multiwfn / Critic2: parsers expect outputs from these tools. Multiwfn is richer and preferred.
- ORCA: used for DFT inputs; scripts include `--orca_path` to point to executables.
- RDKit: recommended from conda-forge for full functionality.
- LMDB & DGL: outputs are formatted for LMDB-backed datasets for ML (see `data/lmdb_conv/`).

Where to look for behavior changes
- Parsing/validation: `qtaim_gen/source/utils/validation.py` and `qtaim_gen/source/scripts/helpers/`.
- Job creation and runners: `qtaim_gen/source/scripts/create_files.py`, `run.py`, `full_runner*.py`.
- Tests: `tests/` contains unit/integration tests illustrating expected data shapes and normalization.

Examples (concrete)
- Create inputs (after install):
  `create-files --file data/inputs/my_dataset.json --root out/jobs --parser Multiwfn --options_qm_file options.json`
- Run parse step:
  `parse-data --root out/jobs --file_in data/inputs/my_dataset.json --file_out parsed.json --impute`
- Run test suite:
  `pytest tests/test_parse.py::test_parse_json -q`

Notes for an AI agent
- Prefer using the CLI entrypoints rather than invoking modules directly unless making unit-level changes.
- When modifying parsing logic, update corresponding tests in `tests/` and add a small fixture/input under `tests/test_files/`.
- Watch for filesystem assumptions: many scripts expect a folder-per-job layout (see `README.md` examples and `tracking_db.py`).
- Preserve editable install behavior and `pyproject.toml` entrypoints when renaming scripts.

If anything here is unclear or you want more examples (argument combos, common failing cases), tell me which area to expand.
