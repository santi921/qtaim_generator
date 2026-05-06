# qtaim_generator

<img src="qtaim_gen/assets/TOC.png" width=50% height=50%>

A high-throughput post-processing package for quantum chemistry calculations. It wraps Multiwfn and ORCA to compute a rich set of descriptors - QTAIM critical points, partial charges (Hirshfeld, ADCH, CM5, Becke, Mulliken, Loewdin, Mayer), bond orders (fuzzy, IBSI, Laplacian, Mayer), and fuzzy atomic densities - and converts them into graph-based LMDB datasets for ML training with [qtaim_embed](https://github.com/santi921/qtaim_embed).

## Install

```bash
conda env create -f environment.yml
pip install -e .
```

RDKit must be installed from conda-forge for full functionality. The `environment.yml` handles this.

Optional extras:

```bash
pip install -e ".[parsl]"   # Parsl-based HPC runners
pip install -e ".[wandb]"   # W&B tracking integration
pip install -e ".[dev]"     # pytest, ruff
```

## Overview

The package has two main workflows:

**QTAIM-only (3-step):** Generate ORCA inputs, run DFT + QTAIM analysis, parse outputs to JSON.

**Full pipeline:** Run all descriptors (charges, bond orders, fuzzy densities, QTAIM), convert to typed LMDBs, build graph LMDBs for ML.

---

## QTAIM Workflow

### Step 1: Generate input files

```bash
create-files \
  -file dataset.pkl \
  -root /path/to/jobs \
  -options_qm_file options_qm.json \
  -parser Multiwfn
```

Key flags: `-reaction` for reaction datasets, `--molden_sub` for ECP jobs (converts `.gbw` to `.molden.input` before Multiwfn).

See `qtaim_gen/source/scripts/options_qm.json` for all QM options.

### Step 2: Run jobs

```bash
run-qtaim-gen \
  -dir_active /path/to/jobs \
  -orca_path /path/to/orca \
  -num_threads 8 \
  -folders_to_crawl 1000
```

Key flags: `-redo_qtaim` to clear and rerun QTAIM results, `-just_dft` to skip QTAIM.

### Step 3: Parse outputs

```bash
parse-data \
  --root /path/to/jobs \
  --file_in dataset.pkl \
  --file_out results.pkl
```

Key flags: `--impute` to fill missing values with mean, `--reaction` for reaction datasets, `--update_bonds_w_qtaim` to override bond definitions with QTAIM bond paths.

---

## Full Pipeline: JSON -> Typed LMDBs -> Graph LMDBs

### Step 1: Convert job outputs to typed LMDBs

```bash
json-to-lmdb --file jobs.pkl --root /path/to/job_folders --out /path/to/lmdbs
```

Produces one LMDB per data type: `structure.lmdb`, `charge.lmdb`, `qtaim.lmdb`, `bond.lmdb`, `fuzzy.lmdb`, `other.lmdb`, `orca.lmdb`.

For large datasets with variable-depth job folder hierarchies (e.g. OMol4M), pass a flat list of absolute job paths:

```bash
json-to-lmdb \
  --root_dir /path/to/root \
  --out_dir /path/to/lmdbs \
  --folder_list /path/to/job_paths.txt \
  --all --shard_index 0 --total_shards 6
```

LMDB keys are derived from each folder's relpath under `--root_dir` with path separators replaced by `__`. See `docs/SHARDING_GUIDE.md` for sharded conversion details.

### Step 2: Build graph LMDBs

```bash
generator-to-embed --config /path/to/converter_config.json
```

Reads typed LMDBs and writes serialized graph objects (DGL or PyG `HeteroData`) for direct use with `qtaim_embed`.

Add `--split` to produce train/val/test LMDBs with train-only scaler fitting:

```bash
generator-to-embed --config config.json --split
```

### Converter types

| Converter | Use case | Required LMDBs |
|-----------|----------|----------------|
| `BaseConverter` | Structural info only | `geom_lmdb` |
| `QTAIMConverter` | QTAIM bond paths + critical point properties | `geom_lmdb`, `qtaim_lmdb` |
| `GeneralConverter` | Flexible: any combination of charge/bond/fuzzy/QTAIM/ORCA | `geom_lmdb` + any |
| `ASELMDBConverter` | ASE-formatted LMDB input | ASE LMDB file |

Config files live in `qtaim_gen/source/scripts/helpers/configs_converter/`. See that directory's `README.md` for field documentation.

### Minimal config (BaseConverter)

```json
{
  "chunk": -1,
  "filter_list": ["length", "scaled"],
  "restart": false,
  "allowed_ring_size": [3, 4, 5, 6, 7, 8],
  "keys_target": { "atom": [], "bond": [], "global": ["n_atoms"] },
  "keys_data": { "atom": [], "bond": [], "global": ["n_atoms"] },
  "lmdb_path": "/path/to/output_dir",
  "lmdb_name": "graphs.lmdb",
  "lmdb_locations": { "geom_lmdb": "/path/to/structure.lmdb" },
  "n_workers": 8,
  "batch_size": 500
}
```

### GeneralConverter config reference

| Key | Description |
|-----|-------------|
| `bonding_scheme` | `"structural"` (coordinate-based), `"bonding"` (bond orders), or `"qtaim"` (bond paths) |
| `bond_list_definition` | Bond order type for the bond list: `"fuzzy"`, `"ibsi"`, `"laplacian"` |
| `bond_cutoff` | Minimum bond order threshold (e.g. `0.3`) |
| `bond_filter` | Bond features to include: `["fuzzy"]`, `["ibsi"]`, `["fuzzy", "ibsi"]` |
| `charge_filter` | Charge schemes: `["hirshfeld", "adch", "cm5", "becke"]` |
| `fuzzy_filter` | Fuzzy density features: `["becke_fuzzy_density", "hirsh_fuzzy_density"]` |
| `orca_filter` | Keys from `orca.json` to surface as features (`null` = chemistry globals only) |
| `missing_data_strategy` | `"skip"` or `"impute"` |
| `allowed_charges` / `allowed_spins` | Filter by molecular charge/spin (`null` = no filter) |

### Multi-vertical merge

For combining multiple dataset verticals into a unified train/val/test split:

```bash
multi-vertical-merge --config pipeline_config.json
```

Three phases: Plan (validate + census + split assignment), Build (parallel graph construction per vertical/split), Scale (fit scaler on all train data, apply to all). See `qtaim_gen/source/scripts/helpers/configs_converter/multi_vertical_example.json`.

### Reading the output graph LMDB

```python
import lmdb, pickle
from qtaim_embed.data.lmdb import load_graph_from_serialized

env = lmdb.open("graphs.lmdb", readonly=True, subdir=False, lock=False)
with env.begin() as txn:
    value = txn.get(b"molecule_key")
    graph = load_graph_from_serialized(pickle.loads(value))
    print(graph.node_types)        # ['atom', 'bond', 'global']
    print(graph['atom'].feat.shape)
env.close()
```

---

## Dataset Evaluation Pipeline

Tools for building evaluation holdouts, splitting descriptor LMDBs, and auditing splits. These operate on raw descriptor LMDBs (before graph conversion) and produce the train/val/test partitions reported in the paper.

### Compute transition-metal neighbor lists

Required input for the H1 metal-ligand holdout:

```bash
tm-neighbor-lists --bond_root /path/to/bond_lmdbs --out_dir /path/to/output
```

### Build holdout filter CSVs

Produces H1/H3/H6/H7/H8 evaluation holdout definitions:

```bash
build-holdout-csvs --manifest_dir /path/to/manifest --output_dir /path/to/filter_csvs
```

### Pull holdout records into separate LMDBs

```bash
pull-holdout-records \
  --lmdb_root /path/to/descriptor_lmdbs \
  --holdout_index /path/to/filter_csvs/INDEX.csv \
  --out_dir /path/to/holdout_lmdbs
```

### Split descriptor LMDBs into train/val/test

Composition-ordered split via deterministic key hashing:

```bash
split-descriptor-lmdbs \
  --lmdb_root /path/to/descriptor_lmdbs \
  --splits_dir /path/to/splits_output \
  --holdout_parquet /path/to/manifest_holdout.parquet
```

### Merge per-vertical splits into combined LMDBs

```bash
merge-split-descriptors \
  --splits_dir /path/to/splits_output \
  --output_dir /path/to/merged
```

Output layout: `<output_dir>/train/<descriptor>.lmdb`, `val/`, `test/`.

### Audit split integrity

```bash
audit-splits \
  --splits_dir /path/to/splits_output \
  --lmdb_root /path/to/descriptor_lmdbs
```

Reports HEALTHY / DRIFT / NEVER_SPLIT / SOURCE_BAD per vertical. DRIFT and NEVER_SPLIT verticals are printed as a rerun list.

---

## Analysis Tools

All analysis commands correspond to sections in the accompanying paper.

| Command | Paper section | Description |
|---------|---------------|-------------|
| `analysis-census` | Stream C / T1 | Per-vertical molecule counts, element coverage, ring statistics |
| `analysis-charge-alignment` | Section 6.2 / B1 | Pairwise agreement between charge schemes |
| `analysis-dipole-alignment` | Stream E2 / Section 6.7 | Cross-vertical dipole magnitude agreement |
| `analysis-bond-agreement` | Stream D | Cross-vertical bond order agreement |
| `analysis-noise-floors` | Stream F | Cross-method noise floor estimation |
| `analysis-soap-featurize` | Stream G | SOAP descriptor computation for UMAP embedding |
| `analysis-soap-umap` | Stream G | UMAP projection of SOAP-featurized structures |

All commands accept `--help` for usage.

---

## Utility Reference

```
json-to-lmdb               Convert job JSON outputs to typed LMDBs (supports sharding)
generator-to-embed         Build graph LMDBs from typed LMDBs via converter config
multi-vertical-merge       Merge multiple dataset verticals with global splits and scaling
build-manifest             Build a dataset manifest (molecule counts, element coverage)
lmdb-status-audit          Audit LMDB completeness across verticals
lmdb-filter-vertical       Filter records from a vertical LMDB by key list
backfill-orca-into-json    Backfill parsed ORCA fields into existing charge.json files
find-bad-json              Find invalid or empty JSON files in a job tree
create-files               Generate ORCA + Multiwfn input files from a molecule dataset
run-qtaim-gen              Run DFT + QTAIM jobs in a job folder tree
parse-data                 Parse DFT/QTAIM outputs into a unified JSON/PKL
full-runner                Orchestrated full analysis (threads)
full-runner-parsl          Orchestrated full analysis (Parsl)
full-runner-parsl-alcf     Orchestrated full analysis (Parsl, ALCF Polaris)
check-res-wfn              Check job completion for molecular QTAIM runs
check-res-rxn-json         Check job completion for reaction QTAIM runs
folder-xyz-molecules-to-pkl  Convert a folder of XYZ files to a dataset PKL
folder-orca-inp-to-pkl     Convert a folder of ORCA inputs to a dataset PKL
```

---

## External Dependencies

- **ORCA** (v5 or v6): DFT calculations
- **Multiwfn**: QTAIM + descriptor analysis
- **orca_2mkl**: Required for ECP jobs (converts `.gbw` to `.molden.input`)
- **RDKit**: Install from conda-forge
- **qtaim_embed**: Required for graph LMDB construction

---

## Citation

If you use this package, please cite:

```bibtex
@Article{D4DD00057A,
  author  = {Vargas, Santiago and Gee, Winston and Alexandrova, Anastassia},
  title   = {High-throughput quantum theory of atoms in molecules (QTAIM) for geometric deep learning of molecular and reaction properties},
  journal = {Digital Discovery},
  year    = {2024},
  volume  = {3},
  issue   = {5},
  pages   = {987--998},
  doi     = {10.1039/D4DD00057A}
}
```
