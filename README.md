# Generator

<img src="https://github.com/santi921/qtaim_generator/blob/main/qtaim_gen/notebooks/TOC.png" width=50% height=50%>

A package to perform post-processing on molecules, reactions, and (soon) periodic systems. It's a wrapper around Multiwfn that handles high-throughput workflows and can compute a rich set of descriptors including QTAIM, partial charges, several bonding schemes, etc. I am currently overhauling the package away from being QTAIM-first and instead integrating many descriptors in tandem. I will be sticking with the JSON file format for QTAIM and the "full" feature implementations so running calculations will remain the same, gathering and parsing utilities for ML, however, will change. Here's a quick status about what is available for QTAIM/full at the moment:

### QTAIM
- [x] Reaction-level feature generation and processing
- [x] Molecule-level feature generation and processing
- [x] Post-processing to LMDBs in DGL for ML


### Full
- [x] Molecule-level feature generation and processing
- [ ] Reaction-level feature generation and processing
- [x] High-throughput runners using python MP
- [x] High-throughput runners using parsl 
- [ ] Post-processing to LMDBs for ML (ongoing)
- [ ] Quacc integration (ongoing)


Simply install this package by cloning the repo and running: 
```
pip install -e .
```

# Overview - QTAIM Usage

We can use QTAIM to define bonds in a system as well as define a rich set of descriptors for machine learning. With a few scripts you can get to generating QTAIM-informatics for analysis and machine learning tasks. Currently, this package supports BondNet(<a href="https://github.com/santi921/bondnet">BonDNet</a>) (for reaction-property predicton) and <a href="https://github.com/santi921/qtaim_embed">QTAIM-Embed</a> and ChemProp <a href="[https://github.com/santi921/qtaim_embed](https://github.com/chemprop/chemprop)">QTAIM-Embed</a> for molecular machine learning tasks. Note that the Chemprop implementation currently only supports atom-level QTAIM descriptors. 

## Overview
To get started you will need to decide a few things: 
1) DFT Software: We currently have input file writers for Orca though creating custom writers for other software should be easy to integrate. For ORCA, we add a few options such a relativistic corrections and atom-specific basis sets. See the <a href="[https://github.com/santi921/qtaim_generator/blob/main/qtaim_gen/source/scripts/options_qm.json]">example JSON</a>  for more options 
2) QTAIM software: The implementation with Critic2 works but is relatively experimental and we suggest you use Multiwfn as it yields a richer set of QTAIM features. 
3) Level of theory: QTAIM is pretty resistant to low levels of theory. Take care, however, when your dataset contains metals (especially heavy metals where this assertion is  less tested). 


## Usage
Three scripts will be needed to generate QTAIM features readily formatted for your dataset. These scripts generate job files, run jobs, and parse outputs to a single json, respectively. For the following we will assume you have a properly formatted json/pickle/bson and will return to this later. 

1) <code> create_files.py </code> - generates input files for DFT and QTAIM jobs and has severate arguments:
    - <code> -reaction </code> : specifies whether the dataframe 
    - <code> -parser </code> Multiwfn or Critic2
    - <code> -file </code> specifies the dataset file
    - <code> -root </code> specifies where to write job files
    - <code> -options_qm_file </code> options for your electronic structure job
    - <code> --molden_sub </code> whether to use <code> orca_2mkl </code> to convert the a gbw to a .molden.input file prior to Multiwfn. Use this if you intend on using ECPs.
2) <code> run.py </code> - runs DFT and QTAIM jobs in selected folder
    - <code> -redo_qtaim </code> - whether to clear QTAIM results file and redo 
    - <code> -just_dft </code> - whether to scriptly run DFT jobs
    - <code> --reactions </code> : specifies whether the root folder contains reaction or molecule jobs
    - <code> -dir_active </code> - root folder of QTAIM/DFT jobs
    - <code> -orca_path </code> - path to ORCA executable
    - <code> -num_threads </code> - number of threads for DFT jobs
    - <code> -folders_to_crawl </code> - how many folders to check for complete jobs
3) <code> parse_data.py </code> takes DFT/QTAIM output files and merges QTAIM data into a the original data structure:
    - <code> --root </code> root folder of QTAIM/DFT jobs
    - <code> --file_in </code> - input dataframe used to construct QTAIM/DFT jobs
    - <code> --impute </code> - whether or not to fill in missing values with mean values from computed statistics
    - <code> --file_out </code> - where to write to
    - <code> --reaction </code> - where your data is a reaction dataset
    - <code> --update_bonds_w_qtaim </code> -whether to overwrite existing bond definitions
    - <code> -define_bonds </code> - method ("distances" or "qtaim") of determining bonds
   
## Extra Scripts
1) <code> parse_stop.py </code> computes and prints statistics of QTAIM values in selected folder
2) <code> check_res_rxn_json.py </code> checks the number of complete jobs for reaction QTAIM run
3) <code> check_res_wfn.py </code> checks the number of complete jobs for molecular QTAIM run
4) <code> folder_xyz_molecules_to_pkl.py </code> converts a folder of xyz files into a single dataset for subsequent QTAIM generation.

## Data Structure
Jsons, pkls, and bson can all be parsed. 



# Overview - Full Usage

The "full" pipeline computes a rich set of descriptors (partial charges, bond orders, fuzzy densities, QTAIM critical points) and converts them into PyTorch Geometric (PyG) heterograph LMDBs for training with [qtaim_embed](https://github.com/santi921/qtaim_embed).

## Pipeline: JSON → LMDB → Graphs

### Step 1: JSON → Typed LMDBs
After running `parse-data`, convert the per-job JSON outputs into typed LMDB files:

```bash
json-to-lmdb --file jobs.pkl --root /path/to/job_folders --out /path/to/lmdbs
```

This produces separate LMDB files for each data type: `structure.lmdb`, `charge.lmdb`, `qtaim.lmdb`, `bond.lmdb`, `fuzzy.lmdb`, `other.lmdb`.

For large datasets, use sharded mode:
```bash
json-to-lmdb --file jobs.pkl --root /path/to/job_folders --out /path/to/lmdbs --sharded --n_shards 4
```

See `docs/SHARDING_GUIDE.md` for details on sharded conversion and merging.

### Step 2: LMDB → PyG Heterographs
Run a converter to build graph LMDBs from the typed LMDBs:

```bash
generator-to-embed --config /path/to/converter_config.json
```

The converter reads from typed LMDBs (Step 1 output) and writes a single graph LMDB containing serialized PyG `HeteroData` objects.

## Converter Types

| Converter | Use Case | Required LMDBs |
|-----------|----------|-----------------|
| `BaseConverter` | Structural info only (positions, elements, connectivity) | `geom_lmdb` |
| `QTAIMConverter` | QTAIM bond paths + critical point properties | `geom_lmdb`, `qtaim_lmdb` |
| `GeneralConverter` | Flexible — supports multiple bonding/charge/bond/fuzzy schemes | `geom_lmdb` + any combination |
| `ASELMDBConverter` | ASE-formatted LMDB inputs | ASE LMDB file |

## Converter Config Reference

Configs are JSON files. See `qtaim_gen/source/scripts/helpers/configs_converter/` for working examples.

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
  "lmdb_locations": {
    "geom_lmdb": "/path/to/structure.lmdb"
  },
  "n_workers": 8,
  "batch_size": 500
}
```

### GeneralConverter config keys

| Key | Description |
|-----|-------------|
| `lmdb_locations` | Dict mapping LMDB type keys to file paths (e.g. `"geom_lmdb"`, `"charge_lmdb"`, `"qtaim_lmdb"`, `"bonds_lmdb"`, `"fuzzy_lmdb"` or `"fuzzy_full_lmdb"`, `"other_lmdb"`, `"orca_lmdb"`) |
| `data_inputs` | List of data sources to use: `["geom", "qtaim", "charge", "fuzzy", "bond", "other", "orca"]`. If omitted, auto-detected from `lmdb_locations`. |
| `bonding_scheme` | How bonds are defined: `"structural"` (coordinate-based), `"bonding"` (bond orders), or `"qtaim"` (bond paths) |
| `bond_list_definition` | Which bond order type defines the bond list: `"fuzzy"`, `"ibsi"`, `"laplacian"` |
| `bond_cutoff` | Minimum bond order to count as a bond (e.g. `0.3`) |
| `bond_filter` | Bond features to include: `["fuzzy"]`, `["ibsi"]`, `["fuzzy", "ibsi"]` |
| `charge_filter` | Charge schemes to include: `["hirshfeld", "adch", "cm5", "becke"]` |
| `fuzzy_filter` | Fuzzy density features: `["becke_fuzzy_density", "hirsh_fuzzy_density"]` |
| `orca_filter` | Top-level keys from `orca.json` to surface as features. `null` (default) uses `DEFAULT_ORCA_FILTER` (chemistry globals only); pass an explicit list to opt in to per-atom (e.g. `"mulliken_charges"`, `"gradient"`) or per-bond (e.g. `"mayer_bond_orders"`) data. |
| `keys_data` | Feature keys for atom/bond/global node types |
| `keys_target` | Target keys for training labels |
| `allowed_ring_size` | Filter molecules by ring sizes |
| `allowed_charges` / `allowed_spins` | Filter by molecular charge/spin (null = no filter) |
| `missing_data_strategy` | `"skip"` (drop molecules with missing data) or `"impute"` |

### Example: GeneralConverter with fuzzy bonds + charges
```json
{
  "chunk": -1,
  "filter_list": ["length", "scaled"],
  "restart": false,
  "allowed_ring_size": [3, 4, 5, 6, 7, 8],
  "keys_target": { "atom": [], "bond": [], "global": ["n_atoms"] },
  "keys_data": {
    "atom": ["charge_hirshfeld", "charge_adch"],
    "bond": [],
    "global": ["n_atoms", "charge_hirshfeld_dipole_mag"]
  },
  "lmdb_path": "/path/to/output",
  "lmdb_name": "general_graphs.lmdb",
  "lmdb_locations": {
    "geom_lmdb": "/path/to/structure.lmdb",
    "charge_lmdb": "/path/to/charge.lmdb",
    "bonds_lmdb": "/path/to/bond.lmdb",
    "fuzzy_full_lmdb": "/path/to/fuzzy.lmdb"
  },
  "bonding_scheme": "bonding",
  "bond_list_definition": "fuzzy",
  "bond_cutoff": 0.3,
  "bond_filter": ["fuzzy"],
  "charge_filter": ["hirshfeld", "adch"],
  "fuzzy_filter": ["becke_fuzzy_density", "hirsh_fuzzy_density"],
  "missing_data_strategy": "skip",
  "n_workers": 8,
  "batch_size": 500
}
```

### ORCA features in the converter

`orca.json` (parsed from `orca.out`) is converted by `json-to-lmdb` to `orca.lmdb` and consumed by `GeneralConverter` when `orca_lmdb` is in `lmdb_locations`. Features are emitted under the `orca_*` prefix using a scheme-as-suffix convention that mirrors `charge_<scheme>` (e.g. `orca_charge_mulliken`, `orca_spin_loewdin`, `orca_population_mayer`, `orca_bond_order_mayer`).

The `orca_filter` config key controls which `orca.json` top-level keys are surfaced. `null` falls back to `DEFAULT_ORCA_FILTER` (chemistry globals only):

| Granularity | Surfaced by default | Opt-in (explicit `orca_filter`) |
|---|---|---|
| Global scalar | `final_energy_eh`, `homo_eh`/`homo_ev`, `lumo_eh`/`lumo_ev`, `homo_lumo_gap_eh`, `s_squared`, `dipole_magnitude_au`, `gradient_rms` | `scf_cycles`, `n_alpha`/`n_beta`/`n_total`/`n_electrons`/`n_orbitals`, `gradient_norm`, `gradient_max` |
| Global nested dict | `energy_components` (flattens to `orca_energy_*`) | `scf_convergence` (flattens to `orca_scf_*`) |
| Global vector | `dipole_au` (`_x/_y/_z`), `rotational_constants_cm1` (`_a/_b/_c`), `quadrupole_au` (`_xx/_yy/_zz/_xy/_xz/_yz`) | — |
| Per-atom | — | `mulliken_charges`, `mulliken_spins`, `loewdin_charges`, `loewdin_spins`, `mayer_charges`, `mayer_population`, `gradient` (`_x/_y/_z`) |
| Per-bond | — | `loewdin_bond_orders`, `mayer_bond_orders` |

**Double-dip note:** `parse_orca.merge_orca_into_charge_json` already copies Mulliken/Loewdin/Mayer charges from `orca.out` into `charge.json`, so combining a non-empty `charge_filter` with a non-`null` `orca_filter` may produce duplicate features (`charge_mulliken_*` and `orca_charge_mulliken`). `GeneralConverter` emits a one-line warning when both are set; pick one source per scheme.

## Output Graph Structure

The output LMDB contains serialized PyG `HeteroData` graphs with node types:
- `atom`: Per-atom features (charges, QTAIM descriptors, fuzzy densities)
- `bond`: Per-bond features (bond orders, IBSI, fuzzy bond values)
- `global`: Molecular-level features (dipole moments, atom count)

Edges connect atoms to bonds (`atom_to_bond`) and atoms to global (`atom_to_global`).

```python
import lmdb, pickle
from qtaim_embed.data.lmdb import load_graph_from_serialized

env = lmdb.open("graphs.lmdb", readonly=True, subdir=False, lock=False)
with env.begin() as txn:
    value = txn.get(b"molecule_key")
    graph = load_graph_from_serialized(pickle.loads(value))
    print(graph.node_types)       # ['atom', 'bond', 'global']
    print(graph['atom'].feat.shape)  # [n_atoms, n_features]
env.close()
```


 ## Citation 
 If you use this package please cite the following, thanks!
 
 @Article{D4DD00057A,
author ="Vargas, Santiago and Gee, Winston and Alexandrova, Anastassia",
title  ="High-throughput quantum theory of atoms in molecules (QTAIM) for geometric deep learning of molecular and reaction 
properties",
journal  ="Digital Discovery",
year  ="2024",
volume  ="3",
issue  ="5",
pages  ="987-998",
publisher  ="RSC",
doi  ="10.1039/D4DD00057A",
url  ="http://dx.doi.org/10.1039/D4DD00057A"
}



## install 
