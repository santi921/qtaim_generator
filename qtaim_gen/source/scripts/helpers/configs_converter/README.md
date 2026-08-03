# Converter Configuration Files

This directory contains validated configuration files for different converter types and bonding schemes.

## Quick Reference

| Config File | Converter | Bonding Scheme | Features | Status |
|------------|-----------|----------------|----------|--------|
| `base_unsharded.json` | BaseConverter | Structural | Basic atomic structure | ✓ Working |
| `qtaim_unsharded.json` | QTAIMConverter | QTAIM | QTAIM bond paths | ✓ Working |
| `general_fuzzy_bonds.json` | GeneralConverter | Bonding (Fuzzy) | Fuzzy bond orders | ⚠️ Testing needed |
| `general_qtaim_bonds.json` | GeneralConverter | QTAIM | QTAIM bond paths | ⚠️ Testing needed |
| `general_ibsi_bonds.json` | GeneralConverter | Bonding (IBSI) | IBSI bond orders | ⚠️ Testing needed |
| `general_with_global_dipoles.json` | GeneralConverter | QTAIM | Global dipole features | ⚠️ Testing needed |
| `base_sharded_shard0.json` | BaseConverter (sharded) | Structural | Shard 0 of 2 | ⚠️ Known issue |
| `base_sharded_shard1.json` | BaseConverter (sharded) | Structural | Shard 1 of 2 | ⚠️ Known issue |

## Configuration Descriptions

### Base Converter

**Purpose**: Minimal converter for basic molecular graphs with only structural information.

**Features**:
- Atomic positions
- Element types
- Connectivity (bonds from structure)
- n_atoms count

**Config**: `base_unsharded.json`

```json
{
  "lmdb_locations": {
    "geom_lmdb": "/path/to/structure.lmdb"
  },
  "bonding_scheme": "structural"  // Implicit
}
```

### QTAIM Converter

**Purpose**: Converter for QTAIM (Quantum Theory of Atoms in Molecules) analysis.

**Features**:
- All Base features
- QTAIM critical points
- Bond paths from QTAIM
- Atomic basin properties

**Config**: `qtaim_unsharded.json`

```json
{
  "lmdb_locations": {
    "geom_lmdb": "/path/to/structure.lmdb",
    "qtaim_lmdb": "/path/to/qtaim.lmdb"
  },
  "bonding_scheme": "qtaim"
}
```

### General Converter

**Purpose**: Flexible converter supporting multiple data sources and bonding schemes.

#### Fuzzy Bonding (`general_fuzzy_bonds.json`)

**Bonding**: Uses fuzzy bond orders (Becke/Hirshfeld fuzzy density)

**Features**:
- Atomic charges (Hirshfeld, ADCH, CM5)
- Fuzzy atomic densities
- Fuzzy bond orders (continuous values)

**Bond List Definition**: `"fuzzy"` - bonds defined by fuzzy bond order threshold

**Bond Cutoff**: `0.3` - minimum fuzzy bond order to include

```json
{
  "bonding_scheme": "bonding",
  "bond_list_definition": "fuzzy",
  "bond_filter": ["fuzzy"],
  "bond_cutoff": 0.3,
  "charge_filter": ["hirshfeld", "adch", "cm5"],
  "fuzzy_filter": ["becke_fuzzy_density", "hirsh_fuzzy_density"]
}
```

#### QTAIM Bonding (`general_qtaim_bonds.json`)

**Bonding**: Uses QTAIM bond paths (topological bonds)

**Features**:
- Atomic charges
- QTAIM bond paths
- Critical point properties

**Bond List Definition**: Implicit from QTAIM data

```json
{
  "bonding_scheme": "qtaim",
  "charge_filter": ["hirshfeld", "adch", "cm5"]
}
```

#### IBSI Bonding (`general_ibsi_bonds.json`)

**Bonding**: Uses IBSI (Intrinsic Bond Strength Index)

**Features**:
- Atomic charges
- IBSI bond orders
- Bond strength indicators

**Bond List Definition**: `"ibsi"` - bonds defined by IBSI threshold

**Bond Cutoff**: `0.1` - minimum IBSI value to include

```json
{
  "bonding_scheme": "bonding",
  "bond_list_definition": "ibsi",
  "bond_filter": ["ibsi"],
  "bond_cutoff": 0.1,
  "charge_filter": ["hirshfeld", "adch", "cm5"]
}
```

#### With Global Dipoles (`general_with_global_dipoles.json`)

**Purpose**: QTAIM converter with global dipole moment features

**Additional Features**:
- `becke_dipole_mag` - Dipole magnitude from Becke partitioning
- `hirshfeld_dipole_mag` - Dipole magnitude from Hirshfeld partitioning
- `adch_dipole_mag` - Dipole magnitude from ADCH charges
- `cm5_dipole_mag` - Dipole magnitude from CM5 charges

**Note**: Dipoles are added as global graph features, useful for predicting molecular properties.

```json
{
  "keys_data": {
    "global": [
      "n_atoms",
      "becke_dipole_mag",
      "hirshfeld_dipole_mag",
      "adch_dipole_mag",
      "cm5_dipole_mag"
    ]
  },
  "charge_filter": ["hirshfeld", "adch", "cm5", "becke"]
}
```

## Sharded Converter Configs

### Known Issue: "Invalid key '0'" Error

**Status**: ⚠️ Under investigation

**Problem**: When using sharded base converter with merge, scaling fails:
```
WARNING - Failed to scale graph b'...': Invalid key "0". Must be one of the edge types.
```

**Context**:
- Unsharded base converter: ✓ Works
- Sharded base converter (merge with `skip_scaling=True`): ✓ Works
- Sharded base converter (merge with `skip_scaling=False`): ✗ Fails on scaling

**Workaround**: Use Option 2 from SHARDING_GUIDE.md:
1. Merge with `skip_scaling=True`
2. Scale separately using `converter.scale_graph_lmdb()`

**Related Docs**:
- [docs/solutions/performance-issues/merge-scaling-deserialization-bug.md](../../../../docs/solutions/performance-issues/merge-scaling-deserialization-bug.md)
- [docs/SHARDING_GUIDE.md](../../../../docs/SHARDING_GUIDE.md)

## Testing Configurations

### Quick Test Script

Run all converter tests:

```bash
cd qtaim_gen/source/scripts/helpers/configs_converter
./test_configs.sh
```

This will test each config with a small subset of data (100 graphs) and report which ones work.

### Individual Testing

Test a specific config:

```python
import json
from qtaim_gen.source.core.converter import GeneralConverter

with open('general_fuzzy_bonds.json') as f:
    config = json.load(f)

# Modify paths if needed
config['lmdb_path'] = '/your/output/path'

# Limit data for testing
config['batch_size'] = 100

converter = GeneralConverter(config)
converter.process()
```

### Validating Output Graphs

After conversion, verify the graphs contain expected features:

```python
import lmdb
import pickle
from qtaim_embed.data.lmdb import load_dgl_graph_from_serialized

# Open output LMDB
env = lmdb.open('output_path/graphs.lmdb', readonly=True, subdir=False, lock=False)

with env.begin() as txn:
    cursor = txn.cursor()
    cursor.first()
    key, value = next(cursor)

    # Skip metadata
    if key == b'scaled' or key == b'length':
        key, value = next(cursor)

    # Load graph
    serialized = pickle.loads(value)
    graph = load_dgl_graph_from_serialized(serialized)

    # Check features
    print("Node types:", graph.ntypes)
    print("Edge types:", graph.canonical_etypes)
    print("Global features:", graph.ndata['atom'].keys())
    print("Bond features:", graph.edata.keys() if graph.num_edges() > 0 else "No bonds")

env.close()
```

## Common Config Parameters

### Required Parameters

| Parameter | Type | Description | Example |
|-----------|------|-------------|---------|
| `lmdb_path` | str | Output directory | `"/path/to/output"` |
| `lmdb_name` | str | Output LMDB filename | `"graphs.lmdb"` |
| `lmdb_locations` | dict | Input LMDB paths | See examples above |
| `bonding_scheme` | str | Bonding definition | `"qtaim"`, `"bonding"`, `"structural"` |

### Optional Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_workers` | int | 8 | Parallel workers for graph building |
| `batch_size` | int | 500 | Graphs per write batch |
| `chunk` | int | -1 | Process specific chunk (-1 = all) |
| `restart` | bool | false | Skip already-processed keys |
| `filter_list` | list | `["length", "scaled"]` | Metadata keys to skip |
| `allowed_ring_size` | list | `[3,4,5,6,7,8]` | Ring sizes to detect |
| `allowed_charges` | list | null | Filter by charge (null = all) |
| `allowed_spins` | list | null | Filter by spin (null = all) |
| `missing_data_strategy` | str | `"skip"` | How to handle missing data |

### Bonding Parameters

| Parameter | Type | Description | Used By |
|-----------|------|-------------|---------|
| `bond_filter` | list | Which bond descriptors to include | GeneralConverter |
| `bond_cutoff` | float | Minimum bond order threshold | Fuzzy/IBSI bonding |
| `bond_list_definition` | str | How to define bond list | `"fuzzy"`, `"ibsi"`, `"qtaim"` |
| `charge_filter` | list | Which charge schemes to include | GeneralConverter, QTAIMConverter |
| `fuzzy_filter` | list | Which fuzzy descriptors to include | GeneralConverter |

## Creating New Configs

### Template

```json
{
  "chunk": -1,
  "filter_list": ["length", "scaled"],
  "restart": false,
  "allowed_ring_size": [3, 4, 5, 6, 7, 8],
  "allowed_charges": null,
  "allowed_spins": null,
  "keys_target": {
    "atom": [],
    "bond": [],
    "global": ["n_atoms"]
  },
  "keys_data": {
    "atom": [],
    "bond": [],
    "global": ["n_atoms"]
  },
  "lmdb_path": "/path/to/output",
  "lmdb_name": "graphs.lmdb",
  "lmdb_locations": {
    "geom_lmdb": "/path/to/structure.lmdb"
  },
  "bonding_scheme": "structural",
  "data_inputs": ["geom"],
  "missing_data_strategy": "skip",
  "n_workers": 8,
  "batch_size": 500
}
```

### Adding Features

To add atom/bond/global features, list them in both `keys_data` and `keys_target`:

**Atom features** (per-atom properties):
- From QTAIM: various critical point properties
- From charge LMDBs: partial charges
- From fuzzy LMDBs: atomic densities

**Bond features** (per-bond properties):
- From bond LMDB: fuzzy bond orders, IBSI values
- From QTAIM: bond path properties

**Global features** (per-molecule properties):
- Always include: `"n_atoms"`
- Optional: dipole magnitudes, molecular properties

**Example** - Adding Hirshfeld charges and fuzzy density as atom features:

```json
{
  "keys_data": {
    "atom": ["hirshfeld_charge", "becke_fuzzy_density"],
    "bond": [],
    "global": ["n_atoms"]
  }
}
```

## Troubleshooting

### "AssertionError: The config file must contain a key 'X_lmdb'"

**Fix**: Add the required LMDB to `lmdb_locations`:

```json
{
  "lmdb_locations": {
    "geom_lmdb": "/path/to/structure.lmdb",
    "qtaim_lmdb": "/path/to/qtaim.lmdb"  // Add this
  }
}
```

### "KeyError: 'fuzzy_bond'" or similar

**Fix**: Ensure the requested descriptor exists in your data:

```python
# Check what's available
import lmdb, pickle
env = lmdb.open('/path/to/bond.lmdb', readonly=True, subdir=False, lock=False)
with env.begin() as txn:
    cursor = txn.cursor()
    cursor.first()
    cursor.next()  # Skip 'length' key
    key, value = next(cursor)
    data = pickle.loads(value)
    print("Available descriptors:", list(data.keys()))
env.close()
```

### Graphs missing expected features

**Check**:
1. Feature is listed in `keys_data`
2. Corresponding filter includes the descriptor (e.g., `"bond_filter": ["fuzzy"]`)
3. Source LMDB contains the data
4. `missing_data_strategy` is set appropriately

### Performance issues

**Tips**:
- Use `n_workers=1` for best performance (GIL limitations)
- Set `batch_size` to balance memory and write frequency (500 is good default)
- For large datasets, use sharding (see SHARDING_GUIDE.md)

## References

- Main converter code: `qtaim_gen/source/core/converter.py`
- Test suite: `tests/test_general_converter_filters.py`
- Sharding guide: `docs/SHARDING_GUIDE.md`
- Example usage: `examples/` directory
