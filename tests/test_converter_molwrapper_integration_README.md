# Converter MoleculeWrapper Integration Tests

## Overview

This test suite ([test_converter_molwrapper_integration.py](test_converter_molwrapper_integration.py)) validates that the converter classes correctly produce `MoleculeWrapper` objects from LMDB data. These are **integration tests** that require the `qtaim-embed` dependency to run.

## What These Tests Verify

### 1. BaseConverter Tests
- Creates valid `MoleculeWrapper` instances
- Includes correct global features (n_atoms, etc.)
- Preserves bond connectivity from coordinate-based detection

### 2. QTAIMConverter Tests
- Creates `MoleculeWrapper` with QTAIM data
- Includes QTAIM atom features (eta, lol, density_alpha, density_beta, etc.)
- Uses QTAIM bond paths for connectivity

### 3. GeneralConverter Tests
- **Charge Features**: Correctly parses and filters charge types (mbis, hirshfeld, etc.)
- **Fuzzy Features**: Includes fuzzy density descriptors (mbis_fuzzy_density, elf_fuzzy, etc.)
- **Bond Features**: Includes bond order data (ibsi, fuzzy, etc.)
- **Global Features**: Includes molecular descriptors (mpp_full, sdp_full, etc.)
- **Bonding Schemes**: Supports multiple bonding definitions (structural, qtaim, bonding)

### 4. End-to-End Tests
- **Roundtrip**: MoleculeWrapper → DGL Graph conversion works correctly
- **Consistency**: Multiple molecules have consistent feature structure

## Requirements

These tests require:
- `qtaim-embed` package (contains `MoleculeWrapper` and graph utilities)
- Test LMDB fixtures in `tests/test_files/lmdb_tests/`

## Running the Tests

### Standard Run
```bash
# Run all integration tests
pytest tests/test_converter_molwrapper_integration.py -v

# Run specific test class
pytest tests/test_converter_molwrapper_integration.py::TestGeneralConverterMolWrapper -v

# Run specific test
pytest tests/test_converter_molwrapper_integration.py::TestGeneralConverterMolWrapper::test_general_converter_charge_features -v
```

### With qtaim-embed Not Installed
If `qtaim-embed` is not installed, all tests will be **automatically skipped** with a clear message:
```
SKIPPED [100%] - qtaim-embed dependency not available. Install with: pip install qtaim-embed
```

### Installing qtaim-embed
To run these tests, install qtaim-embed:
```bash
pip install qtaim-embed
# or from source
git clone https://github.com/santi921/qtaim_embed
cd qtaim_embed
pip install -e .
```

## Test Structure

```
test_converter_molwrapper_integration.py
├── Fixtures
│   ├── test_paths()                    # Paths to test LMDBs
│   ├── _base_converter_config()        # BaseConverter config
│   ├── _qtaim_converter_config()       # QTAIMConverter config
│   └── _general_converter_config()     # GeneralConverter config
│
├── TestBaseConverterMolWrapper
│   ├── test_base_converter_creates_molwrapper
│   ├── test_base_converter_global_features
│   └── test_base_converter_bonds
│
├── TestQTAIMConverterMolWrapper
│   ├── test_qtaim_converter_creates_molwrapper
│   ├── test_qtaim_converter_atom_features
│   └── test_qtaim_converter_bond_paths
│
├── TestGeneralConverterMolWrapper
│   ├── test_general_converter_creates_molwrapper
│   ├── test_general_converter_charge_features
│   ├── test_general_converter_fuzzy_features
│   ├── test_general_converter_bond_features
│   ├── test_general_converter_global_features
│   ├── test_general_converter_bonding_scheme_structural
│   └── test_general_converter_bonding_scheme_qtaim
│
└── TestConverterEndToEnd
    ├── test_molwrapper_to_graph_roundtrip
    └── test_multiple_molecules_consistency
```

## Key Test Patterns

### Extracting MoleculeWrapper Objects
Tests use `_extract_molwrapper_from_converter()` to get MoleculeWrapper instances before they're serialized:

```python
converter = GeneralConverter(config, config_path=str(tmp_path / "config.json"))
key_str, mol_wrapper, failures = _extract_molwrapper_from_converter(converter)

# Now inspect mol_wrapper attributes
assert mol_wrapper.atom_features is not None
assert "charge_mbis" in mol_wrapper.atom_features[0]
```

### Testing Feature Filtering
Tests verify that filters correctly include/exclude features:

```python
config["charge_filter"] = ["mbis", "hirshfeld"]
converter = GeneralConverter(config, ...)

# Should have filtered types
assert "charge_mbis" in atom_feats
assert "charge_hirshfeld" in atom_feats

# Should NOT have unfiltered types
assert "charge_bader" not in atom_feats
```

### Testing Bonding Schemes
Tests verify different bonding scheme behaviors:

```python
# Structural bonding: coordinate-based bonds
config["bonding_scheme"] = "structural"

# QTAIM bonding: use QTAIM bond paths
config["bonding_scheme"] = "qtaim"

# Bonding: use bond order data (ibsi, fuzzy, etc.)
config["bonding_scheme"] = "bonding"
config["bond_list_definition"] = "ibsi"
```

## Why Integration Tests?

These are **integration tests** (not unit tests) because they:
1. Require the `qtaim-embed` external dependency
2. Test the full pipeline from LMDB → parsing → MoleculeWrapper
3. Use real test fixtures (not mocks)
4. Verify behavior of multiple components working together

This ensures that:
- Data flows correctly through the conversion pipeline
- Features are parsed and assigned to the right atoms/bonds
- The MoleculeWrapper objects are ML-ready for downstream use
- Breaking changes to parsers or converters are caught early

## Test Data

Tests use fixtures in `tests/test_files/lmdb_tests/`:
- `generator_lmdbs_merged/`: Merged LMDB files with all feature types
  - `merged_geom.lmdb`: Molecular structures
  - `merged_charge.lmdb`: Partial charges (mbis, hirshfeld, bader, etc.)
  - `merged_fuzzy.lmdb`: Fuzzy density descriptors
  - `merged_bond.lmdb`: Bond orders (ibsi, fuzzy, etc.)
  - `merged_qtaim.lmdb`: QTAIM critical point data
  - `merged_other.lmdb`: Global descriptors (mpp, sdp, ESP, etc.)

## Debugging Failed Tests

If a test fails:

1. **Check the error message**: Tests provide detailed assertions
   ```
   AssertionError: Should have charge_mbis. Found: ['charge_hirshfeld']
   ```

2. **Run with verbose output**:
   ```bash
   pytest tests/test_converter_molwrapper_integration.py::test_name -vv
   ```

3. **Inspect the MoleculeWrapper**:
   ```python
   # Add to test for debugging
   print(f"Atom features: {list(mol_wrapper.atom_features[0].keys())}")
   print(f"Bond features: {list(mol_wrapper.bond_features.keys())}")
   print(f"Global features: {list(mol_wrapper.global_features.keys())}")
   ```

4. **Check converter config**: Verify filters and settings
   ```python
   print(f"Charge filter: {config['charge_filter']}")
   print(f"Bonding scheme: {config['bonding_scheme']}")
   ```

## Adding New Tests

To add a new test:

1. **Follow the naming convention**: `test_<converter>_<feature>`
   ```python
   def test_general_converter_dipole_features(self, tmp_path, test_paths):
       """Test that GeneralConverter includes dipole features."""
       ...
   ```

2. **Use the helper fixtures**:
   ```python
   config = _general_converter_config(tmp_path, test_paths)
   config["other_filter"] = ["dipole_x", "dipole_y", "dipole_z"]
   ```

3. **Extract and inspect MoleculeWrapper**:
   ```python
   converter = GeneralConverter(config, ...)
   key_str, mol_wrapper, failures = _extract_molwrapper_from_converter(converter)

   # Add your assertions
   assert "other_dipole_x" in mol_wrapper.global_features
   ```

4. **Add skip decorators if needed**:
   ```python
   @pytest.mark.skipif(
       not os.path.exists(...),
       reason="Special test fixtures not available"
   )
   ```

## CI/CD Integration

These tests can be integrated into CI/CD pipelines:

```yaml
# .github/workflows/test.yml
- name: Install qtaim-embed
  run: pip install qtaim-embed

- name: Run integration tests
  run: pytest tests/test_converter_molwrapper_integration.py -v
```

Or run them separately from unit tests:
```bash
# Run only unit tests (fast)
pytest tests/ -m "not integration"

# Run integration tests (requires dependencies)
pytest tests/test_converter_molwrapper_integration.py -v
```

## Related Files

- `qtaim_gen/source/core/converter.py`: Converter implementations
- `qtaim_gen/source/utils/lmdbs.py`: LMDB parsing utilities
- `tests/test_general_converter_filters.py`: Unit tests for filter logic
- `tests/test_sharded_converter.py`: Integration tests for sharded processing
