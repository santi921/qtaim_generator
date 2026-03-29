"""Tests for multi-vertical pipeline utilities."""

import json
import os
import pickle
import tempfile

import lmdb
import pytest

from qtaim_gen.source.utils.splits import (
    SplitConfig,
    SPLIT_NAMES,
    assign_formula_to_split,
)
from qtaim_gen.source.utils.multi_vertical import (
    MultiVerticalPipelineConfig,
    VerticalConfig,
    SplitPlan,
    load_pipeline_config,
    plan_phase,
    validate_schema_compatibility,
)


# ---------------------------------------------------------------------------
# assign_formula_to_split (extracted function)
# ---------------------------------------------------------------------------

class TestAssignFormulaToSplit:
    def test_deterministic(self):
        result1 = assign_formula_to_split("C6H12O6", (0.8, 0.1, 0.1), seed=42)
        result2 = assign_formula_to_split("C6H12O6", (0.8, 0.1, 0.1), seed=42)
        assert result1 == result2

    def test_returns_valid_split_name(self):
        for formula in ["H2O", "CH4", "C6H12O6", "NaCl", "Fe2O3"]:
            result = assign_formula_to_split(formula, (0.8, 0.1, 0.1), seed=42)
            assert result in SPLIT_NAMES

    def test_different_seed_can_change_assignment(self):
        # With enough formulas, at least one should differ between seeds
        results_42 = [
            assign_formula_to_split(f"mol_{i}", (0.8, 0.1, 0.1), seed=42)
            for i in range(50)
        ]
        results_99 = [
            assign_formula_to_split(f"mol_{i}", (0.8, 0.1, 0.1), seed=99)
            for i in range(50)
        ]
        assert results_42 != results_99

    def test_same_formula_same_split_regardless_of_context(self):
        """Core guarantee: same formula always maps to same split."""
        split1 = assign_formula_to_split("C6H12O6", (0.8, 0.1, 0.1), seed=42)
        split2 = assign_formula_to_split("C6H12O6", (0.8, 0.1, 0.1), seed=42)
        assert split1 == split2

    def test_custom_split_names(self):
        result = assign_formula_to_split(
            "H2O", (0.5, 0.5), seed=42, split_names=("a", "b")
        )
        assert result in ("a", "b")


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------

class TestVerticalConfig:
    def test_valid(self):
        vc = VerticalConfig(name="spice", converter_config="/tmp/cfg.json")
        assert vc.name == "spice"

    def test_underscore_allowed(self):
        vc = VerticalConfig(name="my_dataset_v2", converter_config="/tmp/cfg.json")
        assert vc.name == "my_dataset_v2"

    def test_invalid_name_special_chars(self):
        with pytest.raises(ValueError, match="alphanumeric"):
            VerticalConfig(name="my-dataset", converter_config="/tmp/cfg.json")

    def test_invalid_name_spaces(self):
        with pytest.raises(ValueError, match="alphanumeric"):
            VerticalConfig(name="my dataset", converter_config="/tmp/cfg.json")


class TestMultiVerticalPipelineConfig:
    def _make_config(self, **overrides):
        defaults = dict(
            output_dir="/tmp/out",
            verticals=(
                VerticalConfig(name="a", converter_config="/tmp/a.json"),
                VerticalConfig(name="b", converter_config="/tmp/b.json"),
            ),
            split_config=SplitConfig(method="composition", ratios=(0.8, 0.1, 0.1)),
        )
        defaults.update(overrides)
        return MultiVerticalPipelineConfig(**defaults)

    def test_valid(self):
        cfg = self._make_config()
        assert len(cfg.verticals) == 2

    def test_empty_verticals(self):
        with pytest.raises(ValueError, match="At least one"):
            self._make_config(verticals=())

    def test_duplicate_names(self):
        with pytest.raises(ValueError, match="unique"):
            self._make_config(verticals=(
                VerticalConfig(name="a", converter_config="/tmp/a.json"),
                VerticalConfig(name="a", converter_config="/tmp/b.json"),
            ))

    def test_invalid_shards(self):
        with pytest.raises(ValueError, match="n_shards_per_split"):
            self._make_config(n_shards_per_split=0)


# ---------------------------------------------------------------------------
# load_pipeline_config
# ---------------------------------------------------------------------------

class TestLoadPipelineConfig:
    def test_loads_valid_json(self, tmp_path):
        config = {
            "output_dir": str(tmp_path / "out"),
            "verticals": [
                {"name": "spice", "converter_config": "/tmp/spice.json"},
                {"name": "qm9", "converter_config": "/tmp/qm9.json"},
            ],
            "split_method": "composition",
            "split_ratios": [0.8, 0.1, 0.1],
            "split_seed": 42,
        }
        config_path = str(tmp_path / "pipeline.json")
        with open(config_path, "w") as f:
            json.dump(config, f)

        result = load_pipeline_config(config_path)
        assert isinstance(result, MultiVerticalPipelineConfig)
        assert len(result.verticals) == 2
        assert result.verticals[0].name == "spice"
        assert result.split_config.method == "composition"
        assert result.n_shards_per_split == 1

    def test_with_optional_shards(self, tmp_path):
        config = {
            "output_dir": str(tmp_path / "out"),
            "verticals": [{"name": "a", "converter_config": "/tmp/a.json"}],
            "split_method": "random",
            "split_ratios": [0.8, 0.1, 0.1],
            "split_seed": 99,
            "n_shards_per_split": 4,
        }
        config_path = str(tmp_path / "pipeline.json")
        with open(config_path, "w") as f:
            json.dump(config, f)

        result = load_pipeline_config(config_path)
        assert result.n_shards_per_split == 4
        assert result.split_config.seed == 99


# ---------------------------------------------------------------------------
# Schema compatibility validation
# ---------------------------------------------------------------------------

class TestSchemaCompatibility:
    def test_identical_configs_pass(self):
        configs = {
            "a": {
                "keys_data": {"atom": ["x"], "bond": [], "global": []},
                "keys_target": {"atom": [], "bond": [], "global": ["e"]},
                "bonding_scheme": "structural",
                "charge_filter": None,
                "fuzzy_filter": None,
                "bond_filter": None,
                "other_filter": None,
            },
            "b": {
                "keys_data": {"atom": ["x"], "bond": [], "global": []},
                "keys_target": {"atom": [], "bond": [], "global": ["e"]},
                "bonding_scheme": "structural",
                "charge_filter": None,
                "fuzzy_filter": None,
                "bond_filter": None,
                "other_filter": None,
            },
        }
        validate_schema_compatibility(configs)  # should not raise

    def test_mismatched_bonding_scheme(self):
        configs = {
            "a": {"keys_data": {}, "keys_target": {}, "bonding_scheme": "structural"},
            "b": {"keys_data": {}, "keys_target": {}, "bonding_scheme": "qtaim"},
        }
        with pytest.raises(ValueError, match="bonding_scheme"):
            validate_schema_compatibility(configs)

    def test_mismatched_keys_data(self):
        configs = {
            "a": {"keys_data": {"atom": ["x"]}, "keys_target": {}, "bonding_scheme": "s"},
            "b": {"keys_data": {"atom": ["y"]}, "keys_target": {}, "bonding_scheme": "s"},
        }
        with pytest.raises(ValueError, match="keys_data"):
            validate_schema_compatibility(configs)

    def test_single_vertical_always_passes(self):
        configs = {"a": {"keys_data": {"atom": ["x"]}}}
        validate_schema_compatibility(configs)  # should not raise


# ---------------------------------------------------------------------------
# Plan phase (integration with real-ish LMDB fixtures)
# ---------------------------------------------------------------------------

class _FakeElements:
    """Picklable stand-in for species.elements — str() returns '[Element X]'."""
    def __init__(self, symbol):
        self._symbol = symbol

    def __repr__(self):
        return f"[Element {self._symbol}]"

    def __str__(self):
        return f"[Element {self._symbol}]"


class _FakeSpecies:
    """Picklable stand-in for pymatgen Species (for element extraction)."""
    def __init__(self, symbol):
        self._symbol = symbol
        self.elements = _FakeElements(symbol)


class _FakeSite:
    """Picklable stand-in for pymatgen Site."""
    def __init__(self, symbol):
        self.species = _FakeSpecies(symbol)


class _FakeComposition:
    """Picklable stand-in for pymatgen Composition."""
    def __init__(self, formula):
        self.formula = formula


class _FakeMolecule(list):
    """Picklable stand-in for pymatgen Molecule.

    Acts as a list of sites (for get_elements_from_structure_lmdb iteration)
    and has .composition.formula (for build_formula_map_from_structure_lmdb).
    """
    def __init__(self, formula, symbols):
        sites = [_FakeSite(s) for s in symbols]
        super().__init__(sites)
        self.composition = _FakeComposition(formula)


class _FakeMoleculeGraph:
    """Picklable stand-in for pymatgen MoleculeGraph."""
    def __init__(self, molecule):
        self.molecule = molecule


def _create_structure_lmdb(path: str, entries: dict[str, str]):
    """Create a minimal structure LMDB with {key: formula} entries.

    Mimics the real format expected by:
    - build_formula_map_from_structure_lmdb: value["molecule_graph"].molecule.composition.formula
    - get_elements_from_structure_lmdb: iterates value["molecule"] sites, reads site.species.elements
    """
    env = lmdb.open(path, map_size=int(1099511627776), subdir=False,
                     meminit=False, map_async=True)
    with env.begin(write=True) as txn:
        for key, formula in entries.items():
            symbols = _formula_to_symbols(formula)
            mol = _FakeMolecule(formula, symbols)
            mol_graph = _FakeMoleculeGraph(mol)
            # value["molecule_graph"] for formula extraction
            # value["molecule"] for element set extraction (iterable of sites)
            value = {"molecule_graph": mol_graph, "molecule": mol}
            txn.put(key.encode("ascii"), pickle.dumps(value, protocol=-1))
        txn.put(b"length", pickle.dumps(len(entries), protocol=-1))
    env.sync()
    env.close()


def _formula_to_symbols(formula: str) -> list[str]:
    """Rough formula->element symbols for test fixtures.

    Handles space-separated format like 'H2 O1' and compact like 'CH4'.
    """
    symbols = []
    # Handle space-separated format: "H2 O1"
    parts = formula.split()
    for part in parts:
        sym = ""
        count_str = ""
        for ch in part:
            if ch.isalpha():
                sym += ch
            elif ch.isdigit():
                count_str += ch
        if sym:
            count = int(count_str) if count_str else 1
            symbols.extend([sym] * count)
    return symbols if symbols else ["H"]


def _create_converter_config(
    tmp_path, name: str, geom_lmdb_path: str, **overrides
) -> str:
    """Create a minimal converter config JSON for testing."""
    config = {
        "lmdb_path": str(tmp_path / name),
        "lmdb_name": "graphs.lmdb",
        "lmdb_locations": {"geom_lmdb": geom_lmdb_path},
        "keys_data": {"atom": ["eta"], "bond": [], "global": ["n_atoms"]},
        "keys_target": {"atom": [], "bond": [], "global": []},
        "bonding_scheme": "structural",
        "charge_filter": None,
        "fuzzy_filter": None,
        "bond_filter": None,
        "other_filter": None,
        "restart": False,
    }
    config.update(overrides)
    config_path = str(tmp_path / f"{name}_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f)
    return config_path


class TestPlanPhase:
    def test_basic_plan(self, tmp_path):
        """Two verticals with overlapping formulas produce consistent splits."""
        # Create structure LMDBs
        spice_lmdb = str(tmp_path / "spice_geom.lmdb")
        _create_structure_lmdb(spice_lmdb, {
            "s0": "H2 O1", "s1": "C1 H4", "s2": "C6 H12 O6", "s3": "N1 H3",
        })
        qm9_lmdb = str(tmp_path / "qm9_geom.lmdb")
        _create_structure_lmdb(qm9_lmdb, {
            "q0": "H2 O1", "q1": "C1 H4", "q2": "C2 H6",
        })

        # Create converter configs
        spice_cfg = _create_converter_config(tmp_path, "spice", spice_lmdb)
        qm9_cfg = _create_converter_config(tmp_path, "qm9", qm9_lmdb)

        # Build pipeline config
        config = MultiVerticalPipelineConfig(
            output_dir=str(tmp_path / "output"),
            verticals=(
                VerticalConfig(name="spice", converter_config=spice_cfg),
                VerticalConfig(name="qm9", converter_config=qm9_cfg),
            ),
            split_config=SplitConfig(method="composition", ratios=(0.8, 0.1, 0.1)),
        )

        plan = plan_phase(config)

        # Basic structure checks
        assert isinstance(plan, SplitPlan)
        assert "spice" in plan.assignment
        assert "qm9" in plan.assignment
        assert len(plan.global_element_set) > 0

        # All keys assigned
        all_spice_keys = set()
        all_qm9_keys = set()
        for split in SPLIT_NAMES:
            all_spice_keys.update(plan.assignment["spice"][split])
            all_qm9_keys.update(plan.assignment["qm9"][split])
        assert all_spice_keys == {"s0", "s1", "s2", "s3"}
        assert all_qm9_keys == {"q0", "q1", "q2"}

        # Composition consistency: H2O keys in same split across verticals
        h2o_split_spice = None
        h2o_split_qm9 = None
        for split in SPLIT_NAMES:
            if "s0" in plan.assignment["spice"][split]:
                h2o_split_spice = split
            if "q0" in plan.assignment["qm9"][split]:
                h2o_split_qm9 = split
        assert h2o_split_spice == h2o_split_qm9, \
            f"H2O in different splits: spice={h2o_split_spice}, qm9={h2o_split_qm9}"

        # CH4 keys in same split across verticals
        ch4_split_spice = None
        ch4_split_qm9 = None
        for split in SPLIT_NAMES:
            if "s1" in plan.assignment["spice"][split]:
                ch4_split_spice = split
            if "q1" in plan.assignment["qm9"][split]:
                ch4_split_qm9 = split
        assert ch4_split_spice == ch4_split_qm9, \
            f"CH4 in different splits: spice={ch4_split_spice}, qm9={ch4_split_qm9}"

    def test_split_plan_json_written(self, tmp_path):
        """Verify split_plan.json is written to output_dir."""
        lmdb_path = str(tmp_path / "geom.lmdb")
        _create_structure_lmdb(lmdb_path, {"k0": "H2 O1", "k1": "C1 H4"})
        cfg_path = _create_converter_config(tmp_path, "v1", lmdb_path)

        config = MultiVerticalPipelineConfig(
            output_dir=str(tmp_path / "output"),
            verticals=(VerticalConfig(name="v1", converter_config=cfg_path),),
            split_config=SplitConfig(method="composition", ratios=(0.8, 0.1, 0.1)),
        )

        plan_phase(config)

        plan_file = tmp_path / "output" / "split_plan.json"
        assert plan_file.exists()
        with open(plan_file) as f:
            data = json.load(f)
        assert "assignment" in data
        assert "summary" in data
        assert data["summary"]["total_keys"] == 2

    def test_schema_mismatch_raises(self, tmp_path):
        """Verticals with different bonding schemes should fail validation."""
        lmdb_path = str(tmp_path / "geom.lmdb")
        _create_structure_lmdb(lmdb_path, {"k0": "H2 O1"})

        cfg1 = _create_converter_config(tmp_path, "v1", lmdb_path, bonding_scheme="structural")
        cfg2 = _create_converter_config(tmp_path, "v2", lmdb_path, bonding_scheme="qtaim")

        config = MultiVerticalPipelineConfig(
            output_dir=str(tmp_path / "output"),
            verticals=(
                VerticalConfig(name="v1", converter_config=cfg1),
                VerticalConfig(name="v2", converter_config=cfg2),
            ),
            split_config=SplitConfig(method="composition", ratios=(0.8, 0.1, 0.1)),
        )

        with pytest.raises(ValueError, match="bonding_scheme"):
            plan_phase(config)

    def test_missing_geom_lmdb_raises(self, tmp_path):
        """Missing structure LMDB should fail with clear error."""
        cfg_path = _create_converter_config(
            tmp_path, "v1", "/nonexistent/geom.lmdb"
        )
        config = MultiVerticalPipelineConfig(
            output_dir=str(tmp_path / "output"),
            verticals=(VerticalConfig(name="v1", converter_config=cfg_path),),
            split_config=SplitConfig(method="composition", ratios=(0.8, 0.1, 0.1)),
        )
        with pytest.raises(FileNotFoundError, match="v1"):
            plan_phase(config)

    def test_element_set_union(self, tmp_path):
        """Global element set is the union across verticals."""
        # v1 has H, O; v2 has H, C
        lmdb1 = str(tmp_path / "v1_geom.lmdb")
        _create_structure_lmdb(lmdb1, {"k0": "H2 O1"})
        lmdb2 = str(tmp_path / "v2_geom.lmdb")
        _create_structure_lmdb(lmdb2, {"k0": "C1 H4"})

        cfg1 = _create_converter_config(tmp_path, "v1", lmdb1)
        cfg2 = _create_converter_config(tmp_path, "v2", lmdb2)

        config = MultiVerticalPipelineConfig(
            output_dir=str(tmp_path / "output"),
            verticals=(
                VerticalConfig(name="v1", converter_config=cfg1),
                VerticalConfig(name="v2", converter_config=cfg2),
            ),
            split_config=SplitConfig(method="composition", ratios=(0.8, 0.1, 0.1)),
        )

        plan = plan_phase(config)
        # Should have H, C, O (element symbols from both verticals)
        assert "H" in plan.global_element_set
        assert "C" in plan.global_element_set
        assert "O" in plan.global_element_set
