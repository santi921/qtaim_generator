"""
Multi-vertical pipeline config and planning utilities.

Provides dataclasses for pipeline configuration and the Plan phase logic:
validation, composition census, element set unification, and global split
assignment across multiple dataset verticals.
"""

import json
import os
import pickle
from dataclasses import dataclass, field

import lmdb

from qtaim_gen.source.utils.lmdbs import (
    get_elements_from_structure_lmdb,
    parse_config_gen_to_embed,
)
from qtaim_gen.source.utils.splits import (
    METADATA_KEYS,
    SPLIT_NAMES,
    SplitConfig,
    assign_formula_to_split,
    build_formula_map_from_structure_lmdb,
)

# Type alias for the split assignment structure
SplitAssignment = dict[str, dict[str, list[str]]]  # {vertical: {split: [keys]}}


@dataclass(frozen=True)
class VerticalConfig:
    name: str
    converter_config: str  # path to existing converter JSON

    def __post_init__(self):
        if not self.name.replace("_", "").isalnum():
            raise ValueError(
                f"Vertical name must be alphanumeric/underscore, got '{self.name}'"
            )


@dataclass(frozen=True)
class MultiVerticalPipelineConfig:
    output_dir: str
    verticals: tuple[VerticalConfig, ...]
    split_config: SplitConfig
    n_shards_per_split: int = 1

    def __post_init__(self):
        if len(self.verticals) == 0:
            raise ValueError("At least one vertical is required")
        names = [v.name for v in self.verticals]
        if len(names) != len(set(names)):
            raise ValueError(f"Vertical names must be unique, got {names}")
        if self.n_shards_per_split < 1:
            raise ValueError(f"n_shards_per_split must be >= 1, got {self.n_shards_per_split}")


def load_pipeline_config(config_path: str) -> MultiVerticalPipelineConfig:
    """Load and validate a multi-vertical pipeline config from JSON."""
    with open(config_path) as f:
        raw = json.load(f)

    verticals = tuple(
        VerticalConfig(name=v["name"], converter_config=v["converter_config"])
        for v in raw["verticals"]
    )
    split_config = SplitConfig(
        method=raw["split_method"],
        ratios=tuple(raw["split_ratios"]),
        seed=raw["split_seed"],
    )
    return MultiVerticalPipelineConfig(
        output_dir=raw["output_dir"],
        verticals=verticals,
        split_config=split_config,
        n_shards_per_split=raw.get("n_shards_per_split", 1),
    )


# ---------- Plan phase types ----------

@dataclass
class SplitPlan:
    """Output of Phase 1 (Plan). Typed result passed directly to Phase 2."""
    assignment: SplitAssignment
    global_element_set: list[int]
    converter_configs: dict[str, dict]  # {vertical_name: parsed config dict}
    summary: dict  # stats for split_plan.json


# ---------- Validation ----------

# Keys that must match across all verticals for graph schema compatibility
_SCHEMA_KEYS = ("keys_data", "keys_target", "bonding_scheme")
_FILTER_KEYS = ("charge_filter", "fuzzy_filter", "bond_filter", "other_filter")


def validate_schema_compatibility(
    converter_configs: dict[str, dict],
) -> None:
    """Check that all vertical converter configs produce compatible graph schemas.

    Raises ValueError with details on first mismatch.
    """
    names = list(converter_configs.keys())
    if len(names) < 2:
        return  # nothing to compare

    ref_name = names[0]
    ref_cfg = converter_configs[ref_name]

    for name in names[1:]:
        cfg = converter_configs[name]
        for key in _SCHEMA_KEYS:
            ref_val = ref_cfg.get(key)
            cur_val = cfg.get(key)
            if ref_val != cur_val:
                raise ValueError(
                    f"Schema mismatch on '{key}' between verticals "
                    f"'{ref_name}' and '{name}':\n"
                    f"  {ref_name}: {ref_val}\n"
                    f"  {name}: {cur_val}"
                )
        for key in _FILTER_KEYS:
            ref_val = ref_cfg.get(key)
            cur_val = cfg.get(key)
            if ref_val != cur_val:
                raise ValueError(
                    f"Filter mismatch on '{key}' between verticals "
                    f"'{ref_name}' and '{name}':\n"
                    f"  {ref_name}: {ref_val}\n"
                    f"  {name}: {cur_val}"
                )


# ---------- Plan phase ----------

def plan_phase(config: MultiVerticalPipelineConfig) -> SplitPlan:
    """Phase 1: Validate, census, assign splits.

    Returns a typed SplitPlan and writes split_plan.json to output_dir.
    """
    # 1. Parse and validate all converter configs
    converter_configs: dict[str, dict] = {}
    for vert in config.verticals:
        if not os.path.isfile(vert.converter_config):
            raise FileNotFoundError(
                f"Converter config for vertical '{vert.name}' not found: "
                f"{vert.converter_config}"
            )
        cfg = parse_config_gen_to_embed(vert.converter_config)
        converter_configs[vert.name] = cfg

    validate_schema_compatibility(converter_configs)

    # 2. Census: build formula maps and collect element sets per vertical
    all_formula_maps: dict[str, dict[str, str]] = {}
    all_element_sets: list[set] = []

    for vert in config.verticals:
        cfg = converter_configs[vert.name]
        geom_path = cfg["lmdb_locations"]["geom_lmdb"]
        if not os.path.exists(geom_path):
            raise FileNotFoundError(
                f"Structure LMDB for vertical '{vert.name}' not found: {geom_path}"
            )

        print(f"Census: reading {vert.name} from {geom_path}...")
        formula_map = build_formula_map_from_structure_lmdb(geom_path)
        all_formula_maps[vert.name] = formula_map
        print(f"  {vert.name}: {len(formula_map)} keys, "
              f"{len(set(formula_map.values()))} unique formulas")

        # Element set from the LMDB env
        env = lmdb.open(
            geom_path, subdir=False, readonly=True, lock=False,
            readahead=True, meminit=False,
        )
        elems = get_elements_from_structure_lmdb(env)
        env.close()
        all_element_sets.append(set(elems))

    # Global element set (union)
    global_element_set = sorted(set().union(*all_element_sets))
    print(f"Global element set ({len(global_element_set)} elements): {global_element_set}")

    # 3. Split assignment
    assignment: SplitAssignment = {}
    total_per_split: dict[str, int] = {s: 0 for s in SPLIT_NAMES}

    for vert in config.verticals:
        fmap = all_formula_maps[vert.name]
        vert_assignment: dict[str, list[str]] = {s: [] for s in SPLIT_NAMES}

        for key, formula in fmap.items():
            split = assign_formula_to_split(
                formula, config.split_config.ratios, config.split_config.seed
            )
            vert_assignment[split].append(key)

        assignment[vert.name] = vert_assignment
        for s in SPLIT_NAMES:
            total_per_split[s] += len(vert_assignment[s])

    # Summary stats
    total_keys = sum(total_per_split.values())
    actual_ratios = {
        s: total_per_split[s] / total_keys if total_keys > 0 else 0.0
        for s in SPLIT_NAMES
    }
    requested_ratios = dict(zip(SPLIT_NAMES, config.split_config.ratios))
    deviations = {
        s: abs(actual_ratios[s] - requested_ratios[s]) for s in SPLIT_NAMES
    }

    # Per-vertical breakdown
    per_vertical_counts = {}
    for vert_name, vert_assign in assignment.items():
        per_vertical_counts[vert_name] = {s: len(vert_assign[s]) for s in SPLIT_NAMES}

    summary = {
        "total_keys": total_keys,
        "per_split_counts": total_per_split,
        "actual_ratios": {s: round(v, 4) for s, v in actual_ratios.items()},
        "requested_ratios": {s: round(v, 4) for s, v in requested_ratios.items()},
        "deviations": {s: round(v, 4) for s, v in deviations.items()},
        "per_vertical_counts": per_vertical_counts,
        "global_element_set": global_element_set,
        "split_method": config.split_config.method,
        "split_seed": config.split_config.seed,
    }

    # Print summary
    print(f"\nSplit assignment ({config.split_config.method}, seed={config.split_config.seed}):")
    for s in SPLIT_NAMES:
        pct = actual_ratios[s] * 100
        dev = deviations[s] * 100
        warn = " WARNING: >5pp deviation!" if dev > 5.0 else ""
        print(f"  {s}: {total_per_split[s]} ({pct:.1f}%){warn}")
    for vert_name, counts in per_vertical_counts.items():
        print(f"  {vert_name}: {counts}")

    # Write split_plan.json
    os.makedirs(config.output_dir, exist_ok=True)
    plan_data = {
        "assignment": assignment,
        "summary": summary,
    }
    plan_path = os.path.join(config.output_dir, "split_plan.json")
    with open(plan_path, "w") as f:
        json.dump(plan_data, f, indent=2)
    print(f"\nWrote split plan to {plan_path}")

    return SplitPlan(
        assignment=assignment,
        global_element_set=global_element_set,
        converter_configs=converter_configs,
        summary=summary,
    )
