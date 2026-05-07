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
    exclude_keys_path: str = ""  # optional manifest_holdout.parquet path

    def __post_init__(self):
        if len(self.verticals) == 0:
            raise ValueError("At least one vertical is required")
        names = [v.name for v in self.verticals]
        if len(names) != len(set(names)):
            raise ValueError(f"Vertical names must be unique, got {names}")
        if self.n_shards_per_split < 1:
            raise ValueError(f"n_shards_per_split must be >= 1, got {self.n_shards_per_split}")
        if self.exclude_keys_path and not os.path.isfile(self.exclude_keys_path):
            raise FileNotFoundError(
                f"exclude_keys_path is set but file not found: {self.exclude_keys_path}"
            )


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
        exclude_keys_path=raw.get("exclude_keys_path", ""),
    )


def _manifest_rel_path_to_lmdb_key(vertical: str, rel_path: str) -> str:
    """Mirror of pull_holdout_records.manifest_rel_path_to_lmdb_key.

    Manifest rel_path is `<vertical>/<path with / separators>`; LMDB keys
    use the jagged-hierarchy convention with `__` separators after a
    per-vertical root strip. Strip a leading `<vertical>/` if present,
    then replace remaining `/` with `__`.

    Duplicated here (rather than imported from scripts.helpers) because
    utils should not depend on scripts. The two definitions must stay in
    sync.
    """
    prefix = f"{vertical}/"
    if rel_path.startswith(prefix):
        rel_path = rel_path[len(prefix):]
    return rel_path.replace("/", "__")


def load_exclusion_set(parquet_path: str) -> dict[str, set[str]]:
    """Load `manifest_holdout.parquet` into a per-vertical set of LMDB keys.

    Reads the 'vertical' and 'rel_path' columns and re-derives the LMDB
    key per row (the same transformation pull_holdout_records uses to
    look up records during the pull). The parquet's 'key' column stores
    the cross-vertical merged form `{vertical}__{rel_path}` and is NOT
    suitable as a direct LMDB lookup key for some verticals (manifest
    rel_paths can include `/` separators that get converted to `__` in
    the LMDB key).

    Returns `{vertical: set(lmdb_key)}` keyed by source vertical name.
    """
    # Local import: pandas is heavy and only needed when exclusion is requested.
    import pandas as pd

    df = pd.read_parquet(parquet_path)
    required = {"vertical", "rel_path"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"exclude_keys_path {parquet_path} missing columns {missing}; "
            f"got {list(df.columns)}"
        )

    by_vert: dict[str, set[str]] = {}
    for v, rp in zip(df["vertical"], df["rel_path"]):
        lmdb_key = _manifest_rel_path_to_lmdb_key(v, rp)
        by_vert.setdefault(v, set()).add(lmdb_key)
    return by_vert


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

    # 2b. Optional: load exclusion set (held-out keys to skip from train/val/test).
    exclusion: dict[str, set[str]] = {}
    if config.exclude_keys_path:
        print(f"\nLoading exclusion set from {config.exclude_keys_path}...")
        exclusion = load_exclusion_set(config.exclude_keys_path)
        total_excl_requested = sum(len(s) for s in exclusion.values())
        print(f"  exclusion set: {total_excl_requested:,} keys across "
              f"{len(exclusion)} verticals")
        # Warn about exclusion-vertical names not present in this run.
        unknown_verts = sorted(
            set(exclusion) - {v.name for v in config.verticals}
        )
        if unknown_verts:
            print(f"  WARNING: exclusion lists {len(unknown_verts)} vertical(s) "
                  f"not in this pipeline: {unknown_verts}")

    # 3. Split assignment
    assignment: SplitAssignment = {}
    total_per_split: dict[str, int] = {s: 0 for s in SPLIT_NAMES}
    excluded_per_vertical: dict[str, int] = {}
    exclusion_misses_per_vertical: dict[str, int] = {}

    for vert in config.verticals:
        fmap = all_formula_maps[vert.name]
        vert_assignment: dict[str, list[str]] = {s: [] for s in SPLIT_NAMES}

        excl = exclusion.get(vert.name, set())
        n_excluded = 0
        for key, formula in fmap.items():
            if key in excl:
                n_excluded += 1
                continue
            split = assign_formula_to_split(
                formula, config.split_config.ratios, config.split_config.seed
            )
            vert_assignment[split].append(key)

        assignment[vert.name] = vert_assignment
        excluded_per_vertical[vert.name] = n_excluded
        # Exclusion keys requested for this vertical but not present in
        # structure.lmdb. Surface so misalignments between
        # manifest_holdout.parquet and the per-vertical LMDBs are visible.
        exclusion_misses_per_vertical[vert.name] = max(0, len(excl) - n_excluded)
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

    n_excluded_total = sum(excluded_per_vertical.values())
    n_exclusion_misses = sum(exclusion_misses_per_vertical.values())

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
        "exclude_keys_path": config.exclude_keys_path,
        "n_excluded_total": n_excluded_total,
        "n_excluded_per_vertical": excluded_per_vertical,
        "n_exclusion_misses_total": n_exclusion_misses,
        "n_exclusion_misses_per_vertical": exclusion_misses_per_vertical,
    }

    # Print summary
    print(f"\nSplit assignment ({config.split_config.method}, seed={config.split_config.seed}):")
    if n_excluded_total:
        print(f"  excluded {n_excluded_total:,} keys via exclude_keys_path")
        if n_exclusion_misses:
            print(f"  WARNING: {n_exclusion_misses:,} exclusion keys not "
                  f"found in any structure.lmdb (per-vertical: "
                  f"{ {k: v for k, v in exclusion_misses_per_vertical.items() if v} })")
    for s in SPLIT_NAMES:
        pct = actual_ratios[s] * 100
        dev = deviations[s] * 100
        warn = " WARNING: >5pp deviation!" if dev > 5.0 else ""
        print(f"  {s}: {total_per_split[s]} ({pct:.1f}%){warn}")
    for vert_name, counts in per_vertical_counts.items():
        excl_note = (f"  excluded={excluded_per_vertical[vert_name]}"
                     if excluded_per_vertical[vert_name] else "")
        print(f"  {vert_name}: {counts}{excl_note}")

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
