"""Validate the committed converter configs in scripts/helpers/configs_converter.

Regression: the config harness (run_configs.py, formerly test_configs.py) was
collected by pytest and launched real converter jobs against machine-local data
paths. These tests cover the same configs hermetically: every config must parse
as strict JSON, and each runnable converter config must build graphs from the
committed test LMDBs with only its data paths patched.
"""

import json
import os
import shutil
from pathlib import Path

import pytest

from qtaim_gen.source.core.converter import (
    BaseConverter,
    GeneralConverter,
    QTAIMConverter,
)
from qtaim_gen.source.utils.lmdbs import json_2_lmdbs

CONFIG_DIR = (
    Path(__file__).parent.parent
    / "qtaim_gen"
    / "source"
    / "scripts"
    / "helpers"
    / "configs_converter"
)
LMDB_TESTS = Path(__file__).parent / "test_files" / "lmdb_tests"
MERGED = LMDB_TESTS / "generator_lmdbs_merged"

# location-key -> committed fixture file
FIXTURE_LOCATIONS = {
    "geom_lmdb": "merged_geom.lmdb",
    "charge_lmdb": "merged_charge.lmdb",
    "qtaim_lmdb": "merged_qtaim.lmdb",
    "bond_lmdb": "merged_bond.lmdb",
    "bonds_lmdb": "merged_bond.lmdb",
    "fuzzy_lmdb": "merged_fuzzy.lmdb",
    "fuzzy_full_lmdb": "merged_fuzzy.lmdb",
    "other_lmdb": "merged_other.lmdb",
}

# configs that drive a converter run directly (the generator_to_embed_* and
# multi_vertical/sharded_example configs belong to other entry points)
RUNNABLE_CONFIGS = {
    "base_unsharded": BaseConverter,
    "base_sharded_shard0": BaseConverter,
    "base_sharded_shard1": BaseConverter,
    "qtaim_unsharded": QTAIMConverter,
    "general_fuzzy_bonds": GeneralConverter,
    "general_qtaim_bonds": GeneralConverter,
    "general_ibsi_bonds": GeneralConverter,
    "general_with_global_dipoles": GeneralConverter,
    "general_all_features": GeneralConverter,
}


@pytest.fixture(scope="module")
def orca_lmdb_path(tmp_path_factory):
    """Build orca.lmdb from the committed orca.json fixtures (no merged
    orca fixture is committed)."""
    out_dir = tmp_path_factory.mktemp("orca_lmdb_configs")
    staging = out_dir / "_staging"
    staging.mkdir()
    for name in ("orca5", "orca5_rks", "orca5_uks", "orca6_rks"):
        dst = staging / name
        dst.mkdir()
        shutil.copy(LMDB_TESTS / name / "orca.json", dst / "orca.json")
    json_2_lmdbs(
        root_dir=str(staging) + os.sep,
        out_dir=str(out_dir) + os.sep,
        data_type="orca",
        out_lmdb="orca.lmdb",
        chunk_size=10,
        clean=True,
        merge=True,
        move_files=False,
    )
    shutil.rmtree(staging)
    return str(out_dir / "orca.lmdb")


def _drop_hirsh_fuzzy(config):
    """The committed fixtures carry no hirsh_fuzzy scheme; drop just those
    features so the config still exercises the fuzzy bonding path verbatim
    otherwise."""
    config["fuzzy_filter"] = [
        f for f in config["fuzzy_filter"] if f != "hirsh_fuzzy_density"
    ]
    for level in ("atom", "global"):
        config["keys_data"][level] = [
            k for k in config["keys_data"][level] if "hirsh_fuzzy" not in k
        ]


FIXTURE_ADJUSTMENTS = {
    "general_fuzzy_bonds": _drop_hirsh_fuzzy,
}


def _patched_config(name, tmp_path, orca_lmdb):
    with open(CONFIG_DIR / f"{name}.json") as f:
        config = json.load(f)
    config["lmdb_path"] = str(tmp_path)
    config["n_workers"] = 1
    config["restart"] = False
    for key in list(config["lmdb_locations"]):
        if key == "orca_lmdb":
            config["lmdb_locations"][key] = orca_lmdb
        elif key in FIXTURE_LOCATIONS:
            config["lmdb_locations"][key] = str(MERGED / FIXTURE_LOCATIONS[key])
        else:
            pytest.fail(f"{name}: no fixture mapped for lmdb_locations[{key!r}]")
    if name in FIXTURE_ADJUSTMENTS:
        FIXTURE_ADJUSTMENTS[name](config)
    return config


def test_every_config_is_strict_json():
    """A config that does not parse cannot have been validated by anything."""
    paths = sorted(CONFIG_DIR.glob("*.json"))
    assert paths, f"no configs found under {CONFIG_DIR}"
    for path in paths:
        with open(path) as f:
            json.load(f)


@pytest.mark.parametrize("name", sorted(RUNNABLE_CONFIGS))
def test_config_builds_graphs_from_fixtures(name, tmp_path, orca_lmdb_path):
    cls = RUNNABLE_CONFIGS[name]
    config = _patched_config(name, tmp_path, orca_lmdb_path)
    # config_path must NOT be the committed fixture: converters call
    # overwrite_config() during process() for restart bookkeeping, which
    # would clobber the tracked file with tmp_path-patched values.
    config_path = tmp_path / f"{name}.json"
    with open(config_path, "w") as f:
        json.dump(config, f)
    converter = cls(config, config_path=str(config_path))
    converter.process(return_info=True)

    # sharded runs name their outputs per shard/chunk, so glob rather than
    # assuming lmdb_name verbatim
    out_files = [
        os.path.join(config["lmdb_path"], f)
        for f in os.listdir(config["lmdb_path"])
        if f.endswith(".lmdb")
    ]
    assert out_files, f"{name}: no output LMDB written"

    import lmdb
    import pickle

    n_graphs = 0
    for out_file in out_files:
        env = lmdb.open(out_file, subdir=False, readonly=True, lock=False)
        with env.begin() as txn:
            length = txn.get(b"length")
            if length is not None:
                n_graphs += pickle.loads(length)
        env.close()
    assert n_graphs > 0, f"{name}: output LMDBs hold zero graphs"
