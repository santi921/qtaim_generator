"""Tests for qtaim_gen.source.analysis.census (Stream C, C4a-C4e)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from qtaim_gen.source.analysis.census import CENSUS_FIELDS, census


# C4a -------------------------------------------------------------------

def test_census_orchestrator(tiny_vertical):
    root, _ = tiny_vertical(real_shape=True)
    result = census(root)

    assert set(result) == set(CENSUS_FIELDS)
    assert result["vertical"] == root.name
    # Fixture seeds: 3 structures (H2O, H2O, CH4)
    assert result["n_structures"] == 3
    assert result["n_unique_formulas"] == 2
    assert result["n_atom_records"] == 3 + 3 + 5
    # 3 charge schemes per record
    assert result["n_charge_scheme_records"] == 3 * 3
    # BCP counts: 2, 1, 0
    assert result["n_bcps"] == 3
    # 3 bond entries per record (mayer 2 + loewdin 1) across 3 records
    assert result["n_bonds_total"] == 9
    assert result["n_bonds_mayer"] == 6
    assert result["n_bonds_loewdin"] == 3
    assert result["n_bonds_fuzzy"] == 0
    assert result["n_bonds_ibsi"] == 0


# C4b -------------------------------------------------------------------

def test_census_cli_corpus_root(tiny_vertical, tmp_path):
    # Two level-0 verticals plus a NON_VERTICAL_DIRS-style subdir and a
    # level-2-only vertical (merged/ subdir). The latter two must be skipped.
    parent = tmp_path / "corpus"
    tiny_vertical(real_shape=True, dest=parent / "alpha")               # level-0, kept
    tiny_vertical(real_shape=True, dest=parent / "beta")                # level-0, kept
    tiny_vertical(real_shape=True, dest=parent / "geom_orca6" / "merged")  # level-2, skipped
    tiny_vertical(real_shape=True, dest=parent / "holdout_lmdbs")       # blocklisted

    out = tmp_path / "census.parquet"
    rc = subprocess.call(
        [sys.executable, "-m", "qtaim_gen.source.scripts.analysis_census",
         "--root", str(parent), "--output", str(out)]
    )
    assert rc == 0
    assert out.exists()

    df = pd.read_parquet(out)
    assert list(df.columns) == list(CENSUS_FIELDS)
    # geom_orca6 (level-2-only) and holdout_lmdbs (blocklisted) skipped.
    assert list(df["vertical"]) == ["alpha", "beta"]
    for col in ["n_structures", "n_unique_formulas", "n_atom_records",
                "n_charge_scheme_records", "n_bcps", "n_bonds_total",
                "n_bonds_mayer", "n_bonds_loewdin", "n_bonds_fuzzy", "n_bonds_ibsi"]:
        assert df[col].iloc[0] == df[col].iloc[1]
    assert df["n_structures"].iloc[0] == 3


# C4c -------------------------------------------------------------------

def test_census_empty_lmdb(tiny_vertical):
    root, _ = tiny_vertical(keys=[], real_shape=True)
    result = census(root)
    assert result["vertical"] == root.name
    assert result["n_structures"] == 0
    assert result["n_unique_formulas"] == 0
    assert result["n_atom_records"] == 0
    assert result["n_charge_scheme_records"] == 0
    assert result["n_bcps"] == 0
    assert result["n_bonds_total"] == 0


# C4d -------------------------------------------------------------------

def test_census_missing_record_key(tiny_vertical):
    # Inject a structure record that lacks the "molecule" key. census
    # treats it as a structure that exists (counted in n_structures) but
    # contributes no atoms/formula. charge/qtaim/bond have no entry for
    # this key, so those streams drop it via inner-join semantics.
    root, _ = tiny_vertical(
        real_shape=True,
        bad_records={"structure": {"k_no_molecule": {"spin": 0, "charge": 0}}},
    )
    result = census(root)
    assert result["n_structures"] == 4  # 3 valid + 1 malformed
    assert result["n_unique_formulas"] == 2  # malformed has no formula
    assert result["n_atom_records"] == 11   # malformed contributes 0
    # Other LMDBs lack the bad key entirely, so their counts are unchanged
    assert result["n_charge_scheme_records"] == 9
    assert result["n_bcps"] == 3
    assert result["n_bonds_total"] == 9


# C4f -------------------------------------------------------------------

def test_census_cli_include_verticals_and_merge(tiny_vertical, tmp_path):
    parent = tmp_path / "corpus"
    tiny_vertical(real_shape=True, dest=parent / "alpha")
    tiny_vertical(real_shape=True, dest=parent / "beta")
    tiny_vertical(real_shape=True, dest=parent / "gamma")

    shards_dir = tmp_path / "shards"
    shards_dir.mkdir()

    # One shard per vertical (parallel-ready). bogus_name is warned + skipped.
    for v in ["alpha", "beta", "gamma"]:
        rc = subprocess.call(
            [sys.executable, "-m", "qtaim_gen.source.scripts.analysis_census",
             "--root", str(parent),
             "--include_verticals", v, "bogus_name",
             "--output", str(shards_dir / f"{v}.parquet")]
        )
        assert rc == 0
        df = pd.read_parquet(shards_dir / f"{v}.parquet")
        assert list(df["vertical"]) == [v]

    # Merge step
    merged = tmp_path / "merged.parquet"
    rc = subprocess.call(
        [sys.executable, "-m", "qtaim_gen.source.scripts.analysis_census",
         "--merge_from", *[str(shards_dir / f"{v}.parquet") for v in ["gamma", "alpha", "beta"]],
         "--output", str(merged)]
    )
    assert rc == 0
    df = pd.read_parquet(merged)
    assert list(df["vertical"]) == ["alpha", "beta", "gamma"]
    assert list(df.columns) == list(CENSUS_FIELDS)


# C4e -------------------------------------------------------------------

@pytest.mark.slow
def test_census_smoke_noble_gas():
    root = Path("data/OMol4M_lmdbs/noble_gas")
    if not (root / "structure.lmdb").exists():
        pytest.skip(f"{root}/structure.lmdb not present; smoke test skipped")
    result = census(root)
    assert result["vertical"] == "noble_gas"
    assert result["n_structures"] > 0
    assert result["n_atom_records"] > 0
    assert result["n_bcps"] > 0
    # Probed 2026-05-05: every noble_gas record carries at least 7 charge
    # schemes; some records carry up to 10 (mixed RKS/UKS jobs).
    assert result["n_charge_scheme_records"] >= result["n_structures"] * 7
    # Universal bond schemes; ibsi only on ~4% of noble_gas records.
    assert result["n_bonds_mayer"] > 0
    assert result["n_bonds_loewdin"] > 0
    assert result["n_bonds_fuzzy"] > 0
    assert result["n_bonds_total"] == (
        result["n_bonds_mayer"] + result["n_bonds_loewdin"]
        + result["n_bonds_fuzzy"] + result["n_bonds_ibsi"]
    )
