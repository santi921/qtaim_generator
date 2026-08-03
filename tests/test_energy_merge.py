"""Tests for merge_omol_energy.py: aselmdb -> split-level energy.lmdb."""

import os
import pickle

import lmdb
import numpy as np
import pytest

pytest.importorskip("ase_db_backends")
pd = pytest.importorskip("pandas")
pytest.importorskip("pyarrow")
pytest.importorskip("pymatgen")

from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.db import connect
from pymatgen.core import Composition

from qtaim_gen.source.scripts.helpers.merge_omol_energy import (
    SPLIT_RATIOS,
    SPLIT_SEED,
    merge,
)
from qtaim_gen.source.utils.splits import assign_formula_to_split

RECORDS = [
    # (rel_path, formula) flat vertical, jagged vertical, holdout, extra
    ("vertA/job_1_0_1", "H2 O"),
    ("vertA/job_2_-1_2", "C H4"),
    ("omol/sub/outputs_1/spf_9_0_1/step0", "N H3"),
    ("vertA/job_hold_0_1", "C2 H6"),
    ("not_ours/job_x_0_1", "H2"),
]

HOLDOUT_KEY = "vertA__job_hold_0_1"


def _expected_split(formula_hill):
    comp = Composition(formula_hill).formula.replace(" ", "")
    return assign_formula_to_split(comp, SPLIT_RATIOS, SPLIT_SEED)


def _write_fixture_aselmdb(path):
    db = connect(path, type="aselmdb")
    for rel_path, formula in RECORDS:
        n = int(Composition(formula).num_atoms)
        atoms = Atoms(formula.replace(" ", ""))
        atoms.positions = np.random.default_rng(n).random((n, 3))
        atoms.calc = SinglePointCalculator(
            atoms, energy=-10.0 * n, forces=np.full((n, 3), 0.5)
        )
        db.write(
            atoms,
            data={
                "source": f"{rel_path}/orca.tar.zst",
                "data_id": rel_path.split("/")[0],
                "charge": 0,
                "spin": 1,
                "nbo_charges": np.zeros(n),
            },
        )


def _write_fixture_manifests(manifest_dir):
    os.makedirs(manifest_dir, exist_ok=True)
    vert_a = [(r, f) for r, f in RECORDS if r.startswith("vertA/")]
    pd.DataFrame(
        {"rel_path": [r for r, _ in vert_a], "formula_hill": [f for _, f in vert_a]}
    ).to_parquet(os.path.join(manifest_dir, "manifest_vertA.parquet"))
    pd.DataFrame(
        {
            "rel_path": ["omol/sub/outputs_1/spf_9_0_1/step0"],
            "formula_hill": ["N H3"],
        }
    ).to_parquet(os.path.join(manifest_dir, "manifest_omol.parquet"))


def _write_fixture_holdout(holdout_dir):
    suite_dir = os.path.join(holdout_dir, "H1")
    os.makedirs(suite_dir, exist_ok=True)
    env = lmdb.open(
        os.path.join(suite_dir, "structure.lmdb"), subdir=False, meminit=False
    )
    with env.begin(write=True) as txn:
        txn.put(HOLDOUT_KEY.encode("ascii"), pickle.dumps({}))
        txn.put(b"length", pickle.dumps(1))
    env.close()


def _read_all(lmdb_path):
    env = lmdb.open(lmdb_path, subdir=False, readonly=True, lock=False)
    out = {}
    with env.begin() as txn:
        for k, v in txn.cursor():
            out[k.decode("ascii")] = pickle.loads(v)
    env.close()
    return out


@pytest.fixture
def fixture_dirs(tmp_path):
    aselmdb_dir = tmp_path / "aselmdbs"
    manifest_dir = tmp_path / "manifests"
    holdout_dir = tmp_path / "holdouts"
    out_dir = tmp_path / "out"
    aselmdb_dir.mkdir()
    _write_fixture_aselmdb(str(aselmdb_dir / "data0000.aselmdb"))
    _write_fixture_manifests(str(manifest_dir))
    _write_fixture_holdout(str(holdout_dir))
    return str(aselmdb_dir), str(manifest_dir), str(holdout_dir), str(out_dir)


def test_merge_writes_split_energy_lmdbs(fixture_dirs):
    aselmdb_dir, manifest_dir, holdout_dir, out_dir = fixture_dirs
    mapping_out = os.path.join(out_dir, "mapping.parquet")
    os.makedirs(out_dir, exist_ok=True)

    report = merge(
        aselmdb_dirs=[aselmdb_dir],
        manifest_dir=manifest_dir,
        out_dir=out_dir,
        holdout_dir=holdout_dir,
        mapping_out=mapping_out,
    )

    assert report["scanned"] == 5
    assert report["extra_not_in_manifest"] == 1

    # every non-holdout record lands in the split its formula hashes to
    splits = {
        rel: _expected_split(f)
        for rel, f in RECORDS
        if not rel.startswith("not_ours") and "hold" not in rel
    }
    for rel_path, split in splits.items():
        key = rel_path.replace("/", "__")
        data = _read_all(os.path.join(out_dir, split, "energy.lmdb"))
        assert key in data, f"{key} not in {split}"

    # the holdout record is in H1 and in no main split
    h1 = _read_all(os.path.join(out_dir, "H1", "energy.lmdb"))
    assert HOLDOUT_KEY in h1
    assert h1["length"] == 1
    for split in ("train", "val", "test"):
        path = os.path.join(out_dir, split, "energy.lmdb")
        if os.path.exists(path):
            assert HOLDOUT_KEY not in _read_all(path)

    # record payload
    rec = h1[HOLDOUT_KEY]
    assert rec["energy_ev"] == pytest.approx(-80.0)
    assert rec["forces_ev_per_ang"].shape == (8, 3)
    assert rec["source"] == "vertA/job_hold_0_1/orca.tar.zst"
    assert rec["charge"] == 0 and rec["spin"] == 1
    assert rec["unique_id"]
    assert rec["aselmdb_file"] == "data0000.aselmdb"

    # report bookkeeping: expected == written everywhere, no dups
    for name, stats in report["outputs"].items():
        assert stats["written"] == stats["expected"], name
        assert stats["duplicates"] == 0, name
    assert report["outputs"]["H1"]["written"] == 1

    mapping = pd.read_parquet(mapping_out)
    assert len(mapping) == 4
    assert mapping.set_index("key").loc[HOLDOUT_KEY, "destination"] == "H1"

    assert os.path.exists(os.path.join(out_dir, "energy_merge_report.json"))


def test_merge_vertical_filter(fixture_dirs):
    aselmdb_dir, manifest_dir, holdout_dir, out_dir = fixture_dirs
    report = merge(
        aselmdb_dirs=[aselmdb_dir],
        manifest_dir=manifest_dir,
        out_dir=out_dir,
        holdout_dir=holdout_dir,
        verticals=["vertA"],
    )
    written = sum(v["written"] for v in report["outputs"].values())
    assert written == 3  # two split records + one holdout, no omol
    omol_key = "omol__sub__outputs_1__spf_9_0_1__step0"
    for name in report["outputs"]:
        path = os.path.join(out_dir, name, "energy.lmdb")
        if os.path.exists(path):
            assert omol_key not in _read_all(path)


def test_merge_rerun_replaces_cleanly(fixture_dirs):
    """Regression: rerunning into the same out_dir must rebuild from empty.
    The old in-place writer counted already-present keys as duplicates and
    stamped `length` with only the new-key count (0 on a full rerun), silently
    truncating the split for any consumer that trusts `length`."""
    aselmdb_dir, manifest_dir, holdout_dir, out_dir = fixture_dirs
    kwargs = dict(
        aselmdb_dirs=[aselmdb_dir],
        manifest_dir=manifest_dir,
        out_dir=out_dir,
        holdout_dir=holdout_dir,
    )
    merge(**kwargs)
    report = merge(**kwargs)
    for name, stats in report["outputs"].items():
        assert stats["written"] == stats["expected"], name
        assert stats["duplicates"] == 0, name
    h1 = _read_all(os.path.join(out_dir, "H1", "energy.lmdb"))
    assert h1["length"] == 1
    assert not os.path.exists(os.path.join(out_dir, "H1", "energy.lmdb.tmp"))


def test_merge_discards_stale_partial_tmp(fixture_dirs):
    """A tmp file left by a crashed run must not leak keys into the rebuild."""
    aselmdb_dir, manifest_dir, holdout_dir, out_dir = fixture_dirs
    os.makedirs(os.path.join(out_dir, "H1"), exist_ok=True)
    stale = os.path.join(out_dir, "H1", "energy.lmdb.tmp")
    env = lmdb.open(stale, subdir=False)
    with env.begin(write=True) as txn:
        txn.put(b"stale_key", pickle.dumps({}))
    env.close()
    merge(
        aselmdb_dirs=[aselmdb_dir],
        manifest_dir=manifest_dir,
        out_dir=out_dir,
        holdout_dir=holdout_dir,
    )
    h1 = _read_all(os.path.join(out_dir, "H1", "energy.lmdb"))
    assert "stale_key" not in h1
    assert h1["length"] == 1


def test_merge_raises_on_zero_written(fixture_dirs, tmp_path):
    aselmdb_dir, _, _, out_dir = fixture_dirs
    empty_manifests = tmp_path / "empty_manifests"
    empty_manifests.mkdir()
    pd.DataFrame(
        {"rel_path": ["vertB/nothing_0_1"], "formula_hill": ["H2"]}
    ).to_parquet(str(empty_manifests / "manifest_vertB.parquet"))
    with pytest.raises(RuntimeError, match="zero records"):
        merge(
            aselmdb_dirs=[aselmdb_dir],
            manifest_dir=str(empty_manifests),
            out_dir=out_dir,
        )
