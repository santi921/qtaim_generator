"""
Build split-level energy.lmdb files from the OMol25 source aselmdb files.

Joins each aselmdb record to the descriptor corpus via
row.data["source"] == "{rel_path}/orca.tar.zst", where rel_path matches the
manifest rel_path column exactly (verified 2026-07-28). LMDB keys use the same
convention as every other data type: rel_path with os.sep replaced by "__".

Output layout mirrors the released dataset: one energy.lmdb per split and per
holdout suite, NOT per vertical:

    {out_dir}/train/energy.lmdb
    {out_dir}/val/energy.lmdb
    {out_dir}/test/energy.lmdb
    {out_dir}/H1/energy.lmdb   (and H3/H6/H7/H8, when --holdout_dir is given)

Split assignment reproduces the shipped composition split: pymatgen
Composition(formula_hill).formula (spaces stripped), hashed via
assign_formula_to_split (SHA-256 of "{formula}_{seed}", seed 42, ratios
0.8/0.1/0.1). Holdout membership comes from the keys of each suite's
structure.lmdb; holdout structures go to their suite file(s) and are excluded
from train/val/test (a structure in two suites is written to both).

Each record carries the official OMol25 energy (eV) and forces (eV/A), the
source string, unique_id, provenance back to the aselmdb file/row, and the full
row.data payload (charge, spin, SCF metadata, OMol25 mulliken/lowdin/nbo
charges, composition).

Requires ase + ase_db_backends (aselmdb backend), pyarrow (manifests) and
pymatgen (composition formula).

Example:
    python merge_omol_energy.py \
        --aselmdb_dir ~/dev/data/train_4M \
        --manifest_dir data/omol_manifest \
        --holdout_dir data/holdouts \
        --out_dir /path/to/energy_split_lmdbs \
        --mapping_out /path/to/energy_mapping.parquet
"""

import argparse
import glob
import json
import os
import pickle

import lmdb
import pandas as pd
from ase.db import connect
from pymatgen.core import Composition

from qtaim_gen.source.utils.splits import assign_formula_to_split

SOURCE_SUFFIX = "/orca.tar.zst"
SPLIT_RATIOS = (0.8, 0.1, 0.1)
SPLIT_SEED = 42


def load_split_assignments(manifest_dir, verticals=None, seed=SPLIT_SEED):
    """Return {rel_path: split_name} using the shipped composition split."""
    formula_cache = {}
    assignments = {}
    for f in sorted(glob.glob(os.path.join(manifest_dir, "manifest_*.parquet"))):
        vertical = os.path.basename(f)[len("manifest_") : -len(".parquet")]
        if verticals is not None and vertical not in verticals:
            continue
        df = pd.read_parquet(f, columns=["rel_path", "formula_hill"])
        for rel_path, formula_hill in zip(df["rel_path"], df["formula_hill"]):
            split = formula_cache.get(formula_hill)
            if split is None:
                comp = Composition(formula_hill).formula.replace(" ", "")
                split = assign_formula_to_split(comp, SPLIT_RATIOS, seed)
                formula_cache[formula_hill] = split
            assignments[rel_path] = split
    return assignments


def load_holdout_membership(holdout_dir):
    """Return {lmdb_key: [suite, ...]} from each suite's structure.lmdb keys."""
    membership = {}
    suite_counts = {}
    for suite_path in sorted(glob.glob(os.path.join(holdout_dir, "*"))):
        structure = os.path.join(suite_path, "structure.lmdb")
        if not os.path.isfile(structure):
            continue
        suite = os.path.basename(suite_path)
        env = lmdb.open(structure, subdir=False, readonly=True, lock=False)
        n = 0
        with env.begin() as txn:
            for k, _ in txn.cursor():
                key = k.decode("ascii")
                if key == "length":
                    continue
                membership.setdefault(key, []).append(suite)
                n += 1
        env.close()
        suite_counts[suite] = n
    return membership, suite_counts


class SplitWriter:
    """Streaming writer for one energy.lmdb, lazily opened per split/suite."""

    def __init__(self, out_dir, name):
        split_dir = os.path.join(out_dir, name)
        os.makedirs(split_dir, exist_ok=True)
        self.db = lmdb.open(
            os.path.join(split_dir, "energy.lmdb"),
            map_size=int(1099511627776 * 2),
            subdir=False,
            meminit=False,
            map_async=True,
        )
        self.count = 0
        self.duplicates = 0

    def put(self, key, record):
        with self.db.begin(write=True) as txn:
            existed = txn.get(key.encode("ascii")) is not None
            txn.put(key.encode("ascii"), pickle.dumps(record, protocol=-1))
        if existed:
            self.duplicates += 1
        else:
            self.count += 1

    def close(self):
        with self.db.begin(write=True) as txn:
            txn.put("length".encode("ascii"), pickle.dumps(self.count, protocol=-1))
        self.db.sync()
        self.db.close()


def merge(
    aselmdb_dirs,
    manifest_dir,
    out_dir,
    holdout_dir=None,
    verticals=None,
    mapping_out=None,
    limit_files=None,
    seed=SPLIT_SEED,
):
    assignments = load_split_assignments(manifest_dir, verticals, seed)
    print(f"manifests: {len(assignments)} rel_paths assigned to splits")

    holdout_membership, suite_counts = {}, {}
    if holdout_dir is not None:
        holdout_membership, suite_counts = load_holdout_membership(holdout_dir)
        print(
            f"holdouts: {len(holdout_membership)} unique keys across "
            f"{len(suite_counts)} suites {suite_counts}"
        )

    aselmdb_files = []
    for d in aselmdb_dirs:
        aselmdb_files.extend(sorted(glob.glob(os.path.join(d, "*.aselmdb"))))
    if limit_files:
        aselmdb_files = aselmdb_files[:limit_files]
    if not aselmdb_files:
        raise FileNotFoundError(f"no .aselmdb files under {aselmdb_dirs}")

    writers = {}
    mapping_rows = []
    n_extra = 0
    n_bad_source = 0
    n_scanned = 0

    def write_to(name, key, record):
        if name not in writers:
            writers[name] = SplitWriter(out_dir, name)
        writers[name].put(key, record)

    for fpath in aselmdb_files:
        fname = os.path.basename(fpath)
        db = connect(fpath, type="aselmdb", readonly=True)
        for row in db.select():
            n_scanned += 1
            data = dict(row.data)
            source = data.get("source", "")
            if not source.endswith(SOURCE_SUFFIX):
                n_bad_source += 1
                continue
            rel_path = source[: -len(SOURCE_SUFFIX)]
            split = assignments.get(rel_path)
            if split is None:
                n_extra += 1
                continue
            key = rel_path.replace("/", "__")

            record = {
                "energy_ev": row.energy,
                "forces_ev_per_ang": row.forces,
                "unique_id": row.unique_id,
                "aselmdb_file": fname,
                "aselmdb_row_id": row.id,
            }
            record.update(data)

            suites = holdout_membership.get(key)
            if suites:
                # holdout structures live in their suite file(s) only,
                # mirroring their exclusion from the released main splits
                for suite in suites:
                    write_to(suite, key, record)
                dest = ",".join(suites)
            else:
                write_to(split, key, record)
                dest = split
            if mapping_out is not None:
                vertical = rel_path.split("/", 1)[0]
                mapping_rows.append(
                    (key, vertical, dest, fname, row.id, row.unique_id)
                )
        print(f"{fname}: scanned total {n_scanned}", flush=True)

    for w in writers.values():
        w.close()

    # expected counts: split assignment minus holdout members, suites as-is
    expected = {}
    for rel_path, split in assignments.items():
        key = rel_path.replace("/", "__")
        if key in holdout_membership:
            for suite in holdout_membership[key]:
                expected[suite] = expected.get(suite, 0) + 1
        else:
            expected[split] = expected.get(split, 0) + 1

    report = {
        "scanned": n_scanned,
        "extra_not_in_manifest": n_extra,
        "bad_source": n_bad_source,
        "outputs": {},
    }
    for name, exp in sorted(expected.items()):
        w = writers.get(name)
        written = w.count if w else 0
        dups = w.duplicates if w else 0
        report["outputs"][name] = {
            "expected": exp,
            "written": written,
            "missing": exp - written,
            "duplicates": dups,
        }
        status = "OK" if written == exp and dups == 0 else "MISMATCH"
        print(f"{name}: expected {exp}, written {written}, dups {dups} [{status}]")

    if all(v["written"] == 0 for v in report["outputs"].values()):
        raise RuntimeError("zero records written across all outputs - bad inputs?")

    report_path = os.path.join(out_dir, "energy_merge_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"report: {report_path}")

    if mapping_out is not None:
        pd.DataFrame(
            mapping_rows,
            columns=[
                "key",
                "vertical",
                "destination",
                "aselmdb_file",
                "aselmdb_row_id",
                "unique_id",
            ],
        ).to_parquet(mapping_out)
        print(f"mapping: {mapping_out} ({len(mapping_rows)} rows)")

    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--aselmdb_dir",
        action="append",
        required=True,
        help="directory of OMol25 .aselmdb files; repeat flag for multiple dirs",
    )
    parser.add_argument(
        "--manifest_dir",
        required=True,
        help="directory of manifest_{vertical}.parquet files",
    )
    parser.add_argument(
        "--out_dir",
        required=True,
        help="output root; writes {out_dir}/{train,val,test,H*}/energy.lmdb",
    )
    parser.add_argument(
        "--holdout_dir",
        default=None,
        help="directory of holdout suites ({dir}/{suite}/structure.lmdb); "
        "suite members are written per suite and excluded from train/val/test",
    )
    parser.add_argument(
        "--verticals",
        default=None,
        help="comma-separated subset of verticals (default: all manifests)",
    )
    parser.add_argument(
        "--mapping_out",
        default=None,
        help="optional parquet path for the key -> (destination, file, row) mapping",
    )
    parser.add_argument(
        "--limit_files",
        type=int,
        default=None,
        help="only scan the first N aselmdb files (debugging)",
    )
    parser.add_argument(
        "--split_seed",
        type=int,
        default=SPLIT_SEED,
        help="seed for the composition split hash (default 42, the shipped split)",
    )
    args = parser.parse_args()

    verticals = args.verticals.split(",") if args.verticals else None
    merge(
        aselmdb_dirs=args.aselmdb_dir,
        manifest_dir=args.manifest_dir,
        out_dir=args.out_dir,
        holdout_dir=args.holdout_dir,
        verticals=verticals,
        mapping_out=args.mapping_out,
        limit_files=args.limit_files,
        seed=args.split_seed,
    )


if __name__ == "__main__":
    main()
