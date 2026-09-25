"""Regression tests for scripts/merge_l1_into_l0.py root-file handling.

The merge tool copies root-level files from a source job folder into the
destination. Replace actions overwrite the destination's geometry input with
the source's, because the source record validated against it. A failed merge
must then put the original input back: neither the copied-file sweep nor the
generator/ backup covers the job root, so without an explicit save the
destination was left with no input at all.
"""
import importlib.util
import json
import os
import shutil
import sys
import zipfile

import pytest

from qtaim_gen.source.utils.validation import get_expected_timing_keys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIX = os.path.join(REPO, "tests", "test_files", "lmdb_tests", "orca6_rks")

_spec = importlib.util.spec_from_file_location(
    "merge_l1_into_l0", os.path.join(REPO, "scripts", "merge_l1_into_l0.py")
)
m = importlib.util.module_from_spec(_spec)
sys.modules["merge_l1_into_l0"] = m
_spec.loader.exec_module(m)

L0_CHARGE = ["adch", "becke", "hirshfeld", "cm5"]
L1_CHARGE = ["vdd", "mbis", "chelpg"]
TAG = "# vast-original\n"


def _load(name):
    with open(os.path.join(FIX, name)) as f:
        return json.load(f)


def make_job(job, level, drop=(), qtaim_out=True):
    """A job folder shaped like the pipeline leaves it, from the orca6_rks fixture."""
    gen = os.path.join(job, "generator")
    os.makedirs(gen)
    shutil.copy2(os.path.join(FIX, "orca.inp"), os.path.join(job, "orca.inp"))
    charge, bond, fuzzy = _load("charge.json"), _load("bond.json"), _load("fuzzy_full.json")
    other, qtaim = _load("other.json"), _load("qtaim.json")
    for k in ("ALIE_Volume", "ALIE_Overall_skewness", "mpp_full"):
        other.setdefault(k, 0.1)
    keys, _ = get_expected_timing_keys(full_set=level, spin_tf=False)
    fz_entry = fuzzy["becke_fuzzy_density"]
    files = {
        "timings.json": {k: 1.5 for k in keys},
        "charge.json": {k: charge[k] for k in L0_CHARGE + (L1_CHARGE if level else [])},
        "bond.json": {"fuzzy_bond": bond["fuzzy_bond"], **({"ibsi_bond": bond["ibsi_bond"]} if level else {})},
        "fuzzy_full.json": {
            "becke_fuzzy_density": fz_entry,
            "hirsh_fuzzy_density": fz_entry,
            **({"elf_fuzzy": fuzzy["elf_fuzzy"], "mbis_fuzzy_density": fuzzy["mbis_fuzzy_density"]} if level else {}),
        },
        "other.json": other,
        "qtaim.json": qtaim,
        "orca.json": _load("orca.json"),
    }
    for name, data in files.items():
        if name in drop:
            continue
        with open(os.path.join(gen, name), "w") as f:
            json.dump(data, f)
    n_bcp = sum(1 for k in qtaim if "_" in k)
    with zipfile.ZipFile(os.path.join(gen, "out_files.zip"), "w") as z:
        for o in ("hirshfeld.out", "adch.out", "cm5.out", "becke.out", "fuzzy_bond.out"):
            z.writestr(o, "Main function menu\nrun\nMain function menu\n")
        if qtaim_out:
            z.writestr("qtaim.out", f"Number of (3,-1) CPs: {n_bcp}\nhave been outputted to CPprop.txt\n")
    with open(os.path.join(job, "gbw_analysis.log"), "w") as f:
        f.write("log\n")


@pytest.fixture
def pair(tmp_path):
    """Destination with an invalid level-0 record and a tagged input; source
    with a valid level-1 record. The plan must choose REPLACE."""
    dst = str(tmp_path / "vast" / "job")
    src = str(tmp_path / "lustre" / "job")
    make_job(dst, 0, drop=("bond.json",))
    make_job(src, 1)
    with open(os.path.join(dst, "orca.inp"), "a") as f:
        f.write(TAG)
    return dst, src


def _plan(dst, src):
    m._plan_init(m.STRICT_DEFAULT, 0, 1e-3, 6.0, 0.5)
    return m.plan_one(("job", dst, src, []))


def _apply(row, force=False):
    m._apply_init(m.STRICT_DEFAULT, False, force, 6.0)
    return m.apply_one(row)


def _saved_leftovers(job):
    return [n for n in os.listdir(job) if n.endswith(m.SAVED_SUFFIX)]


def test_plan_chooses_replace(pair):
    dst, src = pair
    assert _plan(dst, src)["action"] == "REPLACE"


def test_failed_replace_restores_overwritten_input(pair):
    dst, src = pair
    row = _plan(dst, src)
    # strict check_orca now fails on the merged folder, forcing a rollback
    os.remove(os.path.join(src, "generator", "orca.json"))
    out = _apply(row, force=True)
    assert out["result"] == "FAILED"
    with open(os.path.join(dst, "orca.inp")) as f:
        assert f.read().endswith(TAG), "original destination input was not restored"
    assert _saved_leftovers(dst) == []


def test_successful_replace_installs_source_input(pair):
    dst, src = pair
    out = _apply(_plan(dst, src))
    assert out["result"] == "OK", out["detail"]
    with open(os.path.join(dst, "orca.inp")) as a, open(os.path.join(src, "orca.inp")) as b:
        assert a.read() == b.read()
    assert _saved_leftovers(dst) == []


def test_decoy_input_is_never_copied(pair):
    dst, src = pair
    with open(os.path.join(src, "orca.property.inp"), "w") as f:
        f.write("$ SCF_Energy\n")
    out = _apply(_plan(dst, src))
    assert out["result"] == "OK", out["detail"]
    assert not os.path.exists(os.path.join(dst, "orca.property.inp"))


def test_stale_root_qtaim_out_does_not_shadow_replaced_record(pair):
    dst, src = pair
    # vast's broken run left an incomplete qtaim.out at the job root; it is read
    # before the zip, so without the sync it outranks the installed record
    with open(os.path.join(dst, "qtaim.out"), "w") as f:
        f.write("progress 12%\n")
    row = _plan(dst, src)
    assert row["action"] == "REPLACE"
    out = _apply(row)
    assert out["result"] == "OK", out["detail"]
    assert not os.path.exists(os.path.join(dst, "qtaim.out"))
    assert _saved_leftovers(dst) == []


def test_failed_replace_restores_generator_without_backup(pair):
    dst, src = pair
    before = sorted(os.listdir(os.path.join(dst, "generator")))
    row = _plan(dst, src)
    os.remove(os.path.join(src, "generator", "orca.json"))
    out = _apply(row, force=True)
    assert out["result"] == "FAILED"
    assert sorted(os.listdir(os.path.join(dst, "generator"))) == before
    assert not os.path.exists(os.path.join(dst, "generator" + m.SAVED_SUFFIX))


@pytest.fixture
def patch_pair(tmp_path):
    """Destination: level 1 whose zip is unreadable, so provenance fails.
    Source: valid level 0. The plan must choose PATCH_QTAIM."""
    dst = str(tmp_path / "vast" / "job")
    src = str(tmp_path / "lustre" / "job")
    make_job(dst, 1)
    make_job(src, 0)
    with open(os.path.join(dst, "generator", "out_files.zip"), "wb") as f:
        f.write(b"not a zip")
    return dst, src


def test_patch_qtaim_survives_corrupt_destination_zip(patch_pair):
    dst, src = patch_pair
    row = _plan(dst, src)
    assert row["action"] == "PATCH_QTAIM"
    out = _apply(row)
    assert out["result"] == "OK", out["detail"]
    gen = os.path.join(dst, "generator")
    with zipfile.ZipFile(os.path.join(gen, "out_files.zip")) as z:
        assert "qtaim.out" in z.namelist()
    assert os.path.exists(os.path.join(gen, "out_files.zip.corrupt"))
    assert not [n for n in os.listdir(gen) if n.endswith(m.SAVED_SUFFIX)]


def test_failed_patch_restores_qtaim_json(tmp_path):
    dst = str(tmp_path / "vast" / "job")
    src = str(tmp_path / "lustre" / "job")
    make_job(dst, 1, qtaim_out=False)
    make_job(src, 0)
    with open(os.path.join(dst, "generator", "qtaim.json"), "a") as f:
        f.write(" ")  # distinguishable bytes, still valid JSON
    row = _plan(dst, src)
    assert row["action"] == "PATCH_QTAIM"
    with open(os.path.join(dst, "generator", "qtaim.json")) as f:
        original = f.read()
    os.remove(os.path.join(dst, "generator", "orca.json"))  # post-merge check_orca now fails
    out = _apply(row, force=True)
    assert out["result"] == "FAILED"
    with open(os.path.join(dst, "generator", "qtaim.json")) as f:
        assert f.read() == original
