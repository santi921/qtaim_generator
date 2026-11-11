"""migrate-provenance: engine folders' timings.json records become provenance.json, and the
migrated folder reads as the old readers read it (scripts/helpers/migrate_provenance.py)."""

import json

from qtaim_gen.source.scripts.helpers.migrate_provenance import LEGACY_KEYS, main, migrate
from qtaim_gen.source.utils import provenance


def _folder(tmp_path, name, timings, charge=None, fuzzy=None, other=None, orca=None, bond=None):
    folder = tmp_path / "tree" / "cat" / name
    gen = folder / "generator"
    gen.mkdir(parents=True)
    (gen / "timings.json").write_text(json.dumps(timings))
    for fname, d in (("charge.json", charge), ("fuzzy_full.json", fuzzy), ("other.json", other),
                     ("orca.json", orca), ("bond.json", bond)):
        if d is not None:
            (gen / fname).write_text(json.dumps(d))
    return folder


def test_stamped_engine_folder(tmp_path):
    timings = {"qtaim": 3.0, "chelpg": 2.0, "charge_engine": 5.0, "surface_engine": 1.0, "qtaim_engine": 0.0,
               "_engine_versions": {"charge_engine": 2, "surface_engine": 2}, "_converter": "native:fixed",
               "_skipped_routines": {"mbis": {"reason": "ecp_disabled", "by": "charge_engine"}},
               "_timings_patched": {"qtaim": "log"}}
    folder = _folder(tmp_path, "a", timings,
                     charge={"hirshfeld": {}, "chelpg": {}, "loewdin_orca": {}},
                     fuzzy={"elf_fuzzy": {}, "becke_fuzzy_density": {}}, bond={"fuzzy_bond": {"x": 1}},
                     other={"ALIE_Volume": 1.0, "mpp_full": 0.1}, orca={"orca_parser_version": 2})
    row = migrate(str(folder), apply=False)
    assert row["status"] == "would_migrate"
    assert not (folder / "generator" / provenance.PROVENANCE_FILE).exists()

    assert migrate(str(folder), apply=True)["status"] == "migrated"
    prov = provenance.load(str(folder))
    r = prov["routines"]
    assert {k for k, e in r.items() if e["by"] == "charge_engine"} == {"hirshfeld", "elf_fuzzy", "becke_fuzzy_density",
                                                                       "fuzzy_bond"}
    assert {e["engine_version"] for e in r.values() if e["by"] == "charge_engine"} == {2}
    assert r["other_alie"] == {"by": "surface_engine", "engine_version": 2, "legacy": True}
    assert r["other_geometry"]["by"] == "surface_engine"
    assert r["qtaim"]["by"] == "multiwfn" and r["chelpg"]["by"] == "multiwfn"
    assert r["orca"] == {"by": "orca", "parser_version": 2, "legacy": True}
    assert provenance.converter_tag(prov) == "native:fixed"
    assert prov["skipped"] == timings["_skipped_routines"]
    left = json.loads((folder / "generator" / "timings.json").read_text())
    assert not set(LEGACY_KEYS) & set(left) and left["_timings_patched"] == {"qtaim": "log"}
    assert left["charge_engine"] == 5.0
    # idempotent
    assert migrate(str(folder), apply=True)["status"] == "already"


def test_unstamped_v1_engine_folder_and_multiwfn_folder(tmp_path):
    v1 = _folder(tmp_path, "v1", {"charge_engine": 4.0, "elf_fuzzy": 2.0, "qtaim": 1.0},
                 fuzzy={"elf_fuzzy": {}, "becke_fuzzy_density": {}})
    assert migrate(str(v1), apply=True)["status"] == "migrated"
    r = provenance.load(str(v1))["routines"]
    # version 1 left elf_fuzzy to Multiwfn
    assert r["becke_fuzzy_density"]["engine_version"] == 1 and r["elf_fuzzy"]["by"] == "multiwfn"
    assert "converter" not in provenance.load(str(v1))

    mwfn = _folder(tmp_path, "mwfn", {"hirshfeld": 1.0, "qtaim": 1.0})
    assert migrate(str(mwfn), apply=True)["status"] == "nothing"
    assert not (mwfn / "generator" / provenance.PROVENANCE_FILE).exists()


def test_existing_provenance_wins_and_locked_folders_are_left_alone(tmp_path):
    folder = _folder(tmp_path, "b", {"charge_engine": 4.0, "_engine_versions": {"charge_engine": 3}},
                     charge={"hirshfeld": {}, "adch": {}})
    prov = provenance.empty()
    prov["routines"]["hirshfeld"] = {"by": "charge_engine", "engine_version": 4}
    (folder / "generator" / provenance.PROVENANCE_FILE).write_text(json.dumps(prov))
    assert migrate(str(folder), apply=True)["status"] == "migrated"
    r = provenance.load(str(folder))["routines"]
    assert r["hirshfeld"] == {"by": "charge_engine", "engine_version": 4}
    assert r["adch"]["engine_version"] == 3 and r["adch"]["legacy"]

    locked = _folder(tmp_path, "c", {"charge_engine": 4.0})
    (locked / ".processing.lock").write_text("")
    assert migrate(str(locked), apply=True)["status"] == "locked"


def test_cli_walks_a_tree_and_reports(tmp_path, capsys):
    _folder(tmp_path, "a", {"charge_engine": 4.0, "_converter": "multiwfn"}, charge={"hirshfeld": {}})
    _folder(tmp_path, "m", {"hirshfeld": 1.0})
    report = tmp_path / "report.tsv"
    assert main(["--root", str(tmp_path / "tree"), "--report", str(report)]) == 0
    assert "would_migrate: 1" in capsys.readouterr().out
    lines = report.read_text().splitlines()
    assert lines[0].startswith("folder\tstatus") and len(lines) == 2 and "multiwfn" in lines[1]


def test_a_folder_with_root_timings_and_an_empty_generator_is_found(tmp_path):
    from qtaim_gen.source.scripts.helpers.migrate_provenance import job_folders

    folder = tmp_path / "tree" / "cat" / "interrupted"
    (folder / "generator").mkdir(parents=True)
    (folder / "generator" / "orca.json").write_text("{}")
    (folder / "timings.json").write_text(json.dumps({"convert": 1.0, "_converter": "native:fixed"}))
    _folder(tmp_path, "done", {"charge_engine": 4.0})
    assert sorted(job_folders(str(tmp_path / "tree"))) == sorted([str(folder), str(tmp_path / "tree" / "cat" / "done")])
    assert migrate(str(folder), apply=True)["status"] == "migrated"
    assert provenance.converter_tag(provenance.load(str(folder))) == "native:fixed"
    assert "_converter" not in json.loads((folder / "timings.json").read_text())
