"""Tests for the shared LMDB audit module and the json-to-lmdb --validate flag.

Covers:
  - validate_record per data_type / per status (ok, empty, malformed,
    missing_critical, no_bonds)
  - scan_lmdb on synthetic LMDBs (counts, samples, limit, unpickle errors)
  - audit_lmdb_paths cross-LMDB drop vs structure.lmdb
  - write_audit_report file outputs
  - end-to-end --validate flag in json-to-lmdb (non-sharded, sharded, merged)
"""

import json
import os
import pickle
import subprocess
import sys
from pathlib import Path

import lmdb
import pytest

from qtaim_gen.source.utils.lmdb_audit import (
    EXPECTED_METHODS,
    audit_lmdb_paths,
    render_audit_md,
    render_summary_line,
    scan_lmdb,
    validate_record,
    write_audit_report,
)


# ---- helpers --------------------------------------------------------------


def _write_synthetic_lmdb(path, items):
    """Write a single-file LMDB at `path` containing pickled `items`.

    `items` values that are already `bytes` are written verbatim (so tests can
    inject corrupt pickle payloads). All other values are pickled.
    """
    env = lmdb.open(
        str(path), subdir=False, map_size=64 * 1024 * 1024, lock=False, meminit=False
    )
    with env.begin(write=True) as txn:
        for k, v in items.items():
            payload = v if isinstance(v, bytes) else pickle.dumps(v, protocol=-1)
            txn.put(k.encode("ascii"), payload)
        txn.put(b"length", pickle.dumps(len(items), protocol=-1))
    env.sync()
    env.close()


def _ok_charge_record():
    return {
        "hirshfeld": {"charge": {"1_C": 0.1, "2_H": -0.05}, "dipole": {"mag": 0.5}},
        "adch": {"charge": {"1_C": 0.12, "2_H": -0.06}},
    }


def _ok_bond_record():
    return {"fuzzy_bond": {"1_C_to_2_H": 0.95, "1_C_to_3_H": 0.94}}


def _ok_qtaim_record():
    return {
        "0": {"density_all": 0.34, "lap_e_density": -1.2, "energy_density": -0.7},
        "1": {"density_all": 0.21},
    }


def _ok_structure_record():
    return {
        "molecule": object(),  # opaque - validator only checks key presence
        "molecule_graph": object(),
        "ids": "key_0",
        "bonds": [(0, 1), (0, 2)],
        "spin": 1,
        "charge": 0,
    }


def _ok_orca_record():
    return {
        "final_energy_eh": -75.123,
        "scf_converged": True,
        "gradient": {"1_C": [0.0, 0.0, 0.0]},
        "dipole_au": [0.1, 0.0, 0.0],
    }


def _ok_fuzzy_record():
    return {
        "becke_fuzzy_density": {"1_C": 0.5},
        "hirsh_fuzzy_density": {"1_C": 0.5},
    }


# ---- TestValidateRecord ---------------------------------------------------


class TestValidateRecord:
    """Pure unit tests for validate_record()."""

    @pytest.mark.parametrize(
        "data_type,record",
        [
            ("structure", _ok_structure_record()),
            ("charge", _ok_charge_record()),
            ("bond", _ok_bond_record()),
            ("qtaim", _ok_qtaim_record()),
            ("orca", _ok_orca_record()),
            ("fuzzy", _ok_fuzzy_record()),
            ("other", {"mpp_full": 1.0}),
            ("timings", {"qtaim": 1.5}),
        ],
    )
    def test_ok_per_data_type(self, data_type, record):
        status, _present = validate_record(data_type, record)
        assert status == "ok", f"{data_type} expected ok, got {status}"

    @pytest.mark.parametrize(
        "data_type",
        ["structure", "charge", "bond", "qtaim", "orca", "fuzzy", "other", "timings"],
    )
    def test_empty_dict_returns_empty(self, data_type):
        assert validate_record(data_type, {}) == ("empty", [])

    @pytest.mark.parametrize(
        "value", [None, "string", 42, 3.14, [1, 2, 3], (1, 2)]
    )
    def test_non_dict_returns_malformed(self, value):
        assert validate_record("charge", value) == ("malformed", [])

    def test_structure_missing_required_returns_missing_critical(self):
        partial = {"molecule": object(), "bonds": []}  # missing charge, spin
        assert validate_record("structure", partial)[0] == "missing_critical"

    def test_orca_missing_final_energy_returns_missing_critical(self):
        rec = {"scf_converged": True, "dipole_au": [0, 0, 0]}
        assert validate_record("orca", rec)[0] == "missing_critical"

    def test_orca_final_energy_none_returns_missing_critical(self):
        rec = {"final_energy_eh": None, "scf_converged": True}
        assert validate_record("orca", rec)[0] == "missing_critical"

    def test_qtaim_no_atom_features_returns_missing_critical(self):
        rec = {"0": {"cp_num": 5, "element": "C"}}  # no density_all etc.
        assert validate_record("qtaim", rec)[0] == "missing_critical"

    def test_qtaim_skips_bond_keys_finds_atom(self):
        rec = {
            "0_1": {"density_all": 0.1},  # bond pair, ignored
            "5": {"density_all": 0.34},  # numeric atom key, valid
        }
        assert validate_record("qtaim", rec)[0] == "ok"

    def test_charge_method_with_inner_charge_returns_ok(self):
        rec = {"hirshfeld": {"charge": {"1_C": 0.1}}}
        status, present = validate_record("charge", rec)
        assert status == "ok" and "hirshfeld" in present

    def test_bond_legacy_key_fuzzy_returns_ok(self):
        # parse_bond_data accepts both 'fuzzy' (legacy) and 'fuzzy_bond' (current)
        rec = {"fuzzy": {"1_C_to_2_H": 0.95}}
        status, present = validate_record("bond", rec)
        assert status == "ok" and "fuzzy_bond" in present

    def test_bond_orca_only_returns_ok(self):
        # ORCA-derived bond orders are valid bond evidence
        rec = {"mayer_orca": {"1_C_to_2_H": 0.92}}
        status, present = validate_record("bond", rec)
        assert status == "ok" and "mayer_orca" in present

    def test_bond_all_methods_empty_returns_no_bonds(self):
        # All expected method keys present but every dict is {} -> no_bonds branch
        rec = {"fuzzy_bond": {}, "ibsi_bond": {}, "laplacian_bond": {}}
        status, present = validate_record("bond", rec)
        assert status == "no_bonds" and present == []

    def test_bond_unknown_keys_returns_no_bonds(self):
        # No expected method present at all (also returns no_bonds for bond type)
        rec = {"unknown_bond_scheme": {"a": 1}}
        status, _ = validate_record("bond", rec)
        assert status == "no_bonds"

    def test_charge_no_methods_returns_missing_critical(self):
        # For non-bond types, missing all methods is missing_critical
        rec = {"unknown_charge_scheme": {"a": 1}}
        assert validate_record("charge", rec)[0] == "missing_critical"

    def test_present_methods_lists_populated(self):
        rec = {
            "hirshfeld": {"charge": {"1_C": 0.1}},
            "cm5": {"charge": {"1_C": 0.0}},
        }
        status, present = validate_record("charge", rec)
        assert status == "ok"
        assert "hirshfeld" in present
        assert "cm5" in present

    def test_charge_outer_dict_with_empty_inner_returns_missing_critical(self):
        # A charge method with metadata only (e.g. dipole) but empty per-atom
        # 'charge' dict is a real silent failure mode (the method ran but
        # produced no per-atom values). Tightened validator surfaces this.
        rec = {"adch": {"charge": {}, "dipole": {"mag": 0.5}}}
        status, present = validate_record("charge", rec)
        assert status == "missing_critical"
        assert "adch" not in present

    def test_charge_only_dipole_no_charge_key_returns_missing_critical(self):
        # Even more degenerate: outer dict has only metadata keys, no 'charge'
        # or 'spin' inner dict at all.
        rec = {"hirshfeld": {"dipole": {"mag": 0.7}}}
        assert validate_record("charge", rec)[0] == "missing_critical"

    def test_charge_inner_spin_only_returns_ok(self):
        # Open-shell systems may report only spin densities for some methods.
        rec = {"hirshfeld": {"spin": {"1_C": 0.1}, "charge": {}}}
        status, present = validate_record("charge", rec)
        assert status == "ok"
        assert "hirshfeld" in present


# ---- TestScanLmdb ---------------------------------------------------------


class TestScanLmdb:
    """Unit tests for scan_lmdb against synthetic single-file LMDBs."""

    def test_missing_file_returns_exists_false(self, tmp_path):
        result = scan_lmdb(str(tmp_path / "nope.lmdb"), "charge")
        assert result["exists"] is False
        assert result["entries"] == 0

    def test_basic_counts(self, tmp_path):
        path = tmp_path / "charge.lmdb"
        items = {
            "k_ok_1": _ok_charge_record(),
            "k_ok_2": _ok_charge_record(),
            "k_empty": {},
            "k_missing_critical": {"unknown": {"a": 1}},
        }
        _write_synthetic_lmdb(path, items)

        result = scan_lmdb(str(path), "charge")
        assert result["exists"] is True
        assert result["entries"] == 4
        assert result["ok"] == 2
        assert result["empty"] == 1
        assert result["missing_critical"] == 1
        assert result["unpickle_error"] == 0
        assert result["malformed"] == 0
        assert result["all_keys"] == set(items.keys())

    def test_unpickle_error_recorded(self, tmp_path):
        path = tmp_path / "bond.lmdb"
        items = {
            "k_ok": _ok_bond_record(),
            "k_corrupt": b"\x00\x01\x02not_a_pickle",
        }
        _write_synthetic_lmdb(path, items)

        result = scan_lmdb(str(path), "bond")
        assert result["entries"] == 2
        assert result["ok"] == 1
        assert result["unpickle_error"] == 1
        assert any(
            "k_corrupt" in s for s in result["failed_samples"]["unpickle_error"]
        )

    def test_failed_samples_capped_by_sample_failed(self, tmp_path):
        path = tmp_path / "bond.lmdb"
        items = {f"k_{i}": {} for i in range(10)}  # all empty
        _write_synthetic_lmdb(path, items)

        result = scan_lmdb(str(path), "bond", sample_failed=3)
        assert result["empty"] == 10
        assert len(result["failed_samples"]["empty"]) == 3

    def test_limit_stops_iteration(self, tmp_path):
        path = tmp_path / "charge.lmdb"
        items = {f"k_{i:03d}": _ok_charge_record() for i in range(20)}
        _write_synthetic_lmdb(path, items)

        result = scan_lmdb(str(path), "charge", limit=5)
        assert result["entries"] == 5
        assert result["ok"] == 5

    def test_method_counts_only_populated(self, tmp_path):
        path = tmp_path / "charge.lmdb"
        items = {
            "k1": {"hirshfeld": {"charge": {"1_C": 0.1}}, "adch": {"charge": {"1_C": 0.2}}},
            "k2": {"hirshfeld": {"charge": {"1_C": 0.1}}},
            "k3": {"cm5": {"charge": {"1_C": 0.0}}},
        }
        _write_synthetic_lmdb(path, items)

        result = scan_lmdb(str(path), "charge")
        assert result["method_counts"]["hirshfeld"] == 2
        assert result["method_counts"]["adch"] == 1
        assert result["method_counts"]["cm5"] == 1
        assert result["method_counts"]["mbis"] == 0

    def test_length_field_skipped_from_counts(self, tmp_path):
        # length is a metadata key written by write_lmdb; should not be counted
        path = tmp_path / "charge.lmdb"
        _write_synthetic_lmdb(path, {"k1": _ok_charge_record(), "k2": _ok_charge_record()})

        result = scan_lmdb(str(path), "charge")
        assert result["entries"] == 2
        assert result["length_field"] == 2
        assert "length" not in result["all_keys"]


# ---- TestAuditLmdbPaths ---------------------------------------------------


class TestAuditLmdbPaths:
    """Tests for cross-LMDB orchestrator and structure-vs-other drop logic."""

    def _build_two_lmdbs(self, tmp_path, structure_keys, charge_keys):
        struct_path = tmp_path / "structure.lmdb"
        charge_path = tmp_path / "charge.lmdb"
        _write_synthetic_lmdb(
            struct_path, {k: _ok_structure_record() for k in structure_keys}
        )
        _write_synthetic_lmdb(
            charge_path, {k: _ok_charge_record() for k in charge_keys}
        )
        return {"structure": str(struct_path), "charge": str(charge_path)}

    def test_cross_drop_vs_structure(self, tmp_path):
        paths = self._build_two_lmdbs(
            tmp_path,
            structure_keys=["a", "b", "c", "d", "e"],
            charge_keys=["a", "b", "c", "d"],  # 'e' missing
        )
        report = audit_lmdb_paths(paths, sample_failed=10, workers=1)

        assert report["per_type"]["charge"]["entries"] == 4
        assert report["per_type"]["structure"]["entries"] == 5

        drop = report["cross_drop_vs_structure"]["charge"]
        assert drop["structure_count"] == 5
        assert drop["lmdb_count"] == 4
        assert drop["missing_from_lmdb"] == 1
        assert drop["missing_sample"] == ["e"]

    def test_cross_drop_extra_in_lmdb(self, tmp_path):
        paths = self._build_two_lmdbs(
            tmp_path, structure_keys=["a", "b"], charge_keys=["a", "b", "extra1"]
        )
        report = audit_lmdb_paths(paths, sample_failed=10, workers=1)

        drop = report["cross_drop_vs_structure"]["charge"]
        assert drop["missing_from_lmdb"] == 0
        assert drop["extra_in_lmdb"] == 1
        assert drop["extra_sample"] == ["extra1"]

    def test_workers_parallel_matches_serial(self, tmp_path):
        paths = self._build_two_lmdbs(
            tmp_path, structure_keys=["a", "b", "c"], charge_keys=["a", "b"]
        )
        serial = audit_lmdb_paths(paths, workers=1)
        parallel = audit_lmdb_paths(paths, workers=4)

        for dt in ("structure", "charge"):
            for k in ("entries", "ok", "empty", "missing_critical"):
                assert serial["per_type"][dt][k] == parallel["per_type"][dt][k]
        # cross_drop content should match exactly
        assert (
            serial["cross_drop_vs_structure"]["charge"]["missing_from_lmdb"]
            == parallel["cross_drop_vs_structure"]["charge"]["missing_from_lmdb"]
        )

    def test_no_structure_means_no_cross_drop(self, tmp_path):
        # Audit a charge-only set (no structure key) -> cross_drop empty
        path = tmp_path / "charge.lmdb"
        _write_synthetic_lmdb(path, {"a": _ok_charge_record()})
        report = audit_lmdb_paths({"charge": str(path)})
        assert report["cross_drop_vs_structure"] == {}

    def test_missing_structure_lmdb_logs_warning(self, tmp_path, caplog):
        # structure.lmdb in the audit set but file does not exist: should
        # warn loudly so the empty cross_drop doesn't silently mask a broken
        # conversion.
        charge_path = tmp_path / "charge.lmdb"
        _write_synthetic_lmdb(charge_path, {"a": _ok_charge_record()})
        paths = {
            "structure": str(tmp_path / "structure.lmdb"),  # does not exist
            "charge": str(charge_path),
        }
        with caplog.at_level("WARNING", logger="qtaim_gen.source.utils.lmdb_audit"):
            report = audit_lmdb_paths(paths)
        assert report["cross_drop_vs_structure"] == {}
        assert any("structure.lmdb missing" in m for m in caplog.messages)


# ---- TestWriteAuditReport -------------------------------------------------


class TestWriteAuditReport:
    """Verify write_audit_report side-effects."""

    def _trivial_report(self, tmp_path):
        struct = tmp_path / "structure.lmdb"
        charge = tmp_path / "charge.lmdb"
        _write_synthetic_lmdb(struct, {"a": _ok_structure_record(), "b": _ok_structure_record()})
        _write_synthetic_lmdb(charge, {"a": _ok_charge_record()})  # 'b' dropped
        return audit_lmdb_paths({"structure": str(struct), "charge": str(charge)})

    def test_writes_md_and_json(self, tmp_path):
        report = self._trivial_report(tmp_path)
        out_dir = tmp_path / "audit_out"
        written = write_audit_report(report, str(out_dir), ["structure", "charge"], label="validate")

        assert (out_dir / "validate.md").exists()
        assert (out_dir / "validate.json").exists()
        assert "json" in written and "md" in written

        loaded = json.loads((out_dir / "validate.json").read_text())
        assert "per_type" in loaded
        assert loaded["per_type"]["structure"]["entries"] == 2

    def test_writes_failed_keys_dir_when_drop_present(self, tmp_path):
        report = self._trivial_report(tmp_path)
        out_dir = tmp_path / "audit_out"
        write_audit_report(report, str(out_dir), ["structure", "charge"], label="validate")

        failed_dir = out_dir / "validate_failed_keys"
        assert failed_dir.exists()
        # structure had key 'b' that charge is missing
        missing_file = failed_dir / "charge__missing_from_lmdb.txt"
        assert missing_file.exists()
        assert missing_file.read_text().strip() == "b"

    def test_no_failed_keys_dir_when_clean(self, tmp_path):
        # When everything is ok and no drop, failed_keys/ should not be created
        struct = tmp_path / "structure.lmdb"
        charge = tmp_path / "charge.lmdb"
        for k in ("a", "b"):
            pass
        _write_synthetic_lmdb(struct, {"a": _ok_structure_record(), "b": _ok_structure_record()})
        _write_synthetic_lmdb(charge, {"a": _ok_charge_record(), "b": _ok_charge_record()})
        report = audit_lmdb_paths({"structure": str(struct), "charge": str(charge)})

        out_dir = tmp_path / "audit_out_clean"
        written = write_audit_report(report, str(out_dir), ["structure", "charge"], label="validate")

        assert (out_dir / "validate.md").exists()
        assert "failed_keys" not in written
        assert not (out_dir / "validate_failed_keys").exists()


# ---- TestRenderHelpers ----------------------------------------------------


class TestRenderHelpers:
    """Smoke tests for the markdown / one-line summary helpers."""

    def test_render_summary_line_clean(self, tmp_path):
        path = tmp_path / "charge.lmdb"
        _write_synthetic_lmdb(path, {"a": _ok_charge_record(), "b": _ok_charge_record()})
        report = audit_lmdb_paths({"charge": str(path)})
        line = render_summary_line(report, ["charge"])
        assert "charge=2 ok" in line

    def test_render_summary_line_flags_no_bonds(self, tmp_path):
        path = tmp_path / "bond.lmdb"
        _write_synthetic_lmdb(
            path, {"a": {"fuzzy_bond": {}}, "b": _ok_bond_record()}
        )
        report = audit_lmdb_paths({"bond": str(path)})
        line = render_summary_line(report, ["bond"])
        assert "no_bonds" in line

    def test_render_audit_md_has_expected_sections(self, tmp_path):
        path = tmp_path / "charge.lmdb"
        _write_synthetic_lmdb(path, {"a": _ok_charge_record()})
        report = audit_lmdb_paths({"charge": str(path)})
        md = render_audit_md(report, ["charge"], label="validate")
        assert "# LMDB validation: validate" in md
        assert "## Per-record validity" in md
        assert "## Per-method coverage" in md


# ---- TestJsonToLmdbValidateFlag (end-to-end CLI) --------------------------


FIXTURE_ROOT = Path(__file__).parent / "test_files" / "lmdb_tests"


def _run_json_to_lmdb(args, env=None):
    cmd = [
        sys.executable,
        "-m",
        "qtaim_gen.source.scripts.json_to_lmdb",
    ] + args
    res = subprocess.run(cmd, capture_output=True, text=True, env=env or os.environ.copy())
    return res


@pytest.mark.skipif(
    not FIXTURE_ROOT.exists(), reason="lmdb_tests fixture not available"
)
class TestJsonToLmdbValidateFlag:
    """End-to-end: --validate writes the expected report files."""

    def test_no_validate_writes_no_audit_files(self, tmp_path):
        out = tmp_path / "out"
        res = _run_json_to_lmdb([
            "--root_dir", str(FIXTURE_ROOT),
            "--out_dir", str(out),
            "--all",
        ])
        assert res.returncode == 0, res.stderr
        # No validate report produced when flag is absent.
        assert not (out / "validate.md").exists()
        assert not (out / "validate.json").exists()
        # But LMDBs should still be there
        assert (out / "structure.lmdb").exists()

    def test_validate_writes_report_non_sharded(self, tmp_path):
        out = tmp_path / "out"
        res = _run_json_to_lmdb([
            "--root_dir", str(FIXTURE_ROOT),
            "--out_dir", str(out),
            "--all",
            "--validate",
            "--validate_workers", "2",
        ])
        assert res.returncode == 0, res.stderr
        assert (out / "validate.md").exists()
        assert (out / "validate.json").exists()

        report = json.loads((out / "validate.json").read_text())
        assert "per_type" in report
        for dt in ("structure", "charge", "qtaim", "bond", "fuzzy", "other", "orca"):
            assert dt in report["per_type"], f"missing {dt} in report"
            assert report["per_type"][dt]["entries"] == 4

    def test_validate_per_shard_writes_shard_label(self, tmp_path):
        out = tmp_path / "out"
        res0 = _run_json_to_lmdb([
            "--root_dir", str(FIXTURE_ROOT),
            "--out_dir", str(out),
            "--all",
            "--total_shards", "2",
            "--shard_index", "0",
            "--validate",
        ])
        assert res0.returncode == 0, res0.stderr
        assert (out / "shard_0" / "validate_shard_0.md").exists()
        assert (out / "shard_0" / "validate_shard_0.json").exists()

        report = json.loads(
            (out / "shard_0" / "validate_shard_0.json").read_text()
        )
        # Each shard gets ~half the fixtures (4 fixtures / 2 shards = 2)
        assert report["per_type"]["structure"]["entries"] == 2

    def test_validate_with_auto_merge_writes_merged_label(self, tmp_path):
        out = tmp_path / "out"
        # shard 0 first
        r0 = _run_json_to_lmdb([
            "--root_dir", str(FIXTURE_ROOT),
            "--out_dir", str(out),
            "--all",
            "--total_shards", "2",
            "--shard_index", "0",
            "--validate",
        ])
        assert r0.returncode == 0, r0.stderr
        # shard 1 + auto_merge
        r1 = _run_json_to_lmdb([
            "--root_dir", str(FIXTURE_ROOT),
            "--out_dir", str(out),
            "--all",
            "--total_shards", "2",
            "--shard_index", "1",
            "--auto_merge",
            "--validate",
        ])
        assert r1.returncode == 0, r1.stderr

        merged_md = out / "merged" / "validate_merged.md"
        merged_json = out / "merged" / "validate_merged.json"
        assert merged_md.exists()
        assert merged_json.exists()

        merged = json.loads(merged_json.read_text())
        # merged should have all 4 fixtures
        assert merged["per_type"]["structure"]["entries"] == 4
