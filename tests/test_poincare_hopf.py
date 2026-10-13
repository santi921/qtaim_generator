"""--enforce_poincare_hopf: QTAIM records whose n - b + r - c != 1 rerun once with the
exhaustive search; the exhaustive result is accepted whatever its sum (a rerun
reproduces it), and a record carrying no topology information always passes.
The sum is stored in qtaim.json under _meta, which no CP-key reader may count."""

import json
import logging
import shutil
from pathlib import Path

from qtaim_gen.source.core.omol import (
    _poincare_hopf_needs_escalation,
    _qtaim_output_complete,
    parse_multiwfn,
)
from qtaim_gen.source.utils.lmdbs import drop_qtaim_fields, parse_qtaim_data
from qtaim_gen.source.utils.validation import (
    QTAIM_SPHERE_SEARCH_MARKER,
    cpprop_cp_counts,
    poincare_hopf_needs_exhaustive,
    qtaim_poincare_hopf,
    qtaim_topology_meta,
    validate_qtaim_dict,
)

TEST_FILES = Path(__file__).parent / "test_files"
CPPROP = TEST_FILES / "CPprop_w_bond_paths.txt"  # 13 NCP, 13 BCP, 1 RCP, 0 CCP
INP = TEST_FILES / "input_bond_paths.in"
COUNT_LINE = " Number of (3,-1) CPs:    13    Generating topology paths...\n"
EXPORT_LINE = " Done! The results have been outputted to CPprop.txt in current folder\n"
LOG = logging.getLogger("t")


def _record(ph, search, n_atoms=13, n_bcp=13):
    rec = {str(i): {"density_all": 1.0} for i in range(n_atoms)}
    rec.update({f"{i}_{i + 1}": {"density_all": 0.2} for i in range(n_bcp)})
    rec["_meta"] = {"poincare_hopf": ph, "cp_counts": {}, "qtaim_search": search}
    return rec


def _job(tmp_path, record):
    gen = tmp_path / "generator"
    gen.mkdir()
    (gen / "qtaim.json").write_text(json.dumps(record))
    (tmp_path / "qtaim.out").write_text(COUNT_LINE + EXPORT_LINE)
    return tmp_path, gen / "qtaim.json"


class TestTopology:
    def test_counts_and_sum_from_cpprop(self):
        data = CPPROP.read_bytes()
        assert cpprop_cp_counts(data) == {"NCP": 13, "BCP": 13, "RCP": 1, "CCP": 0}
        assert qtaim_topology_meta(data, COUNT_LINE)["poincare_hopf"] == 1

    def test_search_from_qtaim_out(self):
        data = CPPROP.read_bytes()
        assert qtaim_topology_meta(data, COUNT_LINE)["qtaim_search"] == "standard"
        assert qtaim_topology_meta(data, QTAIM_SPHERE_SEARCH_MARKER)["qtaim_search"] == "exhaustive"
        assert qtaim_topology_meta(data, None)["qtaim_search"] is None

    def test_escalation_rule(self):
        assert not poincare_hopf_needs_exhaustive(None)
        assert not poincare_hopf_needs_exhaustive({"poincare_hopf": 1, "qtaim_search": "standard"})
        assert poincare_hopf_needs_exhaustive({"poincare_hopf": 0, "qtaim_search": "standard"})
        assert poincare_hopf_needs_exhaustive({"poincare_hopf": 2, "qtaim_search": None})
        assert not poincare_hopf_needs_exhaustive({"poincare_hopf": 0, "qtaim_search": "exhaustive"})

    def test_legacy_record_falls_back_to_the_archived_cpprop(self, tmp_path):
        shutil.copy(CPPROP, tmp_path / "CPprop.txt")
        (tmp_path / "qtaim.out").write_text(COUNT_LINE + EXPORT_LINE)
        topo = qtaim_poincare_hopf({"0": {"density_all": 1.0}}, str(tmp_path))
        assert topo["poincare_hopf"] == 1 and topo["qtaim_search"] == "standard"

    def test_no_information_is_none(self, tmp_path):
        assert qtaim_poincare_hopf({"0": {"density_all": 1.0}}, str(tmp_path)) is None


class TestParseWritesMeta:
    def test_meta_is_stored_last(self, tmp_path):
        shutil.copy(CPPROP, tmp_path / "CPprop.txt")
        shutil.copy(INP, tmp_path / "input.in")
        (tmp_path / "qtaim.out").write_text(COUNT_LINE + EXPORT_LINE)
        parse_multiwfn(str(tmp_path), separate=False, logger=LOG)
        rec = json.loads((tmp_path / "qtaim.json").read_text())
        assert list(rec)[-1] == "_meta"
        assert rec["_meta"] == {"poincare_hopf": 1, "cp_counts": {"NCP": 13, "BCP": 13, "RCP": 1, "CCP": 0},
                                "qtaim_search": "standard"}


class TestValidatorAndGate:
    def test_meta_is_not_counted_as_a_bond(self, tmp_path):
        # 12 stored BCPs + _meta against 13 reported: a deficit of 1 either way, but
        # with tolerance 0 the miscount would hide it
        folder, p = _job(tmp_path, _record(1, "standard", n_bcp=12))
        assert not validate_qtaim_dict(str(p), n_atoms=13, folder=str(folder), check_bcp_count=True,
                                       bcp_tolerance=0)

    def test_violation_from_standard_search_reruns(self, tmp_path):
        folder, p = _job(tmp_path, _record(0, "standard"))
        assert validate_qtaim_dict(str(p), n_atoms=13, folder=str(folder))
        assert not validate_qtaim_dict(str(p), n_atoms=13, folder=str(folder), enforce_poincare_hopf=True)
        assert _qtaim_output_complete(str(folder), n_atoms=13)
        assert not _qtaim_output_complete(str(folder), n_atoms=13, enforce_poincare_hopf=True)
        assert _poincare_hopf_needs_escalation(str(folder), LOG)

    def test_violation_after_exhaustive_search_is_accepted(self, tmp_path):
        folder, p = _job(tmp_path, _record(0, "exhaustive"))
        assert validate_qtaim_dict(str(p), n_atoms=13, folder=str(folder), enforce_poincare_hopf=True)
        assert _qtaim_output_complete(str(folder), n_atoms=13, enforce_poincare_hopf=True)
        # accepted, and any later rerun keeps the exhaustive search
        assert _poincare_hopf_needs_escalation(str(folder), LOG)

    def test_satisfied_record_passes(self, tmp_path):
        folder, p = _job(tmp_path, _record(1, "standard"))
        assert validate_qtaim_dict(str(p), n_atoms=13, folder=str(folder), enforce_poincare_hopf=True)
        assert _qtaim_output_complete(str(folder), n_atoms=13, enforce_poincare_hopf=True)

    def test_record_without_meta_or_cpprop_passes(self, tmp_path):
        rec = _record(1, "standard")
        del rec["_meta"]
        folder, p = _job(tmp_path, rec)
        assert validate_qtaim_dict(str(p), n_atoms=13, folder=str(folder), enforce_poincare_hopf=True)
        assert _qtaim_output_complete(str(folder), n_atoms=13, enforce_poincare_hopf=True)


class TestLmdbBoundary:
    def test_meta_never_reaches_qtaim_lmdb(self):
        assert "_meta" not in drop_qtaim_fields(_record(1, "standard"))

    def test_bond_features_ignore_a_leading_meta(self):
        rec = {"_meta": {"poincare_hopf": 1}, "0": {"density_all": 1.0}, "1": {"density_all": 1.0},
               "0_1": {"density_all": 0.2, "connected_bond_paths": [1, 2]}}
        _, bond_keys, _, bond_feats, _ = parse_qtaim_data(rec, {}, {})
        assert bond_keys == ["density_all"]
        assert bond_feats[(0, 1)] == {"density_all": 0.2}


class TestEscalationPlumbing:
    """The escalation only works if it reaches create_jobs(exhaustive_qtaim=True)."""

    def _gbw(self, monkeypatch, tmp_path, record, **kwargs):
        from qtaim_gen.source.core import omol
        seen = {}
        monkeypatch.setattr(omol, "create_jobs", lambda *a, **k: seen.update(create=k))
        monkeypatch.setattr(omol, "run_jobs", lambda *a, **k: seen.update(run=k))
        monkeypatch.setattr(omol, "parse_multiwfn", lambda *a, **k: None)
        monkeypatch.setattr(omol, "validation_checks", lambda *a, **k: seen.update(validate=k) or False)
        folder, _ = _job(tmp_path, record)
        (folder / "orca.wfn").write_text("wfn")
        omol.gbw_analysis(str(folder), multiwfn_cmd="x", orca_2mkl_cmd="y", restart=False, overwrite=False,
                          logger=LOG, move_results=True, **kwargs)
        return seen

    def test_violation_writes_the_exhaustive_script(self, monkeypatch, tmp_path):
        seen = self._gbw(monkeypatch, tmp_path, _record(0, "standard"), enforce_poincare_hopf=True)
        assert seen["create"]["exhaustive_qtaim"] is True
        assert seen["run"]["restart"] is True and seen["run"]["enforce_poincare_hopf"] is True

    def test_satisfied_record_keeps_the_standard_script(self, monkeypatch, tmp_path):
        seen = self._gbw(monkeypatch, tmp_path, _record(1, "standard"), enforce_poincare_hopf=True)
        assert seen["create"]["exhaustive_qtaim"] is False

    def test_exhaustive_record_stays_exhaustive(self, monkeypatch, tmp_path):
        # a rerun for another reason must not fall back to standard and escalate again
        seen = self._gbw(monkeypatch, tmp_path, _record(1, "exhaustive"), enforce_poincare_hopf=True)
        assert seen["create"]["exhaustive_qtaim"] is True

    def test_flag_off_never_escalates(self, monkeypatch, tmp_path):
        seen = self._gbw(monkeypatch, tmp_path, _record(0, "standard"))
        assert seen["create"]["exhaustive_qtaim"] is False

    def test_parse_only_ignores_the_flag(self, monkeypatch, tmp_path):
        # nothing can rerun QTAIM under parse_only, so enforcing would fail every pass
        seen = self._gbw(monkeypatch, tmp_path, _record(0, "standard"), enforce_poincare_hopf=True,
                         parse_only=True)
        assert seen["validate"]["enforce_poincare_hopf"] is False

    def test_clean_first_decides_before_the_record_moves(self, monkeypatch, tmp_path):
        # _clean_first moves generator/ aside; deciding afterwards found nothing and
        # reran the standard search forever
        from qtaim_gen.source.core import workflow
        seen = {}
        monkeypatch.setattr(workflow, "gbw_analysis", lambda *a, **k: seen.update(k) or False)
        folder, _ = _job(tmp_path, _record(0, "standard"))
        workflow.process_folder(str(folder), clean_first=True, enforce_poincare_hopf=True, move_results=True)
        assert seen["exhaustive_qtaim"] is True and seen["enforce_poincare_hopf"] is True

    def test_prevalidation_passes_the_flag(self, monkeypatch, tmp_path):
        from qtaim_gen.source.utils import io
        seen = {}
        monkeypatch.setattr(io, "validation_checks", lambda folder, **k: seen.update(k) or True)
        (tmp_path / "a").mkdir()
        lst = tmp_path / "jobs.txt"
        lst.write_text(f"{tmp_path / 'a'}\n")
        io.get_folders_from_file(str(lst), num_folders=10, pre_validate=True, enforce_poincare_hopf=True)
        assert seen.get("enforce_poincare_hopf") is True

    def test_runners_accept_the_flag(self):
        import importlib
        for mod in ("full_runner", "full_runner_parsl", "full_runner_parsl_alcf", "helpers.refine_list_of_jobs",
                    "helpers.sweep_truncated_steps"):
            src = Path(importlib.import_module(f"qtaim_gen.source.scripts.{mod}").__file__).read_text()
            assert '"--enforce_poincare_hopf"' in src, mod
