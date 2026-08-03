"""Tests for the audit->select->verify classification predicates.

The selector (select-qtaim-rerun) and verifier (verify-qtaim-rerun) must agree
with the runner's validator on what counts as a defect, or the campaign loops:
a selector stricter than the runner picks jobs the runner accepts unchanged,
and they never leave the queue.
"""

from qtaim_gen.source.scripts.helpers.select_qtaim_rerun import classify
from qtaim_gen.source.scripts.helpers.verify_qtaim_rerun import classify_state

# a healthy folder-mode audit CSV row (all strings, as csv.DictReader yields)
BASE_ROW = {
    "have_qtaim_json": "True",
    "have_qtaim_out": "True",
    "search_done": "True",
    "export_done": "True",
    "n_atoms": "12",
    "n_ncp": "12",
    "n_bcp": "9",
    "reported_bcp": "11",
    "bcp_shortfall": "2",
    "storable_bcp": "",
    "bcp_shortfall_storable": "2",
    "ncp_matches_atoms": "1",
    "n_cov_bonds": "9",
    "n_isolated_bonded": "0",
    "error": "",
}


def row(**overrides):
    r = dict(BASE_ROW)
    r.update(overrides)
    return r


class TestClassifyStorableBasis:
    def test_raw_shortfall_rescued_by_storable_is_control(self):
        """3 CPs with no storable atom pair: raw shortfall 3 > tolerance, but
        the validator rescues via storable_bcp_count and accepts the record.
        Selecting it would rerun -> accepted unchanged -> selected again."""
        assert (
            classify(row(bcp_shortfall="3", bcp_shortfall_storable="0"))
            == "control"
        )

    def test_real_storable_loss_still_selected(self):
        assert (
            classify(row(bcp_shortfall="5", bcp_shortfall_storable="5"))
            == "shortfall"
        )

    def test_blank_storable_column_falls_back_to_raw(self):
        assert (
            classify(row(bcp_shortfall="5", bcp_shortfall_storable=""))
            == "shortfall"
        )

    def test_old_csv_without_storable_column_falls_back_to_raw(self):
        r = row(bcp_shortfall="5")
        del r["bcp_shortfall_storable"]
        del r["storable_bcp"]
        assert classify(r) == "shortfall"

    def test_tolerance_respected_on_storable_basis(self):
        assert (
            classify(row(bcp_shortfall="7", bcp_shortfall_storable="2"))
            == "control"
        )


class TestClassifyNcpMismatch:
    def test_ncp_mismatch_is_not_a_control(self):
        """Fatal in validate_qtaim_dict regardless of flags, so the runner
        always reruns these; a 'control' label would make verify condemn a
        safe remedy as CONTROL_PERTURBED."""
        assert (
            classify(row(ncp_matches_atoms="0", bcp_shortfall="0",
                         bcp_shortfall_storable="0"))
            == "ncp_mismatch"
        )

    def test_old_csv_without_column_does_not_misfire(self):
        r = row(bcp_shortfall="0", bcp_shortfall_storable="0")
        del r["ncp_matches_atoms"]
        assert classify(r) == "control"

    def test_absence_modes_still_win(self):
        assert classify(row(have_qtaim_json="False", ncp_matches_atoms="0")) == (
            "no_qtaim_json"
        )


class TestEmptyBcpCompleteRuns:
    """A covalently-bonded geometry whose complete run reported <= tolerance
    bond CPs is accepted by the runner's validator: a standard rerun cannot
    change it, so the default modes must not loop on it. It keeps its own
    mode (empty_bcp_complete) for exhaustive-search campaigns."""

    def test_complete_zero_report_gets_own_mode(self):
        r = row(n_bcp="0", n_cov_bonds="9", reported_bcp="0",
                bcp_shortfall="0", bcp_shortfall_storable="0")
        assert classify(r) == "empty_bcp_complete"

    def test_lost_cps_with_empty_set_still_selected(self):
        r = row(n_bcp="0", n_cov_bonds="9", reported_bcp="11",
                bcp_shortfall="11", bcp_shortfall_storable="11")
        assert classify(r) == "empty_bcp"

    def test_bond_free_geometry_is_control(self):
        r = row(n_bcp="0", n_cov_bonds="0", reported_bcp="0",
                bcp_shortfall="0", bcp_shortfall_storable="0")
        assert classify(r) == "control"

    def test_classify_state_mirrors_complete_zero(self):
        after = {
            "error": "", "have_qtaim_json": True, "have_qtaim_out": True,
            "search_done": True, "export_done": True, "n_bcp": 0,
            "n_cov_bonds": 9, "reported_bcp": 0, "bcp_shortfall": 0,
            "bcp_shortfall_storable": 0, "ncp_matches_atoms": 1,
        }
        assert classify_state(after) == "empty_bcp_complete"
        after["reported_bcp"] = 11
        after["bcp_shortfall"] = after["bcp_shortfall_storable"] = 11
        assert classify_state(after) == "empty_bcp"


class TestClassifyStateMirrorsSelector:
    """verify's classify_state consumes audit_folder dicts (native types)."""

    @staticmethod
    def _after(**overrides):
        d = {
            "error": "",
            "have_qtaim_json": True,
            "have_qtaim_out": True,
            "search_done": True,
            "export_done": True,
            "n_bcp": 9,
            "n_cov_bonds": 9,
            "reported_bcp": 11,
            "bcp_shortfall": 2,
            "bcp_shortfall_storable": 2,
            "ncp_matches_atoms": 1,
        }
        d.update(overrides)
        return d

    def test_storable_rescue_reads_ok(self):
        after = self._after(bcp_shortfall=3, bcp_shortfall_storable=0)
        assert classify_state(after) == "ok"

    def test_real_loss_reads_shortfall(self):
        after = self._after(bcp_shortfall=5, bcp_shortfall_storable=5)
        assert classify_state(after) == "shortfall"

    def test_missing_storable_falls_back_to_raw(self):
        after = self._after(bcp_shortfall=5, bcp_shortfall_storable=None)
        assert classify_state(after) == "shortfall"

    def test_ncp_mismatch_is_not_ok(self):
        after = self._after(ncp_matches_atoms=0, bcp_shortfall=0,
                            bcp_shortfall_storable=0)
        assert classify_state(after) == "ncp_mismatch"
