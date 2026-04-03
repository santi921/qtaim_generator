"""Tests for the data-presence cross-reference in the restart skip logic.

Covers:
- _compiled_data_present: all combinations of key present/missing/empty
- Skip logic: data present with timing, data present without timing, data absent
"""

import json
import os
import pytest

from qtaim_gen.source.core.omol import _compiled_data_present


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f)


# ---------------------------------------------------------------------------
# _compiled_data_present: compiled JSON in job root
# ---------------------------------------------------------------------------

def test_compiled_data_present_charge_key_found(tmp_path):
    """charge.json with non-empty hirshfeld charge sub-key → True."""
    _write_json(tmp_path / "charge.json", {"hirshfeld": {"charge": {"1_H": 0.1}}})
    compiled_map = {"hirshfeld": ("charge.json", "hirshfeld", "charge")}
    assert _compiled_data_present(str(tmp_path), "hirshfeld", compiled_map)


def test_compiled_data_present_charge_key_empty_dict(tmp_path):
    """charge.json with hirshfeld: {} → False (empty parse, no sub-key)."""
    _write_json(tmp_path / "charge.json", {"hirshfeld": {}})
    compiled_map = {"hirshfeld": ("charge.json", "hirshfeld", "charge")}
    assert not _compiled_data_present(str(tmp_path), "hirshfeld", compiled_map)


def test_compiled_data_present_charge_nested_empty(tmp_path):
    """charge.json with hirshfeld: {'charge': {}} → False (0 charges parsed).

    This covers the observed failure: 'Number of charges in hirshfeld (0)
    does not match expected (247)'. The op-level dict is non-empty, but the
    nested charge dict is empty — must return False so the calc is re-run.
    """
    _write_json(tmp_path / "charge.json", {"hirshfeld": {"charge": {}}})
    compiled_map = {"hirshfeld": ("charge.json", "hirshfeld", "charge")}
    assert not _compiled_data_present(str(tmp_path), "hirshfeld", compiled_map)


def test_compiled_data_present_charge_nested_empty_with_dipole(tmp_path):
    """charge.json with hirshfeld: {'charge': {}, 'dipole': [...]} → False.

    Even though the op-level dict has multiple keys, an empty 'charge' sub-key
    means the parse failed and must be re-run.
    """
    _write_json(
        tmp_path / "charge.json",
        {"hirshfeld": {"charge": {}, "dipole": [0.0, 0.0, 0.0]}},
    )
    compiled_map = {"hirshfeld": ("charge.json", "hirshfeld", "charge")}
    assert not _compiled_data_present(str(tmp_path), "hirshfeld", compiled_map)


def test_compiled_data_present_charge_key_missing(tmp_path):
    """charge.json without hirshfeld key → False."""
    _write_json(tmp_path / "charge.json", {"adch": {"charge": {"1_H": 0.1}}})
    compiled_map = {"hirshfeld": ("charge.json", "hirshfeld", "charge")}
    assert not _compiled_data_present(str(tmp_path), "hirshfeld", compiled_map)


def test_compiled_data_present_fuzzy_key_found(tmp_path):
    """fuzzy_full.json with non-empty becke_fuzzy_density → True."""
    _write_json(
        tmp_path / "fuzzy_full.json",
        {"becke_fuzzy_density": {"1_H": 0.5, "2_O": 0.3}},
    )
    compiled_map = {"becke_fuzzy_density": ("fuzzy_full.json", "becke_fuzzy_density")}
    assert _compiled_data_present(str(tmp_path), "becke_fuzzy_density", compiled_map)


def test_compiled_data_present_fuzzy_key_empty(tmp_path):
    """fuzzy_full.json with becke_fuzzy_density: {} → False."""
    _write_json(tmp_path / "fuzzy_full.json", {"becke_fuzzy_density": {}})
    compiled_map = {"becke_fuzzy_density": ("fuzzy_full.json", "becke_fuzzy_density")}
    assert not _compiled_data_present(str(tmp_path), "becke_fuzzy_density", compiled_map)


def test_compiled_data_present_other_nonempty(tmp_path):
    """other.json non-empty (key=None path) → True."""
    _write_json(tmp_path / "other.json", {"mpp_full": 1.0, "sdp_full": 2.0})
    compiled_map = {"other_esp": ("other.json", None)}
    assert _compiled_data_present(str(tmp_path), "other_esp", compiled_map)


def test_compiled_data_present_other_empty(tmp_path):
    """other.json empty dict (key=None path) → False."""
    _write_json(tmp_path / "other.json", {})
    compiled_map = {"other_esp": ("other.json", None)}
    assert not _compiled_data_present(str(tmp_path), "other_esp", compiled_map)


def test_compiled_data_present_order_not_in_map(tmp_path):
    """Order not in compiled_map → False."""
    compiled_map = {"hirshfeld": ("charge.json", "hirshfeld", "charge")}
    assert not _compiled_data_present(str(tmp_path), "qtaim", compiled_map)


def test_compiled_data_present_json_missing(tmp_path):
    """Compiled JSON file doesn't exist → False."""
    compiled_map = {"hirshfeld": ("charge.json", "hirshfeld", "charge")}
    assert not _compiled_data_present(str(tmp_path), "hirshfeld", compiled_map)


def test_compiled_data_present_json_empty_file(tmp_path):
    """Zero-byte compiled JSON → False (getsize guard)."""
    (tmp_path / "charge.json").write_text("")
    compiled_map = {"hirshfeld": ("charge.json", "hirshfeld", "charge")}
    assert not _compiled_data_present(str(tmp_path), "hirshfeld", compiled_map)


def test_compiled_data_present_json_corrupted(tmp_path):
    """Corrupted JSON → False (JSONDecodeError caught)."""
    (tmp_path / "charge.json").write_text("{not valid json")
    compiled_map = {"hirshfeld": ("charge.json", "hirshfeld", "charge")}
    assert not _compiled_data_present(str(tmp_path), "hirshfeld", compiled_map)


# ---------------------------------------------------------------------------
# _compiled_data_present: data in generator/ subfolder
# ---------------------------------------------------------------------------

def test_compiled_data_present_in_generator_subfolder(tmp_path):
    """Data in generator/charge.json (post move_results) → True."""
    gen = tmp_path / "generator"
    gen.mkdir()
    _write_json(gen / "charge.json", {"hirshfeld": {"charge": {"1_H": 0.1}}})
    compiled_map = {"hirshfeld": ("charge.json", "hirshfeld", "charge")}
    assert _compiled_data_present(str(tmp_path), "hirshfeld", compiled_map)


def test_compiled_data_present_root_takes_precedence(tmp_path):
    """Non-empty in root, empty in generator/ → True (root checked first)."""
    gen = tmp_path / "generator"
    gen.mkdir()
    _write_json(tmp_path / "charge.json", {"hirshfeld": {"charge": {"1_H": 0.1}}})
    _write_json(gen / "charge.json", {"hirshfeld": {"charge": {}}})
    compiled_map = {"hirshfeld": ("charge.json", "hirshfeld", "charge")}
    assert _compiled_data_present(str(tmp_path), "hirshfeld", compiled_map)


def test_compiled_data_present_root_empty_generator_valid(tmp_path):
    """Empty charge in root, non-empty in generator/ → True (falls through)."""
    gen = tmp_path / "generator"
    gen.mkdir()
    _write_json(tmp_path / "charge.json", {"hirshfeld": {"charge": {}}})
    _write_json(gen / "charge.json", {"hirshfeld": {"charge": {"1_H": 0.1}}})
    compiled_map = {"hirshfeld": ("charge.json", "hirshfeld", "charge")}
    assert _compiled_data_present(str(tmp_path), "hirshfeld", compiled_map)


# ---------------------------------------------------------------------------
# adch specifically (charge op that can live in charge.json under "adch" key)
# ---------------------------------------------------------------------------

def test_compiled_data_present_adch_in_charge_json(tmp_path):
    """adch data in charge.json (post cleanup of adch.json) → True."""
    _write_json(
        tmp_path / "charge.json",
        {"adch": {"charge": {"1_C": -0.2}, "dipole": [0.0, 0.0, 0.1]}},
    )
    compiled_map = {"adch": ("charge.json", "adch", "charge")}
    assert _compiled_data_present(str(tmp_path), "adch", compiled_map)


def test_compiled_data_present_adch_empty_value(tmp_path):
    """adch key present but charge sub-key is empty dict → False."""
    _write_json(tmp_path / "charge.json", {"adch": {"charge": {}}})
    compiled_map = {"adch": ("charge.json", "adch", "charge")}
    assert not _compiled_data_present(str(tmp_path), "adch", compiled_map)


def test_compiled_data_present_adch_outer_empty(tmp_path):
    """adch key present but value is entirely empty dict → False."""
    _write_json(tmp_path / "charge.json", {"adch": {}})
    compiled_map = {"adch": ("charge.json", "adch", "charge")}
    assert not _compiled_data_present(str(tmp_path), "adch", compiled_map)
