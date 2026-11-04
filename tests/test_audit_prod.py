"""scripts/nersc/audit_prod.py on synthetic full_set 1 folders."""
import importlib.util
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _audit_module():
    spec = importlib.util.spec_from_file_location("audit_prod", os.path.join(ROOT, "scripts", "nersc", "audit_prod.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _folder(path, mbis_shift=0.0, meta=None):
    gen = path / "generator"
    gen.mkdir(parents=True)
    (gen / "gbw_analysis.log").write_text("Validation status: True\n")
    q = {"1_Th": 0.6 + mbis_shift, "2_O": -0.6 - mbis_shift}
    charge = {s: {"charge": {"1_Th": 0.6, "2_O": -0.6}} for s in ("hirshfeld", "adch", "cm5", "becke", "vdd", "chelpg")}
    charge["mbis"] = {"charge": q}
    (gen / "charge.json").write_text(json.dumps(charge))
    fuzzy = {s: {"1_Th": 89.4, "2_O": 8.6} for s in ("hirsh_fuzzy_density", "becke_fuzzy_density", "mbis_fuzzy_density")}
    fuzzy.update({s: {"1_Th": 1.5, "2_O": 0.5} for s in ("hirsh_fuzzy_spin", "becke_fuzzy_spin", "mbis_fuzzy_spin")})
    (gen / "fuzzy_full.json").write_text(json.dumps(fuzzy))
    (gen / "timings.json").write_text(json.dumps({"charge_engine": 12.5}))
    (gen / "qtaim.json").write_text(json.dumps({"1": {}, "2": {}, "_meta": {"poincare_hopf": 1}}))
    if meta is not None:
        (gen / "charge_engine_meta.json").write_text(json.dumps(meta))
    return str(path)


def test_audit_reads_level1_schemes_and_mbis_meta(tmp_path, monkeypatch, capsys):
    A = _audit_module()
    mol = {"1": {"element": "Th"}, "2": {"element": "O"}}
    monkeypatch.setattr(A, "get_charge_spin_n_atoms_from_folder", lambda f: {"mol": mol, "charge": 0, "spin": 3})
    monkeypatch.setattr(A, "all_electron_count", lambda dft: 98)
    meta = {"mbis_selection": {"selected": "sigma_x1.5", "converged": {"default": True, "sigma_x0.6": False,
                                                                      "sigma_x1.5": True},
                               "solutions": 2, "spread": 0.02},
            "mbis_merged": {"charge": [{"atom": 1, "z": 90, "shells_guess": 6, "shells_fit": 5}], "fuzzy": []}, "mbis_cycles": {"charge": 2400, "fuzzy": 900},
            "mbis_degenerate": {"charge": False, "fuzzy": False}}
    new = _folder(tmp_path / "new" / "job", mbis_shift=1e-4, meta=meta)
    rec = A.audit(new)
    assert rec["validated"] and rec["flags"] == []
    for s in ("vdd", "mbis", "chelpg"):
        assert rec[f"sum_{s}"] < 1e-12
    assert rec["sum_mbis_fuzzy_spin"] == 0.0 and rec["rel_mbis_fuzzy_density"] == 0.0
    assert (rec["mbis_selected"], rec["mbis_converged"], rec["mbis_guesses_converged"]) == ("sigma_x1.5", True, 2)
    assert (rec["mbis_merged_atoms"], rec["mbis_cycles"], rec["mbis_solutions"]) == (1, 2400, 2)

    meta["mbis_selection"]["converged"]["sigma_x1.5"] = False
    meta["mbis_degenerate"]["fuzzy"] = True
    (tmp_path / "new" / "job" / "generator" / "charge_engine_meta.json").write_text(json.dumps(meta))
    assert set(A.audit(new)["flags"]) == {"mbis_unconverged", "mbis_degenerate"}

    ref = _folder(tmp_path / "ref" / "job")
    out = A.compare(new, ref)
    assert abs(out["d_mbis"] - 1e-4) < 1e-12 and out["d_vdd"] == 0.0 and out["d_chelpg"] == 0.0
    assert out["d_mbis_fuzzy_density"] == 0.0 and out["d_mbis_fuzzy_spin"] == 0.0

    monkeypatch.setattr("sys.argv", ["audit_prod.py", str(tmp_path / "new"), "--ref", str(tmp_path / "ref")])
    A.main()
    printed = capsys.readouterr().out
    assert "| mbis | 1 |" in printed and "MBIS (charge_engine_meta.json, 1 folders)" in printed
