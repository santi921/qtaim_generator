"""Criterion A of the QTAIM engine plan: at Multiwfn's own CP positions (from
CPprop.txt, printed to 12 decimals in Bohr), compare every property the engine
evaluates (qtaim_engine.point_properties) with Multiwfn's printed value.

Reports per property the largest relative difference |e - m| / |m| (absolute
for ellipticity and eta, printed with 6 decimals; spin relative to
max(|spin|, 1e-6 rho); the gradient against the position-rounding bound),
split by CP type. A NaN on either side,
or a property missing from CPprop.txt, reports inf, so it fails any threshold.

    python scripts/qtaim_compare_props.py --pairs pairs.txt
where each line of pairs.txt is "path/to/CPprop.txt path/to/orca.wfx".
"""

import argparse

import numpy as np

from qtaim_gen.source.core.charge_engine import prepare_basis, read_wfx
from qtaim_gen.source.core.parse_qtaim import load_cpprop_full
from qtaim_gen.source.core.qtaim_engine import point_properties

SCALARS = {
    "Density of all electrons": "rho",
    "Density of Alpha electrons": "rho_alpha",
    "Density of Beta electrons": "rho_beta",
    "Spin density of electrons": "spin",
    "Lagrangian kinetic energy G(r)": "G",
    "Hamiltonian kinetic energy K(r)": "K",
    "Potential energy density V(r)": "V",
    "Energy density E(r) or H(r)": "H",
    "Laplacian of electron density": "laplacian",
    "Electron localization function (ELF)": "elf",
    "Localized orbital locator (LOL)": "lol",
    "Average local ionization energy (ALIE)": "alie",
    "Determinant of Hessian": "det_hessian",
    "Ellipticity of electron density": "ellipticity",
    "eta index": "eta",
    "Delta-g (under Hirshfeld partition)": "delta_g_hirsh",
    "ESP from nuclear charges": "esp_nuc",
    "ESP from electrons": "esp_e",
    "Total ESP": "esp_total",
}
# printed with 6 decimals, so compared as absolute differences
ABSOLUTE = {"ellipticity", "eta"}
# Not compared at NCPs: there they are set by the CP's distance to its nucleus (1e-8
# to 1e-5 Bohr, where Newton stopped), which the 12-decimal positions do not resolve;
# esp_nuc is ~Z/d, and a printed position equal to the nucleus gives Multiwfn's 1000.
NCP_POSITION_LIMITED = {"esp_nuc", "esp_total", "delta_g_hirsh"}
# spin is relative to max(|spin|, SPIN_FLOOR * rho): an absolute difference would pass
# large relative errors where |spin| is small and flag print rounding where |spin| ~ 10.
# Spin is a difference of alpha and beta densities, so its roundoff scales with rho
# (~1e-16 rho, measured on a ROKS radical's SOMO node); below 1e-6 rho the relative
# measure is roundoff, not a spin difference.
SPIN_FLOOR = 1e-6


def _worst(d):
    """Largest difference, inf if any is NaN (np.nanmax would drop it)."""
    d = np.asarray(d, dtype=float)
    return float(np.inf) if np.isnan(d).any() else float(np.max(d))


def compare_cps(cps, wfx, basis, boys_mode="multiwfn"):
    """Per CP type: (count, {property: max difference}) between Multiwfn's
    printed values for cps (load_cpprop_full) and the engine at the printed
    positions. The ESP uses Multiwfn's Boys function (Fmch) by default, so the
    comparison checks the integrals; boys_mode="exact" measures the divergence. gradient_bound is max |g_engine - g_mwfn| over the rounding
    bound ||H||_2 * 8.7e-13 + 5e-11 |g| + 1e-20 (positions printed to 1e-12
    Bohr per coordinate, values to E18.10); <= 1 means consistent."""
    pts = np.array([c["pos_bohr"] for c in cps])
    eng = point_properties(pts, wfx, basis, esp=True, boys_mode=boys_mode)
    out = {}
    for lab in ("NCP", "BCP", "RCP", "CCP"):
        idx = [i for i, c in enumerate(cps) if c["label"] == lab]
        if not idx:
            continue
        res = {}
        for name, key in SCALARS.items():
            if lab == "NCP" and key in NCP_POSITION_LIMITED:
                continue
            ref = np.array([cps[i]["props"].get(name, np.nan) for i in idx])
            val = eng[key][idx]
            if key in ABSOLUTE:
                d = np.abs(val - ref)
            elif key == "spin":
                rho_ref = np.array([cps[i]["props"].get("Density of all electrons", np.nan) for i in idx])
                d = np.abs(val - ref) / np.maximum(np.abs(ref), SPIN_FLOOR * np.abs(rho_ref))
            else:
                d = np.abs(val - ref) / np.maximum(np.abs(ref), 1e-300)
            res[key] = _worst(d)
        ev_ref = np.array([cps[i]["eigenvalues"] for i in idx])
        res["eigenvalues"] = _worst(np.abs(eng["eigenvalues"][idx] - np.sort(ev_ref, axis=1)) / np.maximum(np.abs(ev_ref), 1e-300))
        g_ref = np.array([cps[i]["gradient"] for i in idx])
        dg = np.linalg.norm(eng["gradient"][idx] - g_ref, axis=1)
        bound = np.abs(eng["eigenvalues"][idx]).max(axis=1) * 8.7e-13 + 5e-11 * np.linalg.norm(g_ref, axis=1) + 1e-20
        res["gradient_bound"] = _worst(dg / bound)
        out[lab] = (len(idx), res)
    return out


def compare(cpprop, wfx_path):
    wfx = read_wfx(wfx_path)
    return compare_cps(load_cpprop_full(cpprop), wfx, prepare_basis(wfx))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pairs", required=True)
    args = p.parse_args()
    worst = {}
    for line in open(args.pairs):
        if not line.strip():
            continue
        cpprop, wfx = line.split()
        res = compare(cpprop, wfx)
        name = cpprop.split("/")[-2]
        for lab, (n, r) in res.items():
            print(f"{name[:34]} | {lab} x{n} | " + " ".join(f"{k} {v:.1e}" for k, v in r.items()))
            for k, v in r.items():
                worst[(lab, k)] = max(worst.get((lab, k), 0.0), v)  # v is never NaN (_worst)
    print("\nworst over all jobs (relative; absolute for ellipticity, eta; gradient_bound <= 1 is consistent):")
    for lab in ("NCP", "BCP", "RCP", "CCP"):
        row = {k: v for (l2, k), v in worst.items() if l2 == lab}
        if row:
            print(f"{lab}: " + " ".join(f"{k} {v:.1e}" for k, v in row.items()))


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------- whole-step comparisons (P3)

LABEL_TYPE = {"NCP": 0, "BCP": 1, "RCP": 2, "CCP": 3}
SAME_CP = 0.03  # Bohr, Multiwfn's minicpdis


def compare_cp_sets(ref_cps, res, weak_drop=1e-5):
    """Engine CPs (cp_search.analyze) against Multiwfn's (load_cpprop_full).
    Per type: counts [multiwfn, engine]; Multiwfn CPs with no engine CP of the
    same type within SAME_CP (missing; those below weak_drop, which analyze
    drops, are counted apart); engine CPs with no Multiwfn CP (extra); the
    largest position difference of the matched ones. BCP atom pairs, compared
    unordered as parse_qtaim does: same, differ, engine-only, Multiwfn-only."""
    out = {"counts": {}, "missing": {}, "missing_weak": {}, "extra": {}, "max_pos_diff": 0.0}
    pairs = {"same": 0, "differ": 0, "engine_only": 0, "multiwfn_only": 0, "both_unpaired": 0}
    for lab, t in LABEL_TYPE.items():
        ref = [c for c in ref_cps if c["label"] == lab]
        rp = np.array([c["pos_bohr"] for c in ref]).reshape(-1, 3)
        idx = np.flatnonzero(res["type"] == t)
        ep = res["pos"][idx]
        out["counts"][lab] = [len(rp), len(ep)]
        if len(rp) and len(ep):
            d = np.linalg.norm(rp[:, None] - ep[None], axis=2)
            near, dmin = d.argmin(axis=1), d.min(axis=1)
            matched = dmin < SAME_CP
            out["extra"][lab] = int((d.min(axis=0) >= SAME_CP).sum())
        else:
            near, dmin, matched = np.zeros(len(rp), int), np.full(len(rp), np.inf), np.zeros(len(rp), bool)
            out["extra"][lab] = len(ep)
        weak = np.array([c["props"].get("Density of all electrons", 1.0) < weak_drop for c in ref], dtype=bool)
        out["missing"][lab] = int((~matched & ~weak).sum())
        out["missing_weak"][lab] = int((~matched & weak).sum())
        if matched.any():
            out["max_pos_diff"] = max(out["max_pos_diff"], float(dmin[matched].max()))
        if lab == "BCP":
            for c, m, k in zip(ref, matched, near):
                if not m:
                    continue
                e = res["bcp_atoms"][idx[k]]
                ep_pair = None if e[0] < 0 else tuple(sorted(int(a) + 1 for a in e))
                mp = None if c["connected"] is None else tuple(sorted(c["connected"]))
                key = ("both_unpaired" if mp is None and ep_pair is None else "engine_only" if mp is None
                       else "multiwfn_only" if ep_pair is None else "same" if mp == ep_pair else "differ")
                pairs[key] += 1
    out["bcp_pairs"] = pairs
    return out


# qtaim.json fields: skipped, compared absolutely (printed f12.6), or set by the CP's
# distance to its nucleus at NCPs (see NCP_POSITION_LIMITED)
JSON_SKIP = {"cp_num", "delta_g_promolecular", "element", "number", "pos_ang", "connected_bond_paths"}
JSON_ABS = {"ellip_e_dens", "eta"}
JSON_NCP_LIMITED = {"esp_nuc", "esp_total", "delta_g_hirsh"}


def compare_qtaim_json(ref, eng):
    """Two qtaim.json dicts (parse_qtaim output or loaded JSON; keys compared as
    strings, "_" keys such as _meta ignored): keys on one side only, and per
    field the largest difference. Relative, except: ellipticity and eta
    absolute; grad_norm absolute (it is the convergence residual); spin
    relative to rho; esp_nuc, esp_total, delta_g_hirsh skipped at atom CPs."""
    ref = {str(k): v for k, v in ref.items() if not str(k).startswith("_")}
    eng = {str(k): v for k, v in eng.items() if not str(k).startswith("_")}
    worst = {}
    for k in set(ref) & set(eng):
        atom = "_" not in k
        r, e = ref[k], eng[k]
        for f, v in r.items():
            if f in JSON_SKIP or not isinstance(v, (int, float)) or not isinstance(e.get(f), (int, float)):
                continue
            if atom and f in JSON_NCP_LIMITED:
                continue
            if f in JSON_ABS or f == "grad_norm":
                d = abs(e[f] - v)
            elif f == "spin_density":
                d = abs(e[f] - v) / max(abs(r.get("density_all", 0.0)), 1e-300)
            else:
                d = abs(e[f] - v) / max(abs(v), 1e-300)
            if not np.isfinite(d):
                d = float("inf")
            if d > worst.get(f, (-1.0, ""))[0]:
                worst[f] = (d, k)
    return {"only_ref": sorted(set(ref) - set(eng)), "only_engine": sorted(set(eng) - set(ref)),
            "worst": {f: {"diff": d, "key": k} for f, (d, k) in worst.items()}}
