"""Per-suite statistics for the five held-out stress sets (H1, H3, H6, H7, H8).

Manifest-only. Reports composition overlap with the training split, element
coverage (the paper claims pair-level holdouts keep every element in training),
and the same nearest-neighbour statistics used for the main split.
"""
import glob, hashlib, os, re, sys
import numpy as np
import pandas as pd
from pymatgen.core.composition import Composition

MANIFEST_DIR = "data/omol_manifest"
HOLDOUT_PARQUET = "data/holdouts/manifest_holdout.parquet"
COLS = ["vertical", "rel_path", "job_id", "formula_hill", "element_set",
        "n_atoms", "net_charge_abs"]
_TOK = re.compile(r"([A-Z][a-z]?)(\d*)")


def canon_tail(v, r):
    for pre in (v + "/", v + "__"):
        if r.startswith(pre):
            r = r[len(pre):]
            break
    return r.replace("/", "__")


def parse_formula(f):
    d = {}
    for el, n in _TOK.findall(f):
        if el:
            d[el] = d.get(el, 0) + (int(n) if n else 1)
    return d


def shipped(f):
    pm = Composition(f).formula.replace(" ", "")
    h = int(hashlib.sha256(f"{pm}_42".encode()).hexdigest(), 16) % 10000 / 10000.0
    return "train" if h < 0.8 else ("val" if h < 0.9 else "test")


def by_element_set(formulas):
    g = {}
    for f in formulas:
        d = parse_formula(f)
        g.setdefault(frozenset(d), []).append(d)
    return {k: np.array([[d.get(e, 0) for e in sorted(k)] for d in v], dtype=np.int32)
            for k, v in g.items()}


def nn_l1(formulas, groups):
    has, dist = [], []
    for f in formulas:
        d = parse_formula(f)
        k = frozenset(d)
        G = groups.get(k)
        if G is None:
            has.append(False); continue
        v = np.array([d.get(e, 0) for e in sorted(k)], dtype=np.int32)
        has.append(True)
        dist.append(np.abs(G - v).sum(axis=1).min() / max(sum(d.values()), 1))
    return np.array(has), np.array(dist, dtype=float)


def main():
    df = pd.concat([pd.read_parquet(f, columns=COLS)
                    for f in sorted(glob.glob(f"{MANIFEST_DIR}/manifest_*.parquet"))],
                   ignore_index=True)
    df["ctail"] = [canon_tail(v, r) for v, r in zip(df.vertical, df.rel_path)]

    ho = pd.read_parquet(HOLDOUT_PARQUET)
    ho["ctail"] = [canon_tail(v, r) for v, r in zip(ho.vertical, ho.rel_path)]
    key2suites = {}
    for v, t, h in zip(ho.vertical, ho.ctail, ho.holdout_id):
        key2suites.setdefault((v, t), set()).add(h)

    df["suites"] = [key2suites.get((v, t)) for v, t in zip(df.vertical, df.ctail)]
    df["is_ho"] = df.suites.notna()
    keep = df[~df.is_ho].copy()

    uf = keep.formula_hill.unique()
    smap = {f: shipped(f) for f in uf}
    keep["split"] = keep.formula_hill.map(smap)
    train = keep[keep.split == "train"]
    train_formulas = train.formula_hill.unique()
    train_set = set(train_formulas)
    train_groups = by_element_set(train_formulas)
    train_elements = {e for f in train_formulas for e in parse_formula(f)}
    print(f"corpus {len(df):,} | held out {int(df.is_ho.sum()):,} | "
          f"train rows {len(train):,} | train unique formulas {len(train_formulas):,} | "
          f"train elements {len(train_elements)}")

    rows = []
    for suite in ["H1", "H3", "H6", "H7", "H8"]:
        m = df.suites.apply(lambda s: s is not None and suite in s)
        sub = df[m]
        f_uniq = sub.formula_hill.unique()
        els = {e for f in f_uniq for e in parse_formula(f)}
        missing = sorted(els - train_elements)
        has, dist = nn_l1(f_uniq, train_groups)
        rows.append(dict(
            suite=suite, n_structures=len(sub), n_unique_formulas=len(f_uniq),
            median_n_atoms=int(sub.n_atoms.median()),
            max_abs_charge=int(sub.net_charge_abs.max()),
            n_elements=len(els),
            elements_absent_from_train=len(missing),
            frac_exact_composition_in_train=float(np.mean([f in train_set for f in f_uniq])),
            frac_same_element_set_peer=float(has.mean()),
            l1_median=float(np.median(dist)) if len(dist) else np.nan,
            l1_p90=float(np.percentile(dist, 90)) if len(dist) else np.nan,
            n_verticals=sub.vertical.nunique(),
        ))
        if missing:
            print(f"  {suite}: elements absent from train -> {missing}")

    out = pd.DataFrame(rows)
    os.makedirs("data/analysis_outputs/split_similarity", exist_ok=True)
    p = "data/analysis_outputs/split_similarity/holdout_suite_stats.csv"
    out.to_csv(p, index=False)
    pd.set_option("display.width", 250)
    print()
    print(out.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nwrote {p}")


if __name__ == "__main__":
    sys.exit(main())
