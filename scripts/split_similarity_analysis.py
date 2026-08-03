"""Split leakage analysis: composition NN similarity + CH2 homologue probe.

Answers mgPn W4 / AC 3 without re-splitting. Manifest-only, all 34 verticals.
Reproduces the as-shipped split (sha256 of pymatgen Composition.formula).
"""
import argparse, glob, hashlib, os, re, sys
import numpy as np
import pandas as pd
from pymatgen.core.composition import Composition

MANIFEST_DIR = "data/omol_manifest"
HOLDOUT_PARQUET = "data/holdouts/manifest_holdout.parquet"


def canon_tail(vertical, rel_path):
    """Strip the vertical prefix and normalise separators to '__'."""
    for pre in (vertical + "/", vertical + "__"):
        if rel_path.startswith(pre):
            rel_path = rel_path[len(pre):]
            break
    return rel_path.replace("/", "__")


def load_corpus():
    frames = []
    for f in sorted(glob.glob(os.path.join(MANIFEST_DIR, "manifest_*.parquet"))):
        frames.append(pd.read_parquet(f, columns=["vertical", "rel_path", "formula_hill"]))
    df = pd.concat(frames, ignore_index=True)
    df["ctail"] = [canon_tail(v, r) for v, r in zip(df.vertical, df.rel_path)]
    return df


def mark_holdouts(df):
    ho = pd.read_parquet(HOLDOUT_PARQUET)
    keys = set(zip(ho.vertical, [canon_tail(v, r) for v, r in zip(ho.vertical, ho.rel_path)]))
    df["is_ho"] = [(v, t) in keys for v, t in zip(df.vertical, df.ctail)]
    return len(keys)


def shipped_split(formula_hill_values, seed=42):
    """The as-shipped assignment: sha256 of pymatgen's formula string."""
    out = {}
    for f in formula_hill_values:
        pm = Composition(f).formula.replace(" ", "")
        h = int(hashlib.sha256(f"{pm}_{seed}".encode()).hexdigest(), 16) % 10000 / 10000.0
        out[f] = "train" if h < 0.8 else ("val" if h < 0.9 else "test")
    return out


_TOK = re.compile(r"([A-Z][a-z]?)(\d*)")


def parse_formula(f):
    d = {}
    for el, n in _TOK.findall(f):
        if el:
            d[el] = d.get(el, 0) + (int(n) if n else 1)
    return d


def homologue_probe(test_formulas, train_set, max_k=5):
    """Fraction of test compositions with a train composition differing by k*CH2."""
    hits = np.zeros(max_k + 1, dtype=int)
    for f in test_formulas:
        d = parse_formula(f)
        c, h = d.get("C", 0), d.get("H", 0)
        if c == 0:
            continue
        rest = {k: v for k, v in d.items() if k not in ("C", "H")}
        for k in range(1, max_k + 1):
            found = False
            for sign in (+1, -1):
                nc, nh = c + sign * k, h + sign * 2 * k
                if nc < 1 or nh < 0:
                    continue
                cand = dict(rest); cand["C"] = nc
                if nh: cand["H"] = nh
                s = Composition(cand).formula.replace(" ", "")
                if s in train_set:
                    found = True; break
            if found:
                hits[k] += 1
    return hits


def by_element_set(formulas):
    """Group formulas by their element set: homologues must share it."""
    groups = {}
    for f in formulas:
        d = parse_formula(f)
        groups.setdefault(frozenset(d), []).append(d)
    return {k: np.array([[d.get(e, 0) for e in sorted(k)] for d in v], dtype=np.int32)
            for k, v in groups.items()}


def nn_within_element_set(test_formulas, train_groups):
    """Min L1 count distance to a train composition sharing the element set.

    Returns (has_peer, min_l1_norm). Compositions with no same-element-set peer in
    train get has_peer=False and are excluded from the distance distribution.
    """
    has, dist = [], []
    for f in test_formulas:
        d = parse_formula(f)
        k = frozenset(d)
        G = train_groups.get(k)
        if G is None or len(G) == 0:
            has.append(False); continue
        v = np.array([d.get(e, 0) for e in sorted(k)], dtype=np.int32)
        l1 = np.abs(G - v).sum(axis=1).min()
        has.append(True)
        dist.append(l1 / max(sum(d.values()), 1))
    return np.array(has), np.array(dist, dtype=np.float64)


def summarise(name, has_peer, dist, exact_frac):
    q = np.percentile(dist, [10, 50, 90]) if len(dist) else [np.nan] * 3
    return dict(scheme=name, n=len(has_peer),
                frac_exact_composition_in_train=exact_frac,
                frac_same_element_set_peer=float(has_peer.mean()),
                l1_p10=q[0], l1_median=q[1], l1_p90=q[2])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_test", type=int, default=10000, help="test formulas to sample")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/analysis_outputs/split_similarity")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    os.makedirs(a.out, exist_ok=True)

    df = load_corpus()
    n_ho_keys = mark_holdouts(df)
    print(f"corpus {len(df):,} rows | holdout keys {n_ho_keys:,} | matched {int(df.is_ho.sum()):,}")

    keep = df[~df.is_ho].copy()
    uf = keep.formula_hill.unique()
    print(f"post-holdout rows {len(keep):,} | unique formulas {len(uf):,}")
    smap = shipped_split(uf)
    keep["split"] = keep.formula_hill.map(smap)
    print("as-shipped split rows:", keep.split.value_counts().to_dict())

    elements = sorted({e for f in uf for e in parse_formula(f)})
    print(f"element dimensions: {len(elements)}")

    train_f = np.array([f for f in uf if smap[f] == "train"])
    test_f = np.array([f for f in uf if smap[f] == "test"])
    train_set = {Composition(f).formula.replace(" ", "") for f in train_f}
    print(f"unique train formulas {len(train_f):,} | unique test formulas {len(test_f):,}")

    rows = []

    def run(name, train_formulas, test_formulas, test_rows_formulas):
        tset = set(train_formulas)
        exact = float(np.mean([f in tset for f in test_rows_formulas]))
        groups = by_element_set(train_formulas)
        samp = rng.choice(test_formulas, size=min(a.n_test, len(test_formulas)), replace=False)
        has, dist = nn_within_element_set(samp, groups)
        rows.append(summarise(name, has, dist, exact))
        print(f"  {name} done")
        return samp

    # 1. the shipped composition split
    test_rows = keep[keep.split == "test"].formula_hill.values
    samp = run("composition (as shipped)", train_f, test_f, test_rows)

    # 2. ROW-LEVEL random split: the real leakage upper bound. Splitting unique
    #    formulas at random is still composition-consistent and is NOT a valid
    #    upper bound -- the rows themselves must be permuted.
    n = len(keep)
    perm = rng.permutation(n)
    r_tr_rows = keep.formula_hill.values[perm[: int(0.8 * n)]]
    r_te_rows = keep.formula_hill.values[perm[int(0.9 * n):]]
    run("random ROW-level (leakage upper bound)", np.unique(r_tr_rows),
        np.unique(r_te_rows), r_te_rows)

    # 3. whole-vertical: extrapolation lower bound
    for v in ["tm_react", "droplet", "rna"]:
        vt_rows = keep[keep.vertical == v].formula_hill.values
        vtr = keep[keep.vertical != v].formula_hill.unique()
        run(f"whole-vertical held out: {v}", vtr, np.unique(vt_rows), vt_rows)

    sim = pd.DataFrame(rows)
    sim.to_csv(os.path.join(a.out, "nn_similarity.csv"), index=False)
    print("\n=== composition nearest-neighbour: exact-composition leakage + same-element-set L1 ===")
    print(sim.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # ---- per-vertical post-split distribution table (paper App. reference) ----
    pv = (keep.groupby(["vertical", "split"]).size().unstack(fill_value=0)
          .reindex(columns=["train", "val", "test"], fill_value=0))
    excl = df[df.is_ho].groupby("vertical").size()
    pv["held_out"] = excl.reindex(pv.index).fillna(0).astype(int)
    pv["total"] = pv[["train", "val", "test", "held_out"]].sum(axis=1)
    pv = pv.sort_values("total", ascending=False)
    pv.to_csv(os.path.join(a.out, "per_vertical_split.csv"))

    tex = [
        "% Generated by scripts/split_similarity_analysis.py -- do not hand-edit.",
        "\\begin{table}[h]", "\\centering", "\\small",
        "\\caption{Per-vertical post-split distribution. \\emph{Held-out} counts are",
        "structures removed into the five stress suites before hashing; a structure in two",
        "suites is counted once. Train/val/test are the realized as-shipped assignment.}",
        "\\label{tab:per_vertical_split}",
        "\\begin{tabular}{lrrrrr}", "\\toprule",
        "Vertical & Total & Held-out & Train & Val & Test \\\\", "\\midrule",
    ]
    def g(x): return f"{int(x):,}".replace(",", "{,}")
    for v, r in pv.iterrows():
        tex.append(f"\\texttt{{{v.replace('_', chr(92)+'_')}}} & {g(r.total)} & "
                   f"{g(r.held_out)} & {g(r.train)} & {g(r.val)} & {g(r.test)} \\\\")
    tot = pv.sum()
    tex += ["\\midrule",
            f"\\textbf{{Total}} & {g(tot.total)} & {g(tot.held_out)} & {g(tot.train)} & "
            f"{g(tot.val)} & {g(tot.test)} \\\\",
            "\\bottomrule", "\\end{tabular}", "\\end{table}"]
    tex_path = "docs/neurips/tables/per_vertical_split.tex"
    os.makedirs(os.path.dirname(tex_path), exist_ok=True)
    with open(tex_path, "w") as fh:
        fh.write("\n".join(tex) + "\n")
    print(f"\nwrote {tex_path} ({len(pv)} verticals)")
    print(pv.to_string())

    hits = homologue_probe(samp, train_set)
    hp = pd.DataFrame({"k_CH2": range(1, 6), "n_test_with_train_homologue": hits[1:],
                       "frac": hits[1:] / len(samp)})
    hp.to_csv(os.path.join(a.out, "homologue_probe.csv"), index=False)
    print(f"\n=== CH2 homologue probe (n_test sampled = {len(samp):,}) ===")
    print(hp.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nwrote {a.out}/nn_similarity.csv and homologue_probe.csv")


if __name__ == "__main__":
    sys.exit(main())
