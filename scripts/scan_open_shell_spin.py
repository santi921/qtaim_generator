"""Count records whose Multiwfn spin quantities are corrupted (open-shell .wfn read as all-alpha).

A record is corrupted when becke_fuzzy_spin does not sum to (multiplicity - 1).
Multiplicity is structure.lmdb's `spin` field. Reports per set and per vertical,
and writes the corrupted keys (these also carry wrong fuzzy_bond and qtaim
alpha/beta/spin/ELF/LOL values).

    python scan_open_shell_spin.py --root $ROOT --sets train val test --workers 64 --out spin_scan
    python scan_open_shell_spin.py --root $ROOT/holdouts --sets H1 H3 H6 H7 H8 --out spin_scan_holdouts
"""
import argparse
import json
import os
import pickle
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

import lmdb

TOL = 0.05


def _open(p):
    return lmdb.open(p, readonly=True, lock=False, subdir=False, readahead=False, max_readers=512)


def check_chunk(args):
    d, keys = args
    fz, st = _open(f"{d}/fuzzy.lmdb"), _open(f"{d}/structure.lmdb")
    c, bad = Counter(), []
    with fz.begin() as tf, st.begin() as ts:
        for k in keys:
            key = k.decode()
            v = key.split("__", 1)[0]
            s_raw = ts.get(k)
            if s_raw is None:
                continue
            mult = int(pickle.loads(s_raw)["spin"])
            shell = "open" if mult > 1 else "closed"
            c[(v, shell, "n")] += 1
            if shell == "closed":
                continue
            f_raw = tf.get(k)
            spin = pickle.loads(f_raw).get("becke_fuzzy_spin") if f_raw else None
            if not spin:
                c[(v, shell, "no_becke_spin")] += 1
                continue
            vals = [float(x) for a, x in spin.items() if a not in ("sum", "abs_sum")]
            if abs(sum(vals) - (mult - 1)) < TOL:
                c[(v, shell, "ok")] += 1
            else:
                c[(v, shell, "corrupted")] += 1
                bad.append(key)
    return c, bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--sets", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--chunk", type=int, default=20000)
    ap.add_argument("--out", default="spin_scan")
    a = ap.parse_args()

    report, all_bad = {}, []
    for s in a.sets:
        d = os.path.join(a.root, s)
        with _open(f"{d}/structure.lmdb").begin() as t:
            keys = [k for k in t.cursor().iternext(values=False) if k != b"length"]
        chunks = [(d, keys[i:i + a.chunk]) for i in range(0, len(keys), a.chunk)]
        tot = Counter()
        with ProcessPoolExecutor(a.workers) as ex:
            for c, bad in ex.map(check_chunk, chunks):
                tot.update(c)
                all_bad += [(s, k) for k in bad]
        per_v = {}
        for (v, shell, what), n in tot.items():
            per_v.setdefault(v, Counter())[f"{shell}_{what}"] += n
        summ = Counter()
        for cnt in per_v.values():
            summ.update(cnt)
        report[s] = {"total": dict(summ), "per_vertical": {v: dict(c) for v, c in sorted(per_v.items())}}
        print(s, json.dumps(dict(summ)), flush=True)

    with open(f"{a.out}.json", "w") as fh:
        json.dump(report, fh, indent=1)
    with open(f"{a.out}_corrupted_keys.tsv", "w") as fh:
        fh.write("set\tkey\n")
        for s, k in all_bad:
            fh.write(f"{s}\t{k}\n")
    print(f"corrupted: {len(all_bad)} -> {a.out}_corrupted_keys.tsv")


if __name__ == "__main__":
    main()
