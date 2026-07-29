"""audit-qtaim-connectivity: find qtaim.json records with missing bond CPs.

Motivation: a truncated Multiwfn `CPprop.txt` silently loses critical points
from the *tail* of the CP list. `validate_qtaim_dict` cannot detect this because
it only checks that the nuclear-CP count matches the atom count, and Multiwfn
numbers nuclear CPs first - so the surviving prefix always passes.

The signature it does leave is a **broken molecular graph**: with
`bonding_scheme="qtaim"` the BCP set is the graph edge set, so lost BCPs show up
as atoms with no bond critical point at all, or as covalently-close atom pairs
with no BCP between them.

Distinguishing a real defect from legitimate topology needs a reference, since
several verticals (5A_elytes, droplet, noble_gas) are genuinely multi-fragment
and can contain bare ions with no bonds. This audit uses covalent-radius
bonding from the structure LMDB (`get_bonds_from_coords`) as that reference and
reports two signals per record:

- `n_isolated_bonded`: atoms with zero BCPs that the covalent reference says
  are bonded to something. Near-unambiguous: a covalently bound atom always has
  at least one BCP.
- `n_missing_cov_bonds`: covalent reference bonds with no corresponding BCP.
  Broader, and expected to be nonzero for ionic/metal contacts, so use it as a
  ranking signal rather than a pass/fail.

Runs entirely on existing LMDBs - no wavefunctions, no recomputation.

Example:
    audit-qtaim-connectivity --lmdb_root data/OMol4M_lmdbs \
        --out_csv qtaim_connectivity_audit.csv
"""

import argparse
import csv
import os
import pickle
import sys
import warnings
from typing import List, Optional

warnings.filterwarnings("ignore")


def bcp_pairs(record: dict) -> List[tuple]:
    """0-based atom pairs from a qtaim.json/lmdb record."""
    pairs = []
    for k in record:
        if k == "_meta" or "_" not in k:
            continue
        try:
            i, j = (int(x) for x in k.split("_"))
        except ValueError:
            continue
        pairs.append((min(i, j), max(i, j)))
    return pairs


def n_components(n_atoms: int, pairs) -> int:
    """Connected components of the BCP graph."""
    adj = {i: set() for i in range(n_atoms)}
    for i, j in pairs:
        if i in adj and j in adj:
            adj[i].add(j)
            adj[j].add(i)
    seen, comps = set(), 0
    for a in range(n_atoms):
        if a in seen:
            continue
        comps += 1
        stack = [a]
        while stack:
            u = stack.pop()
            if u in seen:
                continue
            seen.add(u)
            stack.extend(adj[u] - seen)
    return comps


def audit_record(structure_rec: dict, qtaim_rec: dict, covalent_factor: float) -> dict:
    from qtaim_gen.source.utils.io import get_bonds_from_coords

    mol = structure_rec["molecule"]
    species = [str(el) for el in mol.species]
    coords = mol.cart_coords
    n_atoms = len(species)

    pairs = set(bcp_pairs(qtaim_rec))
    n_ncp = sum(1 for k in qtaim_rec if k != "_meta" and "_" not in k)

    cov = {(min(i, j), max(i, j)) for i, j in
           get_bonds_from_coords(species, coords, covalent_factor)}
    cov_degree = {a: 0 for a in range(n_atoms)}
    for i, j in cov:
        cov_degree[i] += 1
        cov_degree[j] += 1

    bcp_degree = {a: 0 for a in range(n_atoms)}
    for i, j in pairs:
        if i in bcp_degree:
            bcp_degree[i] += 1
        if j in bcp_degree:
            bcp_degree[j] += 1

    isolated_bonded = [
        a for a in range(n_atoms) if bcp_degree[a] == 0 and cov_degree[a] > 0
    ]
    missing_cov = sorted(cov - pairs)

    return {
        "n_atoms": n_atoms,
        "n_ncp": n_ncp,
        "n_bcp": len(pairs),
        "n_cov_bonds": len(cov),
        "n_components": n_components(n_atoms, pairs),
        "n_isolated_bonded": len(isolated_bonded),
        "isolated_bonded": " ".join(str(a) for a in isolated_bonded[:12]),
        "n_missing_cov_bonds": len(missing_cov),
        "missing_cov_bonds": " ".join(f"{i}_{j}" for i, j in missing_cov[:12]),
        "ncp_matches_atoms": int(n_ncp == n_atoms),
    }


def find_job_folders(root: str, max_depth: int = 8):
    """Yield job folders under root, i.e. those holding a qtaim.json.

    The OMol4M hierarchy is jagged, so this walks rather than globbing a fixed
    depth. Checks the folder root and generator/ (post-cleanup layout).
    """
    root = os.path.abspath(root)
    base_depth = root.rstrip(os.sep).count(os.sep)
    for dirpath, dirnames, filenames in os.walk(root):
        if dirpath.count(os.sep) - base_depth >= max_depth:
            dirnames[:] = []
        if "qtaim.json" in filenames:
            yield dirpath
            dirnames[:] = [d for d in dirnames if d != "generator"]
        elif "generator" in dirnames and os.path.isfile(
            os.path.join(dirpath, "generator", "qtaim.json")
        ):
            yield dirpath
            dirnames[:] = [d for d in dirnames if d != "generator"]


def audit_folder(folder: str, covalent_factor: float) -> dict:
    """Audit one job folder on disk (no LMDBs needed).

    On a production folder this also gets the stronger signal: Multiwfn's own
    reported (3,-1) count from qtaim.out, which survives inside
    generator/out_files.zip after cleanup.
    """
    import json as _json

    from qtaim_gen.source.core.parse_qtaim import dft_inp_to_dict
    from qtaim_gen.source.utils.validation import count_reported_bcps

    qpath = os.path.join(folder, "qtaim.json")
    if not os.path.isfile(qpath):
        qpath = os.path.join(folder, "generator", "qtaim.json")
    with open(qpath) as f:
        qtaim_rec = _json.load(f)

    inp = None
    for cand in ("orca.inp", "input.in", "orca.in", "input.inp"):
        for base in (folder, os.path.join(folder, "generator")):
            p = os.path.join(base, cand)
            if os.path.isfile(p):
                inp = p
                break
        if inp:
            break
    if inp is None:
        raise FileNotFoundError("no ORCA input found for geometry")

    dft = dft_inp_to_dict(inp)
    species = [dft[k]["element"] for k in sorted(dft)]
    coords = [dft[k]["pos"] for k in sorted(dft)]

    class _Mol:
        pass

    mol = _Mol()
    mol.species = species
    mol.cart_coords = coords
    row = audit_record({"molecule": mol}, qtaim_rec, covalent_factor)

    reported = count_reported_bcps(folder)
    row["reported_bcp"] = reported
    row["bcp_shortfall"] = (
        reported - row["n_bcp"] if reported is not None else None
    )

    # Leading hypothesis for lost CPs is the QTAIM step being killed mid-write
    # (walltime/OOM), which would leave a partial CPprop.txt that still parses.
    # Carrying the step's wall time lets that be tested directly: affected jobs
    # should cluster at long qtaim times, or pile up against a ceiling.
    row["qtaim_time_s"] = None
    row["total_time_s"] = None
    for base in (folder, os.path.join(folder, "generator")):
        tpath = os.path.join(base, "timings.json")
        if os.path.isfile(tpath) and os.path.getsize(tpath) > 0:
            try:
                with open(tpath) as f:
                    timings = _json.load(f)
            except (ValueError, OSError):
                continue
            if isinstance(timings, dict):
                row["qtaim_time_s"] = timings.get("qtaim")
                numeric = [
                    v for v in timings.values() if isinstance(v, (int, float)) and v > 0
                ]
                row["total_time_s"] = round(sum(numeric), 2) if numeric else None
            break
    return row


def _run_folder_mode(args) -> int:
    import concurrent.futures
    import csv as _csv

    roots = args.folder_root.split(",") if "," in args.folder_root else [args.folder_root]
    if args.verticals:
        roots = [os.path.join(args.folder_root, v) for v in args.verticals]

    folders = []
    for r in roots:
        if not os.path.isdir(r):
            print(f"skip missing root: {r}", file=sys.stderr)
            continue
        found = list(find_job_folders(r, args.max_depth))
        print(f"{os.path.basename(r.rstrip(os.sep)):<24} job folders: {len(found)}", flush=True)
        folders.extend(found)
        if args.limit and len(folders) >= args.limit:
            folders = folders[: args.limit]
            break
    if not folders:
        print("no job folders with qtaim.json found", file=sys.stderr)
        return 1

    def one(folder):
        try:
            row = audit_folder(folder, args.covalent_factor)
        except Exception as e:
            row = {"n_isolated_bonded": -1, "error": f"{type(e).__name__}: {e}"[:120]}
        row["vertical"] = os.path.relpath(folder, args.folder_root).split(os.sep)[0]
        row["key"] = os.path.basename(folder)
        row["folder"] = folder
        return row

    rows = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        for i, row in enumerate(ex.map(one, folders)):
            rows.append(row)
            if (i + 1) % 2000 == 0:
                print(f"  {i + 1}/{len(folders)} audited", flush=True)

    fields = [
        "vertical", "key", "folder", "n_atoms", "n_ncp", "n_bcp", "reported_bcp",
        "bcp_shortfall", "qtaim_time_s", "total_time_s",
        "n_cov_bonds", "n_components", "n_isolated_bonded",
        "isolated_bonded", "n_missing_cov_bonds", "missing_cov_bonds",
        "ncp_matches_atoms", "error",
    ]
    with open(args.out_csv, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    ok = [r for r in rows if r.get("n_atoms")]
    errs = [r for r in rows if r.get("error")]
    empty = [r for r in ok if r.get("n_bcp") == 0]
    short = [r for r in ok if (r.get("bcp_shortfall") or 0) > 0]
    iso = [r for r in ok if r.get("n_isolated_bonded", 0) > 0]
    severe = [
        r for r in iso
        if r["n_atoms"] and r["n_isolated_bonded"] / r["n_atoms"] > 0.10
    ]
    n = len(ok) or 1
    with_prov = [r for r in ok if r.get("reported_bcp") is not None]
    print(f"\n{len(rows)} folders audited -> {args.out_csv}  ({len(errs)} errors)")
    print(
        f"  qtaim.out provenance available: {len(with_prov):>7} "
        f"({100*len(with_prov)/n:.1f}%)"
    )
    if not with_prov:
        print(
            "  !! WARNING: no folder had qtaim.out (checked root, generator/, and\n"
            "     generator/out_files.zip), so the BCP-shortfall check below is\n"
            "     VACUOUS -- a 0% result means 'not checked', not 'clean'. Only the\n"
            "     connectivity signals are meaningful for this run."
        )
    elif len(with_prov) < 0.5 * n:
        print(
            f"  !! WARNING: only {100*len(with_prov)/n:.1f}% of folders had qtaim.out, so the "
            "shortfall rate\n     below is computed over that subset only."
        )
    m = len(with_prov) or 1
    print(f"  empty BCP set:                 {len(empty):>7} ({100*len(empty)/n:.3f}%)")
    print(
        f"  fewer BCPs than Multiwfn said: {len(short):>7} "
        f"({100*len(short)/m:.3f}% of those with provenance)  <- strongest signal"
    )
    print(f"  severe (>10% atoms isolated):  {len(severe):>7} ({100*len(severe)/n:.3f}%)")
    print(f"  any isolated bonded atom:      {len(iso):>7} ({100*len(iso)/n:.3f}%)")
    unamb = {id(r) for r in empty} | {id(r) for r in short} | {id(r) for r in severe}
    print(f"  UNAMBIGUOUS DEFECTS:           {len(unamb):>7} ({100*len(unamb)/n:.3f}%)")

    import statistics as _st

    def _aff(r):
        return (r.get("bcp_shortfall") or 0) > 0

    timed = [r for r in with_prov if isinstance(r.get("qtaim_time_s"), (int, float))]
    if timed:
        bad = [r["qtaim_time_s"] for r in timed if _aff(r)]
        good = [r["qtaim_time_s"] for r in timed if not _aff(r)]
        if bad and good:
            print(
                f"\n  qtaim step wall time (s), median: "
                f"affected={_st.median(bad):.1f}  unaffected={_st.median(good):.1f}"
            )
            print(
                f"    max affected={max(bad):.1f}  max unaffected={max(good):.1f}"
            )
            if max(good) >= max(bad):
                print(
                    "    note: the longest job is UNAFFECTED, so there is no runtime\n"
                    "    ceiling -- this argues against a walltime kill."
                )

    # Runtime and size are confounded: larger systems take longer AND have more
    # CPs to lose. Bin by size, then compare times within a bin, to see which
    # actually drives the loss.
    sized = [r for r in with_prov if isinstance(r.get("n_atoms"), int)]
    if sized:
        print("\n  shortfall rate by system size (controls for the size/time confound):")
        print(f"    {'atoms':<12}{'n':>7}{'affected':>10}{'rate':>9}"
              f"{'med t aff':>11}{'med t un':>10}{'med BCPs':>10}")
        for lo, hi in ((0, 50), (50, 100), (100, 200), (200, 400), (400, 10**6)):
            sub = [r for r in sized if lo <= r["n_atoms"] < hi]
            if not sub:
                continue
            a = [r for r in sub if _aff(r)]
            u = [r for r in sub if not _aff(r)]
            ta = [r["qtaim_time_s"] for r in a
                  if isinstance(r.get("qtaim_time_s"), (int, float))]
            tu = [r["qtaim_time_s"] for r in u
                  if isinstance(r.get("qtaim_time_s"), (int, float))]
            bcps = [r["reported_bcp"] for r in sub
                    if isinstance(r.get("reported_bcp"), int)]
            label = f"{lo}-{hi}" if hi < 10**6 else f"{lo}+"
            print(
                f"    {label:<12}{len(sub):>7}{len(a):>10}{100*len(a)/len(sub):>8.2f}%"
                f"{(_st.median(ta) if ta else float('nan')):>11.1f}"
                f"{(_st.median(tu) if tu else float('nan')):>10.1f}"
                f"{(_st.median(bcps) if bcps else float('nan')):>10.1f}"
            )
        print(
            "    if the rate rises with size but times match within a bin, size (memory\n"
            "    or write volume) is the driver, not elapsed time."
        )
    if errs:
        import collections as _c
        print("\n  error kinds:", dict(_c.Counter(
            r["error"].split(":")[0] for r in errs).most_common(5)))
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    import lmdb

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lmdb_root",
        default=None,
        help="audit from per-vertical LMDBs ({root}/{vertical}/{qtaim,structure}.lmdb)",
    )
    parser.add_argument(
        "--folder_root",
        default=None,
        help=(
            "audit raw job-folder trees instead (what production HPC layouts "
            "look like). Also cross-checks Multiwfn's reported BCP count from "
            "qtaim.out, which the LMDB mode cannot see."
        ),
    )
    parser.add_argument(
        "--workers", type=int, default=8, help="folder-mode parallelism"
    )
    parser.add_argument("--max_depth", type=int, default=8)
    parser.add_argument(
        "--verticals",
        nargs="+",
        default=None,
        help="default: every subdirectory holding both qtaim.lmdb and structure.lmdb",
    )
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--covalent_factor", type=float, default=1.3)
    parser.add_argument("--limit", type=int, default=None, help="records per vertical")
    args = parser.parse_args(argv)

    if not args.lmdb_root and not args.folder_root:
        print("give --lmdb_root or --folder_root", file=sys.stderr)
        return 2

    if args.folder_root:
        return _run_folder_mode(args)

    verticals = args.verticals
    if verticals is None:
        verticals = sorted(
            d
            for d in os.listdir(args.lmdb_root)
            if os.path.isfile(os.path.join(args.lmdb_root, d, "qtaim.lmdb"))
            and os.path.isfile(os.path.join(args.lmdb_root, d, "structure.lmdb"))
        )
    if not verticals:
        print("no verticals with both qtaim.lmdb and structure.lmdb", file=sys.stderr)
        return 2

    rows = []
    for vert in verticals:
        qpath = os.path.join(args.lmdb_root, vert, "qtaim.lmdb")
        spath = os.path.join(args.lmdb_root, vert, "structure.lmdb")
        qenv = lmdb.open(qpath, readonly=True, lock=False, subdir=False)
        senv = lmdb.open(spath, readonly=True, lock=False, subdir=False)
        n_seen = n_flag = 0
        try:
            with qenv.begin() as qt, senv.begin() as st:
                for k, v in qt.cursor():
                    key = k.decode()
                    if key in ("length", "__len__"):
                        continue
                    raw_s = st.get(k)
                    if raw_s is None:
                        continue
                    try:
                        row = audit_record(
                            pickle.loads(raw_s), pickle.loads(v), args.covalent_factor
                        )
                    except Exception as e:
                        row = {
                            "n_atoms": None,
                            "error": f"{type(e).__name__}: {e}"[:120],
                            "n_isolated_bonded": -1,
                        }
                    row["vertical"] = vert
                    row["key"] = key
                    rows.append(row)
                    n_seen += 1
                    if row.get("n_isolated_bonded", 0) > 0:
                        n_flag += 1
                    if args.limit and n_seen >= args.limit:
                        break
        finally:
            qenv.close()
            senv.close()
        pct = 100 * n_flag / n_seen if n_seen else 0.0
        print(f"{vert:<22} records={n_seen:>7} flagged={n_flag:>5} ({pct:.2f}%)", flush=True)

    if not rows:
        print("no records audited", file=sys.stderr)
        return 1

    fields = [
        "vertical", "key", "n_atoms", "n_ncp", "n_bcp", "n_cov_bonds",
        "n_components", "n_isolated_bonded", "isolated_bonded",
        "n_missing_cov_bonds", "missing_cov_bonds", "ncp_matches_atoms", "error",
    ]
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    flagged = [r for r in rows if r.get("n_isolated_bonded", 0) > 0]
    passed_validation = [r for r in flagged if r.get("ncp_matches_atoms") == 1]
    print(f"\n{len(rows)} records -> {args.out_csv}")
    print(
        f"flagged (an atom is covalently bonded but has no BCP): {len(flagged)} "
        f"({100 * len(flagged) / len(rows):.2f}%)"
    )
    print(
        f"  of those, passing the current nuclear-CP validator: "
        f"{len(passed_validation)} - i.e. invisible to validate_qtaim_dict"
    )
    worst = sorted(flagged, key=lambda r: -r.get("n_isolated_bonded", 0))[:10]
    if worst:
        print("\nworst records:")
        for r in worst:
            print(
                f"  {r['vertical']}/{r['key'][:44]:<44} "
                f"atoms={r['n_atoms']} bcp={r['n_bcp']} cov={r['n_cov_bonds']} "
                f"isolated={r['n_isolated_bonded']} [{r['isolated_bonded']}]"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
