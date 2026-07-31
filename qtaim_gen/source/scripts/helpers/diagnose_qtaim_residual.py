"""diagnose-qtaim-residual: per-job forensics on the jobs a repair left behind.

explain-qtaim-shortfall says *which* cause applies. This says *why*, by pulling
the evidence that distinguishes a resource kill from ordinary parse behaviour:

- **truncation point**: whether CPprop.txt stops mid-line / mid-block, which is
  a SIGKILL signature (OOM), versus at a clean block boundary, which is not. On
  the first 28 residuals every file ended cleanly, which ruled out an OOM
  truncation for that set and moved the question to parse/report mismatch.
- **process end state**: whether qtaim.out reaches Multiwfn's completion lines
  or simply stops, plus any error text it did manage to print.
- **memory proxies**: the "Atoms / Basis functions / GTFs" header Multiwfn
  prints. Multiwfn's CP search allocates per-GTF, so GTF count is a better
  memory proxy than atom count. Comparing truncated jobs against jobs that
  completed on the same hardware is what makes the memory hypothesis testable
  rather than plausible.
- **collision detail**: for duplicate_pair, both colliding CPs with their
  densities, since a real second CP being silently overwritten is data loss by
  a different name than a CP that was never storable.
- **counts side by side**: Multiwfn's reported (3,-1) count, the (3,-1) blocks
  actually in CPprop.txt, and the "Connected atoms:" lines. A "truncated"
  verdict against a complete, cleanly-ended file means these three disagree,
  which is a reporting/parse question rather than lost output.

Example:
    diagnose-qtaim-residual --causes_csv qtaim_shortfall_causes.csv \
        --compare_csv qtaim_rerun_test_verified.csv --out_csv residual_forensics.csv
"""

import argparse
import collections
import csv
import json
import os
import re
import sys
import zipfile
from typing import List, Optional

# Multiwfn's own wavefunction summary; GTFs drive CP-search allocations
HEADER_RE = re.compile(
    r"Atoms:\s*(\d+),\s*Basis functions:\s*(\d+),\s*GTFs:\s*(\d+)"
)
BLOCK_RE = re.compile(r"^ -{4,}\s+CP\s+\d+", re.M)
# things Multiwfn or the shell print when a run dies badly
ERROR_SIGNATURES = (
    "Error", "error", "insufficient memory", "Insufficient", "allocat",
    "forrtl", "SIGSEGV", "Killed", "killed", "cannot", "Warning: Unable",
)
COMPLETION_MARKER = "have been outputted to CPprop.txt"


def read_from_zip_or_disk(folder: str, name: str) -> Optional[str]:
    for rel in (name, os.path.join("generator", name)):
        path = os.path.join(folder, rel)
        if os.path.isfile(path) and os.path.getsize(path) > 0:
            with open(path, "r", errors="replace") as f:
                return f.read()
    zip_path = os.path.join(folder, "generator", "out_files.zip")
    if os.path.isfile(zip_path):
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                if name in zf.namelist():
                    return zf.read(name).decode("utf-8", errors="replace")
        except (zipfile.BadZipFile, OSError, KeyError):
            return None
    return None


def zip_inventory(folder: str) -> str:
    zip_path = os.path.join(folder, "generator", "out_files.zip")
    if not os.path.isfile(zip_path):
        return "NO_ZIP"
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            names = zf.namelist()
        return f"{len(names)} entries" + ("" if "CPprop.txt" in names else " NO_CPPROP")
    except (zipfile.BadZipFile, OSError):
        return "BAD_ZIP"


def _fmt_time(value):
    try:
        return f"{float(value):.0f}s"
    except (TypeError, ValueError):
        return "?"


def diagnose(folder: str, cause: str) -> dict:
    from qtaim_gen.source.core.parse_qtaim import get_qtaim_descs, only_atom_cps

    out = {"folder": folder, "cause": cause, "zip": zip_inventory(folder)}

    qout = read_from_zip_or_disk(folder, "qtaim.out")
    # The "Atoms / Basis functions / GTFs" line is printed when Multiwfn *loads*
    # the wavefunction, which happens in the convert step -- qtaim.out usually
    # does not carry it, so search the other step logs too.
    for src in ("qtaim.out", "convert.out", "orca.out"):
        text = qout if src == "qtaim.out" else read_from_zip_or_disk(folder, src)
        if not text:
            continue
        m = HEADER_RE.search(text)
        if m:
            out["n_atoms_mwfn"] = int(m.group(1))
            out["n_basis"] = int(m.group(2))
            out["n_gtf"] = int(m.group(3))
            out["header_from"] = src
            break
    if qout:
        out["qtaim_out_bytes"] = len(qout)
        out["export_marker"] = COMPLETION_MARKER in qout
        tail = qout.rstrip().splitlines()[-1:] or [""]
        out["qtaim_out_last_line"] = tail[0].strip()[:90]
        hits = sorted({sig for sig in ERROR_SIGNATURES if sig in qout})
        out["error_signatures"] = " ".join(hits)[:90]
    else:
        out["qtaim_out_bytes"] = 0

    if qout:
        found = re.findall(r"Number of \(3,-1\) CPs:\s*(\d+)", qout)
        out["reported_bcp"] = int(found[-1]) if found else None
        out["cpprop_is_loose"] = os.path.isfile(os.path.join(folder, "CPprop.txt")) or (
            os.path.isfile(os.path.join(folder, "generator", "CPprop.txt"))
        )

    cpprop = read_from_zip_or_disk(folder, "CPprop.txt")
    if cpprop:
        out["cpprop_bytes"] = len(cpprop)
        out["n_cp_blocks"] = len(BLOCK_RE.findall(cpprop))
        out["n_bcp_blocks"] = len(re.findall(r"Type \(3,-1\)", cpprop))
        out["n_nuclear_blocks"] = len(re.findall(r"Type \(3,-3\)", cpprop))
        out["n_connected_lines"] = len(re.findall(r"Connected atoms:", cpprop))
        # A cleanly finished file ends with a complete final line. Stopping
        # mid-line means the process died while writing -- the SIGKILL/OOM
        # signature we are looking for.
        out["ends_with_newline"] = cpprop.endswith("\n")
        last = cpprop.rstrip().splitlines()[-1:] or [""]
        out["cpprop_last_line"] = last[0].strip()[:90]
        # a complete CP block ends on a property line, not a header
        out["ends_mid_block"] = bool(
            BLOCK_RE.search(cpprop[-400:] if len(cpprop) > 400 else cpprop)
        )
    else:
        out["cpprop_bytes"] = 0

    if cause.startswith("duplicate_pair") and cpprop:
        try:
            tmp = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), ".cpprop_tmp"
            )
            with open(tmp, "w") as f:
                f.write(cpprop)
            _atoms, bonds = only_atom_cps(get_qtaim_descs(tmp))
            os.remove(tmp)
            by_pair = collections.defaultdict(list)
            for v in bonds.values():
                if v.get("connected_bond_paths"):
                    by_pair[tuple(sorted(v["connected_bond_paths"]))].append(
                        round(float(v.get("density_all", 0.0)), 5)
                    )
            collisions = {p: r for p, r in by_pair.items() if len(r) > 1}
            out["collision_detail"] = "; ".join(
                f"{a}-{b}: rho={sorted(rhos, reverse=True)}"
                for (a, b), rhos in list(collisions.items())[:4]
            )[:180]
        except Exception as e:
            out["collision_detail"] = f"{type(e).__name__}: {e}"[:60]

    for base in (folder, os.path.join(folder, "generator")):
        tpath = os.path.join(base, "timings.json")
        if os.path.isfile(tpath):
            try:
                with open(tpath) as f:
                    out["qtaim_time_s"] = json.load(f).get("qtaim")
            except (ValueError, OSError):
                pass
            break
    return out


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--causes_csv", required=True, help="explain-qtaim-shortfall output")
    parser.add_argument(
        "--compare_csv",
        default=None,
        help="verify-qtaim-rerun output; 'fixed' jobs become the control group "
        "for the memory comparison",
    )
    parser.add_argument("--out_csv", default="qtaim_residual_forensics.csv")
    parser.add_argument(
        "--causes",
        nargs="+",
        default=None,
        help="restrict to these causes (default: everything except explained_none)",
    )
    args = parser.parse_args(argv)

    with open(args.causes_csv) as f:
        rows = list(csv.DictReader(f))
    if args.causes:
        rows = [r for r in rows if r.get("cause") in args.causes]
    if not rows:
        print("no rows to diagnose", file=sys.stderr)
        return 2

    print(f"diagnosing {len(rows)} residual jobs\n")
    diags = [diagnose(r["folder"], r.get("cause", "")) for r in rows]

    control = []
    if args.compare_csv:
        with open(args.compare_csv) as f:
            fixed = [r["folder"] for r in csv.DictReader(f) if r.get("verdict") == "fixed"]
        control = [diagnose(f, "fixed_control") for f in fixed]

    fields = [
        "cause", "n_atoms_mwfn", "n_basis", "n_gtf", "header_from", "qtaim_time_s",
        "reported_bcp", "n_bcp_blocks", "n_nuclear_blocks", "n_connected_lines",
        "cpprop_is_loose", "cpprop_bytes", "n_cp_blocks", "ends_with_newline",
        "ends_mid_block",
        "cpprop_last_line", "qtaim_out_bytes", "export_marker",
        "qtaim_out_last_line", "error_signatures", "zip", "collision_detail",
        "folder",
    ]
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(diags + control)

    for cause in sorted({d["cause"] for d in diags}):
        sub = [d for d in diags if d["cause"] == cause]
        print(f"== {cause}  (n={len(sub)})")
        for d in sub:
            bits = [
                f"atoms={d.get('n_atoms_mwfn','?')}",
                f"GTFs={d.get('n_gtf','?')}",
                f"t={_fmt_time(d.get('qtaim_time_s'))}",
                f"reported={d.get('reported_bcp','?')}",
                f"bcp_blocks={d.get('n_bcp_blocks','?')}",
                f"connected_lines={d.get('n_connected_lines','?')}",
                f"export={d.get('export_marker','?')}",
            ]
            print("   " + "  ".join(str(b) for b in bits))
            if d.get("cpprop_bytes"):
                print(
                    f"     CPprop ends: newline={d.get('ends_with_newline')} "
                    f"mid_block={d.get('ends_mid_block')} | "
                    f"{d.get('cpprop_last_line','')[:70]}"
                )
            if d.get("error_signatures"):
                print(f"     errors seen: {d['error_signatures']}")
            if d.get("collision_detail"):
                print(f"     collisions: {d['collision_detail']}")
            if d.get("zip") and "NO_" in str(d["zip"]) or d.get("zip") == "BAD_ZIP":
                print(f"     zip: {d['zip']}")
            print(f"     {os.path.basename(d['folder'])}")
        print()

    # the memory question: are truncated jobs bigger than jobs that completed?
    trunc = [d for d in diags if d["cause"] == "truncated" and d.get("n_gtf")]
    ctrl = [d for d in control if d.get("n_gtf")]
    if trunc and ctrl:
        import statistics as st

        print("== memory hypothesis: truncated vs jobs that completed")
        for label, key in (("GTFs", "n_gtf"), ("basis fns", "n_basis"), ("atoms", "n_atoms_mwfn")):
            t = [d[key] for d in trunc if d.get(key)]
            c = [d[key] for d in ctrl if d.get(key)]
            if t and c:
                print(
                    f"   {label:<10} truncated median={st.median(t):>10.0f} "
                    f"max={max(t):>10.0f} | completed median={st.median(c):>10.0f} "
                    f"max={max(c):>10.0f}"
                )
        over = [d for d in ctrl if d["n_gtf"] >= max(x["n_gtf"] for x in trunc)]
        print(
            f"   {len(over)} completed job(s) are at least as large (by GTFs) as the "
            f"largest truncated one"
        )
        if over:
            print(
                "   -> size alone does not determine truncation, so a hard memory\n"
                "      ceiling is not the whole story; concurrency at the time of the\n"
                "      run would be the next thing to check."
            )
        else:
            print(
                "   -> every truncated job is larger than every completed one, which is\n"
                "      what a memory ceiling looks like."
            )
    elif trunc:
        print("== memory hypothesis: pass --compare_csv to get a control group")

    with_file = [d for d in diags if d.get("cpprop_bytes")]
    mid = [
        d for d in with_file
        if d.get("ends_mid_block") or d.get("ends_with_newline") is False
    ]
    if with_file:
        print(
            f"\n  CPprop.txt write integrity: {len(mid)} of {len(with_file)} stop "
            "mid-line or mid-block."
        )
        if not mid:
            print(
                "  Every file ends cleanly, so none of these was killed while writing.\n"
                "  That rules out an OOM/SIGKILL truncation for this set -- a count\n"
                "  shortfall against a complete, cleanly-ended file is a parse or\n"
                "  reporting mismatch, not lost output."
            )
    no_file = [d for d in diags if not d.get("cpprop_bytes")]
    if no_file:
        print(
            f"\n  {len(no_file)} job(s) produced no CPprop.txt at all"
            + (
                f" (qtaim ran {_fmt_time(no_file[0].get('qtaim_time_s'))}+)"
                if no_file[0].get("qtaim_time_s")
                else ""
            )
            + ".\n  With no export marker either, these are the only genuine "
            "killed-run candidates here."
        )
    print(f"\n  -> {args.out_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
