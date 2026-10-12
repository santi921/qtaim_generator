# LIBRETA ESP for the QTAIM step (parked)

Status, 2026-10-12: parked on branch `feat/qtaim-libreta-esp`, flag `--libreta_esp`
off by default. Not used for the OMol4M dataset: every existing vertical has eleesp1
ESP, and LIBRETA's values differ from eleesp1 by more than print precision, so turning
it on would mix two ESP codes in one dataset.

## Summary

- Production's CPprop.txt ESP (esp_e, esp_total) comes from Multiwfn's old `eleesp1`
  on every cluster, whatever `iESPcode` says, because nothing on the topology path
  initializes LIBRETA. A 7-line stdin prefix makes Multiwfn use LIBRETA, with the CP
  set and every non-ESP field unchanged.
- LIBRETA (fast, `iESPcode= 2`) is about 1.8x faster at 1.5-3k GTFs and 2.7x at 7.4k
  GTFs on the default variant (1.1-1.2x exhaustive), but esp_e is off from exact by up
  to 4.8e-5 relative (1.3e-3 a.u.), against 6.7e-9 for eleesp1.
- Its init briefly needs about 3.9e-7 x GTF^2 GB (20 GB at 7.4k GTFs). With 92
  single-core jobs on a 512 GB Tuolumne node, synchronized starts would run the node
  out of memory above about 3,400 GTFs.

## Why production never uses LIBRETA

- `eleesp` falls back to `eleesp1` whenever `if_initlibreta == 0`
  (Multiwfn 3.8 `function.f90:4269-4280`).
- `doinitlibreta` is called only from specific menus; in topology only from `vmin`
  and from `-11` when the selected function involves ESP (`topology.f90:105-127`).
  Option `7 0` (`showptprop`) just calls `eleesp`.
- Census of production qtaim.out files on Tuolumne and ALCF: 0 of 529 sampled runs
  initialized LIBRETA. Builds seen: 3.8(dev) 2024-Oct-6, 2025-Jun-26,
  2025-Dec-8. The 2026.10.1 source has the same code paths.

## The flag

`--libreta_esp` (full-runner-parsl, full-runner-parsl-alcf) prepends
`LIBRETA_ESP_PREFIX` (`data/multiwfn.py`) to both QTAIM stdins:

```
4 12 1 2,2 1 0 -5        2x2 ESP plane map, only to call doinitlibreta, then back to the main menu
2 2 3 4 5 8 7 0 -10 q    unchanged production stdin (exhaustive: 2 2 3 4 5 6 -1 -9 8 7 0 -10 q)
```

- The plane map runs before topology, so every topology setting stays at its default.
  The topology `-11` route was rejected: it tightens gradconv/dispconv to 1e-7/1e-8.
- `initlibreta` snapshots the primitive density matrix; topology's `delvirorb` only
  hides unoccupied orbitals, so the snapshot stays valid.
- A first version used a 500-point far-field line plot
  (`3 12 2 1000,1000,1000,1000,1000,1000.1 0`); it costs about 25 s on 7k-GTF jobs,
  the plane map 4 points. The verification and timings below used the line plot. The
  plane map gave a CPprop.txt byte-identical to it on rmechdb_264_step6_0_2, and ran
  cleanly on the 6 jobs of the slow-version test; rerun the 87-job check with it
  before relying on it.
- `qtaim_esp_code(qtaim_out_text)` (`utils/validation.py`) returns `eleesp1`,
  `libreta_fast` or `libreta_slow` from Multiwfn's own init line. It is meant for the
  qtaim.json `_meta` block of `fix/cpprop-integrity` (PR #53) and is not written
  anywhere yet.
- The restart gate and `--check_bcp_count` read the prefixed qtaim.out exactly as the
  production one (same run status, BCP count and banner; `tests/test_libreta_esp.py`).

## Verification: nothing but the ESP changes

Local Multiwfn 3.8(dev) 2024-Oct-6, 4 threads. References: the 87 jobs of
`data/cross_validation_wfns/wfx_pull` (25 with EDF), plus two 227-260 atom jobs of
`data/edge_cases/0831`.

| set | variant | jobs | CPs | CP count/type mismatch | max d(grad norm), prod vs libreta (a.u.) | same, prod vs prod rerun (a.u.) |
|---|---|---|---|---|---|---|
| wfx_pull | default | 87 | 3,059 | 0 | 2.06e-14 | 2.06e-14 |
| wfx_pull | exhaustive | 87 | 3,071 | 0 | 2.06e-14 | 2.06e-14 |
| edge 1930 (260 atoms, 7,432 GTFs) | default | 1 | 657 | 0 | 6.7e-16 | - |
| edge 3467 (227 atoms, 7,411 GTFs) | default | 1 | 699 | 0 | 1.7e-15 | - |

- Positions, rho, Laplacian, Hessian, ELF, LOL, G, K and the rest are byte-identical.
  The only other differences are gradient-limited values at roundoff (IRI, RDG,
  gradient norm) and spin density in its last printed digit; production differs from
  its own rerun by the same amount.
- qtaim.out from the topology menu on is identical except the "Searching CPs took up"
  timing lines.
- Not verified: the 391 edge case (11,379 GTFs, cut for memory); the exhaustive variant
  on edge cases; the 2025-Jun-26 and 2025-Dec-8 builds (same source paths only).

## Accuracy

Exact reference: gbasis `electrostatic_potential` (horton env) at the CP positions,
electronic part only, MO density without the EDF (neither code includes it).
`scripts/libreta_esp/esp_ref.py`. Default variant, 87 jobs.

esp_e against exact:

| subset | CPs | eleesp1 rel (median / max) | libreta rel (median / max) | libreta abs, a.u. (median / max) |
|---|---|---|---|---|
| all | 3,059 | 4.8e-9 / 6.7e-9 | 1.3e-7 / 4.8e-5 | 3.2e-6 / 1.3e-3 |
| (3,-3) | 1,355 | 3.4e-9 / 6.0e-9 | 9.9e-8 / 4.8e-5 | 3.5e-6 / 1.3e-3 |
| (3,-1) | 1,475 | 5.1e-9 / 6.7e-9 | 1.4e-7 / 3.6e-5 | 2.8e-6 / 1.2e-3 |
| (3,+1) | 219 | 5.8e-9 / 6.3e-9 | 3.2e-7 / 2.8e-5 | 4.1e-6 / 1.1e-3 |
| (3,+3) | 10 | 6.0e-9 / 6.1e-9 | 1.1e-6 / 1.2e-5 | 5.5e-5 / 6.0e-4 |
| EDF jobs | 768 | 4.8e-9 / 6.2e-9 | 2.1e-6 / 3.4e-5 | 6.7e-5 / 1.3e-3 |
| no-EDF jobs | 2,291 | 4.8e-9 / 6.7e-9 | 3.3e-8 / 4.8e-5 | 1.5e-6 / 9.1e-4 |

- LIBRETA is closer to exact than eleesp1 on 34 of 3,059 CPs, and more negative than
  eleesp1 on 69% of them (biased, not noise).
- esp_total: both codes share esp_nuc, so its absolute error equals esp_e's. On the
  1,704 off-nucleus CPs (|total| a median 9.5% of |esp_nuc|, down to 7.85e-5 a.u.)
  the total's relative error vs exact is, median / p99 / max: eleesp1 2.1e-7 / 5.6e-5 /
  2.5e-3, libreta 1.0e-5 / 1.6e-2 / 0.72.
- At nuclear (3,-3) CPs the printed "Total ESP" (1e8-1e13 a.u.) has 10 significant
  digits; compare components, not that line.

## Where LIBRETA's error comes from

1. Screening (fast version only): shell pairs with max |P| <= 1e-7 are skipped,
   `ZeroTol` in `libreta_hybrid/ean.f90:116,127`. A compile-time parameter, not a
   settings.ini key, in 3.8 and 2026.10.1. This is the 1e-3 a.u. part.
2. A Multiwfn bug: `ean.f90:71` declares `real*8,parameter :: PI2 = 6.28318530717958647693`
   without a `d0` suffix, so the literal is single precision. Every LIBRETA esp_e
   (fast and slow) is scaled by float32(2 pi)/(2 pi) = 1 + 2.78e-8. Present in
   2026.10.1; no promotion flag in either Makefile. Worth reporting upstream.
3. Boys function: LIBRETA's standard `boys_libreta` unless `iaccurateESP=1`, which only
   topology's `vmin` and `-11` set. Small next to 1 and 2.

The high-precision Boys alone does not help the fast version: on the 6 worst jobs the
max error is the same to 2 digits (screening dominates).

## Exact-LIBRETA recipe (tested on 6 jobs only)

The slow version (`iESPcode= 3`, `eleesp2_slow`) does not screen. With

- `iESPcode= 3` for the QTAIM step only (other steps that use ESP must keep 2, so the
  runner would need a per-step settings.ini),
- stdin prefix `2 -11 12 -11 1 -1 3 1E-6 4 1E-7 0 -3 1 40 3 0.02 0` before
  `2 3 4 5 8 7 0 -10 q`: `-11` turns on the high-precision Boys, the `-1` and `-3`
  menus put back the four parameters `-11` changes (gradconv, dispconv,
  nsurfpathpercp, surfpathstpsiz), the others it sets already equal the defaults,
- esp_e divided by float32(2 pi)/(2 pi) when parsing, esp_total = esp_nuc + esp_e,

esp_e matched eleesp1 to a median 9.5e-11 and max 8.5e-10 relative on the 6 worst
jobs (342 CPs), which is print precision; CP sets and non-ESP fields unchanged.
Not measured: speed and memory at large sizes. The slow pair loop is single-threaded,
and the per-pair array (about 3.6 KB per primitive shell pair) is kept for the whole
run instead of freed after init.

## Speed and memory (fast version)

Contended runs (several jobs at once), 500-point line prefix, 4 threads:

| size | jobs | median speedup, default | median speedup, exhaustive |
|---|---|---|---|
| < 700 GTFs | 45 | 0.92-0.94x | 0.99x |
| 700-1,500 GTFs | 36 | 1.35-1.40x | 1.06x |
| 1,500-3,000 GTFs | 6 | 1.83-1.95x | 1.17x |
| 7,411 GTFs (3467) | 1 | 2.65x (1:06:00 -> 24:53) | - |
| 7,432 GTFs (1930) | 1 | 2.72x (53:41 -> 19:46) | - |

- Peak RSS is a transient spike in `formblock` (a temporary record per primitive shell
  pair, sized for MaxL=5), under 10 s: 1.8 GB at 2.5k GTFs, 18.5-20.2 GB at 7.4k. The
  steady state after it is about 2 GB at 7.4k, against 0.18 GB for eleesp1.
- `scripts/libreta_esp/gtf_census.py` reads GTF counts from existing qtaim.out headers
  (`Atoms: N, GTFs: P`) or estimates 1.61 x ORCA basis functions. Tuolumne, 200 jobs per
  vertical: median 3,632 GTFs (lustre5), 2,373 (vast1); 47.9% and 68.8% at or under
  3,500 GTFs.

## Surface ESP (other_esp, full_set > 1)

Main function 12 initializes LIBRETA itself, so with the shipped `iESPcode= 2` its
surface statistics come from the fast version. Against eleesp1 (`iESPcode= 1`) on 11
jobs: ESP extrema up to 0.49 kcal/mol, positive/negative surface areas up to 1.3%,
polar/nonpolar areas up to 2.3%, the overall average up to 24% relative (it is near
zero). The slow version brings every field 100-1000x closer. No dataset so far has
full_set > 1.
