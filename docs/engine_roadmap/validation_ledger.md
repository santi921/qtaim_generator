# Engine validation ledger

Every validation run of the native engines (charge, surface, QTAIM) against Multiwfn, HORTON,
gbasis, critic2, extended precision or physical invariants. One row per run, with the
reference, the data, the result and where the raw output lives.

State as of 2026-11-03, branch `feat/charge-engine` (cfac9a3), plus `feat/engine-fuzzy-elf` and `feat/engine-esp-surface`. Companion documents:
`README.md` (decisions, divergence changelog), `qtaim_verification_campaign.md` (QTAIM
tiers 1-3), `mbis_converged.md` (MBIS).

## How to read this

- **Reference kinds.**
  - *Multiwfn parity*: the engine reproduces Multiwfn 3.8 on the same wavefunction. The
    bar is Multiwfn's print precision, since Multiwfn's output is text.
  - *Independent code*: HORTON 3 (gbasis, horton-part), critic2. Checks the kernels or
    the method, not the parity.
  - *Extended precision*: an 80-bit long-double evaluation, used to decide which code is
    right where two disagree.
  - *Invariants*: physical checks that need no reference (charge and spin sums, electron
    count on the grid, Poincare-Hopf, an NCP on every nucleus).
- **Engine version.** Results are only as current as the code they ran on. The column
  "code" gives the branch or commit where it was recorded. Section 7 lists which current
  behaviours have no validation on current code.
- **Numbers** are the largest per-atom (or per-point) absolute difference unless marked
  relative ("rel"). Charges and populations are in e, positions in Bohr, planarity in
  Angstrom.
- "Same as Multiwfn" in this document always means "within the precision Multiwfn
  prints".

## 1. Summary

- **Charge and fuzzy routines** (full_set 0 and 1) match Multiwfn at print precision on
  every set run so far: 6 to 419 jobs per set, up to 342 atoms, ECP with EDF, lanthanides,
  actinides (X2C and ECP), open shells up to multiplicity 11. The largest charge
  difference seen anywhere is 1.12e-6 e (ADC-Becke, LRC full_set 1). The ELF, gradient
  norm and Laplacian fuzzy integrals (full_set 1 and 2) match at print precision too, apart
  from Laplacian values on ECP heavy atoms, which agree to 2.3e-11 of their integrand scale
  (local sets only; no cluster run yet).
- **QTAIM kernels** agree with gbasis to a median 7.5e-14 rel (rho) on 100
  wavefunctions; where they differ more, 80-bit evaluation sides with the engine. The CP
  search reproduces Multiwfn's CP sets exactly in its Multiwfn-compatible mode. In engine
  mode it finds every CP Multiwfn finds (0 missing on 1498 LRC jobs and 419 actinide
  jobs) plus CPs Multiwfn misses.
- **Gaps:** converged MBIS (the current default) has no independent reference; the ALIE
  difference on the NERSC actinides (3.4e-5 rel) is unexplained; engine-only CPs are not
  yet confirmed by critic2; no run covers double hybrids, HF, Cartesian basis sets or a
  systematic ORCA 5 vs 6 comparison.

## 2. Charge engine (`core/charge_engine.py`)

Routines: hirshfeld, adch, cm5, becke (Multiwfn's ADC-corrected Becke), hirsh and becke
fuzzy density and spin, fuzzy_bond (full_set 0); vdd, mbis, mbis fuzzy density and spin,
elf_fuzzy (full_set 1); grad_norm_rho_fuzzy, laplacian_rho_fuzzy (full_set 2).

| date | reference | data | result | code | raw output |
|---|---|---|---|---|---|
| 2026-10-04 | Multiwfn parity, local | 6 jobs, 12-178 atoms (ECP, lanthanide, open shell) | all 8 full_set 0 schemes within 5.8e-9 e | first engine (one-pass) | `tests/test_files/charge_engine/` fixtures |
| 2026-10-04 | Multiwfn parity, local | 87 wfx_pull jobs (`data/cross_validation_wfns/wfx_pull`): 7 OMol verticals, 61 elements, 25 ECP, 43 open shell, mult 1-11, charge -5 to +4, ORCA 6.0.0 | all 9 schemes pass; charges within 7.9e-8 e, fuzzy_bond within 5e-7 | fuzzy_bond added | session scratchpad (not kept) |
| 2026-10-06 | Multiwfn parity, local | 18 references (vdd, mbis, mbis fuzzy; ECP+EDF, lanthanides, mult up to 11) | print precision | 0ae37d9..df2db4e | `tests/test_files/charge_engine/` |
| 2026-10-06 | Multiwfn parity, LRC | tm_react, 194 jobs, engine from scratch vs fresh Multiwfn on the same node | all 9 schemes pass; charges within 1.1e-6 e, fuzzy_bond 5e-7 | feat/charge-engine | LRC `engine_test` tree (runbook `data/engine_test_2026-10/`) |
| 2026-10-06 | stored Multiwfn data, LRC | tm_react, 1853 folders merged into stored results | all 9 schemes pass on clean records; non-engine data changed in 5 | feat/charge-engine | same |
| 2026-10-12 | Multiwfn parity, LRC, full_set 1 | tm_react, 191 of 200 jobs (113 open shell; 9 lost to preemption stale locks) | 13 schemes + 8 ALIE fields pass; max 5.03e-9 e except adch 4.73e-7, becke 1.12e-6, fuzzy_bond 5e-7; non-engine data changed in 0 | 568d6ad | LRC trees `ref_l1`, `scratch_l1c` |
| 2026-10-13 | Multiwfn parity, local | 156 actinide nitrates (`data/actinide_dev/nitrates_dev`): Th, U, Np, Pu; 8 functionals x X2C-SVP/TZVP/TZVPPall + wB97M-V def2-SVP/TZVPD (ECP); ORCA 6.1 | charges within 1.3e-8 e (Pu ADCH/Becke), fuzzy_bond 5e-7, full_set 1 | fix/engine-mixed-records | `data/actinide_dev/work/{multiwfn,engine}` |
| 2026-10-25 | Multiwfn parity, NERSC | openac: the 419 jobs with a Multiwfn-mode run (`mwfn_l0`, 440-job sample) | all charge and fuzzy schemes within 7.4e-8 e (median 5e-9), fuzzy_bond 5e-7 | feat/charge-engine, PRs #63-#68 | NERSC `actinide_test/prod_l1_audit.tsv` (`audit_prod.py --ref`) |
| 2026-11-02 | Multiwfn parity, local, fresh Multiwfn | 2 fixtures (open shell; Xe ECP + EDF): elf_fuzzy, grad_norm_rho_fuzzy, laplacian_rho_fuzzy | ELF 4.8e-9, gradient norm 4.8e-9, Laplacian 2.2e-8 (Xe, value -114.9) | feat/engine-fuzzy-elf | `tests/test_files/charge_engine/*/multiwfn_reference_level{1,2}.json` |
| 2026-11-02 | stored Multiwfn data (production `elf_fuzzy.out`), local | 156 actinide nitrates (as the 2026-10-13 row; open shells spin-polarized) | elf_fuzzy within 5.0e-9 on every atom of all 156 (Multiwfn prints 8 decimals); engine derivative pass median 14.4 s (4 threads) vs stored Multiwfn elf_fuzzy median 39.0 s | same | `data/engine_test_2026-11/fuzzy_elf/nitrates_elf.jsonl` |
| 2026-11-02 | Multiwfn parity, local, fresh Multiwfn | 87 wfx_pull jobs (as the 2026-10-04 row) | elf_fuzzy within 5.0e-9 and grad_norm_rho_fuzzy within 5.3e-9 on all 87. laplacian_rho_fuzzy: 63 jobs within 1e-8; 24 jobs exceed it, every one on an ECP + EDF heavy atom (Xe, I, Te, Sb, Hf, Tl, Au, Ir, In, lanthanides; max 1.08e-6 on Hf, value 25712.9, 4.2e-11 rel). All 24 are within 2.3e-11 of the atom's integral of abs(integrand); summation order alone moves the result by about 1e-11. Read as per-point roundoff of the Laplacian near the core [I], not checked against 80-bit | same | `data/engine_test_2026-11/fuzzy_elf/` (`wfx_pull_fuzzy.txt`, `lapl_rel.py`, `refs_wfx_pull/`) |
| 2026-11-03 | internal consistency | 2 fixtures, 400 random points each | `density_derivatives(elf_only=True)` (the full_set 1 ELF pass) gives values identical bit for bit to the full call, and `hessian=False` agrees within 1e-12 rel; elf_only takes 0.34 s vs 0.59 s per atom grid on MOR22 (tm_react) and 0.73 s vs 1.05 s on a TZVPPall Np nitrate (4 threads), so the 14.4 s nitrate median above is now lower | feat/engine-fuzzy-elf (review fixes) | `tests/test_qtaim_engine.py` |
| 2026-11-03 | invariant scan | 552 local orca.wfx (85 with ECP) | the per-ECP-atom EDF rule (charge engine version 2) decides every skip as the molecule-wide rule did | same | `data/engine_test_2026-11/fuzzy_elf/edf_scan.py` |

`scripts/compare_engine_merge.py` takes the routines a tree's engine covers from its
version stamp. From charge engine version 2 its Multiwfn baseline time
(`mwfn_engine_routines_s`) includes elf_fuzzy and the level-2 fuzzy timings, so speedups
from it are not comparable with the 2026-10 rows above.

**MBIS** (`mbis_converged.md`, 2026-10-15). The default (`MBIS_MODE = "converged"`)
deliberately diverges from Multiwfn, which stops when no charge moves by 1e-4 e in a cycle.
`--mbis_mode multiwfn` keeps the parity mode, which the rows above test.

| check | data | result |
|---|---|---|
| converged fit vs plain iteration far past Multiwfn's rule | 35 molecules (27 cross-validation, 5 nitrates, 3 clusters) | within 2.5e-6 e on jobs that do not stall |
| Multiwfn's rule vs converged, Z <= 86 | 27 jobs | 3 off by more than 0.03 e, 7 by more than 0.005 e |
| Multiwfn's rule vs converged, actinides | 5 jobs | 0.03-0.38 e |
| fitted guess vs Multiwfn's guess | 35 jobs | 32 within 0.005 e; the 3 that differ have the better objective |
| selection revision (2026-11-03, perf/mbis-converged): check guesses skip the fallback when the default converged, unconverged check fits excluded; old selection reconstructed on the same grid | 112 jobs (87 wfx_pull, 20 nitrates, 5 edge cases), scratchpad `mbis/old_vs_new.py` | 100 the same (42 by construction), max 0 e; 12 change selection, all from an unconverged x1.5 fit, max 3.6e-4 e (MOR39_Er3Ce3), rest 1.8e-5 e or less; 70 compared jobs 15312 s to 9556 s, Sm3Ir3 1366 s to 93 s |
| shell merging in the plain fallback (2026-11-03, perf/mbis-converged) vs the selection revision above, same grid and thread environment | same 112 jobs, scratchpad `mbis/val_merge.py` | 97 jobs whose default fit converged identical (0 e); the 15 whose default failed: all fits converge, selection unchanged, max 4.4e-5 e (Ru, previously unconverged), rest 4.0e-6 e or less, objective equal or better; 15 jobs 8218 s to 1429 s, all 112 9759 s to 2906 s; BLAS thread count alone moves accelerated fits by up to 2.6e-7 e |

## 3. Surface engine (`core/surface_engine.py`, other_alie, other_geometry and other_esp)

**Planarity** (other_geometry: MPP and SDP for all atoms and for heavy atoms, Angstrom).

| date | reference | data | result | code | raw output |
|---|---|---|---|---|---|
| 2026-11-02 | Multiwfn parity, local, wfx nuclei | 353 production `other_geometry.out` with their orca.wfx (mostly actinide nitrates, joe_collab, edge cases) | all four values within 5.0e-7 (Multiwfn prints f12.6) | feat/engine-planarity | scratchpad `mpp_validate.py` |
| 2026-11-02 | stored Multiwfn data, local, orca.inp geometry | 2673 `other.json` (OMol rmechdb, tm_react audit folders, edge cases) | MPP within 5.1e-7 everywhere; SDP over 5.1e-7 in 26 (full) and 18 (heavy) of 2673, max 3.7e-6 on two jobs whose two smallest singular values differ by 5e-5 relative: orca.inp carries fewer digits than the wfx Multiwfn read, and near a tie that moves the plane | same | same |
| 2026-11-02 | local Multiwfn on test geometries | H2, H2O, CH4, SF6 | match, including CH4 and SF6, whose tied singular values make the plane arbitrary; H2 heavy-atom MPP is 0 where Multiwfn prints NaN (divergence) | same | `tests/test_files/charge_engine/planarity_reference.json` |

Tied singular values: the engine takes numpy's SVD vector (LAPACK gesdd), Multiwfn calls
dgesvd; both gave Multiwfn's SDP on every local case. The choice is reproducible for the
same input and LAPACK build, but not unique: rotating CH4 or adding 1e-10 Angstrom of
noise moves its SDP between 1.26 and 1.78 Angstrom (SF6 1.86-3.12). MPP is unaffected.
No local molecule outside these test geometries has a non-trivial exact tie.

**ALIE** (other_alie).

| date | reference | data | result | code | raw output |
|---|---|---|---|---|---|
| 2026-10-06 | Multiwfn parity, local | 16 references (incl. ECP+EDF, lanthanides) | identical mesh (vertex, edge and face counts) and all ALIE statistics at print precision | 0ae37d9..df2db4e | `tests/test_files/charge_engine/` |
| 2026-10-12 | Multiwfn parity, LRC | tm_react 191 jobs (full_set 1 run above) | all 8 ALIE fields pass | 568d6ad | LRC `ref_l1`, `scratch_l1c` |
| 2026-10-13 | Multiwfn parity, local | 156 actinide nitrates | pass | fix/engine-mixed-records | `data/actinide_dev/work` |
| 2026-10-25 | Multiwfn parity, NERSC | openac 419 | ALIE values median 1.2e-5, max 3.4e-5 rel | PRs #63-#68 | `prod_l1_audit.tsv` |

The NERSC difference is larger than the local and LRC runs. Its cause (print rounding of
the statistics, the mesh, or the Multiwfn build) has not been checked.

| date | reference | data | result | code | raw output |
|---|---|---|---|---|---|
| 2026-11-03 | refactor guard (shared mesh builder for ALIE and ESP) | 2 fixtures + 10 wfx_pull jobs | ALIE values and mesh counts bit-identical before and after | feat/engine-esp-surface | `data/engine_test_2026-11/esp_surface/alie_baseline.json`, `alie_check.py` |

**ESP surface** (other_esp, full_set 2; 20 parsed values: extremes in kcal/mol, areas in
Bohr^2, averages, variances and Pi in a.u., polarity index in eV, polar and nonpolar areas in
Angstrom^2).

| date | reference | data | result | code | raw output |
|---|---|---|---|---|---|
| 2026-11-03 | Multiwfn parity, exact ESP (iESPcode=1, eleesp1; engine with Multiwfn's Fmch Boys function) | 2 fixtures (open shell; Xe ECP + EDF) | identical mesh (counts before and after elimination); all 20 values within print precision | feat/engine-esp-surface | `tests/test_files/charge_engine/*/multiwfn_reference_esp.json` |
| 2026-11-03 | production Multiwfn (libreta, iESPcode=2) vs engine (exact ESP) | 99 jobs: 87 wfx_pull (7 OMol verticals, 25 ECP) + 12 actinide nitrates (Th, U, Np, Pu; X2C and ECP) | mesh identical in 99/99; volume, total area, density within print precision. ESP fields (median / p90 / max): extremes 2.2e-3 / 7.9e-2 / 0.47 kcal/mol (min) and 3.8e-3 / 9.9e-2 / 0.49 (max), average 4.1e-6 / 1.3e-4 / 7.4e-4 a.u., positive area 2.4e-6 / 0.33 / 3.7 Bohr^2, polarity index 7.0e-5 / 2.7e-3 / 2.0e-2 eV, skewness 5.0e-5 / 2.8e-3 / 4.0e-2; nitrates alone at most 0.048 kcal/mol. A divergence by design (README changelog); attributed to libreta [I], with exact-code parity shown on the 2 fixtures only (the 27-job eleesp1 run was stopped). Engine median 2.8x faster than Multiwfn (0.97-12.1x; nitrates 9.8x) | same | `data/engine_test_2026-11/esp_surface/` (`esp_vs_libreta.jsonl`, `refs_esp_libreta/`) |

## 4. QTAIM engine (`core/qtaim_engine.py`, `cp_search.py`, `esp.py`)

### 4a. Point properties and kernels

| date | reference | data | result | raw output |
|---|---|---|---|---|
| 2026-10-08 | Multiwfn parity at Multiwfn's CP positions | 5 jobs, all four CP types | every property within 4.8e-10 rel (ellipticity, eta 5e-7) | `scripts/qtaim_reference.py` |
| 2026-10-09 | gbasis (HORTON), full precision | 93 local wavefunctions, 2-274 atoms, 25 with EDF, about 140,000 points | median rel: rho 7.5e-14, gradient 3.4e-13, Hessian 1.7e-12, Laplacian 2.8e-12, G 4.0e-13 | `scripts/qtaim_horton_check.py` |
| 2026-10-09 | finite differences | EDF terms | gradient and Hessian within 2.7e-7 (step-size limit) | same |
| 2026-10-09 | 80-bit extended precision | the one outlier (17-atom Hf ECP, rho at 0.07-0.16 Bohr from Hf) | engine within 7.5e-14 to 1e-12 of 80-bit; gbasis off by 1.2e-7 to 3.2e-7 | same |
| 2026-10-10/12 | gbasis + 80-bit | 7 jobs, 147-342 atoms | worst point rel: rho 1.8e-12, Laplacian 6.3e-11; 80-bit shows the excess is gbasis roundoff (near-zero Laplacian) or the shared exp(-40) screening (far field) | same |
| 2026-10-12 | gbasis | 5 wavefunction types: ROKS, closed shell, UKS, closed shell + EDF, UKS + EDF | ELF 1.9e-12, LOL 9.5e-13, ALIE 5.6e-13, spin 8.3e-10 rel; ROKS alpha/beta 1.8e-15 | same |
| 2026-10-09/12 | stored Multiwfn CPprop.txt, LRC tier 2 | 1500 sampled jobs (5A_elytes, rgd_uks, tm_react) | 1498 ok, 2 corrupt references; 3 BCP Laplacians at 3.2e-8 to 1.5e-7 rel (position rounding suspected, unconfirmed); 15-job control with fresh Multiwfn: identical | `data/engine_test_2026-10/qverify_results` (qverify_v2) |
| 2026-10-12 | Multiwfn parity (ESP at CPs, Multiwfn's Fmch Boys function) | 97 jobs | esp_e within 4.9e-10 | feat/qtaim-esp |
| 2026-10-12 | gbasis (ESP, exact Boys) | 26 jobs | within 8.9e-14 | same |
| 2026-10-12 | Multiwfn (ESP, exact Boys = production default) | 97 jobs | 3.8e-9 to 7.4e-9 rel: Multiwfn's Fmch, a recorded divergence | same |

### 4b. CP search, bond paths, topology

| date | reference | data | result | raw output |
|---|---|---|---|---|
| 2026-10-12 | Multiwfn parity (Multiwfn-compatible seeds) | 88 local jobs + 9 cost-ladder jobs (12-342 atoms) | identical CP sets, positions within 8.4e-13 Bohr, equal seed counts | feat/qtaim-cp-search |
| 2026-10-12 | Multiwfn parity (all-Multiwfn modes through `parse_qtaim`) | 87 jobs | identical qtaim.json | same |
| 2026-10-12 | Multiwfn parity (bond paths) | Multiwfn mode | 1485 of 1486 pairs equal | same |
| 2026-10-12 | Poincare-Hopf repair (reseeding, type-aware dedup) | 5 Multiwfn failures (`data/engine_test_2026-10/ph_fail`) + 88 local jobs | all 5 repaired; nothing Multiwfn finds is lost; Poincare-Hopf 1 on all 88 | qtaim_engine_plan.md 6d |
| 2026-10-13/16 | heavy nuclei far from the origin (nucleus frame) | 9 local X2C U/Th jobs; LRC UCl6 (U 26 Bohr from origin) | NCP found in all (Multiwfn lost it in 5 of 9 and in UCl6); density and Laplacian unchanged to 4e-16 rel at shifts up to 100 Bohr | README.md divergence changelog |
| 2026-10-13 | Multiwfn parity | 156 actinide nitrates (actinide at the origin) | identical CP sets, Poincare-Hopf 1, properties at print precision | `data/actinide_dev/work` |
| 2026-10-22 | stored Multiwfn data, LRC (qeng2) | 1498 jobs | Poincare-Hopf (kept set) 1 in 1494 vs 1455 for stored Multiwfn; 0 missing non-weak CPs; every CP-set difference is an extra engine CP on a job where stored Multiwfn fails Poincare-Hopf; properties within 2e-9 rel except esp_total (4.6e-2 rel, likely cancellation, unchecked) | `data/engine_test_2026-10/qverify_results/qeng2_*.jsonl` |
| 2026-10-25 | Multiwfn parity, NERSC | openac 419 | identical atom and bond sets in 327; engine-only bonds in 92, Multiwfn-only in 0; 74 of the 92 involve the actinide, 68 of them far from the origin with Multiwfn Poincare-Hopf failing | `prod_l1_audit.tsv` |

## 4b. Wavefunction converter (`core/wavefunction.py`, `--native_convert`)

Reads the molden orca_2mkl writes and writes orca.wfx without Multiwfn. `multiwfn_h` mode
reproduces Multiwfn's alpha-only sign flip of ORCA h functions; `fixed` flips both spins, as
ORCA's own orca_plot requires (README.md divergence changelog). Since 2026-11-03 `fixed` is
the runner's default; `--native_convert off` runs Multiwfn's convert step instead.

| date | reference | data | result | raw output |
|---|---|---|---|---|
| 2026-11-02 | Multiwfn's orca.wfx (read_wfx dict) | 87 wfx_pull (7 OMol verticals, ORCA 6.0.0, 25 ECP) | all fields equal; coefficients within 4.9e-12 of the largest per MO, density within 2.9e-11 rel (E20.12 floor) | scratchpad `conv/val_reader.py` |
| 2026-11-02 | same, compat mode | 156 actinide nitrates (ORCA 6.1, X2C and ECP) | coefficients 5.0e-12, density 5.7e-12 rel | same |
| 2026-11-02 | same, ORCA 5 raw moldens + orca.out charges | 3 omol_tests jobs (Y and Sb ECP) | coefficients 4.7e-12, density 5.4e-12 rel | same |
| 2026-11-02 | same | 312-atom job, 1.95 GB molden | coefficients 4.7e-12, density 5.3e-12 rel; 13 s and 265 MB to read (Multiwfn convert 70 s, 4.6 GB) | same |
| 2026-11-02 | Multiwfn's orca.wfx text (write_wfx) | all of the above | every section but the title identical token for token; 99.983-99.998% of MO coefficient tokens identical, the rest differ in the last digit (worst 2.5e-10 rel) | scratchpad `conv/val_writer.py` |
| 2026-11-02 | ORCA orca_plot cubes (ORCA 6.0.1) | NO UKS PBE0/cc-pV5Z, virtual MOs dominated by f, g, h | fixed mode within 6.3e-6 (cube print precision); Multiwfn's convention off by 1.1 on the h-dominated beta MO | `tests/test_files/wavefunction/no_cc_pv5z` |
| 2026-11-02 | effect of the h fix (fixed vs multiwfn_h, current engines) | 72 open-shell X2C nitrates | charges up to 5.8e-5 e, fuzzy 8.9e-5 e, fuzzy_bond 2.2e-5; ALIE unchanged; CP sets identical; actinide NCP unchanged; actinide-ligand BCPs rho 3.3e-4 rel, Laplacian 1.3e-3 rel, position 5.8e-4 Bohr; other CPs unchanged except esp_total 4.6e-5 rel | scratchpad `hbug/` |
| 2026-11-02 | Multiwfn's orca.wfx, ORCA 5 | 93 Pt4Hx clusters (ORCA 5, all-electron SARC-DKH, s-f shells, 53 open shell; `data/cross_validation_wfns/ptcluster_*`), molden from ORCA 5 orca_2mkl, reference wfx from local Multiwfn | reader: coefficients 5.0e-12, density 3.3e-12 rel; writer: every section identical, 99.997% of coefficient tokens identical (worst 5.5e-11 rel). The 94th folder has no gbw | scratchpad `conv/pt_reader.log`, `pt_writer.log` |
| 2026-11-02 | programs that read orca.wfx, run on Multiwfn's file and on ours | 6 OMol jobs (ECP open and closed shell, open-shell organics, HXeOH) + 2 open-shell X2C nitrates (multiwfn_h mode) | Multiwfn QTAIM CPprop.txt: identical token for token on the 6 OMol jobs; nitrates differ only at values near 1e-11 and by at most 8.5e-7 rel elsewhere. Multiwfn other_esp, laplacian_bond, grad_norm_rho_fuzzy: identical. critic2: same bond CPs; values identical single-threaded (multithreaded critic2 varies run to run by itself). HORTON: identical charges on the 6 OMol jobs; it fails on Np and Pu with either file (no tables for Z 93-94) | scratchpad `conv/consumers/` |
| 2026-11-02 | NERSC openac sample, full runner, `--native_convert multiwfn_h` vs `prod_l1` | 440 jobs (all families, bases, functionals; ORCA 6.1; molden-only families included) | charges and fuzzy within 2.6e-12 e; QTAIM sets identical 440/440, density, Laplacian and positions identical; spin density 8.6e-6 a.u. at heavy nuclei (1e-11 of rho) | NERSC `actinide_test/nc_compat_audit.tsv` |
| 2026-11-02 | NERSC openac sample, `--native_convert fixed` vs `prod_l1` | 440 jobs | only open-shell X2C jobs change: charges median 3.7e-6, max 9.8e-4 e; CP density max 6.4e-3 rel, Laplacian 0.10 rel (with a 1e-2 a.u. floor) and 1.9e-2 a.u. absolute at actinide CPs, positions 8.5e-3 Angstrom; CP sets identical; ALIE unchanged. Concentrated in 4 structures (PuN3O11 t, NpC15H13N4O6 t, PuC6H14O8 t, UC32H47ClN d); every other job 5e-5 e or less | NERSC `actinide_test/nc_fixed_audit.tsv` |
| 2026-11-03 | LRC OMol, full runner at full_set 1, Multiwfn convert vs `--native_convert fixed` | first 18 tm_react jobs validated in both trees (run continuing) | charges, fuzzy, fuzzy_bond, VDD and ALIE within 1.9e-12; MBIS within 3.0e-7 e (iteration noise of the converged fit); QTAIM sets and properties identical; Multiwfn's CHELPG on our wfx within 2e-10 e (its last printed digit) | LRC `engine_test/nc_ref_l1`, `nc_fixed_l1` |
| 2026-11-02 | full runner, three trees (Multiwfn convert, multiwfn_h, fixed), engines on | 12 nitrates (open and closed shell, X2C and def2 ECP) | multiwfn_h vs Multiwfn convert: charges and fuzzy within 8.5e-14 e, QTAIM sets and properties identical; fixed vs Multiwfn: changes only on open-shell X2C (charges 5.8e-5 e, BCP rho 2.9e-4 rel); 12/12 validate in each, no Multiwfn convert in the native trees | scratchpad `rehearse/` |

## 5. Invariant audits (no reference)

`scripts/nersc/audit_prod.py`: charge sums, spin sums, fuzzy populations and grid electron
count against the all-electron count, an NCP on every actinide, Poincare-Hopf.

| date | data | result | raw output |
|---|---|---|---|
| 2026-10-25/27 | openac `prod_l1`, 4235 jobs (8 functionals x 3 X2C bases + 2 OMol levels) | no folder flagged; electron count within 3.9e-5 rel (median 1.1e-6, on the first 3602); Poincare-Hopf (kept set) 1 everywhere; nucleus frame used in 454 | NERSC `actinide_test/prod_l1_audit.tsv` |
| 2026-10-27 | multi-spin benchmarks `prod_l1_santi`, 179 jobs | no folder flagged | NERSC `actinide_test/prod_l1_santi_audit.tsv` |

## 6. Reference cross-checks (Multiwfn itself)

These test Multiwfn and the conventions, not the engine. They decide which conventions
the engine copies.

| date | comparison | data | result | raw output |
|---|---|---|---|---|
| 2026-07-28 | HORTON Hirshfeld vs Multiwfn hirshfeld | 87 wfx_pull (62 all-electron for Hirshfeld) | r = 0.9993, median 0.004 e over 1030 atoms | `data/cross_validation_wfns/horton_comparison.csv` |
| 2026-07-28 | HORTON Becke with Multiwfn's radii vs raw Multiwfn Becke | 87 jobs | r = 0.994, median 0.005 e; the shipped `becke` is ADC-corrected, not raw | same |
| 2026-07-27 | HORTON MBIS vs Multiwfn MBIS | 1 molecule (aniBB, 12 atoms) | max 0.0028 e | session scratchpad |
| 2026-07-28 | critic2 vs Multiwfn QTAIM | 1438 shared BCPs | median difference 0, r = 1.0000 for rho, Laplacian, energy densities; ELF/LOL differ by convention (r about 0.98); BCP sets identical in 89% of jobs | `data/cross_validation_wfns/critic2_comparison*.csv` |

## 7. Gaps

Current behaviours without a validation on current code, and checks not yet done:

| gap | why it matters | suggested check |
|---|---|---|
| converged MBIS has no independent reference | it is the production default and moves actinide charges by up to 0.38 e from Multiwfn's rule | HORTON MBIS (horton-part) on the 87 wfx_pull + nitrates |
| engine-only CPs not confirmed by critic2 | 92 openac jobs and the qeng2 extras rest on Poincare-Hopf alone | critic2 on the engine's CP positions (QTAIM P4) |
| NERSC ALIE 3.4e-5 rel | larger than every other ALIE comparison | rerun Multiwfn and the engine on a few openac jobs locally |
| qeng2 esp_total 4.6e-2 rel | unchecked | 80-bit or gbasis ESP at those CPs |
| tier 2 3 BCP Laplacians (up to 1.5e-7 rel) | unconfirmed as position rounding | evaluate at perturbed positions |
| determinism across thread counts (QTAIM criterion E) | not recorded | run 1, 4, 8 threads on the ladder |
| level-of-theory matrix | double hybrids, HF, Cartesian basis sets, ROKS beyond one radical, ORCA 5 vs ORCA 6 not covered | tier 3 supplementary set |
| results recorded on older code | the 87-job, 18-reference and 16-reference local sets predate the threading fixes (#63-#65) and the converged MBIS default | rerun the local sets on current code; they take minutes |
| full_set 2 routines | still Multiwfn (bader, laplacian_bond) | when engines exist |
| ESP surface vs Multiwfn's exact code | exact-code parity on 2 fixtures only; the libreta differences on 99 jobs are attributed, not shown | eleesp1 (iESPcode=1) references on more jobs; slow (about 25x libreta) |
| fuzzy ELF / gradient / Laplacian on a cluster | only local sets so far; the Laplacian ECP-core differences are read as roundoff, not proven | LRC tm_react `nc_ref_l1` (elf_fuzzy); 80-bit Laplacian on one Hf or Tl job |

## 7b. Runs in progress (no results yet)

Cluster runs submitted or planned whose results are not in the tables above. Each moves
into its section as a row when it is audited.

| cluster | job id | name | what | trees | code | status (2026-11-03) |
|---|---|---|---|---|---|---|
| LRC | 27498703, 27498704 | nc_ref, nc_fixed | converter check, full_set 1, `jobs_ref.txt` (200 tm_react) | `engine_test/nc_ref_l1` (`--native_convert off`) vs `nc_fixed_l1` | clone `qtaim_generator_nc`, feat/charge-engine before the MBIS changes | all 8 tasks TIMEOUT at 12 h; 18 of 200 compared (row in 4b). Not continued: the rerun uses fresh trees `nc_ref_l1b`/`nc_fixed_l1b` from perf/mbis-converged, so no tree mixes MBIS versions |
| LRC | 27525266 | fz1 | elf_fuzzy and level 1 engine vs Multiwfn, full_set 1, `jobs_ref.txt` | `engine_test/fz_l1` (engine) vs `ref_l1`, `compare_engine_merge.py` (`report_fz_l1.json`) | clone `qtaim_generator_fz` (fuzzy-ELF work) | running |
| LRC | 27527328, 27527329 | ref2, fz2 | level 2 engine vs Multiwfn, full_set 2, `jobs_l2.txt`, 5 jobs per task | `engine_test/ref_l2` (Multiwfn) vs `fz_l2` (engine) | clone `qtaim_generator_fz` or `qtaim_generator_esp`: the submitting session drafted both, which one ran is to be confirmed there | running; an earlier fz2 (27525268) ended after 10 s |
| NERSC | - | l1_off, l1_fixed, l2_off, l2_fixed | converter at full_set 1 and 2 with every engine routine: MBIS (merge), VDD, CHELPG, elf_fuzzy, grad/Laplacian fuzzy, other_esp | `actinide_test/{l1,l2}_{off,fixed}`, 440-job sample (level 2: every 11th job), `audit_prod.py` fixed vs off, off vs `prod_l1` | integration branch: perf/mbis-converged + feat/engine-esp-surface | planned |

## 8. Adding a row

For every new engine routine or validation run, add one row with: date, reference kind,
data (where it lives, how many jobs, which elements and levels of theory), the worst
difference with units, the code (commit or PR), and the raw-output location. A divergence
from Multiwfn by design also goes in the changelog in `README.md`.
