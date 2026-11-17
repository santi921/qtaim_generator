# Engine validation ledger

Every validation run of the native engines (charge, surface, QTAIM, DDEC6) against Multiwfn,
Chargemol, HORTON, gbasis, critic2, extended precision or physical invariants. One row per run, with the
reference, the data, the result and where the raw output lives.

State as of 2026-11-11, branch `feat/charge-engine` (c309d2a), plus `feat/engine-esp-surface`. Companion documents:
`README.md` (decisions, divergence changelog), `qtaim_verification_campaign.md` (QTAIM
tiers 1-3), `mbis_converged.md` (MBIS).

## How to read this

- **Reference kinds.**
  - *Multiwfn parity*: the engine reproduces Multiwfn 3.8 on the same wavefunction. The
    bar is Multiwfn's print precision, since Multiwfn's output is text.
  - *Chargemol parity*: the DDEC6 port reproduces Chargemol 09_26_2017 on the same
    wavefunction. Locally the reference is a serial -O2 build that prints es25.16 and dumps
    its grids, so the bar is bitwise; on the clusters it is the OpenMP build, whose summation
    order differs (about 1e-10).
  - *Independent code*: HORTON 3 (gbasis, horton-part), critic2. Checks the kernels or
    the method, not the parity.
  - *Extended precision*: an 80-bit long-double evaluation, used to decide which code is
    right where two disagree.
  - *Invariants*: physical checks that need no reference (charge and spin sums, electron
    count on the grid, Poincare-Hopf, an NCP on every nucleus).
- **Engine version.** Results are only as current as the code they ran on. The column
  "code" gives the branch or commit where it was recorded. Since 2026-11-11 each job
  folder's provenance.json records the ENGINE_VERSION of every engine routine (README.md,
  "Engine versions and provenance"). Section 7 lists which current
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
  (on a cluster: ELF on 70 LRC tm_react jobs, gradient and Laplacian on 9 and on 33 NERSC
  actinide jobs).
- **Bond orders** (ibsi_bond at full_set 1, laplacian_bond at full_set 2) reproduce
  Multiwfn's pair sets everywhere and its printed values (IBSI on Int(dg_pair) to 5.0e-6,
  LBO to 5.0e-7) on 99 local jobs and 40 NERSC actinides up to 211 atoms. IBSI runs on the
  75 x 434 grid by design; against production's 30 x 110 that moves values by up to 3.0e-2
  (2.96e-2 on 440 NERSC and 2.58e-2 on 200 LRC jobs through the full runner).
- **QTAIM kernels** agree with gbasis to a median 7.5e-14 rel (rho) on 100
  wavefunctions; where they differ more, 80-bit evaluation sides with the engine. The CP
  search reproduces Multiwfn's CP sets exactly in its Multiwfn-compatible mode. In engine
  mode it finds every CP Multiwfn finds (0 missing on 1498 LRC jobs and 419 actinide
  jobs) plus CPs Multiwfn misses.
- **DDEC6** (opt-in, up to 120 atoms) is bitwise identical to serial Chargemol on the 3
  dump fixtures (grids, charges, iteration count, multipoles, spin) and within 1.0e-10 of
  the OpenMP build on 79 openac actinide jobs and 76 OMol jobs, with equal iteration counts
  everywhere. Since the density-kernel work the port takes a median 0.37 of Chargemol's time
  on openac and 0.83-0.89 on OMol.
- **Gaps:** converged MBIS (the current default) has no independent reference; the ALIE
  difference on the NERSC actinides (3.4e-5 rel) is unexplained; engine-only CPs are not
  yet confirmed by critic2; no run covers double hybrids, HF, Cartesian basis sets or a
  systematic ORCA 5 vs 6 comparison.

## Runs in progress

Cluster runs submitted but not finished. When a run ends, its result becomes a row in the
section it validates and its entry here is removed; an abandoned run is removed with a note
in its row or in section 7.

LRC `fz1` (27525266) and `ref2`/`fz2` (27527328/27527329) ended at the 12 h limit on
2026-11-04 and were not resubmitted; their results are rows in section 2 (and the ESP row in
section 3 on `feat/engine-esp-surface`).

LRC `ncref`/`ncfix` (27498703/27498704) hit the 12 h limit and were not continued: fresh
trees `nc_ref_l1b`/`nc_fixed_l1b` replaced them so no tree mixes MBIS versions. The NERSC
level 1/2 batches and the LRC `l1b` pair are done (rows in section 4b, 2026-11-12).

The 2026-11-16 batch (NERSC 61414540/43/45/48, LRC `v8_l1`) is done; rows in sections 2
and 6. The reruns of the 8 full_set 2 folders that hit the 4 h limit (61476672/73) were
cancelled.

No runs in progress (last checked 2026-11-17).

## 2. Charge engine (`core/charge_engine.py`)

Routines: hirshfeld, adch, cm5, becke (Multiwfn's ADC-corrected Becke), hirsh and becke
fuzzy density and spin, fuzzy_bond (full_set 0); vdd, mbis, mbis fuzzy density and spin,
elf_fuzzy, chelpg, ibsi_bond (full_set 1); grad_norm_rho_fuzzy, laplacian_rho_fuzzy,
laplacian_bond (full_set 2).

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
| 2026-11-13 | Multiwfn parity, local, fresh Multiwfn (iESPcode=1 and 2) | 2 fixtures (XeCl4 ECP + EDF; rmechdb_652 open shell): chelpg | engine on Multiwfn's Boys function vs iESPcode=1 within 4.6e-11 e, points identical (6022, 5241); exact kernel vs iESPcode=1 1.25e-6 / 3.7e-7 e; exact vs libreta (production) 3.7e-6 / 8.2e-5 e; RRMSE equal to Multiwfn's printed 0.480105 / 0.379236 | feat/engine-chelpg | `tests/test_files/charge_engine/*/multiwfn_reference_chelpg.json` |
| 2026-11-13 | stored Multiwfn data (production `chelpg.out`), local | 340 stored outputs with a wavefunction: 8 without actinides, 7 actinide runs with the radii file, 325 actinide runs from before the Batsanov radii file | fitting-point count identical 340 of 340 when the engine's grid uses the radii each run used; with the engine's own radii identical on all 8 + 7 current-mode runs; 169 of the 325 old-mode actinide runs differ, as expected from the radii change | feat/engine-chelpg | `data/engine_test_2026-11/chelpg/point_sweep.json` |
| 2026-11-13 | Multiwfn parity (production libreta, iESPcode=2), local | 99 jobs: 87 wfx_pull (7 OMol verticals) + 12 nitrates (`list99.txt`) | points identical 99 of 99; charges median 9.7e-5 e per job, p90 2.6e-3, max 1.89e-2 (mo_hydrides 5383, 3_Cl); 17 jobs above 1e-3 e; 12 nitrates at most 1.4e-3. Attributed to libreta by the row below. Speed at 8 threads each, one job at a time, on 10 of the 99: engine 1.27x Multiwfn (median; 0.89-1.55x, aggregate 1.37x; engine time excludes reading the wfx). An earlier 3.0x median compared the engine at 8 threads against Multiwfn at 4 threads with 4 jobs running at once and is withdrawn | feat/engine-chelpg | `data/engine_test_2026-11/chelpg/cmp99_libreta.jsonl`, `cmp_speed.jsonl` |
| 2026-11-13 | Multiwfn parity (exact code, iESPcode=1), local, fresh Multiwfn | 33 jobs from the 99: the 6 most libreta-divergent (all above 5.8e-3 e), plus a spread over verticals and condition numbers; 4 actinides (U and Pu ECP, Pu and Th X2C), lanthanides (Sm, Er, Ce), cond up to 1.13e6 | engine on Multiwfn's Boys function within 2.04e-10 e (median 4.6e-11; MOR22 Sm3Ir3), points identical 33 of 33, 0 jobs above 1e-9; exact kernel within 1.02e-5 e (median 1.25e-6; 18 above 1e-6, 2 above 1e-5: Pu X2C on Pu, K+ elytes on S). Multiwfn iESPcode=1 vs its own iESPcode=2 matches the engine-vs-libreta difference on all 33 (same worst atom 33 of 33, ratio median 1.00) | feat/engine-chelpg | `data/engine_test_2026-11/chelpg/cmp_iesp1_{mw,exact}.jsonl`, `attrib.py` |
| 2026-11-03 | Multiwfn parity, LRC, full_set 1, charge engine version 2 | tm_react, 70 of 200 jobs compared (73 finished before the 12 h limit; not resubmitted), 43 open shell; engine on native-converted wfx (`multiwfn_h`, per its timings.json), reference on Multiwfn's | elf_fuzzy within 5.00e-9 (median 2.51e-9) over 5097 atoms; hirshfeld, cm5, vdd, Becke and Hirshfeld fuzzy within 5.0e-9 e, adch 7.8e-8, becke 8.11e-8, fuzzy_bond 5e-7; ALIE as the 2026-10-12 row; no routine missing. Non-engine data (first 34 jobs): chelpg within 2.0e-10 e (Multiwfn's last printed digit, as in section 4b); the `other.json` flags are the planarity fields (mpp, sdp within 5e-7 Angstrom, print precision), which `compare_engine_merge.py` does not yet exempt as engine output; ORCA Mulliken charges identical (the Ru4 step27 flag on the ORCA charge entries and qtaim.json is in another field, not yet identified). MBIS: mbis and mbis_fuzzy_density up to 0.71 e (median 3.6e-4), mbis_fuzzy_spin 3.4e-2, from Multiwfn's stopping rule: on MOBH12_V4 (mult 2, the 0.71 e job) `--mbis_mode multiwfn` reproduces Multiwfn exactly (85 cycles, V +1.50) and the converged fit (139 cycles, V +0.79) is the one fixed point of all three guesses (local rerun, `data/engine_test_2026-11/mbis_lrc/`). Speed: median 4.6x, p10 0.4x, aggregate 1.2x; the slow jobs spend their time in converged MBIS (MOBH10_Ag1Nb5 step0, one Ag and one Nb: 8362 s in the charge engine on LRC; locally 6057 plain-fallback cycles with the outer Ag shells merged, vs 137 cycles in multiwfn mode), handed to the MBIS work (`mbis_lrc/mbis_handoff.md`) | 59ac8ef | LRC `engine_test/fz_l1` vs `ref_l1`, `report_fz_l1.json` |
| 2026-11-04 | Multiwfn parity, LRC, full_set 2 (full runner) | tm_react, 9 of the first 20 jobs of `jobs_ref.txt` compared (Re, Ag + Nb, Ti, Pt, Pd, Hf; 4 open shell; 16 of 20 finished in `ref_l2`, 10 in `fz_l2` before the 12 h limit) | grad_norm_rho_fuzzy within 4.99e-9 (median 2.44e-9); laplacian_rho_fuzzy median 2.68e-9, max 1.08e-6 (the size of the local Hf ECP-core case; atom not checked); elf_fuzzy 5.00e-9; every other non-MBIS scheme within 5.1e-9 e, fuzzy_bond 5e-7; MBIS up to 0.43 e (as in the full_set 1 row). Multiwfn's own bader on our wfx differs by 1e-6 to 2.9e-4 e (not yet tested whether Multiwfn's bader varies by that much run to run); chelpg 1e-10 e. Time: the level-2 derivative pass adds 0 to 33 s (charge engine at full_set 1 vs 2 on the 2 jobs in both trees: 4140 vs 4103 s, 194 vs 227 s); the charge engine's 3500-17600 s on 7 of the 9 jobs is the converged-MBIS cost of the full_set 1 row (inferred from that overlap and the local profile, `_meta` not written by this code). Multiwfn's bader alone takes 2200-11500 s per job and has no engine | `feat/engine-esp-surface` (qtaim_generator_esp clone) | LRC `engine_test/ref_l2` vs `fz_l2`, `report_fz_l2.json` |
| 2026-11-16 | Multiwfn parity (Multiwfn 3.8(dev) 2024-Oct-6, production settings.ini), local | 2 fixtures (rmechdb_652 open shell, XeCl4 ECP + EDF) | ibsi_bond against IBSI grid option 4 (75 x 434), compared on Int(dg_pair) (f8.5, independent of the H2 reference value): pair sets identical, within 4.5e-6; laplacian_bond (45 x 302): pair sets identical, within 3.9e-7 | feat/engine-ibsi-lbo | `tests/test_files/charge_engine/*/multiwfn_reference_bond.json` |
| 2026-11-16 | same | 99 jobs: 87 wfx_pull (7 OMol verticals, 25 ECP) + 12 actinide nitrates (`chelpg/list99.txt`), 2-39 atoms | ibsi_bond: pair sets identical 99 of 99, Int(dg_pair) within 5.00e-6 (median of the per-job max 4.9e-6: print rounding); laplacian_bond: pair sets identical 99 of 99, within 5.00e-7. With the reference value of this build (0.500791) every printed IBSI is reproduced. Time, 16 threads, median per job: IBSI 0.046 s in the engine (shared pass) vs 0.85 s Multiwfn at 30 x 110 and 8.5 s at 75 x 434; LBO 1.73 s vs 2.25 s (totals 241 s vs 514 s) | feat/engine-ibsi-lbo | `data/engine_test_2026-11/ibsi_lbo/` (`refs/`, `cmp99.jsonl`, `compare.py`) |
| 2026-11-16 | production grid (IBSI option 1, 30 x 110), divergence by design | same 99 | engine (75 x 434) vs production at the same reference value: up to 2.3e-2 absolute, relative up to 18% and median 4.1% over pairs with IBSI >= 0.05 (worst: omol UN3O11, a U nitrate). Changelog entry in README.md | feat/engine-ibsi-lbo | same |
| 2026-11-16 | Multiwfn parity on the same orca.wfx, NERSC (Multiwfn 3.8(dev) 2025-Jun-26, H2 reference 0.566653), `campaign.py` | openac `sample_l2.txt`, 40 actinide jobs (X2C and def2, 16-211 atoms, median 73) | ibsi_bond: pair sets identical 40 of 40, Int(dg_pair) within 5.00e-6 of grid option 4 (print rounding); laplacian_bond: pair sets identical 40 of 40, within 5.00e-7. Grid vs production 30 x 110 at the same reference, relative over pairs with IBSI >= 0.05: per-job max median 9.9%, worst job 20.6%, per-job median 0.2%. Time, median per job: IBSI 2.0 s in the engine vs 40 s Multiwfn at 30 x 110 and 372 s at 75 x 434; LBO 22.7 s vs 96.5 s | 25ae633 | NERSC `actinide_test/ibsi_lbo_ref/`, summary in `logs/ibsilbo_61414548.out` |
| 2026-11-16 | full runner from scratch, NERSC, full_set 1, charge + QTAIM engine (array 61414540) | openac 440-job sample: `v8_l1` against `chelpg_l1_v7` (version 7 restart) and `l1b_fixed` (f66f277: Multiwfn IBSI at 30 x 110, libreta chelpg) | 440 of 440 validated. Against `chelpg_l1_v7`: every scheme identical except ibsi_bond. Against `l1b_fixed`: ibsi_bond pair sets identical (216602 values), median 1.25e-4, max 2.96e-2 (the grid change; same reference value on this build); chelpg max 3.26e-2 e (libreta, as in the CHELPG cluster rows on feat/engine-chelpg); mbis 8.4e-5 e (version 5 selection); every other scheme, ALIE and planarity 0; QTAIM key sets identical 440 of 440. Time (provenance `timings_s` vs Multiwfn steps): IBSI 943 s total, median 2.0 s per job, vs 36290 s and 40.8 s (38.5x); chelpg 67712 s vs 191657 s (2.8x) | 25ae633 | NERSC `actinide_test/v8_l1_vs_l1b_fixed.{txt,json,tsv}`, `v8_l1_audit.txt`, `timing_breakdown.py` |
| 2026-11-17 | full runner from scratch, LRC, full_set 1 (run by the user) | tm_react `jobs_ref.txt`, 200 jobs, 116 open shell: `v8_l1` against `nc_fixed_l1b` (f66f277) | 200 of 200 compared. ibsi_bond pair sets identical (84936 values), median 1.16e-4, max 2.58e-2; chelpg 4.52e-3 e and mbis 2.45e-4 e (mbis_fuzzy_density 1.52e-4), both as in the LRC CHELPG run; every other scheme, ALIE and planarity 0; no engine routine missing, no non-engine data changed. Per-routine time not pulled | 25ae633 | LRC `engine_test/reports/v8_l1_vs_nc_fixed_l1b.json` |
| 2026-11-17 | full runner from scratch, NERSC, full_set 2: engine (`v8_l2`, 61414543) vs Multiwfn mode from the same checkout (`mw_l2`, 61414545) and vs `l2b_fixed` | openac `sample_l2.txt`, 40 jobs; 39 finished in `v8_l2`, 33 in `mw_l2` within 4 h (the rest stopped in Multiwfn's bader on the largest X2C systems; their reruns were cancelled) | Against `mw_l2` (33 jobs): laplacian_bond pair sets identical, within 5.00e-7; laplacian_rho_fuzzy within 8.85e-7, grad_norm_rho_fuzzy 6.19e-9, elf_fuzzy 5.00e-9, every other non-MBIS charge and fuzzy scheme within 6.1e-8 e, fuzzy_bond 5e-7; ibsi_bond 2.72e-2 (grid), chelpg 1.07e-2 e (libreta); ALIE and planarity at print precision. These are the first full_set 2 fuzzy integrals on cluster actinides. Against `l2b_fixed` (39 jobs): every engine scheme 0 except ibsi_bond and chelpg (as above); laplacian_bond within 5.0e-7 on shared pairs, 353 pairs on one side only because `l2b_fixed` was parsed by the old parser (101 keys containing ")" in 33 of 39 jobs, several pairs collapsed into each). Time against `mw_l2`: LBO 644 s vs 2902 s (4.5x), IBSI 32 s vs 1311 s, chelpg 2494 s vs 6596 s; every non-bader step 15362 s vs 53995 s (3.5x); bader is 86% of the engine tree's time (95463 s) | 25ae633 | NERSC `actinide_test/v8_l2_vs_{mw_l2,l2b_fixed}.{txt,json}` |

`compare_engine_merge.py`'s speedup line puts the new tree's whole charge-engine time
against only the Multiwfn steps the new engine replaced. Against a baseline that already ran
an engine (`l1b_fixed`, `nc_fixed_l1b`) it is not a speedup (0.9x and 0.5x there); the rows
above use the per-routine `timings_s` from provenance.json instead
(`scripts/ibsi_lbo_reference/timing_breakdown.py`).

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
| HORTON MBIS (horton-part, own grid and guess) vs converged engine, 2026-11-04 | MOBH12_V4 (tm_react, 54 atoms, V, mult 2; converged vs Multiwfn's rule differ by 0.71 e) | max 0.0083 e, median 0.0011 e (vs Multiwfn's rule 0.706 e): the first independent reference for converged MBIS |
| every-atom selection objective + MBIS_TIE_REL 1e-8 + stalled-check continuation + fuzzy refit from the charge fit (2026-11-04, perf/mbis-converged 337370c) vs b9d9d30 | 112 jobs (as above) | 111 identical; omol ThN6O18 def2-TZVPD moves to another of its 3 solutions (0.026 e); the old objective (atmrhocut cutoff) reversed rankings on a Pu nitrate (mbis_converged.md) |
| fused map kernel + pair cache (2026-11-04, perf/mbis-kernel) vs perf/mbis-converged | 112 jobs | charges within 2.5e-7 e (median 3.2e-13), selection and solution counts identical on all 112; evaluation counts ratio 0.64-1.31 (last-digit exp shifts where accelerated fits stop); MBIS selection 2836 s to 2184 s on a loaded workstation |
| MBIS between the Multiwfn-convert and native trees, NERSC (2026-11-12, f66f277) | openac 440-job sample, full_set 1 | 2 of 440 differ: NpN6O18 PBE X2C-SVPall 0.163 e (one tree's x1.5 stalled and merged to a worse point), PuN3O11 R2SCAN X2C-SVPall 0.297 e (one tree's x1.5 stalled and was dropped). Locally PuN3O11 reproduces and NpN6O18 does not; both fixed points exist on both wavefunctions (within 0.0012 e of each other), so the flip is a search miss; a third, better fixed point on NpN6O18 at x2.5 (mbis_converged.md) |
| MBIS between the Multiwfn-convert and native trees, LRC (2026-11-12, f66f277) | tm_react, 200 jobs, full_set 1 | mbis within 1.4e-6 e, mbis_fuzzy_density 1.3e-6, mbis_fuzzy_spin 3.6e-8; no flips. 7 select an unconverged fit (same in both trees); the tie rule below moves 3 to a converged fit on the same point (up to 2.5e-4 e); 4 (Ta, Re) never converge on the selected point |
| ties within MBIS_TIE_REL go to a converged fit (2026-11-12, perf/mbis-kernel 7b9f0f5) | unit test with the openac PuN3O11 objectives (unconverged default and converged x0.6 2.8e-9 relative apart) | converged fit selected; with both converged the earlier guess stays |
| second accelerated window for a stalled fit (tested, not adopted) | the 4 wavefunctions of the two NERSC flips, locally | PuN3O11 native stalls again and falls back to the same point (+1000 evaluations); the other 3 unchanged |

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
| 2026-11-04 | production Multiwfn (libreta), LRC full runner at full_set 2 | tm_react, 9 jobs (Re, Ag + Nb, Ti, Pt, Pd, Hf; `ref_l2` vs `fz_l2`) | extremes within 0.067 kcal/mol (maximum) and 0.027 (minimum); positive and negative areas 5.8e-2 Bohr^2, polar and nonpolar 3.1e-2 Angstrom^2; overall average 6.2e-5 a.u., polarity index 1.7e-3 eV, skewness 1.1e-4; volume, total area and density within print precision (4.9e-6). All inside the range of the 99-job row. Surface engine 114-741 s vs Multiwfn other_esp 133-1292 s (1.2-2.9x) | `feat/engine-esp-surface` (qtaim_generator_esp clone) | LRC `engine_test/ref_l2`, `fz_l2` |

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
| 2026-11-13 | determinism across thread counts (criterion E), engine defaults | 8 local ladder jobs, 12-274 atoms (25-779 CPs; ECP, lanthanide, open shell) | CP arrays, CPprop.txt and qtaim.out byte-identical at 1, 4 and 8 threads on all 8; 274 atoms: 395 / 190 / 172 s | `data/engine_test_2026-11/qtaim_p4/determinism.jsonl` |
| 2026-11-13 | Multiwfn and critic2 searches started at the engine-only CPs (`qtaim_verify_stored.py --adjudicate`) | 3 local LRC jobs with NNAs (`data/engine_test_2026-10/ph_fail`: MOBH32_Pt2, MOR16_Ir4, ROST59_Ir3) | 5 engine-only CPs (3 RCPs at rho 2.4e-5 to 2.8e-5, 2 vacuum NNAs at rho 1.5e-5, 6.9-7.4 Bohr from any nucleus): all 5 confirmed by both programs (same type within 1e-4 Bohr) | `data/engine_test_2026-11/qtaim_p4/adj.jsonl` |
| 2026-11-13 | rounding check at the stored positions (`qtaim_compare_props.rounding_check`) | same 3 jobs | every Laplacian difference within the 1e-12 Bohr position spread plus half the last printed digit (NCP up to 4.3e-10 rel); esp_total differences 4.4e-9 to 9.8e-9 of abs(esp_e), the exact vs Fmch Boys function difference (decision 9), large relative to esp_total only where it nearly cancels | same |
| 2026-11-14 | Multiwfn and critic2 (694cb1ee, built on the cluster) searches started at the engine-only CPs, LRC (`qadj`, 27826590, feat/qtaim-p4 45ba764) | 57 qeng2 jobs: 42 with engine-only CPs, 9 with NNAs in the stored set, 4 with kept-set Poincare-Hopf 2, the 3 tier-2 Laplacian outliers, the 5 largest esp_total cases | 48 engine-only CPs (18 BCP, 21 RCP, 7 CCP, 2 NNA; 23 below rho 1e-3): all 48 confirmed by both programs. NNA jobs: the engine has the stored NCP count in all 9 (Sn2 job: 6 NNAs, 14 BCPs unpaired by both codes); in 2 a stored NNA below 1e-5 is dropped and another weak NNA found. No non-weak Multiwfn CP missing; kept-set Poincare-Hopf 1 on 53, 2 on the 4 known (full set 1). Laplacian: every difference within position and print rounding, the tier-2 outliers included (worst 1.45e-7 rel). esp_total: at most 9.8e-9 of abs(esp_e) (Boys function, decision 9) except MOBH8_Sn2 (Sn ECP): 2.8e-8 to 1.0e-7 at 3 CPs, also the Boys function (next row) | `data/engine_test_2026-11/qtaim_p4/lrc/` |
| 2026-11-14 | same, NERSC (61245039) against the stored CPprop.txt of `mwfn_l0` | 92 openac folders with engine-only QTAIM keys in `prod_l1_audit.tsv` | 89 engine-only CPs: 73 actinide NCPs (one per job, within 3.5e-11 Bohr of the nucleus) confirmed by critic2 73/73 and by Multiwfn 0/73 (its search fails at those nuclei even when started on them; the origin-distance loss, section 4b 2026-10-13/16), 14 RCPs and 2 BCPs confirmed by both. BCP pairs: 6116 same, 0 differ, 592 engine-only (the 73 jobs with the lost actinide NCP, and 17 jobs with no extra CP where Multiwfn's paths failed), 0 Multiwfn-only. No non-weak Multiwfn CP missing; Poincare-Hopf 1 on 92. Laplacian within rounding (worst 1.5e-8 rel); esp_total at most 6.1e-9 of abs(esp_e) | `data/engine_test_2026-11/qtaim_p4/nersc/` |
| 2026-11-14 | the LRC open case, locally (wavefunction regenerated with the production convert step) | MOBH8_Sn2 (74 atoms, charge -2, Sn ECP with EDF) | its 6 NNAs are vacuum maxima: rho 3.0e-4 to 8.6e-4, 2.5-3.5 Angstrom from the nearest H, none near a metal; the engine flags all 6 in `nna_vacuum`, and each of its 15 unpaired BCPs has a path ending at one of them. esp_total with Multiwfn's Boys function: at most 2.1e-12 of abs(esp_e) (7.9e-11 a.u.), against 1.0e-7 with the exact one, so the gap is the Boys function difference (decision 9); why it is larger here is not checked (Sn core primitives suspected) | `data/engine_test_2026-11/qtaim_p4/sn2/` |

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
| 2026-11-04 | NERSC openac sample at full_set 1, every engine routine (integration/nersc-l12-batch 5effd0e), `l1_off` (Multiwfn convert) vs `prod_l1` | 440 jobs | every level-0 scheme, ALIE and QTAIM property identical (0 difference): the new code reproduces production | NERSC `actinide_test/l1_off_vs_prod.tsv` |
| 2026-11-04 | same, `l1_fixed` vs `l1_off` | 440 jobs, all validated, none flagged | charges 9.8e-4 e, vdd 2.4e-3, chelpg 8.2e-3, fuzzy 9.6e-4, fuzzy_bond 8.1e-4, elf_fuzzy 3.4e-2 (ELF units, open-shell X2C carbenes); QTAIM sets identical 440/440; MBIS up to 0.30 e on 3 nitrates from the MBIS selection, not the converter (mbis_converged.md); 2 of 440 still differ on the next batch (rows below) | NERSC `actinide_test/l1_fixed_vs_off.tsv` |
| 2026-11-04 | same at full_set 2, `l2_fixed` vs `l2_off` | 40-job subset, 39 validated (one timed out at 4 h in both trees) | charges 4.0e-4 e, mbis 5.9e-4, grad_norm_rho_fuzzy 6.4e-3, laplacian_rho_fuzzy 1.6e-3 (a.u.); QTAIM sets identical; other_esp not compared | NERSC `actinide_test/l2_fixed_vs_off.tsv` |
| 2026-11-03 | LRC OMol, full runner at full_set 1, Multiwfn convert vs `--native_convert fixed` | first 18 tm_react jobs validated in both trees (run not continued; the `l1b` row below replaces it) | charges, fuzzy, fuzzy_bond, VDD and ALIE within 1.9e-12; MBIS within 3.0e-7 e (iteration noise of the converged fit); QTAIM sets and properties identical; Multiwfn's CHELPG on our wfx within 2e-10 e (its last printed digit) | LRC `engine_test/nc_ref_l1`, `nc_fixed_l1` |
| 2026-11-12 | NERSC openac sample, full_set 1 and 2, every engine routine (integration/nersc-l12-batch2 f66f277: MBIS kernel on the ESP-surface stack), fresh trees `l1b_off`, `l1b_fixed`, `l2b_off`, `l2b_fixed` | 440 jobs at full_set 1, 40 at full_set 2; all 520 validated (the 4 h level-2 timeout of the first batch is gone) | `l1b_off` reproduces `prod_l1`; `l1b_fixed` vs `l1b_off` and the level-2 pair differ as in the 2026-11-04 rows apart from MBIS, which differs on 2 of 440 (section 2); charge-engine time about 19% lower at the median and 55% at the maximum than the first batch | NERSC `actinide_test/l1b_fixed_vs_off.tsv` |
| 2026-11-12 | LRC OMol, full runner at full_set 1 (f66f277), `nc_ref_l1b` (Multiwfn convert) vs `nc_fixed_l1b` (fixed) | `jobs_ref.txt`, 200 tm_react, 116 open shell; 200 validated in both | every non-MBIS charge and fuzzy scheme, fuzzy_bond and ALIE within 2.9e-12 (chelpg 2.0e-10, Multiwfn's last digit); mbis 1.4e-6 e, mbis_fuzzy_density 1.3e-6; QTAIM key sets identical 200/200, properties identical (spin density 2.0e-11); no routine missing. `compare_engine_merge.py` lists 169 jobs with "non-engine data changed": chelpg and elf_fuzzy at the sizes above, compared exactly | LRC `engine_test/nc_l1b_audit.tsv` |
| 2026-11-02 | full runner, three trees (Multiwfn convert, multiwfn_h, fixed), engines on | 12 nitrates (open and closed shell, X2C and def2 ECP) | multiwfn_h vs Multiwfn convert: charges and fuzzy within 8.5e-14 e, QTAIM sets and properties identical; fixed vs Multiwfn: changes only on open-shell X2C (charges 5.8e-5 e, BCP rho 2.9e-4 rel); 12/12 validate in each, no Multiwfn convert in the native trees | scratchpad `rehearse/` |

## 4c. DDEC6 (`core/ddec6.py`, opt-in `--ddec6`)

A port of Chargemol 09_26_2017 (DDEC6 charges, spin, atomic dipoles and quadrupoles, tail
fits), not a Multiwfn routine. Molecules above 120 atoms are skipped (permanent skip in
provenance.json). Atomic dipoles and quadrupoles in atomic units. Speed ratios are port
wall time over Chargemol's with the same thread count.

| date | reference | data | result | code | raw output |
|---|---|---|---|---|---|
| 2026-11-05 | Chargemol parity, local, serial -O2 build with grid dumps | rmechdb_652_step10_0_2 (open shell), XeCl4_step2_0_1 (ECP), synthetic diffuse_h2 (12x) | pair data, valence/core/spin grids, charges, iteration count, multipoles, tail fits and spin moments bitwise identical; re-run after every density-kernel change since (last 0af383c) | 78de148, 2a75f16, 02aabc8 | `tests/test_ddec6.py` (DDEC6_SLOW=1), `data/ddec6_ref/dumps/` |
| 2026-11-12 | Chargemol parity, NERSC, OpenMP precise build | openac, 96 jobs from `actinide_test/sample.txt` (8 per node at 16 threads each) | 79 compared, 13 skipped above 120 atoms, 4 without `orca.gbw` (chalcogenides TZVP, Pu/U with Se6/Te6); 0 Chargemol stops; charges within 7.60e-11 e, atomic dipoles 2.97e-11, quadrupoles 1.61e-10, spin moments 2.70e-12 (53 open shell); iteration counts equal 79/79. Port 1.76x Chargemol's time (median, 1.26-2.00); a local profile put 98% of the port's time in the density grids | b24bc06 | NERSC `actinide_test/ddec6/ddec6_*.jsonl`, local copy `data/ddec6_ref/nersc_campaign/` |
| 2026-11-13 | same run, port only | 16 of the 96 (13 compared, 3 above 120 atoms), debug queue | `ddec6_port.json` identical to the 2026-11-12 run on 13/13 after the strided kernel order (56462a8), the fused accumulate steps (8ff3f19) and the per-point power tables (0af383c). Port over Chargemol median 0.37, total 0.46; slowest relative job omol def2-TZVPD at 0.99 (few rows per point, so the power tables do not help) | 0af383c | NERSC `actinide_test/ddec6_fused/` |
| 2026-11-15 | Chargemol parity, LRC lr7, OpenMP precise build | OMol, 90 jobs (30 each tm_react, 5A_elytes, rgd_uks) from `ddec6_test/ddec6_sample.txt`, one job at a time on 16 cores | 76 compared, 14 skipped above 120 atoms (all 5A_elytes), 0 errors or Chargemol stops; charges within 5.59e-11 e, atomic dipoles 3.09e-11, quadrupoles 9.45e-11, spin moments 3.60e-12; iteration counts equal 76/76. Port over Chargemol median 0.89 (5A_elytes), 0.83 (rgd_uks), 0.89 (tm_react) | 0af383c | LRC `ddec6_test/ddec6_*.jsonl` |
| 2026-11-15 | peak memory (`ru_maxrss`), local, 16 threads | UN3O11 def2-TZVPD (closed shell, 13.7M grid points, 0.30M pair rows), NpN6O18 X2C-TZVPP (open shell, 18.4M points, 3.45M rows), UCl4DMF4 (open shell, 23.7M points, 4.22M rows) | 1.09, 4.21 and 5.22 GB. `ddec6.estimate_peak_bytes` (0.3 GB + 48 or 96 bytes per grid point for closed or open shells + twice the pair list) was fitted on the first two and predicts 5.28 GB for the third. The closed-shell coefficient rests on one job | this PR | session scratchpad `mem_probe.py` |

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
| 2026-11-17 | Multiwfn bader, three runs of the same build (2025-Jun-26) on the same native wfx, 8 threads each | NERSC openac `l2b_fixed`, `mw_l2`, `v8_l2`, 33-39 actinide jobs per pair | not reproducible run to run: charges differ by up to 4.55e-3 e (50_C, AmC27H39 PBE0 X2C-TZVP), spin populations by up to 7.70e-3 (49_C, same job). Answers the open question in the 2026-11-04 LRC full_set 2 row | NERSC `actinide_test/v8_l2_vs_{mw_l2,l2b_fixed}.json` |
| 2026-07-28 | critic2 vs Multiwfn QTAIM | 1438 shared BCPs | median difference 0, r = 1.0000 for rho, Laplacian, energy densities; ELF/LOL differ by convention (r about 0.98); BCP sets identical in 89% of jobs | `data/cross_validation_wfns/critic2_comparison*.csv` |

## 7. Gaps

Current behaviours without a validation on current code, and checks not yet done:

| gap | why it matters | suggested check |
|---|---|---|
| converged MBIS has one independent reference (HORTON, one 3d cluster) | it is the production default and moves actinide charges by up to 0.38 e from Multiwfn's rule | HORTON MBIS (horton-part) on the 87 wfx_pull + nitrates |
| NERSC ALIE 3.4e-5 rel | larger than every other ALIE comparison | rerun Multiwfn and the engine on a few openac jobs locally |
| level-of-theory matrix | double hybrids, HF, Cartesian basis sets, ROKS beyond one radical, ORCA 5 vs ORCA 6 not covered | tier 3 supplementary set |
| results recorded on older code | the 87-job, 18-reference and 16-reference local sets predate the threading fixes (#63-#65) and the converged MBIS default | rerun the local sets on current code; they take minutes |
| full_set 2 routines | bader is still Multiwfn, and it is not reproducible run to run (up to 4.6e-3 e, section 6) | when an engine exists |
| stored laplacian_bond from the old parser | keys of pairs with a two-letter element are garbled (101 in 33 of 39 NERSC `l2b_fixed` jobs) and several pairs collapse into one key | audit or regenerate laplacian_bond in every stored tree before it is used |
| IBSI reference value between Multiwfn builds | 2024-Oct-6 prints 0.500791, 2025-Jun-26 and the 2026 sources 0.566653 (IBSI scales as its inverse, 13%); stored production IBSI mixes both by cluster | audit the build per tree from the out_files headers before IBSI is compared or released |
| ESP surface vs Multiwfn's exact code | exact-code parity on 2 fixtures only; the libreta differences on 99 jobs are attributed, not shown | eleesp1 (iESPcode=1) references on more jobs; slow (about 25x libreta) |
| ddec6 through the full runner on a cluster | the cluster runs called the port from `scripts/ddec6_reference/campaign.py`; `--ddec6` in the runner (sidecar, merge into charge.json, provenance, restart) is covered by unit tests only | `--ddec6 --restart` on a copy of a finished tree, e.g. LRC `nc_fixed_l1b` |
| fuzzy Laplacian on ECP cores | the ELF, gradient and Laplacian integrals now match on cluster actinides too (33 NERSC jobs, section 2); the Laplacian ECP-core differences are read as roundoff, not proven | 80-bit Laplacian on one Hf or Tl job |

## 8. Adding a row

For every new engine routine or validation run, add one row with: date, reference kind,
data (where it lives, how many jobs, which elements and levels of theory), the worst
difference with units, the code (commit or PR), and the raw-output location. A divergence
from Multiwfn by design also goes in the changelog in `README.md`.
