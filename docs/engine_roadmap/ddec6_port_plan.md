# DDEC6 port plan: Chargemol in the native engine

Date: 2026-11-03. Status: plan only, no code. Branch `feat/engine-ddec6` (worktree
`~/dev/qtaim_generator-ddec6`), off `feat/charge-engine` cfac9a3.

## Summary

- Port the DDEC6 path of Chargemol 09_26_2017 (v3.5, the latest public release) for
  nonperiodic wfx input into Python/numba. This covers the density-grid generation, the
  core and DDEC6 valence iterators, collinear spin moments, and later the DDEC6 bond
  orders. Thomas Manz approved the port (user, 2026-11-03).
- No Chargemol dependency in the pipeline. The reference ion densities become a
  generated data file in the package, like `data/multiwfn_edf.py`. The Chargemol binary
  is used only as the validation reference: locally and, for the campaign, on LRC and
  NERSC.
- Parity target: Chargemol's own output on the same orca.wfx. Stock Chargemol prints
  charges as f13.6, so the bar is 1e-6 e. A local precision patch on the reference build
  (patch files are allowed by its license) gives a tighter comparison.

Evidence labels: [V] verified in the Chargemol source or a file, [J] judgment, [L] low
confidence. Chargemol paths are relative to
`chargemol_09_26_2017/chargemol_FORTRAN_09_26_2017/sourcecode_linux/`.

## 1. Decisions on record

| # | Decision | Source |
|---|---|---|
| 1 | Port Chargemol's implementation (not a reimplementation from the papers); Manz approved | user, 2026-11-03 |
| 2 | No Chargemol runtime dependency; tables generated into the package | user, 2026-11-03 |
| 3 | Stage 1 reference harness, stage 2 native port validated against it | user, 2026-11-03 |
| 4 | Validation campaign on LRC and NERSC: actinides and some OMol verticals | user, 2026-11-03 |
| 5 | Cite at publication: DDEC6 part 1 and part 2 (RSC Adv 2016), part 3 (bond orders, RSC Adv 2017) if bond orders ship, and the Chargemol program (license item 6) | Chargemol license, instructions PDF section A4 |
| 6 | Buried volume deferred (still wanted, after the feat/engine-esp-surface merge), not part of this work | user, 2026-11-03 |
| 7 | Reference tables ship as `.npz` package data (D1) | user, 2026-11-03 |
| 8 | Tolerance for the pair-renormalization step chosen when P2 gets there, if bit-level parity is impractical (D2) | user, 2026-11-03 |
| 9 | Chargemol binary allowed on LRC and NERSC for reference runs only | user, 2026-11-03 |
| 10 | Durable source copy in ~/dev/chargemol_src (D5) | user, 2026-11-03 |
| 11 | Size cap 120 atoms (skip above it, recorded like other skipped routines); DDEC6 need not cover the full OMol or OpenActinides sets. At 120 atoms a dense primitive 1PDM is about 1 GB per spin, so K2 still uses tiles | user, 2026-11-04 |

## 2. What Chargemol does on a nonperiodic wfx

Main flow [V] chargemol.f08:115-180:
`read_job_control` -> `run_valence_core_densities` (read_wfx +
`generate_density_grids_from_gaussian_basis_set_coefficients` + electron-count check) ->
`core_iterator` -> `DDEC6_valence_iterator` -> `local_multipole_moment_analysis` ->
`total_multipole_moment_analysis` -> `cloud_penetration` ->
`generate_net_atomic_charge_file` -> (if spin) `collinear_spin_moments_iterator` -> (if
compute_BOs, default true) `perform_bond_order_analysis` +
`compute_atomic_radial_moments` + `print_overlap_populations`.

Key facts that shape the port:

- [V] The spherical averages are radial bins on a uniform grid. There are 100 shells out
  to 500 pm (module_global_parameters.f08:12, 19; shell volumes at chargemol.f08:113-116).
  The uniform grid is part of the method, so a faithful port keeps it. The engine's
  Lebedev/Becke grids are not reused here.
- [V] Density generation is not plain sampling
  (module_gen_dens_grids_from_gaussian_basis_set_coeff.f08, 2641 lines, steps at
  61/232/326/460/577/837/876):
  - pick the grid (preferred spacing 0.14 Bohr, module_global_parameters.f08:57)
  - build primitive-pair density matrices
  - a cutoff test per primitive pair
  - an electron-count check
  - per-pair analytic corrections (electron count, spin, dipole, quadrupole)
  - coarser grids (2, 3, 4, 6, 8, 12x) for diffuse pairs (alpha sum < 0.4,
    module_global_parameters.f08:61)
  - a pairwise renormalization pass so each pair's numeric integral matches its analytic
    overlap
  So the engine's `orbital_values`/`density` cannot simply be substituted; the pair
  machinery has to be ported.
- [V] EDF: read from the wfx when present (module_read_wfx.f08:94-98, 423-446).
  Otherwise `add_missing_core_density` builds the core from Chargemol's core reference
  densities. Our native wfx carries the Multiwfn EDF, so the two codes see the same core
  density on ECP atoms. The fallback runs only for (Z, ncore) pairs Multiwfn's library
  lacks.
- [V] Primitive types up to h (L=5, wfx types 1-56) are accepted
  (module_read_wfx.f08:189-422), so X2C bases with g functions are covered.
- [V] Convergence tolerances are 1e-5 for charges and 5e-5 for spin
  (module_global_parameters.f08:26-27). The parity bar is set by convergence plus print
  precision (f13.6, module_generate_net_atomic_charge_file.f08:90), not by the grid.
- [V] Reference densities: `atomic_densities/` holds c2 (conditioned reference ions,
  Z = 1-109, several charge states) and core files, each in 400 pm/75-shell and
  500 pm/100-shell variants; 16 MB of text. With nshells = 100 and cutoff 500 pm, only the
  `_500_100` files should be needed [J, confirm in `initialize_atomic_densities`].

### Port scope (nonperiodic, wfx, collinear)

| Block | Chargemol modules (lines) | Port | Notes |
|---|---|---|---|
| wfx read and density matrices | read_wfx (869) | partial | Reuse `charge_engine.read_wfx` / `wavefunction.read_orca_molden`; port only Chargemol's interpretation (occupations, spin types, primitive normalization, density matrix) |
| Density grids | gen_dens_grids (2641), add_missing_core_density (131), check_grid_spacing (125), gaussian_functions (151), charge_center_positions_and_parallelpiped (90), generate_kApoints (91), matrix_operations (149) | yes, nonperiodic branches only | Biggest block; the part that decides parity |
| Partition | core_iterator (374), DDEC6_valence_iterator (678), update/initialize_atomic_densities (132), reshaping_functions (499), compute_dominant_atom_volumes (80), oxidation_density (119) | yes | |
| Spin | collinear_spin_moments_iterator (240), spin_functions (195) | yes | Noncollinear skipped |
| Atomic moments | local and total multipole (339), compute_atomic_radial_moments (174), cloud_penetration (51) | yes | Cheap once the partition exists |
| Bond orders | perform_bond_order_analysis, prepare_BO_density_grids, initialize_bond_pair_matrix, integrate_bonding_terms, calculate_final_BOs, compute_local_atomic_exchange_vectors, determine_whether_pair_overlap_is_significant, print_overlap_populations (about 1130) | later phase | Separate PR |
| Skip | DDEC3 iterator, VASP/cube/xsf readers, periodic alignment, noncollinear spin, polarizability upper bound (commented out in chargemol.f08), CM5 (the engine has CM5) | no | |

About 6,000-8,000 Fortran lines are in scope, part of them periodic-only branches [J].

## 3. Design

- `qtaim_gen/source/core/ddec6.py`: the port, in the charge_engine style (numpy plus
  numba kernels on the existing worker pool). Entry point
  `run(wfx_path, compute_bos=False) -> dict`, plus a `ddec6-engine` CLI for standalone
  runs.
- `scripts/gen_chargemol_data.py` -> package data with the c2 and core reference
  densities, shipped as `.npz` package data (D1).
- Output, option B (D3, user 2026-11-03): scalar features in the existing shapes, raw
  vectors and tensors kept beside them so nothing needs a rerun later.
  - charge.json:
    `ddec6: {charge: {"1_C": q}, spin: {"1_C": m}, dipole: {mag, xyz}, atomic_dipole: {"1_C": [x, y, z]}, atomic_quadrupole: {"1_C": [...]}}`.
    `atomic_dipole` follows the existing becke/adch layout ([V] tests/test_files/QTAIM/1/charge.json).
    `atomic_quadrupole` is new; its component convention is set at P4 from
    module_local_multipole_moment_analysis.f08.
  - fuzzy_full.json: per-atom scalars `ddec6_r2`, `ddec6_r3`, `ddec6_r4` (plus `sum`,
    `abs_sum`) and per-atom multipole magnitudes, so they become node features through
    `parse_fuzzy_data`.
  - bond.json: `ddec6_bond` (bond-order phase).
  - [V] Checked 2026-11-03 that extra keys are safe:
    - json-to-lmdb stores charge.json verbatim (utils/lmdbs.py:316-331).
    - `parse_charge_data` reads only `charge`, `spin` and `dipole.mag` (utils/lmdbs.py:590-617).
    - `validate_charge_dict` checks expected schemes and their `charge` (utils/validation.py:564).
    - `lmdb_audit` checks `charge`/`spin` (utils/lmdb_audit.py:153-169).
    - `flatten_entry` skips lists (core/lmdb_analysis.py:201-218).
    - census counts schemes; parse_json reads only `charge`.
    - P5 adds `ddec6` to `validate_charge_dict`'s expected keys at its full_set level.
  - Not chosen now: a separate descriptors.json/descriptors.lmdb (option C). It is the
    planned home once ring-level features (PDI, FLU, ring CPs, aromaticity) or other
    entities that are not atoms or bonds arrive, with a schema designed once for all of
    them.
- Provenance: an engine version stamp in timings.json `_engine_versions` and a `ddec6`
  timing key, as the other engine routines do.
- full_set level: decided after the stage 1 timings (D4).

## 4. Phases

| Phase | Content | Exit criterion |
|---|---|---|
| P0 | Reference harness. Build Chargemol from source with gfortran (serial build for determinism), plus a local precision patch (more digits in the charge, spin and bond-order files, plus optional dumps of intermediate arrays for P2). `scripts/ddec6_reference.py` writes job_control.txt (net charge, nonperiodic, atomic-density path, input orca.wfx), runs it and parses the outputs to `ddec6_ref.json` | Chargemol's shipped Gaussian ozone wfx examples reproduce their bundled outputs; our native orca.wfx (EDF block included) is accepted; timings on the 87 wfx_pull set and the 156 nitrates |
| P1 | Tables. `gen_chargemol_data.py` and a loader | Table values identical to the source files |
| P2 | Density grids | Electron count, pair counts and analytic corrections equal to Chargemol's printed values; valence, spin and core grids equal to the patched dumps within float tolerance on fixtures |
| P3 | Core and DDEC6 valence iterators, giving charges | Charges within 1e-6 e of stock Chargemol, and much tighter against the patched build, on fixtures, then the 87 + 156 sets |
| P4 | Collinear spin moments, atomic multipoles, radial moments | Same bar as P3 for spin; moments at print precision |
| P5 | Wiring: runner routine, full_set level, validation key, json-to-lmdb parse, converter filters, tests and fixtures | Engine run produces `ddec6` in charge.json through the runner |
| P6 | LRC and NERSC campaign (section 5) | Ledger rows recorded |
| P7 | Bond orders (separate PR) | Bond orders vs patched Chargemol |

### P0 results (2026-11-04)

Harness: `scripts/ddec6_reference/` (`build_chargemol.sh`, `chargemol_build_fix.patch`,
`chargemol_precision.patch`, `run_chargemol.py`). Binaries in `~/dev/chargemol_src/bin`
(recipe `-O0`) and `bin_O2`.

- **Build fix needed.** `modulo(k, gridspacing)` at
  module_gen_dens_grids_from_gaussian_basis_set_coeff.f08:1031, 1050, 1069, 1852, 1871
  and 1890 is evaluated even for gridspacing = 0 (analytic pairs), because Fortran does
  not short-circuit `.and.`. Every gfortran run died with SIGFPE. `max(gridspacing, 1)`
  leaves every result unchanged. The port must keep the intended semantics (skip the test
  when gridspacing <= 1).
- **Golden test.** The four bundled Gaussian ozone wfx examples reproduce Chargemol's
  shipped outputs exactly at printed precision. This holds at both `-O0` and `-O2`, for
  charges, multipoles, spins, bond orders, overlap populations (10 decimals), radial
  moments and iteration counts.
- **Reproducibility.** Serial `-O0` and OpenMP `-O2` agree to 7e-13 e on charges and
  5e-13 on bond orders (rmechdb fixture), so a parity bar near 1e-10 is realistic for P3.
- **EDF types.** Multiwfn writes library EDF primitive types as 0 (fileIO.f90:4058,
  written at 7936), and core/wavefunction.py copies that. Chargemol accepts only wfx
  type codes and stops. The harness stages a copy with the types set to 1 (s) and
  records `edf_types_rewritten`. The writer is unchanged.
- **Cost.**
  - Density-grid generation is 91% of runtime (rmechdb: 13.5 of 14.8 s at 16 threads,
    216 x 192 x 168 = 7.0M points for 4 atoms). The iterators are cheap, so P2's kernel
    sets the engine cost.
  - wfx_pull (87 jobs, 2-39 atoms, 16 threads each, 2 jobs at a time): median 117 s,
    max 817 s.
  - Nitrates X2C-SVPall (first 24 jobs, 10-25 atoms): median 529 s.
- **Sum rules.**
  - Charges sum to the net charge within 7.5e-14 e on all 111 jobs.
  - Spin moments are off from N_alpha - N_beta by up to 5.5e-3: 9 of the wfx_pull
    jobs, including two UKS singlets at -5.5e-3 and +4.3e-3. Nitrates are off by
    6.8e-5 at most.
  - Cause not established. Possibly the spin density is not renormalized the way the
    electron count is. To trace in P4; the port reproduces Chargemol either way.

### P2 map: module_gen_dens_grids_from_gaussian_basis_set_coeff.f08 (nonperiodic path)

Read 2026-11-04. Line numbers are in the unmodified source.

| Kernel | Source | What it does | Port notes |
|---|---|---|---|
| K0 primitive order | read_wfx 730-819 | Selection sort with swaps: primitives by centre, then by exponent within the centre; MO coefficients move with them. A single block is a run of equal centre and exponent (1e-10). Blocks then span shells: def2 jobs up to 10 primitives, X2C-TZVPPall actinides up to 60 (pair block 3600 values). rmechdb_652: 111 blocks, as Chargemol reports | Port the swap sequence literally; it fixes pair order and so summation order |
| K1 grid | Step 1, 61-185; charge_center_positions, parallelpiped | Cubic 0.14 Bohr pixels; origin and totnumA/B/C from atom extents plus delta_na (cutoff 500 pm) plus 11 buffer pixels, rounded up to a multiple of 24 by a mixed integer/real expression | Port literally; the rounding expression must match bit for bit |
| K2 density matrices | Step 2, 232-262 | Dense nprim x nprim alpha and beta primitive 1PDMs from occupations and MO coefficients | Build per pair block from C (BLAS); the full matrix is about 7 GB at 30k primitives |
| K3 pair screening | Step 3, 326-457; single blocks from read_wfx | For each (block_i <= block_j, i <= j): (\|P_a\| + \|P_b\|) times the overlap with powers rounded to even numbers, against gaussian_overlap_tolerance. Included pairs are "analytic" (alpha1 min(Z1, 1) + alpha2 min(Z2, 1) > analytic_alpha_cutoff) or "numeric" | Keep the even-rounding rules: down when alpha_sum > 1, else up; same-centre and two-centre cases differ |
| K4 pair data | Steps 4-5, 460-835 | Electron check over all pairs (STOP if off by > 0.001). Per pair: 1PDM sum and difference (x2 off-diagonal), analytic overlap, cutoff radius, product centre and its pixel, renormalization factor = 1, grid spacing 1/2/3/4/6/8/12 by alpha_sum (numeric pairs only). Analytic pairs: alpha^10 weights split their electrons, spin, dipole and quadrupole onto atoms (core_electrons, occupancy_correction rows 2-11) | Needs gaussian_overlap_integral with moment shifts (module_gaussian_functions) |
| K5 sort | Step 6, 837-875 | Within each pair block, selection sort by cutoff radius, descending | Port as written; the point loop EXITs on the first pair whose radius is exceeded |
| K6 point loop | Step 7, 876-1530 (spin) and 1775-2309 (no spin) | Per block, per pixel inside the block cutoff: exp(-alpha_sum d^2) times the common factor times the sum over pairs of polynomial x renormalization factor x 1PDM. Spacing 0 (analytic) pairs go to core_density, with their spin subtracted from occupancy_correction. Numeric pairs go to the fine grid or coarse grids 2-12 (index nint((k+1)/s)) | Two passes. Pass 1 accumulates signed and absolute numeric overlaps (columns 39, 40). Then factor = clip((analytic - signed)/abs, +-0.05), or 0 below tolerance. Pass 2 adds factor x \|polynomial\|. numba kernel; threads need a deterministic reduction |
| K7 interpolation | 1530-1775, 2309-2490 | Coarse grids cascade back to the fine grid by 1/8 sums of the 8 neighbours | Port literally |
| K8 core grid and checks | Step 8, 2496-2620; add_missing_core_density; check_grid_spacing; run_valence_core_densities | EDF/core density on the grid, missing-core fill from core_*.txt, initialize_atomic_densities, electron-count checks | Last P2 piece |

Validation for P2: `chargemol_dump.patch` (chargemol_serial_dump; reader
`scripts/ddec6_reference/read_dumps.py`), a third build-time patch (dump only) writes valence_density,
spin_density, core_density, occupancy_correction, core_electrons and the pair counts to
stream-access binary files after Step 8. The port compares arrays element by element
(about 56 MB per grid for a 7M-point fixture).

## 5. Validation

Reference tiers, as in the ledger:

1. **Chargemol parity.** The engine against Chargemol on the same orca.wfx: stock output
   at 1e-6 e, patched output for the tighter bar. Run the reference serially, because
   OpenMP atomic adds make the parallel build order-dependent [J].
2. **Invariants.**
   - charges sum to the net charge
   - spin moments sum to 2S
   - the grid electron count matches nvalence + ncore (Chargemol's own check,
     run_valence_core_densities)
   - an isolated atom gives charge 0
   - translation by 50-100 Bohr gives the same result (it moves the uniform grid
     relative to the atoms, a real test here)
3. **Physical cross-checks.**
   - DDEC6 against MBIS and Hirshfeld on the same jobs (correlation, outliers listed)
   - ESP reproduction: RRMSE of DDEC6 point charges (plus atomic dipoles) against the
     exact `esp.py` ESP on the vdW surface, compared with the RRMSE Chargemol reports
     where it prints one

Data, local first:
- `tests/test_files/charge_engine/` fixtures
- 87 wfx_pull (61 elements, 25 ECP, 43 open shell)
- 156 actinide nitrates (Th-Pu, X2C all-electron and def2 ECP)
- 93 ORCA 5 Pt4Hx clusters

Campaign (P6):
- **LRC:** tm_react plus two or three OMol verticals (candidates rmechdb, 5A_elytes,
  mo_hydrides), up to 100 jobs per vertical as in earlier campaigns.
- **NERSC:** openac `prod_l1` sample (X2C all-electron across functionals and bases, and
  def2 ECP).
- **Reference side on the clusters:** the Chargemol binary is needed there, built from
  the same patched source. It is used only for validation and never enters the runner.
- **Results:** ledger rows in `validation_ledger.md`; any intentional divergence goes in
  the README changelog.

## 6. Risks

- **Pair renormalization and the coarser grids are the hardest part.** Bit-level parity
  depends on porting summation order and cutoffs exactly. If that fails, the fallback is
  documented tolerance-based parity (D2).
- **Cost.** A uniform 0.14 Bohr grid over a 100-atom molecule is of order 1e7 points.
  Chargemol's own timings in P0 set the target. The numba kernels must at least match
  the serial Fortran [J].
- **X2C all-electron actinides** have very steep primitives. They stress the pair
  cutoffs and renormalization, which is why openac is in the campaign.
- **ORCA-derived wfx.** Spherical-to-Cartesian primitives with Multiwfn's g ordering:
  Chargemol maps wfx type numbers itself, so P0 checks it reads our files identically to
  how the engine does (electron count, dipole).
- **Natural orbitals.** Chargemol's reader looks for natural-orbital keywords
  (module_read_wfx.f08:127, 696-728). P0 confirms it accepts a canonical-orbital wfx
  from our converter. [L] until run.

## 7. Open decisions

- D1. RESOLVED 2026-11-03: `.npz` package data, shipped.
- D2. RESOLVED 2026-11-03: if exact parity with Chargemol's density grids proves
  impractical, pick a reasonable tolerance at P2.
- D3. RESOLVED 2026-11-03: option B (section 3). Scalars go in charge.json and
  fuzzy_full.json; raw atomic dipoles and quadrupoles go under `ddec6` in charge.json.
  Option C (descriptors.json) is deferred to the ring-level features.
- D4. full_set level for ddec6, after the P0 timings.
- D5. RESOLVED 2026-11-03: source and zip copied to ~/dev/chargemol_src.

## References

- T. A. Manz and N. Gabaldon Limas, RSC Adv. 6 (2016) 47771-47801 (DDEC6 part 1).
- N. Gabaldon Limas and T. A. Manz, RSC Adv. 6 (2016) 45727-45747 (DDEC6 part 2).
- T. A. Manz, RSC Adv. 7 (2017) 45552-45581 (DDEC6 part 3, bond orders).
- T. A. Manz and D. S. Sholl, J. Chem. Theory Comput. 7 (2011) 4146-4164 (spin moments).
- T. A. Manz and N. Gabaldon Limas, Chargemol program for performing DDEC analysis,
  Version 3.5, 2017, ddec.sourceforge.net.
