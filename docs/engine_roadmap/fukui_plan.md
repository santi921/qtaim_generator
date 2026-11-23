# Single-point Fukui functions in the charge engine: plan

Date: 2026-11-19. Status: plan only, no code changed.
Branch feat/engine-fukui off origin/feat/charge-engine 0a0523b (engine version 9), worktree
~/dev/qtaim_generator-fukui. Multiwfn source: ~/dev/Multiwfn_src/Multiwfn_3.8_src_Linux (MW/).
Labels: [V] verified in code or a file; [I] inference; [E] effort estimate; [U] unverified.

Scope (user, 2026-11-19): frontier-orbital (FMO) and orbital-weighted (OW) Fukui functions, f- and
f+ and what derives from them, condensed with Hirshfeld, for future campaigns only. No backfill of
existing trees. Finite-difference Fukui (N-1/N/N+1 SCFs) stays a separate campaign.

## Summary

- Every quantity is a linear combination of one matrix: the Hirshfeld atomic composition
  C[A, i] = sum over A's grid of w_H,A(r) * w(r) * |phi_i(r)|^2, for a window of spin-orbitals i
  around the gap. This is one extra orbital evaluation per atom grid in the existing main pass,
  with the Hirshfeld weights the pass already computes.
- f+ needs virtual orbitals, and only the molden from orca_2mkl has them. orca.wfx has occupied
  orbitals only [V]. The molden exists in the job folder from the convert step until clean_jobs
  deletes it [V], and the charge engine runs between those two steps [V]. So conversion is
  unchanged: the engine reads the molden itself.
- Multiwfn has reference implementations for closed shells: CDFT option 6 (condensed OW) and
  orbital composition menu 8 option 8 (Hirshfeld composition per orbital, unrestricted included).
  Open-shell OW has no Multiwfn version (CDFT refuses wfntype /= 0) [V]. Its reference is
  Multiwfn's per-orbital compositions combined with the OW weights in a short script.

## 1. What Multiwfn does [V]

| Quantity | Code | Definition |
|---|---|---|
| condensed OW f+, f-, f0, DD | MW/CDFT.f90:758-866 (option 6) | mu = (e_HOMO + e_LUMO)/2; weight_i = exp(-((mu - e_i)/delta)^2), normalized separately over virtuals (idxLUMO..nmo) and occupied (1..idxHOMO); f+(r) = sum_i weight_i phi_i^2, same for f-; condensed f_A = sum over A's grid of selfdens/promol * w * f(r), points with promol = 0 skipped; f0 = (f+ + f-)/2, DD = f+ - f-; printed f15.5 |
| delta | MW/define.f90:486 | orbwei_delta = 0.1 Hartree by default |
| grid for option 6 | MW/CDFT.f90:799-806 | with iautointgrid=1 (production settings.ini:15): 30 x 302, radcut 15 Bohr |
| restriction | MW/CDFT.f90:741-751 | needs orbital coefficients (mwfn/fch/molden/gms) and wfntype = 0 |
| condensed option 6 orbital range | MW/CDFT.f90:823-830 | all orbitals; the 0.001 weight cut exists only in the grid function orbwei_Fukui (MW/function.f90:6766-6800) |
| Hirshfeld orbital composition | MW/orbcomp.f90:38 (menu 8 option 8), gen_orbatmcomp_space MW/orbcomp.f90:1065 | same weights (built-in free-atom densities via calcatmdens, single-centre integration on each atom's grid), per orbital, both spins |
| HOMO index | MW/sub.f90:4650 getHOMOidx | closed shell: last orbital with occupation 2 |
| degeneracy | MW/CDFT.f90:85-140 | for finite differences only: orbitals within 0.01 eV of the HOMO/LUMO are suggested as degenerate |

The engine's Hirshfeld charges already use exactly this weight and quadrature (pop_h on each
atom's own grid, zero-promolecule points skipped; charge_engine.py run, `_hirshfeld_mwfn`) [V].

## 2. Definitions

Spin-orbital convention (proposed, decision F2). Every orbital is a spin-orbital with an energy
and an occupation. In a closed shell each spatial orbital appears twice, once per spin.

- e_H = highest occupied spin-orbital energy, e_L = lowest unoccupied, mu = (e_H + e_L)/2.
- FMO: f-_A = mean of C[A, i] over occupied spin-orbitals with e_H - e_i <= tol;
  f+_A = mean over unoccupied spin-orbitals with e_i - e_L <= tol. The mean over a degenerate set
  makes the value independent of the arbitrary rotation inside that set.
- OW: f+_A = sum over unoccupied i of weight_i C[A, i] / sum weight_i, with weight_i =
  exp(-((mu - e_i)/delta)^2), and the same over occupied orbitals for f-.
- Derived: f0 = (f+ + f-)/2, dual descriptor DD = f+ - f-.

For a closed shell both definitions reduce exactly to Multiwfn's. Each spatial orbital is
counted twice, and that cancels in the normalization. For an open shell they extend Multiwfn
consistently: f- comes from the electron that is easiest to remove, whichever spin it has, and
f+ from the lowest empty level. The alternative is spin-resolved values (alpha and beta
separately); see F2.

Window: orbitals farther than 0.6 Hartree from mu carry OW weight below exp(-36) = 2.3e-16 of the
frontier orbital's at delta 0.1, so only orbitals with |e_i - mu| <= 0.6 Ha (and the FMO sets) are
evaluated. That matches Multiwfn's all-orbital sum to double precision. The window scales with
delta (6 delta). The count goes in `_meta`.

## 3. Design

### 3a. Reader (core/wavefunction.py)

read_orca_molden gains `orbital_window` (Hartree, default None = current behaviour). With a window
it keeps the orbitals within it of mu for each spin channel, occupied or not. A first pass over
the headers (`Ene=`, `Spin=`, `Occup=`) finds e_H and e_L, so nothing depends on the order of the
orbitals in the file. ORCA writes them in ascending energy within each spin on one local file [V],
but this is not relied on. Coefficients outside the window are never stored. Keeping every virtual
would cost about nmo x nprim x 8 bytes: [I] about 1.4 GB for a 300-atom UKS def2-TZVPD job.

The h-function flip follows the folder's converter mode (fixed by default; multiwfn_h when
provenance says so), so the orbitals match the wfx that the other routines read.

### 3b. Engine (core/charge_engine.py)

- `prepare_basis(wfx, keep=None)`: a mask parameter replaces the hard-coded `occ != 0`, so a
  window basis can hold virtuals. Grouping of the primitives is unchanged.
- `run(..., molden_path=None)`: when Fukui is requested and the molden is given, read it with the
  window, build a window basis, and in the existing per-atom loop, where `h` is already computed,
  add `comp[b] = (h[:, b] * grid_wts) @ orbital_values(pts, coords, wbasis) ** 2` (as built: see
  F4b and F7 in section 6 for the full radial grid, the reuse of the occupied orbital values and
  the truncation). Cost: nwin extra orbital columns per point [I] (planned as 10-40; measured up
  to 642 in the window, 384 after truncation, against tens to hundreds of
  occupied orbitals).
- FMO and OW vectors are formed from comp after the loop; f0 and DD are derived.
- `_meta.fukui`: e_H, e_L, mu, delta, tol, the FMO set sizes and spins, nwin per spin, and the
  sums of f+ and f- (both about 1, the grid error).

### 3c. Output schema

Per-atom entries in fuzzy_full.json, in the `fuzzy_entry` format the converter's `fuzzy_filter`
already reads (per-atom values plus sum and abs_sum) [V]. One routine per key, because the
compile step maps one routine to one top-level key [V] (omol.py fuzzy compile). Proposed names:
`fukui_plus_fmo`, `fukui_minus_fmo`, `fukui_plus_ow`, `fukui_minus_ow`, plus
`fukui_zero_*`/`dual_*` if F3 says to store them.

### 3d. Runner wiring

- `data/multiwfn.py`: a FUKUI_ROUTINES set at the level F1 picks, in ENGINE_ROUTINES and
  `charge_engine_routines`.
- `data/engine_versions.py`: ENGINE_VERSION 10, "added": FUKUI_ROUTINES.
- `omol._run_charge_engine`: locate the molden the same way `_native_convert` does (one
  `*.molden.input`, `orca.molden.input` preferred) and pass it on. With no molden, the routines
  get a permanent skip with reason `no_molden` (`_skipped_routines`). Validation then passes, and
  a restart of an old tree does not loop.
- Hirshfeld skips (no_builtin_proatom, ecp_without_edf) apply to Fukui as to the Hirshfeld family
  (F6).
- Validation (`validate_timing_dict`, `_engine_outputs_present`) and the compile step pick up the
  new routines through the registry. To check: parse_fuzzy_data and json-to-lmdb handle new keys
  generically [U].

### 3e. Edge cases

| Case | Handling |
|---|---|
| degenerate HOMO/LUMO (symmetric molecules, closed-shell alpha/beta pairs) | mean over the set (tol, F5) |
| no beta electrons (mult = nelec + 1, e.g. H atom) | the spin-orbital convention needs no beta HOMO |
| molden with no virtuals | f+ entries NaN plus a `_meta` flag, f- still written [I]: rare, check whether any vertical has it |
| ROKS / ROHF moldens | occupation 1 orbitals are alpha only, and their beta counterpart is empty. Handle it or skip it with a reason. Check whether any vertical uses ROKS [U] |
| artificial zero-energy virtuals with diffuse bases (MW/CDFT.f90:953 comment) | outside the window unless the gap is tiny; record in `_meta` if any orbital in the window has e = 0 exactly |
| ECP | Fukui is a valence quantity. Orbitals only, no EDF involved |

## 4. Validation

| Step | Reference | Data | Bar |
|---|---|---|---|
| V1 closed-shell OW | Multiwfn CDFT option 6, iautointgrid=0, radpot 75, sphpot 434 (CDFT option 6 ignores radcut; menu 8 run with radcut=0, F4b) | XeCl4 fixture (ECP + EDF) + closed-shell wfx_pull jobs, moldens regenerated with local ORCA 6.0.1 orca_2mkl from gbw.zstd0 | print precision (5e-6, f15.5) |
| V2 FMO (open and closed) | Multiwfn menu 8 option 8, same grid, compositions of the frontier spin-orbitals | rmechdb_652 fixture (open shell) + both-shell wfx_pull subset | print precision of the % output |
| V3 open-shell OW | menu 8 option 8 compositions of every window spin-orbital, OW weights applied in a script | open-shell wfx_pull subset | print precision |
| V4 invariants | none | all local jobs | sum f+ = sum f- = 1 to the grid error of the Hirshfeld charges; closed shell: spin-orbital result = spatial result; rotating a degenerate pair (synthetic test) leaves FMO unchanged |
| V5 grid divergence | Multiwfn CDFT option 6 at production settings (iautointgrid=1: 30 x 302) | V1 set | report; changelog row (the engine uses 75 x 434) |
| V6 cluster | V1-V3 references via a campaign script, full runner from scratch | the next batched run: LRC tm_react 200 (116 open shell), NERSC openac sample (X2C, h functions) | as V1-V3; all other routines identical to the version 9 trees |

Local work stays on fixtures and small sets; scale goes to the batched cluster run
(feedback-batch-cluster-validation).

## 5. Phases

| Phase | Content | Effort [E] |
|---|---|---|
| F0 | Reference harness: Multiwfn stdin for CDFT 6 and menu 8 option 8 at a set grid, parsers, local moldens for the fixtures and wfx_pull | 0.5 day |
| F1 | Reader window + `prepare_basis(keep=)`; tests against a full-virtual read | 0.5 day |
| F2 | Engine computation, `_meta`, fixture references (`multiwfn_reference_fukui.json`) | 1 day |
| F3 | Runner wiring, skips, ENGINE_VERSION 10, validation, compile; tests | 1 day |
| F4 | Local V1-V5, ledger rows, divergence changelog entries | 0.5-1 day |
| F5 | Cluster V6 in the next batched run | queue time |

## 6. Decisions

Decided 2026-11-19 (user): F1 full_set 0, F2 spin-orbital, F3 4 keys (`fukui_plus_fmo`,
`fukui_minus_fmo`, `fukui_plus_ow`, `fukui_minus_ow`), F4 main grid, F5 tol 0.01 eV and delta
0.1 Ha, F6 follow the Hirshfeld-family skips. Every recommendation below was taken.

Revised after the first local runs (user, 2026-11-20):

- F4b, radial grid: CDFT option 6 integrates every radial point of the atom grid (no `iradcut`
  skip, MW/CDFT.f90:822), while menu 8, the Hirshfeld charges and the engine drop r >= radcut
  (10 Bohr). Frontier orbitals of charged clusters with diffuse bases reach past 10 Bohr: on
  49 wfx_pull jobs at radcut 10, closed-shell OW differed from CDFT option 6 by up to 8.9e-2
  (F- cluster, charge -3, unbound LUMO at +0.27 Ha) and 12 of 23 exceeded 5e-6; f+ summed to
  0.61-0.95 on such jobs. Decision: the Fukui integrals use the 75 x 434 grid with every radial
  point (the main pass plus the shells beyond radcut, for the Fukui orbitals only); outer
  points where every free-atom density is 0 are skipped (weight 0, as Multiwfn skips them).
- F7, cost: the 0.6 Ha window holds up to 642 orbitals (39-atom water cluster, def2-TZVPD,
  451 occupied), and the first version's Fukui pass took 14.2 s against 9.5 s for the density
  pass. Decision: take the occupied orbital values from the density pass (wfx orbitals, mapped
  by occupied rank and checked by energy), evaluate only the virtuals from the molden, and
  drop the orbitals with the smallest OW shares while they sum to less than 1e-10 per side
  (FUKUI_DROP; 642 -> 384 orbitals there). On that job the charge engine went from 7.2 s
  without Fukui to 11.6 s with it (8 threads, quiet machine).

| # | Question | Options | Recommendation |
|---|---|---|---|
| F1 | Level | full_set 0 / full_set 1 / opt-in `--fukui` flag (like `--ddec6`) | full_set 0: cost is small and future campaigns are mostly level 0. Restarting an old level-0 tree on new code would rerun the charge engine once and then record `no_molden` |
| F2 | Open-shell convention | spin-orbital (one set of keys, reduces to Multiwfn for closed shells) / spin-resolved alpha and beta keys / both | spin-orbital: fixed schema across closed and open shells |
| F3 | Stored keys | f+ and f- only (FMO, OW: 4 keys) / also f0 and DD (8 keys) | 4 keys: f0 and DD are linear combinations the converter can form |
| F4 | Grid | main 75 x 434, radcut 10 (no extra pass, divergence in the changelog) / Multiwfn's option 6 grid 30 x 302, radcut 15 (separate pass) | main grid, as was done for IBSI |
| F5 | Degeneracy tolerance and delta | tol 0.01 eV (Multiwfn's suggestion), delta 0.1 Ha (Multiwfn default) | as stated |
| F6 | Hirshfeld-family skips (ECP without EDF, no built-in proatom) | follow the family / compute anyway (orbital densities do not involve the core) | follow the family, for one consistent rule |

Out of scope: finite-difference Fukui, superdelocalizabilities (MW/CDFT.f90:930; they need every
virtual and are sensitive to diffuse-function artefacts), Fukui potential, global CDFT indices
(mu, eta, omega from orbital energies: derivable from orca.json later), backfill.

## Caveats

- The two-pass header read and the window bound are argued from the OW formula, not yet measured.
- Effort figures assume the existing engine patterns and exclude queue time.
- Whether any production vertical has ROKS references or moldens without virtuals is not checked.
