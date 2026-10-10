# QTAIM engine: robust, accurate, scalable replacement for Multiwfn main function 2

Date: 2026-10-08. Repo: /home/santiagovargas/dev/qtaim_generator (branch feat/charge-engine). MW/ = ~/dev/Multiwfn_src/Multiwfn_3.8_src_Linux. Its topology.f90 is identical to the 2026.10.1 copy after stripping CRs. Companions: remaining_routines_plan.md, descriptor_engine_vision.md.

Tags: [V] verified by reading or running, [I] inference, [E] estimate, [U] unverified.

## Summary

- Most of the QTAIM step's time goes to the ESP, not the CP search. [V]
  - On 4 local jobs at 4 threads, the electronic ESP that Multiwfn writes into CPprop.txt is 58-78% of the step's wall time.
  - The seeding is combinatorial. A 342-atom job ran about 75k Newton searches, and its triad and quad stages found only 12 new CPs between them.
  - [E] Delaunay seeds, batched Newton and a shell-pair ESP kernel should make the step about 10x faster on mid-size jobs. Large jobs should gain far more; today they cannot finish inside 5 h.
- "Robust" has to cover five things production does not check today:
  - every nucleus has a nuclear CP (NCP), using the EDF core density for ECP atoms;
  - every bond CP (BCP) is assigned an atom pair;
  - the Poincare-Hopf (PH) relation is satisfied;
  - the CP set does not depend on which seeds happened to converge;
  - the output does not change with thread count or between runs.
  - [V] Production never runs Multiwfn's PH check: it is reachable only from menu options 0/00 (MW/topology.f90:2766, 904-906).
  - [V] The parser throws away ring and cage CPs (parse_qtaim.py:27-30, 201-202).
- Proposed meaning of "accurate", in three layers. Choosing between this and a faithful copy of Multiwfn is decision D1.
  - At Multiwfn's own CP positions, the engine's property values match Multiwfn to print precision.
  - The engine's CP set contains Multiwfn's default set.
  - Every extra CP the engine finds is confirmed by an independent run, not adopted on trust. We should not reproduce Multiwfn's misses.

## 1. What production does today

### 1a. Stdin and code path [V]

`qtaim_data()` (data/multiwfn.py:144-152):
- Default: `2 2 3 4 5 8 7 0 -10 q`. This enters the topology module and seeds CP searches from nuclei, pair midpoints, triangle centers and pyramid centers. `8` then traces bond paths, and `7 0` writes every CP's properties to CPprop.txt.
- Exhaustive (`--exhaustive_qtaim`) adds `6 -1 -9`. This scatters 1000 random points in a 3-Bohr sphere around each nucleus (MW/topology.f90:30-31, 1357).
- The input is written at omol.py:315-319 and CPprop.txt is parsed at omol.py:888-927.

Settings in effect:
- The real-space function stays rho, because production never selects option -11.
- Defaults (MW/define.f90:526-527, 536-537):

| Parameter | Value |
|---|---|
| gradconv | 1e-6 |
| dispconv | 1e-7 |
| minicpdis | 0.03 Bohr |
| vdwsumcrit | 1.5 |
| singularcrit | 5e-22 |
| trust radius | none |
| max cycles | 120 |

- If an ECP atom's wfx carries no EDF block, Multiwfn supplies the core density from its built-in library (MW/fileIO.f90:173-192).

### 1b. Algorithm [V]

| Step | Code (MW/topology.f90) | Behaviour |
|---|---|---|
| Seeds | 1013-1305 | Nuclei, then every pair, triad and quad whose pairwise distances are all <= 1.5 x the vdW sum (1073, 1156, 1224). No other locality limit, so the count grows like N k^3 |
| Nuclear seeds | 1018 | Gradient criterion switched off (gradconv = 1) |
| Newton | findcp 2080-2273 | disp = -H^-1 g, analytic Hessian including EDF (MW/function.f90:2552-2580). Stops on \|det H\| < 5e-22. Converged when \|disp\| < 1e-7 and \|g\| < 1e-6 (2200). No check that the result lies near its seed |
| Dedup and typing | 2210-2226 | A CP within 0.03 Bohr of an existing one is dropped, first come first served. Type = number of positive eigenvalues. No rho floor |
| Ordering | sortCP 2279 | Each seed stage is sorted by a linear projection of position, so the order is stable only while the CP set is |
| Bond paths | findpath 1903-2021 | RK2, step 0.03 Bohr, at most 451 points. Stops within 0.05 Bohr of *any* CP (1964). Aborts if rho decreases (1999) |
| Atom mapping | 2351-2392 | NCP = within 0.15 A of a nucleus. A BCP gets "Connected atoms" only if both path ends are NCPs mapped to atoms |

Local runs (~/dev/Multiwfn_3_8/Multiwfn_noGUI) [V]:
- Default and exhaustive gave the same CP set, with PH satisfied, on 5 jobs from 4 to 39 atoms.
- The 39-atom K+ job gave identical CPprop.txt at 1 and at 4 threads.
- Doc 07 (docs/plans/2026-07-27-neurips-r1-07-cross-package-validation-plan.md:494) found 1 job in 87 where only the exhaustive search finds a real BCP (HCNKrF+, rho 0.288).

### 1c. Properties per CP and what they need [V]

showptprop (MW/sub.f90:2347-2510) writes about 30 quantities. The parser keeps 26 of them (parse_qtaim.py:33-85).

| qtaim.json key(s) | Multiwfn definition | Needs |
|---|---|---|
| density_all | rho, including EDF | MO values + EDF |
| density_alpha, density_beta, spin_density | (rho +- spin)/2 | alpha/beta occupations |
| Lagrangian_K (G) | Lagkin, MOs only (function.f90:2795) | MO gradients |
| Hamiltonian_K (K); energy_density = -K | Hamkin, MOs only (2817) | MO Laplacians |
| lap_e_density, lap_norm, eig_hess | Laplacian including EDF. eig_hess is the *sum* of the eigenvalues (parse_qtaim.py:118) | rho Hessian |
| e_loc_func, lol | Becke ELF/LOL, MOs only. Closed-shell or spin-polarized formula chosen by wfntype (3109-3215) | tau, MO gradients |
| ave_loc_ion_E | avglocion (3406) | MO energies (already used in surface_engine.py:58-75) |
| delta_g_promolecular | IGM over an STO-fit promolecule (3920-3990) | STO table (not ported) |
| delta_g_hirsh | IGMH over interpolated proatoms (4043) | multiwfn_atmraddens tables (ported) + their derivative |
| esp_nuc | sum Zeff/r; set to 1000 exactly at a nucleus (4250-4262) | coordinates |
| esp_e | libreta eleesp2, no EDF (MW/libreta_hybrid/libreta.f90:29-46) | nuclear-attraction integrals |
| esp_total | sum of the two | - |
| grad_norm | gradient norm at the converged point, i.e. the convergence residual | - |
| det_hessian, ellip_e_dens, eta | det H, l1/l2 - 1, \|l1\|/l3 | Hessian eigenvalues |

Redundant keys:
- lap_e_density = lap_norm = eig_hess;
- energy_density = -Hamiltonian_K;
- esp_total = esp_nuc + esp_e.

cp_num is dropped before features are built (utils/lmdbs.py:666, 676).

### 1d. Engine infrastructure today [V]

The engine already has:
- a wfx reader that keeps the EDF block (charge_engine.py:55-97);
- primitive grouping by (center, exponent) (:100-136);
- value-only primitive and EDF kernels (:162-217);
- a Z-order-blocked GEMM in a worker pool with single-threaded BLAS (:220-279).

It has no derivatives. Every part of QTAIM needs primitive first and second derivatives, the same kernel as P0 in remaining_routines_plan.md.

## 2. Known problems, and what "robust" means concretely

| Problem | Evidence | Requirement |
|---|---|---|
| BCPs lost: no attributable pair, two BCPs for one pair, unmatched attractor | 18 / 2 / 1 of 28 residual cases; tolerance of 2 (validation.storable_bcp_count and DEFAULT_BCP_TOLERANCE; regen plan section 3) | Every BCP gets a pair from a robust path tracer. Same-pair duplicates are kept |
| Search misses real BCPs | HCNKrF+; 6 S-O/S-F bonds lost in a shipped K+ record (doc 07:494-495) | Completeness checks + adaptive reseeding |
| ECP atom without core density: missing or phantom NCP, det_hessian up to 1e21 | NOTE_2026-10-05_fuzzy_bond_singlets.md:80-84. Local UCl6: 6 NCPs for 7 atoms, all 6 BCPs unpaired, PH = 0 [V] | Require EDF, or inject it from ported tables (D5) |
| Open-shell .wfn read as all-alpha | tracker #28; 557,098 records fixed (data/rerun_lists_2026-09-29/STATUS.md:404-407) | wfx only; spin from MO types |
| NCPs filed under a nearby H atom | 38,409 folders (STATUS.md:393-403) | NCP-to-atom map must be a bijection, asserted inside the engine |
| Killed runs, missing provenance | qtaim_run_status (validation.py:574-631) | Atomic write + a `_meta` block |
| delta_g_promolecular changes between Multiwfn builds | All 2024-Oct vs 2025-Jun pairs differ (STATUS.md:124-130). Dropped in 9c375ac, which is not on this branch | D2 |
| cp_num instability | ignored by commit 4e6d8d1 | Canonical order (section 4) |
| Weak-CP ill-conditioning | ellipticity/eta diverge across codes below rho ~0.05 (doc 07:446-470) | Flag near-degenerate CPs |

[V] A PH scan of all 54 local CPprop.txt files: 38 satisfy PH. The 16 failures are truncated edge cases, broken-wfx proteins and uranium ECP complexes. This sample is biased toward known edge cases.

## 3. Timing

[V] 2,797 local timings.json files have a 'qtaim' key. They come from mixed clusters and thread counts.
- QTAIM wall time: median 16.3 s, mean 264 s, p90 362 s, max 36,852 s.
- Share of total summed step time: 15.1% (bader excluded). For comparison, 'other' is 21.4% and the charge/fuzzy steps about 49%.
- Median share by size: 2.4% (1-20 atoms), 4.4% (20-50), 36.9% (50-100), 23.6% (100-200), 23.7% (200+).
- [I] Once the charge and surface engines are in production, QTAIM is about 94% of the Multiwfn time left at full_set 0.

[V] Local Multiwfn, 4 threads, cumulative wall time in seconds:

| Job (atoms, primitives) | Search | + paths | + props, no ESP | Full | Exhaustive |
|---|---|---|---|---|---|
| mo_hydrides 5383 (12, 700) | 0.13 | 0.22 | 0.24 | 0.64 | 4.4 |
| tm_react Mn3, mult 5 (26, 1075) | 1.06 | 1.46 | 1.58 | 3.79 | 19.6 |
| 5A_elytes K+ (39, 2529) | 4.61 | 6.72 | 7.30 | 32.8 | 85.5 |
| 5A_elytes Cs+, EDF (39, 2384) | 2.02 | 7.76 | 8.72 | 39.9 | 142.8 |

- ESP is 58-78% of the full step. Multiwfn's own prompt calls ESP "the most expensive one" (topology.f90:1463).
- The other costs are the search and bond paths. The exhaustive variant costs 2.6-7x the default run in total.
- Seed counts, for pairs / triads / quads:
  - K+ job: 282 / 916 / 1,671 seeds, finding 66 / 6 / 0 new CPs.
  - 342-atom data/edge_cases/0831/2123_C3H8O: 4,076 / 19,285 / 51,781 seeds, finding 577 / 12 / 0.
- [I] Superlinear growth comes from the quad seeds and from the ESP, which is about O(N) CPs times O(N^2) primitive pairs.
- [V] The engine's value-only kernel takes 10.2 us per point on 1 thread (K+, 203 occupied MOs). [E] Value + gradient + Hessian should be about 8-10x that.

[I] Which part dominates changes with size: at 12-39 atoms the ESP at CPs is most of the step, but seed counts grow superlinearly (75k searches at 342 atoms), so CP search may dominate large jobs, as the user expects. P0 measures search, paths, properties and ESP separately across size bins up to 350 atoms before the design is fixed, and the CP-search algorithm is planned first.

## 4. Design (core/qtaim_engine.py, CLI `qtaim-engine`)

**Kernels**
- `orbital_derivatives`: 10 columns per primitive (value, gradient, Hessian) in the existing block/screen/GEMM path. It yields MO value, gradient and Hessian, which give rho, its gradient and Hessian, tau, and the MO Laplacian. EDF gets analytic s-Gaussian derivatives.
- ESP:
  - McMurchie-Davidson nuclear-attraction integrals over shell pairs;
  - a tabulated Boys function up to n = 10 (h functions are the limit, charge_engine.py:110-111);
  - shell pairs screened by exp(-mu R^2) max\|P\|;
  - loop over pairs outside and CPs inside;
  - no EDF term and Zeff nuclear charges, as Multiwfn does.
- Proatom radial derivatives (Lagrange interpolation, as in lagintpol) for delta_g_hirsh.

**Seeding**, in canonical order:
- nuclei;
- edge, face and tetrahedron centroids of the Delaunay triangulation of the nuclei (scipy), skipping edges longer than 1.5 x the vdW sum;
- three points along each vdW-close atom pair (0.3, 0.5, 0.7 of the way).
- [I] This is O(N) seeds instead of O(N k^3).
- A `--multiwfn_seeds` mode reproduces Multiwfn's seed sets exactly, for validation only.

**Newton**
- All seeds iterate together as one batch.
- 0.5 Bohr trust radius.
- Eigenvector-following when the Hessian is indefinite.
- Convergence criteria as Multiwfn, plus 2 polishing iterations.
- No gradient criterion for nuclear seeds, as Multiwfn.

**Deduplication**
- 0.03 Bohr spatial hash.
- A deterministic representative: lowest \|g\|, then lexicographic position.
- CP pairs closer than 0.1 Bohr are flagged as near-catastrophe.

**Completeness**
- Required:
  - the NCP count equals the number of atoms plus non-nuclear attractors (NNAs);
  - the NCP-to-nucleus map is a bijection within 0.15 A;
  - n - b + r - c = 1.
- On failure, up to 3 reseed rounds:
  - deterministic Sobol points in 1.5-3 Bohr spheres around atoms with a missing or unpaired CP;
  - seeds inside bond-path rings that have no RCP, and inside cages that have no CCP.
- Optional cross-check against the promolecular topology [I].
- Anything still failing goes into `_meta.flags`.

**Bond paths**
- Batched adaptive RK4 (step 0.01-0.1 Bohr) from each BCP along +-v(l3).
- A path ends when it enters a nuclear capture radius [E: 0.2-0.5 Bohr] or comes within 0.05 Bohr of an NCP or NNA.
- Maximum length 30 Bohr.
- A drop in rho halves the step instead of aborting the path.
- [I] This removes Multiwfn's three no-pair modes: the 451-step cap, ending at a foreign CP, and the rho-drop abort.

**Properties**
- One batched pass over all CPs.
- ELF/LOL follow Multiwfn's wfntype rules, with ELF_addminimal = 1 (MW/define.f90:483).

**Determinism**
- Canonical order: NCPs by atom; BCPs by sorted pair, then position; RCPs and CCPs by nearest-atom tuple, then position rounded to 1e-6 Bohr. cp_num is assigned afterwards.
- [I] Blocking can change BLAS summation order, giving differences around 1e-16. The tolerances above absorb them.
- Tested at 1, 4 and 8 threads. Fallback: a fixed-order numba reduction in the final pass.

**Output**
- qtaim.json in the current schema, written atomically and directly.
- An optional qtaim_cp.json holding RCPs, CCPs, NNAs and duplicate BCPs (D3).
- A `qtaim_engine` timing key, gated in validation like charge_engine (validation.py:265-293).

**Scale**
- CPU first: numba plus the BLAS worker pool; one job per core for throughput. Later, multipoles for distant shell pairs.
- GPU only in P6. It needs FP64 (rho reaches 1e7 at EDF nuclei; Boys function). A long-lived worker per GPU should batch many jobs, since [I] a 50-atom job's ~1e5 point evaluations underfill a GPU. [U] The workstation's A5000s have weak FP64, so the targets are cluster A100/H100 cards.
- [E] Targets on the same hardware: 3-10x for search plus paths, 10-30x for ESP; 10x or more for the full step at 20-50 atoms, more than 30x at 200+ atoms.

## 5. Accuracy: criteria and validation

| Level | Acceptance (proposal) |
|---|---|
| A. Kernels | At Multiwfn's printed CP positions (12 decimals, Bohr), every parsed field matches to 1e-9 relative or 1e-12 absolute, as the E18.10 print allows. Exceptions: ellip_e_dens and eta to 1e-6 absolute, unless the two lowest eigenvalues are near-degenerate (flagged); grad_norm both below 1e-8. Holds on 100% of references, including ECP+EDF, lanthanides and multiplicity up to 11 |
| B. CP set | Contains Multiwfn's default set on 100% of jobs (type match, position within 1e-5 Bohr; median below 1e-8). Every extra CP is confirmed by Multiwfn option 1-4, which searches from a points file (topology.f90:913-1002), or by critic2 |
| C. Pairs | Same "Connected atoms" for every BCP Multiwfn pairs; pairs for at least 99% of the BCPs it leaves unpaired; reasons for the rest |
| D. PH | Satisfied on 100% of jobs with an intact wfx |
| E. Determinism | Byte-identical output across thread counts and repeat runs |
| F. Speed | Faster than Multiwfn in every size bin on the same hardware; target 10x [E] |

Validation set:
1. References: the 87 wfx_pull jobs (critic2.json sidecars exist), the 2 charge_engine fixtures and the rest of data/cross_validation_wfns. Each gets local Multiwfn default and exhaustive CPprop.txt, plus timings with and without ESP (`7 -1`).
2. Synthetic cases (benzene, cubane, Li clusters, H-bond dimers, near-catastrophe geometries) from ORCA at ~/dev/orca5/orca [U].
3. An LRC batch (data/engine_test_2026-10 runbook) against *fresh* Multiwfn runs. Stored QTAIM differs from fresh in 57% of tm_react records (memory note, 2026-10-06).
4. Stress: the 200-350 atom jobs in data/edge_cases/0831 [U: wfx present].

## 6. Phases [E]

| Phase | Content | Effort |
|---|---|---|
| P0 | Reference harness: Multiwfn runs, CPprop loader, comparison report, fixtures; cost split (search / paths / properties / ESP) per size bin up to 350 atoms | 3-4 days |
| P1 | Derivative kernel; rho, gradient, Hessian, tau, ELF, LOL, G, K, ALIE and spin at given points; criterion A without ESP and delta-g | 1-1.5 weeks |
| P2 | ESP kernel, delta_g_hirsh, D2; criterion A complete | 1.5-2 weeks |
| P3 | Seeds, batched Newton, dedup, ordering, bond paths, atom mapping; criteria B, C, E | 2 weeks |
| P4 | Completeness layer, NNAs, EDF policy, flags; LRC validation; critic2 adjudication | 1.5-2 weeks |
| P5 | Wiring into gbw_analysis, validation and restart gates, tests | 1 week |
| P6 | Optional GPU and multi-job batching | 2-4 weeks |

P0-P5, i.e. production on CPU: about 7-9 weeks.

## 7. Risks

- The ESP must match libreta to about 1e-10, including high angular momentum and large Boys arguments. This is the largest piece of new numerics.
- Properties at ECP nuclear CPs are extreme. One Tl NCP in tests/test_files/lmdb_tests/orca5_uks/qtaim.json has rho 9.6e6, Laplacian -1.9e15 and det -2.6e44 [V]. Tiny position differences get amplified, so criterion A is scored at Multiwfn's positions.
- Near-degenerate topology (a BCP close to an RCP, flat vdW regions) is ill-posed. We flag these cases; we do not force a match.
- A different CP set means different graph edges. The no-mixed-engine rule applies: new verticals or full regeneration only.
- Delaunay seeds could miss CPs. Keep the Multiwfn-compatible seeds plus reseeding until P3 measurements show they can go.

## 8. Open decisions

- D1. RESOLVED 2026-10-08: robust superset. Every CP Multiwfn finds plus the ones it misses, flagged; divergences go in the release changelog (README.md).
- D2. Drop delta_g_promolecular, or port Multiwfn 3.8's STO table?
- D3. RESOLVED 2026-10-08 for ring and cage CPs: keep them, found mathematically, as a phase-two QTAIM task. Storage (qtaim.json or sidecar), NNAs and duplicate same-pair BCPs still open. The regeneration campaign is saving CPprop.txt now, so Multiwfn's RCP/CCP can be reparsed for validation.
- D4. Keep the ESP fields (most of Multiwfn's cost) as features?
- D5. Pending: the user is asking Tian Lu whether Multiwfn's EDF tables may be redistributed. Until then, wfx files keep supplying the EDF.
- D6. Weak-CP policy: keep everything, flag rho < 1e-3, or discard rho < 1e-5 as critic2 does?
- D7. Invest in GPU, and on which clusters?
- D8. Is the `_meta` block enough provenance for --require_qtaim_provenance?

## Caveats

- Stage timings come from 4 jobs on one workstation. The size-bin shares use NCP counts from qtaim.json.
- The PH scan covers 54 files, mostly edge cases.
- The internals of the STO fit, libreta and edflib were located, not read.
- All effort figures are estimates.
