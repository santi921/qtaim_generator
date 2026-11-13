# Engine roadmap: replacing Multiwfn with a native descriptor engine

Planning branch `plan/descriptor-engine`. Status as of 2026-10-08: the charge engine
(`core/charge_engine.py`, full_set 0 and 1 charge/fuzzy/bond routines) and the surface
engine (`core/surface_engine.py`, other_alie) are done on `feat/charge-engine`. They
reproduce Multiwfn 3.8 to print precision on 191 fresh LRC tm_react jobs and run 11.3x
faster per job (median, same hardware, full_set 1).

Since 2026-11-03 the wavefunction conversion is native as well: `full-runner-parsl-alcf`
writes orca.wfx with `core/wavefunction.py` by default (`--native_convert fixed`). orca_2mkl
still writes the molden. `--native_convert off` runs Multiwfn's convert step, and is needed
to continue a tree Multiwfn converted (prod_l1, existing OMol trees): with a native mode,
validation requires a native converter record (provenance.json), and folders converted
another way are discarded and recomputed when a gbw is there. With `--wfn` the default is
`off`. With the charge and QTAIM engines, a full_set 0 run makes no Multiwfn call.

## Documents

| File | Scope |
|---|---|
| qtaim_engine_plan.md | QTAIM engine: CP search, properties, bond paths, ESP, validation, phases |
| remaining_routines_plan.md | every other Multiwfn / orca_2mkl step, by cost; wavefunction reader; EDF tables |
| descriptor_engine_vision.md | descriptors Multiwfn has that we do not use, descriptors beyond Multiwfn, standalone engine sketch, shortlist |
| validation_ledger.md | every engine validation run: reference, data, worst difference, code, raw output; known gaps |

## Decisions (2026-10-08)

| # | Question | Decision |
|---|---|---|
| 1 | Copy Multiwfn's quirks (missed CPs, bader spin under ispecial=1, the 30x110 IBSI grid) or fix them | Fix them. Do not carry the issues over; future campaigns regenerate. Every divergence from Multiwfn is recorded in a changelog released with publications (see below) |
| 2 | Native wavefunction reader and EDF tables | Later, unless it becomes blocking. The user is asking Tian Lu about redistributing Multiwfn's EDF / ECP core-density tables |
| 3 | Reader: orca_2mkl + molden, or orca_2json | Both, one as fallback when the other fails on a given gbw. The gbw format is ORCA-major-version specific (6.0.1 tools fail on ORCA 5 gbw), so tools are chosen by the ORCA version that wrote the job |
| 4 | QTAIM: Multiwfn-faithful or superset | Superset: every CP Multiwfn finds, plus the ones it misses, flagged. Ring and cage CPs are found mathematically and kept, as a QTAIM phase-two task |
| 5 | full_set 2 (bader etc.) | Produced in few places; some calculations exist. Bader engine deferred |
| 6 | Parity bar for descriptors with no Multiwfn analogue | Pull reference values from HORTON, PySCF and other implementations for the descriptors we find compelling |
| - | Density-based steric descriptors | Pinned: must be included (descriptor_engine_vision.md, shortlist) |
| - | gbw availability | All gbw files are on the HPC systems, so gbw-based descriptors can be backfilled |

## Decisions (2026-10-12, QTAIM P2)

| # | Question | Decision |
|---|---|---|
| 7 | delta_g_promolecular (QTAIM plan D2): drop or port Multiwfn's STO table | Drop. Its values differ between Multiwfn builds |
| 8 | Keep the ESP fields at CPs as features (QTAIM plan D4) | Keep |
| 9 | Boys function for the ESP kernel | Exact by default. A copy of Multiwfn's `Fmch` (series cut at 1e-8, sqrt(pi)/2 truncated to 0.88622692) exists only for tests; the difference goes in the divergence changelog |
| 10 | Weak CPs (QTAIM plan D6) | Revised 2026-10-12: Poincare-Hopf is checked (and reseeding triggered) on the CPs with rho >= 1e-5, because near-vacuum CPs below that are found or missed depending on where seeds land (LRC case 40_Li+: kept set sums to 1, full set to 0); both sums go in `_meta`. Originally: check Poincare-Hopf on the full CP set first, then drop CPs with rho < 1e-5 and flag rho < 1e-3 in `_meta` (list of CP keys; the per-CP schema is unchanged). Whether flagged CPs become graph edges is a converter setting. Local data (97 jobs, 6,563 CPs): none below 1e-5, 141 (2.1%) between 1e-5 and 1e-3 |
| 11 | esp_nuc, esp_total, delta_g_hirsh at nuclear CPs | Keep, for parity, although they are set by where the search stopped near the nucleus |

Production ESP at CPs comes from Multiwfn's old `eleesp1` routine, not libreta: the QTAIM
stdin never initializes libreta, so `eleesp` falls back (MW/function.f90:4269-4280).
Verified locally: iESPcode 1 and 2 give byte-identical CPprop.txt.

Still open: see the "Open decisions" section of each plan; the resolved ones are marked there.

## Engine versions and provenance (2026-11-11)

One number, `ENGINE_VERSION` (`data/engine_versions.py`), covers the charge, surface and
QTAIM engines (the native converter is identified by its mode). `ENGINE_CHANGES` lists per version the routines it
added and the routines whose stored values it changed. Every job folder has a
`provenance.json` (`utils/provenance.py`, plan in `provenance_plan.md`): per routine, what
produced it (an engine, Multiwfn with its build, or ORCA with its parser version), the engine
version and its settings; the wavefunction converter; permanent skips; and the engines' run
records (the charge engine's `_meta`, the surface engine's grids and failure counts).
`json-to-lmdb` converts it to `provenance.lmdb`.

A stored routine is current when its recorded version is at least the last version that
added or changed it. Validation, the restart checks and the fuzzy recheck use that rule, so
a folder from an older engine still needs, and keeps, its Multiwfn values for routines added
later, and a release that changes a routine's values recomputes it. Rule: a new routine goes
in `added` under a new version; a change to stored values goes in `changed`.

`timings.json` holds timings only (and `_timings_patched`, which describes them). Engine
folders written before 2026-11-11 kept their records in timings.json (`_engine_versions`,
`_surface_engine_routines`, `_converter`, `_skipped_routines`); `migrate-provenance` rewrites
them (dry run by default) and must run on those trees before their next restart. Their
entries are marked `legacy`. Value changes before the first stamp (2026-11-03) have no version
of their own (converged MBIS as the default, 2026-10-15; planarity in the surface engine,
2026-11-02).

| engine version | change |
|---|---|
| 1 | all runs before the stamp: charges, fuzzy density and spin, fuzzy_bond (full_set 0); vdd, mbis, mbis fuzzy (full_set 1); ALIE and planarity (surface engine); QTAIM engine |
| 2 | adds elf_fuzzy (full_set 1), grad_norm_rho_fuzzy and laplacian_rho_fuzzy (full_set 2); EDF-dependent schemes skipped when any ECP atom lacks EDF (was: when the wavefunction has no EDF at all; no local wavefunction differs, 552 checked) |
| 3 | adds other_esp (full_set 2, surface engine, exact ESP kernel) |
| 4 | one ENGINE_VERSION and provenance.json; no routine values change. An ESP surface failure keeps ALIE and planarity, is retried once, then recorded as a permanent skip |
| 5 | changes mbis, mbis_fuzzy_density, mbis_fuzzy_spin: MBIS selection objective over every atom, stalled check fits continue from their state, shell merging in the plain fallback, ties within 1e-8 relative go to a converged fit, fuzzy refit from the charge fit (mbis_converged.md) |

## Cross-cutting requirements

### Divergence changelog

Each intentional difference from Multiwfn (a fixed bug, a different grid, extra CPs, a
new convention) gets an entry: what Multiwfn does (source file:line), what the engine
does, which descriptors change and by how much on the validation set, and the engine
version that introduced it. The changelog ships with each dataset release and paper.
Seed entries already known: ESP at CPs uses an exact Boys function, where Multiwfn's
eleesp1 uses Fmch (series cut at 1e-8, sqrt(pi)/2 truncated to 0.88622692): esp_e differs by
3.8e-9 to 7.4e-9 relative on 97 local jobs (qtaim_engine_plan.md section 6b); delta_g_promolecular
is no longer produced; bader spin integrated as Shannon/Fisher under ispecial=1
(verified in data/omol_test_spin_skip/orca6_uks/bader.out), IBSI on the 30x110 grid,
QTAIM CPs Multiwfn misses (superset), ring/cage CPs (currently discarded by the parser).
Heavy all-electron nuclei: Multiwfn relaxes findcp's gradient criterion to |g| < 1 for
nuclear seeds (MW/topology.f90:1018, "for very heavy atoms"), but at an X2C uranium or
thorium nucleus (Hessian about -3e15) Newton stops within 5e-15 Bohr with |g| about 16
left by roundoff, so the seed runs out of cycles and the actinide NCP is lost: no atomic
descriptors for it, every actinide-ligand BCP unpaired, Poincare-Hopf 0. The cause is the
spacing of doubles at the nuclear coordinate: a nucleus within about 0.1 Bohr of the origin
converges (every production nitrate has its actinide there), one 18-50 Bohr away does not.
In the engine modes such a seed is rerun with its nucleus at the origin (the molecule
translated; status 3, `nucleus_frame` in the nuclei stage stats), and its properties are
evaluated in that frame. It converges there to |g| about 1e-13, with the density and
Laplacian of the unshifted molecule to 4e-16 relative at shifts up to 100 Bohr. The earlier
rescue (accept the seed when its last step was below dispconv and its Hessian is negative
definite: status 2, `precision_limited`) stays as the last resort. Seeds that converge
normally are unchanged. Local X2C jobs: 5 of 9 U/Th jobs had lost the NCP; all 9 now have
it through the nucleus frame, with Poincare-Hopf 1 and every actinide BCP paired.
MBIS (mbis_converged.md): converged to the fixed point instead of Multiwfn's 1e-4 e per-cycle
stop (population.f90 MBIS loop; 3 of 27 local jobs moved by more than 0.03 e, actinides
0.03-0.38 e); initial guess from a free-atom table fitted to the Hirshfeld proatoms instead of
Multiwfn's shell rule; of several fixed points, the one with the smallest KL divergence among
the converged, non-degenerate fits from three guesses; an atom's shells that stall at nearly
equal widths are merged and the smaller model refit, so such atoms report fewer shells
(cations and electropositive metals); Z 87-118 supported (6 shells), where Multiwfn refuses beyond Rn.
Planarity (other_geometry, surface_engine.planarity): a molecule with no heavy atoms gets
mpp_heavy = sdp_heavy = 0; Multiwfn fits a plane to an empty selection and prints MPP NaN
and SDP 0 (otherfunc.f90 calcMPP). Values otherwise match Multiwfn at print precision
(validation_ledger.md section 3).
Wavefunction conversion (`--native_convert`, core/wavefunction.py): ORCA normalizes the
h(+3,-3,+4,-4) functions to -1, so their coefficients change sign. Multiwfn's readmolden does
this for alpha orbitals only (fileIO.f90:4613-4615; the f and g branches flip both spins);
the converter flips both spins, which ORCA's orca_plot confirms. It changes open-shell jobs
with h functions in the basis (the X2C-*all actinide bases). On the 440-job openac sample:
charges up to 9.8e-4 e (in four structures; every other job 5e-5 e or less), CP density up to
6.4e-3 relative, Laplacian up to 1.9e-2 a.u. at actinide CPs (validation_ledger.md section
4b). `multiwfn_h` mode reproduces Multiwfn for parity checks.
ESP surface (other_esp, full_set 2, surface_engine.esp_surface; engine version 3): production
Multiwfn evaluates the ESP at the surface vertices with libreta (iESPcode=2 in settings.ini),
whose pair screening and single-precision 2 pi constant were measured on 2026-10-12 to cost up
to 4.8e-5 relative in esp_e (branch feat/qtaim-libreta-esp, docs/LIBRETA_ESP.md). The engine
evaluates it exactly (core/esp.py, decision 9). The surface mesh is identical (vertex, edge and
facet counts on 99 of 99 local jobs) and the geometry fields (volume, total area, density) agree
at print precision. The ESP fields move: on 99 local jobs (87 OMol wfx_pull, 12 nitrates) the
surface extremes by a median 2.2e-3 / 3.8e-3 kcal/mol and at most 0.49 kcal/mol, the average by
at most 7.4e-4 a.u., the positive and negative areas by at most 3.7 Bohr^2 (facets near ESP = 0
change sign), the polarity index by 2.0e-2 eV and the skewness by 4.0e-2. With Multiwfn's exact
code (iESPcode=1, eleesp1) and the engine on Multiwfn's Boys function, both fixtures agree on
every field at print precision; the attribution of the larger set's differences to libreta rests
on those two and the earlier measurement [I] (validation_ledger.md section 3).

### Parity across levels of theory

All engine validation so far uses one level of theory (the OMol25 production setup).
Parity must also be established against Multiwfn, HORTON, PySCF or other references on
a matrix of settings before the engine is used beyond it:

- functionals: GGA, hybrid, range-separated hybrid, double hybrid (relaxed densities if
  used), HF
- basis sets: minimal to quadruple zeta, with and without diffuse functions, spherical
  and Cartesian, high angular momentum (g, h)
- core treatment: all-electron, small- and large-core ECPs with EDF, relativistic
  all-electron (DKH/ZORA) if in scope
- references: RKS, UKS, ROKS; singlets, high-spin open shells, broken-symmetry
- ORCA 5 and ORCA 6 wavefunctions; other codes (fchk, molden) once the reader exists

A small fixed molecule set (organics, a TM complex, a lanthanide, an anion, a radical)
computed at each setting gives the benchmark; acceptance stays print-precision parity
with Multiwfn where Multiwfn is the reference, and documented tolerances otherwise.

### Data being collected now

The QTAIM regeneration campaign saves CPprop.txt files, so ring and cage CPs can be
reparsed later, and those files are large-scale Multiwfn references for validating the
QTAIM engine's CP sets.
