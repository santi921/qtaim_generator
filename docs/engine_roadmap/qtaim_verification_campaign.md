# QTAIM engine verification campaign (plan)

Status: tiers 1 and 2 running since 2026-10-08 (results below). Companion to qtaim_engine_plan.md (criteria A-F)
and README.md (parity across levels of theory).

Tags: [V] verified, [I] inference, [E] estimate.

## Summary

- Three tiers:
  1. Full-precision kernel checks against independent code, locally. No text rounding.
  2. In-folder comparison against fresh Multiwfn at scale on LRC, stratified across
     verticals.
  3. A level-of-theory matrix computed for this purpose.
- Tiers 1 and 2 start now and cover the P1 point properties. The CP-set, pairing,
  Poincare-Hopf and determinism criteria (B-E) reuse the same samples once the P3 search
  lands.
- The current comparison is limited by Multiwfn's printed output, not by either code. [V]
  - Positions are printed with 12 decimals in Bohr. CPs.txt has only 6 (MW/topology.f90:588).
  - Values are printed as E18.10.
  - At Multiwfn's printed CP positions every P1 property agrees to 4.8e-10 relative on 5
    jobs and all four CP types.
  - The nuclear-CP gradient differs by up to 1.3e-3 absolute. That is 5e-13 Bohr of
    position rounding times Hessian eigenvalues of about 1e9 at a Br nucleus.

## Tier 1: full-precision kernels (local)

Goal: verify the engine's density kernels to about 1e-13 relative, independent of
Multiwfn's print format.

- Reference implementation: HORTON in the existing `horton` conda env.
  - iodata reads the same orca.wfx.
  - gbasis evaluates rho, its gradient and Hessian, the kinetic energy densities and the
    MO values at arbitrary points.
  - HORTON does not apply the EDF.
- EDF terms: checked separately against the closed-form s-Gaussian expressions, plus a
  finite-difference consistency test. The finite-difference check already passes:
  derivatives agree to 1e-7 relative, the step-size limit, through g functions. [V]
- Points per job:
  - Multiwfn's CP positions;
  - 40 points around each atom at radii 0.01-6 Bohr, log-spaced so they reach the nuclear
    cusps, in random directions;
  - 500 points uniform in the molecular box (plus 3 Bohr).
- Quantities compared (`scripts/qtaim_horton_check.py`): rho, alpha and beta densities,
  gradient, Hessian, Laplacian, G, and K through the identity K = G - lap/4. ELF, LOL, ALIE,
  spin and G_xyz are not compared here; they rest on the tier 2 comparison with Multiwfn's
  printed values only (code review, 2026-10-09).
- Known limitation: for ROKS wavefunctions the alpha/beta split on the HORTON side is
  wrong (iodata loads them as restricted, occupations 2/1, and the script halves them);
  total rho is unaffected.
- Jobs: every local wavefunction with a wfx.
  - The 87 wfx_pull files (25 with EDF; transition metals, lanthanides, multiplicities up
    to 11).
  - The 2 fixtures.
  - The cross_validation_wfns edge cases from 110 to 342 atoms.
- Acceptance [E]: at most 1e-12 relative wherever |value| > 1e-10, otherwise 1e-20
  absolute. Every failure gets a written explanation.
- Optional, only if tier 1 shows discrepancies we cannot explain: build a patched local
  Multiwfn that prints E25.16 in CPprop.txt, making Multiwfn itself a full-precision
  reference. Its build requirements are unverified.

## Results so far (2026-10-09)

- Tier 1, 93 of 100 local wavefunctions (2-274 atoms, 25 with EDF, about 140,000
  points) [V]:
  - Median agreement with HORTON: rho 7.5e-14, gradient 3.4e-13, Hessian 1.7e-12,
    Laplacian 2.8e-12, G 4.0e-13 relative.
  - EDF gradient and Hessian match central differences to 2.7e-7, the step-size limit.
  - One outlier: a 17-atom Hf ECP job, 3.2e-7 relative in rho at 0.07-0.16 Bohr from
    the Hf nucleus.
    - There the valence-only density is about 1e-6 and comes from cancelling
      tight-primitive terms.
    - An 80-bit extended-precision evaluation matches the engine to 7.5e-14 to 1e-12
      and HORTON to only 1.2e-7 to 3.2e-7, so HORTON loses precision there, not the
      engine.
  - The patched high-precision Multiwfn is not needed.
- Tier 1, the remaining 7 wavefunctions (147-342 atoms, 6,380-14,180 points each, run
  with chunked HORTON evaluation) [V]: worst point over each job, relative, rho
  1.2e-13 to 1.8e-12, gradient 7.1e-13 to 3.8e-12, Hessian 4.5e-12 to 1.4e-11, Laplacian
  5.1e-12 to 6.3e-11. These maxima exceed the 1e-12 acceptance figure for rho on 2 jobs
  and for the Laplacian on all 7; the pattern (growing with size, largest where the
  Laplacian's terms cancel) points to summation roundoff over thousands of primitives
  rather than a kernel error [I], which an extended-precision check on one job would settle.
- Tier 2 on LRC, 1500 sampled jobs (5A_elytes, rgd_uks, tm_react; 2026-10-09) [V]:
  1432 compared, no engine disagreement beyond print precision; 2 stored CPprop.txt
  files corrupt (one NUL-spliced, one truncated; their qtaim.json are intact); 3 BCP
  Laplacians at 3.2e-8 to 1.5e-7 relative, unconfirmed as position rounding [I]; 68 jobs
  lost to a parser bug (Fortran three-digit exponents), since fixed. The summary was
  NaN-blind until the 2026-10-09 review fix; the NaN recount is pending.
  Control (15 jobs, fresh Multiwfn on the regenerated wfx): identical CP sets, position
  differences 0.
- P0 cost ladder, local Multiwfn, 4 threads, 12-274 atoms [V]:
  - The ESP is 60-78% of the QTAIM step at every size.
  - The CP search grows from 0.2 s to 620 s and is 19-31% of the step above 100 atoms.
  - All 8 runs satisfy Poincare-Hopf with no unpaired BCPs.
  - Conclusion: the ESP kernel (P2) is the largest saving; the search (P3) is second
    and needs the O(N) seeding for large jobs.

## Tier 2: stored regeneration-campaign references at scale

Goal: the P1 properties (and later the CP sets) against production Multiwfn on the real
data distribution, reusing the QTAIM regeneration campaign instead of rerunning Multiwfn.

- References: the CPprop.txt files the regeneration campaign is saving. They are fresh,
  .wfx-based (with EDF) and come from one Multiwfn build.
  - Older stored qtaim.json is not a valid reference. On tm_react, 57% of stored records
    differ from a fresh run: .wfn-era alpha/beta loss, ECP without EDF, mixed builds [V].
- Mechanism: a verification script, `scripts/qtaim_verify_stored.py`, over a list of job
  folders. It only reads them; all work happens in a temporary directory.
  1. Regenerate orca.wfx from the gbw in a temporary directory with the production convert
     step (orca_2mkl, then Multiwfn molden -> wfx). Median 0.56 s locally [V].
  2. Load the stored CPprop.txt with load_cpprop_full.
  3. Evaluate qtaim_engine.point_properties at the stored CP positions.
  4. Append one JSON record per job to `--out`: per property and CP type, the max
     difference (a NaN or a property missing from CPprop.txt reports inf); CP counts,
     Poincare-Hopf, unpaired BCPs, NCPs without a nucleus. A stored CPprop.txt that fails
     `validation.cpprop_integrity` is recorded as `corrupt_reference` and not compared.
  5. Delete the temporary directory. Jobs already in `--out` are skipped, so a requeued
     task resumes; each subprocess has a timeout.
- Control subset: about 5 jobs per vertical also get a fresh Multiwfn QTAIM run on the
  regenerated wfx. This confirms that a wfx regenerated now reproduces the stored values,
  so the comparison measures the engine, not drift in the conversion.
- Sample: at most 100 jobs per vertical (user, 2026-10-08), stratified with a fixed seed;
  500 per vertical on LRC, which has idle compute (user, 2026-10-09).
  - Bins (`scripts/qtaim_verify_sample.py`): nuclear-CP count from the stored CPprop.txt
    (1-20, 21-50, 51-100, 101-200, 200+) and multiplicity (1, 2, 3+), drawn round-robin
    across bins so small bins are not crowded out.
  - Not binned yet: ECP/EDF presence and element class (main group, 3d, 4d/5d,
    lanthanide/actinide). Until they are, tier 2 does not guarantee EDF or heavy-element
    coverage (code review, 2026-10-09).
  - Run on each cluster where the regeneration outputs live: LRC, ALCF and LLNL, as their
    campaigns finish.
- Kept wavefunctions for repeatable offline checks: 10 per vertical locally, 100 per
  vertical on LRC.
- Report (planned, not written yet): scripts/qtaim_verify_report.py aggregates the JSONL
  records.
  - Pass rates per stratum.
  - The worst cases per property.
  - A triage list split into: print-limited, EDF or ECP, near-degenerate Hessian
    (ellipticity or eta), conversion drift (from the control subset), and genuine.
- Acceptance: criterion A on 100% of jobs. Gradients are judged against the position
  rounding bound |delta g| <= ||H||_2 x 8.7e-13 + 5e-11 |g| + 1e-20 (positions printed to
  1e-12 Bohr per coordinate, values to E18.10), not a fixed threshold.
- Cost [E]: per job, a conversion plus engine point properties, seconds. The control
  subset adds production QTAIM runs, a few core-hours.

## Tier 3: levels of theory (OpenActinides first)

Goal: parity beyond the single OMol25 production level (README.md requirement).

- First set: the OpenActinides benchmark. Its DFT is already done, as a sweep over
  functionals, basis sets, relativistic treatments and dispersion corrections. Run
  Multiwfn (production QTAIM, charges, fuzzy, ALIE) on a chunk of it, then the engines and
  HORTON, and compare.
  - This covers heavy elements, scalar-relativistic effects and large or high-angular
    basis sets. These are the places where kernels and EDF handling are most likely to
    break.
  - Expected limits [I]:
    - Multiwfn refuses MBIS above Z=86 (the engine copies that), so there are no MBIS
      references for actinides.
    - ECP actinide runs need EDF core densities. A local uranium ECP complex without
      EDF lost a nuclear CP and had all bonds unpaired [V]. Whether Multiwfn's built-in
      library covers every actinide/ECP combination in the sweep is unverified.
    - Dispersion corrections added after the SCF (D3/D4) leave the density unchanged, so
      those runs can be deduplicated. Non-local dispersion inside the SCF (VV10) changes
      it and stays.
    - The engine supports up to h functions (Multiwfn primitive type 56). Basis sets
      with i functions would need an extension.
- Not covered by OpenActinides: organics, main group, 3d/4d metals, lanthanides. A small
  supplementary ORCA set, or samples from the existing verticals, fills those later.
- Results go in a parity table per setting.

## Later criteria (after P3)

On the tier 1 and tier 2 samples:

- B: CP-set superset with positions within 1e-5 Bohr.
- C: BCP pairs.
- D: Poincare-Hopf.
- E: byte-identical output at 1, 4 and 8 threads.
- F: speed on the same nodes.

Extra CPs need independent confirmation: Multiwfn searching from the engine's points,
critic2 where installed. Ring and cage CPs from the regeneration campaign's saved
CPprop.txt files add Multiwfn references at scale. Matching those needs the wavefunction,
which the in-folder mode provides.

## Implementation pieces

1. scripts/qtaim_horton_check.py (horton env; tier 1).
2. scripts/qtaim_verify_stored.py: regenerate the wfx from the gbw, compare against the
   stored CPprop.txt, write qtaim_verify.json (tier 2).
3. A stratified sampler and Slurm array per cluster (tier 2), following
   data/engine_test_2026-10.
4. A driver for the OpenActinides chunk: Multiwfn, engines and HORTON per setting
   (tier 3).
5. scripts/qtaim_verify_report.py.

## Decisions (2026-10-08)

- Sample: at most 100 jobs per vertical.
- Patched high-precision Multiwfn: only if tier 1 shows discrepancies we cannot explain.
- Kept wavefunctions: 10 per vertical locally, 100 per vertical on LRC.
- Tier 2 uses the regeneration campaign's stored CPprop.txt, not fresh Multiwfn runs (a
  small fresh control subset only).
- Tier 3 starts with OpenActinides.

## Open questions

- Where the regeneration campaign's CPprop.txt files are stored per cluster (job folder
  or out_files.zip), and which Multiwfn build produced them.
- Which chunk of OpenActinides to run first, and where its wavefunctions live.
