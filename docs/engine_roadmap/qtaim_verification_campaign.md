# QTAIM engine verification campaign (plan)

Status: proposed 2026-10-08, for approval. Companion to qtaim_engine_plan.md (criteria A-F)
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
  - 2,000 points drawn around each atom at radii 0.01-6 Bohr, log-spaced so they reach
    the nuclear cusps;
  - a molecular box sample.
- Quantities: rho, alpha/beta/spin, gradient, Hessian, G, K, the Laplacian, ELF and LOL
  (both formed from the compared ingredients), and ALIE.
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

## Tier 2: stored regeneration-campaign references at scale

Goal: the P1 properties (and later the CP sets) against production Multiwfn on the real
data distribution, reusing the QTAIM regeneration campaign instead of rerunning Multiwfn.

- References: the CPprop.txt files the regeneration campaign is saving. They are fresh,
  .wfx-based (with EDF) and come from one Multiwfn build.
  - Older stored qtaim.json is not a valid reference. On tm_react, 57% of stored records
    differ from a fresh run: .wfn-era alpha/beta loss, ECP without EDF, mixed builds [V].
- Mechanism: a verification script, `scripts/qtaim_verify_stored.py`, run per job folder.
  1. Regenerate orca.wfx from the gbw in a temporary directory with the production convert
     step (orca_2mkl, then Multiwfn molden -> wfx). Median 0.56 s locally [V].
  2. Load the stored CPprop.txt with load_cpprop_full.
  3. Evaluate qtaim_engine.point_properties at the stored CP positions.
  4. Write qtaim_verify.json: per property and CP type, the max relative difference and
     the worst CP; CP counts, Poincare-Hopf, unpaired BCPs, NCPs without a nucleus.
  5. Delete the temporary wfx. Nothing else in the folder changes.
- Control subset: about 5 jobs per vertical also get a fresh Multiwfn QTAIM run on the
  regenerated wfx. This confirms that a wfx regenerated now reproduces the stored values,
  so the comparison measures the engine, not drift in the conversion.
- Sample: at most 100 jobs per vertical (user, 2026-10-08), stratified with a fixed seed.
  - Bins: atoms (1-20, 20-50, 50-100, 100-200, 200+), multiplicity (1, 2, 3+), ECP/EDF
    present, element class (main group, 3d, 4d/5d, lanthanide/actinide).
  - Rare bins are filled first.
  - Run on each cluster where the regeneration outputs live: LRC, ALCF and LLNL, as their
    campaigns finish.
- Kept wavefunctions for repeatable offline checks: 10 per vertical locally, 100 per
  vertical on LRC.
- Report: scripts/qtaim_verify_report.py aggregates every qtaim_verify.json.
  - Pass rates per stratum.
  - The worst cases per property.
  - A triage list split into: print-limited, EDF or ECP, near-degenerate Hessian
    (ellipticity or eta), conversion drift (from the control subset), and genuine.
- Acceptance: criterion A on 100% of jobs. Gradients are judged against the position
  rounding bound |delta g| <= ||H|| x 1e-12, not a fixed threshold.
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
