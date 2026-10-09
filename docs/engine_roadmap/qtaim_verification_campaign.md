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

## Tier 2: in-folder comparison at scale (LRC)

Goal: the P1 properties (and later the CP sets) against fresh production Multiwfn, on the
real data distribution, without keeping wfx files.

- Mechanism: a new runner flag, `--qtaim_verify`.
  1. In each job folder, after convert has produced orca.wfx, run production Multiwfn
     QTAIM as usual.
  2. Load its CPprop.txt with load_cpprop_full.
  3. Evaluate qtaim_engine.point_properties at Multiwfn's CP positions.
  4. Write generator/qtaim_verify.json: per property and CP type, the max relative
     difference and the worst CP; CP counts, Poincare-Hopf, unpaired BCPs, NCPs without
     a nucleus; Multiwfn and engine timings.
  - The wfx is cleaned as usual, so only the small JSON stays.
- Sample (stratified, fixed seed), starting with the LRC verticals: tm_react, rgd_uks,
  5A_elytes.
  - Bins: atoms (1-20, 20-50, 50-100, 100-200, 200+), multiplicity (1, 2, 3+),
    ECP/EDF present, element class (main group, 3d, 4d/5d, lanthanide/actinide).
  - [E] 2,000 jobs per vertical, plus every job in the rare bins (200+ atoms, actinides,
    multiplicity 6+).
  - Same Slurm layout as data/engine_test_2026-10 (new MODE=qverify).
- Then ALCF and LLNL verticals with the same flag. Their sizes are a user decision.
- Report: scripts/qtaim_verify_report.py aggregates every qtaim_verify.json.
  - Pass rates per stratum.
  - The worst cases per property.
  - A triage list split into: print-limited (as above), EDF or ECP, near-degenerate
    Hessian (ellipticity or eta), and genuine.
- Acceptance: criterion A on 100% of jobs. Gradients are judged against the position
  rounding bound |delta g| <= ||H|| x 1e-12, not a fixed threshold.
- Cost [E]:
  - Production QTAIM has a median of 16 s and a long tail. Over 2,797 local runs the max
    was 36,852 s, and ESP is 60-78% of the step at 12-39 atoms [V].
  - With a 5 h per-job timeout, 6,000 jobs at 4 cores need about 300-600 core-hours,
    most of it in the 200+ atom bin.
  - The engine side adds seconds per job.

## Tier 3: levels of theory

Goal: parity beyond the single OMol25 production level (README.md requirement).

- Molecule set, 8-10 molecules: an organic, an anion, a radical, a 3d complex
  (high-spin and low-spin), a 4d/5d complex with an ECP, a lanthanide with an ECP, a
  noble-gas compound, and a hydrogen-bonded dimer.
- Matrix in ORCA:
  - functionals: GGA, hybrid, range-separated hybrid, HF;
  - basis sets: def2-SVP, def2-TZVP, def2-TZVPD, def2-QZVPP (g/h functions);
  - all-electron vs ECP;
  - RKS / UKS / ROKS;
  - ORCA 5 and ORCA 6.
  - [E] About 150 single points, minutes each locally.
- Every point runs local Multiwfn (production QTAIM and charges), the engines and HORTON;
  PySCF where its readers allow.
- Results go in a parity table per setting. Settings that are out of scope (relativistic
  all-electron, correlated relaxed densities) are listed as such.

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
2. `--qtaim_verify` in gbw_analysis and the runners, and qtaim_verify.json (tier 2).
3. MODE=qverify plus a stratified sampler in data/engine_test_2026-10 (tier 2).
4. ORCA inputs for the level-of-theory matrix (tier 3).
5. scripts/qtaim_verify_report.py.

## Open decisions

- Sample size per vertical, and when to include the ALCF and LLNL verticals.
- Whether to build the patched high-precision Multiwfn now, or only if tier 1 needs it.
- The exact tier 3 matrix (functionals, basis sets), and whether correlated or
  relativistic densities are ever in scope.
- Whether a small set of wfx files (for example 200 per vertical) should be kept for
  repeatable offline checks.
