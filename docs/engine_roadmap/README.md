# Engine roadmap: replacing Multiwfn with a native descriptor engine

Planning branch `plan/descriptor-engine`. Status as of 2026-10-08: the charge engine
(`core/charge_engine.py`, full_set 0 and 1 charge/fuzzy/bond routines) and the surface
engine (`core/surface_engine.py`, other_alie) are done on `feat/charge-engine`. They
reproduce Multiwfn 3.8 to print precision on 191 fresh LRC tm_react jobs and run 11.3x
faster per job (median, same hardware, full_set 1).

## Documents

| File | Scope |
|---|---|
| qtaim_engine_plan.md | QTAIM engine: CP search, properties, bond paths, ESP, validation, phases |
| remaining_routines_plan.md | every other Multiwfn / orca_2mkl step, by cost; wavefunction reader; EDF tables |
| descriptor_engine_vision.md | descriptors Multiwfn has that we do not use, descriptors beyond Multiwfn, standalone engine sketch, shortlist |

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

Still open: see the "Open decisions" section of each plan; the resolved ones are marked there.

## Cross-cutting requirements

### Divergence changelog

Each intentional difference from Multiwfn (a fixed bug, a different grid, extra CPs, a
new convention) gets an entry: what Multiwfn does (source file:line), what the engine
does, which descriptors change and by how much on the validation set, and the engine
version that introduced it. The changelog ships with each dataset release and paper.
Seed entries already known: bader spin integrated as Shannon/Fisher under ispecial=1
(verified in data/omol_test_spin_skip/orca6_uks/bader.out), IBSI on the 30x110 grid,
QTAIM CPs Multiwfn misses (superset), ring/cage CPs (currently discarded by the parser).

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
