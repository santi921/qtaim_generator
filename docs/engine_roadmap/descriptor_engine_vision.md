# Descriptor engine vision: replacing Multiwfn for molecular descriptors

Date: 2026-10-08. Status: research and ideation only, no code changed.
Scope: molecular (non-periodic) descriptors from a single-point ORCA wavefunction. No plotting, no PBC, no visualization.

## Summary

- The existing engine (charge_engine.py, surface_engine.py) already builds the expensive parts: MO values on atom grids, Becke/Hirshfeld/MBIS weights, atomic overlap matrices (AOMs) and an isosurface mesh. About ten unused Multiwfn descriptors cost almost nothing on top of these: atomic multipoles, Tkatchenko-Scheffler volumes/polarizabilities/C6, localization indices, the full DI matrix, ring aromaticity indices, and per-atom surface statistics.
- The real limit is the input file, not the numerics. The Multiwfn-written .wfx has no contracted basis, and the fixture checked has no virtual orbitals. Cleanup deletes the wfx when a gbw exists. Reading the gbw directly (orca_2json is in the local ORCA 6.0.1) would remove both orca_2mkl and Multiwfn conversion, and would unlock all basis-space indices and frontier-orbital (LUMO) descriptors.
- Removing Multiwfn completely needs three new building blocks. (1) ECP core densities (EDF), which come from Multiwfn's library today. (2) Analytic density gradients and Hessians for QTAIM critical-point search. (3) One-electron integrals for ESP. Of these, the EDF tables are the hard one: a data and licensing question more than a coding one.

Evidence labels used below: [V] verified in code or a file (path:line), [J] judgment, [I] inference from verified facts, [L] low confidence.

## 1. What the generator computes today

| Source | Routine (full_set level) | Level | Fields kept | Where |
|---|---|---|---|---|
| Multiwfn / engine | hirshfeld, adch, cm5, becke(ADC) (0) | atom + global | charge per atom, molecular dipole (mag, xyz) | [V] data/multiwfn.py:41-44; utils/lmdbs.py:544-593 |
| Multiwfn / engine | vdd, mbis (1); chelpg (1, Multiwfn only); bader basin charges (2) | atom | charge (+ spin where printed) | [V] data/multiwfn.py:46-52 |
| Multiwfn / engine | becke/hirsh fuzzy density and spin (0); mbis fuzzy density/spin, elf_fuzzy (1); laplacian, grad_norm fuzzy (2) | atom + global (sum, abs_sum) | integrated function per atom | [V] data/multiwfn.py:98-117; utils/lmdbs.py:596-628 |
| Multiwfn / engine | fuzzy_bond (0), ibsi_bond (1), laplacian_bond (2) | bond | FBO (= fuzzy DI, pairs >= 0.05), IBSI, LBO | [V] data/multiwfn.py:72-78; core/charge_engine.py:706-736 |
| Multiwfn | qtaim: CP search from nuclei, pairs, triangles, pyramids; paths; CP properties | atom (NCP) + bond (BCP) | rho (total/alpha/beta/spin), LOL, ELF, E(r), G(r), K(r), Laplacian, ALIE, delta-g (promol, Hirshfeld), ESP (nuc, el, total), gradient, Hessian eigenvalues and determinant, ellipticity, eta | [V] data/multiwfn.py:144-152; core/parse_qtaim.py:33-85 |
| Multiwfn | qtaim ring/cage CPs | - | discarded | [V] core/parse_qtaim.py:27-30 |
| Multiwfn | other_geometry (0) | global | MPP and SDP (all atoms, heavy atoms) | [V] data/multiwfn.py:136; core/parse_multiwfn.py:902-930 |
| Multiwfn / engine | other_alie (0), other_esp (2) | global | rho=0.001 surface: volume, M/V density, min/max, areas, average, variances, nu, Pi, MPI, polar/nonpolar area, skewness | [V] data/multiwfn.py:137-140; core/parse_multiwfn.py:932-1019 |
| ORCA .out | Mulliken, Loewdin, Mayer (NA, QA, VA, BVA, FA), Hirshfeld, MBIS (charges, populations, spins, valence pops and widths) | atom | merged into charge.json as *_orca | [V] core/parse_orca.py:31-48, 1026-1055 |
| ORCA .out | Mayer and Loewdin bond orders | bond | merged into bond.json | [V] core/parse_orca.py:1080-1085 |
| ORCA .out | energies and components, HOMO/LUMO (alpha/beta), S^2, dipole, quadrupole, rotational constants, gradient stats, SCF convergence, warnings | global | default orca_filter subset | [V] utils/lmdbs.py:950-960 |

Notes:
- [V] The engine currently covers all full_set 0 charge/fuzzy/bond routines plus vdd, mbis, mbis_fuzzy_* and other_alie (data/multiwfn.py:11-18). Still Multiwfn-only: qtaim, other_geometry, other_esp, chelpg, bader, ibsi_bond, laplacian_bond, elf/laplacian/grad_norm fuzzy, and the molden-to-wfx conversion (core/omol.py:377-379).
- [V] Per-job time at full_set 0 before the engine: engine routines 53-68%, other_alie 20-30%, qtaim 4-11% (docs/plans/2026-10-04-feat-one-pass-charge-engine-plan.md:177). So once ALIE is replaced, qtaim matters because it is the last dependency, not because it is slow.
- [V] The other_geometry input string also runs geometry option 8 (molecular diameter, length/width/height), but the parser keeps only MPP/SDP (data/multiwfn.py:136; core/parse_multiwfn.py:902-930).

## 2. Multiwfn descriptor features we do not use

Survey of the Multiwfn 3.8 source (~/dev/Multiwfn_src/Multiwfn_3.8_src_Linux). Line numbers are menu entries. No Multiwfn manual PDF exists on this machine (searched ~/dev, ~ and /), so method details below come from the source and general knowledge.

Input codes: W = wfx alone is enough (occupied MOs + primitives + EDF). B = needs the contracted basis and/or virtual orbitals (gbw/molden/fch). X = needs extra QM calculations. Cost is [J], relative to one existing engine density pass (1x).

### 2a. Atomic charges and populations

| Feature | Source | Level | Input | Cost | ML value [J] |
|---|---|---|---|---|---|
| Hirshfeld-I | population.f90:58 | atom | W | 5-20x (iterative, needs ion proatoms) | medium: near-redundant with MBIS, which we have |
| Merz-Kollman, RESP, CHELPG | population.f90:56,61 | atom | W (ESP on many points) | high (ESP integrals) | low to medium: ill-conditioned for buried atoms, conformer-noisy |
| EEM, Gasteiger | population.f90:60,62 | atom | geometry only | trivial | low: cheminformatics-level, the model can learn it |
| Mulliken, Loewdin, SCPA, Stout-Politzer, Bickelhaupt | population.f90:47-51 | atom | B | trivial once S, P are in hand | low: ORCA already prints Mulliken and Loewdin |
| cMBIS, EMBIS, AEMBIS (2026.10.1 only) | 2026.10.1 population.f90 menu 21-23 | atom | W | ~MBIS | [L] unknown; new methods, little benchmark history |

### 2b. Bond orders and delocalization

| Feature | Source | Level | Input | Cost | ML value [J] |
|---|---|---|---|---|---|
| Mayer, Wiberg (Loewdin basis), Mulliken BO | bondorder.f90:18,22,23 | bond | B | trivial | medium: Mayer/Loewdin already come from ORCA but are thresholded |
| Multicenter bond order (3-6 centers), AV1245, AVmin | bondorder.f90:19,30 | ring / 3-body | B | cheap for 3-6 rings, combinatorial if all rings are enumerated | medium for aromatic and metallacycle verticals |
| Occupancy-perturbed Mayer BO | bondorder.f90:25 | bond | B | cheap | low |
| Bond order density / NAdO | bondorder.f90:31 | bond | B | moderate | low (mainly a visualization tool) |
| Localization index (LI) per atom | fuzzy.f90:161 | atom | W, from the AOMs we already build | ~0 | high: electrons kept on the atom, complements the charge |
| Full DI matrix (no 0.05 cutoff) | fuzzy.f90:161 | atom pair | W | ~0 | medium-high: gives long-range and 1,3 delocalization for edge features |
| Multi-center DI (MCI) | fuzzy.f90:170 | ring | W, closed shell only per the source comment | cheap for small rings | medium |
| CLRK, PLR (linear response) | fuzzy.f90:167-168 | atom pair, ring | B (needs virtuals, per the source comment) | moderate | medium |

### 2c. Atomic multipoles, volumes, polarizabilities

| Feature | Source | Level | Input | Cost | ML value [J] |
|---|---|---|---|---|---|
| Atomic dipole, quadrupole, <r^2> under the current partition | fuzzy.f90:158 | atom + global | W | ~0 (same grid, extra moment weights) | high: charge-only models miss anisotropy; standard input for electrostatics |
| Effective/free atomic volume, TS polarizability and C6, molecular C6 | fuzzy.f90:172, 1110 (C6 = C6_free (V_eff/V_free)^2) | atom + global | W | ~0 (Hirshfeld weights times r^3) | high: physically motivated dispersion and polarizability per atom |
| Atomic polarizability in a molecule (finite field) | hyper_polar.f90:2401 | atom | X (wfx under several external fields) | 6-7 extra SCFs | high value, but out of scope for a single point |
| Basin multipoles (QTAIM) | basin.f90:51 | atom | W, basin grid | high (basin integration) | medium: redundant with fuzzy multipoles |

### 2d. Conceptual DFT and reactivity

| Feature | Source | Level | Input | Cost | ML value [J] |
|---|---|---|---|---|---|
| Finite-difference Fukui f+, f-, f0, dual descriptor (condensed), local softness and electrophilicity | CDFT.f90:60-64, 573-587 | atom + global | X (N+1 and N-1 SCFs at fixed geometry, N-2 for w_cubic) | 2-3 extra SCFs | high, but it is a new-calculation campaign |
| Orbital-weighted Fukui and dual descriptor | CDFT.f90:72 | atom | B (needs virtuals) | cheap | medium-high, and single-point |
| Nucleophilic/electrophilic superdelocalizabilities | CDFT.f90:74 | atom | B | cheap | medium |
| Global CDFT (mu, eta, omega from HOMO/LUMO) | CDFT.f90 | global | ORCA HOMO/LUMO already parsed | trivial | low: linear functions of features we already have |

### 2e. Aromaticity

| Feature | Source | Level | Input | Cost | ML value [J] |
|---|---|---|---|---|---|
| PDI, FLU, FLU-pi | fuzzy.f90:163-165 | ring | W (DI) | ~0 once rings are perceived | medium for aromatic verticals, none elsewhere |
| HOMA, Bird | deloc_aromat.f90:17 | ring | geometry only | trivial | low-medium (geometric) |
| ITA aromaticity, Shannon index at RCPs | fuzzy.f90:171; topology.f90:90 | ring | W | cheap | low |
| ELF/LOL sigma-pi, LOLIPOP | deloc_aromat.f90:16; otherfunc.f90:23 | ring | W, needs sigma/pi orbital separation | moderate | low (planar systems only) |
| RCP properties | deloc_aromat.f90:24 | ring | W, needs ring CPs | ~0 if CP search keeps them | medium: we drop these today |
| NICS, ICSS | deloc_aromat.f90:15 | ring | X (NMR shielding calculation) | expensive | out of scope |

### 2f. Electron localization, basins and domains

| Feature | Source | Level | Input | Cost | ML value [J] |
|---|---|---|---|---|---|
| ELF basin populations (core, V(A), V(A,B), lone pairs) with labels | basin.f90:64 | atom / bond / lone pair | W | high (grid basin assignment) | medium-high for lone-pair and hypervalent chemistry, but basins do not map cleanly to graph nodes |
| AIM basin LI/DI and integrated properties | basin.f90:52 | atom / pair | W | high | medium: fuzzy versions are cheaper and nearly as good [J] |
| Domain analysis (properties inside isosurfaces) | otherfunc2.f90:21 | global | W | moderate | low |
| Core-valence bifurcation (CVB) index | otherfunc2.f90:8 | global / bond | W | moderate | low-medium (H-bond strength) |

### 2g. Weak interactions

| Feature | Source | Level | Input | Cost | ML value [J] |
|---|---|---|---|---|---|
| NCI / RDG, sign(lambda2)rho | visweak.f90:14 | grid (global histograms) | W or promolecular | moderate | low for graph features unless binned per atom pair |
| IGM / IGMH delta-g per atom and per pair | visweak.f90:21 | atom, pair | W (IGMH) or geometry (IGM) | moderate | medium: we already keep delta-g at CPs and IBSI |
| IRI | visweak.f90:17; function.f90:7542 | grid / CP | W | moderate | low-medium |
| Becke/Hirshfeld surfaces of fragments | visweak.f90:20; surfana.f90:103 | fragment | W | moderate | low for single molecules, useful for complexes |

### 2h. Surfaces, orbital composition, spin, other

| Feature | Source | Level | Input | Cost | ML value [J] |
|---|---|---|---|---|---|
| Per-atom surface properties (area, min/max/mean of the mapped function per atom) | surfana.f90:1370 | atom | W (ALIE), W+integrals (ESP) | ~0 on top of the existing mesh for ALIE | high: global ALIE/ESP stats become node features |
| LEA, LEAE, EDR, D(r) mapped surfaces | surfana.f90:190-199 | global / atom | B (LEA needs virtuals) | moderate | medium (LEA) |
| Orbital composition (Hirshfeld, Becke, Mulliken, NAO) | orbcomp.f90:30-41 | atom x orbital | W for occupied, B for virtuals | cheap | high when restricted to HOMO/LUMO (Section 3) |
| LOBA / mLOBA oxidation state | orbcomp.f90:42 | atom | B (localized orbitals) | moderate (Pipek-Mezey or Boys first) | high for TM and lanthanide verticals [J] |
| Atomic and bond dipoles in Hilbert space | otherfunc2.f90:9 | atom, bond | B | cheap | low-medium |
| Energy index, bond polarity index | otherfunc2.f90:19 | bond | B plus reference calculations [L] | moderate | low |
| Hellmann-Feynman forces | otherfunc.f90:28 | atom | W | moderate | low (ORCA gives true gradients) |
| Molecular diameter, L/W/H, vdW volume (MC or marching tetrahedra) | Multiwfn.f90:578, 584 | global | geometry or W | trivial / ~0 | low-medium |

## 3. Descriptors beyond Multiwfn (single-point, molecular)

[J] unless marked. All are computable from one wavefunction plus geometry unless flagged X.

1. Alternative stockholder partitions. ISA, GISA, LISA, Hirshfeld-I and DDEC6-like charges. HORTON already provides becke, hirshfeld and "is" (core/horton.py:21) but drops the EDF on ECP atoms (core/horton.py:32-48) [V]. In the engine they all reuse the same molecular grid and density. MBIS already covers most of this family's value; GISA/LISA add little except robustness on diffuse anions [J]. DDEC6 adds spherical-averaging steps and is a heavier reimplementation [J, L on the exact cost].
2. Basis-space indices, after gbw ingestion (B): Mulliken/Loewdin populations per angular-momentum shell (s/p/d/f occupancy per atom, useful for TMs and lanthanides), unthresholded Mayer/Wiberg matrices, and multicenter indices. IAO charges and IBO bonding (Knizia 2013, JCTC) give a basis-insensitive minimal-basis picture. Per-IBO atom assignment gives bond and lone-pair counts and an oxidation-state estimate. Cheap: one projection plus localization.
3. Orbital-resolved atomic features. Atom contributions (Hirshfeld or IAO) to HOMO, LUMO, HOMO-1, LUMO+1 per spin, which are frontier-orbital condensed Fukui proxies (Fukui function: Parr and Yang 1984, JACS; condensed form: Yang and Mortier 1986, JACS). For occupied orbitals only, these can be computed from today's wfx. LUMO needs B.
4. Spin descriptors for open shells. Per-atom spin populations under several partitions already exist. New options: the Head-Gordon unpaired-electron count per atom from natural-orbital occupations (Head-Gordon 2003, Chem Phys Lett), local <S^2> contributions, and alpha/beta frontier-orbital localization. All B, all cheap.
5. Density-derived atomic shape. Atomic <r^n> moments, effective radii, and anisotropy of the atomic quadrupole (eigenvalue spread). These come free from the multipole pass.
6. Steric descriptors from the density. Percent buried volume around metal centres, computed against the rho = 0.001 isosurface or summed atomic densities instead of fixed vdW spheres. The existing marching-tetrahedra mesh and box grid can be reused. Mainly useful for organometallic verticals.
7. Compressed density representations. Expand the density in an auxiliary basis (def2-universal-jkfit, Coulomb metric) and use per-atom coefficient blocks. Use them directly in equivariant models, or reduced to rotational invariants (power spectra) for invariant ones. Prior local experiment: ~/dev/pyscf/densityfit_q.py (psi4) [V]. Related ideas in the literature: SPAHM (Fabrizio, Briling, Corminboeuf, 2022, Digital Discovery) and OrbNet's symmetry-adapted AO features (Qiao et al., 2020, J Chem Phys). Needs 3-index integrals (libcint). orca_2json exports the aux JK basis per atom (opi output model atoms.py: basisauxjk) [V].
8. Response-free reactivity proxies. Atom-resolved ALIE minima and LEA on the surface. Electrostatic potential at nuclei: ORCA can print it, but parse_orca does not parse it [L on ORCA keyword availability]. Local hardness proxies from frontier-orbital localization.
9. Things to skip [J]: global CDFT indices (linear in HOMO/LUMO); NCI grid histograms (no clean graph mapping); anything that needs N+1/N-1 or field calculations (X) unless a new campaign is planned.

## 4. Standalone engine architecture sketch

### 4a. Ingestion (removes orca_2mkl, the Multiwfn convert step and the ECP molden patch)

- [V] Today: gbw -> orca_2mkl molden -> overwrite_molden_w_ecp -> Multiwfn writes the wfx and adds the EDF from its library (core/omol.py:141-143, 377-379; utils/io.py:538). Cleanup deletes the wfx when a gbw is present (core/omol.py:1092-1095), so the gbw is the durable artifact.
- [V] The Multiwfn-written wfx has no contracted basis. In the rmechdb_652 fixture (tests/test_files/charge_engine/rmechdb_652_step10_0_2/orca.wfx) all 59 MOs are occupied, with no virtuals. The engine discards zero-occupation MOs anyway (core/charge_engine.py:103). [I] Every B-class descriptor is therefore blocked on the current input.
- Proposed reader: orca_2json on the gbw (present in ~/dev/orca_6_0_1_linux_x86-64_shared_openmpi416 [V]). The OPI models list per-atom basis, aux bases, nuclear charge, MOs, and the S, H, F, J, K matrices (~/dev/opi/src/opi/output/models/json/gbw/properties/molecule.py:55-59, atoms.py:41-51) [V]. To verify: ORCA 5 support (part of the dataset was run with ORCA 5), ECP parameters in the export (not in the OPI model), file size, and runtime at 270+ atoms. Fallback: keep orca_2mkl molden but parse it in Python (iodata already reads molden in the horton env).
- ECP core densities: the engine needs EDF Gaussians per ECP element. They come from Multiwfn's EDF library today (edflib.f90), and the Multiwfn source ships no license file (plan doc 2026-10-04, line 27-28) [V]. Options: (a) ask the Multiwfn author for permission to redistribute the tables; (b) fit our own core densities from all-electron atomic calculations, which changes values and violates "no mixed-engine datasets" for ECP elements; (c) valence-only partitioning for ECP atoms, like HORTON, which breaks parity. Decision needed (Q1).

### 4b. Shared numerical core

Already present [V]:
- Multiwfn atom grids (charge_engine.py:139); Becke weights (294); Hirshfeld weights with Multiwfn proatom tables (341); Voronoi mask (399); MBIS fit (426-660)
- Blocked dense-GEMM MO evaluation with distance screening and a Morton point order (231-265); EDF density (202)
- AOM build and DI (706-736)
- Marching tetrahedra, vertex elimination and the ALIE evaluator (surface_engine.py:114, 301, 446)

New building blocks [J], in order of what they unlock:
1. AO-basis layer (contracted shells, S matrix, density matrix P per spin, natural orbitals). This unlocks every B descriptor. libcint through PySCF is the low-effort route (libcint: Sun 2015, J Comput Chem).
2. Analytic first and second derivatives of the primitives. These give grad rho, the rho Hessian, tau/G(r), K(r), ELF, LOL and the Laplacian on any point set. Needed for CP search (Newton steps), bond-path tracing, ellipticity, and the elf/laplacian/grad_norm fuzzy routines. An extension of _primitive_block; moderate effort.
3. CP search and bond paths. Seeds from nuclei, pair midpoints, triangles and pyramids as in qtaim_data, plus Poincare-Hopf checks and the CP property table. Exact parity with Multiwfn CP positions should be feasible: [I] they are converged Newton points, so they are less order-sensitive than the ALIE mesh. Keep ring and cage CPs this time.
4. Nuclear-attraction integrals on point sets for ESP (other_esp, ESP at CPs, MK/RESP/CHELPG). libcint's int1e_grids in PySCF is one option [L on API stability]. Another is a density-fitted ESP (analytic from aux coefficients), approximate but very cheap.
5. Orbital localization (Pipek-Mezey with IAO or Loewdin populations; Boys) for LOBA/IBO.

### 4c. Batching, hardware, provenance

- Batching [J]: a long-lived worker per node (or per GPU) that takes a list of job folders, which avoids re-importing numba/JIT per job. Inside a job, every descriptor that needs rho/MO values on the molecular grid should share one evaluation (the "one-pass" principle that gave the 7-20x speedups [V] plan doc lines 105-113). Density evaluation is still 66-80% of engine time (plan doc line 115) [V], so it is the first GPU target.
- GPU [J]: the blocked-GEMM layout ports directly to CuPy or torch (primitive block -> matmul with MO coefficients). FP64 is needed for parity, so A100/H100-class cards are required (plan doc line 192) [V]. Ring-reduction and CP-search steps stay on CPU.
- Provenance: add an engine_version and per-descriptor spec hash (grid, partition, proatom table, EDF source) to each output, as parse_orca does with ORCA_PARSER_VERSION (core/parse_orca.py:19-25) [V]. Store the timings as the engine already does (charge_engine timing key).
- Validation: (1) parity tier: every Multiwfn-equivalent routine is compared against Multiwfn on the 87-wfx cross-validation set and the 194 LRC reference jobs, with the existing criteria (median <= 0.005 e, max <= 0.03 e) or print precision, as already done for charges [V] plan doc lines 126-142. (2) new-descriptor tier: compare with an independent code where one exists (HORTON for partitions and multipoles, PySCF for Mulliken/Mayer/IAO, ORCA's printed Mayer/Loewdin/MBIS from orca.json). Also check invariants: charges sum to the total charge, LI + 1/2 sum DI = N, multipoles reproduce the molecular dipole, TS C6 matches the reference on free atoms. (3) grid-convergence checks against 150x974 as in the plan doc [V].

### 4d. Mapping into existing schemas

- [V] Atom keys are "{1-based idx}_{Elem}" and bond keys "{i}_{Ei}_to_{j}_{Ej}" (core/parse_orca.py:63-70). parse_charge_data and parse_fuzzy_data index atoms by int(key.split("_")[0]) - 1 (utils/lmdbs.py:569-590, 618-626). New per-atom schemes that follow the charge.json shape {scheme: {charge, spin, dipole}} or the fuzzy shape {name: {atom: value, sum, abs_sum}} flow into GeneralConverter with no converter change. Only filters need updating.
- [J] Things that do not fit today's shapes: per-atom vectors and tensors (atomic dipoles, quadrupoles, density-fitting blocks), ring-level features (PDI, FLU, MCI, RCPs), and lone-pair/basin entities. Proposal: a new descriptors.json -> descriptors.lmdb data type with explicit "atom", "bond", "ring" and "global" sections and a schema version. Rings would enter qtaim_embed either as global aggregates or as a ring-to-atom broadcast (a mean of the ring features onto member atoms), which avoids a new node type at first.
- [V] Decision on record: "No mixed-engine datasets" (plan doc line 19). [I] Additive descriptors that Multiwfn never produced (TS volumes, LI, IAO) do not mix engines inside one descriptor. They can be backfilled on existing verticals from the gbw without touching charge.json, provided the gbw files are still on disk (Q3).

## 5. Prioritized shortlist (top 10)

Prerequisite (not a descriptor): native gbw ingestion with the full MO set and AO basis (4a). Items 4, 7, 8 and 9 depend on it.

| # | Descriptor | Level | Input | Extra cost [J] | Effort [J] | Rationale [J] |
|---|---|---|---|---|---|---|
| 1 | Atomic dipole and quadrupole magnitudes plus <r^2> for Hirshfeld and MBIS | atom (+ global check) | W | ~0 | low | Anisotropy that charges miss; uses the existing grid and weights |
| 2 | TS effective volume ratio, polarizability, C6 per atom; molecular C6 | atom + global | W | ~0 | low | Physical dispersion/polarizability features; exact Multiwfn reference exists (fuzzy.f90:172) |
| 3 | LI per atom and the full DI matrix (no threshold) | atom + edge | W | ~0 (AOMs already built) | low | LI is a strong complement to charge; long-range DI enriches edges |
| 4 | Frontier-orbital atomic contributions (HOMO/LUMO +/-1, per spin) = condensed FMO Fukui | atom | B (W for occupied only) | low | low-medium | Reactivity signal not present in any current feature |
| 5 | Per-atom ALIE surface stats (area, min, mean), later ESP | atom | W | ~0 for ALIE | low-medium | Turns global surface features into node features; reuses the mesh |
| 6 | QTAIM ring/cage CPs and bond-path geometry (path length minus distance, curvature) as part of an engine CP search | ring + bond | W + derivatives | low per job | high | Removes the last big Multiwfn dependency; RCP data is currently discarded |
| 7 | Shell-resolved populations (s/p/d/f per atom) and natural-orbital unpaired electrons per atom | atom | B | ~0 | low-medium | Targets TM, lanthanide and open-shell verticals, where charges alone are ambiguous |
| 8 | LOBA / IBO-based oxidation-state estimate for metals | atom | B | low-moderate | medium | Direct chemical label for TM verticals; also a sanity check on charges |
| 9 | Ring aromaticity set (PDI, FLU, MCI, HOMA) broadcast to ring atoms | ring | W | ~0 after DI | medium (ring perception, schema) | Cheap; valuable only for aromatic-heavy verticals |
| 10 | Per-atom density-fitting coefficient blocks (jkfit, Coulomb metric) or their invariants | atom | B + 3-index integrals | moderate | medium-high | A compact, complete density representation; ML value unproven on this data [L] |

Pinned by the user (2026-10-08), regardless of rank: steric descriptors from the density (section 3, item 6), percent buried volume and related measures computed against the rho = 0.001 isosurface or summed atomic densities, per metal centre and per atom. Reuses surface_engine's box grid and mesh.

Deferred: ESP-fit charges, finite-difference Fukui (X), finite-field atomic polarizabilities (X), NICS (X), ELF basin populations (cost, and the mapping to graph nodes), Hirshfeld-I/ISA variants (redundant with MBIS [J]).

Suggested order [J]: 1-3 and 5 first (pure engine additions on today's wfx, about a day each to add and validate against Multiwfn). Then gbw ingestion with 4, 7 and 8. Then the derivative layer with 6. Then EDF independence, which is what actually lets the project drop Multiwfn.

### Open questions for the user

1. Pending (user is asking Tian Lu). EDF/ECP core densities: is redistributing Multiwfn's EDF tables acceptable (after asking the author), or must the engine supply its own core densities, accepting a parity break for ECP elements?
2. RESOLVED 2026-10-08: pull reference values from HORTON, PySCF and other implementations for the descriptors we find compelling, and establish parity across levels of theory (README.md). Original question - parity bar for new descriptors that have no Multiwfn analogue: is agreement with HORTON/PySCF plus invariants enough?
3. ANSWERED 2026-10-08: yes, all gbw files are on the HPC systems. Original question - backfill: are gbw files still available for all 34 verticals on their clusters? That decides whether items 4, 7, 8 and 10 can be added to existing datasets or only to new ones.
4. RESOLVED 2026-10-08: both orca_2json and orca_2mkl + molden, one as fallback. Original question - should orca_2json be the canonical reader, given part of the dataset is ORCA 5? Its ORCA 5 behaviour and ECP export are unverified.
5. Graph schema: should ring-level and tensor features get first-class support in qtaim_embed, or should they first enter as broadcast and aggregated scalars?
6. Is a new SCF campaign (N+1/N-1 for Fukui, finite fields for polarizabilities) ever in scope? If yes, it changes the ranking substantially.

## Caveats and limitations

- No Multiwfn manual was available, so method descriptions rely on source menus and general knowledge. Input requirements marked B or X were read from source comments (for example fuzzy.f90:167 "Need virtual orbital information"), not from runs.
- The "no virtuals in wfx" finding comes from one fixture. It should be checked on an ECP and an open-shell production wfx before it is relied on.
- All cost, effort and ML-value columns are judgments, not measurements. Nothing here was benchmarked.
- The orca_2json field list comes from the OPI pydantic models, not from an exported file. ECP parameters and ORCA 5 support are unverified.

## References (methods cited above)

- Lu and Chen, 2012, J Comput Chem (Multiwfn).
- Becke, 1988, J Chem Phys (multicenter integration).
- Hirshfeld, 1977, Theor Chim Acta.
- Bultinck, Van Alsenoy, Ayers and Carbo-Dorca, 2007, J Chem Phys (Hirshfeld-I).
- Lillestolen and Wheatley, 2008, Chem Commun (ISA).
- Verstraelen et al., 2016, J Chem Theory Comput (MBIS).
- Manz and Limas, 2016, RSC Adv (DDEC6).
- Marenich, Jerome, Cramer and Truhlar, 2012, J Chem Theory Comput (CM5).
- Mayer, 1983, Chem Phys Lett.
- Tkatchenko and Scheffler, 2009, Phys Rev Lett.
- Knizia, 2013, J Chem Theory Comput (IAO/IBO).
- Thom, Sundstrom and Head-Gordon, 2009, Phys Chem Chem Phys (LOBA).
- Head-Gordon, 2003, Chem Phys Lett (unpaired electrons).
- Parr and Yang, 1984, J Am Chem Soc; Yang and Mortier, 1986, J Am Chem Soc (Fukui, condensed Fukui).
- Becke and Edgecombe, 1990, J Chem Phys (ELF).
- Johnson et al., 2010, J Am Chem Soc (NCI).
- Lu and Chen, 2022, J Comput Chem (IGMH).
- Lu and Chen, 2021, Chemistry-Methods (IRI).
- Klein et al., 2020, J Phys Chem A (IBSI).
- Poater et al., 2003, Chem Eur J (PDI).
- Matito, Duran and Sola, 2005, J Chem Phys (FLU).
- Sun, 2015, J Comput Chem (libcint).
- Qiao et al., 2020, J Chem Phys (OrbNet).
- Fabrizio, Briling and Corminboeuf, 2022, Digital Discovery (SPAHM).
