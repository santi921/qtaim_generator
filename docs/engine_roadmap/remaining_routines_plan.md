# Remaining Multiwfn / orca_2mkl routines: replacement plan

Date: 2026-10-08. Status: research and planning only, no code changed.
Repo: /home/santiagovargas/dev/qtaim_generator, branch feat/charge-engine. Multiwfn source: ~/dev/Multiwfn_src/Multiwfn_3.8_src_Linux (abbreviated MW/). Companion docs: qtaim_engine_plan.md (QTAIM, separate agent), descriptor_engine_vision.md (this folder).
Labels: [V] verified in code, a file or a local run; [I] inference; [E] effort estimate; [U] unverified.

## Summary

- After the charge/surface engines, Multiwfn still runs for: convert (every level), qtaim (every level), other_geometry (level 0); elf_fuzzy, chelpg, ibsi_bond (level 1); bader, laplacian_bond, laplacian_rho_fuzzy, grad_norm_rho_fuzzy, other_esp (level 2). orca_2mkl runs in convert. Three new building blocks cover all of it except bader: (1) primitive gradients/Hessians (shared with QTAIM), (2) a native wavefunction reader with our own EDF tables, (3) nuclear-attraction (ESP) integrals on point sets.
- By local timings, bader is 60% of full_set 2 time; elf_fuzzy (9%), chelpg (6%) and ibsi_bond (5%) lead full_set 1. Convert is cheap (median 0.56 s) and other_geometry trivial (0.07 s), so those two are removed for dependency reasons, not speed. other_geometry plus a native reader would leave QTAIM as the only Multiwfn call at full_set 0.
- Two verified quirks change what "parity" means: with ispecial=1 (settings.ini:115) the bader "spin" field holds the Shannon-entropy column, not spin; IBSI runs on a 30x110 grid that Multiwfn itself flags as too coarse for IGMH. ORCA 6.0.1 tools cannot read our ORCA 5 gbw files; ORCA 5's own orca_2mkl/orca_2json can.

## 1. Inventory

### 1a. Where the calls come from

- [V] Stdin strings: qtaim_gen/source/data/multiwfn.py:32-152. Engine-owned routines: ENGINE_ROUTINES, multiwfn.py:11-18. run_jobs skips them (core/omol.py:650-651) and gbw_analysis runs the engines after run_jobs (omol.py:2872-2877).
- [V] Convert: write_conversion writes convert.in = `orca_2mkl <base> -molden`, then `rm <base>.gbw` if a gbw exists (omol.py:107-154, rm at 151). create_jobs writes the Multiwfn export `100\n2\n4\norca.wfx\n0\nq\n` (omol.py:374-383) and props_convert.mfwn reads the molden (omol.py:446-456). run_jobs runs convert.in, then for orca_6=False patches the molden [Atoms] charge column from orca.out (omol.py:546-575; utils/io.py:506 pull_ecp_dict, 538 overwrite_molden_w_ecp).
- [V] Parse dispatch: omol.py:763-810. Restart markers for other_*: omol.py:2309-2313.

### 1b. Per routine

| Routine (level) | Stdin | Multiwfn path | Algorithm | Parser keeps -> schema | Reuse | Effort [E] |
|---|---|---|---|---|---|---|
| other_geometry (0) | `26\n3\na\nn\n3\nh\nn\n8\n0\n0\nq\n` | menu 26 MW/Multiwfn.f90:566-640; calcMPP MW/otherfunc.f90:3796; ptsfitplane MW/util.f90:551 (SVD plane); option 8 calcmolsize MW/otherfunc.f90:3578 | MPP = RMS distance to the SVD plane, SDP = max(+d) - min(-d), all atoms and non-H atoms; Bohr in, Angstrom out | parse_other_doc_geometry (core/parse_multiwfn.py:902): mpp_full, sdp_full, mpp_heavy, sdp_heavy -> other.json. calcmolsize output not parsed [V] | coordinates only; no wavefunction | 0.5 day. Edge cases: 0-2 heavy atoms (H2, H2O) |
| elf_fuzzy (1) | `15\n1\n9\n0\nq\n` | fuzzyana(0) MW/fuzzy.f90:19, isel=1 loop 640-760; ELF_LOL MW/function.f90:3109 | Becke fuzzy integral of ELF on 75x434, radcut 10 (iautointgrid does not touch isel=1, fuzzy.f90:414-424). Needs orbital gradients and tau; MOs only, no EDF (function.f90:3140-3183); ELF_addminimal=1 adds 1e-5 to D (settings.ini:24); spin-polarized form for open shell | parse_fuzzy_real_space (parse_multiwfn.py:1111): per-atom value, sum, abs_sum -> fuzzy_full.json | grid, Becke weights, print logic of becke_fuzzy_density (charge_engine.py:139, 294, 880) | 2 days after the derivative kernel |
| grad_norm_rho_fuzzy (2) | `15\n1\n2\n0\nq\n` | as above, fgrad MW/function.f90:2390 | integral of abs(grad rho), EDF included (2402) | same | same pass as elf_fuzzy | +0.5 day |
| laplacian_rho_fuzzy (2) | `15\n1\n3\n0\nq\n` | flapl MW/function.f90:2516 (EDF at 2524) | integral of the Laplacian; needs primitive Hessians | same | same pass | +0.5 day |
| ibsi_bond (1) | `9\n10\n1\n1\n0\n0\nq\n` | IBSI MW/bondorder.f90:835; calcatmpairdg MW/visweak.f90:970; proatmgrad MW/function.f90:3947 | IGMH (wavefunction present): grid choice "1" = 30x110 (visweak.f90:996-1006, deprecated for IGMH per its own prompt); Becke weights with covr_tianlu, 3 iterations (gen1cbeckewei MW/sub.f90:3141); centers farther than 6 Bohr from every close pair skipped; per point rho gradient plus free-atom gradients by Lagrange interpolation of built-in densities (8 Bohr cut); pair delta-g summed over all atom pairs; IBSI = Int/d^2/0.566653, printed for d < 3.5 Angstrom | parse_bond_order_ibsi (parse_multiwfn.py:705): "i_E_to_j_E" -> IBSI value -> bond.json | Becke weights, Hirshfeld radial tables (charge_engine.py:341, data/multiwfn_atmraddens.py) | 3 days after the derivative kernel; O(N^2) per point needs pair screening |
| laplacian_bond (2) | `9\n8\nn\n0\nq\n` | fuzzyana(2) MW/fuzzy.f90:190-191, grid 45x302 at 419-424, overlap sum 754-765, output 2051-2101 | Becke weights; LBO_ij = -10 x sum over points of P_i P_j w lapl(rho) where lapl < 0; threshold 0.05 (MW/define.f90:478) | parse_bond_order_laplace (parse_multiwfn.py:672) -> bond.json | Becke code, Laplacian from the derivative kernel | 1.5 days |
| chelpg (1) | `7\n12\n1\nn\n0\n0\nq\n` | population.f90:195 -> fitESP(2) MW/population.f90:2743; setCHELPGpt 3345; setESPfitvdwr 3448; fitESP_calcESP 3580; eleesp MW/function.f90:4269 -> libreta eleesp2 MW/libreta_hybrid/libreta.f90:29 (iESPcode=2, settings.ini:111) | cube of points, 0.3 Angstrom spacing, 2.8 Angstrom margin, points inside the radii dropped; radii 1.45/1.5/1.7/2.0 Angstrom for Z <= 18, UFF/1.2 above because ispecial=1 (population.f90:3489); ESP = effective-charge nuclear term (nucesp, function.f90:4250) + electronic term; total-charge-constrained least squares | parse_charge_chelpg (parse_multiwfn.py:326): per-atom charge only -> charge.json["chelpg"]["charge"] | none for integrals (new kernel); the atom-key scheme | 5 days incl. the ESP kernel |
| other_esp (2) | `12\n0\n-1\n-1\nq\n` | surfana MW/surfana.f90:2; ESP spacing 0.25 Bohr (214); vertex ESP 933-950; statistics 1203-1300 | rho = 0.001 marching-tetrahedra surface (as ALIE but 0.25 Bohr), ESP at surviving vertices, area-weighted stats incl. nu, Pi, MPI, polar area | parse_other_doc_esp (parse_multiwfn.py:932) ESP_* (21 keys) -> other.json | nearly all of surface_engine.py (march 114, eliminate 301, driver 446) | 2 days after the ESP kernel |
| bader (2) | `17\n1\n1\n2\n7\n1\n1\n7\n1\n5\n-10\nq\n` | basinana MW/basin.f90:4; generate 603-741 (medium grid 0.10 Bohr, 1555/1589); integratebasinmix 2963 | basins on a uniform rho grid, attractor clustering, mixed atom-center + uniform integration | parse_charge_doc_bader (parse_multiwfn.py:361): charge (normalized) and "spin" -> charge.json["bader"] | basin assignment from the QTAIM plan | 2-3 weeks; depends on QTAIM |
| qtaim (all) | `2\n2\n3\n4\n5\n8\n7\n0\n-10\nq\n` (multiwfn.py:144-152) | see qtaim_engine_plan.md | - | parse_qtaim | - | see that plan |
| convert (all) | orca_2mkl -molden; Multiwfn `100\n2\n4\norca.wfx\n0\nq\n` | readmolden MW/fileIO.f90:4078; EDF supply 173-192; readEDFlib 4024; EDFLIB MW/edflib.f90:79; outwfx MW/fileIO.f90:7836 | see section 3 | orca.wfx (engine input) | read_wfx/prepare_basis (charge_engine.py:55, 100) | 1-2 weeks incl. EDF tables |

### 1c. Verified quirks a reimplementation must decide on

- [V] Bader spin is not spin. With ispecial=1, integratebasinmix skips the integrand prompt (MW/basin.f90:3010) and integrates Shannon, Fisher and steric terms plus rho (3399-3405). The stdin then desyncs: the "1" meant to pick rho re-enters "1 Regenerate basins" (two extra regenerations appear in the output). In data/omol_test_spin_skip/orca6_uks/bader.out the table at "Atom       Basin" is Shannon/Fisher, and charge.json["bader"]["spin"]["1_W"] = -265.9709839, the Shannon value for W. Charges come from "The atomic charges after normalization" and look sane. data/omol_tests/orca6_uks holds different "spin" values whose source I did not trace [U].
- [V] ispecial=1 also selects UFF/1.2 radii for Z > 18 in CHELPG (population.f90:3489-3492). Any engine must copy this.
- [V] ELF is computed from MOs only; grad and Laplacian include EDF. ESP uses effective nuclear charges and excludes EDF (eleesp has no EDF term; nucesp uses a%charge).

## 2. Cost (local timings.json)

Source: 2,806 timings.json under data/ and tests/. Jobs are binned by inferred level (bader present = 2; elf_fuzzy, chelpg or mbis present = 1; else 0). Seconds are wall time as stored; hardware and thread counts vary.

| Level (jobs) | Engine-covered share | Remaining share |
|---|---|---|
| ~0 (2,695) | 65.3% | 34.7% |
| ~1 (68) | 36.7% | 63.3% |
| ~2 (43) | 8.5% | 90.9% |

| Level | Routine | n | Median (s) | p90 (s) | Total (s) | Share of level |
|---|---|---|---|---|---|---|
| ~0 | other (legacy combined) | 2,401 | 131.8 | 394.3 | 682,994 | 19.0% |
| ~0 | qtaim | 2,694 | 16.2 | 316.8 | 556,986 | 15.5% |
| ~0 | convert | 2,575 | 0.56 | 3.0 | 5,857 | 0.2% |
| ~0 | other_geometry | 210 | 0.07 | 0.1 | 37 | 0.0% |
| ~1 | other (legacy) | 6 | 81,993 | - | 328,015 | 28.5% |
| ~1 | qtaim | 68 | 19.3 | 8,265 | 173,070 | 15.1% |
| ~1 | elf_fuzzy | 64 | 63.7 | 2,834 | 107,304 | 9.3% |
| ~1 | chelpg | 58 | 67.3 | 481 | 65,530 | 5.7% |
| ~1 | ibsi_bond | 65 | 16.8 | 1,971 | 52,010 | 4.5% |
| ~2 | bader | 43 | 1,377 | 17,399 | 223,255 | 59.6% |
| ~2 | other (legacy) | 42 | 277.8 | 2,413 | 40,088 | 10.7% |
| ~2 | chelpg | 34 | 170.3 | 1,450 | 15,136 | 4.0% |
| ~2 | laplacian_rho_fuzzy | 15 | 105.0 | 597 | 4,601 | 1.2% |
| ~2 | elf_fuzzy | 15 | 72.2 | 348 | 2,784 | 0.7% |
| ~2 | grad_norm_rho_fuzzy | 15 | 72.0 | 348 | 2,771 | 0.7% |
| ~2 | ibsi_bond | 15 | 16.2 | 130 | 1,389 | 0.4% |
| ~2 | laplacian_bond | 1 | 2.4 | - | 2 | 0.0% |
| ~2 | other_esp | 2 | 7.4 | - | 15 | - |

Notes:
- [V] "other" is the legacy non-separate step, which ran geometry + ESP surface + ALIE in one session (multiwfn.py:127). [I] Its cost is mostly ESP, because geometry is ~0.1 s and ALIE is a cheaper function on a coarser 0.2 Bohr mesh, but the local data cannot split it.
- [V] 2,394 of the 2,806 files are data/OMol4M/rmechdb_minimal (level 0), so level 0 dominates the totals. Levels 1 and 2 rest on 68 and 43 jobs with heavy tails (p90 >> median). Several stored other_alie timings are ~0.07 s, which looks like restart backfill, not real runs [I].
- [I] Payoff order: bader >> elf/grad/lapl fuzzy (one shared pass) > chelpg > ibsi > other_esp > laplacian_bond > geometry and convert (dependency only).

## 3. The conversion problem

### 3a. What the current chain does [V]

1. orca_2mkl writes orca.molden.input. ORCA 6 adds a [Pseudo] block with effective charges. Verified on HXeOH (wfx_pull/noble_gas_compounds): "Xe 1 26". ORCA 5 has no such block (verified on omol_clean__orca5_uks, Tl listed as 81), hence overwrite_molden_w_ecp.
2. readmolden: reads the [Atoms] third column as nuclear charge (MW/fileIO.f90:4194-4199), [Pseudo] overrides it (4289-4300). It divides ORCA contraction coefficients by renormgau_ORCA (4390-4394; MW/sub.f90:1485), renormalizes shells (renormmoldengau, sub.f90:1457; called at 4400-4403), assumes all-spherical for ORCA (4442), and flips the sign of the f(+3,-3) and g/h(+3,-3,+4,-4) coefficients (4601-4617). Spin: unrestricted if the first occupation is < 1.05 (4499-4506); wfntype from occupations (4673-4685). Spherical shells are expanded to Cartesian with the gensphcartab matrices (sub.f90:3688; 4697-4765). Primitive coefficient = Cartesian MO coefficient x contraction x normgau (sub.f90:1440; fileIO.f90:4790-4803).
3. readinfile: if any Z differs from the nuclear charge and isupplyEDF=2 (settings.ini:46), it loads the built-in EDF (fileIO.f90:173-192 -> readEDFlib 4024 -> EDFLIB edflib.f90:79). EDF = s Gaussians with exponents 0.001 x 1.65^(i-1), coefficients per (Z, ncore), 5,047 lines of tables.
4. outwfx: occupied MOs only, values in E20.12, g functions reordered by convGseq (fileIO.f90:7845), effective <Nuclear Charges>, <Number of Core Electrons>, EDF block (7927-7944). The engine maps g back with WFX_G_TO_MWFN (charge_engine.py:109) and reads the EDF (charge_engine.py:91-96).

### 3b. Reader options

| Option | Drops | Status | Notes |
|---|---|---|---|
| A. Keep orca_2mkl, parse molden in Python | Multiwfn convert, molden ECP patch (charges from [Pseudo] or orca.out) | port of the readmolden ORCA path above | orca_2mkl ships with ORCA; same molden Multiwfn sees, so lowest parity risk |
| B. orca_2json | orca_2mkl and Multiwfn | [V] 6.0.1 works on ORCA 6.0.0 gbw: per-atom basis, ECP N_core, effective NuclearCharge, all 117 MOs incl. virtuals, HFTyp, m-order 0,+1,-1,+2,-2,+3,-3. [V] ORCA 5's orca_2json (~/dev/orca5, takes the basename) works on ORCA 5 gbw but its schema differs ("BasisFunctions" key, no ECPs block). | coordinates in Angstrom (unit-conversion constant matters); contraction coefficients are not molden-normalized (0.00058 vs 0.2967 for the same Xe primitive) - convention to establish [U] |
| C. Parse .gbw binary | everything | not recommended | undocumented and version-specific: [V] ORCA 6.0.1 orca_2json and orca_2mkl both fail on the ORCA 5 gbw ("Unknown HFTYP", "Wrong number of basis-sets") |

Recommendation [I]: A first, behind one `read_wavefunction(path)` that returns the same dict as read_wfx (charge_engine.py:55-97), so prepare_basis and every engine are unchanged. B second, as it also unlocks virtual orbitals and the AO basis (vision doc 4a). Both need matching ORCA-major-version tools. The `rm .gbw` in write_conversion (omol.py:151) must go or move after the reader runs.

### 3c. EDF tables

- Generate a data module from MW/edflib.f90 the same way multiwfn_atmraddens.py was generated; index it by (Z, ncore = Z - Zeff). [E] 1-2 days.
- [V] 25 of the 87 wfx_pull files carry an EDF block, so table lookup can be checked exactly against Multiwfn's output (E20.12).
- Licensing: open, the same as vision doc Q1. readEDFlib credits the Molden2AIM repository (fileIO.f90:4029-4030). That license is not checked [U].

### 3d. Validation against the wfx path

1. Primitive level: on the 87 wfx_pull folders (gbw.zstd0, orca.out and Multiwfn's orca.wfx side by side; all ORCA 6.0.0), compare rho and spin at ~1e4 random points plus Becke-grid points. Compare nelec, core electron count, EDF arrays, MO energies and occupations. Expect about 1e-12 relative error (E20.12 rounding).
2. Engine level: charge_engine.run and alie_surface on both inputs; require print-precision agreement with the existing references (tests/test_files/charge_engine/*/multiwfn_reference*.json).
3. ORCA 5: data/cross_validation_wfns holds ORCA 5 gbw files (e.g. omol_clean__orca5*). Reference wfx must be regenerated with ORCA 5 orca_2mkl + local Multiwfn, because the stored .wfn files lose alpha/beta for open shells.
4. [I] Mesh risk: surface_engine needed D20.13-rounded coordinates to reproduce Multiwfn's mesh exactly (surface_engine.py:7-8). Full-precision inputs could shift single vertices near the isovalue. Either compare ALIE/ESP with a tolerance or round inputs to E20.12 in a compatibility mode (decision D5).

## 4. Phased order

| Phase | Content | Depends on | Removes |
|---|---|---|---|
| P0 | Derivative kernel: first and second derivatives of primitives in the blocked orbital_values (charge_engine.py:162, 231); rho/grad/Hessian/tau API. Co-own with the QTAIM plan | - | (enabler) |
| P1 | other_geometry in the engine | coordinates | last non-QTAIM Multiwfn call at level 0 |
| P2 | Native reader (option A) + EDF data module + electron-count checks ported (omol.py:1942-2060) | P1 not required | Multiwfn convert, molden ECP patch |
| P3 | elf_fuzzy + grad_norm + laplacian_rho fuzzy in one 75x434 pass | P0 | 3 routines (levels 1-2) |
| P4 | ibsi_bond; laplacian_bond | P0 | 2 routines |
| P5 | ESP kernel (Obara-Saika or Rys, numba, primitive pairs with screening); chelpg; other_esp on surface_engine | - | 2 routines |
| P6 | bader with QTAIM basin code | QTAIM plan | last level-2 routine |
| P7 (optional) | option B reader (orca_2json) | P2 | orca_2mkl |

[I] At P2 plus QTAIM, full_set 0 needs neither Multiwfn nor anything but orca_2mkl. P3-P5 finish full_set 1.

References: run Multiwfn locally (~/dev/Multiwfn_3_8/Multiwfn_noGUI) via `scripts/bench_charge_engines.py --full_set 2 --only elf_fuzzy,ibsi_bond,...` (production_steps, scripts/bench_charge_engines.py:191-206). Use the two fixtures in tests/test_files/charge_engine/ and the 87 wfx_pull wavefunctions (25 ECP).

### Risks

- [I] ESP parity: libreta uses its own screening. Agreement to the printed f13.8 a.u. is likely but not certain.
- [I] IBSI and LBO are O(N_atoms^2) per grid point as written. Exact parity needs the same center-skipping rule (visweak.f90:1029-1043) and a pair screen that drops only exact zeros.
- [V] bader output depends on the ispecial=1 stdin desync. Exact parity would mean copying a bug.
- [I] Laplacian fuzzy integrals cancel to ~0 per molecule, so their print-precision match is sensitive to summation order.
- [V] Rule on record: no mixed-engine datasets (memory, plan 2026-10-04). Any deliberate fix (bader spin, IBSI grid) applies only to new verticals or full regeneration.

## 5. Open decisions

- D1. RESOLVED 2026-10-08: fix, do not copy (bader spin, IBSI grid); new data differs from released data and each divergence goes in the release changelog (README.md). Original question - parity or fix: copy the bader spin bug and the 30x110 IBSI grid exactly, or fix them (fix bader by setting ispecial=0 or emitting spin correctly, and IBSI with grid 2 or 3) and accept that new data differs from released data.
- D2. RESOLVED 2026-10-08: defer the bader engine (full_set 2 is produced in few places). Original question - bader scope: implement (2-3 weeks [E], 60% of level-2 time), or first fix the Multiwfn stdin/settings and keep Multiwfn for bader.
- D3. RESOLVED 2026-10-08: both readers, one as fallback when the other fails on a given gbw, tools picked by the ORCA version that wrote the job; reader work waits unless it becomes blocking. Original question - reader: keep orca_2mkl (option A) as the final state, or require orca_2json (B) and the ORCA-5 schema differences it brings.
- D4. Pending: the user is asking Tian Lu. EDF tables: redistribution permission (shared with vision Q1).
- D5. Input precision: emulate E20.12 rounding for mesh parity, or accept tolerance-based validation.
- D6. Drop calcmolsize (option 8) from other_geometry? Its output is never parsed.
- D7. ANSWERED 2026-10-08: limited, but some full_set 2 calculations exist; P6 and the level-2 parts of P4/P5 are deferred. Original question - is full_set 2 (bader, laplacian_bond, other_esp) still produced anywhere? Only 43 local level-2 jobs exist. If not, P6 and part of P4/P5 can be deferred.

## Caveats and limitations

- Effort figures are judgment, assuming the existing engine patterns. They do not include cluster validation runs.
- Timing shares mix hardware, thread counts and eras. The level binning is inferred from which keys are present.
- The bader spin finding rests on one local output (data/omol_test_spin_skip/orca6_uks). It needs a scan of released bader data before anything is claimed about the datasets.
- orca_2json coefficient normalization and the ORCA 5 schema were only inspected, not validated numerically.
