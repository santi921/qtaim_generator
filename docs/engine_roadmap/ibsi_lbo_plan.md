# Engine: laplacian_bond and ibsi_bond

Date: 2026-11-15, implemented 2026-11-16 on feat/engine-ibsi-lbo (off feat/engine-ddec6 c58d250), engine version 8.
Status: done; validated locally and on LRC and NERSC (validation_ledger.md section 2, 2026-11-16 and 2026-11-17).
Decisions D1-D5 below are README decisions 15-18 (D4, the base branch, is not repeated there). MW/ = ~/dev/Multiwfn_src/Multiwfn_3.8_src_Linux (its update date is 2026-Jan-7; IBSI code identical to 2026.10.1).
Tags: [V] verified in code or a run, [I] inference, [E] estimate.

These are the last two Multiwfn routines outside bader. ibsi_bond is the only Multiwfn call left at
full_set 1; laplacian_bond and bader are the ones left at full_set 2.

## 1. What Multiwfn does

### 1a. laplacian_bond (stdin `9\n8\nn\n0\nq\n`, full_set 2)

- [V] fuzzyana(iwork=2) -> isel=8, ifunc=3 (flapl, includes EDF). MW/fuzzy.f90:190-191, 414-424.
- [V] Grid: iautointgrid=1 (production settings.ini:15) switches to 45 radial x 302 Lebedev; radcut 10 Bohr;
  the same per-atom grid as the charge engine's multiwfn_atom_grid (fuzzy.f90:678-692).
- [V] Becke partition with covr_tianlu (iraddefine=-1, fuzzy.f90:99) and 3 iterations: the same BeckePvec the
  charge engine's `_becke` reproduces.
- [V] Accumulation (fuzzy.f90:749-771): at each point of atom a's grid, t = P_a * lapl * w; if t > 0 it goes to the
  positive matrix, else the negative one, with P_i * P_j * t for every pair i <= j. LBO_ij = -10 * neg_ij
  (fuzzy.f90:2051-2072). Printed for i < j when LBO >= 0.05, format f10.6 (print precision 5e-7).
- [V] ispecial=1 does not reach this path (it only affects isel=2).

### 1b. ibsi_bond (stdin `9\n10\n1\n1\n0\n0\nq\n`, full_set 1)

- [V] IBSI (MW/bondorder.f90:835): IGMH because a wavefunction is present; reference 0.566653 in the 2026 sources
  (0.500791 in the 2024-Oct-6 production build, see section 2); distance print
  threshold 3.5 Angstrom; IBSI_ij = Int(dg_pair)_ij / d_ij^2 (Angstrom) / 0.566653, f8.5 (print precision 5e-6).
  Every pair within 3.5 Angstrom is printed, with no threshold on the value.
- [V] calcatmpairdg (MW/visweak.f90:970), grid option "1" = 30 x 110, which Multiwfn's own prompt marks as
  deprecated for IGMH.
- [V] Per atom centre: skipped when it is farther than 6 Bohr from the midpoint of every atom pair closer than
  9 Bohr (visweak.f90:1029-1043). With every atom in both lists the pair (i, i) has distance 0 and midpoint i, so
  the minimum distance from any centre is 0 and the skip never applies; it is not coded. Weight = the centre's Becke weight (gen1cbeckewei, covr_tianlu, 3 iterations;
  sub.f90:3141, the same BeckePvec) x grid weight.
- [V] Per point (visweak.f90:1068-1105, 1143-1152):
  - rho and grad rho of the molecule (calchessmat_dens, MOs + EDF; total density for open shells);
  - free-atom rho_A and grad rho_A by proatmgrad(1): built-in radial densities, Lagrange interpolation with first
    derivative (function.f90:3947; util.f90 lagintpol), zero when r^2 >= atmrhocut^2, and **also zero beyond
    8 Bohr** (iapprox, r^2 > 64);
  - IGMH atom gradient g_A = w_A grad rho - (rho/rho0) grad rho_A + (rho rho_A / rho0^2) grad rho0;
  - dg_ij = |g_i| + |g_j| - |g_i + g_j|, summed with the weight.
- [V] The 8 Bohr cut matters: 144 elements have atmrhocut above 8 Bohr (5.64-17.53 Bohr), so the IGMH promolecule
  differs from the charge engine's Hirshfeld promolecule. It needs its own kernel.

### 1c. What the engine already has [V]

| piece | where |
|-|-|
| per-atom grid, Becke weights (BeckePvec) | charge_engine.multiwfn_atom_grid, `_becke`, becke_pair_tables |
| rho, gradient, Hessian incl. EDF | qtaim_engine.density_derivatives (`elf_only=True` gives rho + gradient incl. EDF) |
| lagintpol with first derivative | qtaim_engine.`_lagintpol_d1` |
| IGMH atom gradients (without the 8 Bohr cut) | qtaim_engine.`_delta_g_hirsh` (the delta_g_hirsh CP property) |
| proatom tables | charge_engine.mwfn_proatom_tables |
| per-step `<routine>.json` -> bond.json compile | omol._run_charge_engine; bond.json keys are routine names (`ibsi_bond`, `laplacian_bond`) |

## 2. Findings that change the plan

- [V] **The H2 reference value depends on the Multiwfn build.** The local production binary (3.8(dev) 2024-Oct-6)
  prints and divides by 0.500791; the 2025-Jun-26 build (LRC Joe runs, `data/omol_full_test/omol_refs_no_clash`) and
  both local sources (2026-Jan-7, 2026.10.1) use 0.566653. Int(dg_pair) is the same: the engine matches the 2024
  build's integrals at print precision on 99 jobs, so only the normalization differs (13%). Of 362 local
  ibsi_bond.out files, 342 are from the 2024 build and 20 from the 2025 build. The engine uses 0.566653
  (`qtaim_engine.IBSI_REF`, recorded in `_meta.ibsi`).

- [V] **Bug in the existing LBO parser.** parse_bond_order_laplace (parse_multiwfn.py:672) splits on whitespace and takes
  tokens 2 and 4. A two-letter element writes `1(Ru)` as one token, so the key comes out as `1_Ru)_to_):`, and
  the third pair of a test file got `1_Ru)_to_0.300000`. Every stored laplacian_bond pair that involves a two-letter element
  (Ru, Pt, Cl, Br, ...) is garbled. Locally there are only 3 laplacian_bond.out files (organics). The cluster trees
  need an audit (keys containing ")").
- [V] The IBSI parser reads fixed columns (line[2:8], line[11:17]), so atom indices of 1000 or more lose their first digit.
  This does not affect the engine, which writes keys directly.

## 3. Design

### 3a. laplacian_bond

`laplacian_bond_orders(basis, wfx, coords, rinv, aij, radcut, labels, nrad=45, nang=302, threshold=0.05)` in
charge_engine.py, next to fuzzy_bond_orders:

```
for each atom a:
    pts = grid45x302 + coords[a]
    lapl = trace(density_derivatives(pts, wfx, basis)["hessian"])
    P = _becke(pts, ...)                    # npts x nat
    t = P[:, a] * wts * lapl
    neg = t <= 0                            # Multiwfn's else branch (t == 0 adds 0)
    M += (P[neg] * t[neg, None]).T @ P[neg]
LBO = -10 * M ; emit i < j with LBO >= 0.05 as {"1_C_to_2_C": value}
```

- Level: full_set 2. Skipped with reason ecp_without_edf when an ECP atom has no EDF (as the derivative fuzzy family).
- Cost [E]: a second derivative pass with the Hessian on 13,590 points per atom, about 40% of the existing
  full_set 2 derivative pass. Multiwfn's own run took 2.4 s on the one local job.

### 3b. ibsi_bond

New numba kernel `_ibsi_accumulate(pts, wcen, rho, grad, coords, atnums, tables, pair_i, pair_j, need, out)`:

- per point: free atoms within 8 Bohr and inside atmrhocut -> rho_A, grad rho_A (`_lagintpol_d1`); rho0,
  grad rho0; g_A for atoms in some pair; for each pair, dg = |g_i| + |g_j| - |g_i + g_j|, out[p] += dg * wcen.
- Pair list = all i < j with d <= 3.5 Angstrom (B2A as the engine's). Computing only printed pairs is exact,
  because each pair's integral is independent. dg is exactly 0 when either atom's g is 0, so only atoms within
  8 Bohr of the point are visited.
- Deterministic parallel sum: per-thread partial arrays over fixed point blocks, reduced in block order.
- Multiwfn's centre-skip rule is not coded: it never applies (section 1b).
- IBSI_ij = Int_ij / (d_ij in Angstrom)^2 / 0.566653.
- Skips: no built-in proatom density for an element (no_builtin_proatom), ECP without EDF (ecp_without_edf).
  These are the same rules as the Hirshfeld family.

Where it runs depends on D1:
- 75 x 434 (recommended): inside the existing full_set 1 main loop, from the ELF pass's rho and gradient
  (`elf_only=True` already includes EDF) and the Becke column `w[:, b] * grid_wts`. Only the pair kernel is extra.
- 60 x 302 or 30 x 110: a separate pass with `density_derivatives(hessian=False)` on that grid.

### 3c. Wiring

- data/multiwfn.py: `ibsi_bond` in ENGINE_LEVEL1_ROUTINES, `laplacian_bond` in ENGINE_LEVEL2_ROUTINES.
- data/engine_versions.py: version 8 adds both.
- omol._run_charge_engine writes ibsi_bond.json and laplacian_bond.json; _engine_outputs_present maps both to
  bond.json; skips go into provenance as for the other families.
- _meta: grid used per routine, number of IBSI pairs, centres skipped, timings.
- scripts/compare_engine_merge.py: compare ibsi_bond and laplacian_bond.
- README divergence changelog: the IBSI grid (D1); the LBO key fix (parser, not Multiwfn).
- Fix parse_bond_order_laplace for two-letter elements (regex on `(\d+)\((\w+)\s*\)`), with a test.
- To read while implementing: the per-step json -> bond.json compile path (omol.py:916 dispatch) for the engine case.

## 4. Validation

| step | data | reference | bar |
|-|-|-|-|
| unit | synthetic | lagintpol derivative vs finite difference; dg_ii = 0; LBO and IBSI symmetric; serial == parallel bit for bit | exact |
| fixtures | rmechdb_652 (open shell), XeCl4 (ECP + EDF) | fresh Multiwfn, IBSI at the chosen grid, LBO at 45 x 302 | IBSI 5e-6, LBO 5e-7 (print precision) |
| sanity | H2 at 0.74144 Angstrom, B3LYP/6-311G** (local ORCA 6.0.1) | definition of the reference value | IBSI about 1.00 on the perfect grid |
| local parity | the 87 wfx_pull + 12 nitrates | Multiwfn at the same grid as the engine | print precision; LBO pairs near 0.05 compared with a tolerance |
| divergence | same set | production Multiwfn (IBSI 30 x 110) | statistics for the changelog, not a pass bar |
| speed | 10 of the above | Multiwfn, same threads, one job at a time | report |
| cluster | folded into the one batched run with the other ledger gaps | Multiwfn trees | as above |

Multiwfn IBSI on the 75 x 434 grid costs 9.9x its 30 x 110 run (its own prompt), so large jobs are expensive as
references. The option-4 reference set should stay at small and mid sizes.

## 5. Phases [E]

| phase | content | effort |
|-|-|-|
| A | LBO function + LBO parser fix + unit tests + fixture references | 1-1.5 days |
| B | IBSI kernel + main-loop integration + unit tests + fixture references | 2-3 days |
| C | wiring (routine sets, version 8, runner, compare script, changelog) | 0.5-1 day |
| D | local parity and divergence runs, ledger rows | 1 day |

## 6. Decisions (answered 2026-11-16: D1 75 x 434, D2 45 x 302, D3 keep both, D4 off feat/engine-ddec6, D5 fix now)

- D1. IBSI grid. Recommended: 75 x 434, shared with the main pass, so nearly free; Multiwfn option 4 gives an
  exact reference. Alternatives: 60 x 302 (option 3, separate pass), or 30 x 110 (production parity, which keeps
  a grid Multiwfn deprecates; README decision 1 says fix).
- D2. LBO grid. Recommended: 45 x 302 (Multiwfn's grid; no known defect, so the parity default). Alternative:
  reuse the 75 x 434 Hessian already computed at full_set 2, which is free but diverges.
- D3. Multiwfn's two IBSI shortcuts (skip centres more than 6 Bohr from every close pair midpoint; free atoms cut
  at 8 Bohr). Recommended: keep them, because they define the IGMH promolecule of the reference and Multiwfn calls the
  loss negligible; measure their effect once on a few jobs and record it.
- D4. Base branch. Recommended: off feat/engine-ddec6, as engine version 8, after the DDEC6 PR lands in
  feat/charge-engine; branching from feat/charge-engine now would collide on version 7.
- D5. Garbled LBO keys in stored data: fix the parser now (yes), and audit the cluster trees for ")" keys before
  any full_set 2 data is used.

## Caveats

- Efforts are estimates and exclude cluster time.
- The Laplacian fuzzy integrals showed roundoff on ECP heavy-atom cores (ledger section 2). In LBO, points
  near lapl = 0 can switch between the positive and negative sums; the effect should be far below 5e-7 [I].
- Cost of IBSI on 75 x 434 inside the engine is estimated from pair counts, not measured.

## 7. Result (2026-11-16)

- 99 local jobs (87 wfx_pull + 12 nitrates, 2-39 atoms) and the 2 fixtures: pair sets identical everywhere;
  IBSI Int(dg_pair) within 5.00e-6 of grid option 4, LBO within 5.00e-7 (print precision).
- Grid change vs production's 30 x 110 at the same reference value: up to 2.3e-2 absolute, median 4.1% and
  at most 18% relative over pairs with IBSI >= 0.05.
- Cost at 16 threads, median per job: IBSI 0.046 s inside the main pass; LBO 1.73 s vs Multiwfn 2.25 s.
- Raw output: `data/engine_test_2026-11/ibsi_lbo/` (refs/, cmp99.jsonl, mwfn_refs.py, compare.py).
- Also changed: `_engine_outputs_present` accepts an empty bond entry (one atom; no LBO above 0.05), as
  validate_bond_dict does, so such folders do not rerun the charge engine on every restart.
