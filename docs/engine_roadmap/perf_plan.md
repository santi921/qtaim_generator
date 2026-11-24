# Engine performance plan (full_set 0 and 1)

Branch `perf/engine-phase0` (off feat/charge-engine ff92ee3, ENGINE_VERSION 10). Scope: the
charge, surface and QTAIM engines and the per-job orchestration at full_set 0 and 1. Out of
scope for now: full_set 2 (bader, laplacian_bond, other_esp beyond what a shared kernel gives
it), DDEC6 (own handoff: ddec6_density_grids_perf_handoff.md on perf/ddec6-density-grids) and
GPU work (after phases 0-3, decision P5).

Research of 2026-11-24 (six agents, local workstation, loaded machine, single timings +/- 15-30%):
reports, scripts and raw results in `data/engine_test_2026-11/perf_2026-11-24/` on the
workstation (git-ignored; `reports/*.md`).

## Where the time goes

Size-weighted over the OMol manifest atom-count histogram (3,986,738 jobs, mean 55.3 atoms),
at 4 threads per job, core-seconds = wall x 4. [I] from a measured 10-job ladder (5-342 atoms)
fitted per stage.

| full_set | mean core-s per job | largest stages (share of core-s) |
|---|---|---|
| 0 | 230 | fukui 18.5%, density 17.2%, surface grid 16.1%, QTAIM Newton search 14.3%, fuzzy_bond 8.6%, becke 6.8%, surface bisect 4.8%, ESP at CPs 3.1%, fixed per-process overhead 2.3%, native convert 2.5% |
| 1 | 1068 | chelpg 65.4%, ELF/IBSI derivative pass 7.8%, fukui 4.0%, density 3.7%, mbis_charge 3.5%, surface grid 3.5%, QTAIM search 3.1% |

91% of core-seconds are in jobs above 50 atoms. Thread scaling 1 -> 4 at full_set 0: whole
worker process 1.6-2.9x; serial or poorly scaling: surface polygonize/eliminate, QTAIM bond
paths and Newton search, fuzzy_bond, fukui. At full_set 0 one thread per job gives 1.4-2.5x
more jobs per core-hour than four (derived from separate 1- and 4-thread runs).

## Expected gain [I]

Per-stage factors are prototype measurements; combining them assumes they are independent.

| | bitwise only | + roundoff-level | + value-changing (phase 3) |
|---|---|---|---|
| full_set 0, code | 1.31x | 1.53x | 1.63x |
| full_set 0, x 1 thread per job (1.4-1.6x) | 1.8-2.1x | 2.1-2.4x | 2.3-2.6x |
| full_set 1, code | about 1.05x | 3.5x | 3.75x |

The full_set 1 gain is the CHELPG ESP kernel (65% of the cost, 14-19x in a prototype).

## Decisions (2026-11-24)

| # | Question | Decision |
|---|---|---|
| P1 | Roundoff-level changes (summation order, GEMM partition; at most about 1e-12 relative on stored values) | Allowed without an ENGINE_VERSION `changed` entry; validation compares with that tolerance instead of exact equality. A change above it is value-changing and follows the usual rule |
| P2 | Threads per job on the clusters | 1 thread per job, many jobs per node. Lands with phase 1: in the local layout the native conversion and ORCA parse of every job run in the one parent process under its GIL, which 128 single-thread jobs per node would saturate |
| P3 | Value-changing options (MBIS stalled-check stop, QTAIM seed thinning + capture radius, bond-path step 0.10 Bohr) | Evaluate each: measure how far stored values move, then decide per option |
| P4 | Surface narrow band: point-level (roundoff, about 2.5x on ALIE) or block-level (bitwise, 1.6x) | Measure how much the derived surface values differ under the point-level variant, then decide |
| P5 | GPU | After phases 0-3 |

## Phase 0: per-job overhead (bitwise, this branch)

| # | Item | Measured cost | Change |
|---|---|---|---|
| 0.1 | `esp.HBOUND_W` built at import by 40,010 Python-level kernel calls | 0.67 s per process that imports the engines (part of the 0.94 s `esp.py` import) | store the 11 values; a test recomputes them |
| 0.2 | The Fukui molden read: a header pass over the whole file, then a second line-by-line pass | 0.79 s (53 atoms), 2.1 s (110), 5.4 s (178), 19.7-20.0 s (227, 342), serial | one read; [MO] headers located by regex; coefficient blocks parsed only for the kept orbitals; same float() conversion. The native converter uses the same reader (0.42 s at 53 atoms, 10.9-12.1 s at 227-342) |
| 0.3 | orca.wfx parsed three times per job (QTAIM, charge, surface engines) | 2 extra reads: 0.03 s (36 atoms) to 2-2.8 s (227, 342) | one parse per process, shared read-only |
| 0.4 | Cold numba cache | 18-29 s per process that compiles | dropped: the cache sits next to the sources in a writable checkout, so it is cold once per checkout and CPU type (about 128 x 25 s, one core-hour, per node type and checkout); persistent workers (phase 1) compile once per worker. A checkout on a read-only filesystem would compile in every process: keep checkouts writable or set NUMBA_CACHE_DIR |

Phase 0 results (2026-11-24, local, measured):

- 0.1: `import qtaim_gen.source.core.esp` 1.11 s -> 0.23 s; test_esp checks the stored values
  against `_hermite_bound_weights()` bit for bit.
- 0.2: the reader maps the file and finds "[" and "=" with `find` (coefficient lines contain
  neither), then parses only the coefficient blocks it keeps, with the same `float()`/`int()`
  per token. Old (git HEAD) and new readers return identical dicts on all 106 distinct local
  moldens (0.1-1954 MB), in every mode (occupied, Fukui window, mwfn_h_compat, keep_virtuals
  below 200 MB), and identical `molden_mo_headers`. Fukui window read: 1.8 -> 0.28 s (120 MB),
  10.3 -> 1.25 s (649 MB), 27.2 -> 3.9 s (1954 MB), 6.5-8.9x. Converter (occupied orbitals):
  0.97 -> 0.13 s (120 MB), 14.7 -> 1.55 s (1954 MB), 7.4-10x.
- 0.3: `omol._load_wfx` (one entry per thread, keyed on the file's path, inode, size and mtime);
  `charge_engine.run(load=...)`. A test runs the three engines in process and checks one parse
  and arrays equal to a fresh read afterwards.
- Engine outputs, trunk (c7672fc) vs this branch, in process at 4 threads: every JSON and
  CPprop.txt identical on 5 jobs (12-53 atoms, full_set 0, and 12 atoms at full_set 1; 79 files).
- Test suite: 1721 passed, 83 skipped, 2 failed (test_critic2 end-to-end tests that need the
  git-ignored data/ tree inside the checkout; they fail the same way on trunk).

## Phase 1: persistent workers and 1 thread per job

- The repo's cluster scripts run `--type_runner local`: a parsl ThreadPoolExecutor in one
  process, one `engine_worker` child per folder (about 2.0-2.2 s per job outside the engines).
- Route: a HighThroughputExecutor on a LocalProvider in local mode (as the hpc layout), workers =
  cores, `cpu_affinity="block"`, engines in process (`engine_subprocess` False). Removes the
  child cost and the parent-GIL limit, which 1 thread per job (P2) needs.
- Between jobs: close the per-folder logger handler (`setup_logger`), pin BLAS to one thread at
  worker start, recycle a worker after N jobs or above an RSS limit, keep DDEC6 in a child.
- Measure on a cluster node: N x T throughput (128 x 1, 64 x 2, 32 x 4) and peak RSS per job.

## Phase 2: kernels (roundoff-level, one routine per PR, cluster validation once per routine)

| # | Item | Prototype | Parity |
|---|---|---|---|
| 2.1 | Exact ESP kernel: pairs outer over 32-64 Morton-sorted points, vectorized asymptotic Boys branch, table branch for near lanes, numba error_model="numpy"; also serves ESP at QTAIM CPs and other_esp | 14-19x on CHELPG ESP | ESP within 7e-14 a.u., CHELPG within 5e-12 e |
| 2.2 | Charge full_set 0 package: Fukui virtuals from the density pass's primitive block, two-phase Becke, column-major primitive kernel | 1.31x charge engine | bitwise |
| 2.3 | Charge full_set 1: one block pass for density, ELF/IBSI derivatives and Fukui (needs the open-shell reductions) | 1.31-1.39x on those stages | roundoff |
| 2.4 | Surface narrow band (P4 decides the variant) | ALIE 1.6x (block) / 2.0-2.85x (point) | bitwise / roundoff; seeding needs the 99-job mesh check |
| 2.5 | QTAIM: drop bitwise-duplicate seeds, one Newton batch across stages, rho/gradient/Hessian-only kernel | 1.28x QTAIM step | roundoff; CP sets identical on 89 jobs |
| 2.6 | MBIS: chord iteration with the analytic Jacobian for the fuzzy refit and for check fits converging to the default's fixed point | fuzzy refit about 10x | within 1.6e-7 e (inside the spread of the three guesses) |

## Phase 3: value-changing options (P3)

| Option | Prototype gain | What to measure before deciding |
|---|---|---|
| MBIS: stop a stalled check fit at about evaluation 200 when its objective is below the best converged fit | selection 2.4-3.7x on jobs with stalled checks | selection flips on the 112-job set and the NERSC openac 440 (actinides untested) |
| QTAIM: drop seeds within 0.2 Bohr of an earlier seed; stop a seed within 0.03 Bohr of a CP of an earlier stage | 1.4-1.8x fewer Newton evaluations | CP-set changes on the LRC tier 2 set; the 0.0115 Bohr BCP-RCP case |
| Bond paths: RK2 step 0.10 Bohr (stop radius unchanged) | 3.3x fewer evaluations | end-CP changes; 0 of 1,671 BCPs so far |

Not pursued: angular pruning of inner shells (changes every stored value), ESP screening at
1e-10 a.u. (6-14% fewer pairs), killing vacuum seeds by density (3-6%), contracted-shell
evaluation (6% fewer primitives per block), SVML (numba dropped it in 0.62).

## Notes for later

- numpy's vector exp is 3.5x faster than numba's scalar exp only with AVX-512 (measured on the
  7950X: 1.18 vs 4.16 ns); without AVX-512 it is 4.78 ns. Perlmutter CPU nodes are Zen 3 (AVX2):
  kernel gains there must come from fewer exp/erf calls and vectorizable loops, not numpy's exp.
- GPU (P5): the orbital-value + density kernel feeds about 71% of full_set 0 today; NERSC GPU
  work goes on account gen0039_g. A Perlmutter GPU node has half the cores of a CPU node at the
  same charge, so the comparison is against the optimized CPU path.
