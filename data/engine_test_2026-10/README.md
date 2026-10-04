# Charge-engine test on LRC (tm_react, 10k)

Plan: docs/plans/2026-10-04-feat-one-pass-charge-engine-plan.md

- A `merge`: full-runner-engine --restart on copies of 10k finished Multiwfn folders
- B `scratch`: full-runner-engine from the raw inputs, same 10k
- `ref`: stock Multiwfn runner, first 200 of the 10k, same node type and threads (timing baseline)

Originals in `$RES` are only read. Everything is written under `$WORK`.

```bash
SRC=/global/scratch/users/santiagovargas/OMol4M_raw
RES=/global/scratch/users/santiagovargas/OMol4M
WORK=/global/scratch/users/santiagovargas/engine_test
```

## 1. Code and env (login node)

```bash
cd <qtaim_generator checkout> && git fetch && git checkout feat/horton-charge-engine && git pull
conda activate qtaim_generator
pip check > /tmp/pipcheck_before.txt; pip install --dry-run numba==0.68.0   # expect only numba + llvmlite
pip install numba==0.68.0 && pip install -e . --no-deps
pip check > /tmp/pipcheck_after.txt; diff /tmp/pipcheck_before.txt /tmp/pipcheck_after.txt
pytest -q tests/test_charge_engine.py          # also compiles and caches the numba kernels
```

## 2. Select and copy (login node)

```bash
bash data/engine_test_2026-10/setup.sh
```

## 3. Submit

```bash
cd data/engine_test_2026-10
sbatch -J merge   --export=ALL,MODE=merge   --array=0-199%50 lrc_engine_test.slurm
sbatch -J scratch --export=ALL,MODE=scratch --array=0-199%50 lrc_engine_test.slurm
sbatch -J ref     --export=ALL,MODE=ref     --array=0-3      lrc_engine_test.slurm
```

## 4. Compare

```bash
# A and B against the original Multiwfn results (accuracy, untouched data, timing vs original runs)
python scripts/compare_engine_merge.py --job_file $WORK/jobs.txt --root_omol_inputs $SRC/ \
    --orig_root $RES/ --new_root $WORK/merge/   --report $WORK/report_merge.json
python scripts/compare_engine_merge.py --job_file $WORK/jobs.txt --root_omol_inputs $SRC/ \
    --orig_root $RES/ --new_root $WORK/scratch/ --report $WORK/report_scratch.json
# same-hardware timing: engine (scratch) vs Multiwfn (ref) on the 200-job subset
python scripts/compare_engine_merge.py --job_file $WORK/jobs_ref.txt --root_omol_inputs $SRC/ \
    --orig_root $WORK/ref/ --new_root $WORK/scratch/
```

Rerunning a mode resumes: prevalidation skips folders that already validate (merge and
scratch require a `charge_engine` timing).
