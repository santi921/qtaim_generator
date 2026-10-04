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

## 1. Code and env (login node), separate from production

The production checkout (/global/scratch/users/santiagovargas/qtaim_generator) and env (qtaim_generator) stay
untouched; the test uses its own clone and a cloned env.

```bash
cd /global/scratch/users/santiagovargas
git clone -b feat/charge-engine git@github.com:santi921/qtaim_generator.git qtaim_generator_engine
conda create -y -n qtaim_engine --clone qtaim_generator
conda activate qtaim_engine
pip install --dry-run numba==0.68.0            # expect only numba + llvmlite
pip install numba==0.68.0
pip install -e qtaim_generator_engine --no-deps
python -c "import qtaim_gen, numba; print(qtaim_gen.__file__, numba.__version__)"   # must be the _engine clone
pip check
cd qtaim_generator_engine && pytest -q tests/test_charge_engine.py   # also builds the numba cache
```

The slurm script activates `qtaim_engine` (override with `--export=ALL,CONDA_ENV=...`).
Run everything below from `qtaim_generator_engine`.

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
