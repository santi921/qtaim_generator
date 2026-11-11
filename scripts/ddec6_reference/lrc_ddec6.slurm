#!/bin/bash
#SBATCH --account=pc_aditik
#SBATCH --partition=lr7
#SBATCH --qos=lr_normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --output=/global/scratch/users/santiagovargas/ddec6_test/logs/%x_%A_%a.out
# DDEC6 validation on LRC lr7: the port (core/ddec6.py) against Chargemol on raw OMol
# jobs (campaign.py). Each array task takes N lines of $LIST, one job at a time with all
# the task's cores. A requeued task resumes from its results JSONL.
#
#   sbatch -J ddec6 --export=ALL,CODE=$CODE,LIST=$WORK/ddec6_sample.txt --array=0-17 lrc_ddec6.slurm
#
# Optional: N (lines per task, default 5), CONDA_ENV (default qtaim_engine), SRC, CHARGEMOL,
# DENSITIES, ORCA_2MKL, SKIP_CHARGEMOL=1 (port timings only).

case "${SLURM_JOB_ACCOUNT:-}" in *blau*) echo "refusing to run on a blau account (${SLURM_JOB_ACCOUNT})"; exit 1 ;; esac
# batch shells start without conda: ~/.bashrc sets it up, as in the other LRC scripts
# (before set -e, so a nonzero return inside ~/.bashrc does not end the job)
source ~/.bashrc
module load gcc
conda activate ${CONDA_ENV:-qtaim_engine}
set -eo pipefail
CODE=${CODE:?CODE is required (the feat/engine-ddec6 checkout)}
export PYTHONPATH=$CODE
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export OPENBLAS_NUM_THREADS=1
export OMP_STACKSIZE=1G
ulimit -s unlimited

SRC=${SRC:-/global/scratch/users/santiagovargas/OMol4M_raw}
WORK=${WORK:-/global/scratch/users/santiagovargas/ddec6_test}
CHARGEMOL=${CHARGEMOL:-$WORK/chargemol/bin/chargemol_omp_precise}
DENSITIES=${DENSITIES:-$WORK/chargemol/chargemol_09_26_2017/atomic_densities/}
ORCA_2MKL=${ORCA_2MKL:-/global/scratch/users/santiagovargas/orca_6_0_1_linux_x86-64_shared_openmpi416/orca_2mkl}
N=${N:-5}
THREADS=${SLURM_CPUS_PER_TASK:-16}
LIST=${LIST:?LIST is required}
ID=${SLURM_ARRAY_TASK_ID}
EXTRA=""
[ "${SKIP_CHARGEMOL:-0}" = "1" ] && EXTRA="--skip_chargemol"

python -c "import qtaim_gen; assert qtaim_gen.__file__.startswith('$CODE'), qtaim_gen.__file__"
echo "branch: $(git -C $CODE rev-parse --abbrev-ref HEAD) $(git -C $CODE rev-parse --short HEAD)"
[ -x "$ORCA_2MKL" ] || { echo "missing $ORCA_2MKL"; exit 1; }
[ -n "$EXTRA" ] || [ -x "$CHARGEMOL" ] || { echo "missing $CHARGEMOL"; exit 1; }

mkdir -p $WORK/slices $WORK/logs
SLICE=$WORK/slices/${SLURM_JOB_NAME}_${ID}.txt
sed -n "$(( ID * N + 1 )),$(( (ID + 1) * N ))p" $LIST > $SLICE
if [ ! -s $SLICE ]; then
  echo "task $ID: empty slice, nothing to do"
  exit 0
fi

export NUMBA_NUM_THREADS=$THREADS
python $CODE/scripts/ddec6_reference/campaign.py \
  --job_file $SLICE --root_omol_inputs $SRC/ --work $WORK/work --out $WORK/${SLURM_JOB_NAME}_${ID}.jsonl \
  --orca_2mkl_cmd $ORCA_2MKL --chargemol_bin $CHARGEMOL --atomic_densities $DENSITIES \
  --workers 1 --threads $THREADS $EXTRA
