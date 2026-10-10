#!/bin/bash
#SBATCH --account=ac_blau
#SBATCH --partition=lr7
#SBATCH --qos=lr_lowprio
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --output=/global/scratch/users/santiagovargas/engine_test/qverify/logs/%x_%A_%a.out

# Tier 2 QTAIM verification (docs/engine_roadmap/qtaim_verification_campaign.md):
# the engine against the CPprop.txt stored in each job's generator/out_files.zip.
# Reads the job folders only. Each array task takes a disjoint slice of N lines
# of LIST and writes $WORK/<job name>_<task>.jsonl; a requeued task skips the jobs
# already in its file. CONTROL=1 adds --control (fresh Multiwfn QTAIM on the
# regenerated wfx; slow on large jobs, use N=1).
#   sbatch -J qverify  --export=ALL,LIST=$WORK/sample_all.txt,N=25 --array=0-59 lrc_qtaim_verify.slurm
#   sbatch -J qcontrol --export=ALL,LIST=$WORK/control_all.txt,N=1,CONTROL=1 --array=0-14 lrc_qtaim_verify.slurm

source ~/.bashrc
module load gcc
conda activate ${CONDA_ENV:-qtaim_engine}
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
Q=${Q:-/global/scratch/users/santiagovargas/qtaim_generator_qtaim}   # feat/qtaim-engine worktree
export PYTHONPATH=$Q

SRC=/global/scratch/users/santiagovargas/OMol4M_raw
RES=/global/scratch/users/santiagovargas/OMol4M
WORK=/global/scratch/users/santiagovargas/engine_test/qverify
ORCA_2MKL=/global/scratch/users/santiagovargas/orca_6_0_1_linux_x86-64_shared_openmpi416/orca_2mkl
MULTIWFN=/global/scratch/users/santiagovargas/Multiwfn/Multiwfn_3.8_dev_bin_Linux/Multiwfn
N=${N:-25}
: "${LIST:?LIST is required (export LIST=...)}"
ID=${SLURM_ARRAY_TASK_ID}
mkdir -p $WORK/slices
SLICE=$WORK/slices/${SLURM_JOB_NAME}_${ID}.txt
sed -n "$(( ID * N + 1 )),$(( (ID + 1) * N ))p" $LIST > $SLICE
if [ ! -s $SLICE ]; then
  echo "task $ID: empty slice, nothing to do"
  exit 0
fi

EXTRA=""
[ "${CONTROL:-0}" = 1 ] && EXTRA="--control"
python $Q/scripts/qtaim_verify_stored.py --job_file $SLICE --out $WORK/${SLURM_JOB_NAME}_${ID}.jsonl \
    --root_omol_inputs $SRC/ --root_omol_results $RES/ --orca_2mkl_cmd $ORCA_2MKL --multiwfn_cmd $MULTIWFN \
    --n_threads $SLURM_CPUS_PER_TASK --tmp_dir ${TMPDIR:-/tmp} $EXTRA
