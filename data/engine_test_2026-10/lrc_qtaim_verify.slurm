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
# Reads the job folders only; writes $OUT. One array task per vertical:
#   sbatch -J qverify --export=ALL,VERTICALS="5A_elytes rgd_uks tm_react" --array=0-2 lrc_qtaim_verify.slurm
# CONTROL=1 runs the first $NCONTROL jobs of each sample with --control (fresh
# Multiwfn QTAIM on the regenerated wfx) into a separate file.

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
V=$(echo $VERTICALS | cut -d' ' -f$(( SLURM_ARRAY_TASK_ID + 1 )))
SAMPLE=$WORK/sample_${V}.txt
TMP=${TMPDIR:-/tmp}

COMMON="--root_omol_inputs $SRC/ --root_omol_results $RES/ --orca_2mkl_cmd $ORCA_2MKL --multiwfn_cmd $MULTIWFN --n_threads $SLURM_CPUS_PER_TASK --tmp_dir $TMP"
if [ "${CONTROL:-0}" = 1 ]; then
  head -${NCONTROL:-5} $SAMPLE > $WORK/control_${V}.txt
  python $Q/scripts/qtaim_verify_stored.py --job_file $WORK/control_${V}.txt --out $WORK/control_${V}.jsonl --control $COMMON
else
  python $Q/scripts/qtaim_verify_stored.py --job_file $SAMPLE --out $WORK/verify_${V}.jsonl $COMMON
fi
