#!/bin/bash
# IBSI / Laplacian bond order: engine vs Multiwfn on the same orca.wfx (campaign.py), one CPU
# node, N_PAR jobs at a time with THREADS threads each.
#
#   sbatch -J ibsilbo --export=ALL,CODE=$CODE,LIST=$WORK/sample_l2.txt,OUT=$WORK/ibsi_lbo_ref \
#     $CODE/scripts/ibsi_lbo_reference/nersc_ibsi_lbo.slurm
# then: PYTHONPATH=$CODE python $CODE/scripts/ibsi_lbo_reference/campaign.py summarize $OUT
#SBATCH -A m3278
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH --cpus-per-task=128
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-user=santiagovargas@lbl.gov
#SBATCH --mail-type=END,FAIL

set -eo pipefail
module load python
conda activate /pscratch/sd/s/santiago/envs/generator
CODE=${CODE:?CODE is required}
LIST=${LIST:?LIST is required}
OUT=${OUT:?OUT is required}
SRC=${SRC:-/pscratch/sd/s/santiago}
N_PAR=${N_PAR:-16}
THREADS=${THREADS:-8}
MULTIWFN=${MULTIWFN:-/pscratch/sd/s/santiago/multiwfn/Multiwfn_3.8_dev_bin_Linux_noGUI/Multiwfn_noGUI}
ORCA_2MKL=${ORCA_2MKL:-/pscratch/sd/s/santiago/orca_6_1_1_linux_x86-64_shared_openmpi418_nodmrg/orca_2mkl}
export PYTHONPATH=$CODE PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
export OMP_STACKSIZE=4G KMP_STACKSIZE=200M OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=$THREADS
ulimit -s unlimited
python -c "import qtaim_gen; assert qtaim_gen.__file__.startswith('$CODE'), qtaim_gen.__file__"
echo "branch: $(git -C $CODE rev-parse --abbrev-ref HEAD) $(git -C $CODE rev-parse --short HEAD)"
mkdir -p $OUT logs
xargs -a $LIST -P $N_PAR -I{} sh -c \
  "python $CODE/scripts/ibsi_lbo_reference/campaign.py one {} $OUT --src $SRC --multiwfn $MULTIWFN \
   --orca_2mkl $ORCA_2MKL --nthreads $THREADS || echo FAILED {}"
python $CODE/scripts/ibsi_lbo_reference/campaign.py summarize $OUT
