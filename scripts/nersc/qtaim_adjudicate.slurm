#!/bin/bash
# QTAIM P4 (docs/engine_roadmap/qtaim_engine_plan.md): the engine's QTAIM step on the
# folders of LIST against the CPprop.txt stored in a Multiwfn tree, with --adjudicate:
# a Multiwfn (and, with CRITIC2, a critic2) CP search started at every engine-only CP,
# and the rounding check of the Laplacian and esp_total differences
# (scripts/qtaim_verify_stored.py). Reads the job folders and the tree only. One node;
# LIST is split into N_PAR slices run side by side, THREADS each. Rerunning resumes
# (jobs already in a slice's output are skipped).
#
#   sbatch --export=ALL,CODE=$CODE,LIST=$WORK/qtaim_adj_jobs.txt qtaim_adjudicate.slurm
# Optional: REF_TREE (Multiwfn tree under WORK with the stored CPprop.txt, default mwfn_l0),
# SRC (input root the list is under), N_PAR (default 16), THREADS (default 8), CRITIC2
# (critic2 executable; without it only Multiwfn checks the extra CPs), MULTIWFN, ORCA_2MKL.

#SBATCH -A m3278
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH --cpus-per-task=128
#SBATCH --time=04:00:00
#SBATCH -J qadj
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -eo pipefail

module load python
conda activate /pscratch/sd/s/santiago/envs/generator
CODE=${CODE:?CODE is required: the branch checkout to import, e.g. /pscratch/sd/s/santiago/qtaim_generator_<name>}
export PYTHONPATH=$CODE
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export OMP_STACKSIZE=4G
export KMP_STACKSIZE=200M
export OPENBLAS_NUM_THREADS=1
ulimit -s unlimited

ORCA_2MKL=${ORCA_2MKL:-/pscratch/sd/s/santiago/orca_6_1_1_linux_x86-64_shared_openmpi418_nodmrg/orca_2mkl}
MULTIWFN=${MULTIWFN:-/pscratch/sd/s/santiago/multiwfn/Multiwfn_3.8_dev_bin_Linux_noGUI/Multiwfn_noGUI}
SRC=${SRC:-/pscratch/sd/s/santiago}
WORK=${WORK:-/pscratch/sd/s/santiago/actinide_test}
REF_TREE=${REF_TREE:-mwfn_l0}
N_PAR=${N_PAR:-16}
THREADS=${THREADS:-8}
LIST=${LIST:?LIST is required}
OUT=$WORK/qtaim_adj
[ -x "$ORCA_2MKL" ] && [ -x "$MULTIWFN" ] || { echo "missing $ORCA_2MKL or $MULTIWFN"; exit 1; }
python -c "import qtaim_gen; print('qtaim_gen from', qtaim_gen.__file__); assert qtaim_gen.__file__.startswith('$CODE')"
echo "branch:    $(git -C $CODE rev-parse --abbrev-ref HEAD) $(git -C $CODE rev-parse --short HEAD)"
C2=""
[ -n "${CRITIC2:-}" ] && C2="--critic2_cmd $CRITIC2"

mkdir -p $OUT/slices $OUT/tmp logs
total=$(grep -c . $LIST)
per=$(( (total + N_PAR - 1) / N_PAR ))
for k in $(seq 0 $(( N_PAR - 1 ))); do
  sed -n "$(( k * per + 1 )),$(( (k + 1) * per ))p" $LIST > $OUT/slices/slice_$k.txt
  [ -s $OUT/slices/slice_$k.txt ] || continue
  NUMBA_NUM_THREADS=$THREADS python $CODE/scripts/qtaim_verify_stored.py --job_file $OUT/slices/slice_$k.txt \
      --out $OUT/adj_$k.jsonl --root_omol_inputs $SRC/ --root_omol_results $WORK/$REF_TREE/ \
      --orca_2mkl_cmd $ORCA_2MKL --multiwfn_cmd $MULTIWFN --n_threads $THREADS --tmp_dir $OUT/tmp \
      --adjudicate $C2 > $OUT/slice_$k.log 2>&1 &
done
wait
echo "done: $(cat $OUT/adj_*.jsonl | wc -l) records in $OUT"
