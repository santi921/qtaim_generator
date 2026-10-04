#!/bin/bash
# Charge-engine LRC test, step 0 (login node, once). Selects 10k random tm_react
# folders that Multiwfn already completed, writes the shared job list, and copies
# those result folders into the merge tree. Originals in $RES are never written.
#   bash setup.sh
set -euo pipefail
SRC=/global/scratch/users/santiagovargas/OMol4M_raw
RES=/global/scratch/users/santiagovargas/OMol4M
WORK=/global/scratch/users/santiagovargas/engine_test
N_JOBS=${N_JOBS:-10000}
SEED=${SEED:-20261004}

mkdir -p $WORK/merge $WORK/scratch $WORK/ref $WORK/slices/logs

# completed = all six compiled jsons non-empty in generator/
if [ ! -s $WORK/tm_react_completed_rel.txt ]; then
  for g in $RES/tm_react/*/generator; do
    ok=1
    for f in charge bond fuzzy_full qtaim other timings; do
      [ -s $g/$f.json ] || { ok=0; break; }
    done
    [ $ok = 1 ] && echo "tm_react/$(basename $(dirname $g))"
  done > $WORK/tm_react_completed_rel.txt
fi
echo "completed tm_react folders: $(grep -c . $WORK/tm_react_completed_rel.txt)"

# fixed-seed sample; the same list drives merge, scratch and ref
shuf -n $N_JOBS --random-source=<(yes $SEED) $WORK/tm_react_completed_rel.txt > $WORK/tm_react_${N_JOBS}_rel.txt
sed "s#^#$SRC/#" $WORK/tm_react_${N_JOBS}_rel.txt > $WORK/jobs.txt
head -200 $WORK/jobs.txt > $WORK/jobs_ref.txt
echo "jobs: $(grep -c . $WORK/jobs.txt), ref subset: $(grep -c . $WORK/jobs_ref.txt)"

# test A works on copies of the finished folders
while read rel; do
  [ -d $WORK/merge/$rel ] || { mkdir -p $(dirname $WORK/merge/$rel); cp -a $RES/$rel $WORK/merge/$rel; }
done < $WORK/tm_react_${N_JOBS}_rel.txt
echo "merge tree: $(du -sh $WORK/merge | cut -f1)"
