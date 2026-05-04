#!/bin/bash
# Run SOAP featurization across all four comparators + every OMol vertical
# under --omol-root that has a structure.lmdb. Designed for HPC so it auto-
# discovers verticals; the caller does not have to enumerate them.
#
# Output layout:
#   <output-root>/<comparator>/soap.parquet           (one per comparator)
#   <output-root>/omol/soap_<vertical>.parquet        (one per OMol vertical)
#
# Each parquet records SOAP hyperparameters and the species list in its
# kv-metadata. Species is locked to the comparator union of 12 elements
# (HBCNOFSiPSClBrI, Z=1,5,6,7,8,9,14,15,16,17,35,53), dim=32,592.
#
# Examples:
#   bash scripts/run_soap_featurization.sh
#   bash scripts/run_soap_featurization.sh \
#     --omol-root /pscratch/sd/.../omol4m_lmdbs \
#     --comparators-root /pscratch/sd/.../comparators \
#     --output-root /pscratch/sd/.../soap \
#     --n-cap 50000 --n-jobs 32

set -euo pipefail

OMOL_ROOT="data/OMol4M_lmdbs"
COMPARATORS_ROOT="data/comparators"
OUTPUT_ROOT="data/comparators"
N_CAP=25000
N_JOBS=8
OMOL_SAMPLE_FRAC=10           # percent
SEED=42
SOURCES_DEFAULT="schnet4aim qm7x pcqm4mv2 qmugs omol"
SOURCES="$SOURCES_DEFAULT"
DRY_RUN=0

usage() {
  cat <<EOF
Usage: $0 [options]

Defaults shown in [].

  --omol-root PATH         Root containing OMol verticals [data/OMol4M_lmdbs]
  --comparators-root PATH  Root containing comparator raw/ subdirs [data/comparators]
  --output-root PATH       Where to write soap.parquet files [data/comparators]
  --n-cap N                Cap per comparator [25000]
  --n-jobs N               dscribe SOAP n_jobs [8]
  --omol-sample-frac PCT   Percent of each OMol vertical to take [10]
  --seed N                 Random seed for sampling [42]
  --sources LIST           Whitespace-separated subset of {schnet4aim,qm7x,pcqm4mv2,qmugs,omol}
                           [$SOURCES_DEFAULT]
  --dry-run                Print invocations without running them
  -h, --help               This help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --omol-root) OMOL_ROOT="$2"; shift 2 ;;
    --comparators-root) COMPARATORS_ROOT="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --n-cap) N_CAP="$2"; shift 2 ;;
    --n-jobs) N_JOBS="$2"; shift 2 ;;
    --omol-sample-frac) OMOL_SAMPLE_FRAC="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --sources) SOURCES="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

run() {
  echo "+ $*"
  if [[ $DRY_RUN -eq 0 ]]; then
    "$@"
  fi
}

want() {
  local src="$1"
  for s in $SOURCES; do
    [[ "$s" == "$src" ]] && return 0
  done
  return 1
}

mkdir -p "$OUTPUT_ROOT/omol"

if want schnet4aim; then
  echo "=== schnet4aim (full) ==="
  run analysis-soap-featurize --source schnet4aim \
    --root "$COMPARATORS_ROOT/schnet4aim/raw" \
    --output "$OUTPUT_ROOT/schnet4aim/soap.parquet" \
    --n-jobs "$N_JOBS" --seed "$SEED" --no-progress
fi

if want qm7x; then
  echo "=== qm7x (cap $N_CAP) ==="
  run analysis-soap-featurize --source qm7x \
    --root "$COMPARATORS_ROOT/qm7x/raw" \
    --output "$OUTPUT_ROOT/qm7x/soap.parquet" \
    --n-sample "$N_CAP" --n-jobs "$N_JOBS" --seed "$SEED" --no-progress
fi

if want pcqm4mv2; then
  echo "=== pcqm4mv2 (cap $N_CAP) ==="
  run analysis-soap-featurize --source pcqm4mv2 \
    --root "$COMPARATORS_ROOT/pcqm4mv2/raw" \
    --output "$OUTPUT_ROOT/pcqm4mv2/soap.parquet" \
    --n-sample "$N_CAP" --n-jobs "$N_JOBS" --seed "$SEED" --no-progress
fi

if want qmugs; then
  echo "=== qmugs (cap $N_CAP) ==="
  run analysis-soap-featurize --source qmugs \
    --root "$COMPARATORS_ROOT/qmugs/raw" \
    --output "$OUTPUT_ROOT/qmugs/soap.parquet" \
    --n-sample "$N_CAP" --n-jobs "$N_JOBS" --seed "$SEED" --no-progress
fi

if want omol; then
  echo "=== omol per-vertical (${OMOL_SAMPLE_FRAC}% each) ==="
  if [[ ! -d "$OMOL_ROOT" ]]; then
    echo "OMol root not found: $OMOL_ROOT" >&2
    exit 1
  fi
  for V in $(ls -1 "$OMOL_ROOT" | sort); do
    ROOT="$OMOL_ROOT/$V"
    LMDB_FILE="$ROOT/structure.lmdb"
    if [[ ! -f "$LMDB_FILE" ]]; then
      echo "  skip $V (no structure.lmdb)"
      continue
    fi
    TOTAL=$(python -c "
import lmdb
e=lmdb.open('$LMDB_FILE',subdir=False,readonly=True,lock=False,readahead=False)
print(e.stat()['entries']);e.close()")
    N=$(python -c "
import math
print(max(1, math.ceil($TOTAL * $OMOL_SAMPLE_FRAC / 100)))")
    echo "  vertical=$V total=$TOTAL n=$N"
    run analysis-soap-featurize --source omol --root "$ROOT" \
      --output "$OUTPUT_ROOT/omol/soap_${V}.parquet" \
      --n-sample "$N" --n-jobs "$N_JOBS" --seed "$SEED" --no-progress
  done
fi

echo "=== summary ==="
ls -lah "$OUTPUT_ROOT"/*/soap*.parquet 2>/dev/null || true
