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
# kv-metadata. Default species is the comparator union of 12 elements
# (HBCNOFSiPSClBrI, Z=1,5,6,7,8,9,14,15,16,17,35,53), dim=32,592. Override
# with --species "1,6,7,8" etc.
#
# Examples:
#   bash scripts/run_soap_featurization.sh
#   bash scripts/run_soap_featurization.sh --n-cap 0.1            # 10% per source
#   bash scripts/run_soap_featurization.sh --vertical droplet --vertical rmechdb
#   bash scripts/run_soap_featurization.sh --parallel 4 --n-jobs 8
#   bash scripts/run_soap_featurization.sh --species "1,6,7,8"

set -euo pipefail

OMOL_ROOT="data/OMol4M_lmdbs"
COMPARATORS_ROOT="data/comparators"
OUTPUT_ROOT="data/comparators"
N_CAP="25000"
N_JOBS=8
PARALLEL=1
OMOL_SAMPLE_FRAC=10           # percent
SEED=42
SOURCES_DEFAULT="schnet4aim qm7x pcqm4mv2 qmugs omol"
SOURCES="$SOURCES_DEFAULT"
SPECIES=""                    # forwarded to CLI when non-empty
VERTICALS=()                  # explicit vertical list; empty = auto-discover
DRY_RUN=0

# Hard-coded total record counts per comparator (for fraction mode). Update
# if you re-subsample upstream. Values reflect the on-disk subsamples we
# already document in data/comparators/README.md.
declare -A COMPARATOR_TOTALS=(
  [schnet4aim]=5925
  [qm7x]=418821
  [pcqm4mv2]=337861
  [qmugs]=199298
)

usage() {
  cat <<EOF
Usage: $0 [options]

Defaults shown in [].

  --omol-root PATH         Root containing OMol verticals [data/OMol4M_lmdbs]
  --comparators-root PATH  Root with comparator raw/ subdirs [data/comparators]
  --output-root PATH       Where to write soap.parquet files [data/comparators]
  --n-cap VALUE            Per comparator: integer >=1 OR fraction 0<x<1 of
                           that source's known total [25000].
  --n-jobs N               dscribe SOAP n_jobs per process [8]
  --parallel N             Run up to N source/vertical jobs concurrently.
                           Total CPU = parallel * n_jobs [1]
  --omol-sample-frac PCT   Percent of each OMol vertical to take [10]
                           Ignored if --n-cap is a fraction (then OMol gets
                           the same fraction).
  --seed N                 Random seed for sampling [42]
  --sources LIST           Whitespace-separated subset of
                           {schnet4aim,qm7x,pcqm4mv2,qmugs,omol}
                           [$SOURCES_DEFAULT]
  --vertical NAME          Restrict OMol to this vertical. Repeatable. If
                           omitted, auto-discovers every <omol-root>/*/structure.lmdb
  --species "Z1,Z2,..."    Override the locked 12-element species set
  --dry-run                Print invocations without running them
  -h, --help               This help

Notes on --n-cap:
  --n-cap 25000   absolute count
  --n-cap 0.1     10% of the source's total (also applied to OMol per-vertical)
  --n-cap 1       absolute 1 record (NOT 100%; pass 1.0 if you want all)

Notes on --parallel:
  Each parallel job spawns its own dscribe SOAP with --n-jobs threads.
  --parallel 4 --n-jobs 8 saturates 32 cores.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --omol-root) OMOL_ROOT="$2"; shift 2 ;;
    --comparators-root) COMPARATORS_ROOT="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --n-cap) N_CAP="$2"; shift 2 ;;
    --n-jobs) N_JOBS="$2"; shift 2 ;;
    --parallel) PARALLEL="$2"; shift 2 ;;
    --omol-sample-frac) OMOL_SAMPLE_FRAC="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --sources) SOURCES="$2"; shift 2 ;;
    --vertical) VERTICALS+=("$2"); shift 2 ;;
    --species) SPECIES="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

# Resolve --n-cap into either an integer count or a fraction. We treat any
# value matching `^0\.[0-9]+$` (or `^[0-9]\.[0-9]+$` excluding 1.0) as a
# fraction; integers are absolute counts.
is_fraction() {
  python -c "
v = '$1'
try:
    f = float(v)
    print('1' if 0 < f < 1 else '0')
except Exception:
    print('0')"
}
N_CAP_IS_FRACTION=$(is_fraction "$N_CAP")

resolve_count() {
  # $1 = source name; echoes the integer count to use for --n-sample. Empty
  # output means \"do not pass --n-sample\" (e.g. schnet4aim full at integer mode).
  local src="$1"
  if [[ "$N_CAP_IS_FRACTION" == "1" ]]; then
    local total="${COMPARATOR_TOTALS[$src]:-0}"
    if [[ "$total" -le 0 ]]; then
      echo ""
      return
    fi
    python -c "import math; print(max(1, math.ceil($total * $N_CAP)))"
  else
    # absolute integer
    if [[ "$src" == "schnet4aim" && "$N_CAP" -ge "${COMPARATOR_TOTALS[schnet4aim]}" ]]; then
      # cap exceeds total - skip flag
      echo ""
      return
    fi
    echo "$N_CAP"
  fi
}

# Job runner: executes either inline (PARALLEL=1) or backgrounds and waits
# for a slot when PARALLEL>1.
queue() {
  if [[ "$PARALLEL" -le 1 ]]; then
    "$@"
  else
    while (( $(jobs -rp | wc -l) >= PARALLEL )); do
      wait -n || true
    done
    "$@" &
  fi
}

invoke() {
  echo "+ $*"
  if [[ "$DRY_RUN" -eq 0 ]]; then
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

species_args() {
  if [[ -n "$SPECIES" ]]; then
    printf -- '--species %s' "$SPECIES"
  fi
}

mkdir -p "$OUTPUT_ROOT/omol"

run_comparator() {
  local src="$1"
  local count
  count=$(resolve_count "$src")
  local sample_args=()
  if [[ -n "$count" ]]; then
    sample_args=(--n-sample "$count")
  fi
  local species_flag=()
  if [[ -n "$SPECIES" ]]; then
    species_flag=(--species "$SPECIES")
  fi
  echo "=== $src (count=${count:-full}) ==="
  invoke analysis-soap-featurize --source "$src" \
    --root "$COMPARATORS_ROOT/$src/raw" \
    --output "$OUTPUT_ROOT/$src/soap.parquet" \
    "${sample_args[@]}" "${species_flag[@]}" \
    --n-jobs "$N_JOBS" --seed "$SEED" --no-progress
}

run_omol_vertical() {
  local v="$1"
  local root="$OMOL_ROOT/$v"
  local lmdb_file="$root/structure.lmdb"
  if [[ ! -f "$lmdb_file" ]]; then
    echo "  skip $v (no structure.lmdb)"
    return 0
  fi
  local total
  total=$(python -c "
import lmdb
e=lmdb.open('$lmdb_file',subdir=False,readonly=True,lock=False,readahead=False)
print(e.stat()['entries']);e.close()")
  local n
  if [[ "$N_CAP_IS_FRACTION" == "1" ]]; then
    n=$(python -c "import math; print(max(1, math.ceil($total * $N_CAP)))")
  else
    n=$(python -c "import math; print(max(1, math.ceil($total * $OMOL_SAMPLE_FRAC / 100)))")
  fi
  echo "  vertical=$v total=$total n=$n"
  local species_flag=()
  if [[ -n "$SPECIES" ]]; then
    species_flag=(--species "$SPECIES")
  fi
  invoke analysis-soap-featurize --source omol --root "$root" \
    --output "$OUTPUT_ROOT/omol/soap_${v}.parquet" \
    --n-sample "$n" "${species_flag[@]}" \
    --n-jobs "$N_JOBS" --seed "$SEED" --no-progress
}

# Comparators
for src in schnet4aim qm7x pcqm4mv2 qmugs; do
  if want "$src"; then
    queue run_comparator "$src"
  fi
done

# OMol verticals
if want omol; then
  if [[ ! -d "$OMOL_ROOT" ]]; then
    echo "OMol root not found: $OMOL_ROOT" >&2
    exit 1
  fi
  if [[ "${#VERTICALS[@]}" -gt 0 ]]; then
    target_verticals=("${VERTICALS[@]}")
  else
    mapfile -t target_verticals < <(ls -1 "$OMOL_ROOT" | sort)
  fi
  for v in "${target_verticals[@]}"; do
    queue run_omol_vertical "$v"
  done
fi

# Wait for any remaining background jobs.
if [[ "$PARALLEL" -gt 1 ]]; then
  wait
fi

echo "=== summary ==="
ls -lah "$OUTPUT_ROOT"/*/soap*.parquet 2>/dev/null || true
