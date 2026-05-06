#!/usr/bin/env bash
# bond_agreement_local.sh -- parallel per-vertical bond agreement, then merge
#
# Usage:
#   ROOT=/path/to/lmdbs bash bond_agreement_local.sh
#   JOBS=8 ROOT=... bash bond_agreement_local.sh
#   ROOT=... SHARDS=/tmp/ba_shards FINAL=/tmp/ba_out bash bond_agreement_local.sh
#
# Optional env vars:
#   ROOT    - directory containing per-vertical LMDB subdirs (default below)
#   SHARDS  - where per-vertical parquets land (default: <ROOT>/../ba_shards)
#   FINAL   - output directory for combined_ba_agg.parquet etc. (default: <ROOT>/../ba_final)
#   JOBS    - max parallel workers (default: nproc)
#   BO_THRESHOLD  - bond-order threshold (default: 0.5)
#   POOL_MULT     - candidate pair pool multiplier (default: 1.4)
#   GEOM_K        - geometric bond threshold multiplier (default: 1.3)

set -euo pipefail

ROOT=${ROOT:-/p/lustre5/vargas58/converters/converters_final}
SHARDS=${SHARDS:-$(dirname "$ROOT")/ba_shards}
FINAL=${FINAL:-$(dirname "$ROOT")/ba_final}
LOGS=${LOGS:-$SHARDS/logs}
JOBS=${JOBS:-$(nproc)}
BO_THRESHOLD=${BO_THRESHOLD:-0.5}
POOL_MULT=${POOL_MULT:-1.4}
GEOM_K=${GEOM_K:-1.3}

mkdir -p "$SHARDS" "$LOGS" "$FINAL"

VERTICALS=(
    omol tm_react ml_mo electrolytes_reactivity low_spin_23
    electrolytes_redox electrolytes_scaled_sep ml_elytes 5A_elytes
    pmechdb scaled_separations_exp mo_hydrides droplet rna dna nakb
    pdb_pockets_300K pdb_pockets_400K pdb_fragments_300K pdb_fragments_400K
    protein_core protein_interface ml_protein_interface
    noble_gas noble_gas_compounds rmechdb
    ani1xbb ani2x trans1x geom_orca6 rgd_uks orbnet_denali rpmd spice
)

echo "[bond-agreement] root=$ROOT"
echo "[bond-agreement] shards=$SHARDS  final=$FINAL"
echo "[bond-agreement] verticals=${#VERTICALS[@]}  jobs=$JOBS"
echo "[bond-agreement] bo_threshold=$BO_THRESHOLD  pool_mult=$POOL_MULT  geom_k=$GEOM_K"

n=0
for V in "${VERTICALS[@]}"; do
    OUT="$SHARDS/${V}_ba.parquet"
    # Skip if already done (non-empty shard exists)
    if [[ -f "$OUT" ]] && [[ -s "$OUT" ]]; then
        echo "[skip] $V"
        continue
    fi
    # Skip if the vertical directory doesn't exist under ROOT
    if [[ ! -d "$ROOT/$V" ]]; then
        echo "[missing] $V (no dir at $ROOT/$V)" >&2
        continue
    fi

    analysis-bond-agreement \
        --root "$ROOT/$V" \
        --output "$OUT" \
        --bo-threshold "$BO_THRESHOLD" \
        --pool-multiplier "$POOL_MULT" \
        --geom-k "$GEOM_K" \
        --no-progress \
        >"$LOGS/$V.out" 2>"$LOGS/$V.err" \
        && echo "[done] $V" || echo "[FAIL] $V (see $LOGS/$V.err)" >&2 &

    n=$((n + 1))
    if (( n >= JOBS )); then
        wait -n
        n=$((n - 1))
    fi
done

wait
echo "[bond-agreement] all verticals done"

# Merge per-vertical agg parquets into combined outputs.
# Reads only the small *_agg.parquet files (not the huge per-pair parquets).
SHARDS_GLOB=("$SHARDS"/*_ba.parquet)
if (( ${#SHARDS_GLOB[@]} == 0 )); then
    echo "[bond-agreement] no shards found under $SHARDS, nothing to merge" >&2
    exit 1
fi

echo "[bond-agreement] merging ${#SHARDS_GLOB[@]} shards -> $FINAL"
analysis-bond-agreement \
    --merge_from "${SHARDS_GLOB[@]}" \
    --output "$FINAL"

echo "[bond-agreement] combined_ba_agg.parquet -> $FINAL/combined_ba_agg.parquet"
