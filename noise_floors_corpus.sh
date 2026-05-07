#!/usr/bin/env bash
# noise_floors_corpus.sh -- parallel Stream F noise floors, one worker per vertical
# Usage (interactive or inside a Flux job):
#   bash noise_floors_corpus.sh
#   JOBS=16 bash noise_floors_corpus.sh   # cap parallel workers
#
# Outputs under $OUT_DIR:
#   {vertical}_nf.parquet          noise-floor table (B4)
#   {vertical}_nf_charge_atoms.parquet
#   {vertical}_nf_bond_pairs.parquet
#   {vertical}_nf_qtaim_bcps.parquet
#   {vertical}_nf_exemplars.parquet  (B5)
#   combined_nf.parquet            corpus-wide concat

set -euo pipefail

ROOT=${ROOT:-/p/lustre5/vargas58/converters/converters_final}
OUT_DIR=${OUT_DIR:-/p/lustre5/vargas58/converters/noise_floors}
LOGS=${LOGS:-$OUT_DIR/logs}
TOPK=${TOPK:-50}
JOBS=${JOBS:-$(nproc)}

mkdir -p "$OUT_DIR" "$LOGS"

VERTICALS=(
    omol tm_react ml_mo electrolytes_reactivity low_spin_23
    electrolytes_redox electrolytes_scaled_sep ml_elytes 5A_elytes
    pmechdb scaled_separations_exp mo_hydrides droplet rna dna nakb
    pdb_pockets_300K pdb_pockets_400K pdb_fragments_300K pdb_fragments_400K
    protein_core protein_interface ml_protein_interface
    noble_gas noble_gas_compounds rmechdb
    ani1xbb ani2x trans1x geom_orca6 rgd_uks orbnet_denali rpmd spice
)

echo "[noise-floors] root=$ROOT  out=$OUT_DIR  verticals=${#VERTICALS[@]}  jobs=$JOBS"

n=0
for V in "${VERTICALS[@]}"; do
    NF_PATH="$OUT_DIR/${V}_nf.parquet"
    if [[ -f "$NF_PATH" ]] && [[ -s "$NF_PATH" ]]; then
        echo "[skip] $V"
        continue
    fi
    conda run -n generator analysis-noise-floors \
        --root "$ROOT/$V" \
        --output "$NF_PATH" \
        --topk "$TOPK" \
        --no-progress \
        >"$LOGS/$V.out" 2>"$LOGS/$V.err" \
        && echo "[done] $V" || echo "[FAIL] $V" >&2 &
    n=$((n + 1))
    if (( n >= JOBS )); then
        wait -n
        n=$((n - 1))
    fi
done

wait
echo "[noise-floors] all verticals done"

# Combine into a single table
conda run -n generator python - "$OUT_DIR" <<'EOF'
import sys
from pathlib import Path
import pandas as pd

out_dir = Path(sys.argv[1])
nf_files = sorted(out_dir.glob("*_nf.parquet"))
nf_files = [f for f in nf_files if f.name != "combined_nf.parquet"]
if not nf_files:
    print("no per-vertical parquets found", file=sys.stderr)
    sys.exit(1)
combined = pd.concat([pd.read_parquet(f) for f in nf_files], ignore_index=True)
out = out_dir / "combined_nf.parquet"
combined.to_parquet(out, index=False)
print(f"combined_nf.parquet -> {out} ({len(combined)} rows, {combined['vertical'].nunique()} verticals)")
summary = combined[combined["element"].isna() & combined["element_pair"].isna()]
print(summary[["vertical","analysis","descriptor","mar","n_obs"]].to_string(index=False))
EOF
