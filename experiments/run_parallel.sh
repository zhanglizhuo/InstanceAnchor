#!/usr/bin/env bash
# run_parallel.sh — 4× V100 parallel CLIP baseline experiment (v2)
# ===============================================================
# Each GPU runs one model across all 3 subsets × 4 CAPE sets.
# 
# Usage:
#   bash experiments/run_parallel.sh
#
# Environment:
#   CAPE_SETS    default "A B C"
#   STRATEGIES   default "cape_set"  (set to "all" for all 5 strategies)
# ===============================================================

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3 || echo python3)}"
RESULTS_ROOT="${ROOT}/results/baseline"
LOG_DIR="${ROOT}/logs_v2"
CAPE_SETS="${CAPE_SETS:-A B C}"
STRATEGIES="${STRATEGIES:-cape_set}"

mkdir -p "$RESULTS_ROOT" "$LOG_DIR"

if [ "$STRATEGIES" = "all" ]; then
    STRAT_FLAG="--strategies label_only simple action detailed cape_set"
else
    STRAT_FLAG="--strategies $STRATEGIES"
fi

echo "Using Python: $PYTHON_BIN"
echo "CAPE sets: $CAPE_SETS"
echo "Strategies: $STRAT_FLAG"
echo ""

# Model → GPU mapping (each GPU gets one model)
declare -A MODEL_GPU_MAP=(
    ["openai"]="0"
    ["dfn"]="1"
    ["laion"]="2"
    ["eva02"]="3"
)
# siglip2 needs ~3.5GB model + features → put on GPU0 after openai or skip
# We'll run it after the main 4 complete.

pids=()

for model in "${!MODEL_GPU_MAP[@]}"; do
    gpu="${MODEL_GPU_MAP[$model]}"
    log_file="$LOG_DIR/main_clip_${model}.log"

    echo "[GPU $gpu] $model → log: $log_file"
    (
        export CUDA_VISIBLE_DEVICES="$gpu"
        export SCB_DATA_DIR="/home/broadsense/works/lizhuo/AutoResearchClaw/datasets_scb"
        cd "$ROOT"
        "$PYTHON_BIN" experiments/main_clip.py \
            --results_dir "$RESULTS_ROOT/${model}" \
            --cape_sets $CAPE_SETS \
            $STRAT_FLAG \
            2>&1 </dev/null | tee "$log_file"
    ) &
    pids+=("$!")
    # Stagger model launches by 30s to avoid simultaneous model loads
    sleep 30
done

# Wait for main 4
failed=0
for idx in "${!pids[@]}"; do
    if wait "${pids[$idx]}"; then
        echo "[${!MODEL_GPU_MAP[$idx]}] done"
    else
        echo "[${!MODEL_GPU_MAP[$idx]}] FAILED (check logs)"
        failed=1
    fi
done

# siglip2 is large (~3.5GB model). Run on GPU0 after openai finishes.
echo ""
echo "[GPU 0] siglip2 (after openai)"
log_file="$LOG_DIR/main_clip_siglip2.log"
export CUDA_VISIBLE_DEVICES="0"
export SCB_DATA_DIR="/home/broadsense/works/lizhuo/AutoResearchClaw/datasets_scb"
cd "$ROOT"
"$PYTHON_BIN" experiments/main_clip.py \
    --results_dir "$RESULTS_ROOT/siglip2" \
    --cape_sets $CAPE_SETS \
    $STRAT_FLAG \
    2>&1 </dev/null | tee "$log_file"

echo ""
echo "=============================================="
echo "All experiments complete."
echo "Individual results: $RESULTS_ROOT/*/baseline_results.json"
echo ""
echo "To merge:"
echo "  python -c \"import json,os; \\
merged={}; \\
for d in os.listdir('$RESULTS_ROOT'): \\
    p='$RESULTS_ROOT/'+d+'/baseline_results.json'; \\
    if os.path.exists(p): \\
        with open(p) as f: data=json.load(f); \\
        for s,m in data.items(): \\
            merged.setdefault(s,{'class_names':m.get('class_names',[])}); \\
            for k,v in m.items(): \\
                if k!='class_names': merged[s][k]=v; \\
print(json.dumps(merged,indent=2))\" \\
  > $RESULTS_ROOT/merged_baseline_results.json"
echo "=============================================="
exit "$failed"
