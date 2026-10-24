#!/usr/bin/env bash
# run_mllm_v2.sh — Full per-image MLLM inference + CLIP extraction pipeline
# ============================================================================
# Phase 1: Extract per-image CLIP logits (fast, batch inference ~5min/model)
# Phase 2: Run MLLM inference (slow, Ollama API, hours/model)
# Phase 3: Per-image complementarity analysis
#
# Usage:
#   # Set models to run (comma-separated)
#   MLLM_MODELS="qwen3.5:27b,gemma4:26b,qwen3.5:9b" bash experiments/run_mllm_v2.sh
#
#   # Or edit default below
#   bash experiments/run_mllm_v2.sh
#
# Requires:
#   - Ollama running with models pre-pulled
#   - SCB_DATA_DIR set (or default below)
#   - 4× V100 (CLIP extraction on GPU 1-3, Ollama on GPU 0)
# ============================================================================

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON_BIN:-python3}"
LOG_DIR="${ROOT}/logs_v2"
RESULTS_DIR="${ROOT}/results/per_image"

export SCB_DATA_DIR="${SCB_DATA_DIR:-/home/broadsense/works/lizhuo/AutoResearchClaw/datasets_scb}"

# ─── Config ──────────────────────────────────────────────────────────
MLLM_MODELS="${MLLM_MODELS:-qwen3.5:27b,gemma4:26b,qwen3.5:9b}"
IFS=',' read -ra MODELS <<< "$MLLM_MODELS"

echo "=============================================="
echo "  Per-Image MLLM + CLIP Pipeline (v2)"
echo "=============================================="
echo "  MLLM models: ${MODELS[*]}"
echo "  Data dir:    $SCB_DATA_DIR"
echo "  Results:     $RESULTS_DIR"
echo "=============================================="
echo ""

mkdir -p "$RESULTS_DIR" "$LOG_DIR"

# ─── Phase 1: Extract CLIP per-image logits (fast) ────────────────────
echo "[Phase 1] Extracting per-image CLIP logits (all models, all subsets)..."
echo ""

for model in openai laion siglip2 eva02 dfn; do
    log_file="$LOG_DIR/extract_clip_${model}.log"
    echo "  CLIP $model → $log_file"
    CUDA_VISIBLE_DEVICES=1 "$PYTHON" "$ROOT/experiments/extract_clip_per_image.py" \
        --model "$model" --subset all \
        > "$log_file" 2>&1 &
done
wait

echo ""
echo "[Phase 1] CLIP extraction complete."
echo ""

# ─── Phase 2: MLLM inference (slow) ───────────────────────────────────
echo "[Phase 2] Running MLLM inference..."
echo "  NOTE: This will take hours. CTRL+C to interrupt safely."
echo "  Resume is automatic — re-run to continue from last checkpoint."
echo ""

for model in "${MODELS[@]}"; do
    sanitized="${model//:/_}"
    log_file="$LOG_DIR/mllm_v2_${sanitized}.log"

    echo "  Starting $model (log: $log_file)"
    echo "  =========================================="

    # Run all 3 subsets sequentially per model
    for subset in teacher_behavior handrise_readwrite bow_turnhead; do
        echo "    [$model] $subset"
        "$PYTHON" "$ROOT/experiments/main_mllm_v2.py" \
            --model "$model" \
            --subset "$subset" \
            2>&1 | tee -a "$log_file"
    done

    echo "  [$model] DONE"
    echo ""
done

# ─── Phase 3: Per-image complementarity analysis ──────────────────────
echo "[Phase 3] Running per-image complementarity analysis..."
echo ""

"$PYTHON" "$ROOT/experiments/analyze_per_image_complementarity.py" \
    2>&1 | tee "$LOG_DIR/analyze_per_image.log"

echo ""
echo "=============================================="
echo "  Full pipeline complete."
echo "  Results: $RESULTS_DIR"
echo "  Report:  results/per_image_complementarity_report.txt"
echo "=============================================="
