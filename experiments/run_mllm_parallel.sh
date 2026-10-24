#!/usr/bin/env bash
# run_mllm_parallel.sh — 4-GPU parallel MLLM inference
# ====================================================
# Launches 4 Ollama instances (one per GPU), then runs MLLM inference
# in parallel across all subsets.
#
# Usage:
#   MLLM_MODELS="qwen3.5:27b,gemma4:26b,qwen3.5:9b" bash experiments/run_mllm_parallel.sh
#
# Each model gets its own Ollama instance on its own GPU.
# Inference is parallelized per model (sequential within model across subsets).
#
# IMPORTANT: Kill existing Ollama first:
#   sudo systemctl stop ollama  OR  pkill ollama
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

if [ ${#MODELS[@]} -gt 4 ]; then
    echo "ERROR: max 4 models (one per GPU)"
    exit 1
fi

echo "=============================================="
echo "  Parallel 4-GPU MLLM Inference Pipeline"
echo "=============================================="
echo "  Models:  ${MODELS[*]}"
echo "  GPUs:    0-$((${#MODELS[@]} - 1))"
echo "  Data:    $SCB_DATA_DIR"
echo "  Results: $RESULTS_DIR"
echo "=============================================="
echo ""

mkdir -p "$RESULTS_DIR" "$LOG_DIR"

# ─── Kill any existing Ollama instances ──────────────────────────────
echo "[Setup] Stopping existing Ollama instances..."
pkill -f "ollama serve" 2>/dev/null || true
sleep 3

# ─── Start one Ollama instance per GPU ───────────────────────────────
OLLAMA_PIDS=()
OLLAMA_BASE_PORT=11434

for i in "${!MODELS[@]}"; do
    model="${MODELS[$i]}"
    gpu=$i
    port=$((OLLAMA_BASE_PORT + i))
    ollama_log="$LOG_DIR/ollama_gpu${gpu}_port${port}.log"

    echo "[Setup] Starting Ollama on GPU $gpu, port $port (model: $model)..."

    CUDA_VISIBLE_DEVICES=$gpu \
        OLLAMA_HOST="0.0.0.0:$port" \
        OLLAMA_NUM_PARALLEL=1 \
        OLLAMA_MAX_LOADED_MODELS=1 \
        nohup ollama serve > "$ollama_log" 2>&1 &

    OLLAMA_PIDS+=($!)

    # Wait for Ollama to be ready
    echo "  Waiting for Ollama on port $port..."
    for attempt in $(seq 1 30); do
        if curl -s "http://localhost:$port/api/tags" > /dev/null 2>&1; then
            echo "  Ollama ready on port $port"
            break
        fi
        sleep 2
    done

    # Pre-load model
    echo "  Pre-loading $model..."
    curl -s "http://localhost:$port/api/generate" \
        -d "{\"model\":\"$model\",\"prompt\":\"hi\",\"stream\":false}" \
        > /dev/null 2>&1 || true
    echo "  Model $model loaded on GPU $gpu"
done

echo ""
echo "[Setup] All Ollama instances ready."
echo ""

# ─── Run MLLM inference in parallel ──────────────────────────────────
MLLM_PIDS=()

for i in "${!MODELS[@]}"; do
    model="${MODELS[$i]}"
    port=$((OLLAMA_BASE_PORT + i))
    sanitized="${model//:/_}"
    log_file="$LOG_DIR/mllm_v2_${sanitized}.log"

    echo "[Launch] $model → GPU $i, port $port"
    (
        export OLLAMA_URL="http://localhost:$port/api/generate"
        for subset in teacher_behavior handrise_readwrite bow_turnhead; do
            echo "  [$model] $subset"
            "$PYTHON" "$ROOT/experiments/main_mllm_v2.py" \
                --model "$model" \
                --subset "$subset" \
                2>&1
        done
        echo "  [$model] DONE"
    ) > "$log_file" 2>&1 &

    MLLM_PIDS+=($!)
done

echo ""
echo "[Running] ${#MLLM_PIDS[@]} MLLM inference jobs in background."
echo "  Monitor: tail -f $LOG_DIR/mllm_v2_*.log"
echo "  Check:   curl http://localhost:11434/api/ps"
echo ""

# ─── Wait for all to finish ──────────────────────────────────────────
echo "[Wait] Waiting for all MLLM jobs to complete..."
failed=0
for i in "${!MLLM_PIDS[@]}"; do
    pid="${MLLM_PIDS[$i]}"
    model="${MODELS[$i]}"
    if wait "$pid"; then
        echo "[Done] $model completed successfully"
    else
        echo "[FAIL] $model failed (check log)"
        failed=1
    fi
done

# ─── Cleanup Ollama instances ────────────────────────────────────────
echo ""
echo "[Cleanup] Stopping Ollama instances..."
for pid in "${OLLAMA_PIDS[@]}"; do
    kill "$pid" 2>/dev/null || true
done
sleep 2

# ─── Phase 3: Per-image complementarity analysis ──────────────────────
if [ $failed -eq 0 ]; then
    echo ""
    echo "[Phase 3] Running per-image complementarity analysis..."
    "$PYTHON" "$ROOT/experiments/analyze_per_image_complementarity.py" \
        2>&1 | tee "$LOG_DIR/analyze_per_image.log"
fi

echo ""
echo "=============================================="
echo "  Pipeline complete (exit=$failed)"
echo "  Results: $RESULTS_DIR"
echo "  Report:  results/per_image_complementarity_report.txt"
echo "=============================================="
exit $failed
