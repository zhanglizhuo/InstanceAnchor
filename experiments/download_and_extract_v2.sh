#!/usr/bin/env bash
# download_and_extract_v2.sh — Download VOC2007 + Stanford40, then run extraction
# Usage: nohup bash experiments/download_and_extract_v2.sh > logs/download_v2.log 2>&1 &
set -euo pipefail

DATA_DIR="/home/broadsense/works/lizhuo/data"
REPO_DIR="/home/broadsense/works/lizhuo/InstanceAnchor"
LOG_FILE="$REPO_DIR/logs/download_v2_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$DATA_DIR" "$REPO_DIR/logs"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ── Step 1: Download VOC2007 ──────────────────────────────────────
VOC_TAR="$DATA_DIR/VOCtrainval_06-Nov-2007.tar"
if [ -f "$VOC_TAR" ] && [ $(stat -c%s "$VOC_TAR") -gt 400000000 ]; then
    log "[VOC2007] already exists, skip download"
else
    log "[VOC2007] downloading (this may take hours, ~10KB/s)..."
    rm -f "$VOC_TAR"
    wget -q --show-progress \
        "http://host.robots.ox.ac.uk:8080/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar" \
        -O "$VOC_TAR"
    log "[VOC2007] download complete ($(stat -c%s $VOC_TAR) bytes)"
fi

# ── Step 2: Download Stanford40 ───────────────────────────────────
STANFORD_ZIP="$DATA_DIR/Stanford40_JPEGImages.zip"
if [ -f "$STANFORD_ZIP" ] && [ $(stat -c%s "$STANFORD_ZIP") -gt 290000000 ]; then
    log "[Stanford40] already exists, skip download"
else
    log "[Stanford40] downloading images..."
    rm -f "$STANFORD_ZIP"
    # Official Stanford URL (HTTP, ~298MB)
    wget -q --show-progress \
        "http://vision.stanford.edu/Datasets/Stanford40_JPEGImages.zip" \
        -O "$STANFORD_ZIP"
    log "[Stanford40] download complete ($(stat -c%s $STANFORD_ZIP) bytes)"
fi

# ── Step 3: Check extraction ──────────────────────────────────────
# VOC2007: torchvision handles extraction, we just need the tar present
if [ -f "$VOC_TAR" ]; then
    log "[VOC2007] tar ready at $VOC_TAR"
fi

# Stanford40: unzip JPEGImages/ if not already
if [ -f "$STANFORD_ZIP" ] && [ ! -d "$DATA_DIR/JPEGImages" ]; then
    log "[Stanford40] extracting..."
    cd "$DATA_DIR"
    unzip -q -o "$STANFORD_ZIP"
    log "[Stanford40] extracted to $DATA_DIR/JPEGImages/"
fi

# ── Step 4: Run CLIP extraction ───────────────────────────────────
log "Starting CLIP extraction for VOC2007 + Stanford40 (5 backbones)..."
cd "$REPO_DIR"

# extract all 5 models × 2 datasets
for dataset in voc2007 stanford40; do
    log "--- dataset: $dataset ---"
    for model in openai laion siglip2 eva02 dfn; do
        out_file="$REPO_DIR/results/per_image/clip_${model}_${dataset}_logits.npy"
        if [ -f "$out_file" ]; then
            log "  [SKIP] $model / $dataset already exists"
            continue
        fi
        log "  [RUN]  $model / $dataset"
        python experiments/extract_clip_per_image_v2.py \
            --dataset "$dataset" --model "$model" \
            --data_root "$DATA_DIR" 2>&1
        log "  [DONE] $model / $dataset (exit=$?)"
    done
done

log "All done!"
