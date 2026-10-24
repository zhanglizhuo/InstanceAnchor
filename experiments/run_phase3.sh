#!/usr/bin/env bash
# run_phase3.sh — Phase 3 cross-family extraction on V100
# nohup bash experiments/run_phase3.sh > logs/phase3_nohup.log 2>&1 &
set -euo pipefail

REPO_DIR="/home/broadsense/works/lizhuo/InstanceAnchor"
cd "$REPO_DIR"
mkdir -p logs

export SCB_DATA_DIR="/home/broadsense/works/lizhuo/AutoResearchClaw/datasets_scb"
export HF_HUB_OFFLINE=1

log() { echo "[$(date '+%H:%M:%S')] $*"; }

log "Phase 3: cross-family extraction (DINOv2 + BLIP-2 on SCB5)..."
python experiments/extract_crossfamily.py --model all --subset all 2>&1
log "Phase 3 done."
