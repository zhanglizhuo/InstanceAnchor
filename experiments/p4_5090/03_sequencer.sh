#!/bin/bash
# experiments/p4_5090/03_sequencer.sh — autonomous: wait downloads -> extract -> signal
# Run on 5090: nohup setsid bash 03_sequencer.sh </dev/null >/dev/null 2>&1 &
LOG=~/ia_p4/logs/sequencer.log
log(){ echo "[$(date '+%m-%d %T')] $*" >> $LOG; }
log 'sequencer started; waiting for model + data downloads'
while pgrep -f 'dl_models.sh' >/dev/null; do sleep 60; done
log 'model download finished'
while pgrep -f 'run_dl_data.sh' >/dev/null; do sleep 60; done
log 'data download finished'
grep -q ALL_DATA_DONE ~/ia_p4/logs/dl_data.log 2>/dev/null || log 'WARN: data dl marker missing'
cd ~/ia_p4
log 'launching extraction (single GPU, all datasets, batch 96)'
~/ia_p4/venv/bin/python experiments/extract_clip_meta.py --datasets all --device cuda:0 --batch 96 > logs/extract.log 2>&1
log "extraction finished, rc=$?"
log 'np file count:' $(ls results/per_image/meta/*.npy 2>/dev/null | wc -l)
log 'P4_5090_EXTRACTION_DONE'
