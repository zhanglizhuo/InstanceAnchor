#!/bin/bash
# run_siglip2_cape_bc.sh — Extract siglip2 CAPE B/C logits on V100
# Run on V100: bash run_siglip2_cape_bc.sh
set -e

cd ~/InstanceAnchor

# V100 has no external network; use offline HF cache
export HF_HUB_OFFLINE=1

echo "=== SigLIP2 CAPE-B extraction ==="
python experiments/extract_clip_per_image.py \
    --model siglip2 --subset all --cape_set B

echo "=== SigLIP2 CAPE-C extraction ==="  
python experiments/extract_clip_per_image.py \
    --model siglip2 --subset all --cape_set C

echo "=== Done. Files in results/per_image/ ==="
ls -lh results/per_image/clip_siglip2_*_cape*.npy
