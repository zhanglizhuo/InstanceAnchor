#!/bin/bash
# COCO extraction on V100
set -e
cd ~/InstanceAnchor
mkdir -p ~/data/coco

echo "=== Step 1: Download COCO annotations ==="
python3 -c "
import urllib.request, zipfile, os
coco_dir = os.path.expanduser('~/data/coco')
os.makedirs(coco_dir, exist_ok=True)
ann_path = os.path.join(coco_dir, 'annotations.zip')
urllib.request.urlretrieve('http://images.cocodataset.org/annotations/annotations_trainval2017.zip', ann_path)
with zipfile.ZipFile(ann_path) as zf: zf.extractall(coco_dir)
print('Annotations OK')
"

echo "=== Step 2: Download COCO val2017 images ==="
python3 -c "
import urllib.request, zipfile, os
coco_dir = os.path.expanduser('~/data/coco')
val_path = os.path.join(coco_dir, 'val2017.zip')
urllib.request.urlretrieve('http://images.cocodataset.org/zips/val2017.zip', val_path)
import subprocess
subprocess.run(['unzip', '-q', '-o', val_path, '-d', coco_dir])
print('Val images OK')
"

echo "=== Step 3: Extract CLIP logits ==="
export HF_HUB_OFFLINE=1 PYTHONPATH=~/scb5-zeroshot
python experiments/extract_clip_per_image_v2.py --dataset coco --model all --data_root ~/data

echo "=== Done ==="
ls -lh results/per_image/clip_*_coco_*.npy
