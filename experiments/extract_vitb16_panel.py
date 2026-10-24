#!/usr/bin/env python3
"""
experiments/extract_vitb16_panel.py — 2x2 cell D: architecture-fixed, data-shared
=================================================================================
Completes the mechanism-attribution 2x2 (paper sec:nonclip). Cell D holds BOTH
architecture (ViT-B/16) and training-data family (ImageNet-supervised) fixed,
varying only the training recipe:

    tv_vitb16            torchvision ViT_B_16, IMAGENET1K_V1 (augreg recipe)
    timm_vitb16_augreg_in1k   timm vit_base_patch16_224.augreg_in1k (IN-1k)
    timm_vitb16_augreg_in21k_ft  timm vit_base_patch16_224.augreg_in21k_ft_in1k
    timm_vitb16_augreg2_in21k_ft timm vit_base_patch16_224.augreg2_in21k_ft_in1k
    timm_vitb16_orig_in21k_ft   timm vit_base_patch16_224.orig_in21k_ft_in1k (JFT+IN-21k+IN-1k)

All five are ViT-B/16 with 1000-class ImageNet heads, varying only the
training recipe. Honest caveat (stated in the paper): the IN-21k variants
pre-train on additional data, so "data-shared" holds for the supervised
classification head but not the full pretraining lineage. (The SWSL variant
was dropped: its timm tag was removed in timm>=1.0.)

Dataset: ImageNet-V2 matched-frequency (vaishaal/ImageNetV2 tar.gz via
hf-mirror), 10,000 images, numeric class-dir labels (same loader as
extract_nonclip_per_image.py).

Output: results/per_image/vitb16_panel/{model}_imagenet_v2_matched_{logits,labels}.npy

Usage (5090, llava env):
  ~/miniconda3/envs/llava/bin/python experiments/extract_vitb16_panel.py
"""
import argparse
import io
import json
import logging
import os
import sys
import tarfile
import time
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_OFFLINE", "0")

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

MODELS = {
    "tv_vitb16": ("torchvision", "vit_b_16", "IMAGENET1K_V1", None),
    "timm_vitb16_augreg_in1k": ("timm", "vit_base_patch16_224.augreg_in1k", None, None),
    "timm_vitb16_augreg_in21k_ft": ("timm", "vit_base_patch16_224.augreg_in21k_ft_in1k", None, None),
    "timm_vitb16_augreg2_in21k_ft": ("timm", "vit_base_patch16_224.augreg2_in21k_ft_in1k", None, None),
    "timm_vitb16_orig_in21k_ft": ("timm", "vit_base_patch16_224.orig_in21k_ft_in1k", None, None),
}
DATASET = "imagenet_v2_matched"
OUT_ROOT = Path(__file__).resolve().parents[1] / "results" / "per_image" / "vitb16_panel"
DL_ROOT = Path(__file__).resolve().parents[1] / "data" / "vitb16_panel_dl"


def setup_logging(ts):
    logdir = Path(__file__).resolve().parents[1] / "logs"
    logdir.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(logdir / f"extract_vitb16_panel_{ts}.log"),
                  logging.StreamHandler()])
    return logging.getLogger()


def get_model(key, device):
    src, name, tv_weights, _ = MODELS[key]
    if src == "torchvision":
        import torchvision.models as tvm
        fn = getattr(tvm, name)
        weights = getattr(tvm, f"ViT_B_16_Weights")[tv_weights]
        model = fn(weights=weights).to(device).eval()
        transform = weights.transforms()
    else:
        import timm
        model = timm.create_model(name, pretrained=True).to(device).eval()
        data_cfg = timm.data.resolve_data_config({}, model=model)
        from timm.data import create_transform
        transform = create_transform(**data_cfg)
    return model, transform


def iter_images_from_tar(tar_path, log):
    with tarfile.open(tar_path) as tf:
        members = [m for m in tf.getmembers()
                   if m.isfile()
                   and m.name.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))]
        members.sort(key=lambda m: m.name)
        log.info(f"  {len(members)} images in archive")
        for m in members:
            parts = m.name.split("/")
            cls = None
            for part in parts[:-1]:
                if part.isdigit() and 0 <= int(part) < 1000:
                    cls = int(part)
                    break
            if cls is None:
                raise RuntimeError(f"no class dir in {m.name}")
            data = tf.extractfile(m).read()
            yield cls, Image.open(io.BytesIO(data)).convert("RGB")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=list(MODELS.keys()))
    parser.add_argument("--batch", type=int, default=64)
    args = parser.parse_args()
    ts = time.strftime("%Y%m%d_%H%M%S")
    log = setup_logging(ts)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # download dataset
    from huggingface_hub import hf_hub_download
    DL_ROOT.mkdir(parents=True, exist_ok=True)
    tar_path = hf_hub_download("vaishaal/ImageNetV2",
                               "imagenetv2-matched-frequency.tar.gz",
                               repo_type="dataset", cache_dir=str(DL_ROOT / "hf_cache"))
    log.info(f"dataset -> {tar_path} ({os.path.getsize(tar_path)/1e6:.0f} MB)")

    accs = {}
    for key in args.models:
        out = OUT_ROOT / f"{key}_{DATASET}_logits.npy"
        if out.exists():
            existing = np.load(out)
            acc = float((existing.argmax(1) == np.load(
                OUT_ROOT / f"{key}_{DATASET}_labels.npy")).mean())
            accs[key] = acc
            log.info(f"skip {key} (exists, top1={acc:.4f})")
            continue
        model, transform = get_model(key, device)
        chunks, labels_all = [], []
        n_done, t0 = 0, time.time()
        with torch.no_grad():
            buf_x, buf_y = [], []
            for cls, img in iter_images_from_tar(tar_path, log):
                buf_x.append(transform(img))
                buf_y.append(cls)
                if len(buf_x) == args.batch:
                    inp = torch.stack(buf_x).to(device)
                    logits = model(inp)
                    chunks.append(logits.float().cpu().numpy())
                    labels_all.append(np.array(buf_y, dtype=np.int32))
                    n_done += len(buf_y)
                    buf_x, buf_y = [], []
                    if n_done % (args.batch * 40) == 0:
                        log.info(f"  {key}: {n_done} images ({time.time()-t0:.0f}s)")
            if buf_x:
                inp = torch.stack(buf_x).to(device)
                chunks.append(model(inp).float().cpu().numpy())
                labels_all.append(np.array(buf_y, dtype=np.int32))
                n_done += len(buf_y)
        logits = np.concatenate(chunks).astype(np.float32)
        labels = np.concatenate(labels_all)
        np.save(out, logits)
        np.save(OUT_ROOT / f"{key}_{DATASET}_labels.npy", labels)
        acc = float((logits.argmax(1) == labels).mean())
        accs[key] = acc
        log.info(f"  {key} done: {logits.shape} top1={acc:.4f} ({time.time()-t0:.0f}s)")
        del model
        torch.cuda.empty_cache()
    log.info("per-model top1: " + json.dumps(accs, indent=2))


if __name__ == "__main__":
    main()
