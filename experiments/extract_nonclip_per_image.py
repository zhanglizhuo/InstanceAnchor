#!/usr/bin/env python3
"""
experiments/extract_nonclip_per_image.py — Non-CLIP (ImageNet-supervised)
ensemble extraction for cross-family kappa replication
==========================================================================
Goal: test whether the plurality ceiling (kappa decomposition, 54-108%
plurality share for the CLIP family) replicates on a NON-CLIP ensemble of
five architecturally diverse ImageNet-supervised classifiers:

    resnet50         (ResNet-50, residual CNN, IMAGENET1K_V2)
    vit_b_16         (ViT-B/16, pure transformer, IMAGENET1K_V1)
    convnext_base    (ConvNeXt-B, modernized CNN, IMAGENET1K_V1)
    swin_b           (Swin-B, hierarchical transformer, IMAGENET1K_V1)
    efficientnet_b4  (EfficientNet-B4, compound-scaled CNN, IMAGENET1K_V1)

Dataset: ImageNet-V2 variants (vaishaal/ImageNetV2 tar.gz; 1000 numeric
class dirs, label = dir index; identical layout to extract_clip_ood.py's
index_dir mode). Default: imagenet_v2_matched (n=10,000).

Protocol notes (differences vs CLIP extraction, by design):
  - logits are RAW model outputs (unbounded), NOT cosine similarities;
    downstream softmax signals therefore use T=1 (see analysis script).
  - each model uses its own canonical weights.transforms() preprocessing
    (per-model ImageNet evaluation protocol).

Output:
  results/per_image/nonclip/nonclip_{model}_{dataset}_logits.npy (N,1000) f32
  results/per_image/nonclip/nonclip_{model}_{dataset}_labels.npy (N,) int32

Usage (on the V100 box, internet reachable):
  nohup python experiments/extract_nonclip_per_image.py \
      --datasets imagenet_v2_matched imagenet_v2_top > logs/nonclip_extract.log 2>&1 &
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

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

MODELS = {
    # key -> (builder func, weights enum class, weights name)
    "resnet50":        ("resnet50",        "ResNet50_Weights",        "IMAGENET1K_V2"),
    "vit_b_16":        ("vit_b_16",        "ViT_B_16_Weights",        "IMAGENET1K_V1"),
    "convnext_base":   ("convnext_base",   "ConvNeXt_Base_Weights",   "IMAGENET1K_V1"),
    "swin_b":          ("swin_b",          "Swin_B_Weights",          "IMAGENET1K_V1"),
    "efficientnet_b4": ("efficientnet_b4", "EfficientNet_B4_Weights", "IMAGENET1K_V1"),
}

DATASETS = {
    "imagenet_v2_matched": {
        "repo": "vaishaal/ImageNetV2",
        "file": "imagenetv2-matched-frequency.tar.gz",
    },
    "imagenet_v2_threshold": {
        "repo": "vaishaal/ImageNetV2",
        "file": "imagenetv2-threshold0.7.tar.gz",
    },
    "imagenet_v2_top": {
        "repo": "vaishaal/ImageNetV2",
        "file": "imagenetv2-top-images.tar.gz",
    },
}

OUT_ROOT = Path(__file__).resolve().parents[1] / "results" / "per_image" / "nonclip"
DL_ROOT = Path(__file__).resolve().parents[1] / "data" / "nonclip_dl"


def setup_logging(ts):
    logdir = Path(__file__).resolve().parents[1] / "logs"
    logdir.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(logdir / f"extract_nonclip_{ts}.log"),
                  logging.StreamHandler()])
    return logging.getLogger()


def download_dataset(key, log):
    from huggingface_hub import hf_hub_download
    cfg = DATASETS[key]
    DL_ROOT.mkdir(parents=True, exist_ok=True)
    p = hf_hub_download(cfg["repo"], cfg["file"], repo_type="dataset",
                        cache_dir=str(DL_ROOT / "hf_cache"))
    log.info(f"  dataset {key} -> {p} ({os.path.getsize(p)/1e6:.0f} MB)")
    return p


def iter_images_from_tar(tar_path, log):
    """Yield (class_index, PIL.Image) in a deterministic global sorted order.
    Layout: <prefix>/<int 0-999>/<hash>.<ext>."""
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


def iter_batches(images_iter, transform, batch, device):
    """Batch (tensor, labels) pairs from the image stream."""
    buf_x, buf_y = [], []
    for cls, img in images_iter:
        buf_x.append(transform(img))
        buf_y.append(cls)
        if len(buf_x) == batch:
            yield torch.stack(buf_x).to(device), torch.tensor(buf_y, dtype=torch.long)
            buf_x, buf_y = [], []
    if buf_x:
        yield torch.stack(buf_x).to(device), torch.tensor(buf_y, dtype=torch.long)


def extract_model(model_key, dataset_key, tar_path, log, batch=64, device="cuda"):
    import torchvision.models as tvm
    fn_name, weights_cls, weights_name = MODELS[model_key]
    fn = getattr(tvm, fn_name)
    weights_enum = getattr(tvm, weights_cls)[weights_name]
    log.info(f"  loading {model_key} ({weights_name})")
    model = fn(weights=weights_enum).to(device).eval()
    transform = weights_enum.transforms()

    out_logits = OUT_ROOT / f"nonclip_{model_key}_{dataset_key}_logits.npy"
    out_labels = OUT_ROOT / f"nonclip_{model_key}_{dataset_key}_labels.npy"
    chunks, labels_all = [], []
    n_done = 0
    t0 = time.time()
    with torch.no_grad():
        for inp, y in iter_batches(iter_images_from_tar(tar_path, log),
                                   transform, batch, device):
            logits = model(inp)
            chunks.append(logits.float().cpu().numpy())
            labels_all.append(y.numpy())
            n_done += len(y)
            if n_done % (batch * 40) == 0:
                log.info(f"    {model_key}: {n_done} images "
                         f"({time.time()-t0:.0f}s)")
    logits = np.concatenate(chunks).astype(np.float32)
    labels = np.concatenate(labels_all).astype(np.int32)
    np.save(out_logits, logits)
    np.save(out_labels, labels)
    acc = float((logits.argmax(1) == labels).mean())
    log.info(f"  {model_key} done: {logits.shape}, top1={acc:.4f} "
             f"({time.time()-t0:.0f}s)")
    del model
    torch.cuda.empty_cache()
    return acc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=list(MODELS.keys()))
    parser.add_argument("--datasets", nargs="+",
                        default=["imagenet_v2_matched"])
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--device", default="cuda:0"
                        if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    ts = time.strftime("%Y%m%d_%H%M%S")
    log = setup_logging(ts)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    accs = {}
    for ds in args.datasets:
        tar_path = download_dataset(ds, log)
        for m in args.models:
            out = OUT_ROOT / f"nonclip_{m}_{ds}_logits.npy"
            if out.exists():
                existing = np.load(out)
                accs[f"{m}/{ds}"] = float(
                    (existing.argmax(1)
                     == np.load(OUT_ROOT / f"nonclip_{m}_{ds}_labels.npy")).mean())
                log.info(f"  skip {m}/{ds} (exists, top1={accs[f'{m}/{ds}']:.4f})")
                continue
            accs[f"{m}/{ds}"] = extract_model(m, ds, tar_path, log,
                                              args.batch, args.device)
    log.info("per-model top1: " + json.dumps(accs, indent=2))


if __name__ == "__main__":
    main()
