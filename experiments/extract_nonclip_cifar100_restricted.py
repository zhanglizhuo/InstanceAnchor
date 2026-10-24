#!/usr/bin/env python3
"""
experiments/extract_nonclip_cifar100_restricted.py — E2: non-ImageNet cross-family
====================================================================================
Closes the paper's stated caveat "the non-ImageNet scope of the MSP edge
remains untested cross-family": runs the same five ImageNet-supervised
torchvision models (ResNet-50, ViT-B/16, ConvNeXt-B, Swin-B, EfficientNet-B4)
on CIFAR-100 test images, with the 1000-class head RESTRICTED to the 63
CIFAR-100 classes that have a defensible ImageNet-1k counterpart
(data/cifar100_to_imagenet.json; the remaining 37 CIFAR classes have no
ImageNet-1k class and are excluded together with their images).

Protocol: images of the 63 mapped classes only; logits restricted to the 63
mapped ImageNet columns; correctness = argmax within the restricted head
equals the image's mapped class. Per-model canonical transforms. Raw logits.

Output: results/per_image/nonclip_cifar100/{model}_cifar100_restricted_{logits,labels}.npy

Usage (5090, llava env):
  ~/miniconda3/envs/llava/bin/python experiments/extract_nonclip_cifar100_restricted.py
"""
import argparse
import io
import json
import logging
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

OUT_ROOT = Path(__file__).resolve().parents[1] / "results" / "per_image" / "nonclip_cifar100"
MODELS = {
    "resnet50": ("resnet50", "ResNet50_Weights", "IMAGENET1K_V2"),
    "vit_b_16": ("vit_b_16", "ViT_B_16_Weights", "IMAGENET1K_V1"),
    "convnext_base": ("convnext_base", "ConvNeXt_Base_Weights", "IMAGENET1K_V1"),
    "swin_b": ("swin_b", "Swin_B_Weights", "IMAGENET1K_V1"),
    "efficientnet_b4": ("efficientnet_b4", "EfficientNet_B4_Weights", "IMAGENET1K_V1"),
}


def setup_logging(ts):
    logdir = Path(__file__).resolve().parents[1] / "logs"
    logdir.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(logdir / f"extract_nonclip_cifar100_{ts}.log"),
                  logging.StreamHandler()])
    return logging.getLogger()


def load_mapping():
    m = json.load(open(Path(__file__).resolve().parents[1]
                       / "data" / "cifar100_to_imagenet.json"))
    # canonical CIFAR-100 order
    CIFAR100_CLASSES = ["apple", "aquarium_fish", "baby", "bear", "beaver", "bed",
                        "bee", "beetle", "bicycle", "bottle", "bowl", "boy",
                        "bridge", "bus", "butterfly", "camel", "can", "castle",
                        "caterpillar", "cattle", "chair", "chimpanzee", "clock",
                        "cloud", "cockroach", "couch", "crab", "crocodile", "cup",
                        "dinosaur", "dolphin", "elephant", "flatfish", "forest",
                        "fox", "girl", "hamster", "house", "kangaroo", "keyboard",
                        "lamp", "lawn_mower", "leopard", "lion", "lizard",
                        "lobster", "man", "maple_tree", "motorcycle", "mountain",
                        "mouse", "mushroom", "oak_tree", "orange", "orchid",
                        "otter", "palm_tree", "pear", "pickup_truck", "pine_tree",
                        "plain", "plate", "poppy", "porcupine", "possum", "rabbit",
                        "raccoon", "ray", "road", "rocket", "rose", "sea", "seal",
                        "shark", "shrew", "skunk", "skyscraper", "snail", "snake",
                        "spider", "squirrel", "streetcar", "sunflower",
                        "sweet_pepper", "table", "tank", "telephone", "television",
                        "tiger", "tractor", "train", "trout", "tulip", "turtle",
                        "wardrobe", "whale", "willow_tree", "wolf", "woman",
                        "worm"]
    mapped = m["cifar_to_imagenet_idx"]  # cifar_name -> imagenet idx
    kept = [c for c in CIFAR100_CLASSES if c in mapped]
    cifar_idx_to_restricted = {i: j for j, c in enumerate(kept)
                               for i in [CIFAR100_CLASSES.index(c)]}
    imagenet_cols = [mapped[c] for c in kept]
    return cifar_idx_to_restricted, imagenet_cols


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=list(MODELS.keys()))
    parser.add_argument("--batch", type=int, default=128)
    args = parser.parse_args()
    ts = time.strftime("%Y%m%d_%H%M%S")
    log = setup_logging(ts)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq
    cache = Path(__file__).resolve().parents[1] / "data" / "nonclip_cifar100_dl"
    p = hf_hub_download("uoft-cs/cifar100", "cifar100/test-00000-of-00001.parquet",
                        repo_type="dataset", cache_dir=str(cache))
    table = pq.read_table(p)
    images = [row['bytes'].as_py() for row in table.column('img')]
    fine = table.column('fine_label').to_pylist()
    cifar_idx_to_restricted, imagenet_cols = load_mapping()
    keep = [i for i, l in enumerate(fine) if l in cifar_idx_to_restricted]
    log.info(f"kept {len(keep)}/{len(fine)} images ({len(cifar_idx_to_restricted)} classes)")

    import torchvision.models as tvm
    for key in args.models:
        out = OUT_ROOT / f"{key}_cifar100_restricted_logits.npy"
        if out.exists():
            log.info(f"skip {key} (exists)")
            continue
        fn_name, wcls, wname = MODELS[key]
        fn = getattr(tvm, fn_name)
        weights = getattr(tvm, wcls)[wname]
        model = fn(weights=weights).to(device).eval()
        transform = weights.transforms()
        chunks, labels = [], []
        t0 = time.time()
        with torch.no_grad():
            buf_x, buf_y = [], []
            for i in keep:
                img = Image.open(io.BytesIO(images[i])).convert("RGB")
                buf_x.append(transform(img))
                buf_y.append(cifar_idx_to_restricted[fine[i]])
                if len(buf_x) == args.batch:
                    inp = torch.stack(buf_x).to(device)
                    logits = model(inp)[:, imagenet_cols]
                    chunks.append(logits.float().cpu().numpy())
                    labels.append(np.array(buf_y, dtype=np.int32))
                    buf_x, buf_y = [], []
            if buf_x:
                inp = torch.stack(buf_x).to(device)
                chunks.append(model(inp)[:, imagenet_cols].float().cpu().numpy())
                labels.append(np.array(buf_y, dtype=np.int32))
        logits = np.concatenate(chunks).astype(np.float32)
        lbls = np.concatenate(labels)
        np.save(out, logits)
        np.save(OUT_ROOT / f"{key}_cifar100_restricted_labels.npy", lbls)
        acc = float((logits.argmax(1) == lbls).mean())
        log.info(f"{key} done: {logits.shape} top1={acc:.4f} ({time.time()-t0:.0f}s)")
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
