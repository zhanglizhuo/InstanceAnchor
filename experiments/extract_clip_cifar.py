"""
experiments/extract_clip_cifar.py — CLIP per-image extraction on CIFAR-100
============================================================================
Extracts per-image CLIP logits for all 5 backbones on CIFAR-100 test set.
Saves .npy files for downstream analysis.

Output:
  results/per_image/cifar100/clip_{model}_cifar100_logits.npy   (10000, 100)
  results/per_image/cifar100/clip_{model}_cifar100_labels.npy   (10000, 1)

Reproducibility (on GPU server with CIFAR-100 data):
  python experiments/extract_clip_cifar.py [--data_root /path/to/data]
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.datasets as datasets

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_clip_per_image_v2 import (
    MODEL_REGISTRY, CLIP_MODELS, load_model, encode_text, encode_images,
)

CIFAR_CLASSES = [
    "apple", "aquarium_fish", "baby", "bear", "beaver", "bed", "bee", "beetle",
    "bicycle", "bottle", "bowl", "boy", "bridge", "bus", "butterfly", "camel",
    "can", "castle", "caterpillar", "cattle", "chair", "chimpanzee", "clock",
    "cloud", "cockroach", "couch", "crab", "crocodile", "cup", "dinosaur",
    "dolphin", "elephant", "flatfish", "forest", "fox", "girl", "hamster",
    "house", "kangaroo", "keyboard", "lamp", "lawn_mower", "leopard", "lion",
    "lizard", "lobster", "man", "maple_tree", "motorcycle", "mountain",
    "mouse", "mushroom", "oak_tree", "orange", "orchid", "otter", "palm_tree",
    "pear", "pickup_truck", "pine_tree", "plain", "plate", "poppy", "porcupine",
    "possum", "rabbit", "raccoon", "ray", "road", "rocket", "rose",
    "sea", "seal", "shark", "shrew", "skunk", "skyscraper", "snail", "snake",
    "spider", "squirrel", "streetcar", "sunflower", "sweet_pepper", "table",
    "tank", "telephone", "television", "tiger", "tractor", "train", "trout",
    "tulip", "turtle", "wardrobe", "whale", "willow_tree", "wolf", "woman",
    "worm",
]


def setup_logging(log_dir, ts, name="extract_clip_cifar"):
    log = logging.getLogger(name)
    log.setLevel(logging.INFO)
    fh = logging.FileHandler(log_dir / f"{name}_{ts}.log")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    log.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    log.addHandler(ch)
    return log


@torch.no_grad()
def extract(model_key, device, data_root, output_dir, log):
    log.info(f"[{model_key}] Loading on {device} ...")
    model, preprocess, tokenizer, loader_type = load_model(model_key, device)
    model.eval()
    log.info(f"  loaded (loader={loader_type})")

    ds_raw = datasets.CIFAR100(root=str(data_root), train=False, download=True, transform=None)
    n = len(ds_raw)
    log.info(f"  {n} images, {len(CIFAR_CLASSES)} classes")

    texts = [f"a photo of a {c}" for c in CIFAR_CLASSES]
    text_feats = encode_text(loader_type, model, tokenizer, texts, device)

    batch_size = 256
    all_logits = []
    for i in range(0, n, batch_size):
        batch_imgs = [ds_raw[j][0] for j in range(i, min(i + batch_size, n))]
        batch_tensors = torch.stack([preprocess(img) for img in batch_imgs]).to(device)
        img_feats = encode_images(loader_type, model, batch_tensors)
        logits = img_feats @ text_feats.T
        all_logits.append(logits.cpu().numpy())
        if (i // batch_size) % 5 == 0:
            log.info(f"  [{model_key}] {min(i+batch_size, n)}/{n}")

    logits_arr = np.concatenate(all_logits, axis=0).astype(np.float32)
    labels_arr = np.array(ds_raw.targets, dtype=np.float32)

    acc = (logits_arr.argmax(1) == labels_arr).mean()
    log.info(f"  [{model_key}] acc={acc:.4f}")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / f"clip_{model_key}_cifar100_logits.npy", logits_arr)
    np.save(out / f"clip_{model_key}_cifar100_labels.npy", labels_arr)
    log.info(f"  saved to {out}/")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="all", choices=CLIP_MODELS + ["all"])
    parser.add_argument("--data_root", default=str(Path(__file__).resolve().parents[1] / "data"))
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--log_dir", default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    ts = time.strftime("%Y%m%d_%H%M%S")
    log_dir = Path(args.log_dir) if args.log_dir else Path("/tmp/cifar_logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log = setup_logging(log_dir, ts)

    if args.device:
        device = args.device
    else:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    log.info(f"device={device}")

    output_dir = Path(args.output_dir) if args.output_dir else (
        Path(__file__).resolve().parents[1] / "results" / "per_image" / "cifar100"
    )
    log.info(f"output_dir={output_dir}")

    models = CLIP_MODELS if args.model == "all" else [args.model]
    for mk in models:
        try:
            extract(mk, device, args.data_root, output_dir, log)
        except Exception as e:
            log.error(f"[{mk}] {e}", exc_info=True)

    log.info("Done.")


if __name__ == "__main__":
    main()
