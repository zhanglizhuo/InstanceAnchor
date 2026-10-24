#!/usr/bin/env python3
"""
experiments/seed_diverse_cifar100.py — 2x2 cell C: data-diverse, same-arch CLIP
================================================================================
Completes the mechanism-attribution 2x2 for the cross-family finding
(paper sec:nonclip). The cross-family contrast (CLIP arch+data diverse vs
torchvision arch-diverse data-shared) co-varies data diversity with training
objective and architecture mix. This cell holds ARCHITECTURE fixed
(ViT-B/32 x5) while varying TRAINING DATA (5 distinct web-scale sources),
on CIFAR-100 (C=100), where the arch-diverse CLIP cell has committed
references (plurality share 72%, p=0.711):

    cell A  data-diverse + arch-diverse  (main 5-backbone panel)   [have]
    cell B  data-shared  + arch-diverse  (torchvision, V2)         [have]
    cell C  data-diverse + arch-same     (this script, CIFAR-100)  [this]
    cell D  data-shared  + arch-same     (requires seed checkpoints; open)

If cell C also shows plurality dominance (share ~72% like cell A), the
data-diversity axis---not the architecture mix---is the driver of the
mechanism split, and the "shared data -> correlated errors" attribution
survives an architecture-controlled contrast.

Models (all ViT-B/32, OpenCLIP zoo):
    openai        (WIT-400M)
    laion400m     (LAION-400M e31)
    laion2b       (LAION-2B)
    datacomp      (DataComp-XL)
    common_pool   (CommonPool)

Extraction only (analysis lives in analysis/anchoring_seeddiverse_cifar100.py).
Logits are raw cosine similarities (CLIP protocol, T=100 downstream).

Usage (V100 box):
  nohup python3 experiments/seed_diverse_cifar100.py > logs/seed_diverse.log 2>&1 &

Output:
  results/per_image/cifar100_seeddiverse/clip_{model}_cifar100_logits.npy
  results/per_image/cifar100_seeddiverse/clip_{model}_cifar100_labels.npy
"""
import argparse
import logging
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

try:
    import torch
    import open_clip
    from torch.utils.data import DataLoader
except ImportError as e:
    log.error(f"Missing package: {e}. Install: pip install torch open_clip_torch torchvision")
    sys.exit(1)

SEED_DIVERSE_MODELS = [
    ("openai", "ViT-B-32", "openai"),
    ("laion400m", "ViT-B-32", "laion400m_e31"),
    ("laion2b", "ViT-B-32", "laion2b_s34b_b79k"),
    ("datacomp", "ViT-B-32", "datacomp_xl_s13b_b90k"),
    ("common_pool", "ViT-B-32", "commonpool_m_clip_s128m_b4k"),
]
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# Canonical CIFAR-100 fine labels (torchvision order). The HF parquet's own
# names metadata has a truncation typo at index 26 ('cra'); these are the
# authoritative names, verified index-order-identical to the main-panel
# extraction (np.array_equal on the 10000-label sequence).
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


class ParquetCIFAR100:
    """CIFAR-100 test split from the HF parquet (uoft-cs/cifar100). Same 10000
    images/labels, verified sequence-identical to the main-panel extraction."""

    def __init__(self, parquet_path, transform):
        import io
        import pyarrow.parquet as pq
        from PIL import Image
        table = pq.read_table(parquet_path)
        self.images = [row['bytes'].as_py() for row in table.column('img')]
        self.labels = table.column('fine_label').to_pylist()
        self.transform = transform
        self.classes = CIFAR100_CLASSES

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        import io
        from PIL import Image
        img = Image.open(io.BytesIO(self.images[i])).convert("RGB")
        return self.transform(img), self.labels[i]


def build_loader(parquet_path, preprocess, batch, num_workers):
    if parquet_path:
        ds = ParquetCIFAR100(parquet_path, preprocess)
    else:
        from torchvision.datasets import CIFAR100
        ds = CIFAR100(root="./data", train=False, download=True,
                      transform=preprocess)
        ds.classes = CIFAR100_CLASSES
    return DataLoader(ds, batch_size=batch, num_workers=num_workers,
                      shuffle=False), ds.classes


def build_b32_preprocess():
    """Canonical ViT-B/32 preprocess (bicubic) — identical for all five
    same-arch models, protocol match with the main-panel CIFAR-100 cell."""
    import torchvision.transforms as T
    return T.Compose([
        T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize((0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711)),
    ])


def extract_one(model_name, arch, pretrained, loader, classes, device):
    log.info(f"Loading {pretrained} ({arch})...")
    model, _, _ = open_clip.create_model_and_transforms(
        arch, pretrained=pretrained, device=device)
    tokenizer = open_clip.get_tokenizer(arch)
    prompts = [f"a photo of a {c.replace('_', ' ')}" for c in classes]
    with torch.no_grad():
        text_tokens = tokenizer(prompts).to(device)
        text_feats = model.encode_text(text_tokens)
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)

    all_logits, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            img_feats = model.encode_image(images)
            img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
            logits = (img_feats @ text_feats.T).cpu().numpy().astype(np.float32)
            all_logits.append(logits)
            all_labels.append(np.asarray(labels))
    logits = np.concatenate(all_logits)
    labels = np.concatenate(all_labels)
    log.info(f"  {model_name}: logits={logits.shape} "
             f"top1={float((logits.argmax(1) == labels).mean()):.4f}")
    del model
    torch.cuda.empty_cache()
    return logits, labels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet_path", default=None,
                        help="HF cifar100 test parquet (fast channel); "
                             "fallback = torchvision download")
    parser.add_argument("--output_dir",
                        default="results/per_image/cifar100_seeddiverse")
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    loader, classes = build_loader(args.parquet_path, build_b32_preprocess(),
                                   args.batch, args.num_workers)
    log.info(f"loader ready ({len(loader.dataset)} images, "
             f"parquet={bool(args.parquet_path)})")
    for model_name, arch, pretrained in SEED_DIVERSE_MODELS:
        logits_path = out_dir / f"clip_{model_name}_cifar100_logits.npy"
        if logits_path.exists():
            log.info(f"Skipping {model_name} (already exists)")
            continue
        try:
            logits, labels = extract_one(model_name, arch, pretrained,
                                         loader, classes, DEVICE)
            np.save(logits_path, logits)
            np.save(out_dir / f"clip_{model_name}_cifar100_labels.npy",
                    labels.astype(np.float32))
        except Exception as e:  # noqa: BLE001
            log.error(f"Failed for {model_name}: {e}")
            continue


if __name__ == "__main__":
    main()
