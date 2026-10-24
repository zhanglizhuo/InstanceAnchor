#!/usr/bin/env python3
"""
experiments/train_resnet18_cifar100_seeds.py — seed-diverse deep ensemble
=========================================================================
Trains five ResNet-18 classifiers on CIFAR-100 with different random seeds
(same architecture, same data, same protocol — seed is the only varying
axis). This closes the paper's stated limitation: "the seed axis remains
untested; we leave it to future work" (Discussion, design rationale).

The ensemble is then fed through the same kappa-decomposition / agreement
analysis (analysis/anchoring_seeddiverse_trained.py) as the other 2x2 cells.

Protocol: ResNet-18 adapted to 32x32 (conv1 3x3/s1, no maxpool), SGD momentum
0.9, weight decay 5e-4, batch 128, 200 epochs, cosine LR 0.1->0, standard
CIFAR augmentation (pad-4 random crop + flip). Test-set raw logits saved per
seed. Idempotent: skips seeds whose logits exist.

Data: CIFAR-100 train/test from uoft-cs/cifar100 parquet via hf-mirror
(torchvision tar route unusable on this network). Labels are canonical-order
(verified np.array_equal against the main-panel extraction locally).

Output: results/per_image/resnet18_seeds/resnet18_seed{s}_{logits,labels}.npy

Usage (5090, llava env):
  ~/miniconda3/envs/llava/bin/python experiments/train_resnet18_cifar100_seeds.py
"""
import argparse
import io
import logging
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

OUT_ROOT = Path(__file__).resolve().parents[1] / "results" / "per_image" / "resnet18_seeds"
SEEDS = [0, 1, 2, 3, 4]
EPOCHS = 200


def setup_logging(ts):
    logdir = Path(__file__).resolve().parents[1] / "logs"
    logdir.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(logdir / f"train_resnet18_seeds_{ts}.log"),
                  logging.StreamHandler()])
    return logging.getLogger()


def load_parquet(split, cache_dir):
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq
    p = hf_hub_download("uoft-cs/cifar100", f"cifar100/{split}-00000-of-00001.parquet",
                        repo_type="dataset", cache_dir=str(cache_dir))
    table = pq.read_table(p)
    images = [row['bytes'].as_py() for row in table.column('img')]
    labels = table.column('fine_label').to_pylist()
    return images, labels


def make_resnet18():
    from torchvision.models import resnet18
    m = resnet18(num_classes=100)
    m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    return m


def train_seed(seed, train_imgs, train_lbls, device, log):
    import torchvision.transforms as T
    torch.manual_seed(seed)
    np.random.seed(seed)
    train_tf = T.Compose([
        T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    test_tf = T.Compose([
        T.ToTensor(),
        T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    model = make_resnet18().to(device)
    opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9,
                          weight_decay=5e-4, nesterov=True)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    crit = nn.CrossEntropyLoss()
    n = len(train_lbls)
    lbls = torch.tensor(train_lbls, dtype=torch.long)
    t0 = time.time()
    for ep in range(EPOCHS):
        model.train()
        perm = torch.randperm(n)
        total = correct = 0
        for i in range(0, n, 128):
            idx = perm[i:i + 128]
            x = torch.stack([train_tf(Image.open(io.BytesIO(train_imgs[j])).convert("RGB"))
                             for j in idx.tolist()]).to(device)
            y = lbls[idx].to(device)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
            total += len(idx)
        sched.step()
        if (ep + 1) % 50 == 0:
            log.info(f"  seed {seed} epoch {ep+1}/{EPOCHS} ({time.time()-t0:.0f}s)")
    return model, test_tf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    args = parser.parse_args()
    ts = time.strftime("%Y%m%d_%H%M%S")
    log = setup_logging(ts)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    cache = Path(__file__).resolve().parents[1] / "data" / "resnet18_seeds_dl"

    train_imgs, train_lbls = load_parquet("train", cache, )
    test_imgs, test_lbls = load_parquet("test", cache)
    log.info(f"train {len(train_imgs)} test {len(test_imgs)}")

    test_lbls_np = np.array(test_lbls, dtype=np.int32)
    for seed in SEEDS:
        out = OUT_ROOT / f"resnet18_seed{seed}_logits.npy"
        if out.exists():
            log.info(f"skip seed {seed} (exists)")
            continue
        model, test_tf = train_seed(seed, train_imgs, train_lbls, device, log)
        model.eval()
        chunks = []
        with torch.no_grad():
            for i in range(0, len(test_imgs), 256):
                x = torch.stack([test_tf(Image.open(io.BytesIO(b)).convert("RGB"))
                                 for b in test_imgs[i:i + 256]]).to(device)
                chunks.append(model(x).float().cpu().numpy())
        logits = np.concatenate(chunks).astype(np.float32)
        np.save(out, logits)
        np.save(OUT_ROOT / f"resnet18_seed{seed}_labels.npy", test_lbls_np)
        acc = float((logits.argmax(1) == test_lbls_np).mean())
        log.info(f"seed {seed} done: top1={acc:.4f}")
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
