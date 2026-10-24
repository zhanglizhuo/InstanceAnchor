"""
extract_crossfamily.py — Phase 3: Cross-family image feature extraction
=====================================================================
Extracts per-image features from non-CLIP vision models on SCB5,
using siglip2 text embeddings as the shared classifier.

Models (true cross-family, all cached on V100):
  - vit_b_16:  ImageNet-supervised ViT (torchvision, weights cached)
  - resnet50:  ImageNet-supervised ResNet (torchvision, weights cached)

These are fundamentally different from CLIP: trained on ImageNet with
classification loss, NOT on image-text pairs with contrastive loss.

Output:
  results/per_image/crossfamily_{model}_{subset}_logits.npy
  results/per_image/crossfamily_{model}_{subset}_labels.npy

Usage:
  python experiments/extract_crossfamily.py --model all --subset all
"""
import argparse, os, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "experiments"))

OUTPUT_DIR = REPO_ROOT / "results" / "per_image"
SCB_DATA_DIR = os.environ.get("SCB_DATA_DIR",
    "/home/broadsense/works/lizhuo/AutoResearchClaw/datasets_scb")

SUBSET_KEY_MAP = {
    "teacher_behavior": "SCB5_TeacherBehavior",
    "handrise_readwrite": "SCB5_HandriseReadWrite",
    "bow_turnhead": "SCB_BowTurnHead",
}
MODELS = ["vit_b_16"]  # resnet50 skipped: 2048-dim != 768-dim CLIP text
SUBSETS = ["teacher_behavior", "handrise_readwrite", "bow_turnhead"]
GPU_ASSIGNMENT = {"vit_b_16": "cuda:1", "resnet50": "cuda:1"}
CLASS_NAMES = {
    "teacher_behavior": [
        "guide", "answer", "on-stage interaction",
        "blackboard-writing", "teacher", "stand", "screen", "blackboard"],
    "handrise_readwrite": ["hand-raise", "read", "write"],
    "bow_turnhead": ["bow-head", "turn-head"],
}


def _load_clip_text_feats(device):
    """Load openai CLIP text encoder via open_clip (cached on V100, no HF download)."""
    import open_clip
    print("[openai text] Loading via open_clip...")
    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained="openai", device=device)
    model.eval()
    tokenizer = open_clip.get_tokenizer("ViT-L-14")
    text_feats = {}
    with torch.no_grad():
        for subset, names in CLASS_NAMES.items():
            tokens = tokenizer(names).to(device)
            feats = model.encode_text(tokens)
            text_feats[subset] = F.normalize(feats.float(), dim=-1).cpu()
            print(f"  [{subset}] {feats.shape}")
    return text_feats


class _MinimalSCBDataset(torch.utils.data.Dataset):
    """Minimal SCBDataset replicating scb5-zeroshot's local loader, no external deps."""
    def __init__(self, subset, data_dir, transform):
        configs = {
            "teacher_behavior": {"classes": 8, "multi": True},
            "handrise_readwrite": {"classes": 3, "multi": False},
            "bow_turnhead": {"classes": 2, "multi": False},
        }
        cfg = configs[subset]
        self.multi = cfg["multi"]
        self.nc = cfg["classes"]
        self.transform = transform
        base = Path(data_dir)
        # Find images/val/ or images/train/ under base or its subdirs
        img_dir = None
        for candidate in [base / "images" / "val", base / "images" / "train"]:
            if candidate.exists(): img_dir = candidate; break
        if img_dir is None:
            for child in sorted(base.iterdir()):
                if not child.is_dir(): continue
                for candidate in [child / "images" / "val", child / "images" / "train"]:
                    if candidate.exists(): img_dir = candidate; break
                if img_dir is not None: break
        if img_dir is None:
            raise RuntimeError(f"No images found under {data_dir}")
        self.samples = sorted([p for p in img_dir.rglob("*.jpg")])
        # Load labels from parallel labels/ dir
        lbl_dir = Path(str(img_dir).replace("/images/", "/labels/"))
        self.labels = {}
        if lbl_dir.exists():
            for lp in lbl_dir.rglob("*.txt"):
                stem = lp.stem
                with open(lp) as f:
                    cls_ids = list(set(int(line.split()[0]) for line in f if line.strip()))
                self.labels[stem] = cls_ids
        print(f"  [{subset}] {len(self.samples)} images @ {img_dir}")
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        from PIL import Image
        img_path = self.samples[idx]
        label_vec = torch.zeros(self.nc)
        stem = img_path.stem
        if stem in self.labels:
            for c in self.labels[stem]:
                if 0 <= c < self.nc: label_vec[c] = 1.0
        if not self.multi:
            label_vec = label_vec.argmax().long()
        return {"image": self.transform(Image.open(img_path).convert("RGB")),
                "label": label_vec}

def _build_loader(subset, transform, batch_size):
    data_dir = str(Path(SCB_DATA_DIR) / SUBSET_KEY_MAP[subset])
    ds = _MinimalSCBDataset(subset=subset, data_dir=data_dir, transform=transform)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True)
    return loader, len(ds)


def _make_tv_transform(input_size=224):
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize(int(input_size * 1.14)),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def _load_model(model_key, device):
    weights_enum = {
        "vit_b_16": torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1,
        "resnet50": torchvision.models.ResNet50_Weights.IMAGENET1K_V1,
    }
    weights = weights_enum[model_key]
    # Remove the classification head (num_classes=0)
    model = torchvision.models.get_model(model_key, weights=weights)
    if hasattr(model, 'heads'):
        model.heads = torch.nn.Identity()
    elif hasattr(model, 'fc'):
        model.fc = torch.nn.Identity()
    elif hasattr(model, 'classifier'):
        model.classifier = torch.nn.Identity()
    model.eval().to(device)
    input_size = 224
    if model_key == "vit_b_16":
        input_size = 224
    transform = _make_tv_transform(input_size)
    return model, transform


@torch.no_grad()
def extract(model_key, subset, text_feats, device, output_dir, batch_size=64):
    print(f"\n{'='*60}")
    print(f"  {model_key}  →  {subset}")
    print(f"{'='*60}")

    out_logits = output_dir / f"crossfamily_{model_key}_{subset}_logits.npy"
    out_labels = output_dir / f"crossfamily_{model_key}_{subset}_labels.npy"
    if out_logits.exists():
        print(f"  [SKIP] already exists"); return

    model, transform = _load_model(model_key, device)
    loader, n = _build_loader(subset, transform, batch_size)
    txt = text_feats[subset].to(device)
    all_logits, all_labels = [], []
    n_seen = 0; t0 = time.time()

    for batch in loader:
        imgs = batch["image"].to(device)
        lbs = batch["label"]
        feats = model(imgs)
        feats = F.normalize(feats.float(), dim=-1)
        logits = feats @ txt.T
        all_logits.append(logits.cpu().numpy())
        all_labels.append(lbs.numpy())
        n_seen += len(imgs)
        if n_seen % 500 == 0 or n_seen >= n:
            print(f"  [{model_key}/{subset}] {n_seen}/{n} ({time.time()-t0:.0f}s)")

    logits = np.concatenate(all_logits, axis=0).astype(np.float32)
    labels = np.concatenate(all_labels, axis=0).astype(np.float32)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_logits, logits); np.save(out_labels, labels)
    if labels.ndim == 2:
        acc = (logits.argmax(1) == labels.argmax(1)).mean()
    else:
        acc = (logits.argmax(1) == labels).mean()
    print(f"  [{model_key}/{subset}] acc={acc:.4f} shape={logits.shape}")
    del model; torch.cuda.empty_cache()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="all", choices=MODELS + ["all"])
    p.add_argument("--subset", default="all", choices=SUBSETS + ["all"])
    p.add_argument("--output_dir", default=str(OUTPUT_DIR))
    args = p.parse_args()
    models = MODELS if args.model == "all" else [args.model]
    subsets = SUBSETS if args.subset == "all" else [args.subset]
    text_feats = _load_clip_text_feats("cpu")  # text encoder is small, load on CPU
    for mk in models:
        dev = GPU_ASSIGNMENT[mk]
        for ss in subsets:
            extract(mk, ss, text_feats, dev, Path(args.output_dir))
    print("\nDone.")


if __name__ == "__main__":
    main()
