"""
extract_clip_per_image.py — Extract per-image CLIP predictions
==============================================================
Runs CLIP+CAPE Set-A inference and saves per-image logits + labels
for downstream per-image complementarity analysis.

This is fast (batch inference, ~2-3 min per model per subset).
Reuses the main project's models/clip_zoo.py and prompts/cape_prompts.py.

Usage:
  # Single model, single subset
  python experiments/extract_clip_per_image.py --model openai --subset teacher_behavior

  # All models, all subsets
  python experiments/extract_clip_per_image.py --model all --subset all

Output:
  results/per_image/clip_{model}_{subset}_logits.npy
  results/per_image/clip_{model}_{subset}_labels.npy  (multi-label binary matrix)

Reproducibility:
  python experiments/extract_clip_per_image.py --model all --subset all
"""

import os
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.scb_dataset import build_dataloader, SUBSET_CONFIG, SUBSET_LOCAL_DIRS
from models.clip_zoo import CLIPModel, build_cape_text_features, batch_inference
from prompts.cape_prompts import get_all_class_prompts

SCB_DATA_DIR = os.environ.get("SCB_DATA_DIR",
    "/home/broadsense/works/lizhuo/AutoResearchClaw/datasets_scb")

CLIP_MODELS = ["openai", "laion", "siglip2", "eva02", "dfn"]
SUBSETS = ["teacher_behavior", "handrise_readwrite", "bow_turnhead"]
SUBSET_KEY_MAP = {
    "teacher_behavior": "SCB5_TeacherBehavior",
    "handrise_readwrite": "SCB5_HandriseReadWrite",
    "bow_turnhead": "SCB_BowTurnHead",
}

GPU_ASSIGNMENT = {
    "openai": "cuda:0", "siglip2": "cuda:1", "laion": "cuda:2",
    "eva02": "cuda:3", "dfn": "cuda:0",
}


def extract(model_key: str, subset: str, output_dir: Path, device: str = "cuda:0",
            cape_set: str = "A"):
    """Extract per-image CLIP logits for one (model, subset, cape_set) combination."""
    cfg = SUBSET_CONFIG[subset]
    classes = cfg["classes"]
    multilabel = cfg["multilabel"]
    num_classes = cfg["num_classes"]

    data_dir = None
    if SCB_DATA_DIR:
        candidate = Path(SCB_DATA_DIR) / SUBSET_KEY_MAP[subset]
        if candidate.exists():
            data_dir = str(candidate)

    print(f"[CLIP] Loading {model_key} (data_dir={data_dir or 'auto/HF'})...")
    model = CLIPModel(model_key, device=device)

    loader, dataset = build_dataloader(
        subset=subset, split="validation",
        batch_size=64, num_workers=0, input_size=model.input_size,
        data_dir=data_dir,
    )

    class_prompts = get_all_class_prompts(classes, "cape", cape_set=cape_set)
    text_feats = build_cape_text_features(model, class_prompts, classes)

    print(f"[CLIP] Running inference on {subset} / CAPE-{cape_set} ({len(dataset)} images)...")
    logits, labels = batch_inference(model, loader, text_feats, multilabel)

    # Convert labels to multi-label binary matrix
    if multilabel:
        lb_matrix = labels.astype(np.float32)
    else:
        lb_matrix = np.eye(num_classes, dtype=np.float32)[labels.astype(int)]

    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if cape_set == "A" else f"_cape{cape_set.lower()}"
    logits_path = output_dir / f"clip_{model_key}_{subset}_logits{suffix}.npy"
    labels_path = output_dir / f"clip_{model_key}_{subset}_labels{suffix}.npy"

    np.save(logits_path, logits.astype(np.float32))
    np.save(labels_path, lb_matrix)
    print(f"[CLIP] Saved: {logits_path} ({logits.shape})")
    print(f"[CLIP] Saved: {labels_path} ({lb_matrix.shape})")

    del model
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description="Extract per-image CLIP predictions")
    parser.add_argument("--model", type=str, default="all",
                        choices=CLIP_MODELS + ["all"],
                        help="CLIP model key (default: all)")
    parser.add_argument("--subset", type=str, default="all",
                        choices=SUBSETS + ["all"],
                        help="Subset (default: all)")
    parser.add_argument("--output_dir", type=str,
                        default=str(REPO_ROOT / "results" / "per_image"),
                        help="Output directory")
    parser.add_argument("--device", type=str, default=None,
                        help="CUDA device (auto-detect if not set)")
    parser.add_argument("--cape_set", type=str, default="A",
                        choices=["A", "B", "C"],
                        help="CAPE prompt set (default: A)")
    args = parser.parse_args()

    models_to_run = CLIP_MODELS if args.model == "all" else [args.model]
    subsets_to_run = SUBSETS if args.subset == "all" else [args.subset]
    output_dir = Path(args.output_dir)

    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        device = "cuda:0"
    else:
        device = args.device or GPU_ASSIGNMENT.get(models_to_run[0], "cuda:0")

    for model_key in models_to_run:
        for subset in subsets_to_run:
            extract(model_key, subset, output_dir, device=device, cape_set=args.cape_set)

    return 0


if __name__ == "__main__":
    exit(main())
