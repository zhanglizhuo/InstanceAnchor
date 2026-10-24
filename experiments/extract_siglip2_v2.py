"""
Quick fix: extract siglip2 for VOC2007 + Stanford40 using open_clip (cached).
The main v2 script uses transformers for siglip2 which needs HTTPS.
"""
import sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "experiments"))
OUTPUT_DIR = REPO_ROOT / "results" / "per_image"
DATA_DIR = Path("/home/broadsense/works/lizhuo/data")

import open_clip

device = "cuda:1"
print("[siglip2] Loading via open_clip...")
model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-L-16-SigLIP2-256", device=device)
model.eval()
tokenizer = open_clip.get_tokenizer("ViT-L-16-SigLIP2-256")

from extract_clip_per_image_v2 import VOC2007Dataset, Stanford40Dataset, SUPPORTED_DATASETS

for dataset_name in SUPPORTED_DATASETS:
    out_l = OUTPUT_DIR / f"clip_siglip2_{dataset_name}_logits.npy"
    if out_l.exists():
        print(f"[SKIP] siglip2/{dataset_name} already exists"); continue

    if dataset_name == "voc2007":
        ds = VOC2007Dataset(str(DATA_DIR), preprocess)
        class_names = VOC2007Dataset.CLASS_NAMES
    else:
        ds = Stanford40Dataset(str(DATA_DIR), preprocess)
        from extract_clip_per_image_v2 import STANFORD40_CLASSES
        class_names = STANFORD40_CLASSES

    print(f"[siglip2/{dataset_name}] {len(ds)} images, {len(class_names)} classes")
    tokens = tokenizer(class_names).to(device)
    with torch.no_grad():
        text_feats = F.normalize(model.encode_text(tokens).float(), dim=-1)

    loader = torch.utils.data.DataLoader(
        ds, batch_size=64, shuffle=False, num_workers=2, pin_memory=True,
        collate_fn=lambda b: b)
    all_logits, all_labels = [], []
    n = len(ds); seen = 0; t0 = time.time()

    for batch in loader:
        imgs = torch.stack([x[0] for x in batch]).to(device)
        lbs = np.stack([x[1] for x in batch])
        img_feats = F.normalize(model.encode_image(imgs).float(), dim=-1)
        logits = img_feats @ text_feats.T
        all_logits.append(logits.cpu().numpy())
        all_labels.append(lbs)
        seen += len(batch)
        if seen % 500 == 0 or seen >= n:
            print(f"  {seen}/{n} ({time.time()-t0:.0f}s)")

    logits = np.concatenate(all_logits).astype(np.float32)
    labels = np.concatenate(all_labels).astype(np.float32)
    np.save(out_l, logits)
    np.save(OUTPUT_DIR / f"clip_siglip2_{dataset_name}_labels.npy", labels)
    acc = (logits.argmax(1) == labels.argmax(1)).mean()
    print(f"[siglip2/{dataset_name}] acc={acc:.4f} shape={logits.shape}")

print("Done.")
