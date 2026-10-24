"""
extract_clip_per_image_v2.py — General per-image CLIP extraction
=================================================================
Runs 5 CLIP-family backbones on any supported dataset.
Saves per-image logits + labels for downstream analysis.

Usage:
  python experiments/extract_clip_per_image_v2.py --dataset voc2007 --model all
  python experiments/extract_clip_per_image_v2.py --dataset stanford40 --model openai

Output:
  results/per_image/clip_{model}_{dataset}_logits.npy   (N, C) float32
  results/per_image/clip_{model}_{dataset}_labels.npy   (N, C) float32

Reproducibility:
  conda run -n dl_env_310 python experiments/extract_clip_per_image_v2.py \\
      --dataset all --model all --data_root /path/to/data
"""

import argparse, json, os, sys, time, logging, urllib, zipfile, shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

OUTPUT_DIR = REPO_ROOT / "results" / "per_image"
LOG_DIR = REPO_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ── exact model registry (mirrors scb5-zeroshot/models/clip_zoo.py) ──
CLIP_MODELS = ["openai", "laion", "siglip2", "eva02", "dfn"]

MODEL_REGISTRY = {
    "openai": {
        "loader": "openclip",
        "arch": "ViT-L-14",
        "pretrained": "openai",
        "input_size": 224,
    },
    "laion": {
        "loader": "openclip",
        "arch": "ViT-L-14",
        "pretrained": "laion2b_s32b_b82k",
        "input_size": 224,
    },
    "siglip2": {
        "loader": "hf_siglip",
        "arch": "ViT-L-16-SigLIP2-256",
        "hf_name": "google/siglip2-large-patch16-256",
        "input_size": 256,
    },
    "eva02": {
        "loader": "openclip",
        "arch": "EVA02-L-14",
        "pretrained": "merged2b_s4b_b131k",
        "input_size": 224,
    },
    "dfn": {
        "loader": "openclip",
        "arch": "ViT-L-14",
        "pretrained": "dfn2b",
        "input_size": 224,
    },
}

GPU_ASSIGNMENT = {
    "openai": "cuda:1",
    "siglip2": "cuda:2",
    "laion": "cuda:3",
    "eva02": "cuda:1",
    "dfn": "cuda:2",
}

SUPPORTED_DATASETS = ["voc2007", "stanford40", "coco"]


# ── model loading ─────────────────────────────────────────────────────
def load_model(model_key: str, device: str):
    cfg = MODEL_REGISTRY[model_key]
    print(f"[{model_key}] Loading on {device} ...")

    if cfg["loader"] == "openclip":
        import open_clip
        model, _, preprocess = open_clip.create_model_and_transforms(
            cfg["arch"], pretrained=cfg["pretrained"], device=device
        )
        model.eval()
        tokenizer = open_clip.get_tokenizer(cfg["arch"])
        return model, preprocess, tokenizer, "openclip"

    elif cfg["loader"] == "hf_siglip":
        from transformers import AutoModel, AutoTokenizer, AutoImageProcessor
        model = AutoModel.from_pretrained(cfg["hf_name"], trust_remote_code=True).to(device)
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(cfg["hf_name"], trust_remote_code=True)
        image_processor = AutoImageProcessor.from_pretrained(cfg["hf_name"], trust_remote_code=True)
        def _preprocess(img):
            return image_processor(images=img, return_tensors="pt")["pixel_values"].squeeze(0)
        def _tokenizer(texts):
            return tokenizer(text=texts, return_tensors="pt", padding=True, truncation=True)
        return model, _preprocess, _tokenizer, "hf_siglip"

    else:
        raise ValueError(f"Unknown loader: {cfg['loader']}")


@torch.no_grad()
def encode_text(loader_type: str, model, tokenizer, texts, device):
    if loader_type == "openclip":
        tokens = tokenizer(texts).to(device)
        feats = model.encode_text(tokens)
        return F.normalize(feats.float(), dim=-1)
    else:
        inputs = tokenizer(texts)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        feats = model.get_text_features(**inputs)
        return F.normalize(feats.float(), dim=-1)


@torch.no_grad()
def encode_images(loader_type: str, model, images):
    if loader_type == "openclip":
        feats = model.encode_image(images)
        return F.normalize(feats.float(), dim=-1)
    else:
        feats = model.get_image_features(pixel_values=images)
        return F.normalize(feats.float(), dim=-1)


def run_inference(model, preprocess, tokenizer, loader_type, dataset, class_names,
                  device, model_key, batch_size=64):
    text_feats = encode_text(loader_type, model, tokenizer, class_names, device).cpu()

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=2,
        pin_memory=True, collate_fn=lambda b: b)

    all_logits, all_labels = [], []
    n_total = len(dataset)
    n_imgs = 0
    t0 = time.time()

    for batch in loader:
        imgs = torch.stack([item[0] for item in batch]).to(device)
        lbs = np.stack([item[1] for item in batch])

        img_feats = encode_images(loader_type, model, imgs).cpu()
        logits = img_feats @ text_feats.T

        all_logits.append(logits.numpy())
        all_labels.append(lbs)

        n_imgs += len(batch)
        if n_imgs % 500 == 0 or n_imgs == n_total:
            elapsed = time.time() - t0
            ips = n_imgs / max(elapsed, 1e-6)
            print(f"  [{model_key}] {n_imgs}/{n_total} ({elapsed:.0f}s, {ips:.1f} img/s)")

    logits = np.concatenate(all_logits, axis=0).astype(np.float32)
    labels = np.concatenate(all_labels, axis=0).astype(np.float32)
    return logits, labels


# ── dataset loaders ────────────────────────────────────────────────────
class VOC2007Dataset(torch.utils.data.Dataset):
    CLASS_NAMES = [
        "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car",
        "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
        "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor",
    ]

    def __init__(self, root, preprocess):
        import torchvision
        self.cname_to_idx = {n: i for i, n in enumerate(self.CLASS_NAMES)}
        self.transform = preprocess
        self.ds = torchvision.datasets.VOCDetection(
            root=root, year="2007", image_set="val", download=True)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        img, ann = self.ds[idx]
        img_t = self.transform(img)
        objs = ann["annotation"]["object"]
        if isinstance(objs, dict):
            objs = [objs]
        label = np.zeros(len(self.CLASS_NAMES), dtype=np.float32)
        for obj in objs:
            name = obj["name"]
            if name in self.cname_to_idx:
                label[self.cname_to_idx[name]] = 1.0
        return img_t, label


STANFORD40_CLASSES = [
    "applauding", "blowing bubbles", "brushing teeth", "cleaning the floor",
    "climbing", "cooking", "cutting trees", "cutting vegetables",
    "drinking", "feeding a horse", "fishing", "fixing a bike",
    "fixing a car", "gardening", "holding an umbrella", "jumping",
    "looking through a microscope", "looking through a telescope", "phoning",
    "playing guitar", "playing violin", "pouring liquid", "pushing a cart",
    "reading", "riding a bike", "riding a horse", "rowing a boat",
    "running", "shooting an arrow", "smoking", "taking photos",
    "texting message", "throwing frisby", "using a computer",
    "walking the dog", "washing dishes", "watching TV", "waving hands",
    "writing on a board", "writing on a book",
]
STANFORD40_URL = "http://vision.stanford.edu/Datasets/Stanford40Actions.zip"


class Stanford40Dataset(torch.utils.data.Dataset):
    def __init__(self, root, preprocess, val_frac=0.2, seed=42):
        self.transform = preprocess
        self.cname_to_idx = {n: i for i, n in enumerate(STANFORD40_CLASSES)}
        jpeg_dir = Path(root) / "JPEGImages"

        if not jpeg_dir.exists():
            self._download(root)

        samples = []
        for f in sorted(jpeg_dir.glob("*.jpg")):
            parts = f.stem.rsplit("_", 1)
            cname = parts[0]
            if cname in self.cname_to_idx:
                samples.append((str(f), cname, int(parts[1]) if len(parts) > 1 else 0))

        rng = np.random.default_rng(seed)
        by_class = {}
        for path, cname, iid in samples:
            by_class.setdefault(cname, []).append((path, cname, iid))

        self.samples = []
        for cname, items in by_class.items():
            items.sort(key=lambda x: x[2])
            rng.shuffle(items)
            n_val = max(1, int(len(items) * val_frac))
            self.samples.extend(items[:n_val])

        self.samples.sort(key=lambda x: x[0])
        print(f"[Stanford40] val: {len(self.samples)} images")

    def _download(self, root):
        from PIL import Image
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        zip_path = root / "Stanford40Actions.zip"
        print(f"[Stanford40] Downloading {STANFORD40_URL} ...")
        urllib.request.urlretrieve(STANFORD40_URL, zip_path)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(root)
        print("[Stanford40] Extracted")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, cname, _ = self.samples[idx]
        from PIL import Image
        img = Image.open(img_path).convert("RGB")
        img_t = self.transform(img)
        label = np.zeros(len(STANFORD40_CLASSES), dtype=np.float32)
        label[self.cname_to_idx[cname]] = 1.0
        return img_t, label


COCO_CLASS_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]


class COCODataset(torch.utils.data.Dataset):
    """MS-COCO 2017 val (multi-label, 80 classes)."""

    CLASS_NAMES = COCO_CLASS_NAMES

    def __init__(self, root, preprocess):
        import torchvision.datasets as dsets
        self.transform = preprocess
        self.cname_to_idx = {n: i for i, n in enumerate(self.CLASS_NAMES)}
        coco_root = Path(root) / "coco"
        ann_file = coco_root / "annotations" / "instances_val2017.json"
        img_dir = coco_root / "val2017"
        self.ds = dsets.CocoDetection(root=str(img_dir), annFile=str(ann_file))
        self.coco = self.ds.coco
        self.cat_ids = self.coco.getCatIds()
        self.cat_id_to_idx = {cat_id: self.cname_to_idx[self.coco.loadCats(cat_id)[0]["name"]]
                              for cat_id in self.cat_ids
                              if self.coco.loadCats(cat_id)[0]["name"] in self.cname_to_idx}

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        img, anns = self.ds[idx]
        img_t = self.transform(img)
        label = np.zeros(len(self.CLASS_NAMES), dtype=np.float32)
        for ann in anns:
            cat_id = ann["category_id"]
            if cat_id in self.cat_id_to_idx:
                label[self.cat_id_to_idx[cat_id]] = 1.0
        return img_t, label


DATASET_FACTORY = {
    "voc2007": lambda root, prep: (VOC2007Dataset(root, prep), VOC2007Dataset.CLASS_NAMES),
    "stanford40": lambda root, prep: (Stanford40Dataset(root, prep), STANFORD40_CLASSES),
    "coco": lambda root, prep: (COCODataset(root, prep), COCODataset.CLASS_NAMES),
}


# ── main ──────────────────────────────────────────────────────────────
def extract(dataset, model_key, device, data_root, output_dir):
    log_path = LOG_DIR / f"extract_{model_key}_{dataset}.log"
    logging.basicConfig(filename=log_path, level=logging.INFO,
                        format="%(asctime)s %(message)s")

    model, preprocess, tokenizer, loader_type = load_model(model_key, device)
    ds, class_names = DATASET_FACTORY[dataset](str(data_root), preprocess)

    print(f"[{dataset}] {len(ds)} images, {len(class_names)} classes")
    logits, labels = run_inference(model, preprocess, tokenizer, loader_type,
                                    ds, class_names, device, model_key)

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / f"clip_{model_key}_{dataset}_logits.npy", logits)
    np.save(output_dir / f"clip_{model_key}_{dataset}_labels.npy", labels)
    # Accuracy: for multi-label, hit = any GT; for single-label, argmax match
    if labels.ndim == 2 and labels.shape[1] > 1:
        acc = np.array([logits[i].argmax() in np.where(labels[i] > 0)[0]
                        for i in range(len(logits))], dtype=bool).mean()
    else:
        acc = (logits.argmax(1) == labels.argmax(1)).mean()
    print(f"[{model_key}] {dataset} — acc={acc:.4f}, logits={logits.shape}")
    logging.info(f"Done: acc={acc:.4f}, shape={logits.shape}")

    del model
    torch.cuda.empty_cache()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=SUPPORTED_DATASETS + ["all"])
    p.add_argument("--model", default="all", choices=CLIP_MODELS + ["all"])
    p.add_argument("--data_root", default=None)
    p.add_argument("--output_dir", default=str(OUTPUT_DIR))
    p.add_argument("--device", default=None)
    args = p.parse_args()

    data_root = Path(args.data_root) if args.data_root else REPO_ROOT / "data"
    output_dir = Path(args.output_dir)

    models = CLIP_MODELS if args.model == "all" else [args.model]
    datasets = SUPPORTED_DATASETS if args.dataset == "all" else [args.dataset]

    for ds in datasets:
        for mk in models:
            dev = args.device or GPU_ASSIGNMENT.get(mk, "cuda:0")
            out_path = output_dir / f"clip_{mk}_{ds}_logits.npy"
            if out_path.exists():
                print(f"[SKIP] {out_path} exists")
                continue
            print(f"\n{'='*60}")
            print(f"  {mk}  →  {ds}")
            print(f"{'='*60}")
            try:
                extract(ds, mk, dev, data_root, output_dir)
            except Exception as e:
                print(f"[ERROR] {mk}/{ds}: {e}")
                import traceback
                traceback.print_exc()

    print("\nDone.")


if __name__ == "__main__":
    main()