"""
experiments/extract_clip_meta.py — P4: 35-dataset meta-study extraction
=======================================================================
Elevation-plan P4 (large-scale, ~35 datasets total). Extracts per-image
CLIP logits + labels for the 5 paper backbones on NEW datasets to feed
the meta-regression / LOO boundary test.

Locked design (matches the paper's protocol):
  - 5 backbones, same registry as the paper (openai/laion/siglip2/eva02/dfn).
  - Zero-shot text classifier: per-class text features averaged over
    generic ImageNet-style templates, L2-normalized.
  - Logits = normalized image-feature @ normalized text-feature.T
    (RAW cosine sims, unscaled — temperature applied downstream, rule 14).
  - Label convention (matches existing loaders):
      single-label -> (N,) int32 class indices
      multi-label  -> (N, C) int8 binary
  - Per-dataset image cap, deterministic seed-sorted subsample.
  - Idempotent: dataset skipped when all requested npy pairs exist.
  - Resumable per dataset; progress logged.

Output: results/per_image/meta/clip_{model}_{dataset}_logits.npy (+labels)

Reproducibility (on A100, `clip` conda env; HF_ENDPOINT=hf-mirror.com):
  python experiments/extract_clip_meta.py --datasets all [--models ...] \
      [--max_images 10000] [--device cuda:0]

"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import os

# HF datasets download ONLINE via HF_ENDPOINT (hf-mirror on 5090); the 5 model
# weights are pre-cached, so transformers stays offline (no network stall).
os.environ.setdefault("HF_HUB_OFFLINE", "0")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Model registry (mirrors experiments/extract_clip_per_image_v2.py, which is
# the canonical V100 loader): openclip via arch+pretrained tags (weights in
# the server HF cache), siglip2 via transformers AutoModel.
MODELS = {
    "openai":  {"loader": "openclip", "arch": "ViT-L-14",             "pretrained": "openai",            "input_size": 224},
    "laion":   {"loader": "openclip", "arch": "ViT-L-14",             "pretrained": "laion2b_s32b_b82k", "input_size": 224},
    "siglip2": {"loader": "hf_siglip", "arch": "ViT-L-16-SigLIP2-256", "hf_name": "google/siglip2-large-patch16-256", "input_size": 256},
    "eva02":   {"loader": "openclip", "arch": "EVA02-L-14",           "pretrained": "merged2b_s4b_b131k", "input_size": 224},
    "dfn":     {"loader": "openclip", "arch": "ViT-L-14",             "pretrained": "dfn2b",             "input_size": 224},
    # P5 panel backbones (k=8: original 5 + these 3)
    "vith14":  {"loader": "openclip", "arch": "ViT-H-14",             "pretrained": "laion2b_s32b_b79k", "input_size": 224},
    "vitb32":  {"loader": "openclip", "arch": "ViT-B-32",             "pretrained": "openai",            "input_size": 224},
    "vitb16":  {"loader": "openclip", "arch": "ViT-B-16",             "pretrained": "laion2b_s34b_b88k", "input_size": 224},
}

DATASETS = {
    # --- single-label (HF hub; downloaded via hf-mirror) ---
    # NOTE: tanganke/sun397, tanganke/gtsrb, tanganke/stanford_cars dropped —
    # hf-mirror stalls on their large parquet files (450MB+/file).
    "food101":       {"loader": "hf", "hf_id": "food101",                    "cap": 10000, "split": "validation", "img": "image", "lbl": "label"},
    "flowers102":    {"loader": "hf", "hf_id": "nelorth/oxford-flowers",     "cap": 8000,  "split": "test",        "img": "image", "lbl": "label"},
    "dtd":           {"loader": "hf", "hf_id": "tanganke/dtd",               "cap": 5000,  "split": "test",        "img": "image", "lbl": "label"},
    "cifar10":       {"loader": "hf", "hf_id": "uoft-cs/cifar10",            "cap": 10000, "split": "test",        "img": "img",   "lbl": "label"},
    "tiny_imagenet": {"loader": "hf", "hf_id": "Maysee/tiny-imagenet",       "cap": 10000, "split": "valid",       "img": "image", "lbl": "label", "wnid_names": True},
    "stl10":         {"loader": "hf", "hf_id": "tanganke/stl10",             "cap": 8000,  "split": "test",        "img": "image", "lbl": "label"},
    "eurosat":       {"loader": "hf", "hf_id": "tanganke/eurosat",           "cap": 5400,  "split": "test",        "img": "image", "lbl": "label"},
    # --- ModelScope-recovered (leafdir / folder layouts) ---
    "sun397":        {"loader": "leafdir", "cap": 10000, "split": None},
    "fgvc_aircraft": {"loader": "leafdir", "cap": 5000,  "split": None,
                      "subdir": "trainval_cls"},
    "stanford_cars": {"loader": "leafdir", "cap": 8000,  "split": None,
                      "subdir": "cars_train_cls"},
    "gtsrb":         {"loader": "folder",  "cap": 10000, "split": None},
    # --- single-label (direct berkeley tar; folder loader + wnid map) ---
    "imagenet_a":    {"loader": "folder", "cap": 7500,  "split": None, "wnid_map": True},
    "imagenet_r":    {"loader": "folder", "cap": 10000, "split": None, "wnid_map": True},
}

OUT_ROOT = Path(__file__).resolve().parents[1] / "results" / "per_image" / "meta"

TEMPLATES = [
    "a photo of a {}.", "a photo of the {}.", "a picture of a {}.",
    "a picture of the {}.", "an image of a {}.", "an image of the {}.",
    "a photograph of a {}.", "a photograph of the {}.",
    "the {} in the photo.", "a rendering of a {}.",
]

def setup_logging(ts):
    logdir = Path(__file__).resolve().parents[1] / "logs"
    logdir.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(logdir / f"extract_clip_meta_{ts}.log"),
                  logging.StreamHandler()])


def load_pipeline(m, device):
    import open_clip
    cfg = MODELS[m]
    if cfg["loader"] == "openclip":
        model, _, preprocess = open_clip.create_model_and_transforms(
            cfg["arch"], pretrained=cfg["pretrained"], device=device)
        tokenizer = open_clip.get_tokenizer(cfg["arch"])
        loader_type = "openclip"
    else:
        from transformers import AutoModel, AutoTokenizer, AutoImageProcessor
        model = AutoModel.from_pretrained(cfg["hf_name"],
                                          trust_remote_code=True).to(device)
        tok_inner = AutoTokenizer.from_pretrained(cfg["hf_name"],
                                                  trust_remote_code=True)
        image_processor = AutoImageProcessor.from_pretrained(
            cfg["hf_name"], trust_remote_code=True)

        def preprocess(img):
            return image_processor(images=img,
                                   return_tensors="pt")["pixel_values"].squeeze(0)

        def tok_fn(texts):
            return tok_inner(text=texts, return_tensors="pt", padding=True,
                             truncation=True)
        tokenizer = tok_fn
        loader_type = "hf_siglip"
    return model.eval(), preprocess, tokenizer, loader_type


def encode_texts(model, tokenizer, loader_type, class_names, device, batch=256):
    """Mean template text features per class, L2-normalized."""
    import torch.nn.functional as F
    feats = []
    with torch.no_grad():
        for i in range(0, len(class_names), batch):
            chunk = class_names[i:i + batch]
            texts = [tpl.format(c.replace("_", " ")) for c in chunk
                     for tpl in TEMPLATES]
            if loader_type == "openclip":
                tok = tokenizer(texts).to(device)
                f = model.encode_text(tok)
            else:
                inputs = tokenizer(texts)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                f = model.get_text_features(**inputs)
            f = F.normalize(f.float(), dim=-1)
            f = f.reshape(len(chunk), len(TEMPLATES), -1).mean(dim=1)
            f = F.normalize(f, dim=-1)
            feats.append(f.cpu())
    return torch.cat(feats, dim=0)  # (C, D)


def encode_images(model, preprocess, loader_type, images, device, batch=64):
    import torch.nn.functional as F
    feats = []
    with torch.no_grad():
        for i in range(0, len(images), batch):
            chunk = images[i:i + batch]
            tensors = [preprocess(im) for im in chunk]
            inp = torch.stack(tensors).to(device)
            if loader_type == "openclip":
                f = model.encode_image(inp)
            else:
                f = model.get_image_features(pixel_values=inp)
            f = F.normalize(f.float(), dim=-1)
            feats.append(f.cpu())
    return torch.cat(feats, dim=0)  # (N, D)


def load_tv_dataset(key, log):
    import torchvision.datasets as dsets
    registry = {
        "food101": dsets.Food101, "flowers102": dsets.Flowers102,
        "fgvc_aircraft": dsets.FGVCAircraft, "dtd": dsets.DTD,
        "oxford_pets": dsets.OxfordIIITPet, "caltech101": dsets.Caltech101,
        "caltech256": dsets.Caltech256, "sun397": dsets.SUN397,
        "ucf101": dsets.UCF101, "gtsrb": dsets.GTSRB, "svhn": dsets.SVHN,
        "stl10": dsets.STL10, "eurosat": dsets.EuroSAT,
        "country211": dsets.Country211, "rendered_sst2": dsets.RenderedSST2,
        "cifar10": dsets.CIFAR10, "imagenet1k": dsets.ImageNet,
    }
    root = Path(__file__).resolve().parents[1] / "data" / "meta_datasets"
    root.mkdir(parents=True, exist_ok=True)
    kw = {"root": str(root), "split": DATASETS[key]["split"]}
    dset = registry[key](download=False, **kw)
    classes = dset.classes
    if key == "imagenet1k" and DATASETS[key].get("classfile"):
        p = Path(__file__).resolve().parents[1] / "data" / "imagenet_classes.txt"
        if not p.exists():
            raise RuntimeError(f"imagenet1k needs {p} (1000 class names, one per line)")
        classes = [l.strip() for l in open(p) if l.strip()]
    return dset, classes, False


def load_folder_dataset(key, log):
    """Folder loader: data/meta_datasets/{key}/ with class-per-subfolder
    (cub200, deepweeds) or wnid subfolders mapped to ImageNet names
    (imagenet_a/r via data/imagenet_class_index.json)."""
    import re
    root = Path(__file__).resolve().parents[1] / "data" / "meta_datasets" / key
    if not root.exists():
        raise RuntimeError(f"{key}: {root} missing (local download -> scp)")
    # safety flatten: if extraction left a single wrapper dir, descend into it
    for _ in range(2):
        kids = sorted(d for d in root.iterdir() if d.is_dir()
                      and not d.name.startswith("."))
        if len(kids) == 1 and not any(
                f.is_file() and f.suffix.lower() in (".jpg", ".jpeg", ".png")
                for f in kids[0].iterdir() if f.is_file()):
            root = kids[0]
        else:
            break
    subdirs = sorted(d for d in root.iterdir() if d.is_dir()
                     and not d.name.startswith("."))
    if DATASETS[key].get("wnid_map"):
        import json as _json
        mp = Path(__file__).resolve().parents[1] / "data" / "imagenet_class_index.json"
        wnid2idx = _json.load(open(mp))  # {"0": ["n01440764", "tench, Tinca tinca"], ...}
        wnid_to_idx = {v[0]: int(k) for k, v in wnid2idx.items()}
        name_of_wnid = {v[0]: v[1].split(", ")[0] for k, v in wnid2idx.items()}
        present = [d.name for d in subdirs if d.name in wnid_to_idx]
        if DATASETS[key].get("wnid_subset"):
            # C = number of present classes (tiny_imagenet: 200)
            classes = [name_of_wnid[w] for w in present]
            cidx = {w: i for i, w in enumerate(present)}
            items = []
            for d in subdirs:
                if d.name not in cidx:
                    continue
                for f in sorted(d.iterdir()):
                    if f.suffix.lower() in (".jpg", ".jpeg", ".png"):
                        items.append((str(f), cidx[d.name]))
            return items, classes, False
        # full 1000-way ImageNet label space (imagenet_a/r)
        classes = [name_of_wnid[wnid2idx[str(i)][0]] for i in range(1000)]
        items = []
        for d in subdirs:
            wnid = d.name
            if wnid not in wnid_to_idx:
                continue
            for f in sorted(d.iterdir()):
                if f.suffix.lower() in (".jpg", ".jpeg", ".png"):
                    items.append((str(f), wnid_to_idx[wnid]))
        return items, classes, False
    def _clean(name):
        return re.sub(r"^\d+\.\s*", "", name).replace("_", " ")
    classes = [_clean(d.name) for d in subdirs]
    items = []
    for d in subdirs:
        for f in sorted(d.iterdir()):
            if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".ppm"):
                items.append((str(f), d.name))
    # class-index map
    cls2idx = {d.name: i for i, d in enumerate(subdirs)}
    items = [(p, cls2idx[c]) for p, c in items]
    return items, classes, False


def load_leafdir_dataset(key, log):
    """Multi-level folder loader (SUN397/FGVC layouts): every directory that
    contains image files is a class; class name = leaf basename, deduped via
    full relative path on collision. cfg['subdir'] restricts the walk root."""
    import re
    root = Path(__file__).resolve().parents[1] / "data" / "meta_datasets" / key
    if not root.exists():
        raise RuntimeError(f"{key}: {root} missing")
    sub = DATASETS[key].get("subdir")
    walk_root = root / sub if sub else root
    if not walk_root.exists():
        raise RuntimeError(f"{key}: {walk_root} missing")
    imgs = list(walk_root.rglob("*.jpg")) + list(walk_root.rglob("*.jpeg")) + \
        list(walk_root.rglob("*.png")) + list(walk_root.rglob("*.ppm"))
    leaf = {}
    for f in imgs:
        leaf.setdefault(f.parent, []).append(f)
    # class name from leaf dir; collision -> relative path
    name_counts = {}
    cls_dirs = sorted(leaf.keys())
    for d in cls_dirs:
        nm = d.name
        name_counts[nm] = name_counts.get(nm, 0) + 1
    classes = []
    dir2idx = {}
    for i, d in enumerate(cls_dirs):
        nm = d.name
        if name_counts[nm] > 1:
            nm = str(d.relative_to(root)).replace("/", "_")
        classes.append(re.sub(r"^\d+\.\s*", "", nm).replace("_", " "))
        dir2idx[str(d)] = i
    items = []
    for d in cls_dirs:
        for f in sorted(leaf[d]):
            items.append((str(f), dir2idx[str(d)]))
    log.info(f"  leafdir: {len(classes)} classes, {len(items)} images")
    return items, classes, False


def load_voc2012(key, log):
    """VOC2012 multi-label classification (20 classes) from the official
    VOCtrainval tar (ImageSets/Main/*_trainval.txt, +1/-1 per class)."""
    base = Path(__file__).resolve().parents[1] / "data" / "meta_datasets" / "voc2012"
    dev = base / "VOCdevkit" / "VOC2012"
    if not dev.exists():
        raise RuntimeError(f"voc2012: {base} missing (local download -> scp)")
    classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car",
               "cat", "chair", "cow", "diningtable", "dog", "horse",
               "motorbike", "person", "pottedplant", "sheep", "sofa", "train",
               "tvmonitor"]
    imgset = {}
    for cls in classes:
        p = dev / "ImageSets" / "Main" / f"{cls}_trainval.txt"
        for line in open(p):
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            imgset.setdefault(parts[0], {})[cls] = parts[1] == "1"
    names = sorted(imgset.keys())
    images = [(str(dev / "JPEGImages" / f"{n}.jpg"), n) for n in names]
    return images, classes, True


def load_tv_dataset_safe(key, log):
    raise RuntimeError("tv loader deprecated; use hf loader")


def load_hf_dataset(key, log):
    from datasets import load_dataset as hf_load
    cfg = DATASETS[key]
    log.info(f"  loading HF dataset {cfg['hf_id']} split={cfg['split']}")
    dset = hf_load(cfg["hf_id"], split=cfg["split"])
    feats = dset.features
    lbl_feat = feats[cfg["lbl"]]
    if hasattr(lbl_feat, "names") and lbl_feat.names:
        classes = list(lbl_feat.names)
    elif hasattr(lbl_feat, "feature") and hasattr(lbl_feat.feature, "names"):
        classes = list(lbl_feat.feature.names)
    else:
        raise RuntimeError(f"{key}: cannot resolve class names from {lbl_feat}")
    if cfg.get("wnid_names"):
        # some mirrors store wnids as class names; remap via ImageNet index
        import json as _json
        root = Path(__file__).resolve().parents[1]
        mp = root / "data" / "imagenet_class_index.json"
        wnid2idx = _json.load(open(mp))
        name_of_wnid = {v[0]: v[1].split(", ")[0] for k, v in wnid2idx.items()}
        # full-ImageNet fallback (words.txt from the official tiny-imagenet zip)
        wf = root / "data" / "words.txt"
        if wf.exists():
            for line in open(wf):
                parts = line.rstrip("\n").split("\t")
                if len(parts) == 2 and parts[0] not in name_of_wnid:
                    name_of_wnid[parts[0]] = parts[1].split(",")[0]
        classes = [name_of_wnid.get(w, w) for w in classes]
        n_mapped = sum(1 for c in classes if not c.startswith("n0"))
        log.info(f"  wnid names remapped ({n_mapped}/{len(classes)} mapped)")
    return dset, classes, False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["all"])
    parser.add_argument("--models", nargs="+",
                        default=["openai", "laion", "siglip2", "eva02", "dfn"])
    parser.add_argument("--max_images", type=int, default=10000)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    ts = time.strftime("%Y%m%d_%H%M%S")
    setup_logging(ts)
    log = logging.getLogger()
    log.info(f"P4 extraction | args={vars(args)}")

    keys = list(DATASETS.keys()) if args.datasets == ["all"] else args.datasets


    log.info("loading model pipelines ...")
    pipelines = {m: load_pipeline(m, args.device) for m in args.models}
    # normalize: (model, preprocess, tokenizer, loader_type)
    log.info("pipelines ready")

    from torch.utils.data import Subset

    for key in keys:
        cfg = DATASETS[key]
        OUT_ROOT.mkdir(parents=True, exist_ok=True)
        if all((OUT_ROOT / f"clip_{m}_{key}_logits.npy").exists() for m in args.models):
            log.info(f"{key}: already extracted, skip")
            continue

        log.info(f"{key}: loading ({cfg['loader']}) ...")
        try:
            if cfg["loader"] == "hf":
                dset, classes, multilabel = load_hf_dataset(key, log)
                from PIL import Image
                images = [r[cfg["img"]] for r in dset]
                if hasattr(images[0], "convert"):
                    images = [im.convert("RGB") for im in images]
                y = np.array([int(r[cfg["lbl"]]) for r in dset], dtype=np.int32)
            elif cfg["loader"] == "folder":
                items, classes, multilabel = load_folder_dataset(key, log)
                from PIL import Image
                images, ys = [], []
                n_bad = 0
                for t in items:
                    try:
                        images.append(Image.open(t[0]).convert("RGB"))
                        ys.append(t[1])
                    except Exception:
                        n_bad += 1
                if n_bad:
                    log.warning(f"  {key}: skipped {n_bad} corrupt images")
                y = np.array(ys, dtype=np.int32)
            elif cfg["loader"] == "leafdir":
                items, classes, multilabel = load_leafdir_dataset(key, log)
                from PIL import Image
                images, ys = [], []
                n_bad = 0
                for t in items:
                    try:
                        images.append(Image.open(t[0]).convert("RGB"))
                        ys.append(t[1])
                    except Exception:
                        n_bad += 1
                if n_bad:
                    log.warning(f"  {key}: skipped {n_bad} corrupt images")
                y = np.array(ys, dtype=np.int32)
            else:
                raise RuntimeError(f"unknown loader {cfg['loader']}")
        except Exception as e:
            log.error(f"{key}: load failed: {e}")
            continue

        cap = min(cfg["cap"] or args.max_images, args.max_images)
        if len(images) > cap:
            rng = np.random.RandomState(args.seed)
            idx = np.sort(rng.choice(len(images), size=cap, replace=False))
            images = [images[i] for i in idx]
            y = y[idx]
        n = len(images)
        C = len(classes)
        log.info(f"{key}: N={n} C={C} multilabel={multilabel}")
        for m in args.models:
            np.save(OUT_ROOT / f"clip_{m}_{key}_labels.npy", y)
        log.info(f"{key}: labels saved for {len(args.models)} models ({y.shape})")

        log.info(f"{key}: {len(images)} images ready, encoding ...")

        for m in args.models:
            model, preprocess, tokenizer, loader_type = pipelines[m]
            txt = encode_texts(model, tokenizer, loader_type, classes, args.device)
            img = encode_images(model, preprocess, loader_type, images, args.device, args.batch)
            logits = (img @ txt.T).numpy().astype(np.float32)  # raw cosine sims
            np.save(OUT_ROOT / f"clip_{m}_{key}_logits.npy", logits)
            del img, logits
            log.info(f"{key}/{m}: logits saved ({n}x{C})")
        del images
        log.info(f"{key}: DONE")
    log.info("extraction finished")
    return 0


if __name__ == "__main__":
    exit(main())
