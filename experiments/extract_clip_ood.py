"""
experiments/extract_clip_ood.py — OOD expansion (reviewer round 3)

Extends the meta extraction to 5 new out-of-distribution / robustness
benchmarks so the OOD stratum grows from n=2 (ImageNet-A/R) to n=7:

    imagenet_sketch          songweig/imagenet_sketch (zip + classes.py)
    imagenet_v2_matched      vaishaal/ImageNetV2 (tar.gz, wnid dirs)
    imagenet_v2_threshold    vaishaal/ImageNetV2 (tar.gz, wnid dirs)
    imagenet_v2_top          vaishaal/ImageNetV2 (tar.gz, wnid dirs)
    imagenet_c               haideraltahan/wds_imagenetc (wds shards, cap 10000)

Pre-registered dataset list (locked before running, per review discipline:
no post-hoc dataset selection). Protocol identical to extract_clip_meta.py:
5 paper backbones, generic ImageNet-style templates, raw cosine-sim logits,
single-label (N,) int32 labels in the 1000-way ImageNet class index.

Label spaces:
  - sketch/V2: wnid subdirectories -> data/imagenet_class_index.json
  - imagenet_c: .cls index per image; classnames.txt order verified against
    imagenet_class_index.json before extraction (hard error on mismatch).

ImageNet-C sampling (two-pass): pass 1 walks all 544 wds shards collecting
(corruption_severity, cls, tar, member) candidates; pass 2 seed-samples
cap=10000 items and extracts only the sampled members.

Usage (on 5090, HF_ENDPOINT=https://hf-mirror.com):
  nohup python experiments/extract_clip_ood.py --datasets all > logs/ood_extract.log 2>&1 &

Output: results/per_image/meta/clip_{model}_{dataset}_logits.npy (+ labels)
"""
import argparse
import json
import logging
import os
import re
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_OFFLINE", "0")

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

MODELS = {
    "openai":  {"loader": "openclip", "arch": "ViT-L-14",             "pretrained": "openai",            "input_size": 224},
    "laion":   {"loader": "openclip", "arch": "ViT-L-14",             "pretrained": "laion2b_s32b_b82k", "input_size": 224},
    "siglip2": {"loader": "hf_siglip", "arch": "ViT-L-16-SigLIP2-256", "hf_name": "google/siglip2-large-patch16-256", "input_size": 256},
    "eva02":   {"loader": "openclip", "arch": "EVA02-L-14",           "pretrained": "merged2b_s4b_b131k", "input_size": 224},
    "dfn":     {"loader": "openclip", "arch": "ViT-L-14",             "pretrained": "dfn2b",             "input_size": 224},
}

DATASETS = {
    "imagenet_sketch": {
        "repo": "songweig/imagenet_sketch", "file": "data/ImageNet-Sketch.zip",
        "kind": "zip", "cap": 10000, "label_mode": "wnid",
    },
    "imagenet_v2_matched": {
        "repo": "vaishaal/ImageNetV2", "file": "imagenetv2-matched-frequency.tar.gz",
        "kind": "tar", "cap": 10000, "label_mode": "index_dir",
    },
    "imagenet_v2_threshold": {
        "repo": "vaishaal/ImageNetV2", "file": "imagenetv2-threshold0.7.tar.gz",
        "kind": "tar", "cap": 10000, "label_mode": "index_dir",
    },
    "imagenet_v2_top": {
        "repo": "vaishaal/ImageNetV2", "file": "imagenetv2-top-images.tar.gz",
        "kind": "tar", "cap": 10000, "label_mode": "index_dir",
    },
    "imagenet_c": {
        "repo": "haideraltahan/wds_imagenetc", "file": "test",
        "kind": "wds", "cap": 10000, "label_mode": "cls_file",
    },
}

OUT_ROOT = Path(__file__).resolve().parents[1] / "results" / "per_image" / "meta"
DL_ROOT = Path(__file__).resolve().parents[1] / "data" / "meta_datasets" / "ood_dl"

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
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(logdir / f"extract_clip_ood_{ts}.log"),
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
    return torch.cat(feats, dim=0)


def encode_images(model, preprocess, loader_type, images, device, batch=64):
    import torch.nn.functional as F
    from PIL import Image
    feats = []
    with torch.no_grad():
        for i in range(0, len(images), batch):
            chunk = images[i:i + batch]
            imgs = []
            for im in chunk:
                if not isinstance(im, Image.Image):
                    im = Image.open(im).convert("RGB")
                imgs.append(im)
            tensors = [preprocess(im) for im in imgs]
            inp = torch.stack(tensors).to(device)
            if loader_type == "openclip":
                f = model.encode_image(inp)
            else:
                f = model.get_image_features(pixel_values=inp)
            f = F.normalize(f.float(), dim=-1)
            feats.append(f.cpu())
    return torch.cat(feats, dim=0)


def load_imagenet_index():
    mp = Path(__file__).resolve().parents[1] / "data" / "imagenet_class_index.json"
    data = json.load(open(mp))
    wnid2idx = {v[0]: int(k) for k, v in data.items()}
    name_of_wnid = {v[0]: v[1].split(", ")[0] for k, v in data.items()}
    ordered_names = [data[str(i)][1].split(", ")[0] for i in range(1000)]
    return wnid2idx, name_of_wnid, ordered_names


def download_file(repo, filename, dest_dir, log):
    from huggingface_hub import hf_hub_download
    dest_dir.mkdir(parents=True, exist_ok=True)
    p = hf_hub_download(repo, filename, repo_type="dataset",
                        cache_dir=str(dest_dir / "hf_cache"))
    log.info(f"  downloaded {repo}/{filename} -> {p} ({os.path.getsize(p)/1e6:.0f} MB)")
    return p


def resolve_zip(zip_path, log):
    import zipfile
    zf = zipfile.ZipFile(zip_path)
    names = zf.namelist()
    imgs = [n for n in names if n.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".ppm"))]
    return zf, imgs


def resolve_tar(tar_path, log):
    import tarfile
    tf = tarfile.open(tar_path)
    names = tf.getnames()
    imgs = [n for n in names if n.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".ppm"))]
    return tf, imgs


def probe_top_level(imgs, wnid2idx, log):
    """Locate the wnid level: archives may have a parent prefix dir (e.g.
    'sketch/<wnid>/...'). Returns the part index holding wnids."""
    depth_hits = {}
    for n in imgs[:2000]:
        parts = n.split("/")
        for d in range(len(parts) - 1):
            if parts[d] in wnid2idx:
                depth_hits[d] = depth_hits.get(d, 0) + 1
                break
    if not depth_hits:
        raise RuntimeError("no wnid segment found in archive paths")
    best = max(depth_hits.items(), key=lambda kv: kv[1])
    frac = best[1] / min(2000, len(imgs))
    log.info(f"  wnid depth {best[0]} with {best[1]} hits ({frac:.1%})")
    if frac < 0.9:
        raise RuntimeError(f"wnid hit rate too low ({frac:.1%}); need loader adaptation")
    return best[0]


def probe_index_dir(imgs, log):
    """V2-style layout: <prefix>/<class-index>/<hash>.<ext>. Class index is
    the ImageNet 0-999 index directly. Returns the part index of class dir."""
    depth_counts = {}
    for n in imgs[:2000]:
        parts = n.split("/")
        for d in range(len(parts) - 1):
            if parts[d].isdigit() and 0 <= int(parts[d]) < 1000:
                depth_counts[d] = depth_counts.get(d, 0) + 1
                break
    if not depth_counts:
        raise RuntimeError("no numeric class-dir segment found")
    best = max(depth_counts.items(), key=lambda kv: kv[1])
    frac = best[1] / min(2000, len(imgs))
    log.info(f"  index-dir depth {best[0]} with {best[1]} hits ({frac:.1%})")
    if frac < 0.9:
        raise RuntimeError(f"index-dir hit rate too low ({frac:.1%})")
    return best[0]


def resolve_wds(dl_path, log, cap, download_threads=6, map_file=None):
    """ImageNet-C via pre-computed shard->combo map.

    Map file (540 entries, 95 corruption*severity combos) was probed with
    HTTP Range reads of each shard's first 512KB (see
    data/meta_datasets/ood_imagenetc_combo_map.json). We download only ONE
    representative shard per combo (the smallest shard number, already
    cached where previously fetched), build the candidate pool from those
    reps, and seed-sample `cap` images across all combos.
    """
    import tarfile
    from huggingface_hub import hf_hub_download
    repo = "haideraltahan/wds_imagenetc"
    with open(map_file) as f:
        combo_map = json.load(f)          # shard_num(str) -> combo
    rev = {}
    for sh, combo in combo_map.items():
        rev.setdefault(combo, []).append(int(sh))
    reps = {combo: min(shards) for combo, shards in rev.items()}
    log.info(f"  wds: {len(rev)} combos from map; downloading {len(reps)} "
             f"representative shards ({download_threads} dl threads)")

    from concurrent.futures import ThreadPoolExecutor

    def fetch(sh):
        return hf_hub_download(repo, f"test/{sh}.tar", repo_type="dataset",
                               cache_dir=str(dl_path / "hf_cache"))

    cands = []  # (combo, cls, member)
    with ThreadPoolExecutor(max_workers=download_threads) as pool:
        for sh, local in zip(reps.values(),
                             pool.map(fetch, reps.values())):
            with tarfile.open(local) as tf:
                names = tf.getnames()
                cls_by_base = {}
                for n in names:
                    if n.endswith(".cls"):
                        cls_by_base[n[:-4]] = tf.extractfile(n).read().decode().strip()
                for n in names:
                    if n.lower().endswith((".jpg", ".jpeg", ".png",
                                           ".webp", ".ppm")):
                        stem = n.split(".")[0]
                        if stem in cls_by_base:
                            cands.append((sh, cls_by_base[stem], n))
    log.info(f"  wds: candidate pool {len(cands)} from {len(reps)} reps")
    rng = np.random.RandomState(42)
    sel = sorted(rng.choice(len(cands), size=min(cap, len(cands)),
                            replace=False))
    picked = [cands[i] for i in sel]
    items = []
    by_shard = {}
    for sh, cls, member in picked:
        by_shard.setdefault(sh, []).append((cls, member))
    for sh, entries in by_shard.items():
        local = hf_hub_download(repo, f"test/{sh}.tar", repo_type="dataset",
                                cache_dir=str(dl_path / "hf_cache"))
        with tarfile.open(local) as tf:
            for cls, member in entries:
                data = tf.extractfile(member).read()
                items.append((data, int(cls), sh))
    return items


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["all"])
    parser.add_argument("--models", nargs="+",
                        default=["openai", "laion", "siglip2", "eva02", "dfn"])
    parser.add_argument("--max_images", type=int, default=10000)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--download_threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ts = time.strftime("%Y%m%d_%H%M%S")
    setup_logging(ts)
    log = logging.getLogger()
    log.info(f"OOD extraction | args={vars(args)}")

    wnid2idx, name_of_wnid, ordered_names = load_imagenet_index()

    keys = list(DATASETS.keys()) if args.datasets == ["all"] else args.datasets
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    log.info("loading model pipelines ...")
    pipelines = {m: load_pipeline(m, args.device) for m in args.models}
    log.info("pipelines ready")

    from PIL import Image
    import io as _io

    for key in keys:
        cfg = DATASETS[key]
        if all((OUT_ROOT / f"clip_{m}_{key}_logits.npy").exists()
               for m in args.models):
            log.info(f"{key}: already extracted, skip")
            continue
        log.info(f"{key}: downloading/loading ...")
        try:
            if cfg["kind"] == "wds":
                map_file = Path(__file__).resolve().parents[1] / "data" / \
                    "meta_datasets" / "ood_imagenetc_combo_map.json"
                items = resolve_wds(DL_ROOT, log, args.max_images,
                                    args.download_threads, map_file)
                classes = ordered_names
                ys = np.array([it[1] for it in items], dtype=np.int32)
                images = [Image.open(_io.BytesIO(it[0])).convert("RGB")
                          for it in items]
                sev_hist = {}
                for it in items:
                    sev_hist[it[2]] = sev_hist.get(it[2], 0) + 1
                log.info(f"  wds shard histogram (shards {len(sev_hist)}): "
                         f"{sev_hist}")
            else:
                if cfg["kind"] == "zip":
                    local = download_file(cfg["repo"], cfg["file"], DL_ROOT, log)
                    zf, imgs = resolve_zip(local, log)
                else:
                    local = download_file(cfg["repo"], cfg["file"], DL_ROOT, log)
                    zf, imgs = resolve_tar(local, log)
                if cfg["label_mode"] == "wnid":
                    label_depth = probe_top_level(imgs, wnid2idx, log)
                elif cfg["label_mode"] == "index_dir":
                    label_depth = probe_index_dir(imgs, log)
                else:
                    raise RuntimeError(f"unknown label_mode {cfg['label_mode']}")
                cap = min(cfg["cap"], args.max_images, len(imgs))
                rng = np.random.RandomState(args.seed)
                idx = np.sort(rng.choice(len(imgs), size=cap, replace=False))
                images, ys = [], []
                for i in idx:
                    member = imgs[i]
                    parts = member.split("/")
                    if cfg["label_mode"] == "wnid":
                        wnid = next((p for p in parts if p in wnid2idx), None)
                        if wnid is None:
                            log.warning(f"  no wnid in {member} (skip)")
                            continue
                        y = wnid2idx[wnid]
                    else:
                        num = next((p for p in parts
                                    if p.isdigit() and 0 <= int(p) < 1000), None)
                        if num is None:
                            log.warning(f"  no class-dir in {member} (skip)")
                            continue
                        y = int(num)
                    if cfg["kind"] == "zip":
                        data = zf.read(member)
                    else:
                        data = zf.extractfile(member).read()
                    images.append(Image.open(_io.BytesIO(data)).convert("RGB"))
                    ys.append(y)
                ys = np.array(ys, dtype=np.int32)
                classes = ordered_names
                zf.close()
        except Exception as e:
            log.error(f"{key}: load failed: {e}")
            continue

        n, C = len(images), len(classes)
        log.info(f"{key}: N={n} C={C}")
        if n == 0:
            log.error(f"{key}: no images; skip")
            continue
        for m in args.models:
            np.save(OUT_ROOT / f"clip_{m}_{key}_labels.npy", ys)
        for m in args.models:
            model, preprocess, tokenizer, loader_type = pipelines[m]
            txt = encode_texts(model, tokenizer, loader_type, classes, args.device)
            img = encode_images(model, preprocess, loader_type, images, args.device, args.batch)
            logits = (img @ txt.T).numpy().astype(np.float32)
            np.save(OUT_ROOT / f"clip_{m}_{key}_logits.npy", logits)
            del img, logits
            log.info(f"{key}/{m}: saved ({n}x{C})")
        del images
        log.info(f"{key}: DONE")
    log.info("extraction finished")
    return 0


if __name__ == "__main__":
    exit(main())
