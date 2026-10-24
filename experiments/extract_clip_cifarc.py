"""
experiments/extract_clip_cifarc.py — CLIP per-image extraction on CIFAR-10-C / CIFAR-100-C
==========================================================================================
Extracts per-image CLIP logits for all 5 backbones on corrupted CIFAR test sets
(CIFAR-10-C / CIFAR-100-C, Hendrycks & Dietterich 2019).

Protocol matches extract_clip_ood.py (ImageNet-C):
  - 10 generic ImageNet templates averaged for text features
  - raw cosine logits, single-label (N,) int32 labels (class index 0..C-1)
  - cap N=10000, seed 42, stratified sampling across (corruption, severity) combos

Data sources (V100 server, no external network except hf-mirror):
  - CIFAR-10-C: robro/cifar10-c-parquet (950k images, 19 corruptions x 5 severities)
  - CIFAR-100-C: generated locally from CIFAR-100 test (uoft-cs/cifar100 parquet)
    using the standard Hendrycks corruption transforms (opencv/numpy),
    stored as data/cifar100c/cifar100c_{corruption}_s{severity}.npy (10000,32,32,3) uint8.

Output:
  results/per_image/cifarc/clip_{model}_{dataset}_logits.npy   (10000, C)
  results/per_image/cifarc/clip_{model}_{dataset}_labels.npy   (10000,) int32

Reproducibility (on GPU server):
  python experiments/extract_clip_cifarc.py --dataset cifar10c --data_root /path/data
  python experiments/extract_clip_cifarc.py --dataset cifar100c --data_root /path/data  # after generation
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_clip_per_image_v2 import (
    CLIP_MODELS, load_model, encode_text, encode_images,
)

IMAGENET_TEMPLATES = [
    "a photo of a {}",
    "a rendering of a {}",
    "a cropped photo of the {}",
    "the photo of a {}",
    "a photo of one {}",
    "a good photo of the {}",
    "a photo of one {}",
    "a rendition of the {}",
    "a photo of the clean {}",
    "a photography of the {}",
    "a good photo of the {}",
    "a photo of the nice {}",
    "a photo of the odd {}",
    "a photo of the larger {}",
    "a black and white photo of the {}",
    "a dark photo of the {}",
    "a blurry photo of the {}",
]

CIFAR10_CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog",
                   "frog", "horse", "ship", "truck"]
CIFAR100_CLASSES = [
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

# 19 corruption types, ordered as in CIFAR-C (Hendrycks & Dietterich 2019)
CORRUPTIONS = [
    "gaussian_noise", "shot_noise", "impulse_noise", "defocus_blur",
    "glass_blur", "motion_blur", "zoom_blur", "snow", "frost", "fog",
    "brightness", "contrast", "elastic_transform", "pixelate",
    "jpeg_compression", "speckle_noise", "gaussian_blur", "spatter",
    "saturate",
]
SEVERITIES = [1, 2, 3, 4, 5]


def setup_logging(log_dir, ts, name="extract_clip_cifarc"):
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


def sample_combo_idx(combo_counts, n_total, cap, seed):
    """Stratified sampling across combos: draw floor(cap * count_i / n_total) per
    combo, then top up remaining slots from the largest combos (rng-shuffled).
    Mirrors extract_clip_ood.py's ImageNet-C sampling."""
    rng = np.random.RandomState(seed)
    n_combo = len(combo_counts)
    frac = cap / n_total
    per_combo = np.maximum(1, np.floor(np.array(combo_counts) * frac).astype(int))
    while per_combo.sum() > cap:
        # shrink largest first (deterministic for reproducibility)
        j = int(np.argmax(per_combo))
        per_combo[j] -= 1
    rem = cap - int(per_combo.sum())
    if rem > 0:
        order = rng.permutation(n_combo)
        for j in order:
            if rem == 0:
                break
            if per_combo[j] < combo_counts[j]:
                per_combo[j] += 1
                rem -= 1
    return per_combo


def load_cifar10c_parquet(data_root, log):
    """Load CIFAR-10-C from robro/cifar10-c-parquet (arrow stream files) via pyarrow.
    Each shard contains only 4-5 corruption types x 5 severities; ALL shards must be
    merged to cover the full 19-corruption protocol."""
    import pyarrow.ipc as ipc
    log.info("  loading CIFAR-10-C arrow (robro/cifar10-c-parquet) ...")
    files = sorted(
        [str(p) for p in Path(data_root).glob("data-*.arrow")]
    )
    if not files:
        raise FileNotFoundError(
            f"no data-*.arrow in {data_root}; download robro/cifar10-c-parquet first"
        )
    tbls = []
    for f in files:
        with open(f, "rb") as fh:
            tbls.append(ipc.open_stream(fh).read_all())
    log.info(f"  loaded {len(files)} shards")
    return tbls


def sample_cifar10c(tbls, cap, seed, log):
    """Stratified sample across (corruption, severity) combos; returns
    images (N,32,32,3) uint8, labels (N,) int32, combo tags (N,) str."""
    import pyarrow as pa
    tbl = pa.concat_tables(tbls, promote_options="default")
    n = tbl.num_rows
    log.info(f"  total rows {n}")
    label_arr = np.asarray(tbl.column("label").to_numpy(), dtype=np.int32)
    cname = np.asarray(tbl.column("corruption_name").to_numpy())
    clevel = np.asarray(tbl.column("corruption_level").to_numpy(), dtype=np.int32)
    combos = [f"{c}_s{l}" for c, l in zip(cname, clevel)]
    u, counts = np.unique(combos, return_counts=True)
    log.info(f"  {len(u)} combos: {dict(zip(u[:10], counts[:10]))} ...")
    per_combo = sample_combo_idx(counts, n, cap, seed)
    rng = np.random.RandomState(seed)
    rows = []
    tags = []
    for j, tag in enumerate(u):
        idx = np.where(np.array(combos) == tag)[0]
        chosen = rng.choice(idx, size=int(per_combo[j]), replace=False)
        rows.extend(chosen.tolist())
        tags.extend([tag] * int(per_combo[j]))
    rows = np.array(rows)
    tags = np.array(tags)
    log.info(f"  sampled {len(rows)} rows across {len(u)} combos")
    images = np.empty((len(rows), 32, 32, 3), dtype=np.uint8)
    from PIL import Image as PILImage
    import io
    for i, r in enumerate(rows):
        raw = tbl.column("image")[r]["bytes"].as_py()
        images[i] = np.asarray(PILImage.open(io.BytesIO(raw)).convert("RGB"))
    return images, label_arr[rows].astype(np.int32), tags


def load_cifar100c_npy(data_root, log):
    """Load locally generated CIFAR-100-C (data/cifar100c/*.npy, one per combo)."""
    d = Path(data_root) / "cifar100c"
    files = sorted(d.glob("cifar100c_*_s[1-5].npy"))
    if not files:
        raise FileNotFoundError(
            f"no cifar100c_*_s1-5.npy in {d}; run generate_cifar100c.py first"
        )
    per_combo = []
    for p in files:
        arr = np.load(p)
        per_combo.append(arr)
    labels_full = np.load(d / "cifar100c_labels.npy")
    n_combo = len(files)
    per_combo_n = per_combo[0].shape[0]
    log.info(f"  loaded {n_combo} combos x {per_combo_n} images")
    return per_combo, labels_full


def sample_cifar100c(per_combo, labels_full, cap, seed, log):
    """Stratified sampling across combos, same protocol as CIFAR-10-C."""
    n_combo = len(per_combo)
    counts = np.array([p.shape[0] for p in per_combo])
    n_total = int(counts.sum())
    per = sample_combo_idx(counts, n_total, cap, seed)
    rng = np.random.RandomState(seed)
    imgs = []
    labs = []
    for j, arr in enumerate(per_combo):
        chosen = rng.choice(arr.shape[0], size=int(per[j]), replace=False)
        imgs.append(arr[chosen])
        labs.append(labels_full[chosen])
    images = np.concatenate(imgs, axis=0)
    labels = np.concatenate(labs, axis=0).astype(np.int32)
    log.info(f"  sampled {len(images)} images across {n_combo} combos")
    return images, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="cifar10c", choices=["cifar10c", "cifar100c"])
    parser.add_argument("--model", default="all", choices=CLIP_MODELS + ["all"])
    parser.add_argument("--data_root", default=str(Path(__file__).resolve().parents[1] / "data"))
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--log_dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--cap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ts = time.strftime("%Y%m%d_%H%M%S")
    log_dir = Path(args.log_dir) if args.log_dir else Path("/tmp/cifarc_logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log = setup_logging(log_dir, ts)

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    log.info(f"device={device} dataset={args.dataset} cap={args.cap} seed={args.seed}")

    if args.dataset == "cifar10c":
        tbls = load_cifar10c_parquet(Path(args.data_root) / "cifar10c", log)
        images, labels, _tags = sample_cifar10c(tbls, args.cap, args.seed, log)
        classes = CIFAR10_CLASSES
    else:
        per_combo, labels_full = load_cifar100c_npy(Path(args.data_root), log)
        images, labels = sample_cifar100c(per_combo, labels_full, args.cap, args.seed, log)
        classes = CIFAR100_CLASSES

    log.info(f"  images {images.shape}, labels {labels.shape}, classes {len(classes)}")

    import extract_clip_per_image_v2 as v2
    out_dir = Path(args.output_dir) if args.output_dir else (
        Path(__file__).resolve().parents[1] / "results" / "per_image" / "cifarc"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    models = CLIP_MODELS if args.model == "all" else [args.model]
    for mk in models:
        try:
            model, preprocess, tokenizer, loader_type = v2.load_model(mk, device)
            model.eval()
            t = [tpl.format(c) for c in classes for tpl in IMAGENET_TEMPLATES]
            # average over templates per class
            per_tpl = []
            for i in range(0, len(t), len(IMAGENET_TEMPLATES)):
                chunk = t[i:i + len(IMAGENET_TEMPLATES)]
                feats = v2.encode_text(loader_type, model, tokenizer, chunk, device)
                per_tpl.append(feats.mean(0))
            text_feats = torch.stack(per_tpl)
            text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
            log.info(f"[{mk}] text feats {text_feats.shape}")

            batch_size = 512
            all_logits = []
            from PIL import Image as PILImage
            with torch.no_grad():
                for i in range(0, len(images), batch_size):
                    b = images[i:i + batch_size]
                    tensors = torch.stack([
                        preprocess(PILImage.fromarray(x)) for x in b
                    ]).to(device)
                    img_feats = v2.encode_images(loader_type, model, tensors)
                    logits = img_feats @ text_feats.T
                    all_logits.append(logits.cpu().numpy())
            logits_arr = np.concatenate(all_logits, axis=0).astype(np.float32)
            acc = (logits_arr.argmax(1) == labels).mean()
            log.info(f"[{mk}] acc={acc:.4f}")
            np.save(out_dir / f"clip_{mk}_{args.dataset}_logits.npy", logits_arr)
            np.save(out_dir / f"clip_{mk}_{args.dataset}_labels.npy", labels)
            log.info(f"  saved clip_{mk}_{args.dataset}_*.npy to {out_dir}/")
        except Exception as e:
            log.error(f"[{mk}] {e}", exc_info=True)

    log.info("Done.")


if __name__ == "__main__":
    main()
