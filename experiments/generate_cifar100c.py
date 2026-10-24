"""
experiments/generate_cifar100c.py — generate CIFAR-100-C locally (V100 server)
==============================================================================
Hendrycks & Dietterich 2019 corruption transforms, numpy/opencv implementation,
applied to the CIFAR-100 test set (10,000 images). Outputs one .npy per
(corruption, severity) combo with the same 19 corruption names as
robro/cifar10-c-parquet, so CIFAR-10-C and CIFAR-100-C share protocol.

Output:
  data/cifar100c/cifar100c_{corruption}_s{severity}.npy   (10000,32,32,3) uint8
  data/cifar100c/cifar100c_labels.npy                     (10000,) int32

Reproducibility:
  python experiments/generate_cifar100c.py --data_root /path/data
  # requires CIFAR-100 test parquet at {data_root}/cifar100_test/ (uoft-cs/cifar100)
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import cv2
from PIL import Image
import io

CORRUPTIONS = [
    "gaussian_noise", "shot_noise", "impulse_noise", "defocus_blur",
    "glass_blur", "motion_blur", "zoom_blur", "snow", "frost", "fog",
    "brightness", "contrast", "elastic_transform", "pixelate",
    "jpeg_compression", "speckle_noise", "gaussian_blur", "spatter",
    "saturate",
]
SEVERITIES = [1, 2, 3, 4, 5]


def setup_logging(log_dir, ts, name="generate_cifar100c"):
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


# ---- standard Hendrycks corruption transforms (numpy/opencv) ----

def gaussian_noise(x, severity):
    c = [0.08, 0.12, 0.18, 0.26, 0.38][severity - 1]
    return np.clip(x + np.random.normal(size=x.shape, scale=c) * 255, 0, 255).astype(np.uint8)


def shot_noise(x, severity):
    c = [60, 25, 12, 5, 3][severity - 1]
    return np.clip(np.random.poisson(x.astype(np.float64) * c) / c, 0, 255).astype(np.uint8)


def impulse_noise(x, severity):
    c = [0.03, 0.06, 0.09, 0.17, 0.27][severity - 1]
    out = x.copy().astype(np.float64)
    for i in range(3):
        mask = np.random.rand(*x.shape[:2]) < c
        out[..., i][mask] = np.random.choice([0, 255], size=mask.sum())
    return np.clip(out, 0, 255).astype(np.uint8)


def disk(radius, alias_blur=0.1, dtype=np.float32):
    if radius <= 8:
        L = np.arange(-8, 8 + 1)
        ksize = (3, 3)
    else:
        L = np.arange(-radius, radius + 1)
        ksize = (5, 5)
    X, Y = np.meshgrid(L, L)
    aliased_disk = ((X ** 2 + Y ** 2) <= radius ** 2).astype(dtype)
    aliased_disk /= np.sum(aliased_disk)
    if alias_blur > 0.05:
        cv2_kernel = cv2.GaussianBlur(aliased_disk, ksize, alias_blur)
        return cv2_kernel
    return aliased_disk


def defocus_blur(x, severity):
    c = [(3, 0.1), (4, 0.5), (6, 0.5), (8, 0.5), (10, 0.5)][severity - 1]
    kernel = disk(*c)
    out = np.zeros_like(x)
    for i in range(3):
        out[..., i] = cv2.filter2D(x[..., i], -1, kernel)
    return np.clip(out, 0, 255).astype(np.uint8)


def glass_blur(x, severity):
    c = [(0.7, 1, 2), (0.9, 2, 1), (1, 2, 3), (1.1, 3, 2), (1.5, 4, 2)][severity - 1]
    sigma, max_delta, iterations = c
    x = np.uint8(cv2.GaussianBlur(x.astype(np.float64), (3, 3), sigma) + 0)
    H, W = x.shape[:2]
    for _ in range(iterations):
        for h in range(H - max_delta, max_delta, -1):
            for w in range(W - max_delta, max_delta, -1):
                dx, dy = np.random.randint(-max_delta, max_delta, 2)
                h_prime, w_prime = h + dy, w + dx
                x[h, w], x[h_prime, w_prime] = x[h_prime, w_prime], x[h, w]
    return np.clip(cv2.GaussianBlur(x, (3, 3), sigma), 0, 255).astype(np.uint8)


def motion_blur(x, severity):
    c = [(6, 1), (8, 1.5), (12, 2), (14, 2.5), (16, 3)][severity - 1]
    radius, sigma = c
    kernel = np.zeros((2 * radius + 1, 2 * radius + 1))
    kernel[radius, radius] = 1
    length = int(2 * radius + 1)
    kernel = cv2.GaussianBlur(kernel, (length, length), sigma)
    out = np.zeros_like(x)
    for i in range(3):
        out[..., i] = cv2.filter2D(x[..., i], -1, kernel)
    return np.clip(out, 0, 255).astype(np.uint8)


def zoom_blur(x, severity):
    c = [np.arange(1, 1.02, 0.002), np.arange(1, 1.05, 0.005),
         np.arange(1, 1.09, 0.008), np.arange(1, 1.14, 0.01),
         np.arange(1, 1.19, 0.012)][severity - 1]
    H, W = x.shape[:2]
    out = np.zeros_like(x, dtype=np.float64)
    for k in c:
        M = cv2.getRotationMatrix2D((H / 2, W / 2), 0, 1.0 / float(k))
        out += cv2.warpAffine(x.astype(np.float64), M, (W, H), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REFLECT)
    return np.clip(out / len(c), 0, 255).astype(np.uint8)


def snow(x, severity):
    c = [(0.1, 0.3, 3, 0.5, 10, 4, 0.8), (0.2, 0.3, 2, 0.5, 12, 4, 0.7),
         (0.55, 0.3, 4, 0.9, 12, 8, 0.7), (0.55, 0.3, 4.5, 0.85, 12, 8, 0.65),
         (0.55, 0.3, 2.5, 0.75, 12, 12, 0.55)][severity - 1]
    snow_layer = np.random.normal(size=x.shape[:2], loc=c[0], scale=c[1])
    snow_layer = np.clip(snow_layer * 255, 0, 255).astype(np.uint8)
    snow_layer = cv2.blur(snow_layer, (int(c[2]), int(c[2])))
    out = np.zeros_like(x, dtype=np.float64)
    for i in range(3):
        out[..., i] = cv2.addWeighted(x[..., i], c[5], snow_layer, 1 - c[5], 0)
    M = cv2.getRotationMatrix2D((x.shape[1] / 2, x.shape[0] / 2), 0, 1.1)
    out = cv2.warpAffine(out, M, (x.shape[1], x.shape[0]),
                         borderMode=cv2.BORDER_REFLECT)
    return np.clip(out, 0, 255).astype(np.uint8)


def frost(x, severity):
    rng = np.random.RandomState(42)
    size = (x.shape[0] // 2, x.shape[1] // 2)
    frost_pat = np.zeros(size, dtype=np.float64)
    for _ in range(50):
        x0, y0 = rng.randint(0, size[0]), rng.randint(0, size[1])
        for k in range(1, 4):
            frost_pat[(x0 + k) % size[0], (y0 + k) % size[1]] = 1
    frost_pat = cv2.blur(frost_pat, (10, 10))
    frost_pat = np.kron(frost_pat, np.ones((2, 2)))
    c = [1, 0.9, 0.7, 0.4, 0.2][severity - 1]
    out = np.zeros_like(x, dtype=np.float64)
    for i in range(3):
        out[..., i] = x[..., i] * (1 - c) + 255 * c * frost_pat
    return np.clip(out, 0, 255).astype(np.uint8)


def fog(x, severity):
    c = [(1.5, 2), (2, 2), (2.5, 1.7), (2.5, 1.5), (3, 1.4)][severity - 1]
    gamma, add = c
    out = x.astype(np.float64)
    for i in range(3):
        out[..., i] = (out[..., i] * (1 - add) + add * 255)
    out = np.clip(out ** (1.0 / gamma), 0, 255)
    return np.clip(out, 0, 255).astype(np.uint8)


def brightness(x, severity):
    c = [0.1, 0.2, 0.3, 0.4, 0.5][severity - 1]
    return np.clip(x * (1 + c), 0, 255).astype(np.uint8)


def contrast(x, severity):
    c = [0.4, 0.3, 0.2, 0.1, 0.05][severity - 1]
    mean = np.mean(x, axis=(0, 1), keepdims=True)
    return np.clip((x - mean) * c + mean, 0, 255).astype(np.uint8)


def elastic_transform(x, severity):
    c = [(244 * 2, 720 * 3, 0.12), (244 * 2, 720 * 3, 0.14),
         (244 * 2, 720 * 3, 0.16), (244 * 2, 720 * 3, 0.18),
         (244 * 2, 720 * 3, 0.2)][severity - 1]
    alpha, sigma, scale = c
    rng = np.random.RandomState(42)
    H, W = x.shape[:2]
    dx = cv2.GaussianBlur((rng.rand(H, W) * 2 - 1).astype(np.float32),
                          (17, 17), sigma) * alpha
    dy = cv2.GaussianBlur((rng.rand(H, W) * 2 - 1).astype(np.float32),
                          (17, 17), sigma) * alpha
    dx, dy = cv2.resize(dx, (W, H)), cv2.resize(dy, (W, H))
    grid_x, grid_y = np.meshgrid(np.arange(W), np.arange(H))
    map_x = (grid_x + dx).astype(np.float32)
    map_y = (grid_y + dy).astype(np.float32)
    out = np.zeros_like(x)
    for i in range(3):
        out[..., i] = cv2.remap(x[..., i], map_x, map_y,
                                interpolation=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_REFLECT_101)
    return np.clip(out, 0, 255).astype(np.uint8)


def pixelate(x, severity):
    c = [0.6, 0.5, 0.4, 0.3, 0.25][severity - 1]
    H, W = x.shape[:2]
    small = cv2.resize(x, (int(W * c), int(H * c)), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(small, (W, H), interpolation=cv2.INTER_NEAREST)


def jpeg_compression(x, severity):
    c = [25, 18, 15, 10, 7][severity - 1]
    im = Image.fromarray(np.ascontiguousarray(x))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=c)
    return np.array(Image.open(io.BytesIO(buf.getvalue())))


def speckle_noise(x, severity):
    c = [0.15, 0.2, 0.35, 0.45, 0.6][severity - 1]
    noise = np.random.normal(size=x.shape, scale=c)
    return np.clip(x + x * noise, 0, 255).astype(np.uint8)


def gaussian_blur(x, severity):
    c = [1, 2, 4, 8, 12][severity - 1]
    return np.clip(cv2.GaussianBlur(x, (3, 3), c), 0, 255).astype(np.uint8)


def spatter(x, severity):
    c = [(0.65, 0.3, 4, 0.69, 0.6, 0), (0.65, 0.3, 3, 0.68, 0.6, 0),
         (0.65, 0.3, 2, 0.68, 0.5, 0), (0.65, 0.3, 1, 0.65, 1.5, 1),
         (0.67, 0.4, 1, 0.65, 1.5, 1)][severity - 1]
    liquid_layer = np.random.normal(size=x.shape[:2], loc=c[0], scale=c[1])
    liquid_layer = np.clip(liquid_layer * 255, 0, 255).astype(np.uint8)
    liquid_layer = cv2.blur(liquid_layer, (int(c[2]), int(c[2])))
    if severity < 5:
        M = cv2.getRotationMatrix2D((x.shape[1] / 2, x.shape[0] / 2), 45, 1.0)
        liquid_layer = cv2.warpAffine(liquid_layer, M, (x.shape[1], x.shape[0]),
                                      borderMode=cv2.BORDER_REFLECT)
    out = np.zeros_like(x, dtype=np.float64)
    for i in range(3):
        out[..., i] = cv2.addWeighted(x[..., i], c[4], liquid_layer, 1 - c[4], 0)
    return np.clip(out, 0, 255).astype(np.uint8)


def saturate(x, severity):
    c = [(1.3, 1.9), (1.3, 2.9), (2.2, 3.9), (2.4, 4.9), (2.5, 5.9)][severity - 1]
    hsv = cv2.cvtColor(x, cv2.COLOR_RGB2HSV)
    hsv[..., 1] = np.clip(hsv[..., 1] * c[0] ** c[1], 0, 255)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


CORRUPTION_FUNCS = {
    "gaussian_noise": gaussian_noise, "shot_noise": shot_noise,
    "impulse_noise": impulse_noise, "defocus_blur": defocus_blur,
    "glass_blur": glass_blur, "motion_blur": motion_blur,
    "zoom_blur": zoom_blur, "snow": snow, "frost": frost, "fog": fog,
    "brightness": brightness, "contrast": contrast,
    "elastic_transform": elastic_transform, "pixelate": pixelate,
    "jpeg_compression": jpeg_compression, "speckle_noise": speckle_noise,
    "gaussian_blur": gaussian_blur, "spatter": spatter, "saturate": saturate,
}


def load_cifar100_test(data_root, log):
    """Load CIFAR-100 test images from uoft-cs/cifar100 parquet (test split)."""
    import pyarrow.parquet as pq
    files = sorted(Path(data_root).glob("cifar100_test/*.parquet"))
    if not files:
        raise FileNotFoundError(
            f"no parquet under {data_root}/cifar100_test; download uoft-cs/cifar100 test split first"
        )
    tbl = pq.read_table(str(files[0]))
    n = tbl.num_rows
    log.info(f"  CIFAR-100 test: {n} rows")
    images = np.empty((n, 32, 32, 3), dtype=np.uint8)
    for i in range(n):
        raw = tbl.column("img")[i]["bytes"].as_py()
        images[i] = np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"))
    labels = np.asarray(tbl.column("fine_label").to_numpy(), dtype=np.int32)
    return images, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default=str(Path(__file__).resolve().parents[1] / "data"))
    parser.add_argument("--log_dir", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cap", type=int, default=10000,
                        help="target total images; each (corruption, severity) combo "
                             "generates ceil(cap / (19*5)) images drawn from CIFAR-100 test")
    args = parser.parse_args()

    ts = time.strftime("%Y%m%d_%H%M%S")
    log_dir = Path(args.log_dir) if args.log_dir else Path("/tmp/cifarc_logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log = setup_logging(log_dir, ts)

    images, labels = load_cifar100_test(Path(args.data_root), log)
    n_combo = len(CORRUPTIONS) * len(SEVERITIES)
    per_combo = int(np.ceil(args.cap / n_combo))
    log.info(f"  cap={args.cap}, n_combos={n_combo}, per_combo={per_combo}")
    assert per_combo <= len(images), "per_combo exceeds test size"
    rng = np.random.RandomState(args.seed)
    sel = rng.permutation(len(images))[:per_combo]

    out = Path(args.data_root) / "cifar100c"
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "cifar100c_labels.npy", labels[sel])

    for corr in CORRUPTIONS:
        fn = CORRUPTION_FUNCS[corr]
        for sev in SEVERITIES:
            p = out / f"cifar100c_{corr}_s{sev}.npy"
            if p.exists():
                log.info(f"  skip {p.name}")
                continue
            np.random.seed(args.seed)
            corrupted = np.stack([fn(im, sev) for im in images[sel]])
            np.save(p, corrupted)
            log.info(f"  {p.name} ({corrupted.shape})")

    log.info("Done.")


if __name__ == "__main__":
    main()
