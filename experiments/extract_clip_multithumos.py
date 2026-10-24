"""
experiments/extract_clip_multithumos.py — Per-frame CLIP extraction on MultiTHUMOS
==================================================================================
MultiTHUMOS: dense frame-level multi-label action annotations over THUMOS14
validation videos (200 videos, 65 classes). Annotations are temporal intervals
[start, end) in seconds with 0.1s precision; videos are 30fps.

Design decisions (locked):
  - Frame sampling: 1 fps via ffmpeg (frame i has timestamp t = i seconds).
  - Frame-label alignment: class c is active at frame i iff there exists an
    annotation interval [s, e) for c with s <= i < e (annotation times are
    multiples of 0.1s, exact float comparison is safe after 1-decimal parse).
  - Per-video isolation: each video is processed independently; completed
    videos are skipped on restart (idempotent).
  - Consensus correctness (downstream): argmax over avg logits in any active GT.

Output:
  results/per_image/clip_{model}_multithumos_logits.npy   (N, 65) float32
  results/per_image/clip_{model}_multithumos_labels.npy   (N, 65) float32
  results/per_image/multithumos_meta.json                 video/frame index map

Usage:
  python experiments/extract_clip_multithumos.py --model openai --videos-dir /data/thumos14/videos \
      --ann-dir /data/thumos14/annotations --fps 1
"""

import argparse, json, logging, shutil, subprocess, sys, time
from pathlib import Path

from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

OUTPUT_DIR = REPO_ROOT / "results" / "per_image"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CLIP_MODELS = ["openai", "laion", "siglip2", "eva02", "dfn"]

MODEL_REGISTRY = {
    "openai":  {"loader": "openclip", "arch": "ViT-L-14", "pretrained": "openai", "input_size": 224},
    "laion":   {"loader": "openclip", "arch": "ViT-L-14", "pretrained": "laion2b_s32b_b82k", "input_size": 224},
    "siglip2": {"loader": "hf_siglip", "arch": "ViT-L-16-SigLIP2-256", "hf_name": "google/siglip2-large-patch16-256", "input_size": 256},
    "eva02":   {"loader": "openclip", "arch": "EVA02-L-14", "pretrained": "merged2b_s4b_b131k", "input_size": 224},
    "dfn":     {"loader": "openclip", "arch": "ViT-L-14", "pretrained": "dfn2b", "input_size": 224},
}

GPU_ASSIGNMENT = {
    "openai": "cuda:0", "siglip2": "cuda:1",
    "laion": "cuda:2", "eva02": "cuda:3", "dfn": "cuda:0",
}


def load_model(model_key: str, device: str):
    cfg = MODEL_REGISTRY[model_key]
    logging.info("[%s] Loading on %s ...", model_key, device)
    if cfg["loader"] == "openclip":
        import open_clip
        model, _, preprocess = open_clip.create_model_and_transforms(
            cfg["arch"], pretrained=cfg["pretrained"], device=device)
        model.eval()
        tokenizer = open_clip.get_tokenizer(cfg["arch"])
        return model, preprocess, tokenizer, "openclip"
    else:
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


@torch.no_grad()
def encode_text(loader_type, model, tokenizer, texts, device):
    if loader_type == "openclip":
        tokens = tokenizer(texts).to(device)
        feats = model.encode_text(tokens)
    else:
        inputs = tokenizer(texts)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        feats = model.get_text_features(**inputs)
    return F.normalize(feats.float(), dim=-1)


@torch.no_grad()
def encode_images(loader_type, model, images):
    if loader_type == "openclip":
        feats = model.encode_image(images)
    else:
        feats = model.get_image_features(pixel_values=images)
    return F.normalize(feats.float(), dim=-1)


ACTION_TEMPLATES = [
    "{}",
    "a photo of {}",
    "an image of {}",
    "a picture of {}",
    "a scene of {}",
]


def camel_to_words(name: str) -> str:
    """'BaseballPitch' -> 'baseball pitch'; 'NoHuman' -> 'no human'."""
    import re
    words = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", name).lower()
    return " ".join(words.split())


def build_class_names(class_list_path: Path) -> tuple:
    """Return (prompt_names, raw_name->idx). class_list.txt lines: '1 BaseballPitch'."""
    prompt_names, raw_to_idx = [], {}
    for line in class_list_path.read_text().strip().splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) == 2:
            idx = len(prompt_names)
            prompt_names.append(camel_to_words(parts[1]))
            raw_to_idx[parts[1]] = idx
    assert len(prompt_names) == 65, f"expected 65 classes, got {len(prompt_names)}"
    return prompt_names, raw_to_idx


@torch.no_grad()
def encode_text_templates(loader_type, model, tokenizer, class_names,
                          templates, device):
    """Encode class_names with logit-ensemble over templates (standard CLIP
    zero-shot protocol). Averages logits, NOT text features — avoids the
    template-dominated embedding collapse seen with text-feature averaging."""
    all_feats = []
    for t in templates:
        texts = [t.format(cn) for cn in class_names]
        feats = encode_text(loader_type, model, tokenizer, texts, device)
        all_feats.append(feats)  # each (C, D), unit-norm
    return torch.stack(all_feats, dim=0)  # (T, C, D) — caller averages logits


def load_annotations(ann_dir: Path, raw_to_idx: dict) -> dict:
    """Return {video_name: {class_idx: [(start, end), ...]}}."""
    ann = {}
    for f in sorted(ann_dir.glob("*.txt")):
        if f.stem not in raw_to_idx:
            continue
        cls_idx = raw_to_idx[f.stem]
        for line in f.read_text().strip().splitlines():
            parts = line.split()
            if len(parts) != 3:
                continue
            vid, s, e = parts[0], float(parts[1]), float(parts[2])
            ann.setdefault(vid, {}).setdefault(cls_idx, []).append((s, e))
    return ann


def extract_frames(video_path: Path, fps: int, out_dir: Path) -> list:
    """Extract frames at {fps} fps into out_dir/frames; returns sorted frame file
    list. The frames/ subdir is wiped first so stale frames from the previous
    video never mix in (out_dir itself is kept — it also holds parts/)."""
    frame_dir = out_dir / "frames"
    shutil.rmtree(frame_dir, ignore_errors=True)
    frame_dir.mkdir(parents=True, exist_ok=True)
    pat = str(frame_dir / "frame_%06d.jpg")
    cmd = ["ffmpeg", "-y", "-i", str(video_path), "-vf", f"fps={fps}",
           "-q:v", "2", pat]
    subprocess.run(cmd, capture_output=True, check=True)
    frames = sorted(frame_dir.glob("frame_*.jpg"))
    return frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=CLIP_MODELS)
    parser.add_argument("--videos-dir", type=Path, required=True)
    parser.add_argument("--ann-dir", type=Path, required=True)
    parser.add_argument("--class-list", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--tmp-frames", type=Path, default=Path("/tmp/mt_frames"))
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler()])
    log = logging.getLogger("extract_multithumos")
    log.info("Model=%s fps=%d", args.model, args.fps)

    class_names, raw_to_idx = build_class_names(args.class_list)
    ann = load_annotations(args.ann_dir, raw_to_idx)
    videos = sorted(args.videos_dir.glob("*.mp4"))
    log.info("Videos: %d, annotated videos: %d", len(videos), len(ann))

    device = GPU_ASSIGNMENT[args.model]
    model, preprocess, tokenizer, loader_type = load_model(args.model, device)
    text_feats = encode_text_templates(loader_type, model, tokenizer,
                                       class_names, ACTION_TEMPLATES, device).cpu()

    tmp_frames = args.tmp_frames / args.model
    parts_dir = tmp_frames / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    all_logits, all_labels, meta = [], [], []
    n_total = len(videos)
    n_done = 0
    for vi, vpath in enumerate(videos):
        vid = vpath.stem
        part_l = parts_dir / f"{vid}_logits.npy"
        part_y = parts_dir / f"{vid}_labels.npy"
        if part_l.exists() and part_y.exists():
            n_done += 1
            log.info("[%d/%d] skip %s (already done)", vi + 1, n_total, vid)
            continue
        if vid not in ann:
            log.info("[%d/%d] skip %s (no annotations)", vi + 1, n_total, vid)
            continue
        t0 = time.time()
        try:
            frames = extract_frames(vpath, args.fps, tmp_frames)
        except subprocess.CalledProcessError as e:
            log.warning("[%d/%d] ffmpeg failed %s: %s", vi + 1, n_total, vid,
                        e.stderr[-200:] if e.stderr else "no stderr")
            continue
        if not frames:
            log.warning("[%d/%d] no frames for %s", vi + 1, n_total, vid)
            continue

        labels = np.zeros((len(frames), 65), dtype=np.float32)
        frame_times = np.arange(len(frames), dtype=np.float64) / args.fps
        for cls_idx, intervals in ann[vid].items():
            for s, e in intervals:
                mask = (frame_times >= s - 1e-9) & (frame_times < e)
                labels[mask, cls_idx] = 1.0

        logits = []
        for b in range(0, len(frames), args.batch_size):
            chunk = frames[b:b + args.batch_size]
            imgs = torch.stack([preprocess(Image.open(p).convert("RGB"))
                                for p in chunk]).to(device)
            feats = encode_images(loader_type, model, imgs).cpu()  # (B, D)
            logits_sum = (feats @ text_feats[0].T)  # (B, C)
            for t_idx in range(1, text_feats.shape[0]):
                logits_sum += (feats @ text_feats[t_idx].T)
            logits.append((logits_sum / text_feats.shape[0]).numpy())
        video_logits = np.concatenate(logits, axis=0).astype(np.float32)

        np.save(part_l, video_logits)
        np.save(part_y, labels)
        n_done += 1
        log.info("[%d/%d] %s: %d frames in %.0fs (acc=%.3f)",
                 vi + 1, n_total, vid, len(frames), time.time() - t0,
                 float((labels[np.arange(len(frames)),
                               np.argmax(video_logits, 1)] > 0).mean()))

    log.info("Done: %d videos", n_done)
    logit_parts = sorted(parts_dir.glob("*_logits.npy"))
    label_parts = sorted(parts_dir.glob("*_labels.npy"))
    if not logit_parts:
        log.error("No frames extracted")
        sys.exit(1)
    if [p.stem[: -len("_logits")] for p in logit_parts] != \
       [p.stem[: -len("_labels")] for p in label_parts]:
        log.error("Mismatched logit/label parts")
        sys.exit(1)
    all_logits = [np.load(p) for p in logit_parts]
    all_labels = [np.load(p) for p in label_parts]
    for p in logit_parts:
        vid = p.stem[: -len("_logits")]
        meta.extend({"video": vid, "frame": i}
                    for i in range(np.load(p).shape[0]))

    L = np.concatenate(all_logits, axis=0)
    Y = np.concatenate(all_labels, axis=0)
    log.info("Total frames: %d", len(L))
    np.save(OUTPUT_DIR / f"clip_{args.model}_multithumos_logits.npy", L)
    np.save(OUTPUT_DIR / f"clip_{args.model}_multithumos_labels.npy", Y)
    with open(OUTPUT_DIR / "multithumos_meta.json", "w") as f:
        json.dump({"model": args.model, "fps": args.fps,
                   "n_classes": 65, "n_frames": len(L), "class_names": class_names,
                   "samples": meta}, f, indent=1)
    log.info("Saved %d frames (model=%s)", len(L), args.model)


if __name__ == "__main__":
    main()
