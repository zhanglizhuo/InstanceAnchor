"""
main_mllm_v2.py — Per-image MLLM zero-shot inference (v2)
==========================================================
Runs Ollama MLLM inference on all images across all 3 SCB subsets
and saves per-image predictions for downstream per-image analysis.

Key differences from main project's main_mllm.py:
  - Saves per-image predictions (not just aggregate metrics)
  - Checkpoint/resume support (crash-safe)
  - Model-agnostic: any Ollama model passed via --model
  - Outputs JSONL with one prediction per image

Usage:
  # Single subset, single model
  python experiments/main_mllm_v2.py --model qwen3.5:27b --subset teacher_behavior

  # All subsets
  python experiments/main_mllm_v2.py --model qwen3.5:27b --subset all

  # Resume from checkpoint
  python experiments/main_mllm_v2.py --model qwen3.5:27b --subset all --resume

Output:
  results/per_image/mllm_{model}_{subset}_preds.jsonl
  Each line: {"index": 0, "gt_labels": [...], "pred_classes": [...], "raw_response": "..."}

Reproducibility:
  Full pipeline: bash experiments/run_mllm_v2.sh
"""

import os
import json
import time
import base64
import io
import argparse
import sys
import signal
from pathlib import Path
from typing import List, Optional

import numpy as np
import requests
from PIL import Image
import torchvision.transforms as T

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.scb_dataset import build_dataloader, SUBSET_CONFIG, SUBSET_LOCAL_DIRS

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
SCB_DATA_DIR = os.environ.get("SCB_DATA_DIR",
    "/home/broadsense/works/lizhuo/AutoResearchClaw/datasets_scb")

SUBSETS = ["teacher_behavior", "handrise_readwrite", "bow_turnhead"]
SUBSET_KEY_MAP = {
    "teacher_behavior": "SCB5_TeacherBehavior",
    "handrise_readwrite": "SCB5_HandriseReadWrite",
    "bow_turnhead": "SCB_BowTurnHead",
}

to_pil = T.ToPILImage()

interrupted = False


def _handle_sigint(sig, frame):
    global interrupted
    interrupted = True
    print("\n[MLLM_v2] SIGINT — finishing current image, then saving checkpoint...")


signal.signal(signal.SIGINT, _handle_sigint)


def load_checkpoint(output_path: Path) -> dict:
    """Load existing predictions for resume."""
    if not output_path.exists():
        return {}
    done = {}
    with open(output_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                done[rec["index"]] = rec
            except json.JSONDecodeError:
                continue
    return done


def build_prompt(classes: List[str], multilabel: bool) -> str:
    """Build zero-shot classification prompt."""
    if multilabel:
        return (
            f"You are a classroom behavior classifier. The image may contain "
            f"multiple behaviors simultaneously. Select ALL applicable classes "
            f"from the following list: {json.dumps(classes)}.\n\n"
            f"Return ONLY a JSON object: {{\"classes\": [\"class1\", \"class2\"]}}\n"
            f"If no class applies, return {{\"classes\": []}}."
        )
    else:
        return (
            f"You are a classroom behavior classifier. Select the SINGLE best "
            f"matching class from: {json.dumps(classes)}.\n\n"
            f"Return ONLY a JSON object: {{\"class\": \"class_name\"}}"
        )


def parse_response(response_text: str, classes: List[str], multilabel: bool) -> List[str]:
    """Extract predicted class(es) from MLLM response. Returns empty list on failure."""
    try:
        # Extract JSON from response (may be wrapped in markdown code block)
        text = response_text.strip()
        if "```" in text:
            # Extract content between first ``` and last ```
            parts = text.split("```")
            for part in parts:
                part = part.strip()
                if part.startswith("json"):
                    part = part[4:].strip()
                if "{" in part:
                    text = part
                    break
        # Find JSON object
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            text = text[start:end]
        parsed = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return []

    if multilabel:
        preds = parsed.get("classes", [])
        if isinstance(preds, list):
            return [c for c in preds if c in classes]
        return []
    else:
        pred = parsed.get("class", "")
        if isinstance(pred, str) and pred in classes:
            return [pred]
        # Fallback: try "classes" key even for single-label
        preds = parsed.get("classes", [])
        if isinstance(preds, list) and preds:
            return [preds[0]] if preds[0] in classes else []
        return []


def encode_image(img_tensor) -> str:
    """Convert CLIP-normalized tensor to base64 JPEG."""
    img = to_pil(img_tensor)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=70)
    return base64.b64encode(buf.getvalue()).decode()


def run_inference(model: str, subset: str, output_dir: Path, resume: bool = True):
    """
    Run MLLM inference on all images in a subset.
    Saves per-image predictions to JSONL.
    """
    cfg = SUBSET_CONFIG[subset]
    classes = cfg["classes"]
    multilabel = cfg["multilabel"]
    data_dir = None
    if SCB_DATA_DIR:
        candidate = Path(SCB_DATA_DIR) / SUBSET_KEY_MAP[subset]
        if candidate.exists():
            data_dir = str(candidate)

    output_dir.mkdir(parents=True, exist_ok=True)
    sanitized_model = model.replace(":", "_").replace("/", "_")
    output_path = output_dir / f"mllm_{sanitized_model}_{subset}_preds.jsonl"

    # Load existing predictions for resume
    done = load_checkpoint(output_path) if resume else {}
    if done:
        print(f"[MLLM_v2] Resuming: {len(done)} images already done")

    # Load dataset
    loader, dataset = build_dataloader(
        subset=subset, split="validation",
        batch_size=1, num_workers=0, input_size=224,
        data_dir=data_dir,
    )

    total = len(dataset)
    prompt = build_prompt(classes, multilabel)

    with open(output_path, "a" if resume else "w") as f:
        for idx in range(total):
            global interrupted
            if interrupted:
                print(f"\n[MLLM_v2] Interrupted at image {idx}/{total}")
                break

            if idx in done:
                continue

            sample = dataset[idx]
            img_tensor = sample["image"]
            raw_label = sample["label"]
            if hasattr(raw_label, "tolist"):
                gt_vector = raw_label.tolist() if hasattr(raw_label, "tolist") else raw_label
            else:
                gt_vector = raw_label
            if isinstance(gt_vector, (int, float, np.integer, np.floating)):
                gt_vector = [0] * len(classes)
                gt_vector[int(raw_label)] = 1

            b64 = encode_image(img_tensor)
            payload = {
                "model": model,
                "prompt": prompt,
                "images": [b64],
                "stream": False,
                "options": {"temperature": 0.0},
            }

            t0 = time.time()
            try:
                resp = requests.post(OLLAMA_URL, json=payload, timeout=180)
                resp.raise_for_status()
                response_text = resp.json().get("response", "")
            except requests.exceptions.Timeout:
                response_text = ""
                print(f"  [{idx}/{total}] TIMEOUT")
            except Exception as e:
                response_text = ""
                print(f"  [{idx}/{total}] ERROR: {e}")

            dur = time.time() - t0
            pred_classes = parse_response(response_text, classes, multilabel)

            record = {
                "index": idx,
                "gt_labels": [classes[i] for i, v in enumerate(gt_vector) if v > 0],
                "gt_vector": [int(v) for v in gt_vector],
                "pred_classes": pred_classes,
                "pred_classes": pred_classes,
                "raw_response": response_text[:500],
                "duration_s": round(dur, 1),
            }
            f.write(json.dumps(record) + "\n")
            f.flush()

            if (idx + 1) % 10 == 0 or idx == 0:
                eta = (dur * (total - idx - 1)) / 60
                print(f"  [{idx+1}/{total}] {dur:.0f}s  pred={pred_classes}  "
                      f"gt={record['gt_labels']}  ETA={eta:.0f}min")

    print(f"[MLLM_v2] Done: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="MLLM v2 per-image inference")
    parser.add_argument("--model", type=str, required=True,
                        help="Ollama model name (e.g. qwen3.5:27b)")
    parser.add_argument("--subset", type=str, default="all",
                        choices=SUBSETS + ["all"],
                        help="Subset to run (default: all)")
    parser.add_argument("--output_dir", type=str,
                        default=str(REPO_ROOT / "results" / "per_image"),
                        help="Output directory for prediction files")
    parser.add_argument("--no_resume", action="store_true",
                        help="Do not resume from existing predictions")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    subsets_to_run = SUBSETS if args.subset == "all" else [args.subset]

    for subset in subsets_to_run:
        print(f"\n[MLLM_v2] Model={args.model}  Subset={subset}")
        run_inference(args.model, subset, output_dir, resume=not args.no_resume)

    return 0


if __name__ == "__main__":
    exit(main())
