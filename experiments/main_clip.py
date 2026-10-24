"""
main_clip.py (v2)
CTRL+C-safe CLIP + CAPE baseline experiment runner.
Supports multiple CAPE sets (A/B/C/LLM) and all prompt strategies.
4× V100 parallel-ready via SINGLE_MODEL env var + --cape_sets.

Usage:
  # Single GPU
  python experiments/main_clip.py --cape_sets A B C

  # Parallel: set env SINGLE_MODEL=openai + CUDA_VISIBLE_DEVICES=0
"""

import os, json, argparse, sys, signal
from pathlib import Path
from typing import Dict, List
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.scb_dataset import build_dataloader, SUBSET_CONFIG, SUBSET_LOCAL_DIRS
from models.clip_zoo import CLIPModel, build_cape_text_features, batch_inference
from prompts.cape_prompts import get_all_class_prompts, ALL_STRATEGIES
from evaluation.metrics import full_evaluation

GPU_ASSIGNMENT = {
    "openai": "cuda:0", "siglip2": "cuda:1", "laion": "cuda:2",
    "eva02": "cuda:3", "dfn": "cuda:0",
}
MODEL_KEYS = ["openai", "laion", "siglip2", "eva02", "dfn"]

env_single = os.environ.get("SINGLE_MODEL")
if env_single:
    MODEL_KEYS = [env_single]

interrupted = False
def _handle_sigint(sig, frame):
    global interrupted
    interrupted = True
    print("\n[main_clip] SIGINT received, finishing current task...")
signal.signal(signal.SIGINT, _handle_sigint)


def run_single(model_key, subset, strategy, cape_set="A", device=None):
    device = device or GPU_ASSIGNMENT.get(model_key, "cuda:0")
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        device = "cuda:0"
    cfg = SUBSET_CONFIG[subset]
    classes = cfg["classes"]
    multilabel = cfg["multilabel"]
    num_classes = cfg["num_classes"]

    model = CLIPModel(model_key, device=device)
    data_dir = SUBSET_LOCAL_DIRS.get(subset)
    loader, dataset = build_dataloader(
        subset=subset, split="validation",
        batch_size=64, num_workers=0, input_size=model.input_size,
        data_dir=str(data_dir) if data_dir else None,
    )

    class_prompts = get_all_class_prompts(classes, strategy, cape_set=cape_set)
    text_feats = build_cape_text_features(model, class_prompts, classes)
    logits, labels = batch_inference(model, loader, text_feats, multilabel)

    if multilabel:
        lb_matrix = labels.astype(float)
    else:
        lb_matrix = np.eye(num_classes)[labels.astype(int)]

    eval_results = full_evaluation(
        logits, lb_matrix, num_classes, multilabel, classes,
        run_bootstrap=False,
    )
    del model
    torch.cuda.empty_cache()
    return eval_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="./results/baseline")
    parser.add_argument("--cape_sets", nargs="+", default=["A", "B", "C"],
                        choices=["A", "B", "C", "LLM"],
                        help="CAPE prompt sets to evaluate")
    parser.add_argument("--strategies", nargs="+", default=None,
                        choices=[s for s in ALL_STRATEGIES if s != "cape"] + ["cape_set"],
                        help="Prompt strategies (default: cape_set evaluates all CAPE sets)")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.is_absolute():
        results_dir = REPO_ROOT / results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    subsets = ["teacher_behavior", "handrise_readwrite", "bow_turnhead"]
    strategies = args.strategies or ["cape_set"]
    use_cape_set = "cape_set" in strategies

    all_results = {}
    for subset in subsets:
        all_results[subset] = {"class_names": SUBSET_CONFIG[subset]["classes"]}
        for model_key in MODEL_KEYS:
            all_results[subset][model_key] = {}

            for strategy in strategies:
                if strategy in ("cape_set", "cape"):
                    continue
                if interrupted:
                    break
                print(f"  {model_key} / {subset} / {strategy}")
                result = run_single(model_key, subset, strategy)
                all_results[subset][model_key][strategy] = {
                    "hit_at_1": result["hit_at_1"],
                    "hit_at_3": result["hit_at_3"],
                    "macro_f1": result["single_label"]["macro_f1"],
                    "confusion_matrix": result["single_label"]["confusion_matrix"],
                    "per_class_recall": result["single_label"]["per_class_recall"],
                    "per_class_f1": result["single_label"]["per_class_f1"],
                }
                print(f"    Hit@1={result['hit_at_1']*100:.2f}%  "
                      f"Macro-F1={result['single_label']['macro_f1']*100:.2f}%")

            if use_cape_set:
                for cape_set in args.cape_sets:
                    if interrupted:
                        break
                    strategy_key = f"cape_set_{cape_set}"
                    print(f"  {model_key} / {subset} / CAPE Set-{cape_set}")
                    try:
                        result = run_single(model_key, subset, "cape", cape_set=cape_set)
                    except Exception as e:
                        print(f"    SKIP: {e}")
                        continue
                    all_results[subset][model_key][strategy_key] = {
                        "hit_at_1": result["hit_at_1"],
                        "hit_at_3": result["hit_at_3"],
                        "macro_f1": result["single_label"]["macro_f1"],
                        "confusion_matrix": result["single_label"]["confusion_matrix"],
                        "per_class_recall": result["single_label"]["per_class_recall"],
                        "per_class_f1": result["single_label"]["per_class_f1"],
                    }
                    print(f"    Hit@1={result['hit_at_1']*100:.2f}%  "
                          f"Macro-F1={result['single_label']['macro_f1']*100:.2f}%")

        if interrupted:
            print("[main_clip] Interrupted, saving partial results...")
            break

    out_path = results_dir / "baseline_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n{'✓' if not interrupted else '⚠ Partial'} Baseline complete: {out_path}")


if __name__ == "__main__":
    main()
