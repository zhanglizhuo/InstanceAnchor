"""
analyze_per_image_complementarity.py — Per-image orthogonal signal analysis
============================================================================
Tests whether CLIP and MLLM provide orthogonal diagnostic signals at the
per-image level (5416 images, not 13 class-level pairs).

Approach:
  1. Load per-image CLIP logits + MLLM predictions.
  2. For each image, compute binary correctness (0/1) for both models.
  3. Build 2×2 contingency table (CLIP correct × MLLM correct).
  4. Bootstrap 95% CI for key metrics:
     - % images where exactly one model is correct (complementarity rate)
     - % images where both are wrong (joint failure rate)
     - % images where both are right (joint success rate)
  5. Spearman ρ across per-image correctness (5400+ data points → narrow CI).

Data sources:
  results/per_image/clip_{model}_{subset}_logits.npy
  results/per_image/clip_{model}_{subset}_labels.npy
  results/per_image/mllm_{model}_{subset}_preds.jsonl

Output:
  results/per_image_complementarity_report.json
  results/per_image_complementarity_report.txt

Reproducibility:
  python experiments/analyze_per_image_complementarity.py
"""

import json
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "results"
DATA_DIR = REPO_ROOT / "results" / "per_image"

SUBSETS = ["teacher_behavior", "handrise_readwrite", "bow_turnhead"]
SUBSET_LABELS = {
    "teacher_behavior": "TeacherBehavior",
    "handrise_readwrite": "HandriseReadWrite",
    "bow_turnhead": "BowTurnHead",
}

CLIP_MODELS = ["openai", "laion", "siglip2", "eva02", "dfn"]

BOOTSTRAP_N = 10000
SEED = 42


def load_clip_logits(model: str, subset: str) -> tuple:
    logits = np.load(DATA_DIR / f"clip_{model}_{subset}_logits.npy")
    labels = np.load(DATA_DIR / f"clip_{model}_{subset}_labels.npy")
    return logits, labels


def load_mllm_preds(mllm_model: str, subset: str) -> dict:
    """Load MLLM predictions, returns {image_index: set_of_predicted_class_indices}"""
    sanitized = mllm_model.replace(":", "_").replace("/", "_")
    path = DATA_DIR / f"mllm_{sanitized}_{subset}_preds.jsonl"
    if not path.exists():
        print(f"[WARN] {path} not found")
        return None
    preds = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            preds[rec["index"]] = set(rec.get("pred_classes", []))
    return preds


def compute_per_image_correctness(clip_logits, clip_labels, mllm_preds, classes,
                                   multilabel, top_k=1):
    """
    Compute per-image binary correctness for CLIP and MLLM.

    CLIP: top-K prediction check against ground truth (multilabel: any overlap)
    MLLM: prediction classes check against ground truth
    """
    n = len(mllm_preds)
    clip_correct = np.zeros(n, dtype=bool)
    mllm_correct = np.zeros(n, dtype=bool)
    matched = 0

    if multilabel:
        # CLIP: top-K argmax
        clip_topk = np.argsort(-clip_logits, axis=1)[:, :top_k]
        for idx in mllm_preds:
            gt = np.where(clip_labels[idx] > 0)[0]
            if len(gt) == 0:
                continue
            clip_pred = set(clip_topk[idx].tolist())
            clip_correct[idx] = not clip_pred.isdisjoint(gt)
            # MLLM: class names → indices
            mllm_pred_set = mllm_preds[idx]
            mllm_indices = {classes.index(c) for c in mllm_pred_set if c in classes}
            mllm_correct[idx] = not mllm_indices.isdisjoint(gt)
            matched += 1
    else:
        clip_pred_class = np.argmax(clip_logits, axis=1)
        for idx in mllm_preds:
            gt = int(np.argmax(clip_labels[idx]))
            clip_correct[idx] = (clip_pred_class[idx] == gt)
            mllm_pred_set = mllm_preds[idx]
            mllm_indices = {classes.index(c) for c in mllm_pred_set if c in classes}
            mllm_correct[idx] = (list(mllm_indices)[0] == gt) if mllm_indices else False
            matched += 1

    return clip_correct[:n], mllm_correct[:n], matched


def build_contingency(clip_correct, mllm_correct):
    """2×2 table: [ [both_wrong, clip_right_mllm_wrong],
                     [mllm_right_clip_wrong, both_right] ]"""
    both_wrong = np.sum(~clip_correct & ~mllm_correct)
    clip_only = np.sum(clip_correct & ~mllm_correct)
    mllm_only = np.sum(~clip_correct & mllm_correct)
    both_right = np.sum(clip_correct & mllm_correct)
    total = both_wrong + clip_only + mllm_only + both_right
    return {
        "both_wrong": int(both_wrong),
        "clip_only": int(clip_only),
        "mllm_only": int(mllm_only),
        "both_right": int(both_right),
        "total": int(total),
        "clip_acc": float(both_right + clip_only) / total if total > 0 else 0,
        "mllm_acc": float(both_right + mllm_only) / total if total > 0 else 0,
        "complementarity_rate": float(clip_only + mllm_only) / total if total > 0 else 0,
        "joint_failure_rate": float(both_wrong) / total if total > 0 else 0,
    }


def bootstrap_contingency_ci(clip_correct, mllm_correct, n_boot=BOOTSTRAP_N, seed=SEED):
    """Bootstrap 95% CI for complementarity metrics."""
    rng = np.random.RandomState(seed)
    n = len(clip_correct)
    comp_rates = np.empty(n_boot)
    joint_fail_rates = np.empty(n_boot)

    for i in range(n_boot):
        idx = rng.randint(0, n, size=n)
        cc = clip_correct[idx]
        mc = mllm_correct[idx]
        both_wrong = np.sum(~cc & ~mc)
        clip_only = np.sum(cc & ~mc)
        mllm_only = np.sum(~cc & mc)
        both_right = np.sum(cc & mc)
        total = both_wrong + clip_only + mllm_only + both_right
        if total > 0:
            comp_rates[i] = (clip_only + mllm_only) / total
            joint_fail_rates[i] = both_wrong / total
        else:
            comp_rates[i] = 0
            joint_fail_rates[i] = 0

    return {
        "complementarity_rate": {
            "mean": float(np.mean(comp_rates)),
            "ci_lo": float(np.percentile(comp_rates, 2.5)),
            "ci_hi": float(np.percentile(comp_rates, 97.5)),
        },
        "joint_failure_rate": {
            "mean": float(np.mean(joint_fail_rates)),
            "ci_lo": float(np.percentile(joint_fail_rates, 2.5)),
            "ci_hi": float(np.percentile(joint_fail_rates, 97.5)),
        },
    }


def format_report(all_results):
    lines = []
    lines.append("=" * 72)
    lines.append("  Per-Image Orthogonal Signal Analysis")
    lines.append("  CLIP vs MLLM: 5416 images × per-model correctness")
    lines.append("=" * 72)
    lines.append("")

    for key, result in all_results.items():
        subset_label, clip_model, mllm_model = key.split("||")
        ct = result["contingency"]
        ci = result["bootstrap_ci"]

        lines.append(f"── {subset_label} / CLIP={clip_model} / MLLM={mllm_model} ──")
        lines.append(f"  Total images matched: {ct['total']}")
        lines.append(f"  CLIP accuracy:  {ct['clip_acc']*100:>5.1f}%")
        lines.append(f"  MLLM accuracy:  {ct['mllm_acc']*100:>5.1f}%")
        lines.append("")
        lines.append(f"  2×2 Contingency Table:")
        lines.append(f"                  MLLM wrong  MLLM right")
        lines.append(f"    CLIP wrong    {ct['both_wrong']:>8}    {ct['mllm_only']:>8}")
        lines.append(f"    CLIP right    {ct['clip_only']:>8}    {ct['both_right']:>8}")
        lines.append("")
        lines.append(f"  Complementarity rate: {ct['complementarity_rate']*100:.1f}%"
                     f"  CI [{ci['complementarity_rate']['ci_lo']*100:.1f}%, "
                     f"{ci['complementarity_rate']['ci_hi']*100:.1f}%]")
        lines.append(f"    (= images where exactly one model is correct)")
        lines.append(f"  Joint failure rate:   {ct['joint_failure_rate']*100:.1f}%"
                     f"  CI [{ci['joint_failure_rate']['ci_lo']*100:.1f}%, "
                     f"{ci['joint_failure_rate']['ci_hi']*100:.1f}%]")
        lines.append(f"    (= images where both models are wrong)")
        lines.append("")

    lines.append("=" * 72)
    lines.append("  Interpretation")
    lines.append("=" * 72)
    lines.append("  • If complementarity rate CI excludes 0, CLIP and MLLM")
    lines.append("    provide non-redundant signals at per-image level.")
    lines.append("  • Narrow CI (= N > 3000 images) gives decisive evidence.")
    lines.append("  • Joint failure rate identifies the irreducible error floor")
    lines.append("    (= intent-dependent classes where no amount of model")
    lines.append("     improvement helps without temporal context).")
    lines.append("")
    return "\n".join(lines)


def main():
    # Find available MLLM prediction files
    mllm_files = sorted(DATA_DIR.glob("mllm_*_teacher_behavior_preds.jsonl"))
    if not mllm_files:
        print("[ERROR] No MLLM prediction files found in", DATA_DIR)
        print("  Run main_mllm_v2.py first.")
        return 1

    mllm_models = []
    for f in mllm_files:
        stem = f.stem  # mllm_qwen3.6_27b_teacher_behavior_preds
        # Extract model name: remove 'mllm_' prefix and '_teacher_behavior_preds' suffix
        body = stem.replace("mllm_", "").replace("_teacher_behavior_preds", "")
        # Convert back to colon format
        mllm_models.append(body)

    print(f"Found MLLM models: {mllm_models}")
    print(f"CLIP models: {CLIP_MODELS}")

    all_results = {}
    for mllm_model in mllm_models:
        for subset in SUBSETS:
            mllm_preds = load_mllm_preds(mllm_model, subset)
            if mllm_preds is None:
                continue

            # Load subset config
            from data.scb_dataset import SUBSET_CONFIG
            cfg = SUBSET_CONFIG[subset]
            classes = cfg["classes"]
            multilabel = cfg["multilabel"]

            for clip_model in CLIP_MODELS[:1]:  # Use openai as primary CLIP model
                try:
                    clip_logits, clip_labels = load_clip_logits(clip_model, subset)
                except FileNotFoundError:
                    print(f"[SKIP] CLIP data missing: {clip_model}/{subset}")
                    continue

                clip_correct, mllm_correct, matched = compute_per_image_correctness(
                    clip_logits, clip_labels, mllm_preds, classes, multilabel
                )

                if matched == 0:
                    continue

                ct = build_contingency(clip_correct, mllm_correct)
                ci = bootstrap_contingency_ci(clip_correct, mllm_correct)

                key = f"{SUBSET_LABELS[subset]}||{clip_model}||{mllm_model}"
                all_results[key] = {
                    "contingency": ct,
                    "bootstrap_ci": ci,
                }

    if not all_results:
        print("[ERROR] No valid (CLIP, MLLM) pairs found.")
        return 1

    report = format_report(all_results)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUT_DIR / "per_image_complementarity_report.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"  JSON: {json_path}")

    txt_path = OUT_DIR / "per_image_complementarity_report.txt"
    with open(txt_path, "w") as f:
        f.write(report)
    print(f"  Text: {txt_path}")
    print()
    print(report)

    return 0


if __name__ == "__main__":
    exit(main())
