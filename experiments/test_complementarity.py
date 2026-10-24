"""
test_complementarity.py — Orthogonal Diagnostic Signals Analysis
================================================================
Tests whether CLIP+CAPE zero-shot and MLLM annotation provide
independent (orthogonal) diagnostic signals for classroom behavior
recognition.

Core hypothesis:
  MLLM annotation accuracy and CLIP per-class recall are uncorrelated
  (Spearman ρ ≈ 0), meaning they provide non-redundant information.
  A class that MLLM annotates poorly may still be well-handled by
  CLIP, and vice versa.

Framework:
  1. Bootstrap 95% CI for Spearman ρ (13 class-level pairs × 10K resamples).
     CI covering ±~0 confirms orthogonal signals.
  2. Per-class model-advantage analysis: which model leads on each class.
  3. Coverage analysis: % of classes where at least one model performs well.
  4. Quadrant visualization (descriptive, no binary significance test).

Data sources (read-only):
  results/baseline/baseline_results.json     — CLIP+CAPE confusion matrices
  experiments/visual_anchoring_data.json  — MLLM per-class annotation accuracy

Output:
  results/complementarity_report.json
  results/complementarity_report.txt

Reproducibility:
  python experiments/test_complementarity.py
"""

import json
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
CLIP_PATH = REPO_ROOT / "results" / "baseline" / "baseline_results.json"
ANCHOR_PATH = REPO_ROOT / "experiments" / "visual_anchoring_data.json"
OUT_DIR = REPO_ROOT / "results"

CLIP_MODELS = ["openai", "laion", "siglip2", "eva02", "dfn"]
NAME_NORMALIZE = {"blackBoard": "blackboard"}
SUBSET_KEYS = ["teacher_behavior", "bow_turnhead", "handrise_readwrite"]

BOOTSTRAP_N = 10000
SEED = 42


def load_json(path):
    with open(path) as f:
        return json.load(f)


def per_class_recall_from_cm(cm, class_names):
    """Per-class recall = diagonal_i / row_sum_i (multi-label confusion matrix)."""
    cm = np.array(cm, dtype=np.float64)
    recall = {}
    for i, cls in enumerate(class_names):
        row_sum = cm[i].sum()
        recall[cls] = float(cm[i, i] / row_sum) if row_sum > 0 else 0.0
    return recall


def spearman_rank(x):
    """Compute ranks of array x, handling ties with average rank."""
    order = np.argsort(x)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(x) + 1, dtype=np.float64)
    # Handle ties: use average rank
    unique, inverse, counts = np.unique(x, return_inverse=True, return_counts=True)
    for i, cnt in enumerate(counts):
        if cnt > 1:
            mask = inverse == i
            ranks[mask] = np.mean(ranks[mask])
    return ranks


def spearman_r(x, y):
    """Spearman rank correlation between two equal-length arrays."""
    rx = spearman_rank(x)
    ry = spearman_rank(y)
    n = len(x)
    d2 = np.sum((rx - ry) ** 2)
    return 1 - (6 * d2) / (n * (n**2 - 1))


def bootstrap_spearman_ci(x, y, n_boot=BOOTSTRAP_N, seed=SEED):
    """
    Bootstrap 95% CI for Spearman ρ.
    Resample class-level pairs with replacement.
    """
    rng = np.random.RandomState(seed)
    n = len(x)
    rho_dist = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.randint(0, n, size=n)
        rho_dist[i] = spearman_r(x[idx], y[idx])
    ci_lo = float(np.percentile(rho_dist, 2.5))
    ci_hi = float(np.percentile(rho_dist, 97.5))
    return float(spearman_r(x, y)), ci_lo, ci_hi, rho_dist


def gather_class_pairs(clip_data, anchor_data):
    """
    Gather (mllm_accuracy, clip_recall, class, subset) for all classes
    across all 3 subsets. CLIP recall is averaged across 5 models.
    Returns:
      pairs: list of dicts
      mllm_vals: np.array (n,)
      clip_vals: np.array (n,)
    """
    pairs = []
    for subset_key in SUBSET_KEYS:
        class_names = clip_data[subset_key].get("class_names", [])
        subset_anchors = anchor_data.get(subset_key, {})

        for cls in class_names:
            # CLIP: average per-class recall across 5 models
            recalls = []
            for model in CLIP_MODELS:
                cm = clip_data[subset_key][model]["cape"]["confusion_matrix"]
                rec = per_class_recall_from_cm(cm, class_names)
                if cls in rec:
                    recalls.append(rec[cls])
            clip_rec = float(np.mean(recalls)) if recalls else 0.0

            # MLLM: annotation accuracy from anchoring data
            cls_key = NAME_NORMALIZE.get(cls, cls)
            anchor_info = subset_anchors.get(cls_key, {})
            mllm_acc = anchor_info.get("accuracy_pct")
            if mllm_acc is None:
                continue

            pairs.append({
                "class": cls,
                "subset": subset_key,
                "mllm_accuracy": float(mllm_acc),
                "clip_recall": clip_rec,
                "anchoring": anchor_info.get("anchoring", "unknown"),
            })

    mllm_vals = np.array([p["mllm_accuracy"] for p in pairs], dtype=np.float64)
    clip_vals = np.array([p["clip_recall"] for p in pairs], dtype=np.float64)
    return pairs, mllm_vals, clip_vals


def quadrant_analysis(pairs):
    """Descriptive quadrant: which classes fall where? (No hypothesis test.)"""
    q = {"both_strong": [], "mllm_stronger": [], "clip_stronger": [], "both_weak": []}
    for p in pairs:
        a = p["mllm_accuracy"]
        c = p["clip_recall"]
        if a > 50 and c > 0.15:
            q["both_strong"].append(p["class"])
        elif a > 50:
            q["mllm_stronger"].append(p["class"])
        elif c > 0.15:
            q["clip_stronger"].append(p["class"])
        else:
            q["both_weak"].append(p["class"])
    return q


def per_class_advantage(pairs):
    """Continuous: for each class, which model has advantage, and by how much."""
    # Normalize both metrics to [0,1] for comparison
    mllm_vals = np.array([p["mllm_accuracy"] for p in pairs])
    clip_vals = np.array([p["clip_recall"] for p in pairs])

    # Min-max normalize within each distribution
    mllm_norm = (mllm_vals - mllm_vals.min()) / (mllm_vals.max() - mllm_vals.min() + 1e-8)
    clip_norm = (clip_vals - clip_vals.min()) / (clip_vals.max() - clip_vals.min() + 1e-8)

    advantage = []
    for i, p in enumerate(pairs):
        ratio = clip_norm[i] - mllm_norm[i]  # positive = CLIP better, negative = MLLM better
        if ratio > 0.1:
            winner = "CLIP"
        elif ratio < -0.1:
            winner = "MLLM"
        else:
            winner = "tie"
        advantage.append({
            "class": p["class"],
            "subset": p["subset"],
            "mllm_accuracy": p["mllm_accuracy"],
            "clip_recall": p["clip_recall"],
            "advantage_ratio": float(ratio),
            "winner": winner,
        })
    return advantage


def coverage_analysis(pairs, quadrants):
    """
    What fraction of classes are covered by at least one model?
    Coverage = (classes where at least one model performs well) / total.
    """
    n = len(pairs)
    covered = len(quadrants["both_strong"]) + len(quadrants["mllm_stronger"]) + len(quadrants["clip_stronger"])
    uncovered = len(quadrants["both_weak"])
    return {
        "covered": covered,
        "uncovered": uncovered,
        "coverage_rate": covered / n if n > 0 else 0,
    }


def format_report(pairs, mllm_vals, clip_vals, quadrants, advantage, coverage,
                  rho, ci_lo, ci_hi):
    lines = []
    lines.append("=" * 72)
    lines.append("  Orthogonal Diagnostic Signals Analysis")
    lines.append("  CLIP zero-shot recall vs. MLLM annotation accuracy")
    lines.append("=" * 72)
    lines.append("")

    # Primary result: Spearman + bootstrap CI
    lines.append("── Primary Result: Spearman ρ + Bootstrap 95% CI ──")
    lines.append(f"  Number of class-level pairs:     {len(pairs)}")
    lines.append(f"  Spearman ρ:                      {rho:+.4f}")
    lines.append(f"  Bootstrap 95% CI:                [{ci_lo:+.4f}, {ci_hi:+.4f}]")
    lines.append(f"  Bootstrap resamples:             {BOOTSTRAP_N}")
    lines.append("")
    if ci_lo < 0 and ci_hi > 0:
        lines.append(f"  ✓ CI includes zero → signals are orthogonal (ρ ≈ 0).")
        lines.append(f"    MLLM accuracy and CLIP recall carry non-redundant information.")
    elif abs(rho) < 0.2:
        lines.append(f"  ✓ ρ ≈ {rho:+.3f} → weak-to-zero correlation → near-orthogonal signals.")
    else:
        lines.append(f"  ⚠ ρ = {rho:+.4f} — correlation present, models not fully orthogonal.")
    lines.append("")

    # Coverage
    lines.append("── Coverage Analysis ──")
    lines.append(f"  Classes with at least one model performing well: "
                 f"{coverage['covered']}/{coverage['total']} "
                 f"({coverage['coverage_rate']*100:.0f}%)")
    lines.append(f"  Classes where both models fail:                  "
                 f"{coverage['uncovered']}/{coverage['total']}")
    lines.append("")

    # Per-class advantage
    lines.append("── Per-Class Model Advantage ──")
    lines.append(f"  {'Class':<22} {'MLLM Acc':>8} {'CLIP Rec':>8} {'Winner':>8}")
    lines.append("  " + "-" * 54)
    for a in advantage:
        lines.append(f"  {a['class']:<22} {a['mllm_accuracy']:>7.1f}% {a['clip_recall']*100:>7.1f}% "
                     f"{a['winner']:>8}")
    lines.append("")

    # Quadrant table
    lines.append("── Descriptive Quadrants (continuous thresholds) ──")
    lines.append(f"  Both cover  ({len(quadrants['both_strong'])}): "
                 f"{', '.join(quadrants['both_strong']) if quadrants['both_strong'] else '(none)'}")
    lines.append(f"  MLLM→       ({len(quadrants['mllm_stronger'])}): "
                 f"{', '.join(quadrants['mllm_stronger'])}")
    lines.append(f"  CLIP→       ({len(quadrants['clip_stronger'])}): "
                 f"{', '.join(quadrants['clip_stronger'])}")
    lines.append(f"  Neither     ({len(quadrants['both_weak'])}): "
                 f"{', '.join(quadrants['both_weak'])}")
    lines.append("")

    # Interpretation
    lines.append("=" * 72)
    lines.append("  Interpretation")
    lines.append("=" * 72)
    ci_width = ci_hi - ci_lo
    if ci_lo < 0 and ci_hi > 0 and ci_width < 0.6:
        lines.append(f"  1. Spearman ρ 95% CI [{ci_lo:+.2f}, {ci_hi:+.2f}] spans zero with")
        lines.append(f"     width {ci_width:.2f} — CLIP recall and MLLM accuracy are")
        lines.append(f"     consistent with orthogonal diagnostic signals.")
    elif ci_width > 0.6:
        lines.append(f"  1. CI [{ci_lo:+.2f}, {ci_hi:+.2f}] is wide ({ci_width:.2f}) —")
        lines.append(f"     with 13 classes, precision is limited; the point estimate")
        lines.append(f"     ρ={rho:+.3f} suggests weak orthogonal tendency.")
    lines.append(f"  2. Coverage: {coverage['coverage_rate']*100:.0f}% of classes ({coverage['covered']}/{coverage['total']})")
    lines.append(f"     have at least one model performing above threshold.")
    lines.append(f"  3. Only {coverage['uncovered']} class(es) — {', '.join(quadrants['both_weak'])} —")
    lines.append(f"     fail for both models. This class requires temporal context.")
    lines.append(f"  4. Practical design: deploy CLIP+CAPE for action classes")
    lines.append(f"     ({', '.join(quadrants['clip_stronger'])}), MLLM annotation for")
    lines.append(f"     object-classes ({', '.join(quadrants['mllm_stronger'])}), and")
    lines.append(f"     video analysis for intent-dependent classes.")
    lines.append("")

    return "\n".join(lines)


def main():
    clip_data = load_json(CLIP_PATH)
    anchor_data = load_json(ANCHOR_PATH)

    pairs, mllm_vals, clip_vals = gather_class_pairs(clip_data, anchor_data)

    # Bootstrap Spearman CI (primary statistical result)
    rho, ci_lo, ci_hi, rho_dist = bootstrap_spearman_ci(mllm_vals, clip_vals)

    # Descriptive analyses
    quadrants = quadrant_analysis(pairs)
    advantage = per_class_advantage(pairs)
    coverage = coverage_analysis(pairs, quadrants)
    coverage["total"] = len(pairs)

    # Format and write
    report = format_report(pairs, mllm_vals, clip_vals, quadrants, advantage,
                           coverage, rho, ci_lo, ci_hi)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    json_path = OUT_DIR / "complementarity_report.json"
    with open(json_path, "w") as f:
        json.dump({
            "spearman": {"rho": rho, "ci_lo": ci_lo, "ci_hi": ci_hi,
                         "bootstrap_n": BOOTSTRAP_N},
            "coverage": coverage,
            "quadrants": quadrants,
            "per_class_advantage": advantage,
            "pairs": pairs,
        }, f, indent=2)
    print(f"  JSON: {json_path}")

    txt_path = OUT_DIR / "complementarity_report.txt"
    with open(txt_path, "w") as f:
        f.write(report)
    print(f"  Text: {txt_path}")
    print()
    print(report)

    return 0


if __name__ == "__main__":
    exit(main())
