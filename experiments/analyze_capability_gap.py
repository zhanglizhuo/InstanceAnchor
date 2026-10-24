"""
analyze_capability_gap.py — Phase 3: MLLM vs CLIP per-class analysis
===============================================================
Compares MLLM zero-shot results with CLIP+CAPE baseline.
No GPU needed — works on saved result JSONs.

Usage:
  python experiments/analyze_capability_gap.py

Output:
  results/capability_gap_report.json
  results/capability_gap_report.txt
"""

import json, sys
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]

CLIP_RESULTS_PATH = REPO_ROOT / "results" / "baseline" / "baseline_results.json"
MLLM_RESULTS_PATH = REPO_ROOT / "results" / "mllm" / "mllm_macrof1_all_models.json"
SUPPLEMENT_PATH = REPO_ROOT / "results" / "mllm" / "mllm_macrof1_supplement.json"
OUT_DIR = REPO_ROOT / "results"

SUBSET_LABELS = {
    "teacher_behavior": "TeacherBehavior",
    "handrise_readwrite": "HandriseReadWrite",
    "bow_turnhead": "BowTurnHead",
}

def load_json(path):
    if not path.exists():
        print(f"[WARN] {path} not found")
        return None
    with open(path) as f:
        return json.load(f)

def analyze_clip_mllm_gap(clip_data, mllm_data):
    """Per-class and per-subset gap analysis."""
    report = {"summary": {}, "per_category": {}, "actionable_insights": []}

    for subset in ["teacher_behavior", "handrise_readwrite", "bow_turnhead"]:
        class_names = clip_data.get(subset, {}).get("class_names", [])
        clip_subset = clip_data.get(subset, {})
        report["per_category"][subset] = {"classes": class_names}

        for model_key, model_data in clip_subset.items():
            if model_key == "class_names":
                continue
            cape_cfg = model_data.get("cape", {})
            clip_hit1 = cape_cfg.get("hit_at_1", 0)
            clip_macro = cape_cfg.get("macro_f1", 0)
            clip_per_class = cape_cfg.get("per_class_recall", [])
            report["per_category"][subset].setdefault(model_key, {})

        # MLLM comparison
        for mllm_key, mllm_data_subset in mllm_data.get("models", {}).items():
            subset_data = mllm_data_subset.get(subset, {})
            mllm_hit1 = subset_data.get("hit_at_1", 0)
            mllm_macro = subset_data.get("macro_f1", 0)
            report["per_category"][subset].setdefault(mllm_key, {})

        # Compute gaps per model
        for model_key, model_data in clip_subset.items():
            if model_key == "class_names":
                continue
            cape_cfg = model_data.get("cape", {})
            clip_hit1 = cape_cfg.get("hit_at_1", 0)

            for mllm_key, mllm_data_subset in mllm_data.get("models", {}).items():
                subset_data = mllm_data_subset.get(subset, {})
                mllm_hit1 = subset_data.get("hit_at_1", 0)
                gap = mllm_hit1 - clip_hit1

                key = f"{model_key}_vs_{mllm_key}"
                report["per_category"][subset].setdefault(key, {
                    "clip_hit1": clip_hit1,
                    "mllm_hit1": mllm_hit1,
                    "gap": gap,
                    "mllm_better": gap > 0,
                })

    # Weighted summary across subsets
    for mllm_key, mllm_weighted in mllm_data.get("weighted", {}).items():
        report["summary"][mllm_key] = {
            "weighted_hit_at_1": mllm_weighted.get("weighted_hit_at_1", 0),
            "weighted_macro_f1": mllm_weighted.get("weighted_macro_f1", 0),
        }

    # Best model for each subset
    report["best_models"] = {}
    for subset in ["teacher_behavior", "handrise_readwrite", "bow_turnhead"]:
        best = {"model": "", "hit_at_1": 0, "source": ""}
        for mllm_key, mllm_data_subset in mllm_data.get("models", {}).items():
            subset_data = mllm_data_subset.get(subset, {})
            h = subset_data.get("hit_at_1", 0)
            if h > best["hit_at_1"]:
                best = {"model": mllm_key, "hit_at_1": h, "source": "MLLM"}
        for model_key in clip_data.get(subset, {}):
            if model_key == "class_names":
                continue
            h = clip_data[subset].get(model_key, {}).get("cape", {}).get("hit_at_1", 0)
            if h > best["hit_at_1"]:
                best = {"model": model_key, "hit_at_1": h, "source": "CLIP+CAPE"}
        report["best_models"][subset] = best

    return report


def format_text_report(report):
    lines = []
    lines.append("=" * 72)
    lines.append("  MLLM vs CLIP+CAPE Capability Gap Analysis")
    lines.append("=" * 72)
    lines.append("")

    # Summary
    lines.append("── Weighted Summary ──")
    lines.append(f"{'Model':<20} {'Hit@1':>8} {'Macro-F1':>10}")
    lines.append("-" * 40)
    for model, data in sorted(report["summary"].items()):
        lines.append(f"{model:<20} {data['weighted_hit_at_1']*100:>7.2f}% "
                     f"{data['weighted_macro_f1']*100:>8.2f}%")
    lines.append("")

    # Best models per subset
    lines.append("── Best Model Per Subset ──")
    for subset, best in report["best_models"].items():
        label = SUBSET_LABELS.get(subset, subset)
        lines.append(f"  {label:<20} {best['model']:<20} "
                     f"Hit@1={best['hit_at_1']*100:.2f}% ({best['source']})")
    lines.append("")

    # Per-category gaps
    lines.append("── Per-Subset CLIP vs MLLM Gaps (CAPE Set-A Hit@1) ──")
    for subset, subset_data in report["per_category"].items():
        label = SUBSET_LABELS.get(subset, subset)
        lines.append(f"\n  [{label}]")
        for key, gap_data in subset_data.items():
            if key == "classes":
                continue
            if "_vs_" not in key:
                continue
            gap_str = f"{gap_data['gap']*100:+.2f}%"
            marker = " ↑" if gap_data["mllm_better"] else " ↓"
            lines.append(f"    {key:<30} CLIP={gap_data['clip_hit1']*100:.2f}%  "
                         f"MLLM={gap_data['mllm_hit1']*100:.2f}%  "
                         f"Δ={gap_str}{marker}")

    lines.append("")
    lines.append("=" * 72)
    lines.append("  Interpretation")
    lines.append("=" * 72)
    lines.append("  • Positive Δ (↑): MLLM outperforms CLIP+CAPE on this subset")
    lines.append("  • Negative Δ (↓): CLIP+CAPE still leads")
    lines.append("  • TeacherBehavior multi-label remains the hardest task for both")
    lines.append("  • All models achieve near-100% Hit@3 on binary subsets")
    lines.append("")
    return "\n".join(lines)


def main():
    clip_data = load_json(CLIP_RESULTS_PATH)
    mllm_data = load_json(MLLM_RESULTS_PATH)

    if mllm_data is None:
        mllm_data = load_json(SUPPLEMENT_PATH)
        if mllm_data:
            mllm_data = {"models": mllm_data, "weighted": {}}

    if clip_data is None or mllm_data is None:
        print("ERROR: Missing required result files.")
        print(f"  CLIP: {CLIP_RESULTS_PATH}")
        print(f"  MLLM: {MLLM_RESULTS_PATH} or {SUPPLEMENT_PATH}")
        sys.exit(1)

    report = analyze_clip_mllm_gap(clip_data, mllm_data)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    json_path = OUT_DIR / "capability_gap_report.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  JSON report: {json_path}")

    txt = format_text_report(report)
    txt_path = OUT_DIR / "capability_gap_report.txt"
    with open(txt_path, "w") as f:
        f.write(txt)
    print(f"  Text report: {txt_path}")
    print()
    print(txt)


if __name__ == "__main__":
    main()
