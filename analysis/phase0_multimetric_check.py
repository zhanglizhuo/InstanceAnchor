"""
analysis/phase0_multimetric_check.py — Cross-metric robustness of the Phase 0 gate
==================================================================================
Motivation: the primary phase0_oracle_analysis.py (sample-F1) gave a per-instance
oracle gain of +3.78pp over the class-level oracle on TeacherBehavior -- below
the 5pp gate. Before deciding to abandon, we check whether this conclusion is
robust across reasonable per-image metrics or is an artifact of sample-F1.

For each metric we compute, on TeacherBehavior (gate subset), for siglip2 + laion:
  - CLIP-only, MLLM-only
  - class-level oracle (route by primary GT class; per-class best model)
  - per-instance oracle (per image, max over the two models)
  - delta = per-instance - class-level  (+ bootstrap 95% CI)

Metrics (all per-image scalars in [0,1]):
  1. sample_f1   : 2|pred&GT| / (|pred|+|GT|)            [symmetric, set/set]
  2. jaccard     : |pred&GT| / |pred|GT|                  [symmetric, set/set]
  3. subset_acc  : 1.0 iff pred == GT exactly             [symmetric, strict]
  4. recall_pi   : |pred&GT| / |GT|                        [symmetric, coverage]
  5. hit1_asym   : CLIP argmax(logit) in GT ; MLLM any-pred in GT
                                                              [asymmetric, deployment]

Reuses loaders/alignment/bootstrap from phase0_oracle_analysis (single source of
truth for data handling).

Output:
  results/phase0_multimetric_report.{json,txt}
  logs/phase0_multimetric_<timestamp>.log

Reproducibility:
  python analysis/phase0_multimetric_check.py
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase0_oracle_analysis import (  # noqa: E402
    SUBSET_CONFIG, DATA_DIR, OUT_DIR, LOG_DIR, THRESHOLD_GRID,
    load_clip, load_mllm, align_and_audit,
    best_clip_threshold, clip_pred_set, per_image_f1,
    class_level_oracle, bootstrap_system_f1, bootstrap_oracle_delta,
    setup_logging,
)


# ── per-image metric functions ──────────────────────────────────────────────
def metric_sample_f1(clip_pred, mllm_pred, gt, clip_logits):
    return per_image_f1(clip_pred, gt), per_image_f1(mllm_pred, gt)


def metric_jaccard(clip_pred, mllm_pred, gt, clip_logits):
    def jac(pred, gt):
        pred = pred.astype(bool); gt = gt.astype(bool)
        inter = (pred & gt).sum(1).astype(np.float32)
        union = (pred | gt).sum(1).astype(np.float32)
        j = np.zeros(pred.shape[0], dtype=np.float32)
        valid = union > 0
        j[valid] = inter[valid] / union[valid]
        j[(~pred.any(1)) & (~gt.any(1))] = 1.0
        return j
    return jac(clip_pred, gt), jac(mllm_pred, gt)


def metric_subset_acc(clip_pred, mllm_pred, gt, clip_logits):
    def sa(pred, gt):
        return (pred == gt).all(axis=1).astype(np.float32)
    return sa(clip_pred, gt), sa(mllm_pred, gt)


def metric_recall_pi(clip_pred, mllm_pred, gt, clip_logits):
    def rec(pred, gt):
        pred = pred.astype(bool); gt = gt.astype(bool)
        inter = (pred & gt).sum(1).astype(np.float32)
        gtsum = gt.sum(1).astype(np.float32)
        r = np.zeros(pred.shape[0], dtype=np.float32)
        valid = gtsum > 0
        r[valid] = inter[valid] / gtsum[valid]
        return r
    return rec(clip_pred, gt), rec(mllm_pred, gt)


def metric_hit1_asym(clip_pred, mllm_pred, gt, clip_logits):
    """CLIP: argmax(logit) in GT.  MLLM: any pred in GT.  Both binary 0/1."""
    top1 = np.argmax(clip_logits, axis=1)
    clip_c = np.array([top1[i] in np.where(gt[i] > 0)[0]
                       for i in range(len(gt))], dtype=np.float32)
    mllm_c = (mllm_pred.astype(bool) & gt.astype(bool)).any(axis=1).astype(np.float32)
    return clip_c, mllm_c


METRICS = [
    ("sample_f1", "sample-F1 (set/set)", metric_sample_f1, "symmetric"),
    ("jaccard", "Jaccard / IoU", metric_jaccard, "symmetric"),
    ("subset_acc", "subset accuracy (exact match)", metric_subset_acc, "symmetric"),
    ("recall_pi", "per-image recall (coverage)", metric_recall_pi, "symmetric"),
    ("hit1_asym", "Hit@1 asymmetric (CLIP argmax / MLLM any)", metric_hit1_asym, "asymmetric"),
]


def evaluate(subset, clip_model, mllm_model, bootstrap_n, seed, log):
    log.info(f"=== {subset} | CLIP={clip_model} | MLLM={mllm_model} ===")
    clip = load_clip(clip_model, subset, log)
    mllm = load_mllm(mllm_model, subset, log)
    if clip is None or mllm is None:
        return None
    clip_logits_full, clip_labels_full = clip
    mllm_indices, mllm_gt_full, mllm_pred_full = mllm
    clip_rows, mllm_rows = align_and_audit(
        clip_labels_full, np.arange(len(clip_logits_full)),
        mllm_indices, mllm_gt_full, log)
    logits = clip_logits_full[clip_rows]
    gt = clip_labels_full[clip_rows]
    mllm_pred = mllm_pred_full[mllm_rows]
    n = len(logits)

    thr, _ = best_clip_threshold(logits, gt, THRESHOLD_GRID)
    clip_pred = clip_pred_set(logits, thr)
    log.info(f"  N={n}  CLIP threshold={thr}")

    out = {"subset": subset, "clip_model": clip_model, "mllm_model": mllm_model,
           "n_images": int(n), "clip_threshold": thr, "metrics": {}}
    for key, label, fn, kind in METRICS:
        c_arr, m_arr = fn(clip_pred, mllm_pred, gt, logits)
        per_inst = np.maximum(c_arr, m_arr)
        ci_c = bootstrap_system_f1(c_arr, bootstrap_n, seed)
        ci_m = bootstrap_system_f1(m_arr, bootstrap_n, seed)
        ci_pi = bootstrap_system_f1(per_inst, bootstrap_n, seed)
        cls_mean, cls_assign = class_level_oracle(c_arr, m_arr, gt)
        cls_arr = np.zeros(n, dtype=np.float32)
        primary = np.array([np.where(gt[i] > 0)[0][0] for i in range(n)])
        for c in range(gt.shape[1]):
            m = primary == c
            if not m.any():
                continue
            chosen = cls_assign[c]["chosen"]
            cls_arr[m] = c_arr[m] if chosen == "clip" else m_arr[m]
        ci_cls = bootstrap_system_f1(cls_arr, bootstrap_n, seed)
        d = bootstrap_oracle_delta(c_arr, m_arr, gt, bootstrap_n, seed)
        out["metrics"][key] = {
            "label": label, "kind": kind,
            "clip_only": ci_c, "mllm_only": ci_m,
            "class_level_oracle": {**ci_cls, "mean_override": cls_mean},
            "per_instance_oracle": ci_pi,
            "delta_point": ci_pi["mean"] - ci_cls["mean"],
            "delta_ci_lo": d["delta_ci_lo"], "delta_ci_hi": d["delta_ci_hi"],
        }
        log.info(f"  [{label}] clip={ci_c['mean']:.4f} mllm={ci_m['mean']:.4f} "
                 f"cls_oracle={ci_cls['mean']:.4f} per_inst={ci_pi['mean']:.4f} "
                 f"delta={ci_pi['mean']-ci_cls['mean']:+.4f}")
    return out


def format_report(results, args):
    lines = []
    lines.append("=" * 84)
    lines.append("  Phase 0 — Multi-metric robustness check (gate = +5pp on TeacherBehavior)")
    lines.append("=" * 84)
    lines.append(f"  Bootstrap B={args.bootstrap_n} seed={args.seed}")
    lines.append("")
    for res in results:
        lines.append(f"── {res['subset']} | CLIP={res['clip_model']} | N={res['n_images']} ──")
        lines.append(f"  {'metric':<44} {'kind':<11} {'CLIP':>6} {'MLLM':>6} "
                     f"{'clsOr':>6} {'insOr':>6} {'delta':>7}  CI")
        lines.append("  " + "-" * 80)
        for key, m in res["metrics"].items():
            d = m["delta_point"] * 100
            flag = " PASS" if d >= 5.0 else ("" if d < 0 else "")
            lines.append(
                f"  {m['label']:<44} {m['kind']:<11} "
                f"{m['clip_only']['mean']*100:>5.2f}% {m['mllm_only']['mean']*100:>5.2f}% "
                f"{m['class_level_oracle']['mean']*100:>5.2f}% "
                f"{m['per_instance_oracle']['mean']*100:>5.2f}% "
                f"{d:>+6.2f}pp [{m['delta_ci_lo']*100:+.2f},{m['delta_ci_hi']*100:+.2f}]{flag}"
            )
        lines.append("")
    lines.append("=" * 84)
    lines.append("  Reading guide:")
    lines.append("  * delta = per-instance oracle - class-level oracle (the granularity ceiling).")
    lines.append("  * Symmetric metrics score CLIP(thresholded set) and MLLM(pred set) identically.")
    lines.append("  * hit1_asym = CLIP argmax-in-GT vs MLLM any-in-GT (deployment-realistic).")
    lines.append("  * delta >= +5pp on TeacherBehavior/siglip2 would keep the project alive.")
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Phase 0 multi-metric robustness check")
    parser.add_argument("--clip_models", nargs="+", default=["siglip2", "laion"])
    parser.add_argument("--mllm_model", default="qwen3.5_27b")
    parser.add_argument("--subsets", nargs="+", default=["teacher_behavior"])
    parser.add_argument("--bootstrap_n", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ts = time.strftime("%Y%m%d_%H%M%S")
    setup_logging(ts, name="phase0_multimetric")
    log = logging.getLogger()
    log.info(f"Multi-metric check start | args={vars(args)}")

    results = []
    for clip_model in args.clip_models:
        for subset in args.subsets:
            r = evaluate(subset, clip_model, args.mllm_model, args.bootstrap_n, seed=args.seed, log=log)
            if r is not None:
                results.append(r)

    if not results:
        log.error("No results."); return 1

    txt = format_report(results, args)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "phase0_multimetric_report.json", "w") as f:
        json.dump({"args": vars(args), "timestamp": ts, "results": results}, f, indent=2)
    with open(OUT_DIR / "phase0_multimetric_report.txt", "w") as f:
        f.write(txt)
    print()
    print(txt)
    return 0


if __name__ == "__main__":
    exit(main())
