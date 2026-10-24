"""
analysis/router_r1_entropy.py — R1 entropy router vs class-level router (Pareto)
================================================================================
CONTEXT: Phase 0 showed the per-instance ORACLE ceiling over the class-level
oracle is only +3.78pp on TeacherBehavior (robust across metrics, <5pp gate).
That is an ACCURACY ceiling. The open question is the COST dimension: can a
realistic per-instance router reach a target accuracy at lower MLLM budget than
class-level routing? This script answers that.

R1 (per-instance entropy router, deployment-realistic):
  signal  = softmax entropy of CLIP logits (temperature T=100, AGENTS.md rule 14)
  rule    = route the k% highest-entropy images to MLLM, rest to CLIP
  output  = CLIP thresholded set if kept, MLLM pred set if routed
  Sweep k in [0,100] -> Pareto curve (system sample-F1 vs MLLM budget).

Class-level router (realistic baseline, VisualAnchor-style):
  signal  = per-class CLIP sample-F1 (an AnchorScore proxy computed on this data)
  rule    = route by PREDICTED primary class (CLIP argmax); greedily route the
            worst-CLIP-F1 predicted-classes to MLLM until image budget = k.
  This uses only CLIP outputs at decision time (no GT) -- a fair realistic baseline.

Endpoints / ceilings (reference):
  CLIP-only (k=0), MLLM-only (k=100), per-instance oracle, class-level oracle.

Primary metric: sample-F1 (locked). Bootstrap 95% CI at fixed operating points.

Output:
  results/router_r1_report.{json,txt}
  logs/router_r1_<timestamp>.log

Reproducibility:
  python analysis/router_r1_entropy.py
  python analysis/router_r1_entropy.py --clip_model siglip2 --budgets 0.1 0.25 0.5 0.75
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
    SUBSET_CONFIG, OUT_DIR, THRESHOLD_GRID,
    load_clip, load_mllm, align_and_audit,
    best_clip_threshold, clip_pred_set, per_image_f1,
    class_level_oracle, bootstrap_system_f1, setup_logging,
)

TEMPERATURE = 100.0  # CLIP logit_scale proxy (AGENTS.md rule 14)


def clip_entropy(logits, temperature=TEMPERATURE):
    """Per-image softmax entropy of (logits * temperature)."""
    z = logits * temperature
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    p = e / e.sum(axis=1, keepdims=True)
    return -np.sum(p * np.log(p + 1e-12), axis=1)


def router_r1_pareto(f1_clip, f1_mllm, entropy):
    """R1: route top-k% highest-entropy images to MLLM. Returns sorted-by-budget
    Pareto array of (budget, system_f1). Budget = fraction routed to MLLM.
    By construction this is the best R1 can do at each budget under the entropy
    signal (greedy on the signal)."""
    n = len(entropy)
    order = np.argsort(-entropy)  # descending entropy: most uncertain first
    f1_sorted_clip = f1_clip[order]
    f1_sorted_mllm = f1_mllm[order]
    # Route the first m (highest-entropy) images to MLLM, rest stay on CLIP.
    # system F1 (mean) = (sum f1_mllm[:m] + sum f1_clip[m:]) / n
    pref_mllm = np.concatenate([[0.0], np.cumsum(f1_sorted_mllm)])
    pref_clip = np.concatenate([[0.0], np.cumsum(f1_sorted_clip)])
    total_clip = pref_clip[-1]
    sys_f1 = (pref_mllm + (total_clip - pref_clip)) / n
    budgets = np.arange(n + 1) / n
    return budgets, sys_f1


def router_r1_at_budget(f1_clip, f1_mllm, entropy, budget):
    """System F1 when routing fraction `budget` (highest-entropy) to MLLM."""
    n = len(entropy)
    m = int(round(budget * n))
    order = np.argsort(-entropy)
    routed = np.zeros(n, dtype=bool)
    routed[order[:m]] = True
    sys_f1 = np.where(routed, f1_mllm, f1_clip)
    return sys_f1, m


def class_level_realistic_pareto(f1_clip, f1_mllm, clip_pred_primary,
                                 per_class_clip_f1):
    """Route by PREDICTED primary class. Sort predicted-classes by CLIP-F1
    ascending; greedily route worst classes to MLLM up to budget. Returns
    list of (budget, system_f1) -- at most num_classes+1 discrete points."""
    n = len(clip_pred_primary)
    classes = np.unique(clip_pred_primary)
    cls_f1 = {int(c): float(per_class_clip_f1[c]) for c in classes}
    class_sizes = {int(c): int(np.sum(clip_pred_primary == c)) for c in classes}
    order_cls = sorted(classes, key=lambda c: cls_f1[int(c)])  # worst CLIP first
    # cumulative images as we add classes to MLLM set
    points = []
    mllm_set = set()
    acc = 0.0
    total = float(n)
    routed_flag = np.zeros(n, dtype=bool)
    # start: all CLIP
    points.append((0.0, float(f1_clip.mean())))
    for c in order_cls:
        mllm_set.add(int(c))
        mask = np.isin(clip_pred_primary, list(mllm_set))
        sys = np.where(mask, f1_mllm, f1_clip)
        budget = float(mask.sum()) / n
        points.append((budget, float(sys.mean())))
    # dedupe + sort by budget
    points = sorted(set(points), key=lambda x: x[0])
    return points


def class_level_at_budget(f1_clip, f1_mllm, clip_pred_primary, per_class_clip_f1, budget):
    """Closest class-level operating point >= budget (class granularity is coarse)."""
    pts = class_level_realistic_pareto(f1_clip, f1_mllm, clip_pred_primary, per_class_clip_f1)
    for b, f in pts:
        if b >= budget - 1e-9:
            mask = _cls_mask_at_budget(clip_pred_primary, per_class_clip_f1, b, pts)
            sys = np.where(mask, f1_mllm, f1_clip)
            return sys, b
    mask = np.ones(len(f1_clip), dtype=bool)
    return f1_mllm, 1.0


def _cls_mask_at_budget(clip_pred_primary, per_class_clip_f1, budget, points):
    classes = np.unique(clip_pred_primary)
    order_cls = sorted(classes, key=lambda c: float(per_class_clip_f1[int(c)]))
    mllm_set = set()
    for c in order_cls:
        mllm_set.add(int(c))
        mask = np.isin(clip_pred_primary, list(mllm_set))
        if float(mask.sum()) / len(mask) >= budget - 1e-9:
            return mask
    return np.ones(len(clip_pred_primary), dtype=bool)


def per_class_clip_f1(f1_clip, clip_pred_primary, num_classes):
    out = np.zeros(num_classes, dtype=np.float32)
    for c in range(num_classes):
        m = clip_pred_primary == c
        out[c] = float(f1_clip[m].mean()) if m.any() else 0.0
    return out


def evaluate(subset, clip_model, mllm_model, budgets, bootstrap_n, seed, log):
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
    num_classes = gt.shape[1]

    thr, _ = best_clip_threshold(logits, gt, THRESHOLD_GRID)
    clip_pred = clip_pred_set(logits, thr)
    log.info(f"  N={n} CLIP threshold={thr} temperature={TEMPERATURE}")

    f1_clip = per_image_f1(clip_pred, gt)
    f1_mllm = per_image_f1(mllm_pred, gt)
    ent = clip_entropy(logits)
    clip_pred_primary = np.argmax(logits, axis=1)
    pc_clip_f1 = per_class_clip_f1(f1_clip, clip_pred_primary, num_classes)

    # ceilings
    per_inst_oracle = np.maximum(f1_clip, f1_mllm)
    cls_oracle_mean, _ = class_level_oracle(f1_clip, f1_mllm, gt)

    # Pareto curves
    r1_budgets, r1_sys = router_r1_pareto(f1_clip, f1_mllm, ent)
    cls_pts = class_level_realistic_pareto(f1_clip, f1_mllm, clip_pred_primary, pc_clip_f1)

    # operating points with bootstrap CI
    ops = {}
    for b in budgets:
        r1_arr, m_r1 = router_r1_at_budget(f1_clip, f1_mllm, ent, b)
        ci_r1 = bootstrap_system_f1(r1_arr, bootstrap_n, seed)
        cls_arr, b_cls = class_level_at_budget(f1_clip, f1_mllm, clip_pred_primary,
                                               pc_clip_f1, b)
        ci_cls = bootstrap_system_f1(cls_arr, bootstrap_n, seed)
        ops[f"{int(b*100)}pct"] = {
            "budget_target": b,
            "r1": {**ci_r1, "actual_budget": m_r1 / n},
            "class_level": {**ci_cls, "actual_budget": float(b_cls)},
            "r1_minus_cls": ci_r1["mean"] - ci_cls["mean"],
        }
        log.info(f"  budget={b:.0%}: R1={ci_r1['mean']:.4f} (n_mllm={m_r1})  "
                 f"CLSlvl={ci_cls['mean']:.4f} (budget={b_cls:.3f})  "
                 f"delta={ci_r1['mean']-ci_cls['mean']:+.4f}")

    # budget to reach MLLM-only accuracy
    mllm_only_acc = float(f1_mllm.mean())
    r1_reach = next((float(r1_budgets[i]) for i in range(len(r1_sys))
                     if r1_sys[i] >= mllm_only_acc), None)

    result = {
        "subset": subset, "clip_model": clip_model, "mllm_model": mllm_model,
        "n_images": int(n), "clip_threshold": thr, "temperature": TEMPERATURE,
        "endpoints": {
            "clip_only": float(f1_clip.mean()),
            "mllm_only": mllm_only_acc,
            "class_level_oracle_ceiling": cls_oracle_mean,
            "per_instance_oracle_ceiling": float(per_inst_oracle.mean()),
        },
        "r1_pareto": [{"budget": float(b), "f1": float(f)}
                      for b, f in zip(r1_budgets[::max(1, len(r1_budgets)//50)],
                                      r1_sys[::max(1, len(r1_sys)//50)])],
        "class_level_pareto": [{"budget": float(b), "f1": float(f)} for b, f in cls_pts],
        "operating_points": ops,
        "r1_budget_to_reach_mllm_only": r1_reach,
    }
    return result


def format_report(results, args):
    lines = []
    lines.append("=" * 84)
    lines.append("  R1 entropy router vs class-level router (Pareto, sample-F1)")
    lines.append("=" * 84)
    lines.append(f"  temperature={TEMPERATURE}  bootstrap B={args.bootstrap_n} seed={args.seed}")
    lines.append("")
    for res in results:
        e = res["endpoints"]
        lines.append(f"── {res['subset']} | CLIP={res['clip_model']} | N={res['n_images']} ──")
        lines.append(f"  endpoints: CLIP-only={e['clip_only']*100:.2f}%  "
                     f"MLLM-only={e['mllm_only']*100:.2f}%  "
                     f"cls-oracle ceiling={e['class_level_oracle_ceiling']*100:.2f}%  "
                     f"per-inst-oracle ceiling={e['per_instance_oracle_ceiling']*100:.2f}%")
        rr = res["r1_budget_to_reach_mllm_only"]
        lines.append(f"  R1 reaches MLLM-only accuracy at budget = "
                     f"{rr*100:.1f}%" if rr is not None else "  R1 never reaches MLLM-only accuracy")
        lines.append("")
        lines.append(f"  {'budget':>7} | {'R1 sample-F1 (CI)':<24} | "
                     f"{'class-level (CI)':<24} | {'R1-cls':>8}")
        lines.append("  " + "-" * 74)
        for key, op in res["operating_points"].items():
            r, c = op["r1"], op["class_level"]
            d = op["r1_minus_cls"]
            star = " *" if d > 0 else ""
            lines.append(
                f"  {op['budget_target']*100:>6.0f}% | "
                f"{r['mean']*100:5.2f}% [{r['ci_lo']*100:5.2f},{r['ci_hi']*100:5.2f}] "
                f"(b={r['actual_budget']*100:4.1f}%) | "
                f"{c['mean']*100:5.2f}% [{c['ci_lo']*100:5.2f},{c['ci_hi']*100:5.2f}] "
                f"(b={c['actual_budget']*100:4.1f}%) | "
                f"{d*100:+5.2f}pp{star}"
            )
        lines.append("")
    lines.append("=" * 84)
    lines.append("  Reading guide:")
    lines.append("  * R1 = entropy router (per-instance); class-level = VisualAnchor-style (predicted class).")
    lines.append("  * 'b=' = actual MLLM fraction routed. class-level granularity is coarse (<=num_classes).")
    lines.append("  * R1-cls > 0 (marked *) = per-instance beats class-level at matched budget.")
    lines.append("  * Compare R1's budget-to-reach-MLLM-only vs 100% (MLLM-only) for cost savings.")
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="R1 entropy router Pareto vs class-level")
    parser.add_argument("--clip_model", default="siglip2")
    parser.add_argument("--mllm_model", default="qwen3.5_27b")
    parser.add_argument("--subsets", nargs="+",
                        default=["teacher_behavior", "handrise_readwrite"])
    parser.add_argument("--budgets", nargs="+", type=float,
                        default=[0.10, 0.25, 0.50, 0.75, 0.90])
    parser.add_argument("--bootstrap_n", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ts = time.strftime("%Y%m%d_%H%M%S")
    setup_logging(ts, name="router_r1")
    log = logging.getLogger()
    log.info(f"R1 router start | args={vars(args)}")

    results = []
    for subset in args.subsets:
        r = evaluate(subset, args.clip_model, args.mllm_model,
                     args.budgets, args.bootstrap_n, args.seed, log)
        if r is not None:
            results.append(r)

    if not results:
        log.error("No results."); return 1

    txt = format_report(results, args)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "router_r1_report.json", "w") as f:
        json.dump({"args": vars(args), "timestamp": ts, "results": results}, f, indent=2)
    with open(OUT_DIR / "router_r1_report.txt", "w") as f:
        f.write(txt)
    print()
    print(txt)
    return 0


if __name__ == "__main__":
    exit(main())
