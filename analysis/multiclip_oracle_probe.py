"""
analysis/multiclip_oracle_probe.py — Multi-CLIP-backbone routing feasibility probe
==================================================================================
CONTEXT: original CLIP<->MLLM per-instance routing was negated (Phase 0 ceiling
+3.78pp on TB, R1 realises ~0pp). Probing direction A: per-instance ROUTING
among the 5 CLIP backbones, whose mutual complementarity may be much larger than
CLIP<->MLLM's. Uses only existing per-image .npy -- zero new data, ~30 s.

Question (gate for direction A): on each subset, what is the per-instance oracle
(best backbone per image) headroom over (a) the best single backbone and
(b) the class-level oracle (best backbone per primary GT class)? If per-instance
- class-level >= 5pp on TeacherBehavior, direction A is worth pursuing.

Metric: sample-F1 (locked). Each backbone's predicted set = thresholded logits
(best threshold per backbone, same protocol as phase0).

Systems:
  - each backbone alone (5)
  - best single backbone
  - class-level oracle (per primary GT class, pick the backbone with highest
    mean-F1 on that class) -- tightest class-level upper bound
  - per-instance oracle (per image, max F1 across the 5 backbones) -- routing ceiling

Output:
  results/multiclip_oracle_probe.{json,txt}
  logs/multiclip_oracle_<timestamp>.log

Reproducibility:
  python analysis/multiclip_oracle_probe.py
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
    SUBSET_CONFIG, DATA_DIR, OUT_DIR, THRESHOLD_GRID,
    load_clip, per_image_f1, best_clip_threshold, clip_pred_set,
    bootstrap_system_f1, setup_logging,
)

BACKBONES = ["openai", "laion", "siglip2", "eva02", "dfn"]
GATE_PP = 5.0


def class_level_oracle_multi(f1_arrays, gt):
    """Per primary GT class, assign all images to the backbone with highest
    mean-F1 on that class. Returns (system_f1_array, assignment dict)."""
    n = len(gt)
    primary = np.array([np.where(gt[i] > 0)[0][0] for i in range(n)])
    num_classes = gt.shape[1]
    k = len(f1_arrays)
    system_f1 = np.zeros(n, dtype=np.float32)
    assignment = {}
    for c in range(num_classes):
        mask = primary == c
        if not mask.any():
            continue
        scores = [float(f1_arrays[kk][mask].mean()) for kk in range(k)]
        best_k = int(np.argmax(scores))
        system_f1[mask] = f1_arrays[best_k][mask]
        assignment[int(c)] = {"backbone": BACKBONES[best_k],
                              "n": int(mask.sum()), "f1": scores[best_k]}
    return system_f1, assignment


def bootstrap_paired_delta(arr_a, arr_b, gt, n_boot, seed):
    """Bootstrap CI of mean(arr_a) - mean(arr_b) with coupled resampling.
    arr_a, arr_b are fixed per-image arrays (here: per-instance-oracle and the
    class-level-oracle system array, both derived from full-data assignment)."""
    rng = np.random.RandomState(seed)
    n = len(arr_a)
    diffs = np.empty(n_boot, dtype=np.float32)
    for b in range(n_boot):
        idx = rng.randint(0, n, size=n)
        diffs[b] = arr_a[idx].mean() - arr_b[idx].mean()
    return {"delta_mean": float(diffs.mean()),
            "ci_lo": float(np.percentile(diffs, 2.5)),
            "ci_hi": float(np.percentile(diffs, 97.5))}


def probe(subset, bootstrap_n, seed, log):
    log.info(f"=== {subset} ===")
    cfg = SUBSET_CONFIG[subset]
    # load all backbones; use the first as reference labels
    per_backbone = {}
    ref_labels = None
    for bb in BACKBONES:
        clip = load_clip(bb, subset, log)
        if clip is None:
            return None
        logits, labels = clip
        if ref_labels is None:
            ref_labels = labels
        else:
            if not np.array_equal(ref_labels, labels):
                log.error(f"Label mismatch between backbones on {subset} ({bb})")
                raise RuntimeError("backbone label mismatch")
        per_backbone[bb] = logits
    gt = ref_labels
    n = len(gt)
    num_classes = gt.shape[1]

    f1_arrays = []
    indiv = {}
    thr_used = {}
    for bb in BACKBONES:
        logits = per_backbone[bb]
        thr, _ = best_clip_threshold(logits, gt, THRESHOLD_GRID)
        pred = clip_pred_set(logits, thr)
        f1 = per_image_f1(pred, gt)
        f1_arrays.append(f1)
        indiv[bb] = {"f1": float(f1.mean()), "threshold": thr}
    thr_used_log = {bb: indiv[bb]["threshold"] for bb in BACKBONES}
    log.info(f"  N={n}  per-backbone F1: " +
             " ".join(f"{bb}={indiv[bb]['f1']:.4f}" for bb in BACKBONES))

    f1_stack = np.stack(f1_arrays, axis=0)  # (K, N)
    best_single_i = int(np.argmax([indiv[bb]["f1"] for bb in BACKBONES]))
    best_single_name = BACKBONES[best_single_i]
    best_single_f1 = indiv[best_single_name]["f1"]

    per_inst_oracle = f1_stack.max(axis=0)  # (N,)
    cls_oracle_arr, cls_assign = class_level_oracle_multi(f1_arrays, gt)

    ci_best = bootstrap_system_f1(f1_arrays[best_single_i], bootstrap_n, seed)
    ci_perinst = bootstrap_system_f1(per_inst_oracle, bootstrap_n, seed)
    ci_cls = bootstrap_system_f1(cls_oracle_arr, bootstrap_n, seed)
    delta_ci = bootstrap_paired_delta(per_inst_oracle, cls_oracle_arr, gt,
                                      bootstrap_n, seed)

    result = {
        "subset": subset, "n_images": int(n), "num_classes": num_classes,
        "per_backbone": indiv,
        "best_single": {"backbone": best_single_name,
                        **bootstrap_system_f1(f1_arrays[best_single_i], bootstrap_n, seed)},
        "class_level_oracle": {**ci_cls, "assignment": cls_assign},
        "per_instance_oracle": ci_perinst,
        "per_inst_minus_best_single": ci_perinst["mean"] - best_single_f1,
        "per_inst_minus_class_level": {"point": ci_perinst["mean"] - ci_cls["mean"],
                                        **delta_ci},
    }
    log.info(f"  best_single={best_single_f1:.4f} ({best_single_name})  "
             f"cls_oracle={ci_cls['mean']:.4f}  per_inst={ci_perinst['mean']:.4f}  "
             f"delta(cls)={ci_perinst['mean']-ci_cls['mean']:+.4f}")
    return result


def format_report(results, args):
    lines = []
    lines.append("=" * 82)
    lines.append("  Multi-CLIP-backbone routing feasibility probe (direction A)")
    lines.append("  Metric: sample-F1.  Backbones: " + ", ".join(BACKBONES))
    lines.append("=" * 82)
    lines.append(f"  Bootstrap B={args.bootstrap_n} seed={args.seed}")
    lines.append("")
    for res in results:
        lines.append(f"── {res['subset']} | N={res['n_images']} | {res['num_classes']} classes ──")
        lines.append("  per-backbone sample-F1:")
        for bb in BACKBONES:
            lines.append(f"    {bb:<8} {res['per_backbone'][bb]['f1']*100:6.2f}%  (thr={res['per_backbone'][bb]['threshold']})")
        bs = res["best_single"]
        co = res["class_level_oracle"]
        pi = res["per_instance_oracle"]
        d1 = res["per_inst_minus_best_single"]
        d2 = res["per_inst_minus_class_level"]
        lines.append(f"  best single backbone : {bs['backbone']} = {bs['mean']*100:6.2f}%")
        lines.append(f"  class-level oracle    : {co['mean']*100:6.2f}%   "
                     f"CI [{co['ci_lo']*100:.2f},{co['ci_hi']*100:.2f}]")
        lines.append(f"  per-instance oracle   : {pi['mean']*100:6.2f}%   "
                     f"CI [{pi['ci_lo']*100:.2f},{pi['ci_hi']*100:.2f}]")
        lines.append(f"  per-inst - best single = {d1*100:+6.2f}pp  (total routing headroom)")
        lines.append(f"  per-inst - class-level = {d2['point']*100:+6.2f}pp   "
                     f"CI [{d2['ci_lo']*100:+.2f},{d2['ci_hi']*100:+.2f}]  "
                     f"{'-> PASS 5pp gate' if d2['point']*100 >= GATE_PP else '-> <5pp'}")
        lines.append("")
    lines.append("=" * 82)
    lines.append("  Reading guide:")
    lines.append("  * per-inst - class-level >= +5pp on TeacherBehavior => direction A viable.")
    lines.append("  * per-inst - best single   = total headroom any routing/ensemble can unlock.")
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subsets", nargs="+",
                        default=["teacher_behavior", "handrise_readwrite", "bow_turnhead"])
    parser.add_argument("--bootstrap_n", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ts = time.strftime("%Y%m%d_%H%M%S")
    setup_logging(ts, name="multiclip_oracle")
    log = logging.getLogger()
    log.info(f"Multi-CLIP probe start | args={vars(args)}")

    results = []
    for subset in args.subsets:
        r = probe(subset, args.bootstrap_n, args.seed, log)
        if r is not None:
            results.append(r)
    if not results:
        log.error("No results."); return 1

    txt = format_report(results, args)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "multiclip_oracle_probe.json", "w") as f:
        json.dump({"args": vars(args), "timestamp": ts, "results": results}, f, indent=2)
    with open(OUT_DIR / "multiclip_oracle_probe.txt", "w") as f:
        f.write(txt)
    print()
    print(txt)
    return 0


if __name__ == "__main__":
    exit(main())
