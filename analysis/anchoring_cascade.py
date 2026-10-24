"""
analysis/anchoring_cascade.py — Exp H: selective cascade (CLIP → MLLM fallback)
=============================================================================
Selective deferral: use CLIP 5-backbone ensemble with agreement signal; defer
low-agreement images to MLLM as a fixed fallback.

Framing: NOT per-instance routing between two active models. The cascade is a
SELECTIVE PREDICTION scheme: CLIP predicts when confident (high agreement); MLLM
serves as a static fallback for the rest.

Signals compared as deferral criteria:
  - agreement_top1
  - ensemble entropy (compute-matched, from 5-backbone avg logits)

Metrics:
  - Accuracy vs MLLM usage rate (risk-usage curve)
  - AUROC of deferral signal for CLIP correctness
  - Accuracy improvement over CLIP-only at fixed MLLM budget

Output: results/anchoring_cascade_report.{json,txt}

Reproducibility:
  python analysis/anchoring_cascade.py [--subsets teacher_behavior]
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase0_oracle_analysis import SUBSET_CONFIG, load_clip, setup_logging, OUT_DIR
from pilot_perimage_anchoring import BACKBONES, softmax_entropy, auroc, bootstrap_auroc_diff

TB_CLASSES = ["blackboard-writing", "on-stage interaction", "guide",
              "answer", "teacher", "stand", "screen", "blackboard"]

MLLM_FILES = {
    "teacher_behavior": "mllm_qwen3.5_27b_teacher_behavior_preds.jsonl",
}


def load_mllm(subset, data_dir):
    import json as _json
    fname = MLLM_FILES.get(subset)
    if fname is None:
        return None
    path = data_dir / fname
    if not path.exists():
        return None
    indices, pred_class_lists, gt_vectors = [], [], []
    with open(path) as f:
        for line in f:
            d = _json.loads(line)
            indices.append(d["index"])
            pred_class_lists.append(d["pred_classes"])
            gt_vectors.append(np.array(d["gt_vector"], dtype=np.float32))
    return np.array(indices), pred_class_lists, np.stack(gt_vectors, 0)


def mllm_hit(pred_classes, gt_labels):
    gt_set = set(gt_labels)
    return float(any(p in gt_set for p in pred_classes))


def evaluate(subset, bootstrap_n, seed, log):
    log.info(f"=== {subset} ===")
    all_logits = {}
    clip_gt = None
    for bb in BACKBONES:
        clip = load_clip(bb, subset, log)
        if clip is None:
            return None
        logits, labels = clip
        if clip_gt is None:
            clip_gt = labels
        elif not np.array_equal(clip_gt, labels):
            raise RuntimeError("label mismatch")
        all_logits[bb] = logits
    n = len(clip_gt)
    classes = SUBSET_CONFIG[subset]["classes"]

    # CLIP signals
    stack = np.stack([all_logits[b] for b in BACKBONES], 0)
    avg = stack.mean(0)
    consensus_argmax = np.argmax(avg, 1)
    clip_correct = np.array([consensus_argmax[i] in np.where(clip_gt[i] > 0)[0]
                             for i in range(n)], dtype=bool)
    ent, _ = softmax_entropy(avg)
    s_ent = -ent
    barg = np.argmax(stack, 2)
    s_agree = (barg == consensus_argmax[None, :]).mean(0)

    # MLLM data
    mllm_data = load_mllm(subset, OUT_DIR.parent / "results" / "per_image")
    if mllm_data is None:
        log.warning(f"MLLM data unavailable for {subset}")
        return None
    mllm_indices, mllm_preds, mllm_gt = mllm_data
    if not np.array_equal(mllm_gt, clip_gt):
        raise RuntimeError(
            "MLLM GT does not match CLIP GT; per-image alignment invalid")
    if not np.array_equal(np.sort(mllm_indices), np.arange(n)):
        raise RuntimeError(
            "MLLM rows are not a full, dataset-ordered cover of the subset; "
            "mllm_preds[i] would be misindexed")

    # Convert MLLM pred_classes (strings) to binary vectors for TB
    gt_labels_for_mllm = np.array([
        [classes[c] for c in np.where(clip_gt[i] > 0)[0]]
        for i in range(n)
    ], dtype=object)
    mllm_correct = np.array([
        mllm_hit(mllm_preds[i], gt_labels_for_mllm[i])
        for i in range(n)
    ], dtype=bool)
    log.info(f"  CLIP consensus acc={clip_correct.mean():.4f}  "
             f"MLLM hit@any={mllm_correct.mean():.4f}  "
             f"MLLM avg preds={np.mean([len(p) for p in mllm_preds]):.2f}")

    # Cascade: defer low-agreement / high-entropy images to MLLM
    n_vals = np.linspace(0, n - 1, min(n, 101), dtype=int)
    thresholds_agree = np.sort(s_agree)[n_vals]
    thresholds_ent = np.sort(ent)[n_vals]

    cascade_agree, cascade_ent = [], []
    for tau_a in thresholds_agree:
        defer = s_agree < tau_a
        final_correct = np.where(defer, mllm_correct, clip_correct)
        mllm_usage = defer.mean()
        cascade_agree.append({"mllm_usage": float(mllm_usage),
                              "accuracy": float(final_correct.mean())})
    for tau_e in thresholds_ent:
        defer = ent >= tau_e
        final_correct = np.where(defer, mllm_correct, clip_correct)
        mllm_usage = defer.mean()
        cascade_ent.append({"mllm_usage": float(mllm_usage),
                            "accuracy": float(final_correct.mean())})

    # AUROC of deferral signals (high signal = likely CLIP error)
    error_signal_agree = 1 - s_agree  # high when disagreement → likely error
    ent = -s_ent  # positive entropy
    error_signal_ent = ent  # high entropy → likely error
    delta = bootstrap_auroc_diff(error_signal_agree, error_signal_ent, ~clip_correct, bootstrap_n, seed)

    # Key operating points: 20%, 50%, 80% MLLM usage
    op_points = [0.2, 0.5, 0.8]
    ops_agree = {}
    for pct in op_points:
        best = min((c for c in cascade_agree if c["mllm_usage"] >= pct),
                   key=lambda c: c["mllm_usage"], default=None)
        ops_agree[f"@{pct*100:.0f}%"] = best
    ops_ent = {}
    for pct in op_points:
        best = min((c for c in cascade_ent if c["mllm_usage"] >= pct),
                   key=lambda c: c["mllm_usage"], default=None)
        ops_ent[f"@{pct*100:.0f}%"] = best

    return {
        "subset": subset,
        "n_images": int(n),
        "clip_consensus_acc": float(clip_correct.mean()),
        "mllm_hit_any": float(mllm_correct.mean()),
        "deferral_signal_auroc_agree_for_clip_error": float(auroc(error_signal_agree, ~clip_correct)),
        "deferral_signal_auroc_ent_for_clip_error": float(auroc(error_signal_ent, ~clip_correct)),
        "delta_agree_minus_ent": delta,
        "cascade_agreement": cascade_agree,
        "cascade_entropy": cascade_ent,
        "ops_agreement_deferral": ops_agree,
        "ops_entropy_deferral": ops_ent,
    }


def format_report(results, args):
    lines = []
    lines.append("=" * 86)
    lines.append("  Exp H — Selective cascade: CLIP agreement → MLLM fallback")
    lines.append("=" * 86)
    lines.append("")
    for res in results:
        lines.append(f"── {res['subset']} | N={res['n_images']} ──")
        lines.append(f"  CLIP consensus acc = {res['clip_consensus_acc']*100:.2f}%")
        lines.append(f"  MLLM hit@any       = {res['mllm_hit_any']*100:.2f}%")
        lines.append(f"  Deferral signal AUROC for detecting CLIP errors:")
        lines.append(f"    agreement: {res['deferral_signal_auroc_agree_for_clip_error']:.4f}")
        lines.append(f"    entropy:   {res['deferral_signal_auroc_ent_for_clip_error']:.4f}")
        d = res["delta_agree_minus_ent"]
        lines.append(f"    agree-entropy delta: {d['mean']:+.4f} CI[{d['ci_lo']:+.4f},{d['ci_hi']:+.4f}]")
        lines.append("")
        lines.append(f"  Operating points (agreement-based deferral; "
                     f"labels are targets, actual usage = nearest grid point at or above target):")
        lines.append(f"    {'MLLM usage':>12} | {'accuracy':>9}")
        lines.append("    " + "-" * 25)
        for label, op in res["ops_agreement_deferral"].items():
            if op:
                lines.append(f"    {label:>12} | {op['accuracy']*100:>7.2f}%  (actual usage {op['mllm_usage']*100:.1f}%)")
        lines.append("")
        lines.append(f"  Operating points (entropy-based deferral; "
                     f"labels are targets, actual usage = nearest grid point at or above target):")
        lines.append(f"    {'MLLM usage':>12} | {'accuracy':>9}")
        lines.append("    " + "-" * 25)
        for label, op in res["ops_entropy_deferral"].items():
            if op:
                lines.append(f"    {label:>12} | {op['accuracy']*100:>7.2f}%  (actual usage {op['mllm_usage']*100:.1f}%)")
        lines.append("")
    lines.append("=" * 86)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subsets", nargs="+", default=["teacher_behavior"])
    parser.add_argument("--bootstrap_n", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ts = time.strftime("%Y%m%d_%H%M%S")
    setup_logging(ts, name="anchoring_cascade")
    log = logging.getLogger()
    log.info(f"Exp H | args={vars(args)}")

    results = []
    for subset in args.subsets:
        r = evaluate(subset, args.bootstrap_n, args.seed, log)
        if r is not None:
            results.append(r)
    if not results:
        log.error("No results."); return 1

    txt = format_report(results, args)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "anchoring_cascade_report.json", "w") as f:
        json.dump({"args": vars(args), "timestamp": ts, "results": results}, f, indent=2)
    with open(OUT_DIR / "anchoring_cascade_report.txt", "w") as f:
        f.write(txt)
    print(); print(txt)
    return 0


if __name__ == "__main__":
    exit(main())
