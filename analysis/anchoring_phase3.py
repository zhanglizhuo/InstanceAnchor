"""
analysis/anchoring_phase3.py — Phase 3: cross-family complementarity analysis
===========================================================================
Evaluates whether vit_b_16 (ImageNet-supervised, non-CLIP) makes errors that
are complementary to or correlated with CLIP's errors on SCB5.

Key question: Do cross-family vision models also fail on the same "confusing"
images that trip up CLIP? If yes → hard examples are model-agnostic, and
CLIP agreement is a general-purpose hardness signal.
If no → different architectures capture complementary information.

Metrics:
  1. Consensus accuracy of vit_b_16 in CLIP text space.
  2. AUROC of CLIP agreement (and entropy) for predicting vit_b_16 correctness.
  3. Confusion matrix: when CLIP is right/wrong × vit_b_16 is right/wrong.
  4. Per-image correlation of CLIP correctness vs vit_b_16 correctness.

Output:
  results/anchoring_phase3_report.{json,txt}

Reproducibility:
  python analysis/anchoring_phase3.py
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
from pilot_perimage_anchoring import BACKBONES, softmax_entropy, auroc


def crossfamily_correctness(subset, log):
    """Load vit_b_16 predictions in CLIP text space, compute correctness per image."""
    log.info(f"[{subset}] loading crossfamily vit_b_16...")
    logits_path = Path(__file__).resolve().parents[1] / "results" / "per_image" / f"crossfamily_vit_b_16_{subset}_logits.npy"
    labels_path = Path(__file__).resolve().parents[1] / "results" / "per_image" / f"crossfamily_vit_b_16_{subset}_labels.npy"
    if not logits_path.exists():
        log.error(f"  MISSING {logits_path}"); return None
    logits = np.load(str(logits_path))
    labels = np.load(str(labels_path))
    n, c = logits.shape
    cfg = SUBSET_CONFIG[subset]
    if cfg["multilabel"]:
        correct = np.array([logits[i].argmax() in np.where(labels[i] > 0)[0] for i in range(n)], dtype=bool)
    elif labels.ndim == 2:
        correct = logits.argmax(1) == labels.argmax(1)
    else:
        correct = logits.argmax(1) == labels
    log.info(f"  vit_b_16 acc={correct.mean():.4f} shape={logits.shape}")
    return logits, labels, correct


def evaluate(subset, bootstrap_n, seed, log):
    log.info(f"\n{'='*60}")
    log.info(f"  Phase 3: {subset}")
    log.info(f"{'='*60}")

    xf = crossfamily_correctness(subset, log)
    if xf is None: return None
    _, _, cf_correct = xf

    all_logits = []
    gt = None
    for bb in BACKBONES:
        clip = load_clip(bb, subset, log)
        if clip is None: return None
        logits, labels = clip
        if gt is None:
            gt = labels
        elif not np.array_equal(gt, labels):
            raise RuntimeError(f"label mismatch on {subset}")
        all_logits.append(logits)
    logits_stack = np.stack(all_logits, axis=0)
    n = logits_stack.shape[1]
    avg_logits = logits_stack.mean(axis=0)
    num_classes = gt.shape[1]
    consensus_argmax = np.argmax(avg_logits, axis=1)
    cfg = SUBSET_CONFIG[subset]
    if cfg["multilabel"]:
        clip_correct = np.array([consensus_argmax[i] in np.where(gt[i] > 0)[0] for i in range(n)], dtype=bool)
    else:
        clip_correct = consensus_argmax == gt.argmax(1)

    # CLIP signals
    ent_avg, _ = softmax_entropy(avg_logits)
    backbone_argmaxes = np.argmax(logits_stack, axis=2)
    agreement = (backbone_argmaxes == consensus_argmax[None, :]).mean(axis=0)

    rng = np.random.RandomState(seed)

    def _bootstrap_auroc(score, target, n_boot):
        boots = []
        for _ in range(n_boot):
            idx = rng.randint(0, n, size=n)
            if target[idx].std() > 0 and score[idx].std() > 0:
                boots.append(auroc(score[idx], target[idx]))
            else:
                boots.append(np.nan)
        arr = np.array(boots)
        return float(np.nanmean(arr)), float(np.nanquantile(arr, 0.025)), float(np.nanquantile(arr, 0.975))

    # AUROC: CLIP signals → predict vit_b_16 correctness
    agreement_auroc, ag_lo, ag_hi = _bootstrap_auroc(agreement, cf_correct, bootstrap_n)
    entropy_auroc, en_lo, en_hi = _bootstrap_auroc(-ent_avg, cf_correct, bootstrap_n)

    # AUROC: CLIP signals → predict CLIP correctness
    clip_agree_auroc, _, _ = _bootstrap_auroc(agreement, clip_correct, bootstrap_n)
    clip_ent_auroc, _, _ = _bootstrap_auroc(-ent_avg, clip_correct, bootstrap_n)

    # Confusion matrix: CLIP vs vit_b_16 correctness
    confusion = {}
    for clip_ok in [True, False]:
        for cf_ok in [True, False]:
            key = f"CLIP_{'right' if clip_ok else 'wrong'}_vB_{'right' if cf_ok else 'wrong'}"
            confusion[key] = int(((clip_correct == clip_ok) & (cf_correct == cf_ok)).sum())

    # Correlation
    agree_rate = (clip_correct == cf_correct).mean()
    from scipy.stats import chi2_contingency
    ct = np.array([[confusion["CLIP_right_vB_right"], confusion["CLIP_right_vB_wrong"]],
                   [confusion["CLIP_wrong_vB_right"], confusion["CLIP_wrong_vB_wrong"]]])
    chi2, p_val, _, _ = chi2_contingency(ct)

    log.info(f"  CLIP consensus acc: {clip_correct.mean():.4f}")
    log.info(f"  vit_b_16 acc:       {cf_correct.mean():.4f}")
    log.info(f"  Agreement rate:     {agree_rate:.4f}")
    log.info(f"  CLIP agreement → predict vB: AUROC={agreement_auroc:.4f} [{ag_lo:.4f},{ag_hi:.4f}]")
    log.info(f"  CLIP entropy   → predict vB: AUROC={entropy_auroc:.4f} [{en_lo:.4f},{en_hi:.4f}]")
    log.info(f"  Confusion: {json.dumps(confusion)}")
    log.info(f"  Chi2 independence: p={p_val:.6f}")

    return {
        "subset": subset, "n_images": n, "num_classes": num_classes,
        "clip_consensus_accuracy": float(clip_correct.mean()),
        "vB_accuracy": float(cf_correct.mean()),
        "clip_vs_vB_agreement_rate": float(agree_rate),
        "chi2_p_value": float(p_val),
        "predict_vB_correctness": {
            "agreement_auroc": {"mean": agreement_auroc, "ci_lo": ag_lo, "ci_hi": ag_hi},
            "entropy_auroc": {"mean": entropy_auroc, "ci_lo": en_lo, "ci_hi": en_hi},
        },
        "predict_CLIP_correctness": {
            "agreement_auroc": {"mean": clip_agree_auroc},
            "entropy_auroc": {"mean": clip_ent_auroc},
        },
        "confusion_matrix": confusion,
    }


def format_report(results):
    lines = []
    lines.append("=" * 72)
    lines.append("  Phase 3 — Cross-family complementarity (vit_b_16 vs CLIP)")
    lines.append("=" * 72)
    for res in results:
        lines.append(f"\n── {res['subset']} | N={res['n_images']} ──")
        lines.append(f"  CLIP consensus acc={res['clip_consensus_accuracy']*100:.2f}%  "
                     f"vit_b_16 acc={res['vB_accuracy']*100:.2f}%")
        lines.append(f"  CLIP↔vB agreement rate={res['clip_vs_vB_agreement_rate']*100:.1f}%  "
                     f"χ² p={res['chi2_p_value']:.6f}")
        lines.append(f"  CLIP agreement → predict vB correctness: "
                     f"AUROC={res['predict_vB_correctness']['agreement_auroc']['mean']:.4f} "
                     f"[{res['predict_vB_correctness']['agreement_auroc']['ci_lo']:.4f},"
                     f"{res['predict_vB_correctness']['agreement_auroc']['ci_hi']:.4f}]")
        lines.append(f"  CLIP entropy   → predict vB correctness: "
                     f"AUROC={res['predict_vB_correctness']['entropy_auroc']['mean']:.4f} "
                     f"[{res['predict_vB_correctness']['entropy_auroc']['ci_lo']:.4f},"
                     f"{res['predict_vB_correctness']['entropy_auroc']['ci_hi']:.4f}]")
        lines.append("  -- Confusion matrix (CLIP_right/wrong × vB_right/wrong) --")
        cm = res["confusion_matrix"]
        total = sum(cm.values())
        lines.append(f"    {'':>15} {'vB_right':>10} {'vB_wrong':>10}")
        lines.append(f"    {'CLIP_right':>15} {cm['CLIP_right_vB_right']:>10} {cm['CLIP_right_vB_wrong']:>10}")
        lines.append(f"    {'CLIP_wrong':>15} {cm['CLIP_wrong_vB_right']:>10} {cm['CLIP_wrong_vB_wrong']:>10}")
        lines.append(f"    CLIP_wrong + vB_wrong (both fail): {cm['CLIP_wrong_vB_wrong']} / {total} "
                     f"({cm['CLIP_wrong_vB_wrong']/total*100:.1f}%)")
        lines.append(f"    CLIP_wrong + vB_right (vB rescues): {cm['CLIP_wrong_vB_right']} / {total} "
                     f"({cm['CLIP_wrong_vB_right']/total*100:.1f}%)")
    lines.append("\n" + "=" * 72)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subsets", nargs="+",
                        default=["teacher_behavior", "handrise_readwrite", "bow_turnhead"])
    parser.add_argument("--bootstrap_n", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ts = time.strftime("%Y%m%d_%H%M%S")
    setup_logging(ts, name="anchoring_phase3")
    log = logging.getLogger()
    log.info(f"Phase 3 anchoring_phase3 | args={vars(args)}")

    results = []
    for subset in args.subsets:
        r = evaluate(subset, args.bootstrap_n, args.seed, log)
        if r is not None:
            results.append(r)
    if not results:
        log.error("No results."); return 1

    txt = format_report(results)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "anchoring_phase3_report.json", "w") as f:
        json.dump({"args": vars(args), "timestamp": ts, "results": results}, f, indent=2)
    with open(OUT_DIR / "anchoring_phase3_report.txt", "w") as f:
        f.write(txt)
    print(); print(txt)
    return 0


if __name__ == "__main__":
    exit(main())
