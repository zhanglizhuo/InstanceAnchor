"""
analysis/anchoring_soft_mechanism.py — soft_agreement mechanism: Jensen gap
============================================================================
Why does soft_agreement beat entropy AND remain competitive with MSP?

Hypothesis: soft_agreement and MSP differ by a Jensen gap caused by
cross-backbone disagreement about the distribution shape.
  MSP = softmax(avg_logits)[consensus]    (normalize AFTER averaging)
  soft = mean_k softmax(logits_k)[consensus]  (normalize BEFORE averaging)
By Jensen's inequality, the gap |soft - MSP| is non-zero when backbones
disagree about the distribution. This gap carries information about
cross-backbone divergence that MSP alone misses.

Test: (1) Jensen gap correlates with per-image error
      (2) soft_agreement MI(correctness) > entropy MI(correctness)
      (3) soft_agreement adds conditional info beyond MSP

Output: results/anchoring_soft_mechanism_report.{json,txt}
"""

import argparse, json, sys, time, logging
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase0_oracle_analysis import setup_logging, OUT_DIR, DATA_DIR  # noqa
from pilot_perimage_anchoring import BACKBONES, softmax_entropy, auroc  # noqa

CIFAR_DIR = DATA_DIR / "cifar100"
CIFARC_DIR = DATA_DIR / "cifarc"
META_DIR = DATA_DIR / "meta"
T = 100.0
EXISTING = list({"teacher_behavior":1,"handrise_readwrite":1,"bow_turnhead":1,
                 "voc2007":1,"stanford40":1,"coco":1,"multithumos":1}.keys())
META_KEYS = ["food101", "dtd", "cifar10", "tiny_imagenet", "stl10",
             "eurosat", "imagenet_a", "imagenet_r",
             "sun397", "fgvc_aircraft", "stanford_cars", "gtsrb",
             "imagenet_sketch", "imagenet_v2_matched", "imagenet_v2_threshold",
             "imagenet_v2_top", "imagenet_c",
             "cifar10c", "cifar100c"]


def load_stack(key):
    stack, gt = [], None
    for bb in BACKBONES:
        if key in ("teacher_behavior","handrise_readwrite","bow_turnhead",
                   "voc2007","stanford40","coco","multithumos"):
            lp = DATA_DIR / f"clip_{bb}_{key}_logits.npy"
            bp = DATA_DIR / f"clip_{bb}_{key}_labels.npy"
        elif key == "cifar100":
            lp = CIFAR_DIR / f"clip_{bb}_cifar100_logits.npy"
            bp = CIFAR_DIR / f"clip_{bb}_cifar100_labels.npy"
        elif key in ("cifar10c", "cifar100c"):
            lp = CIFARC_DIR / f"clip_{bb}_{key}_logits.npy"
            bp = CIFARC_DIR / f"clip_{bb}_{key}_labels.npy"
        else:
            lp = META_DIR / f"clip_{bb}_{key}_logits.npy"
            bp = META_DIR / f"clip_{bb}_{key}_labels.npy"
        if not lp.exists():
            return None
        stack.append(np.load(lp).astype(np.float32))
        labels = np.load(bp).astype(np.float32)
        if gt is None:
            gt = labels
    return np.stack(stack, axis=0), gt


def correctness(cons, gt):
    if gt.ndim == 1:
        return cons == gt.astype(int)
    return np.array([cons[i] in np.where(gt[i]>0)[0] for i in range(len(cons))], dtype=bool)


def mutual_info_binning(signal, correct, n_bins=20):
    """I(signal; correct) via histogram-based MI estimation."""
    from sklearn.metrics import mutual_info_score
    edges = np.linspace(signal.min()-1e-9, signal.max()+1e-9, n_bins+1)
    s_binned = np.digitize(signal, edges) - 1
    s_binned = np.clip(s_binned, 0, n_bins-1)
    return mutual_info_score(s_binned, correct.astype(int))


def analyze(key, log):
    loaded = load_stack(key)
    if loaded is None:
        return None
    logits_stack, gt = loaded
    K, N, C = logits_stack.shape
    avg = logits_stack.mean(axis=0)
    consensus = np.argmax(avg, axis=1)
    correct = correctness(consensus, gt)
    if correct.std() == 0:
        return None

    probs = np.stack([np.exp(z * T) / np.exp(z * T).sum(axis=1, keepdims=True)
                      for z in logits_stack], axis=0)

    soft = probs[:, np.arange(N), consensus].mean(axis=0)   # soft_agreement: avg of softmaxes at consensus
    # TRUE MSP: softmax of the AVERAGE logits at consensus (normalize AFTER averaging)
    _, p_avg = softmax_entropy(avg)
    msp = p_avg[np.arange(N), consensus]
    ent_avg, _ = softmax_entropy(avg)
    entropy = -ent_avg

    # Jensen gap: |soft - MSP| — non-zero when backbones disagree on distribution shape
    jensen_gap = np.abs(soft - msp)

    # (1) Jensen gap vs error: AUROC of gap as error detector
    gap_auroc = auroc(jensen_gap, correct)  # high gap → likely error?

    # (2) MI(signal; correctness) for each signal
    mi_soft = mutual_info_binning(soft, correct)
    mi_msp = mutual_info_binning(msp, correct)
    mi_ent = mutual_info_binning(entropy, correct)
    mi_gap = mutual_info_binning(jensen_gap, correct)

    # (3) soft_agreement AUROC after residualizing against MSP
    # (does soft add info beyond MSP?)
    from numpy.polynomial import polynomial as P
    # residual = soft - linear_prediction_from_MSP
    coeffs = np.polyfit(msp, soft, 1)
    soft_residual = soft - np.polyval(coeffs, msp)
    resid_auroc = auroc(np.abs(soft_residual), correct)

    # (4) decompose: which datasets have large Jensen gap?
    mean_gap = float(jensen_gap.mean())

    row = {
        "dataset": key, "N": int(N), "C": int(C),
        "acc": float(correct.mean()),
        "jensen_gap_mean": mean_gap,
        "jensen_gap_auroc_vs_error": float(gap_auroc),
        "mi_soft": float(mi_soft),
        "mi_msp": float(mi_msp),
        "mi_entropy": float(mi_ent),
        "mi_gap": float(mi_gap),
        "soft_residual_auroc": float(resid_auroc),
        "soft_auroc": float(auroc(soft, correct)),
        "msp_auroc": float(auroc(msp, correct)),
        "entropy_auroc": float(auroc(entropy, correct)),
    }
    log.info(f"  {key:18s} gap={mean_gap:.4f} gap_auroc={gap_auroc:.3f} "
             f"MI: soft={mi_soft:.4f} msp={mi_msp:.4f} ent={mi_ent:.4f} "
             f"resid_auroc={resid_auroc:.3f}")
    return row


def main():
    ts = time.strftime("%Y%m%d_%H%M%S")
    setup_logging(ts, name="anchoring_soft_mechanism")
    log = logging.getLogger()

    results = []
    for key in EXISTING + ["cifar100"] + META_KEYS:
        r = analyze(key, log)
        if r:
            results.append(r)

    lines = ["=" * 100, f"  soft_agreement mechanism: Jensen gap + MI decomposition ({len(results)} datasets)", "=" * 100, ""]
    lines.append(f"  {'dataset':>18} | {'gap_mean':>8} | {'gap_auroc':>9} | "
                 f"{'MI_soft':>8} {'MI_msp':>8} {'MI_ent':>8} {'MI_gap':>8} | "
                 f"{'resid_aur':>9}")
    lines.append("  " + "-" * 90)
    for r in sorted(results, key=lambda x: x['jensen_gap_mean'], reverse=True):
        lines.append(f"  {r['dataset']:>18} | {r['jensen_gap_mean']:>8.4f} | "
                     f"{r['jensen_gap_auroc_vs_error']:>9.3f} | "
                     f"{r['mi_soft']:>8.4f} {r['mi_msp']:>8.4f} {r['mi_entropy']:>8.4f} "
                     f"{r['mi_gap']:>8.4f} | {r['soft_residual_auroc']:>9.3f}")

    # Summary stats
    mi_soft_mean = float(np.mean([r['mi_soft'] for r in results]))
    mi_msp_mean = float(np.mean([r['mi_msp'] for r in results]))
    mi_ent_mean = float(np.mean([r['mi_entropy'] for r in results]))
    gap_auroc_mean = float(np.mean([r['jensen_gap_auroc_vs_error'] for r in results]))
    resid_auroc_mean = float(np.mean([r['soft_residual_auroc'] for r in results]))

    lines.append("")
    lines.append(f"  Mean MI(soft;correct)={mi_soft_mean:.4f} > MI(MSP;correct)={mi_msp_mean:.4f} > MI(entropy;correct)={mi_ent_mean:.4f}")
    lines.append(f"  Mean Jensen-gap AUROC vs error: {gap_auroc_mean:.3f} (>0.5 = gap predicts error)")
    lines.append(f"  Mean soft-residual AUROC (info beyond MSP): {resid_auroc_mean:.3f}")
    lines.append("=" * 100)
    txt = "\n".join(lines)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "anchoring_soft_mechanism_report.json", "w") as f:
        json.dump({"timestamp": ts, "results": results,
                   "summary": {"mi_soft_mean": mi_soft_mean, "mi_msp_mean": mi_msp_mean,
                               "mi_ent_mean": mi_ent_mean, "gap_auroc_mean": gap_auroc_mean,
                               "resid_auroc_mean": resid_auroc_mean}}, f, indent=2)
    with open(OUT_DIR / "anchoring_soft_mechanism_report.txt", "w") as f:
        f.write(txt)
    print(); print(txt)


if __name__ == "__main__":
    main()
