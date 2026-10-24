"""
analysis/pilot_perimage_anchoring.py — Pilot for topic #2 (per-image anchoring)
===============================================================================
CORE BET (gate for topic #2): does cross-backbone agreement -- a per-image
visual-anchoring proxy -- predict per-image CLIP correctness BETTER than the
standard entropy signal? If yes, topic #2 (instance-level anchoring for
selective prediction) is alive; if no, downgrade.

This extends the class-level AnchorScore concept (VisualAnchor, sibling #3) to
instance granularity, but for a DIFFERENT task: selective prediction (abstain
on low-anchoring images), NOT routing. Pure CLIP -- no MLLM needed for the core
bet.

Setup (per subset):
  - 5 CLIP backbones' logits.
  - Consensus prediction = argmax of averaged logits.
  - correct[i] = consensus argmax in GT (multilabel: any true class).
  Signals (per image, all in [higher == keep / lower == abstain] orientation):
    entropy     : entropy of averaged-logit softmax        (baseline; LOW=keep)
    margin      : top1 - top2 of averaged logits            (HIGH=keep)
    agreement   : fraction of the 5 backbones whose argmax == consensus argmax
                                                                   (HIGH=keep)
    logitvar    : mean across-backbone variance of logits   (LOW=keep) [disagreement]
  Evaluation:
    - AUROC + AUPR for predicting `correct` (sign-flipped so higher=better).
    - Risk-coverage: keep top-signal fraction c, report accuracy on kept set.
    - Coupled bootstrap CI for (agreement_AUROC - entropy_AUROC).

GATE: agreement AUROC > entropy AUROC on TeacherBehavior with CI excluding 0.

Output:
  results/pilot_perimage_anchoring.{json,txt}
  logs/pilot_anchoring_<timestamp>.log

Reproducibility:
  python analysis/pilot_perimage_anchoring.py
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
    SUBSET_CONFIG, load_clip, setup_logging, OUT_DIR,
)

BACKBONES = ["openai", "laion", "siglip2", "eva02", "dfn"]
TEMPERATURE = 100.0


def softmax_entropy(logits, temperature=TEMPERATURE):
    z = logits * temperature
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    p = e / e.sum(axis=1, keepdims=True)
    return -np.sum(p * np.log(p + 1e-12), axis=1), p


def _rankdata_average(a):
    """Vectorized average-rank (ties get mean of their positions). Mirrors
    scipy.stats.rankdata(method='average'). O(n log n), no Python loop."""
    arr = np.ravel(np.asarray(a, dtype=np.float64))
    sorter = np.argsort(arr, kind="quicksort")
    inv = np.empty(sorter.size, dtype=np.intp)
    inv[sorter] = np.arange(sorter.size)
    arr_sorted = arr[sorter]
    obs = np.r_[True, arr_sorted[1:] != arr_sorted[:-1]]
    dense = obs.cumsum()[inv]
    count = np.r_[np.nonzero(obs)[0], len(obs)]
    return 0.5 * (count[dense] + count[dense - 1] + 1)


def auroc(score, label):
    """ROC-AUC via Mann-Whitney U with average ranks (tie-correct, vectorized).
    score: higher = more positive."""
    score = np.asarray(score, dtype=np.float64)
    label = np.asarray(label, dtype=bool)
    n1 = int(label.sum()); n0 = int(len(label) - n1)
    if n1 == 0 or n0 == 0:
        return float("nan")
    ranks = _rankdata_average(score)
    sum_pos = float(ranks[label].sum())
    return (sum_pos - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def aupr(score, label):
    """Area under precision-recall curve. score: higher = more positive."""
    score = np.asarray(score, dtype=np.float64)
    label = np.asarray(label, dtype=bool)
    order = np.argsort(-score)
    label_sorted = label[order]
    tp = np.cumsum(label_sorted)
    fp = np.cumsum(~label_sorted)
    precision = tp / (tp + fp + 1e-12)
    recall = tp / (max(label.sum(), 1))
    # prepend (0,1)
    precision = np.concatenate([[1.0], precision])
    recall = np.concatenate([[0.0], recall])
    # area by trapezoid on the recall axis (estimator applied uniformly to
    # every signal in every table; absolute AUPR is non-canonical vs the
    # step interpolation, comparisons are internally consistent)
    trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    area = trapz(precision, recall)
    return float(area)


def bootstrap_auroc_diff(scores_a, scores_b, label, n_boot, seed):
    """Coupled bootstrap CI of AUROC(scores_a) - AUROC(scores_b)."""
    rng = np.random.RandomState(seed)
    n = len(label)
    diffs = np.empty(n_boot, dtype=np.float32)
    for b in range(n_boot):
        idx = rng.randint(0, n, size=n)
        da = auroc(scores_a[idx], label[idx]) - auroc(scores_b[idx], label[idx])
        diffs[b] = da
    return {"mean": float(np.mean(diffs)),
            "ci_lo": float(np.percentile(diffs, 2.5)),
            "ci_hi": float(np.percentile(diffs, 97.5))}


def risk_coverage(scores_keep, correct, coverages):
    """For each coverage c (fraction kept), keep the top-c by score_keep,
    return accuracy on kept set. scores_keep: higher = keep."""
    order = np.argsort(-scores_keep)
    n = len(correct)
    out = {}
    for c in coverages:
        m = max(1, int(round(c * n)))
        kept = order[:m]
        out[f"{int(c*100)}pct"] = {"coverage": float(m / n),
                                     "accuracy": float(correct[kept].mean())}
    return out


def evaluate(subset, bootstrap_n, seed, log):
    log.info(f"=== {subset} ===")
    all_logits = []
    gt = None
    for bb in BACKBONES:
        clip = load_clip(bb, subset, log)
        if clip is None:
            return None
        logits, labels = clip
        if gt is None:
            gt = labels
        elif not np.array_equal(gt, labels):
            raise RuntimeError(f"label mismatch on {subset}")
        all_logits.append(logits)
    logits_stack = np.stack(all_logits, axis=0)  # (K, N, C)
    n = logits_stack.shape[1]
    num_classes = gt.shape[1]
    avg_logits = logits_stack.mean(axis=0)  # (N, C)

    # consensus correctness
    consensus_argmax = np.argmax(avg_logits, axis=1)
    correct = np.array([consensus_argmax[i] in np.where(gt[i] > 0)[0]
                        for i in range(n)], dtype=bool)

    # signals (orient so HIGHER = keep)
    ent_avg, _ = softmax_entropy(avg_logits)
    s_entropy_keep = -ent_avg                       # higher entropy -> lower keep
    sorted_avg = np.sort(avg_logits, axis=1)
    s_margin_keep = (sorted_avg[:, -1] - sorted_avg[:, -2])  # higher margin -> keep
    backbone_argmaxes = np.argmax(logits_stack, axis=2)  # (K, N)
    agree = (backbone_argmaxes == consensus_argmax[None, :]).mean(axis=0)
    s_agreement_keep = agree                        # higher agreement -> keep
    logit_var = logits_stack.var(axis=0).mean(axis=1)
    s_logitvar_keep = -logit_var                    # higher var -> lower keep

    signals = {
        "entropy": s_entropy_keep,
        "margin": s_margin_keep,
        "agreement": s_agreement_keep,
        "logitvar(neg)": s_logitvar_keep,
    }

    per_sig = {}
    for name, s in signals.items():
        per_sig[name] = {
            "auroc": auroc(s, correct),
            "aupr": aupr(s, correct),
            "risk_coverage": risk_coverage(s, correct, [0.5, 0.7, 0.9]),
        }

    # coupled bootstrap: agreement vs entropy (the core bet)
    core = bootstrap_auroc_diff(signals["agreement"], signals["entropy"],
                                correct, bootstrap_n, seed)
    # also agreement vs margin
    ag_vmarg = bootstrap_auroc_diff(signals["agreement"], signals["margin"],
                                    correct, bootstrap_n, seed)

    base_acc = float(correct.mean())
    log.info(f"  N={n} consensus acc={base_acc:.4f}")
    log.info("  AUROC: " + " ".join(f"{k}={v['auroc']:.4f}" for k, v in per_sig.items()))
    log.info(f"  core bet (agreement-entropy) AUROC diff={core['mean']:+.4f} "
             f"CI[{core['ci_lo']:+.4f},{core['ci_hi']:+.4f}]")

    return {
        "subset": subset, "n_images": int(n), "num_classes": num_classes,
        "consensus_accuracy": base_acc,
        "per_backbone_hit1": {
            bb: float(np.mean([np.argmax(logits_stack[k, i]) in np.where(gt[i] > 0)[0]
                               for i in range(n)]))
            for k, bb in enumerate(BACKBONES)
        },
        "signals": per_sig,
        "core_bet_agreement_minus_entropy": core,
        "agreement_minus_margin": ag_vmarg,
    }


def format_report(results, args):
    lines = []
    lines.append("=" * 82)
    lines.append("  Pilot — per-image anchoring (cross-backbone agreement) vs entropy")
    lines.append("  Topic #2 gate: does agreement predict CLIP correctness better than entropy?")
    lines.append("=" * 82)
    lines.append(f"  Bootstrap B={args.bootstrap_n} seed={args.seed}")
    lines.append("")
    for res in results:
        lines.append(f"── {res['subset']} | N={res['n_images']} | consensus acc="
                     f"{res['consensus_accuracy']*100:.2f}% ──")
        lines.append(f"  per-backbone Hit@1: " + " ".join(
            f"{bb}={res['per_backbone_hit1'][bb]*100:.1f}%"
            for bb in BACKBONES))
        lines.append(f"  {'signal':<14} {'AUROC':>7} {'AUPR':>7} | "
                     f"acc@50% acc@70% acc@90%")
        lines.append("  " + "-" * 60)
        for name in ["entropy", "margin", "agreement", "logitvar(neg)"]:
            s = res["signals"][name]
            rc = s["risk_coverage"]
            lines.append(f"  {name:<14} {s['auroc']:>7.4f} {s['aupr']:>7.4f} | "
                         f"{rc['50pct']['accuracy']*100:>5.2f}% "
                         f"{rc['70pct']['accuracy']*100:>5.2f}% "
                         f"{rc['90pct']['accuracy']*100:>5.2f}%")
        c = res["core_bet_agreement_minus_entropy"]
        gate_pass = (c["mean"] > 0 and c["ci_lo"] > 0)
        lines.append(f"  CORE BET: agreement - entropy AUROC = {c['mean']:+.4f} "
                     f"CI[{c['ci_lo']:+.4f},{c['ci_hi']:+.4f}] "
                     f"{'-> PASS (agreement better)' if gate_pass else '-> FAIL'}")
        lines.append("")
    lines.append("=" * 82)
    lines.append("  GATE: agreement AUROC > entropy AUROC on TeacherBehavior, CI excludes 0.")
    lines.append("  acc@c% = accuracy when keeping the top-c% highest-signal images.")
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
    setup_logging(ts, name="pilot_anchoring")
    log = logging.getLogger()
    log.info(f"Per-image anchoring pilot | args={vars(args)}")

    results = []
    for subset in args.subsets:
        r = evaluate(subset, args.bootstrap_n, args.seed, log)
        if r is not None:
            results.append(r)
    if not results:
        log.error("No results."); return 1

    txt = format_report(results, args)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "pilot_perimage_anchoring.json", "w") as f:
        json.dump({"args": vars(args), "timestamp": ts, "results": results}, f, indent=2)
    with open(OUT_DIR / "pilot_perimage_anchoring.txt", "w") as f:
        f.write(txt)
    print(); print(txt)
    return 0


if __name__ == "__main__":
    exit(main())
