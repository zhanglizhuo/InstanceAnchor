"""
analysis/anchoring_main.py — Exp A: main result solidification
==============================================================
Solidifies the pilot finding (cross-backbone agreement beats ensemble entropy
for predicting per-image CLIP correctness on TeacherBehavior) into the paper's
main evidence. All on existing per-image .npy -- CPU, fast.

Deliverables:
  1. AUROC + AUPR for {entropy, margin, agreement_top1, agreement_top3_jac,
     agreement_rankcorr, logitvar} on all 3 subsets.
  2. Full risk-coverage curve (dense grid) + AURC per signal.
  3. Per-class AUROC breakdown on TeacherBehavior (which classes drive the gain).
  4. Coupled bootstrap 95% CI for every (variant - entropy) AUROC delta.

Metric target: consensus correctness (argmax of 5-backbone averaged logits in
GT). All signals are compute-matched (use all 5 backbones).

Output:
  results/anchoring_main_report.{json,txt}
  logs/anchoring_main_<timestamp>.log

Reproducibility:
  python analysis/anchoring_main.py
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase0_oracle_analysis import SUBSET_CONFIG, load_clip, setup_logging, OUT_DIR  # noqa: E402
from pilot_perimage_anchoring import (  # noqa: E402
    BACKBONES, TEMPERATURE, softmax_entropy, auroc, aupr,
    bootstrap_auroc_diff,
)

SIGNALS_ORDER = ["entropy", "msp", "margin", "agreement_top1", "agreement_weighted",
                 "agreement_top3_jac", "agreement_rankcorr", "logitvar"]


def topk_indices(arr, k):
    return np.argsort(-arr, axis=1)[:, :k]


def jaccard_top3_pairwise(logits_stack, consensus, k=3):
    """For each image, mean over backbones of Jaccard(backbone_top3, consensus_top3)."""
    K, N, C = logits_stack.shape
    cons_top = topk_indices(consensus, k)  # (N, k)
    cons_sets = [set(cons_top[i].tolist()) for i in range(N)]
    out = np.zeros(N, dtype=np.float32)
    for kk in range(K):
        bb_top = topk_indices(logits_stack[kk], k)
        for i in range(N):
            inter = len(set(bb_top[i].tolist()) & cons_sets[i])
            union = len(set(bb_top[i].tolist()) | cons_sets[i])
            out[i] += (inter / union) if union > 0 else 1.0
    out /= K
    return out  # higher = more agreement


def mean_pairwise_rankcorr(logits_stack):
    """Mean pairwise Spearman rank correlation across backbones, per image."""
    K, N, C = logits_stack.shape
    # ranks of each backbone's C logits per image
    ranks = np.empty_like(logits_stack)
    for kk in range(K):
        order = np.argsort(logits_stack[kk], axis=1)
        r = np.empty((N, C), dtype=np.float32)
        for i in range(N):
            r[i, order[i]] = np.arange(C)
        ranks[kk] = r
    # mean pairwise Spearman = 1 - 6*sum(d^2)/(C(C^2-1)) for each pair; average
    accum = np.zeros(N, dtype=np.float32)
    cnt = 0
    denom = C * (C * C - 1)
    for a in range(K):
        for b in range(a + 1, K):
            d2 = ((ranks[a] - ranks[b]) ** 2).sum(axis=1)
            accum += 1.0 - 6.0 * d2 / denom
            cnt += 1
    return accum / cnt  # higher = more agreement


def risk_coverage_curve(scores_keep, correct, grid):
    """Dense risk-coverage: for each coverage c in grid, keep top-c by score,
    return (coverage, accuracy, risk=1-accuracy)."""
    order = np.argsort(-scores_keep)
    n = len(correct)
    cov, acc, risk = [], [], []
    for c in grid:
        m = max(1, int(round(c * n)))
        kept = order[:m]
        a = float(correct[kept].mean())
        cov.append(m / n)
        acc.append(a)
        risk.append(1.0 - a)
    return np.array(cov), np.array(acc), np.array(risk)


def aurc(risk, cov):
    """Area under risk-coverage curve (trapezoid on coverage). Lower = better."""
    return float(np.trapezoid(risk, cov)) if hasattr(np, "trapezoid") else float(np.trapz(risk, cov))


def _acc_at_nearest(cov, acc, targets):
    """Snap each target coverage to the nearest grid point; return {pct: acc}."""
    cov = np.asarray(cov); acc = np.asarray(acc)
    out = {}
    for t in targets:
        i = int(np.argmin(np.abs(cov - t)))
        out[f"{int(round(t * 100))}pct"] = float(acc[i])
    return out


def _tie_aware_acc_at(signal, correct, targets, n_seeds=200, seed=42):
    """Tie-aware selective accuracy at target coverages (averaged over n_seeds
    random tie breaks). Agreement has only ~6 distinct values -> huge ties ->
    naive argsort injects arbitrary index-order noise."""
    rng = np.random.RandomState(seed)
    n = len(signal)
    out = {}
    for t in targets:
        m = max(1, int(round(t * n)))
        accs = np.empty(n_seeds)
        for s in range(n_seeds):
            jitter = rng.random(n)
            order = np.lexsort((jitter, -signal))
            accs[s] = correct[order[:m]].mean()
        out[f"{int(round(t * 100))}pct"] = float(accs.mean())
    return out


def compute_signals(logits_stack, avg_logits):
    n = avg_logits.shape[0]
    ent_avg, p_avg = softmax_entropy(avg_logits)
    s = {}
    s["entropy"] = -ent_avg
    s["msp"] = p_avg.max(axis=1)  # max softmax probability (Jaeger2023 baseline)
    sorted_avg = np.sort(avg_logits, axis=1)
    s["margin"] = sorted_avg[:, -1] - sorted_avg[:, -2]
    backbone_argmaxes = np.argmax(logits_stack, axis=2)  # (K, N)
    consensus_argmax = np.argmax(avg_logits, axis=1)
    s["agreement_top1"] = (backbone_argmaxes == consensus_argmax[None, :]).mean(axis=0)

    # Weighted agreement: per-backbone Hit@1 weights
    K, N, C = logits_stack.shape
    gt = None  # We'll pass gt separately
    # Weights computed below in evaluate()

    s["agreement_top3_jac"] = jaccard_top3_pairwise(logits_stack, avg_logits, k=3)
    s["agreement_rankcorr"] = mean_pairwise_rankcorr(logits_stack)
    s["logitvar"] = -logits_stack.var(axis=0).mean(axis=1)
    return s


def compute_weighted_agreement(logits_stack, avg_logits, labels):
    """Weighted agreement: each backbone's vote is weighted by its Hit@1.
    Weights are computed from the same validation set (no leakage concern
    since agreement is compared against other same-data signals)."""
    K, N, C = logits_stack.shape
    consensus_argmax = np.argmax(avg_logits, axis=1)
    backbone_argmaxes = np.argmax(logits_stack, axis=2)

    # Compute per-backbone Hit@1
    weights = np.zeros(K)
    for k in range(K):
        pred_k = backbone_argmaxes[k]
        if labels.ndim == 2:
            correct_k = np.array([pred_k[i] in np.where(labels[i] > 0)[0]
                                   for i in range(N)], dtype=bool)
        else:
            correct_k = (pred_k == labels.astype(int))
        weights[k] = correct_k.mean()

    # Weighted vote fraction
    weighted_votes = np.zeros(N)
    for k in range(K):
        weighted_votes += weights[k] * (backbone_argmaxes[k] == consensus_argmax).astype(float)
    return weighted_votes / weights.sum()


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
    logits_stack = np.stack(all_logits, axis=0)
    n = logits_stack.shape[1]
    num_classes = gt.shape[1]
    avg_logits = logits_stack.mean(axis=0)
    consensus_argmax = np.argmax(avg_logits, axis=1)
    correct = np.array([consensus_argmax[i] in np.where(gt[i] > 0)[0]
                        for i in range(n)], dtype=bool)
    primary = np.array([np.where(gt[i] > 0)[0][0] for i in range(n)])

    signals = compute_signals(logits_stack, avg_logits)
    signals["agreement_weighted"] = compute_weighted_agreement(logits_stack, avg_logits, gt)
    grid = np.arange(0.05, 1.001, 0.05)

    per_sig = {}
    for name in SIGNALS_ORDER:
        sc = signals[name]
        cov, acc, risk = risk_coverage_curve(sc, correct, grid)
        per_sig[name] = {
            "auroc": auroc(sc, correct),
            "aupr": aupr(sc, correct),
            "aurc": aurc(risk, cov),
            "risk_coverage": [{"coverage": float(c), "accuracy": float(a), "risk": float(r)}
                              for c, a, r in zip(cov, acc, risk)],
            "acc_at": _acc_at_nearest(cov, acc, [0.5, 0.7, 0.9]),
            "acc_at_tieaware": _tie_aware_acc_at(sc, correct, [0.5, 0.7, 0.9]),
        }

    # coupled bootstrap: each variant vs entropy
    deltas = {}
    for name in SIGNALS_ORDER:
        if name == "entropy":
            continue
        deltas[f"{name}_minus_entropy"] = bootstrap_auroc_diff(
            signals[name], signals["entropy"], correct, bootstrap_n, seed)

    # per-class AUROC breakdown (TB only, multilabel)
    per_class = None
    if SUBSET_CONFIG[subset]["multilabel"]:
        per_class = {}
        classes = SUBSET_CONFIG[subset]["classes"]
        for c in range(num_classes):
            mask = primary == c
            if mask.sum() < 10:
                continue
            entry = {
                "n": int(mask.sum()),
                "consensus_acc": float(correct[mask].mean()),
            }
            if correct[mask].std() == 0:
                entry["note"] = "degenerate (0% consensus acc)"
                entry["entropy_auroc"] = None
                entry["agreement_auroc"] = None
                entry["margin_auroc"] = None
            else:
                entry["entropy_auroc"] = auroc(signals["entropy"][mask], correct[mask])
                entry["agreement_auroc"] = auroc(signals["agreement_top1"][mask], correct[mask])
                entry["margin_auroc"] = auroc(signals["margin"][mask], correct[mask])
            per_class[classes[c]] = entry

    log.info(f"  N={n} consensus_acc={correct.mean():.4f}")
    log.info("  AUROC: " + " ".join(f"{k}={per_sig[k]['auroc']:.4f}" for k in SIGNALS_ORDER))
    log.info("  AURC:  " + " ".join(f"{k}={per_sig[k]['aurc']:.4f}" for k in SIGNALS_ORDER))
    log.info(f"  core (agree_top1-entropy)={deltas['agreement_top1_minus_entropy']['mean']:+.4f} "
             f"CI[{deltas['agreement_top1_minus_entropy']['ci_lo']:+.4f},"
             f"{deltas['agreement_top1_minus_entropy']['ci_hi']:+.4f}]")

    return {
        "subset": subset, "n_images": int(n), "num_classes": num_classes,
        "consensus_accuracy": float(correct.mean()),
        "signals": per_sig, "deltas_vs_entropy": deltas,
        "per_class": per_class,
    }


def format_report(results, args):
    lines = []
    lines.append("=" * 86)
    lines.append("  Exp A — Instance anchoring main result (cross-backbone agreement vs confidence)")
    lines.append("=" * 86)
    lines.append(f"  Bootstrap B={args.bootstrap_n} seed={args.seed}  (all signals compute-matched, 5 backbones)")
    lines.append("")
    for res in results:
        lines.append(f"── {res['subset']} | N={res['n_images']} | consensus acc="
                     f"{res['consensus_accuracy']*100:.2f}% ──")
        lines.append(f"  {'signal':<20} {'AUROC':>7} {'AUPR':>7} {'AURC':>7} | "
                     f"acc@50 acc@70 acc@90")
        lines.append("  " + "-" * 66)
        for name in SIGNALS_ORDER:
            s = res["signals"][name]
            a50 = s["acc_at"].get("50pct", float("nan"))
            a70 = s["acc_at"].get("70pct", float("nan"))
            a90 = s["acc_at"].get("90pct", float("nan"))
            lines.append(f"  {name:<20} {s['auroc']:>7.4f} {s['aupr']:>7.4f} {s['aurc']:>7.4f} | "
                         f"{a50*100:>5.2f}% {a70*100:>5.2f}% {a90*100:>5.2f}%")
        lines.append("  -- AUROC delta vs entropy (coupled bootstrap, + = better than entropy) --")
        for k, v in res["deltas_vs_entropy"].items():
            lines.append(f"    {k:<32} {v['mean']:+.4f} CI[{v['ci_lo']:+.4f},{v['ci_hi']:+.4f}]")
        if res.get("per_class"):
            lines.append("  -- per-class AUROC (TeacherBehavior) --")
            lines.append(f"    {'class':<22} {'N':>5} {'acc':>6} {'entAUROC':>9} "
                         f"{'agrAUROC':>9} {'mrgAUROC':>9}")
            for cls, d in res["per_class"].items():
                if d.get("note"):
                    lines.append(f"    {cls:<22} {d['n']:>5} {d['consensus_acc']*100:>5.2f}% "
                                 f"  ({d['note']})")
                else:
                    lines.append(f"    {cls:<22} {d['n']:>5} {d['consensus_acc']*100:>5.2f}% "
                                 f"{d['entropy_auroc']:>9.4f} {d['agreement_auroc']:>9.4f} "
                                 f"{d['margin_auroc']:>9.4f}")
        lines.append("")
    lines.append("=" * 86)
    lines.append("  Reading: AUROC/AUPR higher=better; AURC lower=better. delta CI excludes 0 => significant.")
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
    setup_logging(ts, name="anchoring_main")
    log = logging.getLogger()
    log.info(f"Exp A anchoring_main | args={vars(args)}")

    results = []
    for subset in args.subsets:
        r = evaluate(subset, args.bootstrap_n, args.seed, log)
        if r is not None:
            results.append(r)
    if not results:
        log.error("No results."); return 1

    txt = format_report(results, args)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "anchoring_main_report.json", "w") as f:
        json.dump({"args": vars(args), "timestamp": ts, "results": results}, f, indent=2)
    with open(OUT_DIR / "anchoring_main_report.txt", "w") as f:
        f.write(txt)
    print(); print(txt)
    return 0


if __name__ == "__main__":
    exit(main())
