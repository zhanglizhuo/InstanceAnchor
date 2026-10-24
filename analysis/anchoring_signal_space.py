"""
analysis/anchoring_signal_space.py — P3: signal-space expansion
================================================================
Elevation-plan P3. Attacks hard-flaw #5 (trivial signal vs trivial baselines).
Evaluates principled cross-backbone divergence signals against ensemble
entropy on all 8 datasets, compute-matched (5 backbones, same logits).

Signals (high = keep):
  - entropy          baseline (avg-logit softmax entropy, negative)
  - agreement_top1   baseline (vote fraction, from main definitions)
  - logitvar_cons    -variance of consensus-class logits across backbones
  - kl_pairwise      -mean pairwise KL of backbones' softmax distributions
  - soft_agreement   mean softmax mass at the consensus class
  - topk_soft_ovl    rank-weighted soft overlap of top-5 sets vs consensus
  - rank_consensus   -mean rank of consensus class in backbones' logits
  - cos_dispersion   -(1 - mean pairwise cosine of backbone logit vectors)

For each dataset: AUROC/AUPR vs consensus error; delta vs entropy with
coupled bootstrap (B=10000). Count how many datasets each signal wins.

Output: results/anchoring_signal_space_report.{json,txt}

Reproducibility:
  python analysis/anchoring_signal_space.py [--bootstrap_n 10000]
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase0_oracle_analysis import setup_logging, OUT_DIR, DATA_DIR  # noqa: E402
from pilot_perimage_anchoring import (  # noqa: E402
    BACKBONES, softmax_entropy, auroc, aupr, bootstrap_auroc_diff,
)

CIFAR_DIR = DATA_DIR / "cifar100"
DATASETS = ["teacher_behavior", "handrise_readwrite", "bow_turnhead",
            "voc2007", "stanford40", "coco", "multithumos", "cifar100"]
NEW_SIGNALS = ["logitvar_cons", "kl_pairwise", "soft_agreement",
               "topk_soft_ovl", "rank_consensus", "cos_dispersion"]


def load_dataset(dataset, log):
    stack = []
    gt = None
    for bb in BACKBONES:
        if dataset == "cifar100":
            lp, bp = CIFAR_DIR / f"clip_{bb}_cifar100_logits.npy", CIFAR_DIR / f"clip_{bb}_cifar100_labels.npy"
        else:
            lp, bp = DATA_DIR / f"clip_{bb}_{dataset}_logits.npy", DATA_DIR / f"clip_{bb}_{dataset}_labels.npy"
        if not lp.exists():
            log.warning(f"missing {lp}")
            return None
        logits = np.load(lp).astype(np.float32)
        labels = np.load(bp).astype(np.float32)
        if gt is None:
            gt = labels
        elif not np.array_equal(gt, labels):
            raise RuntimeError(f"label mismatch on {dataset}")
        stack.append(logits)
    return np.stack(stack, axis=0), gt


def correctness(gt, consensus):
    if gt.ndim == 1:
        return consensus == gt.astype(int)
    n = gt.shape[0]
    return np.array([consensus[i] in np.where(gt[i] > 0)[0]
                     for i in range(n)], dtype=bool)


def compute_new_signals(logits_stack, avg_logits):
    """Principled divergence signals; all high = keep."""
    K, N, C = logits_stack.shape
    _, p_avg = softmax_entropy(avg_logits)
    backbone_argmaxes = np.argmax(logits_stack, axis=2)
    consensus = np.argmax(avg_logits, axis=1)
    s = {}

    # 1. variance of the consensus-class logit across backbones
    cons_logits = logits_stack[np.arange(K)[:, None], np.arange(N)[None, :],
                               consensus[None, :]]
    s["logitvar_cons"] = -cons_logits.var(axis=0)

    # 2. mean pairwise KL between backbone softmax distributions
    T = 100.0
    probs = np.stack([np.exp(z * T) / np.exp(z * T).sum(axis=1, keepdims=True)
                      for z in logits_stack], axis=0)  # (K, N, C)
    eps = 1e-12
    kl = np.zeros((K, K, N))
    for a in range(K):
        for b in range(K):
            if a == b:
                continue
            kl[a, b] = (probs[a] * np.log((probs[a] + eps) / (probs[b] + eps))).sum(axis=1)
    s["kl_pairwise"] = -kl.sum(axis=(0, 1)) / (K * (K - 1))

    # 3. soft agreement: mean softmax mass at the consensus class
    s["soft_agreement"] = probs[:, np.arange(N), consensus].mean(axis=0)

    # 4. rank-weighted soft overlap of top-5 sets with consensus top-5
    k = min(5, C)
    cons_soft = np.zeros((N, C))
    w = 1.0 / (1 + np.arange(k))
    topk = np.argsort(-avg_logits, axis=1)[:, :k]
    cons_soft[np.arange(N)[:, None], topk] = w[None, :]
    b_soft = np.zeros((K, N, C))
    topk_b = np.argsort(-logits_stack, axis=2)[:, :, :k]
    b_soft[np.arange(K)[:, None, None], np.arange(N)[None, :, None], topk_b] = w[None, None, :]
    s["topk_soft_ovl"] = (b_soft * cons_soft[None, :, :]).sum(axis=2).mean(axis=0)

    # 5. mean rank of the consensus class in each backbone's logits
    rank_of_cons = np.empty((K, N))
    for kk in range(K):
        rank_of_cons[kk] = ranks_k = np.argsort(np.argsort(-logits_stack[kk], axis=1),
                                                axis=1)[np.arange(N), consensus]
    s["rank_consensus"] = -rank_of_cons.mean(axis=0)

    # 6. cosine dispersion: 1 - mean pairwise cosine of backbone logit vectors
    norms = np.linalg.norm(logits_stack, axis=2, keepdims=True) + 1e-9
    unit = logits_stack / norms
    sims = np.einsum("knc,mnc->kmn", unit, unit)
    mask = ~np.eye(K, dtype=bool)
    s["cos_dispersion"] = -(1.0 - sims[mask].reshape(K, K - 1, N).mean(axis=1).mean(axis=0))

    return s


def evaluate(dataset, bootstrap_n, seed, log):
    loaded = load_dataset(dataset, log)
    if loaded is None:
        return None
    logits_stack, gt = loaded
    avg_logits = logits_stack.mean(axis=0)
    consensus = np.argmax(avg_logits, axis=1)
    correct = correctness(gt, consensus)

    ent_avg, _ = softmax_entropy(avg_logits)
    entropy = -ent_avg
    agreement = (np.argmax(logits_stack, axis=2) == consensus[None, :]).mean(axis=0)
    new_signals = compute_new_signals(logits_stack, avg_logits)

    out = {"dataset": dataset, "n": int(len(correct)),
           "consensus_acc": float(correct.mean()),
           "baselines": {
               "entropy": {"auroc": auroc(entropy, correct), "aupr": aupr(entropy, correct)},
               "agreement_top1": {"auroc": auroc(agreement, correct),
                                  "aupr": aupr(agreement, correct)},
           },
           "signals": {}}
    if correct.std() == 0:
        log.info(f"  {dataset}: degenerate target, skipped")
        return out
    for name in NEW_SIGNALS:
        sc = new_signals[name]
        delta = bootstrap_auroc_diff(sc, entropy, correct, bootstrap_n, seed)
        out["signals"][name] = {
            "auroc": auroc(sc, correct),
            "aupr": aupr(sc, correct),
            "delta_mean": delta["mean"],
            "delta_ci_lo": delta["ci_lo"],
            "delta_ci_hi": delta["ci_hi"],
        }
        log.info(f"  {dataset}/{name}: auroc={out['signals'][name]['auroc']:.4f} "
                 f"d={delta['mean']:+.4f} CI[{delta['ci_lo']:+.4f},{delta['ci_hi']:+.4f}]")
    return out


def format_report(results):
    lines = []
    lines.append("=" * 96)
    lines.append("  P3 — signal-space expansion (6 principled divergence signals vs entropy,")
    lines.append("       8 datasets, compute-matched 5 backbones, coupled bootstrap)")
    lines.append("=" * 96)
    lines.append("")
    hdr = f"    {'dataset':>18} | {'acc':>6} | {'ent':>7} | {'agr':>7} | " + \
          " | ".join(f"{n:>9}" for n in NEW_SIGNALS)
    lines.append(hdr)
    lines.append("    " + "-" * 92)
    for r in results:
        ds = r["dataset"]
        ent = r["baselines"]["entropy"]["auroc"]
        agr = r["baselines"]["agreement_top1"]["auroc"]
        cells = []
        for n in NEW_SIGNALS:
            v = r["signals"].get(n)
            cells.append(f"{v['auroc']:>9.3f}" if v else f"{'---':>9}")
        lines.append(f"    {ds:>18} | {r['consensus_acc']*100:>5.1f}% | "
                     f"{ent:>7.3f} | {agr:>7.3f} | " + " | ".join(cells))
    lines.append("")
    lines.append("  Delta vs entropy (signal - entropy AUROC), * = CI excludes 0:")
    for r in results:
        wins = []
        for n in NEW_SIGNALS:
            v = r["signals"].get(n)
            if v is None:
                continue
            star = "*" if v["delta_ci_lo"] > 0 else ("!" if v["delta_ci_hi"] < 0 else " ")
            wins.append(f"{n} {v['delta_mean']:+.3f}{star}")
        lines.append(f"    {r['dataset']:>18} | " + " | ".join(wins))
    lines.append("")
    win_count = {n: 0 for n in NEW_SIGNALS}
    for r in results:
        for n in NEW_SIGNALS:
            v = r["signals"].get(n)
            if v and v["delta_ci_lo"] > 0:
                win_count[n] += 1
    lines.append("  Significant wins (CI>0) per signal across 8 datasets:")
    for n, c in win_count.items():
        lines.append(f"    {n:>18} : {c}/8")
    lines.append("=" * 96)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap_n", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ts = time.strftime("%Y%m%d_%H%M%S")
    setup_logging(ts, name="anchoring_signal_space")
    log = logging.getLogger()

    results = []
    for ds in DATASETS:
        r = evaluate(ds, args.bootstrap_n, args.seed, log)
        if r is None:
            log.error(f"{ds} data missing; abort.")
            return 1
        results.append(r)

    txt = format_report(results)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "anchoring_signal_space_report.json", "w") as f:
        json.dump({"args": vars(args), "timestamp": ts, "results": results}, f, indent=2)
    with open(OUT_DIR / "anchoring_signal_space_report.txt", "w") as f:
        f.write(txt)
    print(); print(txt)
    return 0


if __name__ == "__main__":
    exit(main())
