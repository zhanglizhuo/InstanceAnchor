"""
analysis/anchoring_softagreement_cape.py — soft_agreement CAPE robustness
========================================================================
Decision-gate follow-up #1. The soft_agreement signal (mean softmax mass
at the consensus class) beat ensemble entropy on 5/8 datasets (P3). Here
we test prompt robustness on TeacherBehavior: does soft_agreement still
beat entropy under CAPE-B/C — the two prompt sets where HARD agreement
flips sign (B -0.092, C -0.117)?

For each CAPE set (A/B/C): soft_agreement vs entropy AUROC delta
(coupled bootstrap B=10000) + AUPR, on the same logits as P2.

If soft_agreement stays positive across all three prompt sets, prompt
rephrasing is a hard-agreement-specific weakness, not a soft-agreement one.

Output: results/anchoring_softagreement_cape_report.{json,txt}

Reproducibility:
  python analysis/anchoring_softagreement_cape.py
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

SUBSET = "teacher_behavior"
T = 100.0


def load_prompt_set(cape, log):
    stack = []
    gt = None
    for bb in BACKBONES:
        if cape == "A":
            lp = DATA_DIR / f"clip_{bb}_{SUBSET}_logits.npy"
            bp = DATA_DIR / f"clip_{bb}_{SUBSET}_labels.npy"
        else:
            lp = DATA_DIR / f"clip_{bb}_{SUBSET}_logits_cape{cape.lower()}.npy"
            bp = DATA_DIR / f"clip_{bb}_{SUBSET}_labels_cape{cape.lower()}.npy"
        if not lp.exists():
            log.warning(f"missing {lp}")
            return None
        logits = np.load(lp).astype(np.float32)
        labels = np.load(bp).astype(np.float32)
        if gt is None:
            gt = labels
        elif not np.array_equal(gt, labels):
            raise RuntimeError(f"label mismatch for CAPE-{cape}")
        stack.append(logits)
    return np.stack(stack, axis=0), gt


def soft_agreement(logits_stack, consensus):
    """Mean softmax mass at the consensus class across backbones."""
    probs = np.stack([np.exp(z * T) / np.exp(z * T).sum(axis=1, keepdims=True)
                      for z in logits_stack], axis=0)  # (K, N, C)
    return probs[:, np.arange(consensus.shape[0]), consensus].mean(axis=0)


def evaluate(cape, bootstrap_n, seed, log):
    loaded = load_prompt_set(cape, log)
    if loaded is None:
        return None
    logits_stack, gt = loaded
    n = logits_stack.shape[1]
    avg = logits_stack.mean(axis=0)
    consensus = np.argmax(avg, axis=1)
    correct = np.array([consensus[i] in np.where(gt[i] > 0)[0]
                        for i in range(n)], dtype=bool)

    ent_avg, _ = softmax_entropy(avg)
    entropy = -ent_avg
    s_soft = soft_agreement(logits_stack, consensus)
    agreement = (np.argmax(logits_stack, axis=2) == consensus[None, :]).mean(axis=0)

    delta_soft = bootstrap_auroc_diff(s_soft, entropy, correct, bootstrap_n, seed)
    delta_hard = bootstrap_auroc_diff(agreement, entropy, correct, bootstrap_n, seed)
    out = {
        "cape": cape,
        "n": int(n),
        "consensus_acc": float(correct.mean()),
        "entropy_auroc": auroc(entropy, correct),
        "agreement_hard_auroc": auroc(agreement, correct),
        "soft_agreement_auroc": auroc(s_soft, correct),
        "soft_agreement_aupr": aupr(s_soft, correct),
        "entropy_aupr": aupr(entropy, correct),
        "delta_hard_mean": delta_hard["mean"],
        "delta_hard_ci_lo": delta_hard["ci_lo"],
        "delta_hard_ci_hi": delta_hard["ci_hi"],
        "delta_soft_mean": delta_soft["mean"],
        "delta_soft_ci_lo": delta_soft["ci_lo"],
        "delta_soft_ci_hi": delta_soft["ci_hi"],
        "spearman_soft_entropy": None,
    }
    from scipy.stats import spearmanr
    r, p = spearmanr(s_soft, entropy)
    out["spearman_soft_entropy"] = {"rho": float(r), "p": float(p)}
    log.info(f"CAPE-{cape}: acc={correct.mean():.3f} ent={out['entropy_auroc']:.3f} "
             f"hard={out['agreement_hard_auroc']:.3f} soft={out['soft_agreement_auroc']:.3f} "
             f"d_hard={delta_hard['mean']:+.4f} d_soft={delta_soft['mean']:+.4f} "
             f"CI[{delta_soft['ci_lo']:+.4f},{delta_soft['ci_hi']:+.4f}]")
    return out


def format_report(results):
    lines = []
    lines.append("=" * 84)
    lines.append("  soft_agreement prompt robustness (TeacherBehavior, CAPE-A/B/C)")
    lines.append("=" * 84)
    lines.append("")
    lines.append(f"    {'set':>3} | {'acc':>6} | {'ent':>7} | {'hard':>7} | {'soft':>7} | "
                 f"{'d_hard':>9} | {'d_soft':>9} | CI_soft")
    lines.append("    " + "-" * 72)
    for r in results:
        lines.append(f"    {r['cape']:>3} | {r['consensus_acc']*100:>5.1f}% | "
                     f"{r['entropy_auroc']:>7.3f} | {r['agreement_hard_auroc']:>7.3f} | "
                     f"{r['soft_agreement_auroc']:>7.3f} | {r['delta_hard_mean']:>+9.4f} | "
                     f"{r['delta_soft_mean']:>+9.4f} | "
                     f"[{r['delta_soft_ci_lo']:+.4f},{r['delta_soft_ci_hi']:+.4f}]")
    lines.append("")
    lines.append("  Question: does soft_agreement stay positive under CAPE-B/C where hard")
    lines.append("  agreement flips sign?")
    lines.append("=" * 84)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap_n", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ts = time.strftime("%Y%m%d_%H%M%S")
    setup_logging(ts, name="anchoring_softagreement_cape")
    log = logging.getLogger()

    results = []
    for cape in ("A", "B", "C"):
        r = evaluate(cape, args.bootstrap_n, args.seed, log)
        if r is None:
            log.error(f"CAPE-{cape} missing; abort.")
            return 1
        results.append(r)

    txt = format_report(results)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "anchoring_softagreement_cape_report.json", "w") as f:
        json.dump({"args": vars(args), "timestamp": ts, "results": results}, f, indent=2)
    with open(OUT_DIR / "anchoring_softagreement_cape_report.txt", "w") as f:
        f.write(txt)
    print(); print(txt)
    return 0


if __name__ == "__main__":
    exit(main())
