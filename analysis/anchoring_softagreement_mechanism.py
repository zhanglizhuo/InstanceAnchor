"""
analysis/anchoring_softagreement_mechanism.py — why soft agreement survives
===========================================================================
Decision-gate follow-up #2. Hypothesis: hard agreement saturates because its
variance collapses as consensus accuracy rises (all backbones agree ->
agreement=1 almost everywhere -> no discriminative scale), while soft
agreement (mean softmax mass at the consensus class) keeps a continuous
"degree of certainty" scale even when every backbone votes for the same
class.

Test on all 8 datasets (existing logits, no bootstrap):
  - std of hard agreement and std of soft agreement per dataset
  - logit-scale view: fraction of images where all 5 backbones agree on
    the argmax (hard-agreement ceiling occupancy) vs consensus accuracy
  - Spearman of signal std vs consensus accuracy: does std(hard) collapse
    with acc faster than std(soft)?

Output: results/anchoring_softagreement_mechanism_report.{json,txt}

Reproducibility:
  python analysis/anchoring_softagreement_mechanism.py
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
from pilot_perimage_anchoring import BACKBONES, softmax_entropy  # noqa: E402

CIFAR_DIR = DATA_DIR / "cifar100"
DATASETS = ["teacher_behavior", "handrise_readwrite", "bow_turnhead",
            "voc2007", "stanford40", "coco", "multithumos", "cifar100"]
T = 100.0


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
        stack.append(np.load(lp).astype(np.float32))
        labels = np.load(bp).astype(np.float32)
        if gt is None:
            gt = labels
    return np.stack(stack, axis=0), gt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ts = time.strftime("%Y%m%d_%H%M%S")
    setup_logging(ts, name="anchoring_softagreement_mechanism")
    log = logging.getLogger()

    rows = []
    for ds in DATASETS:
        loaded = load_dataset(ds, log)
        if loaded is None:
            log.error(f"{ds} missing; abort.")
            return 1
        logits_stack, gt = loaded
        n = logits_stack.shape[1]
        avg = logits_stack.mean(axis=0)
        consensus = np.argmax(avg, axis=1)
        if gt.ndim == 1:
            correct = consensus == gt.astype(int)
        else:
            correct = np.array([consensus[i] in np.where(gt[i] > 0)[0]
                                for i in range(n)], dtype=bool)
        acc = correct.mean()

        backbone_argmaxes = np.argmax(logits_stack, axis=2)
        hard_agree = (backbone_argmaxes == consensus[None, :]).mean(axis=0)
        probs = np.stack([np.exp(z * T) / np.exp(z * T).sum(axis=1, keepdims=True)
                          for z in logits_stack], axis=0)
        soft_agree = probs[:, np.arange(n), consensus].mean(axis=0)

        full_unanimity = (hard_agree == 1.0).mean()  # ceiling occupancy
        # among unanimous images, does soft agreement still vary?
        unanimous_soft_std = float(soft_agree[hard_agree == 1.0].std()) \
            if (hard_agree == 1.0).any() else 0.0
        rows.append({
            "dataset": ds, "n": int(n), "acc": float(acc),
            "std_hard": float(hard_agree.std()),
            "std_soft": float(soft_agree.std()),
            "frac_full_unanimity": float(full_unanimity),
            "std_soft_among_unanimous": unanimous_soft_std,
            "mean_soft": float(soft_agree.mean()),
        })
        log.info(f"{ds}: acc={acc:.3f} std_hard={hard_agree.std():.4f} "
                 f"std_soft={soft_agree.std():.4f} unanimity={full_unanimity:.3f} "
                 f"std_soft_unanimous={unanimous_soft_std:.4f}")

    from scipy.stats import spearmanr
    accs = np.array([r["acc"] for r in rows])
    std_hard = np.array([r["std_hard"] for r in rows])
    std_soft = np.array([r["std_soft"] for r in rows])
    r_h, p_h = spearmanr(accs, std_hard)
    r_s, p_s = spearmanr(accs, std_soft)
    r_u, p_u = spearmanr(accs, np.array([r["frac_full_unanimity"] for r in rows]))

    lines = []
    lines.append("=" * 80)
    lines.append("  soft-agreement survival mechanism (signal variance vs consensus acc)")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"    {'dataset':>18} | {'acc':>6} | {'std_hard':>9} | {'std_soft':>9} | "
                 f"{'unanim':>7} | {'std_soft_unanim':>16}")
    lines.append("    " + "-" * 72)
    for r in rows:
        lines.append(f"    {r['dataset']:>18} | {r['acc']*100:>5.1f}% | "
                     f"{r['std_hard']:>9.4f} | {r['std_soft']:>9.4f} | "
                     f"{r['frac_full_unanimity']:>7.3f} | {r['std_soft_among_unanimous']:>16.4f}")
    lines.append("")
    lines.append("  Spearman (acc vs signal std):")
    lines.append(f"    std_hard : rho={r_h:+.3f} p={p_h:.4f}")
    lines.append(f"    std_soft : rho={r_s:+.3f} p={p_s:.4f}")
    lines.append(f"    unanimity: rho={r_u:+.3f} p={p_u:.4f}")
    lines.append("")
    lines.append("  If std_hard collapses with acc while std_soft does not, the survival")
    lines.append("  mechanism is: hard votes lose their scale (unanimous -> constant), soft")
    lines.append("  mass keeps a continuous certainty scale even under unanimity.")
    lines.append("=" * 80)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "anchoring_softagreement_mechanism_report.json", "w") as f:
        json.dump({"args": vars(args), "timestamp": ts,
                   "results": rows,
                   "spearman": {"std_hard": {"rho": float(r_h), "p": float(p_h)},
                                "std_soft": {"rho": float(r_s), "p": float(p_s)},
                                "unanimity": {"rho": float(r_u), "p": float(p_u)}}},
                  f, indent=2)
    with open(OUT_DIR / "anchoring_softagreement_mechanism_report.txt", "w") as f:
        f.write("\n".join(lines))
    print(); print("\n".join(lines))
    return 0


if __name__ == "__main__":
    exit(main())
