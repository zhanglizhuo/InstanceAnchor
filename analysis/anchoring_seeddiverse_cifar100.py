"""
analysis/anchoring_seeddiverse_cifar100.py — 2x2 cell C analysis
=================================================================
Analyses the data-diverse / architecture-fixed CLIP ensemble
(5x ViT-B/32, distinct training data; experiments/seed_diverse_cifar100.py)
on CIFAR-100 with the SAME protocol as the main panel:

  consensus = argmax(mean cosine logits); hard agreement = vote fraction;
  kappa_emp = E[alpha|wrong]*(C-1)/(1-p); kappa_iid via simulate_iid_kappa
  (identical function/seed as the CLIP decomposition); signals at T=100
  (cosine-sim logits, CLIP protocol); coupled bootstrap B=10000, seed 42.

Purpose: isolate the TRAINING-DATA axis of the cross-family mechanism split
(sec:nonclip). If this arch-fixed, data-diverse cell reproduces the
arch-diverse CLIP cell's plurality dominance (CIFAR-100 share 72%), the
data-diversity axis survives an architecture-controlled contrast.

Reference rows (committed):
  arch-diverse CLIP CIFAR-100: p=0.711, kappa=161.7, share=72%
  (anchoring_meta_report.json / anchoring_kappa_meta_decomp_report.json)

Input : results/per_image/cifar100_seeddiverse/clip_{model}_cifar100_*.npy
Output: results/anchoring_seeddiverse_cifar100_report.{json,txt}

Usage: python analysis/anchoring_seeddiverse_cifar100.py [--bootstrap_n 10000]
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase0_oracle_analysis import setup_logging, OUT_DIR  # noqa: E402
from anchoring_kappa_iid_baseline import simulate_iid_kappa  # noqa: E402
from pilot_perimage_anchoring import (  # noqa: E402
    auroc, bootstrap_auroc_diff,
)

MODELS = ["openai", "laion400m", "laion2b", "datacomp", "common_pool"]
DATA_DIR = OUT_DIR / "per_image" / "cifar100_seeddiverse"
T = 100.0  # cosine-sim logits, CLIP protocol


def softmax(z):
    z = z * T
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap_n", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ts = time.strftime("%Y%m%d_%H%M%S")
    setup_logging(ts, name="anchoring_seeddiverse_cifar100")
    log = logging.getLogger()

    stack, gt = [], None
    for m in MODELS:
        lp = DATA_DIR / f"clip_{m}_cifar100_logits.npy"
        bp = DATA_DIR / f"clip_{m}_cifar100_labels.npy"
        if not lp.exists():
            log.error(f"missing {lp}; run experiments/seed_diverse_cifar100.py")
            return 1
        stack.append(np.load(lp).astype(np.float32))
        labels = np.load(bp).astype(np.int64)
        if gt is None:
            gt = labels
        elif not np.array_equal(gt, labels):
            raise RuntimeError(f"label mismatch for {m}")
    logits_stack = np.stack(stack, axis=0)
    K, N, C = logits_stack.shape
    avg = logits_stack.mean(axis=0)
    consensus = np.argmax(avg, axis=1)
    correct = consensus == gt

    per_bb_acc = [(np.argmax(logits_stack[b], 1) == gt).mean() for b in range(K)]
    p = float(np.mean(per_bb_acc))
    backbone_argmaxes = np.argmax(logits_stack, axis=2)
    hard = (backbone_argmaxes == consensus[None, :]).mean(axis=0)
    alpha_correct = float(hard[correct].mean()) if correct.sum() else float("nan")
    alpha_wrong = float(hard[~correct].mean()) if (~correct).sum() else float("nan")
    kappa_emp = alpha_wrong * (C - 1) / (1 - p) if (1 - p) > 0 else float("inf")
    p_star = kappa_emp / (C - 1 + kappa_emp) if np.isfinite(kappa_emp) else float("nan")
    delta_alpha = (p - kappa_emp * (1 - p) / (C - 1)) if np.isfinite(kappa_emp) else float("nan")
    sim = simulate_iid_kappa(per_bb_acc, K, C)
    plurality_share = sim["kappa_iid"] / kappa_emp if kappa_emp > 0 else float("nan")

    p_avg = softmax(avg)
    ent = (p_avg * np.log(p_avg + 1e-12)).sum(axis=1)  # negative entropy
    msp = p_avg.max(axis=1)
    probs = np.stack([softmax(z) for z in logits_stack], axis=0)
    soft = probs[:, np.arange(N), consensus].mean(axis=0)

    d_hard_ent = bootstrap_auroc_diff(hard, ent, correct,
                                      args.bootstrap_n, args.seed)
    d_soft_ent = bootstrap_auroc_diff(soft, ent, correct,
                                      args.bootstrap_n, args.seed)
    d_soft_msp = bootstrap_auroc_diff(soft, msp, correct,
                                      args.bootstrap_n, args.seed)

    row = {
        "cell": "C: data-diverse + arch-fixed (ViT-B/32 x5)",
        "dataset": "cifar100", "N": int(N), "C": int(C), "K": K,
        "models": MODELS,
        "per_backbone_accs": [float(a) for a in per_bb_acc], "p": p,
        "consensus_acc": float(correct.mean()),
        "E_alpha_correct": alpha_correct, "E_alpha_wrong": alpha_wrong,
        "kappa_empirical": float(kappa_emp),
        "kappa_iid_mc": float(sim["kappa_iid"]),
        "plurality_share": float(plurality_share),
        "p_star": float(p_star), "Delta_alpha": float(delta_alpha),
        "entropy_auroc": auroc(ent, correct), "msp_auroc": auroc(msp, correct),
        "hard_auroc": auroc(hard, correct), "soft_auroc": auroc(soft, correct),
        "delta_hard_vs_entropy": d_hard_ent,
        "delta_soft_vs_entropy": d_soft_ent,
        "delta_soft_vs_msp": d_soft_msp,
    }
    # reference: arch-diverse CLIP cell on the same dataset
    meta = json.load(open(OUT_DIR / "anchoring_meta_report.json"))
    ref_m = next(r for r in meta["results"] if r["dataset"] == "cifar100")
    decomp = json.load(open(OUT_DIR / "anchoring_kappa_meta_decomp_report.json"))
    ref_d = next(r for r in decomp["datasets"] if r["name"] == "cifar100")
    row["reference_arch_diverse_clip"] = {
        "p": ref_m["p"], "kappa_empirical": ref_m["kappa"],
        "plurality_share": ref_d["plurality_share"],
        "delta_hard_vs_entropy": ref_m["delta_hard_mean"],
        "delta_soft_vs_entropy": ref_m["delta_soft_mean"],
    }

    out = {"args": vars(args), "timestamp": ts, "results": [row]}
    lines = ["=" * 96, "  2x2 cell C: data-diverse + arch-fixed CLIP (CIFAR-100)",
             "=" * 96, ""]
    lines.append(f"  p={row['p']:.3f}  consensus={row['consensus_acc']:.3f}  "
                 f"E[a|correct]={row['E_alpha_correct']:.3f}  "
                 f"E[a|wrong]={row['E_alpha_wrong']:.3f}")
    lines.append(f"  kappa_emp={row['kappa_empirical']:.1f}  "
                 f"kappa_iid={row['kappa_iid_mc']:.1f}  "
                 f"plurality_share={row['plurality_share']*100:.0f}%  "
                 f"p*={row['p_star']:.3f}  Delta_alpha={row['Delta_alpha']:+.3f}")
    de = row["delta_hard_vs_entropy"]; ds = row["delta_soft_vs_entropy"]
    dm = row["delta_soft_vs_msp"]
    lines.append(f"  d_hard-ent={de['mean']:+.4f} CI[{de['ci_lo']:+.4f},{de['ci_hi']:+.4f}]  "
                 f"d_soft-ent={ds['mean']:+.4f} CI[{ds['ci_lo']:+.4f},{ds['ci_hi']:+.4f}]  "
                 f"d_soft-msp={dm['mean']:+.4f} CI[{dm['ci_lo']:+.4f},{dm['ci_hi']:+.4f}]")
    r = row["reference_arch_diverse_clip"]
    lines.append(f"  ref arch-diverse CLIP: p={r['p']:.3f} kappa={r['kappa_empirical']:.1f} "
                 f"share={r['plurality_share']*100:.0f}% "
                 f"d_hard={r['delta_hard_vs_entropy']:+.4f} "
                 f"d_soft={r['delta_soft_vs_entropy']:+.4f}")
    lines.append("=" * 96)
    txt = "\n".join(lines)

    with open(OUT_DIR / "anchoring_seeddiverse_cifar100_report.json", "w") as f:
        json.dump(out, f, indent=2, default=lambda o: float(o))
    with open(OUT_DIR / "anchoring_seeddiverse_cifar100_report.txt", "w") as f:
        f.write(txt)
    print(); print(txt)
    return 0


if __name__ == "__main__":
    exit(main())
