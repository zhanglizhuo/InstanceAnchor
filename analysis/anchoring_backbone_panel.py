"""
analysis/anchoring_backbone_panel.py — P5: k=5 vs k=8 backbone panel
====================================================================
Compares agreement and soft_agreement signals at k=5 (original paper
backbones) vs k=8 (original 5 + 3 new: ViT-H-14, ViT-B-32, ViT-B-16).
Tests whether soft_agreement's 16/20 win rate depends on the specific
5 backbones or generalizes to a larger panel.

Original 5: openai, laion, siglip2, eva02, dfn (all ViT-L scale)
New 3:      vith14 (ViT-H-14 laion2b), vitb32 (ViT-B-32 openai), vitb16 (ViT-B-16 laion2b)

Datasets: the 13 P4 meta datasets (where all 8 backbones have npy).
          Paper datasets (TB etc.) excluded — their raw images are not on 5090.

Output: results/anchoring_backbone_panel_report.{json,txt}
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase0_oracle_analysis import OUT_DIR, DATA_DIR  # noqa: E402
from pilot_perimage_anchoring import (  # noqa: E402
    softmax_entropy, auroc, aupr, bootstrap_auroc_diff,
)

META_DIR = DATA_DIR / "meta"
CIFAR_DIR = DATA_DIR / "cifar100"
T = 100.0

ORIG5 = ["openai", "laion", "siglip2", "eva02", "dfn"]
NEW3 = ["vith14", "vitb32", "vitb16"]
ALL8 = ORIG5 + NEW3

DATASETS_META = ["food101", "dtd", "cifar10", "tiny_imagenet", "stl10",
                 "eurosat", "imagenet_a", "imagenet_r",
                 "sun397", "fgvc_aircraft", "stanford_cars", "gtsrb"]


def load_stack(models, dataset):
    stack, gt = [], None
    for m in models:
        if dataset == "cifar100":
            lp = CIFAR_DIR / f"clip_{m}_cifar100_logits.npy"
            bp = CIFAR_DIR / f"clip_{m}_cifar100_labels.npy"
        elif dataset in ("teacher_behavior", "handrise_readwrite", "bow_turnhead",
                         "voc2007", "stanford40", "coco", "multithumos"):
            lp = DATA_DIR / f"clip_{m}_{dataset}_logits.npy"
            bp = DATA_DIR / f"clip_{m}_{dataset}_labels.npy"
        else:
            lp = META_DIR / f"clip_{m}_{dataset}_logits.npy"
            bp = META_DIR / f"clip_{m}_{dataset}_labels.npy"
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
    n = gt.shape[0]
    return np.array([cons[i] in np.where(gt[i] > 0)[0]
                     for i in range(n)], dtype=bool)


def soft_agreement(logits_stack, consensus):
    probs = np.stack([np.exp(z * T) / np.exp(z * T).sum(axis=1, keepdims=True)
                      for z in logits_stack], axis=0)
    return probs[:, np.arange(consensus.shape[0]), consensus].mean(axis=0)


def analyze_k(models, dataset, bootstrap_n, seed):
    loaded = load_stack(models, dataset)
    if loaded is None:
        return None
    logits_stack, gt = loaded
    K, N, C = logits_stack.shape
    avg = logits_stack.mean(axis=0)
    consensus = np.argmax(avg, axis=1)
    correct = correctness(consensus, gt)
    if correct.std() == 0:
        return None

    backbone_argmaxes = np.argmax(logits_stack, axis=2)
    hard = (backbone_argmaxes == consensus[None, :]).mean(axis=0)
    soft = soft_agreement(logits_stack, consensus)
    ent_avg, _ = softmax_entropy(avg)
    entropy = -ent_avg

    row = {"dataset": dataset, "k": K, "N": int(N), "C": int(C),
           "p": float(np.mean([
               (backbone_argmaxes[b] == gt.astype(int)).mean() if gt.ndim == 1 else
               np.mean([backbone_argmaxes[b][i] in np.where(gt[i] > 0)[0]
                        for i in range(N)])
               for b in range(K)])),
           "acc": float(correct.mean()),
           "entropy_auroc": auroc(entropy, correct),
           "hard_auroc": auroc(hard, correct),
           "soft_auroc": auroc(soft, correct),
           }
    d_hard = bootstrap_auroc_diff(hard, entropy, correct, bootstrap_n, seed)
    d_soft = bootstrap_auroc_diff(soft, entropy, correct, bootstrap_n, seed)
    row.update({
        "delta_hard": d_hard["mean"], "delta_hard_lo": d_hard["ci_lo"],
        "delta_hard_hi": d_hard["ci_hi"],
        "delta_soft": d_soft["mean"], "delta_soft_lo": d_soft["ci_lo"],
        "delta_soft_hi": d_soft["ci_hi"],
    })
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap_n", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ts = time.strftime("%Y%m%d_%H%M%S")
    results = {"k5": [], "k8": []}

    for ds in DATASETS_META:
        r5 = analyze_k(ORIG5, ds, args.bootstrap_n, args.seed)
        r8 = analyze_k(ALL8, ds, args.bootstrap_n, args.seed)
        if r5 and r8:
            results["k5"].append(r5)
            results["k8"].append(r8)
            print(f"{ds:16s} k5: hard={r5['delta_hard']:+.4f} soft={r5['delta_soft']:+.4f} | "
                  f"k8: hard={r8['delta_hard']:+.4f} soft={r8['delta_soft']:+.4f}", flush=True)

    def win_table(rows, prefix):
        w = t = l = 0
        for r in rows:
            lo, hi = r[f"{prefix}_lo"], r[f"{prefix}_hi"]
            if lo > 0: w += 1
            elif hi < 0: l += 1
            else: t += 1
        return {"win": w, "tie": t, "loss": l, "n": len(rows)}

    wt = {
        "k5_hard": win_table(results["k5"], "delta_hard"),
        "k5_soft": win_table(results["k5"], "delta_soft"),
        "k8_hard": win_table(results["k8"], "delta_hard"),
        "k8_soft": win_table(results["k8"], "delta_soft"),
    }

    # delta_soft k8 vs k5 per dataset
    soft_changes = []
    for r5, r8 in zip(results["k5"], results["k8"]):
        diff = r8["delta_soft"] - r5["delta_soft"]
        soft_changes.append({"dataset": r5["dataset"],
                             "k5_soft": r5["delta_soft"], "k8_soft": r8["delta_soft"],
                             "change": diff})

    lines = ["=" * 96, f"  P5 backbone panel: k=5 vs k=8 ({len(results['k5'])} meta datasets)", "=" * 96, ""]
    lines.append(f"  {'dataset':>16} | {'k5 hard':>9} {'k5 soft':>9} | {'k8 hard':>9} {'k8 soft':>9} | {'soft chg':>9}")
    lines.append("  " + "-" * 72)
    for r5, r8 in zip(results["k5"], results["k8"]):
        lines.append(f"  {r5['dataset']:>16} | {r5['delta_hard']:>+9.4f} {r5['delta_soft']:>+9.4f} | "
                     f"{r8['delta_hard']:>+9.4f} {r8['delta_soft']:>+9.4f} | "
                     f"{r8['delta_soft']-r5['delta_soft']:>+9.4f}")
    lines.append("")
    for tag, v in wt.items():
        lines.append(f"  {tag:10s}: win={v['win']:>2} tie={v['tie']:>2} loss={v['loss']:>2} (n={v['n']})")
    lines.append("")
    mean_change = float(np.mean([c["change"] for c in soft_changes])) if soft_changes else 0
    lines.append(f"  Mean soft_agreement change (k8 - k5): {mean_change:+.4f}")
    lines.append("=" * 96)
    txt = "\n".join(lines)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "anchoring_backbone_panel_report.json", "w") as f:
        json.dump({"args": vars(args), "timestamp": ts, "results": results,
                   "win_table": wt, "soft_changes": soft_changes}, f, indent=2)
    with open(OUT_DIR / "anchoring_backbone_panel_report.txt", "w") as f:
        f.write(txt)
    print(); print(txt)


if __name__ == "__main__":
    main()
