"""
analysis/anchoring_temperature_27.py — R5: temperature sensitivity (27 datasets)
================================================================================
Reviewer (internal audit) finding: the paper's temperature robustness section
sweeps T only on TeacherBehavior (T in {1..1000}, delta +0.046..+0.205). The
27-dataset evaluation uses T=100 everywhere. This script checks whether the
soft-agreement vs entropy comparison is robust to the temperature choice
across all 27 datasets:

  for T in {10, 50, 100, 200}:
    soft_agreement(T)  = mean softmax mass at consensus class (T-scaled)
    entropy(T)         = ensemble entropy of avg logits (T-scaled)
    delta(T)           = AUROC(soft) - AUROC(entropy), coupled bootstrap B=1000
    W/T/L at T vs the T=100 reference win/loss classification (CI excludes 0)

Note: signals share the same consensus (argmax of avg logits, T-independent),
so only the temperature of the softmax/signal scaling changes.

Input : per-image logits/labels under results/per_image/ (same loading as
        anchoring_meta.py)
Output: results/anchoring_temperature_27_report.{json,txt}

Usage: python analysis/anchoring_temperature_27.py [--bootstrap_n 1000]
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
    BACKBONES, auroc, bootstrap_auroc_diff,
)

CIFAR_DIR = DATA_DIR / "cifar100"
META_DIR = DATA_DIR / "meta"
EXISTING = {
    "teacher_behavior": DATA_DIR, "handrise_readwrite": DATA_DIR,
    "bow_turnhead": DATA_DIR, "voc2007": DATA_DIR, "stanford40": DATA_DIR,
    "coco": DATA_DIR, "multithumos": DATA_DIR,
}
CIFAR = {"cifar100": CIFAR_DIR}
CIFARC = {"cifar10c": CIFAR_DIR.parent / "cifarc",
          "cifar100c": CIFAR_DIR.parent / "cifarc"}
META_KEYS = ["food101", "dtd", "cifar10", "tiny_imagenet", "stl10",
             "eurosat", "imagenet_a", "imagenet_r",
             "sun397", "fgvc_aircraft", "stanford_cars", "gtsrb",
             "imagenet_sketch", "imagenet_v2_matched",
             "imagenet_v2_threshold", "imagenet_v2_top", "imagenet_c",
             "cifar10c", "cifar100c"]
TEMPERATURES = [10.0, 50.0, 100.0, 200.0]


def load_stack(key):
    stack, gt = [], None
    for bb in BACKBONES:
        if key in EXISTING:
            lp = EXISTING[key] / f"clip_{bb}_{key}_logits.npy"
            bp = EXISTING[key] / f"clip_{bb}_{key}_labels.npy"
        elif key in CIFAR:
            lp = CIFAR[key] / f"clip_{bb}_{key}_logits.npy"
            bp = CIFAR[key] / f"clip_{bb}_{key}_labels.npy"
        elif key in CIFARC:
            lp = CIFARC[key] / f"clip_{bb}_{key}_logits.npy"
            bp = CIFARC[key] / f"clip_{bb}_{key}_labels.npy"
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
    n = gt.shape[0]
    return np.array([cons[i] in np.where(gt[i] > 0)[0]
                     for i in range(n)], dtype=bool)


def softmax_probs(z, t):
    z = z * t
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def signals_at_t(logits_stack, avg, consensus, t):
    probs = np.stack([softmax_probs(z, t) for z in logits_stack], axis=0)
    soft = probs[:, np.arange(len(consensus)), consensus].mean(axis=0)
    p_avg = softmax_probs(avg, t)
    # negative entropy: higher = more confident (mirrors pilot_perimage_anchoring)
    ent = (p_avg * np.log(p_avg + 1e-12)).sum(axis=1)
    return soft, ent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap_n", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ts = time.strftime("%Y%m%d_%H%M%S")
    setup_logging(ts, name="anchoring_temperature_27")
    log = logging.getLogger()

    keys = list(EXISTING.keys()) + list(CIFAR.keys()) + META_KEYS
    results = []
    for key in keys:
        loaded = load_stack(key)
        if loaded is None:
            continue
        logits_stack, gt = loaded
        K, N, C = logits_stack.shape
        avg = logits_stack.mean(axis=0)
        consensus = np.argmax(avg, axis=1)
        correct = correctness(consensus, gt)
        if correct.std() == 0:
            continue
        row = {"dataset": key, "N": int(N), "C": int(C),
               "acc": float(correct.mean())}
        for t in TEMPERATURES:
            soft, ent = signals_at_t(logits_stack, avg, consensus, t)
            d = bootstrap_auroc_diff(soft, ent, correct,
                                     args.bootstrap_n, args.seed)
            row[f"T{t:g}_soft_auroc"] = auroc(soft, correct)
            row[f"T{t:g}_ent_auroc"] = auroc(ent, correct)
            row[f"T{t:g}_delta_mean"] = d["mean"]
            row[f"T{t:g}_delta_lo"] = d["ci_lo"]
            row[f"T{t:g}_delta_hi"] = d["ci_hi"]
        results.append(row)
        log.info(f"  {key:18s} acc={row['acc']:.3f} "
                 + " ".join(f"T{t:g}={row[f'T{t:g}_delta_mean']:+.4f}"
                            for t in TEMPERATURES))

    def cls(lo, hi):
        if lo > 0:
            return "win"
        if hi < 0:
            return "loss"
        return "tie"

    ref = {r["dataset"]: cls(r["T100_delta_lo"], r["T100_delta_hi"])
           for r in results}
    win_table = {}
    flips = {}
    for t in TEMPERATURES:
        w = t_ = l = 0
        fl = []
        for r in results:
            c = cls(r[f"T{t:g}_delta_lo"], r[f"T{t:g}_delta_hi"])
            w += c == "win"; t_ += c == "tie"; l += c == "loss"
            if c != ref[r["dataset"]]:
                fl.append({"dataset": r["dataset"], "ref": ref[r["dataset"]],
                           "at": c})
        win_table[f"T{t:g}"] = {"win": w, "tie": t_, "loss": l, "n": len(results)}
        flips[f"T{t:g}"] = fl

    out = {"args": vars(args), "timestamp": ts,
           "temperatures": TEMPERATURES, "results": results,
           "win_table": win_table, "class_flips_vs_T100": flips}

    lines = ["=" * 96, f"  R5 temperature sensitivity ({len(results)} datasets)",
             "=" * 96, ""]
    lines.append(f"  {'T':>5} | {'W':>3} {'T':>3} {'L':>3} | mean delta "
                 f"(soft - entropy) | flips vs T=100")
    lines.append("  " + "-" * 76)
    for t in TEMPERATURES:
        wt = win_table[f"T{t:g}"]
        means = [r[f"T{t:g}_delta_mean"] for r in results]
        lines.append(f"  {t:>5.0f} | {wt['win']:>3} {wt['tie']:>3} "
                     f"{wt['loss']:>3} | {np.mean(means):+.4f} | "
                     f"{[f['dataset'] for f in flips[f'T{t:g}']]}")
    lines.append("")
    for r in sorted(results, key=lambda r: r["dataset"]):
        parts = [f"T{t:g}={r[f'T{t:g}_delta_mean']:+.4f}" for t in TEMPERATURES]
        lines.append(f"  {r['dataset']:>22s} acc={r['acc']:.3f}  "
                     + "  ".join(parts))
    lines.append("=" * 96)
    txt = "\n".join(lines)

    with open(OUT_DIR / "anchoring_temperature_27_report.json", "w") as f:
        json.dump(out, f, indent=2, default=lambda o: float(o))
    with open(OUT_DIR / "anchoring_temperature_27_report.txt", "w") as f:
        f.write(txt)
    print(); print(txt)
    return 0


if __name__ == "__main__":
    exit(main())
