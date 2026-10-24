"""
analysis/anchoring_cifar.py — Exp A replication on CIFAR-100
=============================================================
Replicates the main agreement-vs-entropy analysis on CIFAR-100 (100 classes,
10k test images, single-label) to test if agreement generalizes to fine-grained
recognition with many classes.

Signals (compute-matched, from k-backbone ensemble):
  - entropy, margin, agreement_top1
  - Bootstrap CIs on agreement-minus-entropy AUROC

Output: results/anchoring_cifar_report.{json,txt}

Reproducibility:
  python analysis/anchoring_cifar.py [--data_dir results/per_image/cifar100]
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pilot_perimage_anchoring import (
    BACKBONES, softmax_entropy, auroc, aupr, bootstrap_auroc_diff,
)
from phase0_oracle_analysis import setup_logging, OUT_DIR
from anchoring_main import risk_coverage_curve, aurc, _acc_at_nearest

DATA_DIR = OUT_DIR.parent / "results" / "per_image" / "cifar100"
SIGNALS_ORDER = ["entropy", "msp", "margin", "agreement_top1"]


def evaluate(data_dir, bootstrap_n, seed, log):
    log.info("=== cifar100 ===")

    all_logits = {}
    gt = None
    for bb in BACKBONES:
        lp = data_dir / f"clip_{bb}_cifar100_logits.npy"
        bp = data_dir / f"clip_{bb}_cifar100_labels.npy"
        if not lp.exists():
            log.warning(f"Missing {lp}")
            continue
        logits = np.load(lp).astype(np.float32)
        labels = np.load(bp).astype(np.float32)
        if gt is None:
            gt = labels
        elif not np.array_equal(gt, labels):
            raise RuntimeError("label mismatch")
        all_logits[bb] = logits

    missing = set(BACKBONES) - set(all_logits.keys())
    if missing:
        log.warning(f"Missing backbones: {missing}. Using {list(all_logits.keys())}")

    if len(all_logits) < 2:
        log.error("Need at least 2 backbones")
        return None

    n = len(gt)
    models_used = list(all_logits.keys())
    k = len(models_used)
    stack = np.stack([all_logits[b] for b in models_used], 0)
    avg = stack.mean(0)
    consensus_argmax = np.argmax(avg, 1)
    correct = (consensus_argmax == gt.astype(int)).astype(bool)

    # signals
    ent, p = softmax_entropy(avg)
    s_ent = -ent
    s_msp = p.max(axis=1)
    sorted_avg = np.sort(avg, axis=1)
    s_margin = sorted_avg[:, -1] - sorted_avg[:, -2]
    barg = np.argmax(stack, 2)
    s_agree = (barg == consensus_argmax[None, :]).mean(0)

    signals = {
        "entropy": s_ent,
        "msp": s_msp,
        "margin": s_margin,
        "agreement_top1": s_agree,
    }

    grid = np.linspace(0.05, 1.0, 96)
    per_sig = {}
    for name in SIGNALS_ORDER:
        sc = signals[name]
        cov, acc_arr, risk_arr = risk_coverage_curve(sc, correct, grid)
        acc_dict = _acc_at_nearest(cov, acc_arr, [0.50, 0.70, 0.90])
        per_sig[name] = {
            "auroc": float(auroc(sc, correct)),
            "aupr": float(aupr(sc, correct)),
            "aurc": float(aurc(risk_arr, cov)),
            "acc_at_50pct": float(acc_dict["50pct"]),
            "acc_at_70pct": float(acc_dict["70pct"]),
            "acc_at_90pct": float(acc_dict["90pct"]),
        }

    deltas = {}
    for name in list(signals.keys()):
        if name == "entropy":
            continue
        deltas[f"{name}_minus_entropy"] = bootstrap_auroc_diff(
            signals[name], signals["entropy"], correct, bootstrap_n, seed)

    log.info(f"  N={n} k={k} consensus_acc={correct.mean():.4f}")
    log.info(f"  AUROC: " + " ".join(f"{k}={per_sig[k]['auroc']:.4f}" for k in SIGNALS_ORDER))
    core = deltas.get("agreement_top1_minus_entropy", {})
    if core:
        log.info(f"  core (agree_top1-entropy)={core['mean']:+.4f} "
                 f"CI[{core['ci_lo']:+.4f},{core['ci_hi']:+.4f}]")

    return {
        "dataset": "cifar100",
        "n_images": int(n),
        "k_backbones": k,
        "backbones": models_used,
        "consensus_accuracy": float(correct.mean()),
        "signals": per_sig,
        "deltas_vs_entropy": deltas,
    }


def format_report(results, args):
    lines = []
    lines.append("=" * 86)
    lines.append("  Exp A (CIFAR-100) — Instance anchoring on fine-grained recognition")
    lines.append("=" * 86)
    lines.append(f"  Backbones: {results['backbones']}  Bootstrap B={args.bootstrap_n} seed={args.seed}")
    lines.append("")
    lines.append(f"── cifar100 | N={results['n_images']} | k={results['k_backbones']} "
                 f"backbones | consensus acc={results['consensus_accuracy']*100:.2f}% ──")
    lines.append(f"  {'signal':<20} {'AUROC':>7} {'AUPR':>7} {'AURC':>7} | "
                 f"acc@50 acc@70 acc@90")
    lines.append("  " + "-" * 66)
    for name in SIGNALS_ORDER:
        s = results["signals"][name]
        lines.append(f"  {name:<20} {s['auroc']:>7.4f} {s['aupr']:>7.4f} {s['aurc']:>7.4f} | "
                     f"{s['acc_at_50pct']*100:>5.2f}% "
                     f"{s['acc_at_70pct']*100:>5.2f}% "
                     f"{s['acc_at_90pct']*100:>5.2f}%")
    lines.append("  -- AUROC delta vs entropy --")
    for k, v in results["deltas_vs_entropy"].items():
        lines.append(f"    {k:<32} {v['mean']:+.4f} CI[{v['ci_lo']:+.4f},{v['ci_hi']:+.4f}]")
    lines.append("")
    lines.append("=" * 86)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=str(DATA_DIR))
    parser.add_argument("--bootstrap_n", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ts = time.strftime("%Y%m%d_%H%M%S")
    setup_logging(ts, name="anchoring_cifar")
    log = logging.getLogger()
    log.info(f"Exp A CIFAR-100 | args={vars(args)}")

    r = evaluate(Path(args.data_dir), args.bootstrap_n, args.seed, log)
    if r is None:
        log.error("No results."); return 1

    txt = format_report(r, args)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "anchoring_cifar_report.json", "w") as f:
        json.dump({"args": vars(args), "timestamp": ts, "results": r}, f, indent=2)
    with open(OUT_DIR / "anchoring_cifar_report.txt", "w") as f:
        f.write(txt)
    print(); print(txt)
    return 0


if __name__ == "__main__":
    exit(main())
