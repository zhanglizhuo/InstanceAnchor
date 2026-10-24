"""
analysis/anchoring_v2.py — Exp A replication on VOC2007 & Stanford40
====================================================================
Replicates the main Exp A analysis on VOC2007 (20-class object detection)
and Stanford40 (40-class action recognition) to test if cross-backbone
agreement generalizes beyond SCB5.

Datasets: single-label, 5 backbones (siglip2 backfilled 2026-07-25).

Signals (compute-matched, all from the 5-backbone ensemble):
  - entropy
  - margin
  - agreement_top1 (fraction of backbones matching consensus)
  - agreement_top3_jac
  - agreement_rankcorr

Target: consensus correctness (argmax of 4-backbone averaged logits in GT).

Output:
  results/anchoring_v2_report.{json,txt}

Reproducibility:
  python analysis/anchoring_v2.py
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
    softmax_entropy, auroc, aupr, bootstrap_auroc_diff,
)
from anchoring_main import topk_indices, risk_coverage_curve, aurc, _acc_at_nearest

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "results" / "per_image"
OUT_DIR = REPO_ROOT / "results"
LOG_DIR = REPO_ROOT / "logs"

V2_BACKBONES = ["openai", "laion", "siglip2", "eva02", "dfn"]

DATASETS = {
    "voc2007": {"classes": 20, "name": "VOC2007"},
    "stanford40": {"classes": 40, "name": "Stanford40"},
    "coco": {"classes": 80, "name": "COCO"},
}

SIGNALS_ORDER = ["entropy", "msp", "margin", "agreement_top1",
                 "agreement_top3_jac", "agreement_rankcorr"]


def setup_logging(ts, name="anchoring_v2"):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s [%(levelname)s] %(message)s"
    logging.basicConfig(
        level=logging.INFO, format=fmt,
        handlers=[logging.FileHandler(LOG_DIR / f"{name}_{ts}.log"),
                  logging.StreamHandler()],
    )


def load_v2_clip(model, dataset):
    logits_path = DATA_DIR / f"clip_{model}_{dataset}_logits.npy"
    labels_path = DATA_DIR / f"clip_{model}_{dataset}_labels.npy"
    if not logits_path.exists():
        return None
    return np.load(str(logits_path)), np.load(str(labels_path))


def compute_signals(logits_stack, avg_logits):
    n = avg_logits.shape[0]
    ent_avg, p_avg = softmax_entropy(avg_logits)
    s = {}
    s["entropy"] = -ent_avg
    s["msp"] = p_avg.max(axis=1)
    sorted_avg = np.sort(avg_logits, axis=1)
    s["margin"] = sorted_avg[:, -1] - sorted_avg[:, -2]
    backbone_argmaxes = np.argmax(logits_stack, axis=2)
    consensus_argmax = np.argmax(avg_logits, axis=1)
    s["agreement_top1"] = (backbone_argmaxes == consensus_argmax[None, :]).mean(axis=0)
    s["agreement_top3_jac"] = jaccard_top3_pairwise(logits_stack, avg_logits, k=3)
    s["agreement_rankcorr"] = mean_pairwise_rankcorr(logits_stack)
    return s


def jaccard_top3_pairwise(logits_stack, consensus, k=3):
    K, N, C = logits_stack.shape
    cons_top = topk_indices(consensus, k)
    cons_sets = [set(cons_top[i].tolist()) for i in range(N)]
    out = np.zeros(N, dtype=np.float32)
    for kk in range(K):
        bb_top = topk_indices(logits_stack[kk], k)
        for i in range(N):
            inter = len(set(bb_top[i].tolist()) & cons_sets[i])
            union = len(set(bb_top[i].tolist()) | cons_sets[i])
            out[i] += (inter / union) if union > 0 else 1.0
    out /= K
    return out


def mean_pairwise_rankcorr(logits_stack):
    K, N, C = logits_stack.shape
    ranks = np.empty_like(logits_stack)
    for kk in range(K):
        order = np.argsort(logits_stack[kk], axis=1)
        r = np.empty((N, C), dtype=np.float32)
        for i in range(N):
            r[i, order[i]] = np.arange(C)
        ranks[kk] = r
    accum = np.zeros(N, dtype=np.float32)
    cnt = 0
    denom = C * (C * C - 1)
    for a in range(K):
        for b in range(a + 1, K):
            d2 = ((ranks[a] - ranks[b]) ** 2).sum(axis=1)
            accum += 1.0 - 6.0 * d2 / denom
            cnt += 1
    return accum / cnt


def evaluate(dataset, bootstrap_n, seed, log):
    log.info(f"\n=== {dataset} ===")
    all_logits = []
    gt = None
    for bb in V2_BACKBONES:
        clip = load_v2_clip(bb, dataset)
        if clip is None:
            log.warning(f"  [SKIP] {bb}/{dataset} missing"); return None
        logits, labels = clip
        if gt is None:
            gt = labels
        elif not np.array_equal(gt, labels):
            raise RuntimeError(f"label mismatch on {dataset}")
        all_logits.append(logits)
    logits_stack = np.stack(all_logits, axis=0)
    n = logits_stack.shape[1]
    num_classes = gt.shape[1]
    avg_logits = logits_stack.mean(axis=0)
    consensus_argmax = np.argmax(avg_logits, axis=1)
    if dataset in ("voc2007", "coco"):
        correct = np.array([consensus_argmax[i] in np.where(gt[i] > 0)[0]
                           for i in range(n)], dtype=bool)
    else:
        correct = consensus_argmax == gt.argmax(1)
    log.info(f"  N={n} consensus_acc={correct.mean():.4f}")

    signals = compute_signals(logits_stack, avg_logits)
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
        }

    deltas = {}
    for name in SIGNALS_ORDER:
        if name == "entropy":
            continue
        deltas[f"{name}_minus_entropy"] = bootstrap_auroc_diff(
            signals[name], signals["entropy"], correct, bootstrap_n, seed)

    log.info("  AUROC: " + " ".join(f"{k}={per_sig[k]['auroc']:.4f}" for k in SIGNALS_ORDER))
    log.info(f"  core (agree_top1-entropy)={deltas['agreement_top1_minus_entropy']['mean']:+.4f} "
             f"CI[{deltas['agreement_top1_minus_entropy']['ci_lo']:+.4f},"
             f"{deltas['agreement_top1_minus_entropy']['ci_hi']:+.4f}]")

    return {
        "dataset": dataset, "n_images": int(n), "num_classes": num_classes,
        "consensus_accuracy": float(correct.mean()),
        "n_backbones": len(V2_BACKBONES),
        "signals": per_sig, "deltas_vs_entropy": deltas,
    }


def format_report(results, args):
    lines = []
    lines.append("=" * 86)
    lines.append("  Exp A (V2) — Instance anchoring on VOC2007 & Stanford40")
    lines.append("=" * 86)
    lines.append(f"  Backbones: {V2_BACKBONES}  Bootstrap B={args.bootstrap_n} seed={args.seed}")
    lines.append("")
    for res in results:
        lines.append(f"── {res['dataset']} | N={res['n_images']} | "
                     f"{res['n_backbones']} backbones | consensus acc="
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
        lines.append("  -- AUROC delta vs entropy --")
        for k, v in res["deltas_vs_entropy"].items():
            lines.append(f"    {k:<32} {v['mean']:+.4f} CI[{v['ci_lo']:+.4f},{v['ci_hi']:+.4f}]")
        lines.append("")
    lines.append("=" * 86)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["voc2007", "stanford40", "coco"])
    parser.add_argument("--bootstrap_n", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ts = time.strftime("%Y%m%d_%H%M%S")
    setup_logging(ts)
    log = logging.getLogger()
    log.info(f"Exp A V2 anchoring_v2 | args={vars(args)}")

    results = []
    for ds in args.datasets:
        r = evaluate(ds, args.bootstrap_n, args.seed, log)
        if r is not None:
            results.append(r)
    if not results:
        log.error("No results."); return 1

    txt = format_report(results, args)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "anchoring_v2_report.json", "w") as f:
        json.dump({"args": vars(args), "timestamp": ts, "results": results}, f, indent=2)
    with open(OUT_DIR / "anchoring_v2_report.txt", "w") as f:
        f.write(txt)
    print(); print(txt)
    return 0


if __name__ == "__main__":
    exit(main())
