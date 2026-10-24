"""
analysis/anchoring_baseline_deltas.py — reviewer-baseline deltas
================================================================
Computes coupled-bootstrap AUROC deltas of agreement_top1 against ALL
standard confidence baselines (entropy, MSP, margin) on all eight
datasets. The paper's headline compares agreement against ensemble
entropy; margin and MSP are stronger baselines on TeacherBehavior
(margin AUROC 0.662, MSP 0.652 vs entropy 0.639), so the headline
must also report the deltas against them.

Metric target: consensus correctness (argmax of 5-backbone averaged
logits in GT). All signals compute-matched (5 backbones).

Output:
  results/anchoring_baseline_deltas_report.json
  logs/anchoring_baseline_deltas_<timestamp>.log

Reproducibility:
  python analysis/anchoring_baseline_deltas.py
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "analysis"))
from pilot_perimage_anchoring import (  # noqa: E402
    BACKBONES, TEMPERATURE, softmax_entropy, auroc, bootstrap_auroc_diff,
)
from phase0_oracle_analysis import SUBSET_CONFIG, load_clip, setup_logging, OUT_DIR  # noqa: E402

DATA_DIR = REPO_ROOT / "results" / "per_image"

V2_BACKBONES = ["openai", "laion", "siglip2", "eva02", "dfn"]


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
    return s


def load_scb5(subset):
    all_logits, gt = [], None
    for bb in BACKBONES:
        clip = load_clip(bb, subset, logging.getLogger("x"))
        if clip is None:
            return None
        logits, labels = clip
        if gt is None:
            gt = labels
        elif not np.array_equal(gt, labels):
            raise RuntimeError(f"label mismatch on {subset}")
        all_logits.append(logits)
    return np.stack(all_logits, 0), gt


def load_v2(dataset):
    all_l, gt = [], None
    for m in V2_BACKBONES:
        base = DATA_DIR / ("cifar100" if dataset == "cifar100" else "")
        p = base / f"clip_{m}_{dataset}_logits.npy"
        lp = base / f"clip_{m}_{dataset}_labels.npy"
        if not p.exists():
            return None
        l = np.load(str(p)).astype(np.float32)
        lb = np.load(str(lp)).astype(np.float32)
        if gt is None:
            gt = lb
        elif not np.array_equal(gt, lb):
            raise RuntimeError(f"label mismatch for {m}")
        all_l.append(l)
    return np.stack(all_l, 0), gt


def evaluate(ds_cfg, args):
    name, loader, key = ds_cfg["name"], ds_cfg["loader"], ds_cfg["key"]
    loaded = loader()
    if loaded is None:
        return None
    stack, gt = loaded
    K, N, C = stack.shape
    if gt.ndim == 1:
        gt_bin = np.zeros((N, C), dtype=float)
        gt_bin[np.arange(N), gt.astype(int)] = 1.0
        gt = gt_bin
    avg = stack.mean(0)
    cons = np.argmax(avg, 1)
    correct = np.array([cons[i] in np.where(gt[i] > 0)[0] for i in range(N)], dtype=bool)

    s = compute_signals(stack, avg)
    out = {"dataset": name, "N": int(N), "C": int(C), "K": int(K),
           "consensus_accuracy": float(correct.mean()),
           "aurocs": {k: float(auroc(s[k], correct)) for k in s},
           "deltas_agreement_minus": {}}
    for base in ["entropy", "msp", "margin"]:
        ci = bootstrap_auroc_diff(s["agreement_top1"], s[base], correct,
                                  args.bootstrap_n, args.seed)
        out["deltas_agreement_minus"][base] = {
            "mean": float(ci["mean"]), "ci_lo": float(ci["ci_lo"]),
            "ci_hi": float(ci["ci_hi"]),
        }
    log = logging.getLogger("anchoring_baseline_deltas")
    log.info("%-16s N=%5d C=%3d acc=%.3f | agree=%.3f ent=%.3f msp=%.3f margin=%.3f "
             "| dEnt=%+.3f dMSP=%+.3f dMar=%+.3f",
             name, N, C, correct.mean(), out["aurocs"]["agreement_top1"],
             out["aurocs"]["entropy"], out["aurocs"]["msp"], out["aurocs"]["margin"],
             out["deltas_agreement_minus"]["entropy"]["mean"],
             out["deltas_agreement_minus"]["msp"]["mean"],
             out["deltas_agreement_minus"]["margin"]["mean"])
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap-n", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    setup_logging(time.strftime("%Y%m%d_%H%M%S"), "anchoring_baseline_deltas")
    log = logging.getLogger("anchoring_baseline_deltas")
    log.info("bootstrap_n=%d seed=%d", args.bootstrap_n, args.seed)

    datasets = [
        {"name": "TeacherBehavior", "key": "teacher_behavior",
         "loader": lambda: load_scb5("teacher_behavior")},
        {"name": "HandriseReadWrite", "key": "handrise_readwrite",
         "loader": lambda: load_scb5("handrise_readwrite")},
        {"name": "BowTurnHead", "key": "bow_turnhead",
         "loader": lambda: load_scb5("bow_turnhead")},
        {"name": "VOC2007", "key": "voc2007", "loader": lambda: load_v2("voc2007")},
        {"name": "Stanford40", "key": "stanford40", "loader": lambda: load_v2("stanford40")},
        {"name": "COCO", "key": "coco", "loader": lambda: load_v2("coco")},
        {"name": "CIFAR-100", "key": "cifar100",
         "loader": lambda: load_v2("cifar100")},
    ]

    results = []
    for cfg in datasets:
        r = evaluate(cfg, args)
        if r is not None:
            results.append(r)

    # MultiTHUMOS (5 backbones, separate loader path)
    try:
        from anchoring_multithumos import load_multithumos
        stack, gt = load_multithumos()
        avg = stack.mean(0)
        cons = np.argmax(avg, 1)
        correct = np.array([cons[i] in np.where(gt[i] > 0)[0] for i in range(len(gt))], dtype=bool)
        s = compute_signals(stack, avg)
        out = {"dataset": "MultiTHUMOS", "N": int(len(gt)), "C": int(gt.shape[1]),
               "K": int(stack.shape[0]), "consensus_accuracy": float(correct.mean()),
               "aurocs": {k: float(auroc(s[k], correct)) for k in s},
               "deltas_agreement_minus": {}}
        for base in ["entropy", "msp", "margin"]:
            ci = bootstrap_auroc_diff(s["agreement_top1"], s[base], correct,
                                      args.bootstrap_n, args.seed)
            out["deltas_agreement_minus"][base] = {
                "mean": float(ci["mean"]), "ci_lo": float(ci["ci_lo"]),
                "ci_hi": float(ci["ci_hi"]),
            }
        results.append(out)
        log.info("MultiTHUMOS N=%d acc=%.3f | dEnt=%+.3f dMSP=%+.3f dMar=%+.3f",
                 out["N"], out["consensus_accuracy"],
                 out["deltas_agreement_minus"]["entropy"]["mean"],
                 out["deltas_agreement_minus"]["msp"]["mean"],
                 out["deltas_agreement_minus"]["margin"]["mean"])
    except Exception as e:
        log.warning("MultiTHUMOS skipped: %s", e)

    report = {
        "script": __file__,
        "args": vars(args),
        "timestamp": time.strftime("%Y%m%d_%H%M%S"),
        "results": results,
    }
    out_json = OUT_DIR / "anchoring_baseline_deltas_report.json"
    with open(out_json, "w") as f:
        json.dump(report, f, indent=1)
    log.info("Report: %s", out_json)


if __name__ == "__main__":
    main()
