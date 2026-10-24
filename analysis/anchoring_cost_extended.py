"""
analysis/anchoring_cost_extended.py — Exp D extended: per-combination breakdown + bootstrap CIs
============================================================================================
Extends the original cost analysis with:
  1. Best/worst individual k-backbone combinations (not just averages)
  2. Bootstrap CIs on agreement-minus-entropy AUROC per k
  3. Identification of the single best 2-backbone pair (deployment-relevant)

Output: results/anchoring_cost_extended_report.{json,txt}

Reproducibility:
  python analysis/anchoring_cost_extended.py
"""

import argparse
import itertools
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase0_oracle_analysis import SUBSET_CONFIG, load_clip, setup_logging, OUT_DIR
from pilot_perimage_anchoring import BACKBONES, softmax_entropy, auroc, bootstrap_auroc_diff

KS = [1, 2, 3, 4, 5]


def metrics_for_subset(logits_list, gt):
    stack = np.stack(logits_list, 0)
    avg = stack.mean(0)
    carg = np.argmax(avg, 1)
    correct = np.array([carg[i] in np.where(gt[i] > 0)[0]
                        for i in range(len(gt))], dtype=bool)
    ent, _ = softmax_entropy(avg)
    s_ent = -ent
    k = len(logits_list)
    s_agree = None
    if k >= 2:
        barg = np.argmax(stack, 2)
        s_agree = (barg == carg[None, :]).mean(0)
    return correct, s_ent, s_agree


def evaluate(subset, bootstrap_n, seed, log):
    log.info(f"=== {subset} ===")
    all_logits = {}
    gt = None
    for bb in BACKBONES:
        clip = load_clip(bb, subset, log)
        if clip is None:
            return None
        logits, labels = clip
        if gt is None:
            gt = labels
        elif not np.array_equal(gt, labels):
            raise RuntimeError("label mismatch")
        all_logits[bb] = logits
    n = len(gt)

    by_k = {}
    for k in KS:
        keys = list(itertools.combinations(BACKBONES, k))
        combo_results = []
        for combo in keys:
            ll = [all_logits[b] for b in combo]
            correct, s_ent, s_agree = metrics_for_subset(ll, gt)
            ent_auc = auroc(s_ent, correct)
            agr_auc = auroc(s_agree, correct) if s_agree is not None else None
            delta = None
            if agr_auc is not None:
                delta = bootstrap_auroc_diff(s_agree, s_ent, correct, bootstrap_n, seed)
            combo_results.append({
                "backbones": list(combo),
                "base_acc": float(correct.mean()),
                "entropy_auroc": float(ent_auc),
                "agreement_auroc": float(agr_auc) if agr_auc is not None else None,
                "delta_mean": float(delta["mean"]) if delta else None,
                "delta_ci_lo": float(delta["ci_lo"]) if delta else None,
                "delta_ci_hi": float(delta["ci_hi"]) if delta else None,
            })
        ent_aurocs = [c["entropy_auroc"] for c in combo_results]
        agr_aurocs = [c["agreement_auroc"] for c in combo_results if c["agreement_auroc"] is not None]
        best_by_agr = max(combo_results, key=lambda c: c["agreement_auroc"] if c["agreement_auroc"] is not None else -1)
        best_by_delta = max(combo_results, key=lambda c: c["delta_mean"] if c["delta_mean"] is not None else -1)
        entry = {
            "k": k,
            "n_subsets": len(keys),
            "entropy_auroc_mean": float(np.mean(ent_aurocs)),
            "entropy_auroc_std": float(np.std(ent_aurocs)) if len(ent_aurocs) > 1 else 0.0,
            "agreement_auroc_mean": float(np.mean(agr_aurocs)) if agr_aurocs else None,
            "agreement_auroc_std": float(np.std(agr_aurocs)) if len(agr_aurocs) > 1 else 0.0,
            "best_agreement_combo": best_by_agr,
            "best_delta_combo": best_by_delta,
            "combinations": combo_results,
        }
        by_k[f"k={k}"] = entry
        best_agr_str = (f"{best_by_agr['agreement_auroc']:.4f} {best_by_agr['backbones']}"
                        if best_by_agr['agreement_auroc'] is not None else "n/a")
        log.info(f"  k={k}: ent={entry['entropy_auroc_mean']:.4f} "
                 f"agr={entry['agreement_auroc_mean'] or 'n/a'} "
                 f"best_agr={best_agr_str}")

    return {"subset": subset, "n_images": int(n), "by_k": by_k}


def format_report(results, args):
    lines = []
    lines.append("=" * 86)
    lines.append("  Exp D extended — per-combination breakdown + bootstrap CIs per k")
    lines.append("=" * 86)
    lines.append(f"  Bootstrap B={args.bootstrap_n} seed={args.seed}")
    lines.append("")
    for res in results:
        lines.append(f"── {res['subset']} | N={res['n_images']} ──")
        lines.append(f"  {'k':>3} | {'entAUROC':>9} {'agrAUROC':>9} {'Δ mean':>7} "
                     f"{'Δ CI':>19} | best combo (agr)")
        lines.append("  " + "-" * 86)
        for kkey, d in res["by_k"].items():
            agr_m = d["agreement_auroc_mean"]
            best = d["best_agreement_combo"]
            delta = d["best_delta_combo"]
            dstr = (f"{delta['delta_mean']:+.4f} [{delta['delta_ci_lo']:+.4f},"
                    f"{delta['delta_ci_hi']:+.4f}]") if delta["delta_mean"] is not None else "       n/a       "
            best_delta_str = (f"{delta['delta_mean']:+.4f}" if delta["delta_mean"] is not None else "n/a")
            agr_str = f"{agr_m:>9.4f}" if agr_m is not None else "      n/a"
            lines.append(f"  {d['k']:>3} | {d['entropy_auroc_mean']:>9.4f} "
                         f"{agr_str} | {dstr:>27} | "
                         f"{' '.join(best['backbones'])} (Δ={best_delta_str})")
        lines.append("")
    lines.append("=" * 86)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subsets", nargs="+",
                        default=["teacher_behavior", "handrise_readwrite", "bow_turnhead"])
    parser.add_argument("--bootstrap_n", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ts = time.strftime("%Y%m%d_%H%M%S")
    setup_logging(ts, name="anchoring_cost_extended")
    log = logging.getLogger()
    log.info(f"Exp D extended | args={vars(args)}")

    results = []
    for subset in args.subsets:
        r = evaluate(subset, args.bootstrap_n, args.seed, log)
        if r is not None:
            results.append(r)
    if not results:
        log.error("No results."); return 1

    txt = format_report(results, args)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "anchoring_cost_extended_report.json", "w") as f:
        json.dump({"args": vars(args), "timestamp": ts, "results": results}, f, indent=2)
    with open(OUT_DIR / "anchoring_cost_extended_report.txt", "w") as f:
        f.write(txt)
    print(); print(txt)
    return 0


if __name__ == "__main__":
    exit(main())
