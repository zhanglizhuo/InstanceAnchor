"""
analysis/anchoring_cost.py — Exp D: cost-efficiency (k-backbone subsets)
========================================================================
Deployment angle: agreement needs >=2 CLIP backbones (5× cost if all 5). Does a
CHEAP 2-backbone agreement already beat single-backbone entropy? How does
agreement scale with k?

For each k in {1..5} and EVERY backbone subset of size k (C(5,k) combinations):
  - system prediction = argmax of the k-backbone averaged logits (consensus-of-k)
  - correct[i] = consensus-of-k argmax in GT
  - signals:
      k-ensemble-entropy  = entropy(softmax(avg_k_logits * 100))   [compute-matched]
      k-agreement         = fraction of the k backbones whose argmax == consensus
                                                              (k>=2; k=1 degenerate)
  - metrics: AUROC, tie-aware selective accuracy @30%/50%, base accuracy.
Average metrics across all size-k subsets. k=1 reports single-backbone entropy
only (agreement is constant=1).

Headline questions:
  Q1. Does 2-backbone agreement beat 1-backbone entropy?   (cost 2x vs 1x)
  Q2. How does agreement AUROC scale with k (diminishing return)?
  Q3. At each k, does agreement beat the compute-matched k-ensemble-entropy?

Output: results/anchoring_cost_report.{json,txt}

Reproducibility:
  python analysis/anchoring_cost.py
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
from phase0_oracle_analysis import SUBSET_CONFIG, load_clip, setup_logging, OUT_DIR  # noqa: E402
from pilot_perimage_anchoring import BACKBONES, softmax_entropy, auroc, bootstrap_auroc_diff  # noqa: E402
from anchoring_class_vs_instance import _tie_aware_selacc  # noqa: E402

KS = [1, 2, 3, 4, 5]
COVERAGES = [0.30, 0.50]
N_BOOT = 10000


def metrics_for_subset(logits_list, gt, rng):
    """Compute (correct, s_entropy, s_agreement_or_None, base_acc) for a given
    list of backbone logit arrays."""
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


def evaluate(subset, seed, log):
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
    rng = np.random.RandomState(seed)

    by_k = {}
    for k in KS:
        keys = list(itertools.combinations(BACKBONES, k))
        ent_aurocs, agr_aurocs = [], []
        ent_sel = {c: [] for c in COVERAGES}
        agr_sel = {c: [] for c in COVERAGES}
        base_accs = []
        for combo in keys:
            ll = [all_logits[b] for b in combo]
            correct, s_ent, s_agree = metrics_for_subset(ll, gt, rng)
            ent_aurocs.append(auroc(s_ent, correct))
            base_accs.append(float(correct.mean()))
            for c in COVERAGES:
                ent_sel[c].append(_tie_aware_selacc(s_ent, correct, c, 100, rng))
            if s_agree is not None:
                agr_aurocs.append(auroc(s_agree, correct))
                for c in COVERAGES:
                    agr_sel[c].append(_tie_aware_selacc(s_agree, correct, c, 100, rng))
        entry = {
            "k": k, "n_subsets": len(keys),
            "base_accuracy_mean": float(np.mean(base_accs)),
            "ensemble_entropy_auroc": float(np.mean(ent_aurocs)),
            "agreement_auroc": (float(np.mean(agr_aurocs)) if agr_aurocs else None),
            "ensemble_entropy_selacc": {c: float(np.mean(ent_sel[c])) for c in COVERAGES},
            "agreement_selacc": ({c: float(np.mean(agr_sel[c])) for c in COVERAGES}
                                 if agr_aurocs else None),
        }
        by_k[f"k={k}"] = entry
        log.info(f"  k={k}: base={entry['base_accuracy_mean']*100:.2f}%  "
                 f"entAUROC={entry['ensemble_entropy_auroc']:.4f}  "
                 + (f"agrAUROC={entry['agreement_auroc']:.4f}" if entry['agreement_auroc'] else ""))

    # Bootstrap CI for Q3 (5-agreement vs 5-entropy) — compute-matched pair
    ll5 = [all_logits[b] for b in BACKBONES]
    correct5, s_ent5, s_agree5 = metrics_for_subset(ll5, gt, rng)
    q3_ci = bootstrap_auroc_diff(s_agree5, s_ent5, correct5, N_BOOT, seed)
    results_meta = {"q3_delta_ci": q3_ci}

    return {"subset": subset, "n_images": int(n), "by_k": by_k, "bootstrap": results_meta}


def format_report(results, args):
    lines = []
    lines.append("=" * 86)
    lines.append("  Exp D — cost-efficiency: agreement vs ensemble-entropy across k backbones")
    lines.append("=" * 86)
    lines.append(f"  tie-aware selective accuracy (100 seeds), averaged over all C(5,k) subsets")
    lines.append("")
    for res in results:
        lines.append(f"── {res['subset']} | N={res['n_images']} ──")
        lines.append(f"  {'k':>3} | {'baseAcc':>7} {'entAUROC':>9} {'agrAUROC':>9} | "
                     f"{'ent@30%':>8} {'agr@30%':>8} | {'ent@50%':>8} {'agr@50%':>8} | n_subsets")
        lines.append("  " + "-" * 80)
        for kkey, d in res["by_k"].items():
            agr_a = f"{d['agreement_auroc']:.4f}" if d['agreement_auroc'] is not None else "  n/a "
            agr30 = (f"{d['agreement_selacc'][0.30]*100:.2f}%"
                     if d['agreement_selacc'] else "  n/a ")
            agr50 = (f"{d['agreement_selacc'][0.50]*100:.2f}%"
                     if d['agreement_selacc'] else "  n/a ")
            lines.append(f"  {d['k']:>3} | {d['base_accuracy_mean']*100:>6.2f}% "
                         f"{d['ensemble_entropy_auroc']:>9.4f} {agr_a:>9} | "
                         f"{d['ensemble_entropy_selacc'][0.30]*100:>7.2f}% {agr30:>8} | "
                         f"{d['ensemble_entropy_selacc'][0.50]*100:>7.2f}% {agr50:>8} | {d['n_subsets']}")
        # headline deltas (k=2 agreement vs k=1 entropy; k=5 agreement vs k=5 entropy)
        d1 = res["by_k"]["k=1"]; d2 = res["by_k"]["k=2"]; d5 = res["by_k"]["k=5"]
        lines.append("  headline deltas:")
        if d2["agreement_selacc"]:
            lines.append(f"    Q1 2-agr@30% - 1-ent@30% = "
                         f"{(d2['agreement_selacc'][0.30]-d1['ensemble_entropy_selacc'][0.30])*100:+.2f}pp")
        if d2["agreement_auroc"]:
            lines.append(f"    Q1 2-agr AUROC - 1-ent AUROC = "
                         f"{(d2['agreement_auroc']-d1['ensemble_entropy_auroc']):+.4f}")
        lines.append(f"    Q3 5-agr AUROC - 5-ent AUROC = "
                      f"{(d5['agreement_auroc']-d5['ensemble_entropy_auroc']):+.4f}")
        if "bootstrap" in res and res["bootstrap"].get("q3_delta_ci"):
            q3ci = res["bootstrap"]["q3_delta_ci"]
            lines.append(f"    Q3 CI: [{q3ci['ci_lo']:+.4f}, {q3ci['ci_hi']:+.4f}] (B={N_BOOT}, per-image coupled)")
        lines.append("")
    lines.append("=" * 86)
    lines.append("  Reading: Q1>0 => 2-backbone agreement beats 1-backbone entropy (cheap deployment win).")
    lines.append("           agrAUROC rising then flat => diminishing return in k.")
    lines.append("           Q3>0 => agreement beats compute-matched ensemble-entropy at every k.")
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subsets", nargs="+",
                        default=["teacher_behavior", "handrise_readwrite", "bow_turnhead"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ts = time.strftime("%Y%m%d_%H%M%S")
    setup_logging(ts, name="anchoring_cost")
    log = logging.getLogger()
    log.info(f"Exp D | args={vars(args)}")

    results = []
    for subset in args.subsets:
        r = evaluate(subset, args.seed, log)
        if r is not None:
            results.append(r)
    if not results:
        log.error("No results."); return 1

    txt = format_report(results, args)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "anchoring_cost_report.json", "w") as f:
        json.dump({"args": vars(args), "timestamp": ts, "results": results}, f, indent=2)
    with open(OUT_DIR / "anchoring_cost_report.txt", "w") as f:
        f.write(txt)
    print(); print(txt)
    return 0


if __name__ == "__main__":
    exit(main())
