"""
analysis/anchoring_prompt_class_breakdown.py — P2: CAPE per-class breakdown
==========================================================================
Elevation-plan P2. Attacks hard-flaw #2 (headline +0.067 exists only under
CAPE-A; CAPE-B/C reverse it). Question: does the agreement advantage on the
low-anchor class `answer` survive CAPE-B/C, or does it also flip sign?

For each CAPE prompt set (A = default extraction, B, C), on TeacherBehavior:
  - compute consensus (avg-logit argmax), correctness, primary class
  - signals: agreement_top1, entropy (same definitions as anchoring_main.py)
  - global AUROC + delta vs entropy (sanity: must reproduce robustness
    report numbers: A +0.067, B -0.092, C -0.117)
  - per-class AUROC (primary-class mask, N>=10, non-degenerate)
  - Spearman correlation of per-class delta across prompt sets

Output: results/anchoring_prompt_class_breakdown_report.{json,txt}

Reproducibility:
  python analysis/anchoring_prompt_class_breakdown.py
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase0_oracle_analysis import SUBSET_CONFIG, setup_logging, OUT_DIR, DATA_DIR  # noqa: E402
from pilot_perimage_anchoring import (  # noqa: E402
    BACKBONES, softmax_entropy, auroc, aupr, bootstrap_auroc_diff,
)

SUBSET = "teacher_behavior"
CLASSES = SUBSET_CONFIG[SUBSET]["classes"]


def load_prompt_set(cape: str, log):
    """Load 5-backbone logits + labels for a CAPE set.

    cape in {"A", "B", "C"}; A is the default extraction (no suffix)."""
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
            raise RuntimeError(f"label mismatch across backbones for CAPE-{cape}")
        stack.append(logits)
    return np.stack(stack, axis=0), gt


def signals_from(logits_stack, avg_logits):
    """agreement_top1 and entropy (high = keep), matching anchoring_main.py."""
    ent_avg, _ = softmax_entropy(avg_logits)
    backbone_argmaxes = np.argmax(logits_stack, axis=2)
    consensus_argmax = np.argmax(avg_logits, axis=1)
    agreement = (backbone_argmaxes == consensus_argmax[None, :]).mean(axis=0)
    return agreement, -ent_avg


def evaluate(cape, bootstrap_n, seed, log):
    loaded = load_prompt_set(cape, log)
    if loaded is None:
        return None
    logits_stack, gt = loaded
    n = logits_stack.shape[1]
    avg_logits = logits_stack.mean(axis=0)
    consensus = np.argmax(avg_logits, axis=1)
    correct = np.array([consensus[i] in np.where(gt[i] > 0)[0]
                        for i in range(n)], dtype=bool)
    primary = np.array([np.where(gt[i] > 0)[0][0] for i in range(n)])

    agreement, entropy = signals_from(logits_stack, avg_logits)

    delta = bootstrap_auroc_diff(agreement, entropy, correct, bootstrap_n, seed)
    out = {
        "cape": cape,
        "n": int(n),
        "consensus_acc": float(correct.mean()),
        "entropy_auroc": auroc(entropy, correct),
        "agreement_auroc": auroc(agreement, correct),
        "agreement_aupr": aupr(agreement, correct),
        "entropy_aupr": aupr(entropy, correct),
        "delta_mean": delta["mean"],
        "delta_ci_lo": delta["ci_lo"],
        "delta_ci_hi": delta["ci_hi"],
        "per_class": {},
    }
    for c, cls in enumerate(CLASSES):
        mask = primary == c
        if mask.sum() < 10:
            continue
        entry = {"n": int(mask.sum()),
                 "consensus_acc": float(correct[mask].mean())}
        if correct[mask].std() == 0:
            entry["entropy_auroc"] = None
            entry["agreement_auroc"] = None
        else:
            entry["entropy_auroc"] = auroc(entropy[mask], correct[mask])
            entry["agreement_auroc"] = auroc(agreement[mask], correct[mask])
            entry["delta"] = float(entry["agreement_auroc"] - entry["entropy_auroc"])
        out["per_class"][cls] = entry
    log.info(f"CAPE-{cape}: N={n} acc={correct.mean():.3f} "
             f"ent={out['entropy_auroc']:.3f} agr={out['agreement_auroc']:.3f} "
             f"d={delta['mean']:+.4f} CI[{delta['ci_lo']:+.4f},{delta['ci_hi']:+.4f}]")
    return out


def per_class_spearman(results_by_cape, log):
    """Spearman correlation of per-class delta across CAPE sets."""
    def _rho(x, y):
        from scipy.stats import spearmanr
        common = sorted(set(x.keys()) & set(y.keys()))
        xs = [x[c]["delta"] for c in common if x[c].get("delta") is not None]
        ys = [y[c]["delta"] for c in common if y[c].get("delta") is not None]
        if len(xs) < 3:
            return None
        r, p = spearmanr(xs, ys)
        return {"rho": float(r), "p": float(p), "n": len(xs)}

    return {
        "A_vs_B": _rho(results_by_cape["A"]["per_class"], results_by_cape["B"]["per_class"]),
        "A_vs_C": _rho(results_by_cape["A"]["per_class"], results_by_cape["C"]["per_class"]),
        "B_vs_C": _rho(results_by_cape["B"]["per_class"], results_by_cape["C"]["per_class"]),
    }


def format_report(results_by_cape, corr):
    lines = []
    lines.append("=" * 80)
    lines.append("  P2 — CAPE prompt-set per-class breakdown (TeacherBehavior)")
    lines.append("=" * 80)
    lines.append("")
    lines.append("  Global agreement-minus-entropy AUROC delta per prompt set:")
    lines.append(f"    {'set':>4} | {'acc':>7} | {'entropy':>8} | {'agreement':>9} | "
                 f"{'delta':>8} | CI")
    lines.append("    " + "-" * 60)
    for cape in ("A", "B", "C"):
        r = results_by_cape[cape]
        lines.append(f"    {cape:>4} | {r['consensus_acc']*100:>6.1f}% | "
                     f"{r['entropy_auroc']:>8.4f} | {r['agreement_auroc']:>9.4f} | "
                     f"{r['delta_mean']:>+8.4f} | [{r['delta_ci_lo']:+.4f},{r['delta_ci_hi']:+.4f}]")
    lines.append("")
    lines.append("  Per-class delta (agreement - entropy AUROC) per prompt set:")
    lines.append(f"    {'class':>20} | {'N_A':>6} {'A':>9} | {'N_B':>6} {'B':>9} | "
                 f"{'N_C':>6} {'C':>9}")
    lines.append("    " + "-" * 70)
    for cls in CLASSES:
        cells = []
        for cape in ("A", "B", "C"):
            cd = results_by_cape[cape]["per_class"].get(cls)
            if cd is None:
                cells.append(("---", "---"))
            elif cd.get("delta") is None:
                cells.append((str(cd["n"]), "deg"))
            else:
                cells.append((str(cd["n"]), f"{cd['delta']:+.3f}"))
        lines.append(f"    {cls:>20} | {cells[0][0]:>6} {cells[0][1]:>9} | "
                     f"{cells[1][0]:>6} {cells[1][1]:>9} | {cells[2][0]:>6} {cells[2][1]:>9}")
    lines.append("")
    lines.append("  Per-class delta Spearman across prompt sets:")
    for pair, v in corr.items():
        if v is None:
            lines.append(f"    {pair}: n<3, skipped")
        else:
            lines.append(f"    {pair}: rho={v['rho']:+.3f} p={v['p']:.4f} n={v['n']}")
    lines.append("")
    lines.append("  Key question: does the `answer` class advantage survive CAPE-B/C?")
    lines.append("=" * 80)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap_n", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ts = time.strftime("%Y%m%d_%H%M%S")
    setup_logging(ts, name="anchoring_prompt_class_breakdown")
    log = logging.getLogger()

    results_by_cape = {}
    for cape in ("A", "B", "C"):
        r = evaluate(cape, args.bootstrap_n, args.seed, log)
        if r is None:
            log.error(f"CAPE-{cape} data missing; abort.")
            return 1
        results_by_cape[cape] = r

    corr = per_class_spearman(results_by_cape, log)
    txt = format_report(results_by_cape, corr)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "anchoring_prompt_class_breakdown_report.json", "w") as f:
        json.dump({"args": vars(args), "timestamp": ts,
                   "results": results_by_cape, "spearman": corr}, f, indent=2)
    with open(OUT_DIR / "anchoring_prompt_class_breakdown_report.txt", "w") as f:
        f.write(txt)
    print(); print(txt)
    return 0


if __name__ == "__main__":
    exit(main())
