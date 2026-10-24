"""
analysis/anchoring_selective_accuracy_27.py — soft agreement vs entropy
selective accuracy at matched coverage, all 27 datasets (P5 kill-argument
close-out).
=======================================================================
Answers the magnitude objection to the 23/27 AUROC record: selective
accuracy deltas at fixed coverage points (30/50/70/90%), soft agreement
vs ensemble entropy, per dataset + mean across datasets. Both signals are
compute-matched (same 5-backbone logits). Soft agreement and entropy are
continuous scores, so no tie-aware randomization is needed. Uses the same
loader/protocol as anchoring_meta.py (T=100).

Output: results/anchoring_selective_accuracy_27_report.{json,txt}
Run: python analysis/anchoring_selective_accuracy_27.py
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
from anchoring_meta import (  # noqa: E402
    EXISTING, CIFAR, CIFARC, META_KEYS, BACKBONES, load_dataset, correctness,
)

T = 100.0
COVERAGES = [0.30, 0.50, 0.70, 0.90]
ALL_KEYS = list(dict.fromkeys(
    list(EXISTING) + list(CIFAR) + list(CIFARC) + list(META_KEYS)))

IMAGENET_FAMILY = {
    "imagenet_a", "imagenet_r", "imagenet_sketch",
    "imagenet_v2_matched", "imagenet_v2_threshold", "imagenet_v2_top",
    "imagenet_c",
}


def softmax(z):
    z = z * T
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def selective_acc(scores, correct, coverage):
    """Accuracy when keeping the top `coverage` fraction by score (higher
    = more confident)."""
    n = len(scores)
    k = max(1, int(round(coverage * n)))
    idx = np.argsort(-scores)[:k]
    return float(correct[idx].mean())


def main():
    ap = argparse.ArgumentParser()
    args = ap.parse_args()

    setup_logging(time.strftime("%Y%m%d_%H%M%S"),
                  name="anchoring_selective_accuracy_27")
    log = logging.getLogger("anchoring_selective_accuracy_27")

    rows = []
    for key in ALL_KEYS:
        loaded = load_dataset(key, log)
        if loaded is None:
            log.warning("skip %s", key)
            continue
        stack, gt = loaded
        cons = stack.mean(axis=0).argmax(-1)
        correct = correctness(cons, gt).astype(float)
        # soft agreement
        probs = softmax(stack)
        soft = probs[:, np.arange(len(cons)), cons].mean(axis=0)
        # ensemble entropy (negative -> confident)
        avg_logits = stack.mean(axis=0)
        p = softmax(avg_logits)
        ent = (p * np.log(p + 1e-12)).sum(axis=1)  # negative entropy
        row = {"dataset": key, "N": len(cons)}
        for cov in COVERAGES:
            sa = selective_acc(soft, correct, cov)
            ea = selective_acc(ent, correct, cov)
            row[f"soft_acc@{int(cov*100)}"] = round(sa, 4)
            row[f"ent_acc@{int(cov*100)}"] = round(ea, 4)
            row[f"delta_pp@{int(cov*100)}"] = round((sa - ea) * 100, 2)
        rows.append(row)
        log.info("%s done", key)

    report = {
        "script": __file__,
        "claim": ("soft agreement vs entropy selective accuracy at matched "
                  "coverage, 27 datasets"),
        "n_datasets": len(rows),
        "coverages": COVERAGES,
        "results": rows,
    }
    # mean deltas
    means = {}
    for cov in COVERAGES:
        k = f"delta_pp@{int(cov*100)}"
        vals = [r[k] for r in rows]
        means[f"mean_delta_pp@{int(cov*100)}"] = round(
            float(np.mean(vals)), 2)
        means[f"n_wins@{int(cov*100)}"] = int(sum(1 for v in vals if v > 0))
    report["summary"] = means
    # strata
    in_fam = [r for r in rows if r["dataset"] in IMAGENET_FAMILY]
    non_in = [r for r in rows if r["dataset"] not in IMAGENET_FAMILY]
    strata = {}
    for name, subset in [("imagenet_family", in_fam),
                         ("non_imagenet", non_in)]:
        strata[name] = {
            f"mean_delta_pp@{int(cov*100)}": round(
                float(np.mean([r[f"delta_pp@{int(cov*100)}"]
                               for r in subset])), 2)
            for cov in COVERAGES}
        strata[name]["n"] = len(subset)
    report["strata"] = strata

    with open(OUT_DIR / "anchoring_selective_accuracy_27_report.json",
              "w") as f:
        json.dump(report, f, indent=1)
    with open(OUT_DIR / "anchoring_selective_accuracy_27_report.txt",
              "w") as f:
        f.write(f"n_datasets: {len(rows)}\n")
        for cov in COVERAGES:
            k = f"mean_delta_pp@{int(cov*100)}"
            f.write(f"coverage {int(cov*100)}%: mean delta "
                    f"{means[k]} pp, wins {means[f'n_wins@{int(cov*100)}']}/"
                    f"{len(rows)}\n")
        f.write("strata: " + json.dumps(strata) + "\n")
        for r in sorted(rows, key=lambda x: x["delta_pp@50"]):
            f.write(f"{r['dataset']:<18} "
                    + " ".join(f"@{c}:{r[f'delta_pp@{c}']}"
                               for c in [30, 50, 70, 90]) + "\n")
    log.info("done: %s", json.dumps(means))


if __name__ == "__main__":
    main()