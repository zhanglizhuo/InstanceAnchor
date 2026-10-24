"""
analysis/anchoring_pair_validation.py — Reviewer Major-3: cross-dataset validation of the best 2-backbone pair
===============================================================================================================
The Exp-D-extended report identified `openai+dfn` as the best 2-backbone pair
on TeacherBehavior (agreement AUROC 0.688 vs 5-backbone entropy 0.639) but
only on that ONE dataset (winner's-curse concern, stated in the paper). This
script answers the reviewer's question: does the pair generalize?

For every dataset with per-image CLIP logits, compute:
  - agreement[openai+dfn]   (2-backbone vote agreement; the candidate pair)
  - entropy[openai+dfn]     (compute-matched: SAME 2 logits averaged)
  - entropy[all backbones]  (the paper's headline baseline, when available)
  - agreement[all backbones]
and report AUROC + coupled bootstrap CI for
  agree2 - ent2   (does the pair itself beat its own entropy? core question)
  agree2 - entAll (deployment: does the pair match the full ensemble?)
  agreeAll - entAll (paper headline for reference)

Winner's-curse test: if openai+dfn is intrinsically complementary, agree2-ent2
should be >= 0 (or at least not far below) on MOST datasets, not just TB.

Data: results/per_image/clip_{model}_{dataset}_logits.npy (+ _labels).
Datasets: SCB5 (5bb), voc2007/stanford40/coco/cifar100/multithumos.

Multilabel convention: correct = consensus argmax in any GT>0 class
(annoy; SCB5 multilabel subsets yes; cifar100 single-label argmax==GT).
Output: results/anchoring_pair_validation_report.{json,txt}

Reproducibility: python analysis/anchoring_pair_validation.py
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

from pilot_perimage_anchoring import BACKBONES, softmax_entropy, auroc, bootstrap_auroc_diff  # noqa: E402

DATA_DIR = REPO_ROOT / "results" / "per_image"
OUT_DIR = REPO_ROOT / "results"
LOG_DIR = REPO_ROOT / "logs"

PAIR = ("openai", "dfn")
# cifar100 labels are class indices; all other datasets use one-hot GT rows.
CLASS_INDEX_LABELS = {"cifar100"}
DATASETS = ["teacher_behavior", "handrise_readwrite", "bow_turnhead",
            "voc2007", "stanford40", "coco", "cifar100", "multithumos"]


def setup_logging(ts, name="anchoring_pair_validation"):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        handlers=[logging.FileHandler(LOG_DIR / f"{name}_{ts}.log"),
                                  logging.StreamHandler()])


def load_dataset(dataset, log):
    all_logits, gt = {}, None
    ddir = DATA_DIR / ("cifar100" if dataset == "cifar100" else "")
    for bb in BACKBONES:
        lp = ddir / f"clip_{bb}_{dataset}_logits.npy"
        bp = ddir / f"clip_{bb}_{dataset}_labels.npy"
        if not lp.exists():
            continue
        logits, labels = np.load(lp).astype(np.float32), np.load(bp).astype(np.float32)
        if gt is None:
            gt = labels
        elif not np.array_equal(gt, labels):
            raise RuntimeError(f"label mismatch for {bb} on {dataset}")
        all_logits[bb] = logits
    if gt is None:
        return None
    return all_logits, gt


def agreement_and_entropy(logits_list):
    stack = np.stack(logits_list, 0)
    avg = stack.mean(0)
    carg = np.argmax(avg, 1)
    ent, _ = softmax_entropy(avg)
    barg = np.argmax(stack, 2)
    agree = (barg == carg[None, :]).mean(0)
    return agree, -ent, carg


def evaluate(dataset, bootstrap_n, seed, log):
    loaded = load_dataset(dataset, log)
    if loaded is None:
        log.warning(f"no data for {dataset}")
        return None
    all_logits, gt = loaded
    n = len(gt)
    if not set(PAIR).issubset(all_logits):
        log.warning(f"pair {PAIR} incomplete on {dataset}; skipping")
        return None
    single = dataset in CLASS_INDEX_LABELS

    def correct_of(carg):
        if single:
            return (carg == gt.astype(int)).astype(bool)
        return np.array([carg[i] in np.where(gt[i] > 0)[0]
                         for i in range(n)], dtype=bool)

    # candidate pair
    agr2, ent2, carg2 = agreement_and_entropy([all_logits[b] for b in PAIR])
    corr2 = correct_of(carg2)
    # full ensemble
    k_all = [all_logits[b] for b in BACKBONES if b in all_logits]
    agrA, entA, cargA = agreement_and_entropy(k_all)
    corrA = correct_of(cargA)

    def auc(s, y):
        return float(auroc(s, y))

    def diff(sa, sb, y):
        d = bootstrap_auroc_diff(sa, sb, y, bootstrap_n, seed)
        return {"mean": float(d["mean"]), "ci_lo": float(d["ci_lo"]), "ci_hi": float(d["ci_hi"])}

    # NOTE on delta targets (reviewer Major-3): coupled bootstrap requires a
    # SINGLE correctness label. Δ(agr2-ent2) uses the PAIR's own consensus
    # (corr2). Δ(agr2-entAll) uses the FULL ensemble's consensus (corrA): the
    # deployment question "does the cheap pair match the expensive baseline"
    # must be asked against the baseline's own predictions. The headline AUROC
    # numbers in the table below are each reported against their own
    # consensus target, which is why 0.688 (corr2) differs from the agr2 value
    # implied inside Δ(agr2-entAll).
    return {
        "dataset": dataset,
        "n": int(n),
        "n_backbones": len(k_all),
        "pair": list(PAIR),
        "agreement2_acc": float(corr2.mean()),
        "agreement_all_acc": float(corrA.mean()),
        "auroc_agreement2": auc(agr2, corr2),
        "auroc_entropy2": auc(ent2, corr2),
        "auroc_agreement2_on_corrA": auc(agr2, corrA),
        "auroc_agreement_all": auc(agrA, corrA),
        "auroc_entropy_all": auc(entA, corrA),
        "delta_agr2_minus_ent2": diff(agr2, ent2, corr2),
        "delta_agr2_minus_entAll": diff(agr2, entA, corrA),
        "delta_agrAll_minus_entAll": diff(agrA, entA, corrA),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--bootstrap_n", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    ts = time.strftime("%Y%m%d_%H%M%S")
    setup_logging(ts)
    log = logging.getLogger()
    log.info(f"pair validation | args={vars(args)}")

    results = []
    for ds in args.datasets:
        r = evaluate(ds, args.bootstrap_n, args.seed, log)
        if r is not None:
            results.append(r)

    header = ("Dataset            N     agr2ACC ent2   agr2   agrAll entAll "
              "Δ(agr2-ent2)          Δ(agr2-entAll)        Δ(agrAll-entAll)")
    lines = [header]
    for r in results:
        d1 = r["delta_agr2_minus_ent2"]
        d2 = r["delta_agr2_minus_entAll"]
        d3 = r["delta_agrAll_minus_entAll"]
        lines.append(
            f"{r['dataset']:<18} {r['n']:>5} {r['agreement2_acc']:>6.2f} {r['auroc_entropy2']:>6.3f} "
            f"{r['auroc_agreement2']:>6.3f} "
            f"{r['auroc_agreement_all']:>6.3f} {r['auroc_entropy_all']:>6.3f} "
            f"{d1['mean']:+.4f} [{d1['ci_lo']:+.4f},{d1['ci_hi']:+.4f}] "
            f"{d2['mean']:+.4f} [{d2['ci_lo']:+.4f},{d2['ci_hi']:+.4f}] "
            f"{d3['mean']:+.4f} [{d3['ci_lo']:+.4f},{d3['ci_hi']:+.4f}]")
    lines.append("")
    lines.append("Reading: Δ(agr2-ent2) = does openai+dfn agreement beat ITS OWN entropy (winner's-curse test);")
    lines.append("         Δ(agr2-entAll) = deployment question: does the 2-backbone pair match the full ensemble?")

    with open(OUT_DIR / "anchoring_pair_validation_report.json", "w") as f:
        json.dump({"args": vars(args), "timestamp": ts, "results": results}, f, indent=1)
    with open(OUT_DIR / "anchoring_pair_validation_report.txt", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()