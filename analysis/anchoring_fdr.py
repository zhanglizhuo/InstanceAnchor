"""
analysis/anchoring_fdr.py — R4: multiple-comparison FDR correction
==================================================================
Reviewer (internal audit) finding: Table 5 reports soft-agreement wins vs 8
confidence-scoring functions across 27 datasets = 216 pairwise comparisons
without any multiple-comparison control.

This script:
  1. recomputes the coupled-bootstrap AUROC delta (soft_agreement - CSF) per
     (dataset, CSF) pair with a stored two-sided bootstrap p-value
     p = 2*min(P(D<=0), P(D>=0)), B=2000 (B=10000 CI values already in
     anchoring_sota_csf_report.json and are NOT re-derived here);
  2. verifies the B=2000 win classification matches the committed B=10000
     win table (CI excludes 0);
  3. applies Benjamini-Hochberg FDR at q=0.05 across all 216 tests and
     reports per-CSF surviving win counts; also BH within each CSF (27 tests)
     as a sensitivity;
  4. reports the largest p among surviving tests and the post-hoc
     stratification note (OOD/multilabel/standard strata were defined after
     inspection; nominal p-values reported alongside FDR-adjusted ones).

Input : per-image logits/labels under results/per_image/ (same loading as
        anchoring_sota_csf.py)
Output: results/anchoring_fdr_report.{json,txt}

Usage: python analysis/anchoring_fdr.py [--bootstrap_n 2000] [--q 0.05]
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
    BACKBONES, softmax_entropy, auroc,
)

CIFAR_DIR = DATA_DIR / "cifar100"
META_DIR = DATA_DIR / "meta"
T = 100.0
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
CSFS = ["entropy", "msp", "margin", "max_logit", "energy",
        "ensemble_mi", "pred_var", "hard_agree"]


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


def compute_signals(logits_stack, avg_logits):
    K, N, C = logits_stack.shape
    consensus = np.argmax(avg_logits, axis=1)
    backbone_argmaxes = np.argmax(logits_stack, axis=2)
    ent_avg, p_avg = softmax_entropy(avg_logits)
    s = {}
    s["entropy"] = -ent_avg
    s["msp"] = p_avg.max(axis=1)
    sorted_avg = np.sort(avg_logits, axis=1)
    s["margin"] = sorted_avg[:, -1] - sorted_avg[:, -2]
    probs = np.stack([np.exp(z * T) / np.exp(z * T).sum(axis=1, keepdims=True)
                      for z in logits_stack], axis=0)
    s["soft_agree"] = probs[:, np.arange(N), consensus].mean(axis=0)
    s["hard_agree"] = (backbone_argmaxes == consensus[None, :]).mean(axis=0)
    s["max_logit"] = avg_logits.max(axis=1)
    s["energy"] = -np.log(np.exp(avg_logits * T).sum(axis=1)) / T
    mean_probs = probs.mean(axis=0)
    H_mean = -(mean_probs * np.log(mean_probs + 1e-12)).sum(axis=1)
    H_indiv = -(probs * np.log(probs + 1e-12)).sum(axis=2)
    s["ensemble_mi"] = H_mean - H_indiv.mean(axis=0)
    s["pred_var"] = probs.var(axis=0).mean(axis=1)
    return s, consensus


def bootstrap_pvalue(a, b, label, n_boot, seed):
    """Coupled-bootstrap two-sided p for H0: AUROC(a) <= AUROC(b)."""
    rng = np.random.RandomState(seed)
    n = len(label)
    diffs = np.empty(n_boot, dtype=np.float32)
    for i in range(n_boot):
        idx = rng.randint(0, n, size=n)
        diffs[i] = (auroc(a[idx], label[idx]) - auroc(b[idx], label[idx]))
    p_le = float(np.mean(diffs <= 0))   # P(auroc_a - auroc_b <= 0) = P(a <= b)
    p_ge = float(np.mean(diffs >= 0))   # P(auroc_a - auroc_b >= 0) = P(a >= b)
    return {"p_two_sided": float(2.0 * min(p_le, p_ge)),
            "p_one_sided_a_gt_b": p_ge}


def bh_fdr(pvals, q):
    """Benjamini-Hochberg: returns q-values (largest p <= threshold => adj)."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    threshold = np.arange(1, n + 1) / n * q
    # q-value assigned to each test = min over k>=rank of (p_rank*n/k)
    minp = np.empty(n)
    cur = np.inf
    for i in range(n - 1, -1, -1):
        cur = min(cur, ranked[i] * n / (i + 1))
        minp[i] = cur
    qvals = np.empty(n)
    qvals[order] = np.minimum(minp, 1.0)
    return qvals


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap_n", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--q", type=float, default=0.05)
    args = parser.parse_args()

    ts = time.strftime("%Y%m%d_%H%M%S")
    setup_logging(ts, name="anchoring_fdr")
    log = logging.getLogger()

    # reference win table from committed B=10000 report
    ref = json.load(open(OUT_DIR / "anchoring_sota_csf_report.json"))
    ref_wins = {csf: ref["win_table"][csf] for csf in ref["win_table"]}

    keys = list(EXISTING.keys()) + list(CIFAR.keys()) + META_KEYS
    tests = []
    for key in keys:
        loaded = load_stack(key)
        if loaded is None:
            continue
        logits_stack, gt = loaded
        avg = logits_stack.mean(axis=0)
        signals, consensus = compute_signals(logits_stack, avg)
        correct = correctness(consensus, gt)
        if correct.std() == 0:
            continue
        soft = signals["soft_agree"]
        for csf in CSFS:
            bp = bootstrap_pvalue(soft, signals[csf], correct,
                                  args.bootstrap_n, args.seed)
            tests.append({"dataset": key, "csf": csf, **bp})
        log.info(f"  {key:18s} done")
    log.info(f"{len(tests)} tests")

    # classification from B=2000 p-values: significant = p_two_sided < 0.05
    sig_rows = [t for t in tests if t["p_two_sided"] < 0.05]

    # BH across all 216
    qvals_all = bh_fdr([t["p_two_sided"] for t in tests], args.q)
    for t, q in zip(tests, qvals_all):
        t["q_all"] = float(q)
    # BH within CSF (27 tests each)
    for csf in CSFS:
        sub = [t for t in tests if t["csf"] == csf]
        qsub = bh_fdr([t["p_two_sided"] for t in sub], args.q)
        for t, q in zip(sub, qsub):
            t["q_within_csf"] = float(q)

    win_table_b2k = {}
    win_table_b2k_fdr = {}
    win_table_within_fdr = {}
    for csf in CSFS:
        sub = [t for t in tests if t["csf"] == csf]
        w = sum(1 for t in sub if t["p_two_sided"] < 0.05)
        wf = sum(1 for t in sub if t["q_all"] < args.q)
        ww = sum(1 for t in sub if t["q_within_csf"] < args.q)
        win_table_b2k[csf] = w
        win_table_b2k_fdr[csf] = wf
        win_table_within_fdr[csf] = ww

    out = {"args": vars(args), "timestamp": ts,
           "n_tests": len(tests),
           "csfs": CSFS,
           "ref_win_table_b10000": ref_wins,
           "win_counts_nominal_b2000": win_table_b2k,
           "win_counts_fdr_q0p05_all216": win_table_b2k_fdr,
           "win_counts_fdr_q0p05_within_csf": win_table_within_fdr,
           "max_q_surviving_all": float(max(
               (t["q_all"] for t in tests
                if t["csf"] == "entropy" and t["q_all"] < args.q),
               default=float("nan"))),
           "tests": tests,
           "notes": [
               "p_two_sided from coupled bootstrap (B=%d), H0: soft AUROC <= CSF AUROC"
               % args.bootstrap_n,
               "q_all = BH-FDR over all %d tests at q=0.05" % len(tests),
               "q_within_csf = BH-FDR within each CSF (27 tests)",
               "stratification (OOD/multilabel/standard) was defined after "
               "inspection; its nominal stratum p-values should be read "
               "with this post-hoc selection in mind"]}

    lines = ["=" * 96, f"  R4 FDR correction: soft_agreement vs {len(CSFS)} CSFs "
                       f"x {len(tests)//len(CSFS)} datasets = {len(tests)} tests",
             "=" * 96, ""]
    lines.append(f"  {'CSF':>12} | {'ref B=10000':>14} | {'nominal B=2000':>15} "
                 f"| {'FDR all 216':>12} | {'FDR within':>11}")
    lines.append("  " + "-" * 70)
    for csf in CSFS:
        rw = ref_wins[csf]
        sub = [t for t in tests if t["csf"] == csf]
        non_sig_n = len(sub) - win_table_b2k[csf]
        non_sig_f = len(sub) - win_table_b2k_fdr[csf]
        non_sig_w = len(sub) - win_table_within_fdr[csf]
        lines.append(f"  {csf:>12} | W{rw['win']:>2}/T{rw['tie']:>2}/L{rw['loss']:>2} "
                     f"| W{win_table_b2k[csf]:>2}/N{non_sig_n:>2}"
                     f"       | W{win_table_b2k_fdr[csf]:>2}/N{non_sig_f:>2}"
                     f"        | W{win_table_within_fdr[csf]:>2}/N{non_sig_w:>2}")
    lines.append("=" * 96)
    txt = "\n".join(lines)

    with open(OUT_DIR / "anchoring_fdr_report.json", "w") as f:
        json.dump(out, f, indent=2, default=lambda o: float(o))
    with open(OUT_DIR / "anchoring_fdr_report.txt", "w") as f:
        f.write(txt)
    print(); print(txt)
    return 0


if __name__ == "__main__":
    exit(main())
