"""
analysis/anchoring_error_overlap_27.py — direct measurement of pairwise
error overlap across all 27 datasets (Exp P3 kill-argument close-out)
=======================================================================
Addresses the kill-argument point that the kappa-decomposition residual
("correlated errors or non-uniform confusion structure") is asserted but
never independently measured outside TeacherBehavior. For each of the 27
datasets we compute, from per-backbone per-image logits, the pairwise
error-overlap ratio  R_ab = P(B wrong | A wrong) / P(B wrong)  over the
10 backbone pairs (same definition as the TB-only plugin in
anchoring_evidence_plugins.py, median 1.75 there), and report:

  1. per-dataset mean/min/max/median overlap ratio across pairs
  2. correlation of mean overlap ratio with the decomposition residual
     share (1 - plurality_share) across the 27 datasets (Spearman + CI)
  3. the same correlation restricted to the 20 non-ImageNet-family
     datasets, and to the 8 original datasets

Output: results/anchoring_error_overlap_27_report.{json,txt}
Run: python analysis/anchoring_error_overlap_27.py [--bootstrap_n 10000]
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
    EXISTING, CIFAR, CIFARC, META_KEYS, BACKBONES, load_dataset,
)

IMAGENET_FAMILY = {
    "imagenet_a", "imagenet_r", "imagenet_sketch",
    "imagenet_v2_matched", "imagenet_v2_threshold", "imagenet_v2_top",
    "imagenet_c",
}
ORIGINAL_8 = {
    "teacher_behavior", "handrise_readwrite", "bow_turnhead",
    "voc2007", "stanford40", "coco", "multithumos", "cifar100",
}

ALL_KEYS = list(dict.fromkeys(
    list(EXISTING) + list(CIFAR) + list(CIFARC) + list(META_KEYS)))


def pair_overlap_ratios(stack, gt):
    """Per-pair conditional/marginal error-overlap ratios over 10 pairs.

    stack: (K, N, C) logits; gt: (N,) int or (N, C) binary.
    Returns list of (pair_name, ratio, class_coincidence).
    """
    preds = stack.argmax(-1)                      # (K, N)
    if gt.ndim == 1:
        correct = preds == gt[None, :].astype(int)
    else:
        correct = np.stack(
            [gt[np.arange(gt.shape[0]), preds[k]] > 0
             for k in range(preds.shape[0])])            # (K, N)
    C = gt.shape[1] if gt.ndim == 2 else int(gt.max()) + 1
    ratios = []
    names = BACKBONES
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            aw = ~correct[i]
            bw = ~correct[j]
            both = aw & bw
            p_cond = bw[aw].mean() if aw.any() else 0.0
            p_marg = bw.mean()
            if p_marg <= 0 or p_cond <= 0:
                ratios.append((f"{names[i]}+{names[j]}", float("nan"),
                               float("nan")))
            else:
                ratios.append((f"{names[i]}+{names[j]}",
                               float(p_cond / p_marg),
                               float((preds[i][both] == preds[j][both]).mean()
                                     * (C - 1)) if both.any() else float("nan")))
    return ratios


def spearman_ci(x, y, bootstrap_n, seed):
    from scipy.stats import spearmanr
    rng = np.random.default_rng(seed)
    rho = spearmanr(x, y).statistic
    boot = []
    for _ in range(bootstrap_n):
        idx = rng.integers(0, len(x), size=len(x))
        boot.append(spearmanr(x[idx], y[idx]).statistic)
    boot = np.array(boot)
    return rho, float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bootstrap_n", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    setup_logging(time.strftime("%Y%m%d_%H%M%S"),
                  name="anchoring_error_overlap_27")
    log = logging.getLogger("anchoring_error_overlap_27")
    t0 = time.time()

    decomp = json.load(open(OUT_DIR / "anchoring_kappa_meta_decomp_report.json"))
    residual = {d["name"]: 1.0 - d["plurality_share"]
                for d in decomp["datasets"]}

    rows = []
    for key in ALL_KEYS:
        loaded = load_dataset(key, log)
        if loaded is None:
            log.warning("skip %s: missing per-image files", key)
            continue
        stack, gt = loaded
        pairs = pair_overlap_ratios(stack, gt)
        vals = np.array([r for _, r, _ in pairs if not np.isnan(r)])
        cc = np.array([c for _, _, c in pairs if not np.isnan(c)])
        row = {
            "dataset": key,
            "n_pairs": int(len(vals)),
            "mean_ratio": float(vals.mean()) if vals.size else None,
            "min_ratio": float(vals.min()) if vals.size else None,
            "max_ratio": float(vals.max()) if vals.size else None,
            "median_ratio": float(np.median(vals)) if vals.size else None,
            "mean_class_coincidence": float(cc.mean()) if cc.size else None,
            "median_class_coincidence": float(np.median(cc)) if cc.size else None,
            "residual_share": residual.get(key),
            "pairs": [{"pair": p, "ratio": r, "class_coincidence": c}
                      for p, r, c in pairs],
        }
        rows.append(row)
        log.info("%s: mean overlap %.2f (median %.2f), class-coinc %.1f, residual %.2f",
                 key, row["mean_ratio"] or float("nan"),
                 row["median_ratio"] or float("nan"),
                 row["mean_class_coincidence"] or float("nan"),
                 row["residual_share"] or float("nan"))

    valid = [r for r in rows
             if r["mean_ratio"] is not None and r["residual_share"] is not None]
    keys = [r["dataset"] for r in valid]
    x = np.array([r["mean_ratio"] for r in valid])
    xc = np.array([r["mean_class_coincidence"] for r in valid])
    y = np.array([r["residual_share"] for r in valid])

    rho_all, lo_all, hi_all = spearman_ci(x, y, args.bootstrap_n, args.seed)
    rho_cc_all, lo_cc_all, hi_cc_all = spearman_ci(
        xc, y, args.bootstrap_n, args.seed)

    def subset(idx):
        return spearman_ci(x[idx], y[idx], args.bootstrap_n, args.seed)

    def subset_cc(idx):
        return spearman_ci(xc[idx], y[idx], args.bootstrap_n, args.seed)

    nonin = np.array([k not in IMAGENET_FAMILY for k in keys])
    orig = np.array([k in ORIGINAL_8 for k in keys])
    rho_nonin = subset(nonin)
    rho_orig = subset(orig)
    rho_cc_nonin = subset_cc(nonin)
    rho_cc_orig = subset_cc(orig)

    report = {
        "script": __file__,
        "claim": ("per-dataset pairwise error-overlap ratio vs kappa-decomposition "
                  "residual share across 27 datasets"),
        "n_datasets": len(valid),
        "bootstrap_n": args.bootstrap_n,
        "seed": args.seed,
        "correlation_all": {
            "spearman_rho": rho_all,
            "ci_lo": lo_all, "ci_hi": hi_all,
            "n": len(valid)},
        "correlation_class_coincidence_all": {
            "spearman_rho": rho_cc_all,
            "ci_lo": lo_cc_all, "ci_hi": hi_cc_all,
            "n": len(valid)},
        "correlation_non_imagenet": {
            "spearman_rho": rho_nonin[0],
            "ci_lo": rho_nonin[1], "ci_hi": rho_nonin[2],
            "n": int(nonin.sum())},
        "correlation_class_coincidence_non_imagenet": {
            "spearman_rho": rho_cc_nonin[0],
            "ci_lo": rho_cc_nonin[1], "ci_hi": rho_cc_nonin[2],
            "n": int(nonin.sum())},
        "correlation_original_8": {
            "spearman_rho": rho_orig[0],
            "ci_lo": rho_orig[1], "ci_hi": rho_orig[2],
            "n": int(orig.sum())},
        "correlation_class_coincidence_original_8": {
            "spearman_rho": rho_cc_orig[0],
            "ci_lo": rho_cc_orig[1], "ci_hi": rho_cc_orig[2],
            "n": int(orig.sum())},
        "datasets": rows,
    }

    with open(OUT_DIR / "anchoring_error_overlap_27_report.json", "w") as f:
        json.dump(report, f, indent=1)
    with open(OUT_DIR / "anchoring_error_overlap_27_report.txt", "w") as f:
        f.write(f"n_datasets: {len(valid)}\n")
        f.write(f"Spearman(mean overlap, residual) all 27: "
                f"rho={rho_all:.3f} CI[{lo_all:.3f},{hi_all:.3f}]\n")
        f.write(f"Spearman(mean class-coincidence, residual) all 27: "
                f"rho={rho_cc_all:.3f} CI[{lo_cc_all:.3f},{hi_cc_all:.3f}]\n")
        f.write(f"Spearman non-ImageNet (n={int(nonin.sum())}): "
                f"rho={rho_nonin[0]:.3f} CI[{rho_nonin[1]:.3f},{rho_nonin[2]:.3f}]\n")
        f.write(f"Spearman class-coincidence non-ImageNet: "
                f"rho={rho_cc_nonin[0]:.3f} CI[{rho_cc_nonin[1]:.3f},{rho_cc_nonin[2]:.3f}]\n")
        f.write(f"Spearman original 8: "
                f"rho={rho_orig[0]:.3f} CI[{rho_orig[1]:.3f},{rho_orig[2]:.3f}]\n")
        f.write(f"Spearman class-coincidence original 8: "
                f"rho={rho_cc_orig[0]:.3f} CI[{rho_cc_orig[1]:.3f},{rho_cc_orig[2]:.3f}]\n")
        f.write("\ndataset          mean  median  min  max  class-coinc  residual\n")
        for r in sorted(valid, key=lambda r: r["mean_ratio"] or 0):
            f.write(f"{r['dataset']:<18} {r['mean_ratio']:6.2f} "
                    f"{r['median_ratio']:6.2f} {r['min_ratio']:6.2f} "
                    f"{r['max_ratio']:6.2f} "
                    f"{r['mean_class_coincidence']:7.1f} "
                    f"{r['residual_share']:6.2f}\n")
    log.info("done in %.1fs: rho_all=%.3f CI[%.3f,%.3f] rho_cc=%.3f CI[%.3f,%.3f] n=%d",
             time.time() - t0, rho_all, lo_all, hi_all,
             rho_cc_all, lo_cc_all, hi_cc_all, len(valid))


if __name__ == "__main__":
    main()