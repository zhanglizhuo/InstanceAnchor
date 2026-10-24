"""
analysis/anchoring_figure_cis.py — per-point bootstrap CIs for the data
figures (figure-audit close-out).
=======================================================================
Computes, from per-image logits (same loaders as the canonical analyses),
the 95% coupled-bootstrap CIs that the five data figures need:

  1. per-class (TB, 8 classes): agreement/entropy AUROC + delta CI
  2. anchor-level (TB): low/high anchor AUROC + delta CI
  3. B1 multi-label strata (TB): delta CI per stratum
  4. tie-aware risk-coverage bands (TB, VOC2007, Stanford40, CIFAR-100):
     agreement curves averaged over 200 tie-break seeds with 2.5/97.5
     percentile bands (entropy is continuous -> point curve)
  5. cost by_k (TB): per-subset AUROC values + across-subset bootstrap CI

Output: results/anchoring_figure_cis_report.{json,txt}
Run: python analysis/anchoring_figure_cis.py [--bootstrap_n 10000]
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
from phase0_oracle_analysis import setup_logging, OUT_DIR  # noqa: E402
from anchoring_meta import BACKBONES, load_dataset, correctness  # noqa: E402
from pilot_perimage_anchoring import softmax_entropy, auroc  # noqa: E402
from anchoring_mechanism import LOW_ANCHOR_TB, HIGH_ANCHOR_TB  # noqa: E402
from anchoring_cost import metrics_for_subset  # noqa: E402
from anchoring_main import risk_coverage_curve  # noqa: E402

T = 100.0
COVERAGE_GRID = np.arange(0.05, 1.0 + 1e-9, 0.05)
RISK_DATASETS = ["teacher_behavior", "voc2007", "stanford40", "cifar100"]


def signals_and_correct(stack, gt):
    avg = stack.mean(0)
    cons = np.argmax(avg, 1)
    correct = correctness(cons, gt).astype(bool)
    ent_raw, _ = softmax_entropy(avg, temperature=T)
    s_ent = -ent_raw                      # higher = more confident
    barg = np.argmax(stack, 2)
    s_agree = (barg == cons[None, :]).mean(0)   # 6 discrete values
    return s_ent, s_agree, correct, cons


def bootstrap_auroc_ci(scores, correct, bootstrap_n, seed):
    """Bootstrap 95% CI of a single-signal AUROC (resample images).
    Degenerate resamples (all-correct / all-wrong, where AUROC is undefined)
    are skipped."""
    rng = np.random.default_rng(seed)
    n = len(scores)
    vals = []
    for b in range(bootstrap_n):
        idx = rng.integers(0, n, n)
        c = correct[idx]
        if c.std() == 0:
            continue
        vals.append(auroc(scores[idx], c))
    vals = np.array(vals)
    if len(vals) == 0:
        return float("nan"), float("nan")
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def bootstrap_delta_ci(sa, se, correct, bootstrap_n, seed):
    """Coupled bootstrap 95% CI of AUROC(agreement) - AUROC(entropy).
    Degenerate resamples are skipped."""
    rng = np.random.default_rng(seed)
    n = len(correct)
    vals = []
    for b in range(bootstrap_n):
        idx = rng.integers(0, n, n)
        c = correct[idx]
        if c.std() == 0:
            continue
        vals.append(auroc(sa[idx], c) - auroc(se[idx], c))
    vals = np.array(vals)
    if len(vals) == 0:
        return float("nan"), float("nan")
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def tie_aware_risk_band(s_agree, correct, n_seeds, seed):
    """Mean + 2.5/97.5 percentiles of tie-aware selective accuracy across
    seeds, per coverage grid point. Replicates the canonical
    _tie_aware_acc_at protocol exactly (fresh jitter per coverage point,
    lexsort with -signal primary) so the @50/70/90 means reproduce the
    committed 78.52/71.96 values."""
    rng = np.random.RandomState(seed)
    n = len(correct)
    # canonical _tie_aware_acc_at consumes RNG in target order [0.5,0.7,0.9];
    # process those first so the @50/70/90 means reproduce the committed
    # 78.52/71.96 values exactly.
    covs = [0.5, 0.7, 0.9] + [c for c in COVERAGE_GRID
                              if c not in (0.5, 0.7, 0.9)]
    means, los, his = {}, {}, {}
    for c in covs:
        m = max(1, int(round(c * n)))
        accs = np.empty(n_seeds)
        for s in range(n_seeds):
            jitter = rng.random(n)
            order = np.lexsort((jitter, -s_agree))
            accs[s] = correct[order[:m]].mean()
        means[c] = float(accs.mean())
        los[c] = float(np.percentile(accs, 2.5))
        his[c] = float(np.percentile(accs, 97.5))
    return means, los, his


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bootstrap_n", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    setup_logging(time.strftime("%Y%m%d_%H%M%S"), name="anchoring_figure_cis")
    log = logging.getLogger("anchoring_figure_cis")

    report = {"script": __file__, "bootstrap_n": args.bootstrap_n,
              "seed": args.seed}

    # ── TB core quantities ──────────────────────────────────────────
    loaded = load_dataset("teacher_behavior", log)
    stack, gt = loaded
    s_ent, s_agree, correct, cons = signals_and_correct(stack, gt)
    primary = np.array([np.where(gt[i] > 0)[0][0] for i in range(len(gt))])
    ngt = gt.sum(1).astype(int)

    # classes in canonical order: reuse the phase0 module's SUBSET_CONFIG
    from phase0_oracle_analysis import SUBSET_CONFIG  # noqa: E402
    tb_classes = SUBSET_CONFIG["teacher_behavior"]["classes"]

    # 1. per-class CI
    per_class = []
    for ci, cname in enumerate(tb_classes):
        m = primary == ci
        if m.sum() < 30 or correct[m].std() == 0:
            per_class.append({"class": cname, "n": int(m.sum()),
                              "skipped": True})
            continue
        elo, ehi = bootstrap_auroc_ci(s_ent[m], correct[m],
                                      args.bootstrap_n, args.seed)
        alo, ahi = bootstrap_auroc_ci(s_agree[m], correct[m],
                                      args.bootstrap_n, args.seed)
        dlo, dhi = bootstrap_delta_ci(s_agree[m], s_ent[m], correct[m],
                                      args.bootstrap_n, args.seed)
        per_class.append({
            "class": cname, "n": int(m.sum()),
            "base_acc": float(correct[m].mean()),
            "entropy_auroc": float(auroc(s_ent[m], correct[m])),
            "entropy_ci": [elo, ehi],
            "agreement_auroc": float(auroc(s_agree[m], correct[m])),
            "agreement_ci": [alo, ahi],
            "delta_ci": [dlo, dhi],
        })
    report["per_class"] = per_class

    # 2. anchor-level CI
    low_idx = [i for i, c in enumerate(tb_classes) if c in LOW_ANCHOR_TB]
    high_idx = [i for i, c in enumerate(tb_classes) if c in HIGH_ANCHOR_TB]
    anchor = {}
    for label, idx in [("low_anchor", low_idx), ("high_anchor", high_idx)]:
        m = np.isin(primary, idx)
        elo, ehi = bootstrap_auroc_ci(s_ent[m], correct[m],
                                      args.bootstrap_n, args.seed)
        alo, ahi = bootstrap_auroc_ci(s_agree[m], correct[m],
                                      args.bootstrap_n, args.seed)
        dlo, dhi = bootstrap_delta_ci(s_agree[m], s_ent[m], correct[m],
                                      args.bootstrap_n, args.seed)
        anchor[label] = {
            "n": int(m.sum()),
            "entropy_auroc": float(auroc(s_ent[m], correct[m])),
            "entropy_ci": [elo, ehi],
            "agreement_auroc": float(auroc(s_agree[m], correct[m])),
            "agreement_ci": [alo, ahi],
            "delta_ci": [dlo, dhi],
        }
    report["anchor_level"] = anchor

    # 3. B1 strata delta CI
    strata = {}
    for label, mask in [("1_labels", ngt == 1), ("2_labels", ngt == 2),
                        ("3_labels", ngt == 3), ("4+_labels", ngt >= 4)]:
        m = mask
        dlo, dhi = bootstrap_delta_ci(s_agree[m], s_ent[m], correct[m],
                                      args.bootstrap_n, args.seed)
        strata[label] = {
            "n": int(m.sum()),
            "agree_minus_ent": float(auroc(s_agree[m], correct[m])
                                     - auroc(s_ent[m], correct[m])),
            "delta_ci": [dlo, dhi],
        }
    report["strata"] = strata

    # 4. tie-aware risk bands (4 datasets)
    risk = {}
    for key in RISK_DATASETS:
        loaded = load_dataset(key, log)
        if loaded is None:
            continue
        st, g = loaded
        se, sa, corr, _ = signals_and_correct(st, g)
        mean_a, lo_a, hi_a = tie_aware_risk_band(sa, corr, 200, args.seed)
        cov_e, acc_e, _ = risk_coverage_curve(se, corr, COVERAGE_GRID)
        risk[key] = {
            "agreement": [{"coverage": float(c), "mean_acc": float(mean_a[c]),
                           "lo": float(lo_a[c]), "hi": float(hi_a[c])}
                          for c in sorted(mean_a)],
            "entropy": [{"coverage": float(c), "accuracy": float(a)}
                        for c, a in zip(cov_e, acc_e)],
        }
        log.info("risk band done: %s", key)
    report["risk_tieaware"] = risk

    # 5. cost by_k across-subset CIs
    all_logits = {bb: stack[i] for i, bb in enumerate(BACKBONES)}
    by_k = {}
    for k in [1, 2, 3, 4, 5]:
        ent_vals, agr_vals = [], []
        for combo in itertools.combinations(BACKBONES, k):
            ll = [all_logits[b] for b in combo]
            corr2, se2, sa2 = metrics_for_subset(ll, gt, None)
            ent_vals.append(float(auroc(se2, corr2)))
            if sa2 is not None:
                agr_vals.append(float(auroc(sa2, corr2)))
        rng = np.random.default_rng(args.seed)
        def across_subset_ci(vals):
            vals = np.array(vals)
            boot = []
            for _ in range(args.bootstrap_n):
                idx = rng.integers(0, len(vals), len(vals))
                boot.append(vals[idx].mean())
            boot = np.array(boot)
            return float(np.percentile(boot, 2.5)), \
                float(np.percentile(boot, 97.5))
        entry = {"n_subsets": len(ent_vals),
                 "entropy_mean": float(np.mean(ent_vals)),
                 "entropy_ci": list(across_subset_ci(ent_vals))}
        if agr_vals:
            entry["agreement_mean"] = float(np.mean(agr_vals))
            entry["agreement_ci"] = list(across_subset_ci(agr_vals))
        by_k[f"k={k}"] = entry
        log.info("cost k=%d done (%d subsets)", k, len(ent_vals))
    report["cost_by_k"] = by_k

    with open(OUT_DIR / "anchoring_figure_cis_report.json", "w") as f:
        json.dump(report, f, indent=1)
    with open(OUT_DIR / "anchoring_figure_cis_report.txt", "w") as f:
        f.write(json.dumps(report, indent=1))
    log.info("done")


if __name__ == "__main__":
    main()