"""
analysis/anchoring_cka_mechanism.py — Test shared-geometry as mechanism for agreement advantage
=================================================================================================
Hypothesis (from SV/SharedVisual): agreement is informative only when backbone
feature spaces are NOT perfectly aligned. When pairwise backbone CKA on a dataset
is high, all backbones make similar predictions → agreement saturates → entropy
wins. When CKA is lower, backbone divergence carries per-image information →
agreement beats entropy.

This script tests whether mean pairwise backbone CKA predicts the
agreement−entropy AUROC gap (Δ) across the 8 evaluated datasets. If a strong
negative correlation exists (higher CKA → smaller Δ), CKA provides a
SV-independent mechanism explanation for when agreement works.

We compute CKA directly on logits (img_feat @ text_feat.T). Logits are a linear
projection of features; while CKA is invariant to invertible linear transforms,
logits project to C-dim (not always invertible). We therefore interpret
logit-CKA as "prediction-space alignment" rather than pure feature-space CKA.
This is the relevant quantity for agreement, which operates on argmax over the
same prediction space.

Output: results/anchoring_cka_mechanism_report.{json,txt}
"""

import argparse, json, logging, sys, time
from pathlib import Path
from itertools import combinations

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pilot_perimage_anchoring import (  # noqa: E402
    BACKBONES, softmax_entropy, auroc, bootstrap_auroc_diff, OUT_DIR,
)
from phase0_oracle_analysis import setup_logging  # noqa: E402


# ── Dataset registry: (key, logits_path_template, label_format) ──
# label_format: "multilabel" (N, C) binary; "singlelabel" (N,) int
DATASETS = [
    {"key": "teacher_behavior", "name": "TeacherBehavior", "label_fmt": "multilabel",
     "path": "clip_{bb}_teacher_behavior_logits.npy"},
    {"key": "handrise_readwrite", "name": "HandriseReadWrite", "label_fmt": "multilabel",
     "path": "clip_{bb}_handrise_readwrite_logits.npy"},
    {"key": "bow_turnhead", "name": "BowTurnHead", "label_fmt": "multilabel",
     "path": "clip_{bb}_bow_turnhead_logits.npy"},
    {"key": "voc2007", "name": "VOC2007", "label_fmt": "multilabel",
     "path": "clip_{bb}_voc2007_logits.npy"},
    {"key": "coco", "name": "COCO", "label_fmt": "multilabel",
     "path": "clip_{bb}_coco_logits.npy"},
    {"key": "stanford40", "name": "Stanford40", "label_fmt": "multilabel",
     "path": "clip_{bb}_stanford40_logits.npy"},
    {"key": "cifar100", "name": "CIFAR-100", "label_fmt": "singlelabel",
     "path": "cifar100/clip_{bb}_cifar100_logits.npy"},
    {"key": "multithumos", "name": "MultiTHUMOS", "label_fmt": "multilabel",
     "path": "clip_{bb}_multithumos_logits.npy"},
]


def linear_cka(X, Y):
    """Linear CKA (Kornblith et al. 2019) between (N, D1) and (N, D2).
    CKA = ||Y^T X||_F^2 / (||X^T X||_F * ||Y^T Y||_F), with X, Y centered.
    Returns float in [0, 1]; 1 = identical up to isotropic scaling."""
    X = X - X.mean(0, keepdims=True)
    Y = Y - Y.mean(0, keepdims=True)
    xtx = np.linalg.norm(X.T @ X)
    yty = np.linalg.norm(Y.T @ Y)
    if xtx < 1e-12 or yty < 1e-12:
        return float("nan")
    ytx = np.linalg.norm(Y.T @ X)
    return float((ytx ** 2) / (xtx * yty))


def compute_correctness(cons, labels, label_fmt):
    """Consensus correctness: argmax in any active GT (multilabel) or == label (single)."""
    if label_fmt == "singlelabel":
        return (cons == labels.astype(int))
    correct = np.zeros(len(labels), dtype=bool)
    for i in range(len(labels)):
        correct[i] = cons[i] in np.where(labels[i] > 0)[0]
    return correct


def load_dataset(ds_entry):
    """Load 5-backbone logits + labels for a dataset. Returns (stack (K,N,C), labels, n_classes)."""
    stack, labels = [], None
    for bb in BACKBONES:
        p = OUT_DIR / "per_image" / ds_entry["path"].format(bb=bb)
        if not p.exists():
            return None
        L = np.load(p)
        stack.append(L)
        if labels is None:
            lb_p = OUT_DIR / "per_image" / ds_entry["path"].format(bb=bb).replace("logits", "labels")
            labels = np.load(lb_p)
    return np.stack(stack, 0), labels


def compute_delta(stack, labels, label_fmt, n_boot=10000):
    """Δ = agreement AUROC - entropy AUROC (compute-matched: both on 5-backbone avg)."""
    avg = stack.mean(0)
    cons = np.argmax(avg, 1)
    correct = compute_correctness(cons, labels, label_fmt)
    if correct.std() == 0:
        return None, None, None
    ent, _ = softmax_entropy(avg)
    s_ent = -ent
    agr = (np.argmax(stack, 2) == cons[None, :]).mean(0)
    a_auc = auroc(agr, correct)
    e_auc = auroc(s_ent, correct)
    ci = bootstrap_auroc_diff(agr, s_ent, correct, n_boot, 42)
    return float(a_auc - e_auc), ci, float(correct.mean())


def compute_pairwise_cka(stack):
    """Return dict of pairwise CKA values + summary stats."""
    K = stack.shape[0]
    pairs = []
    for i, j in combinations(range(K), 2):
        cka = linear_cka(stack[i], stack[j])
        pairs.append({"pair": f"{BACKBONES[i]}-{BACKBONES[j]}", "cka": cka})
    cka_values = np.array([p["cka"] for p in pairs if not np.isnan(p["cka"])])
    return {
        "pairs": pairs,
        "mean_cka": float(cka_values.mean()) if len(cka_values) else float("nan"),
        "min_cka": float(cka_values.min()) if len(cka_values) else float("nan"),
        "max_cka": float(cka_values.max()) if len(cka_values) else float("nan"),
        "std_cka": float(cka_values.std()) if len(cka_values) else float("nan"),
    }


def correlation_with_bootstrap(x, y, n_boot=10000, seed=42):
    """Spearman-style rank correlation with bootstrap CI. Returns r, ci_lo, ci_hi, p_approx."""
    from scipy.stats import spearmanr
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 4:
        return float("nan"), float("nan"), float("nan"), float("nan")
    r, p = spearmanr(x, y)
    rng = np.random.RandomState(seed)
    n = len(x)
    boot = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.randint(0, n, size=n)
        if len(np.unique(x[idx])) < 2 or len(np.unique(y[idx])) < 2:
            boot[b] = float("nan")
            continue
        rb, _ = spearmanr(x[idx], y[idx])
        boot[b] = rb
    boot = boot[np.isfinite(boot)]
    return float(r), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)), float(p)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap-n", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    setup_logging(time.strftime("%Y%m%d_%H%M%S"), "anchoring_cka_mechanism")
    log = logging.getLogger(__name__)
    log.info("=== CKA mechanism test: does backbone alignment predict agreement advantage? ===")

    rows = []
    for ds in DATASETS:
        loaded = load_dataset(ds)
        if loaded is None:
            log.warning("Skip %s (missing data)", ds["name"])
            continue
        stack, labels = loaded
        delta, ci, cons_acc = compute_delta(stack, labels, ds["label_fmt"], n_boot=args.bootstrap_n)
        if delta is None:
            log.warning("Skip %s (degenerate correctness)", ds["name"])
            continue
        cka_info = compute_pairwise_cka(stack)
        ml_deg = float(labels.sum(1).mean()) if ds["label_fmt"] == "multilabel" else 1.0
        n_classes = int(stack.shape[2])

        log.info("%-18s N=%5d C=%3d acc=%.3f ml=%.2f Δ=%+.4f CKA=%.4f",
                 ds["name"], len(labels), n_classes, cons_acc, ml_deg, delta,
                 cka_info["mean_cka"])
        rows.append({
            "dataset": ds["key"], "name": ds["name"], "n": len(labels),
            "n_classes": n_classes, "consensus_acc": cons_acc, "ml_degree": ml_deg,
            "delta": delta, "delta_ci_lo": ci["ci_lo"], "delta_ci_hi": ci["ci_hi"],
            **cka_info,
        })

    if len(rows) < 4:
        log.error("Insufficient datasets (%d) for correlation analysis", len(rows))
        return

    deltas = np.array([r["delta"] for r in rows])
    predictors = {
        "mean_cka":      np.array([r["mean_cka"]      for r in rows]),
        "min_cka":       np.array([r["min_cka"]       for r in rows]),
        "max_cka":       np.array([r["max_cka"]       for r in rows]),
        "std_cka":       np.array([r["std_cka"]       for r in rows]),
        "n_classes":     np.array([r["n_classes"]     for r in rows]),
        "consensus_acc": np.array([r["consensus_acc"] for r in rows]),
        "ml_degree":     np.array([r["ml_degree"]     for r in rows]),
    }

    log.info("\n=== Correlation with Δ (Spearman, bootstrap B=%d) ===", args.bootstrap_n)
    correlations = {}
    for name, values in predictors.items():
        r, lo, hi, p = correlation_with_bootstrap(values, deltas, args.bootstrap_n, args.seed)
        flag = "***" if (lo > 0 or hi < 0) else ("  " if abs(r) < 0.3 else "")
        log.info("  %-14s ρ=%+6.3f  CI[%+6.3f,%+6.3f]  p=%.3f  %s",
                 name, r, lo, hi, p, flag)
        correlations[name] = {"rho": r, "ci_lo": lo, "ci_hi": hi, "p": p}

    # Key verdict
    cka_r = correlations["mean_cka"]["rho"]
    cka_ci = (correlations["mean_cka"]["ci_lo"], correlations["mean_cka"]["ci_hi"])
    log.info("\n=== Verdict ===")
    log.info("  CKA→Δ correlation: ρ=%+.3f CI[%+.3f,%+.3f]", cka_r, *cka_ci)
    if cka_r < -0.5 and cka_ci[1] < 0:
        log.info("  ✅ SUPPORTED: high backbone alignment → agreement loses (CKA mechanism confirmed)")
    elif cka_r > 0.5 and cka_ci[0] > 0:
        log.info("  ❌ REVERSED: high alignment → agreement wins (refutes hypothesis)")
    else:
        log.info("  ➖ INCONCLUSIVE: CKA does not linearly predict Δ across these datasets")

    results = {
        "script": "anchoring_cka_mechanism",
        "bootstrap_n": args.bootstrap_n, "seed": args.seed,
        "n_datasets": len(rows),
        "datasets": rows,
        "correlations": correlations,
        "verdict": {
            "cka_rho": cka_r, "cka_ci": list(cka_ci),
            "supported": bool(cka_r < -0.5 and cka_ci[1] < 0),
        },
        "interpretation": (
            "If supported: replaces ad-hoc 'spatial annotation multiplicity' theory. "
            "CKA provides a continuous, measurable predictor of when agreement works, "
            "grounded in SV's shared-geometry framework. The boundary condition "
            "becomes 'low backbone alignment' rather than a dataset-specific label."
        ),
    }
    out_json = OUT_DIR / "anchoring_cka_mechanism_report.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    lines = [
        "=" * 72,
        "  CKA Mechanism Test — does backbone alignment predict agreement advantage?",
        "=" * 72,
        f"  Datasets: {len(rows)}  Bootstrap: B={args.bootstrap_n}",
        "",
        "  Per-dataset:",
    ]
    for r in rows:
        lines.append(f"    {r['name']:<18s} Δ={r['delta']:+.4f} CKA={r['mean_cka']:.4f} "
                     f"(min={r['min_cka']:.3f}, max={r['max_cka']:.3f})")
    lines += ["", "  Correlation with Δ (Spearman, 95% bootstrap CI):"]
    for name, c in correlations.items():
        star = " ★" if (c["ci_lo"] > 0 or c["ci_hi"] < 0) else ""
        lines.append(f"    {name:<14s} ρ={c['rho']:+.3f} [{c['ci_lo']:+.3f},{c['ci_hi']:+.3f}] p={c['p']:.3f}{star}")
    lines += ["", "=" * 72]
    txt = "\n".join(lines)
    txt_out = OUT_DIR / "anchoring_cka_mechanism_report.txt"
    with open(txt_out, "w") as f:
        f.write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
