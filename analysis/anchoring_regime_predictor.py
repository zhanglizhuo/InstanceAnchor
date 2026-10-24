"""
analysis/anchoring_regime_predictor.py — Quantitative regime characterization
=============================================================================
Addresses the criticism that the "multi-label co-occurrence ambiguity" regime
is post-hoc. Computes quantitative predictors from each dataset's per-image
.npy files and tests whether they separate "agreement wins" from "agreement
loses" datasets.

Candidate predictors (all computable from existing .npy, no new GPU):
  P1. entropy_AUROC          — how well entropy already works
  P2. mean_agreement         — average backbone consensus rate
  P3. std_agreement          — agreement signal variance (saturation check)
  P4. mean_pairwise_argmax_agreement — how correlated the 5 backbones are
  P5. mean_multilabel_degree — average #GT labels per image
  P6. num_classes            — classification granularity

Datasets (6): TB, HRW, BTH (SCB5), VOC2007, Stanford40, CIFAR-100.
Outcome: Δ(agreement - entropy) AUROC, with win/lose/tie labels.

Output:
  results/anchoring_regime_predictor_report.{json,txt}

Reproducibility:
  python analysis/anchoring_regime_predictor.py
  python analysis/anchoring_regime_predictor.py --bootstrap-n 5000
"""
import argparse, json, sys, time, logging
from pathlib import Path
from itertools import combinations
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase0_oracle_analysis import load_clip, OUT_DIR
from pilot_perimage_anchoring import BACKBONES, TEMPERATURE, softmax_entropy, auroc

log = logging.getLogger(__name__)


def load_v2_logits(dataset_key, backbone):
    """Load V2 (VOC2007/Stanford40) logits."""
    p = OUT_DIR / "per_image" / f"clip_{backbone}_{dataset_key}_logits.npy"
    if not p.exists():
        return None, None
    logits = np.load(p)
    labels = np.load(OUT_DIR / "per_image" / f"clip_{backbone}_{dataset_key}_labels.npy")
    return logits, labels


def load_cifar_logits(backbone):
    """Load CIFAR-100 logits."""
    p = OUT_DIR / "per_image" / "cifar100" / f"clip_{backbone}_cifar100_logits.npy"
    if not p.exists():
        return None, None
    logits = np.load(p)
    labels = np.load(OUT_DIR / "per_image" / "cifar100" / f"clip_{backbone}_cifar100_labels.npy")
    return logits, labels


def compute_predictors(name, key, dtype, log, n_boot=10000, seed=42):
    """Compute all predictors + outcome for one dataset."""
    all_logits = []; gt = None
    for bb in BACKBONES:
        if dtype == "scb5":
            clip = load_clip(bb, key, log)
            if clip is None: return None
            logits, labels = clip
        elif dtype == "v2":
            logits, labels = load_v2_logits(key, bb)
            if logits is None: return None
        elif dtype == "cifar":
            logits, labels = load_cifar_logits(bb)
            if logits is None: return None
        if gt is None: gt = labels
        all_logits.append(logits)
    logits_stack = np.stack(all_logits, axis=0)
    K, N, C = logits_stack.shape
    avg = logits_stack.mean(0)
    cons = np.argmax(avg, 1)

    if gt.ndim == 1:
        correct = (cons == gt.astype(int))
    elif gt.ndim == 2:
        correct = np.array([cons[i] in np.where(gt[i] > 0)[0] for i in range(N)], dtype=bool)
    else:
        return None

    ent, _ = softmax_entropy(avg)
    backbone_args = np.argmax(logits_stack, axis=2)
    alpha = (backbone_args == cons[None, :]).mean(0)

    auroc_ent = auroc(-ent, correct)
    auroc_agr = auroc(alpha, correct)
    delta = auroc_agr - auroc_ent

    # Bootstrap CI for Δ
    rng = np.random.RandomState(seed)
    deltas_boot = []
    for _ in range(n_boot):
        idx = rng.randint(0, N, size=N)
        de = auroc(-ent[idx], correct[idx])
        da = auroc(alpha[idx], correct[idx])
        deltas_boot.append(da - de)
    deltas_boot = np.sort(deltas_boot)
    delta_ci = {"ci_lo": float(deltas_boot[int(0.025 * n_boot)]),
                "ci_hi": float(deltas_boot[int(0.975 * n_boot)])}

    pairwise_agree = np.mean([
        float(np.mean(a == backbone_args[j]))
        for i, j in combinations(range(K), 2)
        for a in [backbone_args[i]]
    ])

    ml_degree = float(np.mean(gt.sum(1)) if gt.ndim == 2 else 1.0)

    return {
        "name": name, "N": N, "C": C,
        "entropy_auroc": float(auroc_ent),
        "agreement_auroc": float(auroc_agr),
        "delta": float(delta),
        "delta_ci": delta_ci,
        "mean_agreement": float(np.mean(alpha)),
        "std_agreement": float(np.std(alpha)),
        "mean_pairwise_argmax_agreement": float(pairwise_agree),
        "mean_multilabel_degree": ml_degree,
    }


DATASETS = [
    {"name": "TeacherBehavior", "key": "teacher_behavior", "type": "scb5", "win": True},
    {"name": "HandriseReadWrite", "key": "handrise_readwrite", "type": "scb5", "win": False},
    {"name": "BowTurnHead", "key": "bow_turnhead", "type": "scb5", "win": False},
    {"name": "VOC2007", "key": "voc2007", "type": "v2", "win": False},
    {"name": "Stanford40", "key": "stanford40", "type": "v2", "win": False},
    {"name": "CIFAR-100", "key": "cifar100", "type": "cifar", "win": False},
]


def main():
    parser = argparse.ArgumentParser(description="Regime predictor analysis")
    parser.add_argument("--subsets", nargs="+", default=None,
                        help="Override dataset list (default: all 6)")
    parser.add_argument("--bootstrap-n", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])
    ts = time.strftime("%Y%m%d_%H%M%S")

    datasets = DATASETS
    if args.subsets:
        datasets = [d for d in DATASETS if d["key"] in args.subsets]

    results = []
    for ds in datasets:
        log.info(f"Processing {ds['name']}...")
        r = compute_predictors(ds["name"], ds["key"], ds["type"], log,
                                n_boot=args.bootstrap_n, seed=args.seed)
        if r:
            r["win"] = ds["win"]
            results.append(r)
            log.info(f"  Δ={r['delta']:+.4f}  ent_auroc={r['entropy_auroc']:.4f}  "
                     f"mean_agr={r['mean_agreement']:.4f}  std_agr={r['std_agreement']:.4f}  "
                     f"pairwise={r['mean_pairwise_argmax_agreement']:.4f}")

    # Try to find the best separating predictor
    from itertools import product
    predictors = ["entropy_auroc", "mean_agreement", "std_agreement",
                  "mean_pairwise_argmax_agreement", "mean_multilabel_degree"]
    log.info(f"\n{'='*70}")
    log.info("REGIME PREDICTOR ANALYSIS")
    log.info(f"{'='*70}")
    log.info(f"\n{'Dataset':20s} {'Δ':>8s} {'CI':>22s} {'Win?':>5s}  " +
              "  ".join(f"{p:>18s}" for p in predictors))
    for r in results:
        ci = r.get('delta_ci', {})
        ci_str = f"[{ci.get('ci_lo',0):+.4f},{ci.get('ci_hi',0):+.4f}]"
        vals = "  ".join(f"{r[p]:>18.4f}" for p in predictors)
        log.info(f"{r['name']:20s} {r['delta']:>+8.4f} {ci_str:>22s} {str(r['win']):>5s}  {vals}")

    # Correlation of each predictor with delta
    deltas = np.array([r["delta"] for r in results])
    log.info(f"\nPredictor correlations with Δ(agreement-entropy):")
    analysis = {}
    for p in predictors:
        vals = np.array([r[p] for r in results])
        if np.std(vals) > 0 and np.std(deltas) > 0:
            corr = float(np.corrcoef(vals, deltas)[0, 1])
        else:
            corr = 0.0
        analysis[p] = {"correlation_with_delta": corr}
        log.info(f"  {p:40s} r = {corr:+.4f}")

    best_p = max(analysis, key=lambda k: abs(analysis[k]["correlation_with_delta"]))
    log.info(f"\nBest predictor: {best_p} (r={analysis[best_p]['correlation_with_delta']:+.4f})")

    # Find threshold on best predictor
    vals = np.array([r[best_p] for r in results])
    wins = np.array([1 if r["win"] else 0 for r in results])
    best_thresh = None; best_acc = 0
    for t in np.linspace(vals.min(), vals.max(), 100):
        pred = (vals < t).astype(int) if analysis[best_p]["correlation_with_delta"] < 0 else (vals > t).astype(int)
        acc = np.mean(pred == wins)
        if acc > best_acc:
            best_acc = acc; best_thresh = float(t)
    log.info(f"Threshold on {best_p}: {'<' if analysis[best_p]['correlation_with_delta'] < 0 else '>'} {best_thresh:.4f} "
             f"-> classification accuracy {best_acc:.2%}")

    report = {"timestamp": ts, "datasets": results, "predictor_analysis": analysis,
              "best_predictor": best_p,
              "best_predictor_correlation": analysis[best_p]["correlation_with_delta"],
              "best_threshold": best_thresh, "threshold_accuracy": best_acc}
    out_json = OUT_DIR / "anchoring_regime_predictor_report.json"
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)
    log.info(f"\nReport: {out_json}")


if __name__ == "__main__":
    main()
