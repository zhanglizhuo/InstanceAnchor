"""
analysis/anchoring_loocv_threshold.py — LOOCV validation of the alpha→Δ relationship
=====================================================================================
Addresses the criticism that the 0.61 mean-agreement threshold is post-hoc.
Performs leave-one-out cross-validation on the regression Δ ~ alpha:
for each of the 8 data points (6 datasets + TB under CAPE-B + TB under CAPE-C),
fit the linear regression on the other 7 and predict Δ on the held-out point.

We report:
  - Sign-agreement accuracy (does predicted Δ have the right sign?)
  - Predicted-vs-actual correlation
  - The threshold (where predicted Δ crosses 0) across all 8 LOOCV folds

Output:
  results/anchoring_loocv_threshold_report.{json,txt}

Reproducibility:
  python analysis/anchoring_loocv_threshold.py
"""
import json, sys, logging
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase0_oracle_analysis import OUT_DIR

log = logging.getLogger(__name__)

BACKBONES = ["openai", "laion", "siglip2", "eva02", "dfn"]


def compute_mean_agreement(subset_key, backbone_list, cape_suffix=""):
    """Compute mean per-image agreement for a subset from per-image .npy files."""
    all_logits = []
    for bb in backbone_list:
        p = OUT_DIR / "per_image" / f"clip_{bb}_{subset_key}_logits{cape_suffix}.npy"
        if not p.exists():
            return None
        all_logits.append(np.load(p))
    logits_stack = np.stack(all_logits, axis=0)
    avg = logits_stack.mean(0)
    cons = np.argmax(avg, 1)
    backbone_args = np.argmax(logits_stack, axis=2)
    alpha = (backbone_args == cons[None, :]).mean(0)
    return float(np.mean(alpha))


def gather_points():
    """Gather (name, mean_alpha, delta, ci_lo, ci_hi) for all 8 data points."""
    # Pull 6 base datasets from regime predictor report + bootstrap CIs
    reg = json.load(open(OUT_DIR / "anchoring_regime_predictor_report.json"))
    # For CIs, use the original source reports
    main = json.load(open(OUT_DIR / "anchoring_main_report.json"))
    v2 = json.load(open(OUT_DIR / "anchoring_v2_report.json"))
    cifar = json.load(open(OUT_DIR / "anchoring_cifar_report.json"))

    ci_lookup = {}
    for r in main["results"]:
        de = r.get("deltas_vs_entropy", {}).get("agreement_top1_minus_entropy", {})
        ci_lookup[r["subset"]] = (de.get("ci_lo"), de.get("ci_hi"))
    for r in v2["results"]:
        de = r.get("deltas_vs_entropy", {}).get("agreement_top1_minus_entropy", {})
        ci_lookup[r["dataset"]] = (de.get("ci_lo"), de.get("ci_hi"))
    de_c = cifar.get("results", {}).get("deltas_vs_entropy", {}).get("agreement_top1_minus_entropy", {})
    ci_lookup["cifar100"] = (de_c.get("ci_lo"), de_c.get("ci_hi"))

    points = []
    for d in reg["datasets"]:
        # Map regime-predictor dataset name to ci_lookup key
        name_to_key = {
            "TeacherBehavior": "teacher_behavior",
            "HandriseReadWrite": "handrise_readwrite",
            "BowTurnHead": "bow_turnhead",
            "VOC2007": "voc2007",
            "Stanford40": "stanford40",
            "CIFAR-100": "cifar100",
        }
        k = name_to_key[d["name"]]
        ci_lo, ci_hi = ci_lookup.get(k, (None, None))
        points.append({
            "name": d["name"],
            "mean_alpha": d["mean_agreement"],
            "delta": d["delta"],
            "ci_lo": ci_lo,
            "ci_hi": ci_hi,
        })

    # CAPE-B and CAPE-C on TB
    rob = json.load(open(OUT_DIR / "anchoring_robustness_report.json"))
    for cape in ["B", "C"]:
        r = rob["results"][cape]["teacher_behavior"]
        alpha = compute_mean_agreement("teacher_behavior", BACKBONES, cape_suffix=f"_cape{cape.lower()}")
        points.append({
            "name": f"TB-CAPE-{cape}",
            "mean_alpha": alpha,
            "delta": r["delta"],
            "ci_lo": r["delta_ci_lo"],
            "ci_hi": r["delta_ci_hi"],
        })

    return points


def loocv_regression(points):
    """LOOCV on linear regression Δ = a + b * alpha."""
    alphas = np.array([p["mean_alpha"] for p in points])
    deltas = np.array([p["delta"] for p in points])
    n = len(points)

    per_point = []
    pred_deltas = []
    actual_deltas = []
    thresholds = []

    for i in range(n):
        mask = np.ones(n, dtype=bool); mask[i] = False
        a_tr = alphas[mask]; d_tr = deltas[mask]
        # Linear regression: d = b*a + c
        b, c = np.polyfit(a_tr, d_tr, 1)
        # Predict held-out
        pred = b * alphas[i] + c
        actual = deltas[i]
        pred_deltas.append(pred); actual_deltas.append(actual)
        # Threshold where predicted delta = 0: -c / b
        thr = -c / b if b != 0 else float("inf")
        thresholds.append(thr)
        # Sign agreement
        sign_agree = (np.sign(pred) == np.sign(actual)) or (pred == 0 and actual == 0)
        per_point.append({
            "held_out": points[i]["name"],
            "held_out_alpha": float(alphas[i]),
            "actual_delta": float(actual),
            "predicted_delta": float(pred),
            "sign_agree": bool(sign_agree),
            "fold_threshold": float(thr),
            "fold_slope": float(b),
            "fold_intercept": float(c),
        })

    pred_deltas = np.array(pred_deltas); actual_deltas = np.array(actual_deltas)
    thresholds = np.array(thresholds)
    r_pred_actual = float(np.corrcoef(pred_deltas, actual_deltas)[0, 1]) if np.std(pred_deltas) > 0 else 0.0
    mae = float(np.mean(np.abs(pred_deltas - actual_deltas)))

    return {
        "n_folds": n,
        "sign_agreement_accuracy": float(np.mean([p["sign_agree"] for p in per_point])),
        "predicted_actual_correlation": r_pred_actual,
        "mean_abs_error": mae,
        "threshold_mean": float(np.mean(thresholds)),
        "threshold_min": float(np.min(thresholds)),
        "threshold_max": float(np.max(thresholds)),
        "threshold_std": float(np.std(thresholds)),
        "threshold_full_data": float(-np.polyfit(alphas, deltas, 1)[1] / np.polyfit(alphas, deltas, 1)[0]),
        "per_point": per_point,
    }


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])

    points = gather_points()
    log.info(f"Gathered {len(points)} data points:")
    for p in points:
        log.info(f"  {p['name']:20s}  alpha={p['mean_alpha']:.4f}  delta={p['delta']:+.4f}  "
                 f"CI[{p['ci_lo']:+.4f},{p['ci_hi']:+.4f}]")

    result = loocv_regression(points)
    log.info(f"\nLOOCV Regression Results (Δ ~ alpha):")
    log.info(f"  Sign-agreement accuracy: {result['sign_agreement_accuracy']:.2%} "
             f"({int(result['sign_agreement_accuracy']*result['n_folds'])}/{result['n_folds']})")
    log.info(f"  Predicted-vs-actual r: {result['predicted_actual_correlation']:+.4f}")
    log.info(f"  Mean abs error: {result['mean_abs_error']:.4f}")
    log.info(f"  Threshold (full data): {result['threshold_full_data']:.4f}")
    log.info(f"  LOOCV threshold range: [{result['threshold_min']:.4f}, {result['threshold_max']:.4f}]")
    log.info(f"  LOOCV threshold mean ± std: {result['threshold_mean']:.4f} ± {result['threshold_std']:.4f}")
    log.info(f"\nPer-point:")
    for r in result["per_point"]:
        log.info(f"  {r['held_out']:20s}  alpha={r['held_out_alpha']:.4f}  "
                 f"actual={r['actual_delta']:+.4f}  pred={r['predicted_delta']:+.4f}  "
                 f"{'✓' if r['sign_agree'] else '✗'}  thr={r['fold_threshold']:.4f}")

    report = {"points": points, "loocv": result}
    out_json = OUT_DIR / "anchoring_loocv_threshold_report.json"
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)
    log.info(f"\nReport: {out_json}")


if __name__ == "__main__":
    main()
