"""
analysis/anchoring_loocv_boundary.py — Leave-one-dataset-out validation of the κ-boundary
==========================================================================================
P1 response to reviewer DA-1 ("the κ-boundary is retro-fitted to the observed outcomes"):

Tests whether the saturation boundary -- parameterized by per-dataset (p, C, kappa) --
predicts the sign of the empirical agreement-minus-entropy AUROC delta OUT OF SAMPLE,
i.e. with the held-out dataset's outcome never seen by the fitted rule.

Rules evaluated (all leave-one-dataset-out across the 8 datasets):
  R1  boundary sign rule   : predict "agreement wins" iff Delta_alpha = p - k(1-p)/(C-1) > 0
                             (the paper's theoretical gap; zero fitted parameters).
  R2  LOO linear regression: fit Delta_emp ~ Delta_alpha on the other 7 datasets,
                             predict the held-out Delta, compare sign.
  R3  LOO threshold rule   : implemented in analysis/anchoring_loocv_threshold.py
                             (zero-crossing threshold fit on the other datasets);
                             NOT computed in this script despite the historical
                             label below.
  R4  LOO regression with all three boundary inputs (p, C, kappa) as features.

Baselines: "always tie" and "always loss" (majority class among the 8 outcomes).

This script ALSO extends the kappa table to the 8th dataset (MultiTHUMOS, C=65),
which is absent from anchoring_saturation_model_report.json, computing kappa from
per-image .npy with the same protocol as anchoring_saturation_model.py.

IMPORTANT honesty caveat (reported, not hidden): p, C, kappa are measured on the
same labeled data as Delta, so this is a *transfer* test of the descriptive
relationship, NOT an ex-ante deployability test (the paper's Sec no-deployable-
diagnostic already disclaims the latter).

Output:
  results/anchoring_loocv_boundary_report.{json,txt}

Reproducibility:
  python analysis/anchoring_loocv_boundary.py
"""
import json, sys, logging, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase0_oracle_analysis import OUT_DIR, load_clip
from pilot_perimage_anchoring import BACKBONES

log = logging.getLogger(__name__)

B = 10000
SEED = 42


def load_v2(dataset_key, bb):
    p = OUT_DIR / "per_image" / f"clip_{bb}_{dataset_key}_logits.npy"
    if not p.exists():
        return None, None
    return (np.load(p),
            np.load(OUT_DIR / "per_image" / f"clip_{bb}_{dataset_key}_labels.npy"))


def load_cifar(bb):
    p = OUT_DIR / "per_image" / "cifar100" / f"clip_{bb}_cifar100_logits.npy"
    if not p.exists():
        return None, None
    return (np.load(p),
            np.load(OUT_DIR / "per_image" / "cifar100" / f"clip_{bb}_cifar100_labels.npy"))


def load_coco(bb):
    p = OUT_DIR / "per_image" / f"clip_{bb}_coco_logits.npy"
    if not p.exists():
        return None, None
    return (np.load(p),
            np.load(OUT_DIR / "per_image" / f"clip_{bb}_coco_labels.npy"))


def load_multithumos(bb):
    p = OUT_DIR / "per_image" / f"clip_{bb}_multithumos_logits.npy"
    if not p.exists():
        return None, None
    return (np.load(p),
            np.load(OUT_DIR / "per_image" / f"clip_{bb}_multithumos_labels.npy"))


def correctness(cons, gt):
    """cons (N,) int argmax, gt (N,) int single-label or (N,C) binary multi-label.
    Multi-label: consensus correct iff its class index is an active GT class,
    i.e. gt[i, cons[i]] > 0 (index-safe; do NOT confuse with gt == class index)."""
    if gt.ndim == 1:
        return cons == gt.astype(int)
    N = len(cons)
    return gt[np.arange(N), cons] > 0


def compute_dataset_boundary(name, loader, log):
    """Compute (p, C, kappa, Delta_alpha, gap_empirical, E[alpha|wrong]) from .npy."""
    all_logits = []
    gt = None
    for bb in BACKBONES:
        logits, labels = loader(bb)
        if logits is None:
            return None
        if gt is None:
            gt = labels
        elif not np.array_equal(gt, labels):
            raise RuntimeError(f"label mismatch for {bb}")
        all_logits.append(logits)
    stack = np.stack(all_logits, axis=0)
    K, N, C = stack.shape
    avg = stack.mean(0)
    cons = np.argmax(avg, 1)
    correct = correctness(cons, gt)

    backbone_args = np.argmax(stack, axis=2)
    alpha = (backbone_args == cons[None, :]).mean(0)

    if gt.ndim == 1:
        per_bb_acc = [(backbone_args[b] == gt.astype(int)).mean() for b in range(K)]
    else:
        per_bb_acc = [gt[np.arange(N), backbone_args[b]].mean() for b in range(K)]
    p = float(np.mean(per_bb_acc))

    alpha_wrong = float(alpha[~correct].mean()) if (~correct).sum() > 0 else float("nan")
    kappa = alpha_wrong * (C - 1) / (1 - p) if (1 - p) > 0 else float("inf")
    gap = p - kappa * (1 - p) / (C - 1) if np.isfinite(kappa) else float("nan")
    gap_emp = float(alpha[correct].mean() - alpha[~correct].mean()) if correct.sum() > 0 else float("nan")

    return {
        "name": name, "N": N, "C": C, "K": K, "p": p,
        "E_alpha_wrong": alpha_wrong, "kappa": kappa,
        "Delta_alpha": gap, "gap_empirical": gap_emp,
    }


def loaders():
    return {
        "TeacherBehavior": lambda bb: load_clip(bb, "teacher_behavior", log),
        "HandriseReadWrite": lambda bb: load_clip(bb, "handrise_readwrite", log),
        "BowTurnHead": lambda bb: load_clip(bb, "bow_turnhead", log),
        "VOC2007": lambda bb: load_v2("voc2007", bb),
        "Stanford40": lambda bb: load_v2("stanford40", bb),
        "COCO": lambda bb: load_coco(bb),
        "CIFAR-100": lambda bb: load_cifar(bb),
        "MultiTHUMOS": lambda bb: load_multithumos(bb),
    }


# Empirical agreement-minus-entropy AUROC delta, read from canonical reports:
#   TB/HRW/BTH   anchoring_main_report.json        (deltas_vs_entropy)
#   VOC/Stanford anchoring_v2_report.json
#   CIFAR-100    anchoring_cifar_report.json
#   COCO         anchoring_coco_stratify_report.json (delta_global)
#   MultiTHUMOS  anchoring_multithumos_report.json   (main.delta)
KEY_TO_NAME = {
    "teacher_behavior": "TeacherBehavior",
    "handrise_readwrite": "HandriseReadWrite",
    "bow_turnhead": "BowTurnHead",
    "voc2007": "VOC2007",
    "stanford40": "Stanford40",
}


def empirical_deltas():
    main = json.load(open(OUT_DIR / "anchoring_main_report.json"))
    v2 = json.load(open(OUT_DIR / "anchoring_v2_report.json"))
    cifar = json.load(open(OUT_DIR / "anchoring_cifar_report.json"))
    coco = json.load(open(OUT_DIR / "anchoring_coco_stratify_report.json"))
    mth = json.load(open(OUT_DIR / "anchoring_multithumos_report.json"))
    out = {}
    for r in main["results"]:
        d = r["deltas_vs_entropy"]["agreement_top1_minus_entropy"]
        out[KEY_TO_NAME[r["subset"]]] = d["mean"]
    for r in v2["results"]:
        d = r["deltas_vs_entropy"]["agreement_top1_minus_entropy"]
        out[KEY_TO_NAME[r["dataset"]]] = d["mean"]
    out["CIFAR-100"] = cifar["results"]["deltas_vs_entropy"]["agreement_top1_minus_entropy"]["mean"]
    out["COCO"] = coco["delta_global"]
    out["MultiTHUMOS"] = mth["main"]["delta"]
    return out


def sign_class(delta, tie_margin=0.015):
    """3-way outcome: win if delta > +margin, loss if delta < -margin, else tie.
    Margin matches the fig9_phase_diagram win/tie/loss convention (0.015)."""
    if delta > tie_margin:
        return "win"
    if delta < -tie_margin:
        return "loss"
    return "tie"


def loocv(datasets, feature_cols, target="delta"):
    """Generic LOO: fit linear regression target ~ features on other 7, predict held-out.
    Returns per-fold predictions + summary."""
    n = len(datasets)
    X = np.array([[d[c] for c in feature_cols] for d in datasets], dtype=float)
    y = np.array([d[target] for d in datasets], dtype=float)
    per_fold = []
    preds = np.empty(n)
    for i in range(n):
        mask = np.ones(n, dtype=bool); mask[i] = False
        A = np.column_stack([X[mask], np.ones(mask.sum())])
        beta, *_ = np.linalg.lstsq(A, y[mask], rcond=None)
        pred = float(np.dot(np.r_[X[i], 1.0], beta))
        preds[i] = pred
        per_fold.append({
            "held_out": datasets[i]["name"],
            "actual_delta": float(y[i]),
            "predicted_delta": pred,
            "sign_agree": bool(np.sign(pred) == np.sign(y[i])),
        })
    sign_acc = float(np.mean([p["sign_agree"] for p in per_fold]))
    # predicted-vs-actual correlation (Pearson; guard degenerate predictions)
    if np.std(preds) > 0:
        r = float(np.corrcoef(preds, y)[0, 1])
    else:
        r = 0.0
    mae = float(np.mean(np.abs(preds - y)))
    return {"n_folds": n, "sign_agreement_accuracy": sign_acc,
            "predicted_actual_correlation": r, "mean_abs_error": mae,
            "per_fold": per_fold}


def bootstrap_sign_acc(preds, actuals, n_boot=B, seed=SEED):
    """Bootstrap CI of sign-agreement accuracy across folds (sample folds w/ replacement)."""
    rng = np.random.RandomState(seed)
    n = len(actuals)
    accs = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.randint(0, n, size=n)
        accs[b] = np.mean(np.sign(preds[idx]) == np.sign(actuals[idx]))
    return {"mean": float(accs.mean()),
            "ci_lo": float(np.percentile(accs, 2.5)),
            "ci_hi": float(np.percentile(accs, 97.5))}


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])
    ts = time.strftime("%Y%m%d_%H%M%S")
    log.info(f"Leave-one-dataset-out validation of the κ-boundary (8 datasets)")

    # 1. Recompute boundary params for all 8 datasets from per-image .npy
    datasets = []
    for name, loader in loaders().items():
        r = compute_dataset_boundary(name, loader, log)
        if r is None:
            log.warning(f"  {name}: data missing, skipping")
            continue
        datasets.append(r)
        log.info(f"  {name:20s} C={r['C']:>3d} p={r['p']:.4f} "
                 f"E[α|w]={r['E_alpha_wrong']:.4f} κ={r['kappa']:8.2f} "
                 f"Δ_α={r['Delta_alpha']:+.4f}")

    # 2. Attach empirical deltas
    deltas = empirical_deltas()
    for d in datasets:
        d["delta"] = deltas[d["name"]]
        d["outcome"] = sign_class(d["delta"])
        log.info(f"  {d['name']:20s} empirical Δ={d['delta']:+.4f} ({d['outcome']})")

    # 3. R1: boundary sign rule (no fitted parameters)
    r1 = []
    for d in datasets:
        pred = d["Delta_alpha"] > 0
        actual = d["outcome"] == "win"
        r1.append({"dataset": d["name"], "predicted_win": bool(pred),
                   "actual_win": bool(actual), "match": bool(pred == actual)})
    r1_acc = float(np.mean([x["match"] for x in r1]))

    # 4. R2/R3: LOO regression on the theoretical gap Δ_α
    r2 = loocv(datasets, ["Delta_alpha"])

    # 5. R4: LOO regression on all boundary inputs (p, C, kappa)
    r4 = loocv(datasets, ["p", "C", "kappa"])

    # 6. Baselines
    n_win = sum(1 for d in datasets if d["outcome"] == "win")
    n_loss = sum(1 for d in datasets if d["outcome"] == "loss")
    n_tie = sum(1 for d in datasets if d["outcome"] == "tie")
    baseline_always_tie = n_tie / len(datasets)
    baseline_always_loss = n_loss / len(datasets)

    # 7. Bootstrap CIs on the LOO sign-accuracy numbers
    preds2 = np.array([p["predicted_delta"] for p in r2["per_fold"]])
    actuals = np.array([p["actual_delta"] for p in r2["per_fold"]])
    ci_r2 = bootstrap_sign_acc(preds2, actuals)
    preds4 = np.array([p["predicted_delta"] for p in r4["per_fold"]])
    ci_r4 = bootstrap_sign_acc(preds4, actuals)

    # 8. In-sample comparison (paper fig9 right panel: gap vs delta)
    gaps = np.array([d["Delta_alpha"] for d in datasets])
    dvs = np.array([d["delta"] for d in datasets])
    rho_in = float(np.corrcoef(gaps, dvs)[0, 1])

    log.info("")
    log.info(f"Outcomes: {n_win} win, {n_tie} tie, {n_loss} loss (margin ±0.015)")
    log.info(f"Baselines: always-tie={baseline_always_tie:.3f}, always-loss={baseline_always_loss:.3f}")
    log.info("")
    log.info(f"R1 (sign rule, no fit):      accuracy={r1_acc:.3f} ({int(r1_acc*len(r1))}/{len(r1)})")
    log.info(f"R2 (LOO reg Δ_α):            sign acc={r2['sign_agreement_accuracy']:.3f} "
             f"({int(r2['sign_agreement_accuracy']*r2['n_folds'])}/{r2['n_folds']}), "
             f"r={r2['predicted_actual_correlation']:+.3f}, MAE={r2['mean_abs_error']:.4f}")
    log.info(f"R4 (LOO reg p,C,κ):          sign acc={r4['sign_agreement_accuracy']:.3f} "
             f"({int(r4['sign_agreement_accuracy']*r4['n_folds'])}/{r4['n_folds']}), "
             f"r={r4['predicted_actual_correlation']:+.3f}, MAE={r4['mean_abs_error']:.4f}")
    log.info(f"In-sample Pearson(gap, Δ) = {rho_in:+.3f}")
    log.info("")
    log.info("R2 per-fold:")
    for p in r2["per_fold"]:
        log.info(f"  {p['held_out']:20s} actual={p['actual_delta']:+.4f} "
                 f"pred={p['predicted_delta']:+.4f}  {'✓' if p['sign_agree'] else '✗'}")

    report = {
        "timestamp": ts,
        "purpose": "P1 reviewer response: leave-one-dataset-out validation of the κ-boundary",
        "caveat": ("p, C, kappa are measured on the same labeled data as Δ; this tests the "
                   "transfer of the descriptive boundary relationship, not ex-ante "
                   "deployability"),
        "tie_margin": 0.015,
        "datasets": datasets,
        "outcome_counts": {"win": n_win, "tie": n_tie, "loss": n_loss},
        "baselines": {"always_tie": baseline_always_tie, "always_loss": baseline_always_loss},
        "R1_boundary_sign_rule": {"accuracy": r1_acc, "n": len(r1), "per_dataset": r1},
        "R2_loocv_regression_gap": {k: v for k, v in r2.items() if k != "per_fold"},
        "R2_loocv_per_fold": r2["per_fold"],
        "R2_bootstrap_sign_ci": ci_r2,
        "R4_loocv_regression_pCkappa": {k: v for k, v in r4.items() if k != "per_fold"},
        "R4_loocv_per_fold": r4["per_fold"],
        "R4_bootstrap_sign_ci": ci_r4,
        "in_sample_gap_delta_pearson": rho_in,
    }
    out_json = OUT_DIR / "anchoring_loocv_boundary_report.json"
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)
    log.info(f"\nReport: {out_json}")


if __name__ == "__main__":
    main()
