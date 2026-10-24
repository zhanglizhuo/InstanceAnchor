"""
analysis/anchoring_saturation_model.py — Formalize the correlated-error saturation
====================================================================================
Extends the i.i.d. backbone model from Method §3.2 to the correlated-error case.

Model:
  - E[α | consensus correct] = p             (per-backbone accuracy)
  - E[α | consensus wrong]   = κ(1-p)/(C-1)  (κ ≥ 1 = "error consistency")
  - Discriminative gap: Δ_α = p - κ(1-p)/(C-1)
  - Saturation point (gap = 0): p* = κ / (C-1+κ)

This script empirically estimates κ for each dataset by directly computing
E[α|correct] and E[α|wrong] from .npy files, then compares the predicted
saturation point p* with the empirical consensus accuracy where Δ changes sign.

Output:
  results/anchoring_saturation_model_report.{json,txt}

Reproducibility:
  python analysis/anchoring_saturation_model.py
"""
import json, sys, logging, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase0_oracle_analysis import OUT_DIR, load_clip
from pilot_perimage_anchoring import BACKBONES

log = logging.getLogger(__name__)


def load_v2_logits(dataset_key, backbone):
    p = OUT_DIR / "per_image" / f"clip_{backbone}_{dataset_key}_logits.npy"
    if not p.exists(): return None, None
    return np.load(p), np.load(OUT_DIR / "per_image" / f"clip_{backbone}_{dataset_key}_labels.npy")


def load_cifar_logits(backbone):
    p = OUT_DIR / "per_image" / "cifar100" / f"clip_{backbone}_cifar100_logits.npy"
    if not p.exists(): return None, None
    return np.load(p), np.load(OUT_DIR / "per_image" / "cifar100" / f"clip_{backbone}_cifar100_labels.npy")


def load_coco_logits(backbone):
    p = OUT_DIR / "per_image" / f"clip_{backbone}_coco_logits.npy"
    if not p.exists(): return None, None
    return np.load(p), np.load(OUT_DIR / "per_image" / f"clip_{backbone}_coco_labels.npy")


def load_multithumos_logits(backbone):
    p = OUT_DIR / "per_image" / f"clip_{backbone}_multithumos_logits.npy"
    if not p.exists(): return None, None
    return np.load(p), np.load(OUT_DIR / "per_image" / f"clip_{backbone}_multithumos_labels.npy")


def correctness(cons, gt):
    if gt.ndim == 1:
        return (cons == gt.astype(int))
    N = len(cons)
    return np.array([cons[i] in np.where(gt[i] > 0)[0] for i in range(N)], dtype=bool)


def analyze_dataset(name, key, dtype, log):
    """Compute empirical E[α|correct], E[α|wrong], per-backbone p, and derived κ, p*."""
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
        elif dtype == "coco":
            logits, labels = load_coco_logits(bb)
            if logits is None: return None
        elif dtype == "thumos":
            logits, labels = load_multithumos_logits(bb)
            if logits is None: return None
        if gt is None:
            gt = labels
        elif not np.array_equal(gt, labels):
            raise RuntimeError(f"label mismatch across backbones for {key}")
        all_logits.append(logits)
    logits_stack = np.stack(all_logits, axis=0)
    K, N, C = logits_stack.shape
    avg = logits_stack.mean(0)
    cons = np.argmax(avg, 1)
    correct = correctness(cons, gt)

    backbone_args = np.argmax(logits_stack, axis=2)
    alpha = (backbone_args == cons[None, :]).mean(0)

    # Per-backbone accuracy p (avg over backbones)
    if gt.ndim == 1:
        per_bb_acc = [(backbone_args[b] == gt.astype(int)).mean() for b in range(K)]
    else:
        per_bb_acc = []
        for b in range(K):
            preds = backbone_args[b]
            acc = np.mean([preds[i] in np.where(gt[i] > 0)[0] for i in range(N)])
            per_bb_acc.append(acc)
    p = float(np.mean(per_bb_acc))

    # Empirical E[α|correct], E[α|wrong]
    alpha_correct = float(alpha[correct].mean()) if correct.sum() > 0 else float("nan")
    alpha_wrong   = float(alpha[~correct].mean()) if (~correct).sum() > 0 else float("nan")
    consensus_acc = float(correct.mean())
    gap_empirical = alpha_correct - alpha_wrong

    # Estimate κ: α|wrong = κ(1-p)/(C-1)  =>  κ = α|wrong * (C-1) / (1-p)
    if (1 - p) > 0:
        kappa = alpha_wrong * (C - 1) / (1 - p)
    else:
        kappa = float("inf")

    # Predicted saturation point p* = κ / (C-1+κ)
    p_star = kappa / (C - 1 + kappa) if np.isfinite(kappa) else float("nan")

    # i.i.d. baseline (κ=1) saturation point
    p_star_iid = 1.0 / C

    # i.i.d. predicted gap (for comparison)
    gap_iid = p - (1 - p) / (C - 1)
    # Correlated predicted gap
    gap_corr = p - kappa * (1 - p) / (C - 1)

    return {
        "name": name, "N": N, "C": C, "K": K,
        "per_backbone_acc_p": p,
        "consensus_acc": consensus_acc,
        "alpha_correct_E": alpha_correct,
        "alpha_wrong_E": alpha_wrong,
        "gap_empirical": gap_empirical,
        "gap_iid_predicted": gap_iid,
        "gap_corr_predicted": gap_corr,
        "kappa": kappa,
        "p_star_correlated": p_star,
        "p_star_iid": p_star_iid,
    }


DATASETS = [
    {"name": "TeacherBehavior",   "key": "teacher_behavior",     "type": "scb5"},
    {"name": "HandriseReadWrite", "key": "handrise_readwrite",   "type": "scb5"},
    {"name": "BowTurnHead",       "key": "bow_turnhead",         "type": "scb5"},
    {"name": "VOC2007",           "key": "voc2007",              "type": "v2"},
    {"name": "Stanford40",        "key": "stanford40",           "type": "v2"},
    {"name": "COCO",              "key": "coco",                 "type": "coco"},
    {"name": "CIFAR-100",         "key": "cifar100",             "type": "cifar"},
    {"name": "MultiTHUMOS",       "key": "multithumos",          "type": "thumos"},
]


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])
    ts = time.strftime("%Y%m%d_%H%M%S")

    results = []
    for ds in DATASETS:
        log.info(f"Processing {ds['name']}...")
        r = analyze_dataset(ds["name"], ds["key"], ds["type"], log)
        if r:
            results.append(r)
            log.info(f"  p={r['per_backbone_acc_p']:.3f}  C={r['C']}  "
                     f"E[α|corr]={r['alpha_correct_E']:.3f}  E[α|wrong]={r['alpha_wrong_E']:.3f}  "
                     f"κ={r['kappa']:.2f}  p*_corr={r['p_star_correlated']:.3f}  "
                     f"p*_iid={r['p_star_iid']:.3f}")

    log.info(f"\n{'='*100}")
    log.info("CORRELATED-ERROR SATURATION MODEL")
    log.info(f"{'='*100}")
    log.info(f"\n{'Dataset':22s} {'C':>3s} {'p':>7s} {'p_cons':>7s} "
             f"{'E[α|c]':>8s} {'E[α|w]':>8s} {'gap_emp':>8s} "
             f"{'κ':>6s} {'p*_corr':>9s} {'p*_iid':>9s} "
             f"{'gap_iid':>9s} {'gap_corr':>9s}")
    for r in results:
        log.info(f"{r['name']:22s} {r['C']:>3d} {r['per_backbone_acc_p']:>7.3f} "
                 f"{r['consensus_acc']:>7.3f} {r['alpha_correct_E']:>8.3f} "
                 f"{r['alpha_wrong_E']:>8.3f} {r['gap_empirical']:>+8.3f} "
                 f"{r['kappa']:>6.2f} {r['p_star_correlated']:>9.3f} "
                 f"{r['p_star_iid']:>9.3f} "
                 f"{r['gap_iid_predicted']:>+9.3f} {r['gap_corr_predicted']:>+9.3f}")

    log.info(f"\nInterpretation:")
    log.info(f"  - i.i.d. baseline (κ=1) predicts saturation at p* = 1/C (always low)")
    log.info(f"  - Correlated model predicts p* = κ/(C-1+κ), much higher for κ >> 1")
    log.info(f"  - Saturation boundary observed empirically ~0.75 (B4)")

    report = {"timestamp": ts, "datasets": results,
              "model": {"gap_iid": "p - (1-p)/(C-1)",
                        "gap_correlated": "p - κ(1-p)/(C-1)",
                        "saturation_p_star": "κ/(C-1+κ)"}}
    out_json = OUT_DIR / "anchoring_saturation_model_report.json"
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)
    log.info(f"\nReport: {out_json}")


if __name__ == "__main__":
    main()
