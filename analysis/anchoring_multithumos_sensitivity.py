"""
analysis/anchoring_multithumos_sensitivity.py — MultiTHUMOS backbone
ablation + class-imbalance quantification (reviewer follow-up)
=====================================================================
Motivation (Table 6 / §4.5 reviewer notes):
  1. SigLIP2 collapses on 256×142 frames (Hit@1 ≈ 3.0% vs 15.5–16.7% for
     others).  The pooled per-backbone accuracy p=0.135 in Table 1 is
     therefore dragged down by one backbone.  Question: does κ, p*, and
     the agreement-vs-entropy AUROC delta survive removal of siglip2?
  2. The "extreme class imbalance" claim for MultiTHUMOS in §5 is
     qualitative. Compute quantitative imbalance statistics from GT labels.

Protocol (identical to anchoring_multithumos.py, k = {4,5} backbones):
  - Consensus: argmax of k-backbone averaged logits.
  - Correctness: consensus argmax in any active GT class (multi-label).
  - Signals: agreement (vote fraction vs consensus), entropy (T=100).
  - Primary metric: coupled-bootstrap AUROC delta (B=10,000, seed 42).
  - κ model quantities (same formulas as anchoring_saturation_model.py):
      p        = mean per-backbone Hit@1
      α|wrong  = E[agreement | consensus wrong]
      κ        = α|wrong * (C-1) / (1-p)
      p*       = κ / (C-1+κ)

Output: results/anchoring_multithumos_sensitivity_report.{json,txt}
"""

import argparse, json, logging, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pilot_perimage_anchoring import (  # noqa: E402
    BACKBONES, softmax_entropy, auroc, bootstrap_auroc_diff,
)
from phase0_oracle_analysis import setup_logging, OUT_DIR  # noqa: E402

N_CLASSES = 65


def load_multithumos():
    """Return (stack (K,N,C), labels (N,C))."""
    all_l, gt = [], None
    for m in BACKBONES:
        l = np.load(OUT_DIR / "per_image" / f"clip_{m}_multithumos_logits.npy")
        lb = np.load(OUT_DIR / "per_image" / f"clip_{m}_multithumos_labels.npy")
        if gt is None:
            gt = lb.astype(float)
        elif not np.array_equal(gt, lb):
            raise RuntimeError(f"label mismatch for {m}")
        all_l.append(l)
    return np.stack(all_l, 0), gt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap-n", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    setup_logging(time.strftime("%Y%m%d_%H%M%S"), "anchoring_multithumos_sensitivity")
    log = logging.getLogger(__name__)

    stack, gt = load_multithumos()
    N, C = gt.shape
    gt_pos = gt > 0
    log.info("MultiTHUMOS: N=%d frames, C=%d, K=%d", N, C, stack.shape[0])

    # ---------- class-imbalance statistics (from labels, not predictions) ----------
    label_counts = gt_pos.sum(axis=0)               # per-class positive frames
    cts = np.sort(label_counts)
    imbalance = {
        "min": float(cts[0]),
        "median": float(np.median(cts)),
        "max": float(cts[-1]),
        "max_min_ratio": float(cts[-1] / cts[0]),
        "top10_share": float(cts[-10:].sum() / cts.sum()),
        "top3_share": float(cts[-3:].sum() / cts.sum()),
        "n_classes_lt_100": int((cts < 100).sum()),
        "n_classes_gt_2000": int((cts > 2000).sum()),
    }
    log.info("label counts: min=%.0f median=%.0f max=%.0f ratio=%.1fx top10=%.1f%%",
             imbalance["min"], imbalance["median"], imbalance["max"],
             imbalance["max_min_ratio"], 100 * imbalance["top10_share"])

    # ---------- ablation over {5 backbones (all), 4 (minus siglip2)} ----------
    subsets = {
        "all_5": {
            "backbones": BACKBONES,
            "excluded": None,
        },
        "minus_siglip2": {
            "backbones": [b for b in BACKBONES if b != "siglip2"],
            "excluded": "siglip2",
        },
    }

    results = {}
    for key, cfg in subsets.items():
        bbs = cfg["backbones"]
        K = len(bbs)
        idx = [BACKBONES.index(b) for b in bbs]
        sub = stack[idx]  # (K, N, C)

        avg = sub.mean(0)
        cons = np.argmax(avg, 1)
        correct = np.array([cons[i] in np.where(gt_pos[i])[0] for i in range(N)], dtype=bool)

        ent_p, _ = softmax_entropy(avg)
        s_ent = -ent_p
        agr = (np.argmax(sub, 2) == cons[None, :]).mean(0)

        a_auc = auroc(agr, correct)
        e_auc = auroc(s_ent, correct)
        ci = bootstrap_auroc_diff(agr, s_ent, correct, args.bootstrap_n, args.seed)
        delta = a_auc - e_auc
        verdict = "agreement" if ci["ci_lo"] > 0 else ("entropy" if ci["ci_hi"] < 0 else "tie")

        # per-backbone p
        per_bb = []
        for b in range(K):
            preds = np.argmax(sub[b], 1)
            acc = np.mean([preds[i] in np.where(gt_pos[i])[0] for i in range(N)])
            per_bb.append(acc)
        p = float(np.mean(per_bb))

        # κ
        alpha_wrong = float(agr[~correct].mean()) if (~correct).sum() else float("nan")
        if (1 - p) > 0:
            kappa = alpha_wrong * (C - 1) / (1 - p)
        else:
            kappa = float("nan")
        p_star = kappa / (C - 1 + kappa) if np.isfinite(kappa) else float("nan")

        r = {
            "k_backbones": K,
            "backbones": bbs,
            "excluded": cfg["excluded"],
            "per_backbone_acc_p": p,
            "per_backbone_acc": [float(a) for a in per_bb],
            "consensus_acc": float(correct.mean()),
            "alpha_wrong_E": alpha_wrong,
            "kappa": kappa,
            "p_star": p_star,
            "agreement_auroc": float(a_auc),
            "entropy_auroc": float(e_auc),
            "delta": float(delta),
            "ci_lo": ci["ci_lo"], "ci_hi": ci["ci_hi"],
            "verdict": verdict,
        }
        results[key] = r
        log.info("[%s] K=%d p=%.3f κ=%.1f p*=%.3f cons_acc=%.3f  Δ=%+.4f [%+.4f, %+.4f] → %s",
                 key, K, p, kappa, p_star, r["consensus_acc"],
                 delta, ci["ci_lo"], ci["ci_hi"], verdict)

    report = {
        "timestamp": time.strftime("%Y%m%d_%H%M%S"),
        "dataset": "multithumos",
        "n_frames": N,
        "n_classes": C,
        "label_imbalance": imbalance,
        "protocol": {
            "correctness": "consensus argmax in any active GT",
            "entropy_temperature": 100.0,
            "bootstrap_n": args.bootstrap_n,
            "seed": args.seed,
        },
        "ablation": results,
    }
    out_json = OUT_DIR / "anchoring_multithumos_sensitivity_report.json"
    out_txt = OUT_DIR / "anchoring_multithumos_sensitivity_report.txt"
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)
    with open(out_txt, "w") as f:
        f.write("MultiTHUMOS backbone-ablation + imbalance report\n")
        f.write("=" * 70 + "\n")
        for label, r in results.items():
            f.write(f"[{label}] K={r['k_backbones']} excluded={r['excluded']}\n")
            f.write(f"  per-backbone acc: {['%.3f' % a for a in r['per_backbone_acc']]}\n")
            f.write(f"  p={r['per_backbone_acc_p']:.4f}  consensus_acc={r['consensus_acc']:.4f}\n")
            f.write(f"  kappa={r['kappa']:.2f}  p*={r['p_star']:.3f}\n")
            f.write(f"  AUROC: agreement {r['agreement_auroc']:.4f} vs entropy {r['entropy_auroc']:.4f}\n")
            f.write(f"  Δ={r['delta']:+.4f} CI[{r['ci_lo']:+.4f},{r['ci_hi']:+.4f}] → {r['verdict']}\n")
    log.info(f"Report: {out_json}")

if __name__ == "__main__":
    main()