"""
analysis/anchoring_multithumos.py — Type A replication on MultiTHUMOS
=====================================================================
MultiTHUMOS (Yeung et al., IJCV 2018): dense frame-level multi-label action
annotations over THUMOS14 validation videos — 200 videos, 65 classes, avg
~1.5 labels per frame. This is the only public dataset whose annotation
structure matches TeacherBehavior's Type A "annotation multiplicity"
(competing labels for the same visual evidence, e.g. Stand + Run + Fall
describe the same person's body).

Research question (Exp G2): does cross-backbone agreement flag CLIP errors
better than 5-backbone ensemble entropy on a *second* Type A dataset?

Protocol (identical to anchoring_main on SCB5):
  - Frames sampled at 1 fps, labels aligned via [s, e) intervals (see
    experiments/extract_clip_multithumos.py).
  - Consensus prediction: argmax of 5-backbone averaged logits.
  - Correctness: consensus argmax in any active GT class (multi-label).
  - Signals: agreement (vote fraction vs consensus), entropy (T=100).
  - Primary metric: coupled-bootstrap AUROC delta (B=10,000).
  - Selective accuracy at 30/50/70% coverage via risk-coverage curves.

Expected outcome per the paper's taxonomy:
  - If Type A structure is the driver: agreement > entropy (positive Δ),
    replicating TeacherBehavior's +0.067 rather than COCO's −0.014.
  - Caveat: C=65 vs TB's 8 → saturation model predicts a smaller edge.

Output: results/anchoring_multithumos_report.{json,txt}
"""

import argparse, json, logging, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pilot_perimage_anchoring import (  # noqa: E402
    BACKBONES, softmax_entropy, auroc, bootstrap_auroc_diff,
    risk_coverage, aupr,
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

    setup_logging(time.strftime("%Y%m%d_%H%M%S"), "anchoring_multithumos")
    log = logging.getLogger(__name__)

    stack, gt = load_multithumos()
    N = len(gt)
    log.info("MultiTHUMOS: N=%d frames, C=%d, K=%d", N, N_CLASSES, stack.shape[0])

    avg = stack.mean(0)
    cons = np.argmax(avg, 1)
    gt_pos = gt > 0
    correct = np.array([cons[i] in np.where(gt_pos[i])[0] for i in range(N)], dtype=bool)

    ent, _ = softmax_entropy(avg)
    s_ent = -ent
    agr = (np.argmax(stack, 2) == cons[None, :]).mean(0)

    n_ml = gt_pos.sum(1)
    log.info("consensus acc=%.4f  ml_labels/frame=%.2f  multilabel_frac=%.3f",
             correct.mean(), n_ml.mean(), (n_ml > 1).mean())
    for v in [2, 3, 4, 5]:
        log.info("  ml>=%d: %d frames (%.1f%%)", v, (n_ml >= v).sum(), 100 * (n_ml >= v).mean())

    # --- main result: agreement vs entropy ---
    a_auc = auroc(agr, correct)
    e_auc = auroc(s_ent, correct)
    ci = bootstrap_auroc_diff(agr, s_ent, correct, args.bootstrap_n, args.seed)
    delta = a_auc - e_auc
    verdict = "agreement" if ci["ci_lo"] > 0 else ("entropy" if ci["ci_hi"] < 0 else "tie")
    log.info("agreement AUROC=%.4f  entropy AUROC=%.4f  Δ=%+.4f [%+.4f,%+.4f] → %s",
             a_auc, e_auc, delta, ci["ci_lo"], ci["ci_hi"], verdict)
    log.info("AUPR: agreement=%.4f entropy=%.4f", aupr(agr, correct), aupr(s_ent, correct))

    # --- selective accuracy ---
    covs = [0.3, 0.5, 0.7]
    sel_a = risk_coverage(agr, correct, covs)
    sel_e = risk_coverage(s_ent, correct, covs)
    for c in covs:
        k = f"{int(c*100)}pct"
        log.info("coverage %d%%: agreement=%.4f entropy=%.4f (Δ=%+.4f)",
                 int(c * 100), sel_a[k]["accuracy"], sel_e[k]["accuracy"],
                 sel_a[k]["accuracy"] - sel_e[k]["accuracy"])

    # --- multi-label degree stratification (mechanism check) ---
    strata = []
    for d in [1, 2, 3, 4]:
        mask = n_ml == d
        if mask.sum() < 100:
            continue
        c_ = correct[mask]
        if c_.std() == 0:
            continue
        ci_d = bootstrap_auroc_diff(agr[mask], s_ent[mask], c_, args.bootstrap_n, args.seed)
        d_delta = auroc(agr[mask], c_) - auroc(s_ent[mask], c_)
        sign = "agr" if ci_d["ci_lo"] > 0 else ("ent" if ci_d["ci_hi"] < 0 else "tie")
        log.info("  %d label(s): N=%5d acc=%.3f Δ=%+.4f [%+.4f,%+.4f] %s",
                 d, mask.sum(), c_.mean(), d_delta, ci_d["ci_lo"], ci_d["ci_hi"], sign)
        strata.append({"n_labels": d, "n": int(mask.sum()), "acc": float(c_.mean()),
                       "delta": d_delta, "ci_lo": ci_d["ci_lo"], "ci_hi": ci_d["ci_hi"],
                       "verdict": sign})

    results = {
        "script": "anchoring_multithumos",
        "dataset": "multithumos", "n_frames": int(N), "n_classes": N_CLASSES,
        "n_backbones": len(BACKBONES),
        "consensus_accuracy": float(correct.mean()),
        "ml_labels_per_frame": float(n_ml.mean()),
        "multilabel_frac": float((n_ml > 1).mean()),
        "main": {
            "agreement_auroc": float(a_auc), "entropy_auroc": float(e_auc),
            "delta": float(delta), "ci_lo": ci["ci_lo"], "ci_hi": ci["ci_hi"],
            "verdict": verdict,
            "agreement_aupr": float(aupr(agr, correct)),
            "entropy_aupr": float(aupr(s_ent, correct)),
        },
        "selective": {
            k: {"agreement": sel_a[k]["accuracy"], "entropy": sel_e[k]["accuracy"],
                "delta": sel_a[k]["accuracy"] - sel_e[k]["accuracy"]}
            for k in [f"{int(c*100)}pct" for c in covs]
        },
        "ml_strata": strata,
        "protocol": {
            "frame_sampling": "1 fps",
            "label_alignment": "annotation intervals [s,e) in seconds",
            "correctness": "consensus argmax in any active GT",
            "entropy_temperature": 100.0,
            "bootstrap_n": args.bootstrap_n, "seed": args.seed,
        },
    }

    out = OUT_DIR / "anchoring_multithumos_report.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)

    lines = [
        "=" * 72,
        "  MultiTHUMOS — Type A replication (frame-level multi-label)",
        "=" * 72,
        f"  N={N}  C={N_CLASSES}  acc={correct.mean():.4f}",
        f"  ml_labels/frame={n_ml.mean():.2f}  multilabel_frac={(n_ml>1).mean():.3f}",
        "",
        f"  Agreement AUROC: {a_auc:.4f}",
        f"  Entropy AUROC:   {e_auc:.4f}",
        f"  Δ: {delta:+.4f} [{ci['ci_lo']:+.4f},{ci['ci_hi']:+.4f}]  → {verdict}",
        f"  AUPR: agreement={aupr(agr, correct):.4f}  entropy={aupr(s_ent, correct):.4f}",
        "",
        "  Selective accuracy:",
    ]
    for c in covs:
        k = f"{int(c*100)}pct"
        lines.append(f"    {k}: agreement={sel_a[k]['accuracy']:.4f}  entropy={sel_e[k]['accuracy']:.4f}"
                     f"  Δ={sel_a[k]['accuracy']-sel_e[k]['accuracy']:+.4f}")
    lines.append("")
    lines.append("  ML-degree strata:")
    for s in strata:
        lines.append(f"    {s['n_labels']} label(s): N={s['n']:5d} acc={s['acc']:.3f} "
                     f"Δ={s['delta']:+.4f} [{s['ci_lo']:+.4f},{s['ci_hi']:+.4f}] {s['verdict']}")
    lines.append("")
    lines.append("  Verdict (vs TB +0.067, COCO −0.014):")
    if ci["ci_lo"] > 0:
        lines.append("    ✅ Agreement wins on MultiTHUMOS — Type A replication")
    elif ci["ci_hi"] < 0:
        lines.append("    ❌ Entropy wins on MultiTHUMOS")
    else:
        lines.append("    ➖ Tie on MultiTHUMOS")
    lines.append("=" * 72)

    txt = "\n".join(lines)
    txt_out = OUT_DIR / "anchoring_multithumos_report.txt"
    with open(txt_out, "w") as f:
        f.write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
