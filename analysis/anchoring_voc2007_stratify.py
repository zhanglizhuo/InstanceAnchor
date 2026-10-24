"""
analysis/anchoring_voc2007_stratify.py — VOC2007 multi-label degree stratification
====================================================================================
Tests whether the co-occurrence ambiguity mechanism (B1) generalizes to VOC2007.
Key question: does agreement beat entropy on VOC2007 images with 2-3 GT labels?

Output: results/anchoring_voc2007_stratify_report.json
"""
import argparse, json, sys, time, logging
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pilot_perimage_anchoring import BACKBONES, softmax_entropy, auroc, bootstrap_auroc_diff, OUT_DIR

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap-n", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger(__name__)

    # Load VOC2007
    models = BACKBONES
    all_l = []; gt = None
    for m in models:
        l = np.load(OUT_DIR / "per_image" / f"clip_{m}_voc2007_logits.npy")
        lb = np.load(OUT_DIR / "per_image" / f"clip_{m}_voc2007_labels.npy")
        if gt is None: gt = lb
        all_l.append(l)

    stack = np.stack(all_l, 0)
    K, N, C = stack.shape
    avg = stack.mean(0)
    cons = np.argmax(avg, 1)
    correct = np.array([cons[i] in np.where(gt[i] > 0)[0] for i in range(N)], dtype=bool)

    ent, _ = softmax_entropy(avg)
    s_ent = -ent
    backbone_args = np.argmax(stack, 2)
    agr = (backbone_args == cons[None, :]).mean(0)

    ml_degree = gt.sum(1)
    results = {"dataset": "voc2007", "n_images": N, "n_classes": C,
               "consensus_accuracy": float(correct.mean()),
               "entropy_auroc_global": float(auroc(s_ent, correct)),
               "agreement_auroc_global": float(auroc(agr, correct)),
               "stratification": []}

    log.info(f"VOC2007: N={N} acc={correct.mean():.4f} "
             f"ent={auroc(s_ent, correct):.4f} agr={auroc(agr, correct):.4f}")

    strata = [
        ("1 label", ml_degree == 1),
        ("2 labels", ml_degree == 2),
        ("3 labels", ml_degree == 3),
        ("4+ labels", ml_degree >= 4),
        ("2-3 labels (B1 sweet spot)", (ml_degree >= 2) & (ml_degree <= 3)),
        ("single-label", ml_degree == 1),
    ]

    for label, mask in strata:
        n = int(mask.sum())
        if n < 30:
            continue
        c = correct[mask]; e = s_ent[mask]; a = agr[mask]
        if c.std() == 0:
            log.info(f"  {label}: N={n} acc={c.mean():.4f} (degenerate)")
            results["stratification"].append({"stratum": label, "n": n, "acc": float(c.mean()),
                                               "degenerate": True})
            continue
        ae = auroc(e, c); aa = auroc(a, c)
        ci = bootstrap_auroc_diff(a, e, c, args.bootstrap_n, args.seed)
        delta = aa - ae
        sign = "agreement wins" if ci["ci_lo"] > 0 else ("entropy wins" if ci["ci_hi"] < 0 else "tie")
        log.info(f"  {label:25s} N={n:4d} acc={c.mean():.4f} Δ={delta:+.4f} "
                 f"CI[{ci['ci_lo']:+.4f},{ci['ci_hi']:+.4f}] → {sign}")
        results["stratification"].append({
            "stratum": label, "n": n, "acc": float(c.mean()),
            "entropy_auroc": float(ae), "agreement_auroc": float(aa),
            "delta": float(delta), "delta_ci": ci, "verdict": sign,
        })

    out = OUT_DIR / "anchoring_voc2007_stratify_report.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"\nReport: {out}")

if __name__ == "__main__":
    main()
