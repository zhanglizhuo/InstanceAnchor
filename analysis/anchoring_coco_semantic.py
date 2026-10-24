"""
analysis/anchoring_coco_semantic.py — COCO intra-category co-occurrence analysis
=================================================================================
Tests whether semantic intra-category co-occurrence (e.g., car+truck in same
image) produces agreement advantage on COCO. Stratifies by:
  1. Multi-label degree (1, 2, 3, 4+ labels)
  2. Intra-category groups (vehicles, kitchenware, animals...)
  3. 2-3 label sweet spot (B1 replication on COCO)

Output: results/anchoring_coco_semantic_report.json
"""
import argparse, json, sys, logging
import numpy as np

sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent))
from pilot_perimage_anchoring import BACKBONES, softmax_entropy, auroc, bootstrap_auroc_diff, OUT_DIR

# COCO category indices (0-based) grouped by supercategory.
# Official COCO 2017 supercategories (verified 2026-08-18, round-4 audit;
# the previous table misplaced backpack/tv/laptop/microwave-group and
# included banana/apple in kitchen).
SEMANTIC_GROUPS = {
    "vehicle":     [1, 2, 3, 5, 6, 7],              # bicycle,car,motorcycle,bus,train,truck
    "kitchen":     [39, 40, 41, 42, 43, 44, 45],    # bottle..bowl
    "animal":      [14, 15, 16, 17, 18, 19, 20, 21, 22, 23],  # bird..giraffe
    "furniture":   [56, 57, 58, 59, 60, 61],        # chair..toilet
    "electronics": [62, 63, 64, 65, 66, 67],        # tv..cell phone
    "sports":      [29, 30, 31, 32, 33, 34, 35, 36, 37, 38],  # frisbee..tennis racket
}
assert sum(len(v) for v in SEMANTIC_GROUPS.values()) == len(
    {c for v in SEMANTIC_GROUPS.values() for c in v}), "groups must not overlap"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap-n", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger(__name__)

    # Load COCO
    models = BACKBONES
    all_l = []; gt = None
    for m in models:
        l = np.load(OUT_DIR / "per_image" / f"clip_{m}_coco_logits.npy")
        lb = np.load(OUT_DIR / "per_image" / f"clip_{m}_coco_labels.npy")
        if gt is None: gt = lb
        all_l.append(l)

    stack = np.stack(all_l, 0); avg = stack.mean(0); cons = np.argmax(avg, 1)
    N = len(gt)
    correct = np.array([cons[i] in np.where(gt[i] > 0)[0] for i in range(N)], dtype=bool)
    ent, _ = softmax_entropy(avg); s_ent = -ent
    agr = (np.argmax(stack,2) == cons[None,:]).mean(0)
    ml_degree = gt.sum(1)

    results = {
        "dataset": "coco", "n_images": N, "n_classes": 80,
        "mean_ml_degree": float(ml_degree.mean()),
        "consensus_accuracy": float(correct.mean()),
        "entropy_auroc": float(auroc(s_ent, correct)),
        "agreement_auroc": float(auroc(agr, correct)),
        "delta_global": float(auroc(agr, correct) - auroc(s_ent, correct)),
        "ml_degree_stratification": [],
        "intra_category_stratification": [],
    }

    log.info(f"COCO: N={N} acc={correct.mean():.4f} Δ={results['delta_global']:+.4f}")

    # 1. Multi-label degree stratification
    for d in range(1, 8):
        mask = ml_degree == d
        if mask.sum() < 30: continue
        c = correct[mask]; e = s_ent[mask]; a = agr[mask]
        if c.std() == 0: continue
        ci = bootstrap_auroc_diff(a, e, c, args.bootstrap_n, args.seed)
        delta = float(auroc(a,c) - auroc(e,c))
        results["ml_degree_stratification"].append({
            "stratum": f"{d} labels", "n": int(mask.sum()),
            "acc": float(c.mean()), "delta": delta, "delta_ci": ci})

    # 2-3 sweet spot
    mask23 = (ml_degree >= 2) & (ml_degree <= 3)
    c = correct[mask23]; e = s_ent[mask23]; a = agr[mask23]
    ci = bootstrap_auroc_diff(a, e, c, args.bootstrap_n, args.seed)
    results["b1_sweet_spot"] = {
        "stratum": "2-3 labels", "n": int(mask23.sum()),
        "acc": float(c.mean()), "delta": float(auroc(a,c)-auroc(e,c)), "delta_ci": ci}

    # 2. Intra-category co-occurrence
    for group_name, class_ids in SEMANTIC_GROUPS.items():
        mask = gt[:, class_ids].sum(1) >= 2
        if mask.sum() < 30: continue
        c = correct[mask]; e = s_ent[mask]; a = agr[mask]
        if c.std() == 0: continue
        ci = bootstrap_auroc_diff(a, e, c, args.bootstrap_n, args.seed)
        delta = float(auroc(a,c) - auroc(e,c))
        # Degenerate resamples (error count < 2 in a bootstrap draw) yield NaN
        # percentiles; emit null instead of non-standard JSON NaN.
        if not all(np.isfinite(v) for v in ci.values()):
            ci = {k: (v if np.isfinite(v) else None) for k, v in ci.items()}
            ci["note"] = "degenerate bootstrap (too few errors in resamples)"
        sign = ("tie" if ci.get("ci_lo") is None or ci.get("ci_hi") is None
                else "agr wins" if ci["ci_lo"] > 0
                else "ent wins" if ci["ci_hi"] < 0 else "tie")
        ci_txt = (f"CI[{ci['ci_lo']:+.4f},{ci['ci_hi']:+.4f}]"
                  if ci.get("ci_lo") is not None and ci.get("ci_hi") is not None
                  else "CI[degenerate]")
        log.info(f"  {group_name:12s}: N={mask.sum():4d} acc={c.mean():.3f} Δ={delta:+.4f} {ci_txt} {sign}")
        results["intra_category_stratification"].append({
            "group": group_name, "class_ids": class_ids, "n": int(mask.sum()),
            "acc": float(c.mean()), "delta": delta, "delta_ci": ci, "verdict": sign})

    # 3. Any vs no intra-category
    has_intra = np.zeros(N, dtype=bool)
    for class_ids in SEMANTIC_GROUPS.values():
        has_intra |= (gt[:, class_ids].sum(1) >= 2)
    for label, mask in [("any intra-group", has_intra), ("no intra-group", ~has_intra)]:
        if mask.sum() < 30: continue
        c = correct[mask]; e = s_ent[mask]; a = agr[mask]
        ci = bootstrap_auroc_diff(a, e, c, args.bootstrap_n, args.seed)
        results[label.replace(" ", "_")] = {
            "n": int(mask.sum()), "acc": float(c.mean()),
            "delta": float(auroc(a,c)-auroc(e,c)), "delta_ci": ci}

    out = OUT_DIR / "anchoring_coco_semantic_report.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"\nReport: {out}")

if __name__ == "__main__":
    main()
