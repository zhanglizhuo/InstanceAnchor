"""
analysis/anchoring_coco_hardsubset.py — COCO hard-subset validation of agreement advantage
=========================================================================================
Tests whether agreement outperforms entropy specifically on images with high
co-occurrence ambiguity. We define ambiguity via two orthogonal criteria:

  1. Multi-label degree (>=3 GT labels)
  2. Semantic similarity of co-occurring GT classes
     (COCO supercategory entropy: high → labels span many categories,
      low → concentrated in 1-2 categories → high semantic ambiguity)

The core thesis: TeacherBehavior's agreement advantage is driven by images where
multiple visually/functionally similar classes co-occur in the same spatial context,
causing different backbones to "choose" different co-occurring classes. This is
proxied by (ml_degree >= 3) & (supercat_entropy is low).

If the thesis holds:
  - Hard subset (high ml_deg + low entropy): agreement > entropy
  - Easy subset (low ml_deg | high entropy): entropy >= agreement

Output: results/anchoring_coco_hardsubset_report.{json,txt}
"""

import argparse, json, logging, sys, time
from pathlib import Path
import numpy as np
from scipy.stats import entropy as scipy_entropy

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pilot_perimage_anchoring import BACKBONES, softmax_entropy, auroc, bootstrap_auroc_diff, OUT_DIR
from phase0_oracle_analysis import setup_logging

# COCO supercategory assignment (80 classes, 0-based index)
# Source: COCO 2017 annotation standard (official supercategories).
# Verified 2026-08-18 against the canonical COCO category list; the
# previous table had a one-group shift from `bench` onward (bench/bird
# misassigned, kitchen/food boundary off, etc.) — see round-4 audit.
COCO_SUPERCAT = {
    0: 0,                                                        # person
    1: 1, 2: 1, 3: 1, 5: 1, 6: 1, 7: 1,                          # vehicle: bicycle,car,motorcycle,bus,train,truck
    4: 2, 8: 2, 9: 2, 10: 2, 11: 2, 12: 2, 13: 2,               # outdoor: airplane,boat,traffic light,fire hydrant,stop sign,parking meter,bench
    14: 3, 15: 3, 16: 3, 17: 3, 18: 3, 19: 3, 20: 3, 21: 3, 22: 3, 23: 3,  # animal: bird..giraffe
    24: 4, 25: 4, 26: 4, 27: 4, 28: 4,                           # accessory: backpack,umbrella,handbag,tie,suitcase
    29: 5, 30: 5, 31: 5, 32: 5, 33: 5, 34: 5, 35: 5, 36: 5, 37: 5, 38: 5,  # sports: frisbee..tennis racket
    39: 6, 40: 6, 41: 6, 42: 6, 43: 6, 44: 6, 45: 6,            # kitchen: bottle..bowl
    46: 7, 47: 7, 48: 7, 49: 7, 50: 7, 51: 7, 52: 7, 53: 7, 54: 7, 55: 7,  # food: banana..cake
    56: 8, 57: 8, 58: 8, 59: 8, 60: 8, 61: 8,                   # furniture: chair..toilet
    62: 9, 63: 9, 64: 9, 65: 9, 66: 9, 67: 9,                   # electronic: tv..cell phone
    68: 10, 69: 10, 70: 10, 71: 10, 72: 10,                      # appliance: microwave..refrigerator
    73: 11, 74: 11, 75: 11, 76: 11, 77: 11, 78: 11, 79: 11,      # indoor: book..toothbrush
}
assert len(COCO_SUPERCAT) == 80 and set(COCO_SUPERCAT.values()) == set(range(12))

N_SUPERCAT = 12  # 0-11, official supercategories
SUPERCAT_NAMES = {
    0: "person", 1: "vehicle", 2: "outdoor", 3: "animal",
    4: "accessory", 5: "sports", 6: "kitchen", 7: "food",
    8: "furniture", 9: "electronic", 10: "appliance", 11: "indoor",
}

def supercat_entropy(gt, supercat_map, n_supercat):
    """Entropy over supercategory distribution of GT labels per image."""
    n = len(gt)
    result = np.zeros(n)
    for i in range(n):
        labels = np.where(gt[i] > 0)[0]
        if len(labels) <= 1:
            result[i] = 0.0
        else:
            cats = [supercat_map[l] for l in labels]
            counts = np.zeros(n_supercat)
            for c in cats:
                counts[c] += 1
            p = counts / counts.sum()
            result[i] = scipy_entropy(p)
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap-n", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hard-frac", type=float, default=0.30,
                        help="Fraction of multi-label images to use as hard subset")
    args = parser.parse_args()

    log_path = setup_logging(time.strftime("%Y%m%d_%H%M%S"), "anchoring_coco_hardsubset")
    log = logging.getLogger(__name__)

    log.info("Loading COCO data...")
    models = BACKBONES
    all_l, gt = [], None
    for m in models:
        l = np.load(OUT_DIR / "per_image" / f"clip_{m}_coco_logits.npy")
        lb = np.load(OUT_DIR / "per_image" / f"clip_{m}_coco_labels.npy")
        if gt is None: gt = lb.astype(float)
        all_l.append(l)
    stack = np.stack(all_l, 0)
    avg = stack.mean(0)
    cons = np.argmax(avg, 1)
    N = len(gt)

    # Correctness: argmax in any positive GT
    correct = np.array([cons[i] in np.where(gt[i] > 0)[0] for i in range(N)], dtype=bool)
    ent, _ = softmax_entropy(avg)
    s_ent = -ent  # negate so higher = more confident
    agr = (np.argmax(stack, 2) == cons[None, :]).mean(0)

    ml_deg = gt.sum(1).astype(int)
    sc_ent = supercat_entropy(gt, COCO_SUPERCAT, N_SUPERCAT)

    # Global metrics
    global_delta = float(auroc(agr, correct) - auroc(s_ent, correct))
    log.info(f"COCO: N={N} acc={correct.mean():.4f} ml_deg={ml_deg.mean():.2f}")
    log.info(f"  Global: agr={auroc(agr, correct):.4f} ent={auroc(s_ent, correct):.4f} Δ={global_delta:+.4f}")

    # Stratify by multi-label degree
    log.info("ML-degree stratification:")
    ml_strata = []
    for d in range(1, 8):
        mask = ml_deg == d
        if mask.sum() < 30: continue
        c, e, a = correct[mask], s_ent[mask], agr[mask]
        if c.std() == 0: continue
        ci = bootstrap_auroc_diff(a, e, c, args.bootstrap_n, args.seed)
        delta = float(auroc(a, c) - auroc(e, c))
        sign = "agr" if ci["ci_lo"] > 0 else ("ent" if ci["ci_hi"] < 0 else "tie")
        log.info(f"  {d} labels: N={mask.sum():4d} acc={c.mean():.3f} Δ={delta:+.4f} [{ci['ci_lo']:+.4f},{ci['ci_hi']:+.4f}] {sign}")
        ml_strata.append({"n_labels": d, "n": int(mask.sum()), "acc": float(c.mean()),
                          "delta": delta, "ci_lo": ci["ci_lo"], "ci_hi": ci["ci_hi"], "verdict": sign})

    # Supercat entropy stratification
    sc_bins = [("sc_ent=0 (1 cat)", sc_ent == 0),
               ("0<sc_ent<1", (sc_ent > 0) & (sc_ent < 1)),
               ("sc_ent>=1", sc_ent >= 1)]
    log.info("\nSupercat-entropy stratification:")
    sc_strata = []
    for label, mask in sc_bins:
        if mask.sum() < 30: continue
        c, e, a = correct[mask], s_ent[mask], agr[mask]
        ci = bootstrap_auroc_diff(a, e, c, args.bootstrap_n, args.seed)
        delta = float(auroc(a, c) - auroc(e, c))
        sign = "agr" if ci["ci_lo"] > 0 else ("ent" if ci["ci_hi"] < 0 else "tie")
        log.info(f"  {label:20s}: N={mask.sum():4d} acc={c.mean():.3f} Δ={delta:+.4f} [{ci['ci_lo']:+.4f},{ci['ci_hi']:+.4f}] {sign}")
        sc_strata.append({"label": label, "n": int(mask.sum()), "acc": float(c.mean()),
                          "delta": delta, "ci_lo": ci["ci_lo"], "ci_hi": ci["ci_hi"], "verdict": sign})

    # Joint condition: high ambiguity = ml_deg >= 3 + low supercat entropy (labels concentrated)
    hard_mask = (ml_deg >= 3) & (sc_ent <= np.median(sc_ent[ml_deg >= 3]) if (ml_deg >= 3).sum() > 0 else True)
    easy_mask = ~hard_mask

    # Alternative: use high ml_deg alone, then within that, take most semantically concentrated
    ml3_mask = ml_deg >= 3
    n_ml3 = ml3_mask.sum()
    log.info(f"\nJoint condition: N_ml>=3={n_ml3}")

    if n_ml3 >= 100:
        # Within ml>=3, sort by supercat entropy ascending (most concentrated = most ambiguous)
        sc_ent_ml3 = sc_ent[ml3_mask]
        order = np.argsort(sc_ent_ml3)
        n_hard = max(30, int(n_ml3 * args.hard_frac))

        hard_idx = np.where(ml3_mask)[0][order[:n_hard]]
        easy_idx = np.where(ml3_mask)[0][order[n_hard:]]

        for label, idx in [("hard (high ambig)", hard_idx), ("easy (low ambig)", easy_idx)]:
            if len(idx) < 30: continue
            c, e, a = correct[idx], s_ent[idx], agr[idx]
            ci = bootstrap_auroc_diff(a, e, c, args.bootstrap_n, args.seed)
            delta = float(auroc(a, c) - auroc(e, c))
            sign = "agr" if ci["ci_lo"] > 0 else ("ent" if ci["ci_hi"] < 0 else "tie")
            log.info(f"  {label:20s}: N={len(idx):4d} acc={c.mean():.3f} Δ={delta:+.4f} [{ci['ci_lo']:+.4f},{ci['ci_hi']:+.4f}] {sign}")

        # Final result: comparison on hard subset
        c_h, e_h, a_h = correct[hard_idx], s_ent[hard_idx], agr[hard_idx]
        c_e, e_e, a_e = correct[easy_idx], s_ent[easy_idx], agr[easy_idx]
        ci_h = bootstrap_auroc_diff(a_h, e_h, c_h, args.bootstrap_n, args.seed)
        ci_e = bootstrap_auroc_diff(a_e, e_e, c_e, args.bootstrap_n, args.seed)
        delta_h = float(auroc(a_h, c_h) - auroc(e_h, c_h))
        delta_e = float(auroc(a_e, c_e) - auroc(e_e, c_e))
    else:
        delta_h, delta_e, ci_h, ci_e = 0, 0, {"ci_lo": 0, "ci_hi": 0}, {"ci_lo": 0, "ci_hi": 0}
        hard_idx, easy_idx = np.array([], dtype=int), np.array([], dtype=int)
        log.warning("  Too few ml>=3 images for joint analysis")

    # Also do simple ml_deg threshold
    ml3_delta = 0; ml3_ci = {"ci_lo": 0, "ci_hi": 0}
    if ml3_mask.sum() >= 30:
        c = correct[ml3_mask]; e = s_ent[ml3_mask]; a = agr[ml3_mask]
        ml3_ci = bootstrap_auroc_diff(a, e, c, args.bootstrap_n, args.seed)
        ml3_ci = {"ci_lo": ml3_ci["ci_lo"], "ci_hi": ml3_ci["ci_hi"]}
        ml3_delta = float(auroc(a, c) - auroc(e, c))

    results = {
        "script": "anchoring_coco_hardsubset",
        "dataset": "coco", "n_images": N, "n_classes": 80,
        "consensus_accuracy": float(correct.mean()),
        "hard_fraction": args.hard_frac,
        "global": {"entropy_auroc": float(auroc(s_ent, correct)),
                   "agreement_auroc": float(auroc(agr, correct)),
                   "delta": global_delta},
        "ml_degree_strata": ml_strata,
        "supercat_entropy_strata": sc_strata,
        "ml_ge3": {"n": int(ml3_mask.sum()),
                   "delta": ml3_delta, "delta_ci": ml3_ci},
        "hard_subset": {"n": int(len(hard_idx)), "acc": float(c_h.mean()) if len(hard_idx) > 0 else 0,
                        "entropy_auroc": float(auroc(e_h, c_h)) if len(hard_idx) > 30 else None,
                        "agreement_auroc": float(auroc(a_h, c_h)) if len(hard_idx) > 30 else None,
                        "delta": delta_h, "delta_ci": ci_h},
        "easy_subset": {"n": int(len(easy_idx)), "acc": float(c_e.mean()) if len(easy_idx) > 0 else 0,
                        "entropy_auroc": float(auroc(e_e, c_e)) if len(easy_idx) > 30 else None,
                        "agreement_auroc": float(auroc(a_e, c_e)) if len(easy_idx) > 30 else None,
                        "delta": delta_e, "delta_ci": ci_e},
    }

    out = OUT_DIR / "anchoring_coco_hardsubset_report.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)

    # Human-readable summary
    def _fmt_ci(ci):
        return f"[{ci['ci_lo']:+.4f},{ci['ci_hi']:+.4f}]"

    lines = [
        "=" * 72,
        "  COCO Hard-Subset Validation",
        "=" * 72,
        f"  N={N}  acc={correct.mean():.4f}  hard_frac={args.hard_frac}",
        "",
        "── Global ──",
        f"  Entropy AUROC:   {results['global']['entropy_auroc']:.4f}",
        f"  Agreement AUROC: {results['global']['agreement_auroc']:.4f}",
        f"  Δ:               {results['global']['delta']:+.4f}",
        "",
        "── ML-degree strata ──",
    ]
    for s in ml_strata:
        lines.append(f"  {s['n_labels']} labels (N={s['n']}): Δ={s['delta']:+.4f} {_fmt_ci(s)} {s['verdict']}")
    lines += ["", "── Supercat-entropy strata ──"]
    for s in sc_strata:
        lines.append(f"  {s['label']} (N={s['n']}): Δ={s['delta']:+.4f} {_fmt_ci(s)} {s['verdict']}")
    lines += [
        "",
        "── Joint condition (ml>=3 + supercat concentrated) ──",
        f"  Hard subset (N={results['hard_subset']['n']}): "
        f"Δ={results['hard_subset']['delta']:+.4f} {_fmt_ci(results['hard_subset']['delta_ci'])}",
        f"  Easy subset (N={results['easy_subset']['n']}): "
        f"Δ={results['easy_subset']['delta']:+.4f} {_fmt_ci(results['easy_subset']['delta_ci'])}",
        "",
        "── Verdict ──",
    ]
    if len(hard_idx) >= 30:
        if ci_h["ci_lo"] > 0:
            lines.append("  ✅ Agreement wins on high-ambiguity COCO subset")
        elif ci_h["ci_hi"] < 0:
            lines.append("  ❌ Entropy wins on high-ambiguity COCO subset")
        else:
            lines.append("  ➖ Tie on high-ambiguity COCO subset")
    lines.append("=" * 72)

    txt = "\n".join(lines)
    txt_out = OUT_DIR / "anchoring_coco_hardsubset_report.txt"
    with open(txt_out, "w") as f:
        f.write(txt + "\n")
    print(txt)

if __name__ == "__main__":
    main()
