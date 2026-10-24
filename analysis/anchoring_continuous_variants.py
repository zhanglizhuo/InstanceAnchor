"""
analysis/anchoring_continuous_variants.py — Cross-validate continuous agreement
================================================================================
Earlier conclusion "discrete > continuous" was based on a SINGLE continuous
variant (V1: mean softmax prob on consensus). This script tests MULTIPLE
continuous operationalizations on ALL datasets to find the right story.

Variants:
  V_disc: discrete top-1 vote fraction (current)
  V1: mean softmax prob on consensus class
  V2: 1 - mean pairwise JS divergence
  V3: -Var(logit on consensus class)
  V4: softmax[cons] - mean softmax[non-cons]
  V5: Borda count = mean (C-1-rank_of_cons) / (C-1)

Output:
  results/anchoring_continuous_variants_report.{json,txt}

Reproducibility:
  python analysis/anchoring_continuous_variants.py
"""
import json, sys, logging, time
from pathlib import Path
from itertools import combinations
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase0_oracle_analysis import OUT_DIR, load_clip
from pilot_perimage_anchoring import BACKBONES, TEMPERATURE, softmax_entropy, auroc

log = logging.getLogger(__name__)


def softmax(logits, T=TEMPERATURE):
    z = logits * T
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def js_div(p, q):
    m = 0.5 * (p + q)
    return 0.5 * np.sum(p * np.log2(p/m + 1e-12), axis=-1) + 0.5 * np.sum(q * np.log2(q/m + 1e-12), axis=-1)


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


def correctness(cons, gt):
    if gt.ndim == 1:
        return (cons == gt.astype(int))
    N = len(cons)
    return np.array([cons[i] in np.where(gt[i] > 0)[0] for i in range(N)], dtype=bool)


def compute_all_variants(name, key, dtype, log, T=100.0):
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
        if gt is None: gt = labels
        all_logits.append(logits)
    logits_stack = np.stack(all_logits, axis=0)
    K, N, C = logits_stack.shape
    avg = logits_stack.mean(0)
    cons = np.argmax(avg, 1)
    correct = correctness(cons, gt)
    ent, _ = softmax_entropy(avg)

    # Discrete top-1 vote
    backbone_args = np.argmax(logits_stack, axis=2)
    alpha_disc = (backbone_args == cons[None, :]).mean(0)

    # V1: mean softmax prob on consensus class
    sp = np.stack([softmax(logits_stack[b], T=T) for b in range(K)], axis=0)
    alpha_v1 = np.mean([sp[b, np.arange(N), cons] for b in range(K)], axis=0)

    # V2: 1 - mean pairwise JS divergence
    alphas_v2 = np.ones(N)
    for n in range(N):
        dists = sp[:, n, :]
        jss = [js_div(dists[i], dists[j]) for i, j in combinations(range(K), 2)]
        alphas_v2[n] = 1.0 - np.mean(jss)
    alpha_v2 = alphas_v2

    # V3: negative variance of logit on consensus class
    logit_cons = logits_stack[:, np.arange(N), cons]
    alpha_v3 = -logit_cons.var(0)

    # V5: Borda count = mean (C-1-rank_of_cons) / (C-1)
    ranks = np.argsort(np.argsort(-logits_stack, axis=2), axis=2)
    cons_ranks = ranks[np.arange(K)[:, None], np.arange(N), cons]
    alpha_v5 = (C - 1 - cons_ranks).mean(0) / (C - 1)

    return {
        "name": name, "N": N, "C": C, "K": K,
        "correct": correct,
        "entropy": -ent,
        "alpha_disc": alpha_disc,
        "alpha_v1": alpha_v1,
        "alpha_v2": alpha_v2,
        "alpha_v3": alpha_v3,
        "alpha_v5": alpha_v5,
    }


DATASETS = [
    {"name": "TeacherBehavior",   "key": "teacher_behavior",     "type": "scb5"},
    {"name": "HandriseReadWrite", "key": "handrise_readwrite",   "type": "scb5"},
    {"name": "BowTurnHead",       "key": "bow_turnhead",         "type": "scb5"},
    {"name": "VOC2007",           "key": "voc2007",              "type": "v2"},
    {"name": "Stanford40",        "key": "stanford40",           "type": "v2"},
    {"name": "COCO",              "key": "coco",                 "type": "coco"},
    {"name": "CIFAR-100",         "key": "cifar100",             "type": "cifar"},
]


def boot_delta(a, b, correct, n_boot=10000, seed=42):
    """Bootstrap CI for AUROC(a, correct) - AUROC(b, correct)."""
    rng = np.random.RandomState(seed)
    N = len(correct)
    deltas = []
    for _ in range(n_boot):
        idx = rng.randint(0, N, size=N)
        deltas.append(auroc(a[idx], correct[idx]) - auroc(b[idx], correct[idx]))
    deltas = np.sort(deltas)
    return {"mean": float(np.mean(deltas)),
            "ci_lo": float(deltas[int(0.025 * n_boot)]),
            "ci_hi": float(deltas[int(0.975 * n_boot)])}


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])
    ts = time.strftime("%Y%m%d_%H%M%S")

    results = []
    for ds in DATASETS:
        log.info(f"Processing {ds['name']}...")
        r = compute_all_variants(ds["name"], ds["key"], ds["type"], log)
        if r is None:
            continue
        correct = r["correct"]
        # AUROCs
        ent_auroc = auroc(r["entropy"], correct)
        aurocs = {
            "entropy": float(ent_auroc),
            "discrete": float(auroc(r["alpha_disc"], correct)),
            "v1_soft_prob": float(auroc(r["alpha_v1"], correct)),
            "v2_inv_js": float(auroc(r["alpha_v2"], correct)),
            "v3_neg_var": float(auroc(r["alpha_v3"], correct)),
            "v5_borda": float(auroc(r["alpha_v5"], correct)),
        }
        # Bootstrap CIs: V5 vs discrete, V5 vs entropy, V1 vs discrete
        deltas = {
            "v5_minus_disc": boot_delta(r["alpha_v5"], r["alpha_disc"], correct),
            "v5_minus_ent":  boot_delta(r["alpha_v5"], r["entropy"], correct),
            "v1_minus_disc": boot_delta(r["alpha_v1"], r["alpha_disc"], correct),
            "disc_minus_ent": boot_delta(r["alpha_disc"], r["entropy"], correct),
        }
        out = {"name": r["name"], "N": r["N"], "C": r["C"], "aurocs": aurocs, "deltas": deltas}
        results.append(out)
        log.info(f"  ent={ent_auroc:.4f}  disc={aurocs['discrete']:.4f}  v1={aurocs['v1_soft_prob']:.4f}  "
                 f"v2={aurocs['v2_inv_js']:.4f}  v3={aurocs['v3_neg_var']:.4f}  v5={aurocs['v5_borda']:.4f}")

    log.info(f"\n{'='*90}")
    log.info("CROSS-VALIDATION: continuous agreement variants on all datasets")
    log.info(f"{'='*90}")
    log.info(f"\n{'Dataset':22s} {'ent':>7s} {'disc':>7s} {'v1':>7s} {'v2':>7s} {'v3':>7s} {'v5_borda':>9s}  "
             f"{'Δ(v5-disc)':>14s}  {'Δ(v5-ent)':>14s}")
    for r in results:
        d5d = r["deltas"]["v5_minus_disc"]
        d5e = r["deltas"]["v5_minus_ent"]
        log.info(f"{r['name']:22s} {r['aurocs']['entropy']:>7.4f} {r['aurocs']['discrete']:>7.4f} "
                 f"{r['aurocs']['v1_soft_prob']:>7.4f} {r['aurocs']['v2_inv_js']:>7.4f} "
                 f"{r['aurocs']['v3_neg_var']:>7.4f} {r['aurocs']['v5_borda']:>9.4f}  "
                 f"{d5d['mean']:>+9.4f} [{d5d['ci_lo']:+.3f},{d5d['ci_hi']:+.3f}]  "
                 f"{d5e['mean']:>+9.4f} [{d5e['ci_lo']:+.3f},{d5e['ci_hi']:+.3f}]")

    report = {"timestamp": ts, "datasets": results}
    out_json = OUT_DIR / "anchoring_continuous_variants_report.json"
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)
    log.info(f"\nReport: {out_json}")


if __name__ == "__main__":
    main()
