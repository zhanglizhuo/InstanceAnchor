"""
analysis/anchoring_continuous_agreement.py — Continuous agreement variant
=========================================================================
Addresses the criticism that the discrete agreement signal (6 distinct values
for 5 backbones) loses ranking information.

We test a continuous analog:
  α_soft(x) = (1/K) Σ_b softmax(ℓ_b(x)/T)[ŷ_cons(x)]
i.e., the average probability mass each backbone assigns to the consensus
class. This preserves the "backbone supports consensus" semantics but with
continuous values, breaking ties.

We compare AUROC of:
  - α_discrete (current, 6 values)
  - α_soft (continuous)
  - entropy (compute-matched baseline)

Output:
  results/anchoring_continuous_agreement_report.{json,txt}

Reproducibility:
  python analysis/anchoring_continuous_agreement.py
"""
import argparse, json, sys, logging, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase0_oracle_analysis import OUT_DIR, load_clip
from pilot_perimage_anchoring import BACKBONES, TEMPERATURE, softmax_entropy, auroc

log = logging.getLogger(__name__)


def softmax(logits, T=TEMPERATURE):
    """Row-wise softmax with temperature."""
    z = logits * T
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def load_v2_logits(dataset_key, backbone):
    p = OUT_DIR / "per_image" / f"clip_{backbone}_{dataset_key}_logits.npy"
    if not p.exists():
        return None, None
    logits = np.load(p)
    labels = np.load(OUT_DIR / "per_image" / f"clip_{backbone}_{dataset_key}_labels.npy")
    return logits, labels


def load_cifar_logits(backbone):
    p = OUT_DIR / "per_image" / "cifar100" / f"clip_{backbone}_cifar100_logits.npy"
    if not p.exists():
        return None, None
    logits = np.load(p)
    labels = np.load(OUT_DIR / "per_image" / "cifar100" / f"clip_{backbone}_cifar100_labels.npy")
    return logits, labels


def load_coco_logits(backbone):
    p = OUT_DIR / "per_image" / f"clip_{backbone}_coco_logits.npy"
    if not p.exists():
        return None, None
    logits = np.load(p)
    labels = np.load(OUT_DIR / "per_image" / f"clip_{backbone}_coco_labels.npy")
    return logits, labels


def correctness(cons, gt):
    """Multi-label aware correctness: predicted class is in any GT>0."""
    if gt.ndim == 1:
        return (cons == gt.astype(int))
    elif gt.ndim == 2:
        N = len(cons)
        return np.array([cons[i] in np.where(gt[i] > 0)[0] for i in range(N)], dtype=bool)
    else:
        raise ValueError("Unexpected gt ndim")


def compute_metrics(name, key, dtype, log, n_boot=10000, seed=42):
    """Compute discrete + continuous agreement and entropy AUROC."""
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
    logits_stack = np.stack(all_logits, axis=0)  # (K, N, C)
    K, N, C = logits_stack.shape
    avg = logits_stack.mean(0)
    cons = np.argmax(avg, 1)
    correct = correctness(cons, gt)

    # Entropy (on averaged logits)
    ent, _ = softmax_entropy(avg)

    # Discrete agreement (current)
    backbone_args = np.argmax(logits_stack, axis=2)  # (K, N)
    alpha_disc = (backbone_args == cons[None, :]).mean(0)

    # Continuous agreement (soft): avg softmax prob on consensus class
    soft_probs = np.stack([softmax(logits_stack[b]) for b in range(K)], axis=0)  # (K, N, C)
    # For each image, take soft_probs[b, n, cons[n]]
    alpha_soft = np.mean([soft_probs[b, np.arange(N), cons] for b in range(K)], axis=0)

    # AUROCs
    auroc_ent = auroc(-ent, correct)
    auroc_disc = auroc(alpha_disc, correct)
    auroc_soft = auroc(alpha_soft, correct)

    # Bootstrap CIs for Δ
    rng = np.random.RandomState(seed)
    def boot(metric_fn):
        vals = []
        for _ in range(n_boot):
            idx = rng.randint(0, N, size=N)
            vals.append(metric_fn(idx))
        vals = np.sort(vals)
        return {"mean": float(np.mean(vals)),
                "ci_lo": float(vals[int(0.025 * n_boot)]),
                "ci_hi": float(vals[int(0.975 * n_boot)])}

    delta_disc = boot(lambda idx: auroc(alpha_disc[idx], correct[idx]) - auroc(-ent[idx], correct[idx]))
    delta_soft = boot(lambda idx: auroc(alpha_soft[idx], correct[idx]) - auroc(-ent[idx], correct[idx]))
    delta_soft_minus_disc = boot(lambda idx: auroc(alpha_soft[idx], correct[idx]) - auroc(alpha_disc[idx], correct[idx]))

    return {
        "name": name, "N": N, "C": C,
        "entropy_auroc": float(auroc_ent),
        "agreement_discrete_auroc": float(auroc_disc),
        "agreement_soft_auroc": float(auroc_soft),
        "alpha_disc_mean": float(np.mean(alpha_disc)),
        "alpha_disc_std": float(np.std(alpha_disc)),
        "alpha_soft_mean": float(np.mean(alpha_soft)),
        "alpha_soft_std": float(np.std(alpha_soft)),
        "delta_disc_minus_ent": delta_disc,
        "delta_soft_minus_ent": delta_soft,
        "delta_soft_minus_disc": delta_soft_minus_disc,
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap-n", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])
    ts = time.strftime("%Y%m%d_%H%M%S")

    results = []
    for ds in DATASETS:
        log.info(f"Processing {ds['name']}...")
        r = compute_metrics(ds["name"], ds["key"], ds["type"], log,
                            n_boot=args.bootstrap_n, seed=args.seed)
        if r:
            results.append(r)
            log.info(f"  ent={r['entropy_auroc']:.4f}  disc={r['agreement_discrete_auroc']:.4f}  "
                     f"soft={r['agreement_soft_auroc']:.4f}  "
                     f"Δ(disc-ent)={r['delta_disc_minus_ent']['mean']:+.4f}  "
                     f"Δ(soft-ent)={r['delta_soft_minus_ent']['mean']:+.4f}  "
                     f"Δ(soft-disc)={r['delta_soft_minus_disc']['mean']:+.4f}")

    log.info(f"\n{'='*80}")
    log.info("CONTINUOUS AGREEMENT ANALYSIS")
    log.info(f"{'='*80}")
    log.info(f"\n{'Dataset':22s} {'ent':>7s} {'disc':>7s} {'soft':>7s}  "
             f"{'Δ(disc-ent)':>14s}  {'Δ(soft-ent)':>14s}  {'Δ(soft-disc)':>14s}")
    for r in results:
        log.info(f"{r['name']:22s} {r['entropy_auroc']:>7.4f} {r['agreement_discrete_auroc']:>7.4f} "
                 f"{r['agreement_soft_auroc']:>7.4f}  "
                 f"{r['delta_disc_minus_ent']['mean']:>+10.4f} [{r['delta_disc_minus_ent']['ci_lo']:+.3f},{r['delta_disc_minus_ent']['ci_hi']:+.3f}]  "
                 f"{r['delta_soft_minus_ent']['mean']:>+10.4f} [{r['delta_soft_minus_ent']['ci_lo']:+.3f},{r['delta_soft_minus_ent']['ci_hi']:+.3f}]  "
                 f"{r['delta_soft_minus_disc']['mean']:>+10.4f} [{r['delta_soft_minus_disc']['ci_lo']:+.3f},{r['delta_soft_minus_disc']['ci_hi']:+.3f}]")

    report = {"timestamp": ts, "datasets": results}
    out_json = OUT_DIR / "anchoring_continuous_agreement_report.json"
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)
    log.info(f"\nReport: {out_json}")


if __name__ == "__main__":
    main()
