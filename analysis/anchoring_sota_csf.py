"""
analysis/anchoring_sota_csf.py — SOTA CSF comparison for Zone-2 targeting
=========================================================================
Adds 4 modern confidence-scoring functions beyond entropy/MSP/margin:
  - max_logit:    raw max cosine similarity (Hendrycks & Gimpel 2017)
  - energy:       -logsumexp(logits * T) (Liu et al. 2020)
  - ensemble_mi:  H(mean softmax) - mean(H(per-backbone softmax))
                  (Smith & Gal 2018; epistemic uncertainty)
  - pred_var:     mean across-backbone variance of softmax vectors

All compute-matched (same 5-backbone logits as existing signals).
Compares soft_agreement win rate against EACH of these CSFs (not just entropy).

Output: results/anchoring_sota_csf_report.{json,txt}
"""

import argparse, json, sys, time, logging
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase0_oracle_analysis import setup_logging, OUT_DIR, DATA_DIR  # noqa
from pilot_perimage_anchoring import (  # noqa
    BACKBONES, softmax_entropy, auroc, aupr, bootstrap_auroc_diff,
)

CIFAR_DIR = DATA_DIR / "cifar100"
META_DIR = DATA_DIR / "meta"
T = 100.0
EXISTING = {
    "teacher_behavior": DATA_DIR, "handrise_readwrite": DATA_DIR,
    "bow_turnhead": DATA_DIR, "voc2007": DATA_DIR, "stanford40": DATA_DIR,
    "coco": DATA_DIR, "multithumos": DATA_DIR,
}
CIFAR = {"cifar100": CIFAR_DIR}
CIFARC = {"cifar10c": CIFAR_DIR.parent / "cifarc",
          "cifar100c": CIFAR_DIR.parent / "cifarc"}
META_KEYS = ["food101", "dtd", "cifar10", "tiny_imagenet", "stl10",
             "eurosat", "imagenet_a", "imagenet_r",
             "sun397", "fgvc_aircraft", "stanford_cars", "gtsrb",
             # OOD expansion (reviewer round 3, pre-registered list)
             "imagenet_sketch", "imagenet_v2_matched",
             "imagenet_v2_threshold", "imagenet_v2_top", "imagenet_c",
             # non-ImageNet OOD (reviewer round 3 confound fix)
             "cifar10c", "cifar100c"]

NEW_SIGS = ["max_logit", "energy", "ensemble_mi", "pred_var"]


def load_stack(key):
    stack, gt = [], None
    for bb in BACKBONES:
        if key in EXISTING:
            lp = EXISTING[key] / f"clip_{bb}_{key}_logits.npy"
            bp = EXISTING[key] / f"clip_{bb}_{key}_labels.npy"
        elif key in CIFAR:
            lp = CIFAR[key] / f"clip_{bb}_{key}_logits.npy"
            bp = CIFAR[key] / f"clip_{bb}_{key}_labels.npy"
        elif key in CIFARC:
            lp = CIFARC[key] / f"clip_{bb}_{key}_logits.npy"
            bp = CIFARC[key] / f"clip_{bb}_{key}_labels.npy"
        else:
            lp = META_DIR / f"clip_{bb}_{key}_logits.npy"
            bp = META_DIR / f"clip_{bb}_{key}_labels.npy"
        if not lp.exists():
            return None
        stack.append(np.load(lp).astype(np.float32))
        labels = np.load(bp).astype(np.float32)
        if gt is None:
            gt = labels
    return np.stack(stack, axis=0), gt


def correctness(cons, gt):
    if gt.ndim == 1:
        return cons == gt.astype(int)
    n = gt.shape[0]
    return np.array([cons[i] in np.where(gt[i] > 0)[0]
                     for i in range(n)], dtype=bool)


def compute_all_signals(logits_stack, avg_logits):
    K, N, C = logits_stack.shape
    consensus = np.argmax(avg_logits, axis=1)
    backbone_argmaxes = np.argmax(logits_stack, axis=2)

    # existing baselines
    ent_avg, p_avg = softmax_entropy(avg_logits)
    s = {}
    s["entropy"] = -ent_avg
    s["msp"] = p_avg.max(axis=1)
    sorted_avg = np.sort(avg_logits, axis=1)
    s["margin"] = sorted_avg[:, -1] - sorted_avg[:, -2]

    # agreement signals
    s["hard_agree"] = (backbone_argmaxes == consensus[None, :]).mean(axis=0)
    probs = np.stack([np.exp(z * T) / np.exp(z * T).sum(axis=1, keepdims=True)
                      for z in logits_stack], axis=0)
    s["soft_agree"] = probs[:, np.arange(N), consensus].mean(axis=0)

    # NEW SOTA CSFs
    s["max_logit"] = avg_logits.max(axis=1)  # raw max cosine sim (Hendrycks 2017)
    s["energy"] = -np.log(np.exp(avg_logits * T).sum(axis=1)) / T  # Liu 2020
    # ensemble MI: H(mean) - mean(H)  (Smith & Gal 2018)
    mean_probs = probs.mean(axis=0)
    H_mean = -(mean_probs * np.log(mean_probs + 1e-12)).sum(axis=1)
    H_indiv = -(probs * np.log(probs + 1e-12)).sum(axis=2)
    s["ensemble_mi"] = H_mean - H_indiv.mean(axis=0)  # high = epistemic uncertainty
    # prediction variance: mean pairwise variance of softmax vectors
    s["pred_var"] = probs.var(axis=0).mean(axis=1)  # high = disagreement

    return s, consensus


def analyze_dataset(key, bootstrap_n, seed, log):
    loaded = load_stack(key)
    if loaded is None:
        return None
    logits_stack, gt = loaded
    K, N, C = logits_stack.shape
    avg = logits_stack.mean(axis=0)
    signals, consensus = compute_all_signals(logits_stack, avg)
    correct = correctness(consensus, gt)
    if correct.std() == 0:
        return None

    all_sigs = list(signals.keys())
    out = {"dataset": key, "N": int(N), "C": int(C), "acc": float(correct.mean())}

    # AUROC per signal
    for name in all_sigs:
        out[f"{name}_auroc"] = auroc(signals[name], correct)

    # soft_agreement vs each other signal (coupled bootstrap)
    soft = signals["soft_agree"]
    for name in all_sigs:
        if name == "soft_agree":
            continue
        d = bootstrap_auroc_diff(soft, signals[name], correct, bootstrap_n, seed)
        out[f"soft_vs_{name}_mean"] = d["mean"]
        out[f"soft_vs_{name}_lo"] = d["ci_lo"]
        out[f"soft_vs_{name}_hi"] = d["ci_hi"]

    log.info(f"  {key:18s} soft={out['soft_agree_auroc']:.3f} "
             + " ".join(f"{n}={out[f'{n}_auroc']:.3f}" for n in NEW_SIGS))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap_n", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ts = time.strftime("%Y%m%d_%H%M%S")
    setup_logging(ts, name="anchoring_sota_csf")
    log = logging.getLogger()

    keys = list(EXISTING.keys()) + list(CIFAR.keys()) + META_KEYS
    results = []
    for key in keys:
        r = analyze_dataset(key, args.bootstrap_n, args.seed, log)
        if r:
            results.append(r)

    # Win table: soft_agreement vs each CSF
    all_csfs = ["entropy", "msp", "margin", "hard_agree"] + NEW_SIGS
    win_table = {}
    for csf in all_csfs:
        w = t = l = 0
        for r in results:
            lo = r[f"soft_vs_{csf}_lo"]
            hi = r[f"soft_vs_{csf}_hi"]
            if lo > 0: w += 1
            elif hi < 0: l += 1
            else: t += 1
        win_table[csf] = {"win": w, "tie": t, "loss": l, "n": len(results)}

    lines = ["=" * 100, f"  SOTA CSF comparison: soft_agreement vs {len(all_csfs)} baselines ({len(results)} datasets)", "=" * 100, ""]
    lines.append(f"  {'CSF':>14} | soft wins | tie | soft loses | soft mean AUROC advantage")
    lines.append("  " + "-" * 70)
    for csf in all_csfs:
        wt = win_table[csf]
        deltas = [r[f"soft_vs_{csf}_mean"] for r in results]
        mean_d = float(np.mean(deltas))
        lines.append(f"  {csf:>14} | {wt['win']:>9} | {wt['tie']:>3} | {wt['loss']:>10} | {mean_d:+.4f}")
    lines.append("=" * 100)
    txt = "\n".join(lines)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "anchoring_sota_csf_report.json", "w") as f:
        json.dump({"args": vars(args), "timestamp": ts, "results": results,
                   "win_table": win_table}, f, indent=2)
    with open(OUT_DIR / "anchoring_sota_csf_report.txt", "w") as f:
        f.write(txt)
    print(); print(txt)


if __name__ == "__main__":
    main()
