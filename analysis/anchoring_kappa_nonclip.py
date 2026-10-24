"""
analysis/anchoring_kappa_nonclip.py — Cross-family kappa replication
=====================================================================
Tests whether the plurality ceiling generalizes beyond the CLIP family:
computes the kappa decomposition (empirical kappa vs i.i.d. plurality-MC
baseline, identical protocol to anchoring_kappa_iid_baseline.py) and the
agreement-vs-entropy selective-prediction comparison for an ensemble of
five architecturally diverse ImageNet-supervised (non-CLIP) classifiers
(see experiments/extract_nonclip_per_image.py) on ImageNet-V2 variants,
and contrasts each row with the CLIP-family result on the SAME dataset
(from results/anchoring_meta_report.json +
results/anchoring_kappa_meta_decomp_report.json).

Protocol notes:
  - non-CLIP logits are raw model outputs -> softmax signals use T=1
    (CLIP rows use T=100 on cosine-sim logits; hard-vote / kappa metrics
    are temperature-free in both cases).
  - consensus = argmax of averaged logits (same rule as the CLIP pipeline).

Input : results/per_image/nonclip/nonclip_{model}_{dataset}_{logits,labels}.npy
Output: results/anchoring_kappa_nonclip_report.{json,txt}

Usage: python analysis/anchoring_kappa_nonclip.py [--bootstrap_n 10000]
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase0_oracle_analysis import setup_logging, OUT_DIR  # noqa: E402
from anchoring_kappa_iid_baseline import simulate_iid_kappa  # noqa: E402
from pilot_perimage_anchoring import (  # noqa: E402
    auroc, bootstrap_auroc_diff,
)

MODELS = ["resnet50", "vit_b_16", "convnext_base", "swin_b", "efficientnet_b4"]
DATA_DIR = OUT_DIR / "per_image" / "nonclip"
T = 1.0


def load_nonclip_stack(dataset):
    stack, gt = [], None
    for m in MODELS:
        lp = DATA_DIR / f"nonclip_{m}_{dataset}_logits.npy"
        bp = DATA_DIR / f"nonclip_{m}_{dataset}_labels.npy"
        if not lp.exists():
            return None
        stack.append(np.load(lp).astype(np.float32))
        labels = np.load(bp).astype(np.int64)
        if gt is None:
            gt = labels
        elif not np.array_equal(gt, labels):
            raise RuntimeError(f"label mismatch for {m} on {dataset}")
    return np.stack(stack, axis=0), gt


def softmax(z):
    z = z * T
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def analyze(dataset, bootstrap_n, seed, log):
    loaded = load_nonclip_stack(dataset)
    if loaded is None:
        log.warning(f"  {dataset}: non-CLIP npy missing, skipped")
        return None
    logits_stack, gt = loaded
    K, N, C = logits_stack.shape
    avg = logits_stack.mean(axis=0)
    consensus = np.argmax(avg, axis=1)
    correct = consensus == gt

    per_bb_acc = [(np.argmax(logits_stack[b], 1) == gt).mean() for b in range(K)]
    p = float(np.mean(per_bb_acc))

    backbone_argmaxes = np.argmax(logits_stack, axis=2)
    hard = (backbone_argmaxes == consensus[None, :]).mean(axis=0)
    alpha_correct = float(hard[correct].mean()) if correct.sum() else float("nan")
    alpha_wrong = float(hard[~correct].mean()) if (~correct).sum() else float("nan")
    kappa_emp = alpha_wrong * (C - 1) / (1 - p) if (1 - p) > 0 else float("inf")
    p_star = kappa_emp / (C - 1 + kappa_emp) if np.isfinite(kappa_emp) else float("nan")
    delta_alpha = p - kappa_emp * (1 - p) / (C - 1) if np.isfinite(kappa_emp) else float("nan")

    sim = simulate_iid_kappa(per_bb_acc, K, C)
    kappa_iid = sim["kappa_iid"]
    plurality_share = kappa_iid / kappa_emp if kappa_emp > 0 else float("nan")

    # signals at T=1 (raw logits)
    p_avg = softmax(avg)
    ent = (p_avg * np.log(p_avg + 1e-12)).sum(axis=1)  # negative entropy
    msp = p_avg.max(axis=1)
    probs = np.stack([softmax(z) for z in logits_stack], axis=0)
    soft = probs[:, np.arange(N), consensus].mean(axis=0)

    d_hard_ent = bootstrap_auroc_diff(hard, ent, correct, bootstrap_n, seed)
    d_soft_ent = bootstrap_auroc_diff(soft, ent, correct, bootstrap_n, seed)
    d_soft_msp = bootstrap_auroc_diff(soft, msp, correct, bootstrap_n, seed)

    row = {
        "dataset": dataset, "N": int(N), "C": int(C),
        "ensemble": "imagenet-supervised (non-CLIP)", "K": K,
        "per_backbone_accs": [float(a) for a in per_bb_acc], "p": p,
        "consensus_acc": float(correct.mean()),
        "E_alpha_correct": alpha_correct, "E_alpha_wrong": alpha_wrong,
        "kappa_empirical": float(kappa_emp), "kappa_iid_mc": float(kappa_iid),
        "plurality_share": float(plurality_share),
        "p_star": float(p_star), "Delta_alpha": float(delta_alpha),
        "entropy_auroc": auroc(ent, correct), "msp_auroc": auroc(msp, correct),
        "hard_auroc": auroc(hard, correct), "soft_auroc": auroc(soft, correct),
        "delta_hard_vs_entropy": d_hard_ent,
        "delta_soft_vs_entropy": d_soft_ent,
        "delta_soft_vs_msp": d_soft_msp,
    }

    # CLIP comparison on the same dataset (meta always; decomp when available)
    row["clip_reference"] = None
    try:
        meta = json.load(open(OUT_DIR / "anchoring_meta_report.json"))
        clip = next(r for r in meta["results"] if r["dataset"] == dataset)
        ref = {"p": clip["p"], "kappa_empirical": clip["kappa"],
               "delta_hard_vs_entropy": clip["delta_hard_mean"],
               "delta_soft_vs_entropy": clip["delta_soft_mean"]}
        try:
            decomp = json.load(open(OUT_DIR / "anchoring_kappa_meta_decomp_report.json"))
            cd = next(r for r in decomp["datasets"] if r["name"] == dataset)
            ref["plurality_share"] = cd["plurality_share"]
        except (FileNotFoundError, StopIteration):
            ref["plurality_share"] = None  # not in the 20-dataset decomp run
        row["clip_reference"] = ref
    except (FileNotFoundError, StopIteration):
        pass

    log.info(f"  {dataset}: p={p:.3f} k_emp={kappa_emp:.1f} "
             f"k_iid={kappa_iid:.1f} share={plurality_share*100:.0f}% "
             f"d_hard={d_hard_ent['mean']:+.4f} d_soft={d_soft_ent['mean']:+.4f}")
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+",
                        default=["imagenet_v2_matched", "imagenet_v2_top"])
    parser.add_argument("--bootstrap_n", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ts = time.strftime("%Y%m%d_%H%M%S")
    setup_logging(ts, name="anchoring_kappa_nonclip")
    log = logging.getLogger()

    rows = [r for r in (analyze(d, args.bootstrap_n, args.seed, log)
                        for d in args.datasets) if r]

    out = {"args": vars(args), "timestamp": ts,
           "ensemble": MODELS, "temperature": T, "results": rows}

    lines = ["=" * 96,
             "  Cross-family kappa replication (ImageNet-supervised non-CLIP ensemble)",
             "=" * 96, ""]
    for r in rows:
        clip = r.get("clip_reference") or {}
        lines.append(f"  {r['dataset']}  N={r['N']} C={r['C']} K={r['K']}")
        lines.append(f"    per-backbone acc: "
                     + ", ".join(f"{a:.3f}" for a in r["per_backbone_accs"]))
        lines.append(f"    p={r['p']:.3f}  consensus={r['consensus_acc']:.3f}  "
                     f"E[a|correct]={r['E_alpha_correct']:.3f}  "
                     f"E[a|wrong]={r['E_alpha_wrong']:.3f}")
        lines.append(f"    kappa_emp={r['kappa_empirical']:.1f}  "
                     f"kappa_iid_mc={r['kappa_iid_mc']:.1f}  "
                     f"plurality_share={r['plurality_share']*100:.0f}%  "
                     f"p*={r['p_star']:.3f}  Delta_alpha={r['Delta_alpha']:+.3f}")
        lines.append(f"    AUROC ent={r['entropy_auroc']:.3f} msp={r['msp_auroc']:.3f} "
                     f"hard={r['hard_auroc']:.3f} soft={r['soft_auroc']:.3f}")
        de = r["delta_hard_vs_entropy"]
        ds = r["delta_soft_vs_entropy"]
        dm = r["delta_soft_vs_msp"]
        lines.append(f"    d_hard-ent={de['mean']:+.4f} CI[{de['ci_lo']:+.4f},"
                     f"{de['ci_hi']:+.4f}]  d_soft-ent={ds['mean']:+.4f} "
                     f"CI[{ds['ci_lo']:+.4f},{ds['ci_hi']:+.4f}]  "
                     f"d_soft-msp={dm['mean']:+.4f} CI[{dm['ci_lo']:+.4f},"
                     f"{dm['ci_hi']:+.4f}]")
        if clip:
            share = clip.get("plurality_share")
            share_s = f"{share*100:.0f}%" if share is not None else "n/a"
            lines.append(f"    CLIP ref: p={clip['p']:.3f} "
                         f"kappa={clip['kappa_empirical']:.1f} "
                         f"share={share_s} "
                         f"d_hard={clip['delta_hard_vs_entropy']:+.4f} "
                         f"d_soft={clip['delta_soft_vs_entropy']:+.4f}")
        lines.append("")
    lines.append("=" * 96)
    txt = "\n".join(lines)

    with open(OUT_DIR / "anchoring_kappa_nonclip_report.json", "w") as f:
        json.dump(out, f, indent=2, default=lambda o: float(o))
    with open(OUT_DIR / "anchoring_kappa_nonclip_report.txt", "w") as f:
        f.write(txt)
    print(); print(txt)
    return 0


if __name__ == "__main__":
    exit(main())
