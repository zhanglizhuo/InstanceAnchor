"""
analysis/anchoring_deep_ensemble.py — generic kappa decomposition for any
5-model ensemble stack
=========================================================================
One protocol for the three deep experiments (E1 cell D, E2 non-ImageNet
cross-family, E3 seed-diverse trained ensemble):

  consensus = argmax(mean logits); hard agreement = vote fraction;
  kappa_emp = E[alpha|wrong]*(C-1)/(1-p); kappa_iid via simulate_iid_kappa
  (same function/seed as the CLIP decomposition); plurality_share;
  signals at temperature T (raw logits -> T=1 for torchvision models;
  cosine logits -> T=100); coupled bootstrap B=10000 seed 42:
  d_hard-ent, d_soft-ent, d_soft-msp.

Usage:
  python analysis/anchoring_deep_ensemble.py --name cellD_vitb16_v2 \
      --stack_dir results/per_image/vitb16_panel --models tv_vitb16,timm_vitb16,timm_vitb16_augreg,timm_vitb16_augreg2,timm_vitb16_swsl \
      --dataset imagenet_v2_matched --T 1 --clip_ref imagenet_v2_matched
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
from pilot_perimage_anchoring import auroc, bootstrap_auroc_diff  # noqa: E402


def softmax(z, t):
    z = z * t
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--stack_dir", required=True)
    parser.add_argument("--models", required=True, help="comma-separated prefixes")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--T", type=float, default=1.0)
    parser.add_argument("--clip_ref", default=None,
                        help="meta-report dataset key for the CLIP reference row")
    parser.add_argument("--bootstrap_n", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ts = time.strftime("%Y%m%d_%H%M%S")
    setup_logging(ts, name="anchoring_deep_ensemble")
    log = logging.getLogger()
    models = [m.strip() for m in args.models.split(",")]
    sdir = OUT_DIR / "per_image" / args.stack_dir

    stack, gt = [], None
    for m in models:
        lp = sdir / f"{m}_{args.dataset}_logits.npy"
        bp = sdir / f"{m}_{args.dataset}_labels.npy"
        if not lp.exists():
            log.error(f"missing {lp}")
            return 1
        stack.append(np.load(lp).astype(np.float32))
        labels = np.load(bp).astype(np.int64)
        if gt is None:
            gt = labels
        elif not np.array_equal(gt, labels):
            raise RuntimeError(f"label mismatch {m}")
    logits_stack = np.stack(stack, axis=0)
    K, N, C = logits_stack.shape
    avg = logits_stack.mean(axis=0)
    consensus = np.argmax(avg, axis=1)
    correct = consensus == gt
    per_bb = [(np.argmax(logits_stack[b], 1) == gt).mean() for b in range(K)]
    p = float(np.mean(per_bb))
    hard = (np.argmax(logits_stack, axis=2) == consensus[None, :]).mean(axis=0)
    a_c = float(hard[correct].mean()) if correct.sum() else float("nan")
    a_w = float(hard[~correct].mean()) if (~correct).sum() else float("nan")
    kemp = a_w * (C - 1) / (1 - p) if (1 - p) > 0 else float("inf")
    sim = simulate_iid_kappa(per_bb, K, C)
    share = sim["kappa_iid"] / kemp if kemp > 0 else float("nan")
    p_avg = softmax(avg, args.T)
    ent = (p_avg * np.log(p_avg + 1e-12)).sum(axis=1)
    msp = p_avg.max(axis=1)
    probs = np.stack([softmax(z, args.T) for z in logits_stack], axis=0)
    soft = probs[:, np.arange(N), consensus].mean(axis=0)
    dh = bootstrap_auroc_diff(hard, ent, correct, args.bootstrap_n, args.seed)
    ds = bootstrap_auroc_diff(soft, ent, correct, args.bootstrap_n, args.seed)
    dm = bootstrap_auroc_diff(soft, msp, correct, args.bootstrap_n, args.seed)

    row = {
        "name": args.name, "N": int(N), "C": int(C), "K": K,
        "per_backbone_accs": [float(x) for x in per_bb], "p": p,
        "consensus_acc": float(correct.mean()),
        "E_alpha_correct": a_c, "E_alpha_wrong": a_w,
        "kappa_empirical": float(kemp), "kappa_iid_mc": float(sim["kappa_iid"]),
        "plurality_share": float(share),
        "hard_auroc": auroc(hard, correct), "entropy_auroc": auroc(ent, correct),
        "msp_auroc": auroc(msp, correct), "soft_auroc": auroc(soft, correct),
        "delta_hard_vs_entropy": dh,
        "delta_soft_vs_entropy": ds,
        "delta_soft_vs_msp": dm,
        "T": args.T,
    }
    if args.clip_ref:
        meta = json.load(open(OUT_DIR / "anchoring_meta_report.json"))
        ref = next((r for r in meta["results"] if r["dataset"] == args.clip_ref), None)
        if ref:
            row["clip_reference"] = {
                "p": ref["p"], "kappa": ref["kappa"],
                "delta_hard": ref["delta_hard_mean"],
                "delta_soft": ref["delta_soft_mean"],
            }

    out = {"args": vars(args), "timestamp": ts, "results": [row]}
    with open(OUT_DIR / f"anchoring_{args.name}_report.json", "w") as f:
        json.dump(out, f, indent=2, default=lambda o: float(o))
    lines = ["=" * 90, f"  deep ensemble: {args.name} (T={args.T})", "=" * 90]
    lines.append(f"  N={row['N']} C={row['C']} K={row['K']} p={row['p']:.3f} "
                 f"consensus={row['consensus_acc']:.3f}")
    lines.append(f"  per-backbone: " + ", ".join(f"{a:.3f}" for a in row["per_backbone_accs"]))
    lines.append(f"  E[a|c]={row['E_alpha_correct']:.3f} E[a|w]={row['E_alpha_wrong']:.3f} "
                 f"kappa_emp={row['kappa_empirical']:.1f} kappa_iid={row['kappa_iid_mc']:.1f} "
                 f"share={row['plurality_share']*100:.0f}%")
    lines.append(f"  d_hard-ent={dh['mean']:+.4f} CI[{dh['ci_lo']:+.4f},{dh['ci_hi']:+.4f}]  "
                 f"d_soft-ent={ds['mean']:+.4f} CI[{ds['ci_lo']:+.4f},{ds['ci_hi']:+.4f}]  "
                 f"d_soft-msp={dm['mean']:+.4f} CI[{dm['ci_lo']:+.4f},{dm['ci_hi']:+.4f}]")
    if "clip_reference" in row:
        r = row["clip_reference"]
        lines.append(f"  CLIP ref: p={r['p']:.3f} kappa={r['kappa']:.1f} "
                     f"d_hard={r['delta_hard']:+.4f} d_soft={r['delta_soft']:+.4f}")
    lines.append("=" * 90)
    txt = "\n".join(lines)
    with open(OUT_DIR / f"anchoring_{args.name}_report.txt", "w") as f:
        f.write(txt)
    print(txt)
    return 0


if __name__ == "__main__":
    exit(main())
