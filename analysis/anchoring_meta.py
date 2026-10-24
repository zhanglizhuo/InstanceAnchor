"""
analysis/anchoring_meta.py — P4: 30+-dataset meta-study analysis
=================================================================
Elevation-plan P4 stage-3. Runs the Exp A protocol (hard agreement,
soft agreement, ensemble entropy) on all available datasets — the 8
paper datasets plus the ~22 new meta datasets extracted by
experiments/extract_clip_meta.py — then:

  1. win-count table (delta CI > 0 / < 0 / straddle) per signal
  2. LOO boundary validation at n ~= 30 (R1 sign rule, R2 Delta_alpha
     regression, R4 joint (p, C, kappa) refit — same protocol as
     anchoring_loocv_boundary.py, now with statistical power)
  3. meta-regression: Spearman of empirical delta vs p, kappa, C,
     consensus accuracy, multi-label fraction
  4. acc-bucket inverted-U shape check
  5. kappa-p regime map table (for fig regeneration)

Boundary params per dataset (mirrors anchoring_saturation_model.py):
  p = mean per-backbone Hit@1; alpha = hard vote fraction;
  E[alpha|wrong]; kappa = E[alpha|wrong]*(C-1)/(1-p);
  p* = kappa/(C-1+kappa); Delta_alpha = p - kappa(1-p)/(C-1).

Signals (compute-matched, 5 backbones):
  entropy (avg-logit softmax entropy, negative), agreement_top1 (hard),
  soft_agreement (mean softmax mass at consensus class, P3 winner).

Output: results/anchoring_meta_report.{json,txt}

Reproducibility (after extraction completes on V100 and npy files are
scp'd back into results/per_image/meta/):
  python analysis/anchoring_meta.py [--bootstrap_n 10000]
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase0_oracle_analysis import setup_logging, OUT_DIR, DATA_DIR  # noqa: E402
from pilot_perimage_anchoring import (  # noqa: E402
    BACKBONES, softmax_entropy, auroc, aupr, bootstrap_auroc_diff,
)

CIFAR_DIR = DATA_DIR / "cifar100"
META_DIR = DATA_DIR / "meta"
T = 100.0

# datasets: key -> (logits_dir, multilabel-known) ; single-label stored (N,)
EXISTING = {
    "teacher_behavior": DATA_DIR, "handrise_readwrite": DATA_DIR,
    "bow_turnhead": DATA_DIR, "voc2007": DATA_DIR, "stanford40": DATA_DIR,
    "coco": DATA_DIR, "multithumos": DATA_DIR,
}
CIFAR = {"cifar100": CIFAR_DIR}
CIFARC = {"cifar10c": CIFAR_DIR.parent / "cifarc",
          "cifar100c": CIFAR_DIR.parent / "cifarc"}
# NOTE: flowers102 (nelorth/oxford-flowers) is EXCLUDED: ClassLabel.names are
# lexicographic ID strings ("1","10",...) -> chance-level p=1/C, verified by
# feature inspection. tiny_imagenet (Maysee) is INCLUDED via wnid->name remap.
META_KEYS = ["food101", "dtd", "cifar10",
             "tiny_imagenet", "stl10", "eurosat",
             "imagenet_a", "imagenet_r",
             "sun397", "fgvc_aircraft", "stanford_cars", "gtsrb",
             # OOD expansion (reviewer round 3, pre-registered list)
             "imagenet_sketch", "imagenet_v2_matched",
             "imagenet_v2_threshold", "imagenet_v2_top", "imagenet_c",
             # non-ImageNet OOD (reviewer round 3 confound fix)
             "cifar10c", "cifar100c"]


def load_dataset(key, log):
    stack = []
    gt = None
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
        elif not np.array_equal(gt, labels):
            raise RuntimeError(f"label mismatch on {key}")
    return np.stack(stack, axis=0), gt


def correctness(cons, gt):
    if gt.ndim == 1:
        return cons == gt.astype(int)
    n = gt.shape[0]
    return np.array([cons[i] in np.where(gt[i] > 0)[0]
                     for i in range(n)], dtype=bool)


def soft_agreement(logits_stack, consensus):
    probs = np.stack([np.exp(z * T) / np.exp(z * T).sum(axis=1, keepdims=True)
                      for z in logits_stack], axis=0)
    return probs[:, np.arange(consensus.shape[0]), consensus].mean(axis=0)


def analyze_dataset(key, bootstrap_n, seed, log):
    loaded = load_dataset(key, log)
    if loaded is None:
        return None
    logits_stack, gt = loaded
    K, N, C = logits_stack.shape
    avg = logits_stack.mean(axis=0)
    consensus = np.argmax(avg, axis=1)
    correct = correctness(consensus, gt)
    if correct.std() == 0:
        log.info(f"  {key}: degenerate target, skip")
        return None

    # boundary params (mirror saturation_model)
    backbone_argmaxes = np.argmax(logits_stack, axis=2)
    per_bb_acc = []
    for b in range(K):
        if gt.ndim == 1:
            per_bb_acc.append((backbone_argmaxes[b] == gt.astype(int)).mean())
        else:
            per_bb_acc.append(np.mean([
                backbone_argmaxes[b][i] in np.where(gt[i] > 0)[0]
                for i in range(N)]))
    p = float(np.mean(per_bb_acc))
    hard = (backbone_argmaxes == consensus[None, :]).mean(axis=0)
    alpha_wrong = float(hard[~correct].mean()) if (~correct).sum() > 0 else float("nan")
    alpha_correct = float(hard[correct].mean()) if correct.sum() > 0 else float("nan")
    kappa = alpha_wrong * (C - 1) / (1 - p) if (1 - p) > 0 else float("inf")
    p_star = kappa / (C - 1 + kappa) if np.isfinite(kappa) else float("nan")
    delta_alpha = p - kappa * (1 - p) / (C - 1) if np.isfinite(kappa) else float("nan")

    # signals
    ent_avg, _ = softmax_entropy(avg)
    entropy = -ent_avg
    soft = soft_agreement(logits_stack, consensus)

    ml_frac = float(gt.sum(1).mean()) if gt.ndim == 2 else 1.0

    row = {
        "dataset": key, "N": int(N), "C": int(C),
        "multilabel": bool(gt.ndim == 2), "ml_frac": ml_frac,
        "p": p, "consensus_acc": float(correct.mean()),
        "E_alpha_wrong": alpha_wrong, "E_alpha_correct": alpha_correct,
        "kappa": kappa, "p_star": p_star, "Delta_alpha": delta_alpha,
        "gap_empirical": float(alpha_correct - alpha_wrong),
        "entropy_auroc": auroc(entropy, correct),
        "hard_auroc": auroc(hard, correct),
        "soft_auroc": auroc(soft, correct),
        "hard_aupr": aupr(hard, correct),
        "soft_aupr": aupr(soft, correct),
    }
    d_hard = bootstrap_auroc_diff(hard, entropy, correct, bootstrap_n, seed)
    d_soft = bootstrap_auroc_diff(soft, entropy, correct, bootstrap_n, seed)
    row.update({
        "delta_hard_mean": d_hard["mean"], "delta_hard_lo": d_hard["ci_lo"],
        "delta_hard_hi": d_hard["ci_hi"],
        "delta_soft_mean": d_soft["mean"], "delta_soft_lo": d_soft["ci_lo"],
        "delta_soft_hi": d_soft["ci_hi"],
    })
    log.info(f"  {key:18s} N={N:>5d} C={C:>3d} p={p:.3f} acc={correct.mean():.3f} "
             f"k={kappa:7.2f} d_hard={d_hard['mean']:+.4f} "
             f"d_soft={d_soft['mean']:+.4f}")
    return row


def win_class(delta_lo, delta_hi):
    if delta_lo > 0:
        return "win"
    if delta_hi < 0:
        return "loss"
    return "tie"


def loocv(rows, feature_cols, target="delta_hard_mean"):
    """Same protocol as anchoring_loocv_boundary.py.loocv."""
    n = len(rows)
    X = np.array([[r[c] for c in feature_cols] for r in rows], dtype=float)
    y = np.array([r[target] for r in rows], dtype=float)
    preds = np.empty(n)
    per_fold = []
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        A = np.column_stack([X[mask], np.ones(mask.sum())])
        beta, *_ = np.linalg.lstsq(A, y[mask], rcond=None)
        pred = float(np.dot(np.r_[X[i], 1.0], beta))
        preds[i] = pred
        per_fold.append({"held_out": rows[i]["dataset"],
                         "actual": float(y[i]), "predicted": pred,
                         "sign_agree": bool(np.sign(pred) == np.sign(y[i]))})
    sign_acc = float(np.mean([f["sign_agree"] for f in per_fold]))
    r = float(np.corrcoef(preds, y)[0, 1]) if np.std(preds) > 0 else 0.0
    mae = float(np.mean(np.abs(preds - y)))
    return {"n_folds": n, "sign_acc": sign_acc,
            "pred_actual_corr": r, "mae": mae, "per_fold": per_fold}


def bootstrap_sign_acc(preds, actuals, n_boot, seed):
    rng = np.random.RandomState(seed)
    preds = np.asarray(preds, dtype=float)
    actuals = np.asarray(actuals, dtype=float)
    n = len(actuals)
    accs = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.randint(0, n, size=n)
        accs[b] = np.mean(np.sign(preds[idx]) == np.sign(actuals[idx]))
    return {"mean": float(accs.mean()),
            "ci_lo": float(np.percentile(accs, 2.5)),
            "ci_hi": float(np.percentile(accs, 97.5))}


def spearman(rows, feat, target="delta_hard_mean"):
    from scipy.stats import spearmanr
    xs = [r[feat] for r in rows if np.isfinite(r[feat]) and np.isfinite(r[target])]
    ys = [r[target] for r in rows if np.isfinite(r[feat]) and np.isfinite(r[target])]
    if len(xs) < 5:
        return None
    r, p = spearmanr(xs, ys)
    return {"rho": float(r), "p": float(p), "n": len(xs)}


def format_report(rows, loo_hard, loo_soft, win_table, spears, buckets):
    lines = []
    lines.append("=" * 96)
    lines.append(f"  P4 meta-study: {len(rows)} datasets | hard vs soft agreement vs entropy")
    lines.append("=" * 96)
    lines.append("")
    lines.append(f"    {'dataset':>18} | {'N':>5} {'C':>3} | {'p':>6} | {'acc':>6} | "
                 f"{'k':>7} | {'d_hard':>8} | {'d_soft':>8} | {'h_cls':>6} {'s_cls':>6}")
    lines.append("    " + "-" * 84)
    for r in sorted(rows, key=lambda r: r["delta_hard_mean"]):
        hc = win_class(r["delta_hard_lo"], r["delta_hard_hi"])
        sc = win_class(r["delta_soft_lo"], r["delta_soft_hi"])
        k = f"{r['kappa']:7.1f}" if np.isfinite(r["kappa"]) else "    inf"
        lines.append(f"    {r['dataset']:>18} | {r['N']:>5} {r['C']:>3} | "
                     f"{r['p']:>6.3f} | {r['consensus_acc']*100:>5.1f}% | {k} | "
                     f"{r['delta_hard_mean']:>+8.4f} | {r['delta_soft_mean']:>+8.4f} | "
                     f"{hc:>6} {sc:>6}")
    lines.append("")
    lines.append("  Win table (delta vs entropy, CI excludes 0):")
    for sig, wt in win_table.items():
        lines.append(f"    {sig:>16} : win {wt['win']:>2} | tie {wt['tie']:>2} | "
                     f"loss {wt['loss']:>2}  (of {wt['n']})")
    lines.append("")
    for tag, loo in (("hard", loo_hard), ("soft", loo_soft)):
        if loo is None:
            continue
        lines.append(f"  LOO ({tag}, {loo['n_folds']} folds): sign_acc={loo['sign_acc']*100:.1f}% "
                     f"(bootstrap CI [{loo['sign_ci']['ci_lo']*100:.1f},"
                     f"{loo['sign_ci']['ci_hi']*100:.1f}]) "
                     f"corr={loo['pred_actual_corr']:+.3f} mae={loo['mae']:.4f}")
    lines.append("")
    lines.append("  Meta-regression (Spearman vs empirical delta):")
    for feat, v in spears.items():
        if v:
            lines.append(f"    {feat:>16} : rho={v['rho']:+.3f} p={v['p']:.4f} n={v['n']}")
    lines.append("")
    lines.append("  Acc-bucket inverted-U check (hard delta):")
    for b in buckets:
        lines.append(f"    acc [{b['lo']*100:.0f}-{b['hi']*100:.0f})% n={b['n']:>2} : "
                     f"mean d_hard={b['mean_delta']:+.4f}")
    lines.append("=" * 96)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap_n", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ts = time.strftime("%Y%m%d_%H%M%S")
    setup_logging(ts, name="anchoring_meta")
    log = logging.getLogger()

    keys = list(EXISTING.keys()) + list(CIFAR.keys()) + META_KEYS
    rows = []
    for key in keys:
        r = analyze_dataset(key, args.bootstrap_n, args.seed, log)
        if r is None:
            log.warning(f"  {key}: data missing, skipped")
            continue
        rows.append(r)
    if len(rows) < 10:
        log.error(f"only {len(rows)} datasets available; meta analysis needs "
                  "the V100 extraction. Aborting report generation.")
        return 1

    # LOO
    def _loo_with_ci(features, target):
        usable = [r for r in rows
                  if all(np.isfinite(r[c]) for c in features)
                  and np.isfinite(r[target])]
        if len(usable) < 5:
            return None
        loo = loocv(usable, features, target)
        preds = [f["predicted"] for f in loo["per_fold"]]
        acts = [f["actual"] for f in loo["per_fold"]]
        loo["sign_ci"] = bootstrap_sign_acc(preds, acts, args.bootstrap_n, args.seed)
        return loo

    loo_hard_da = _loo_with_ci(["Delta_alpha"], "delta_hard_mean")
    loo_hard_joint = _loo_with_ci(["p", "C", "kappa"], "delta_hard_mean")
    loo_soft_joint = _loo_with_ci(["p", "C", "kappa"], "delta_soft_mean")

    win_table = {"hard_agreement": {}, "soft_agreement": {}}
    for sig, prefix in (("hard_agreement", "delta_hard"), ("soft_agreement", "delta_soft")):
        w = t = l = 0
        for r in rows:
            cls = win_class(r[f"{prefix}_lo"], r[f"{prefix}_hi"])
            w += cls == "win"; t += cls == "tie"; l += cls == "loss"
        win_table[sig] = {"win": w, "tie": t, "loss": l, "n": len(rows)}

    spears = {feat: spearman(rows, feat) for feat in
              ("p", "kappa", "C", "consensus_acc", "ml_frac")}

    order = np.argsort([r["consensus_acc"] for r in rows])
    buckets = []
    n_buck = 4
    for b in range(n_buck):
        idx = order[b * len(rows) // n_buck:(b + 1) * len(rows) // n_buck]
        ds = [rows[i]["delta_hard_mean"] for i in idx]
        buckets.append({"lo": float(min(rows[i]["consensus_acc"] for i in idx)),
                        "hi": float(max(rows[i]["consensus_acc"] for i in idx)),
                        "n": len(idx), "mean_delta": float(np.mean(ds))})

    txt = format_report(rows, None, None, win_table, spears, buckets)
    out = {"args": vars(args), "timestamp": ts, "results": rows,
           "win_table": win_table, "spearman": spears, "buckets": buckets,
           "loo": {"hard_Delta_alpha": loo_hard_da,
                   "hard_joint_pCk": loo_hard_joint,
                   "soft_joint_pCk": loo_soft_joint}}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "anchoring_meta_report.json", "w") as f:
        json.dump(out, f, indent=2, default=lambda o: float(o))
    with open(OUT_DIR / "anchoring_meta_report.txt", "w") as f:
        f.write(txt)
    print(); print(txt)
    return 0


if __name__ == "__main__":
    exit(main())
