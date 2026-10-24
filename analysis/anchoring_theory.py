"""
anchoring_theory.py — Phase 2 baselines + Phase 4 variance decomposition
=======================================================================
Computes:
  1) Single-backbone selective prediction baselines (max-logit, softmax, entropy, margin)
  2) Ensemble variants (agreement, ensemble entropy)
  3) Logit variance decomposition — demonstrates why agreement > entropy
  4) Bootstrap CIs on all AUROC deltas

Usage:
  python analysis/anchoring_theory.py

Output:
  results/anchoring_theory_report.json
  results/anchoring_theory_report.txt
"""

import json, os, sys
from pathlib import Path
import numpy as np
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

OUT = Path("results")
OUT.mkdir(parents=True, exist_ok=True)

SUBSETS = ["teacher_behavior", "handrise_readwrite", "bow_turnhead"]
BACKBONES = ["openai", "laion", "siglip2", "eva02", "dfn"]
PER_IMAGE = Path("results/per_image")
N_BOOT = 10000

CLASS_NAMES = {
    "teacher_behavior": [
        "guide", "answer", "on-stage interaction", "blackboard-writing",
        "teacher", "stand", "screen", "blackboard",
    ],
    "handrise_readwrite": [
        "handrise", "read", "write",
    ],
    "bow_turnhead": [
        "bow", "turnhead",
    ],
}


def load_logits(subset):
    """Return (logits: dict model→(N,C) array, labels: (N,C) array).
    Labels are same for all models (just copied per model)."""
    logits = {}
    labels = None
    for mk in BACKBONES:
        p = PER_IMAGE / f"clip_{mk}_{subset}_logits.npy"
        if not p.exists():
            print(f"  [WARN] {p} not found, skipping")
            continue
        logits[mk] = np.load(p)
        lb = np.load(PER_IMAGE / f"clip_{mk}_{subset}_labels.npy")
        if labels is None:
            labels = lb
        elif not np.allclose(labels, lb):
            print(f"  [WARN] label mismatch for {mk}")
    return logits, labels


# ── metric functions ─────────────────────────────────────────────────
TEMPERATURE = 100.0


def softmax(x):
    z = x * TEMPERATURE
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def entropy(x):
    p = softmax(x)
    return -(p * np.log(np.clip(p, 1e-12, 1.0))).sum(axis=1)


def max_logit(x):
    return x.max(axis=1)


def softmax_response(x):
    return softmax(x).max(axis=1)


def margin(x):
    x_sorted = np.sort(x, axis=1)
    return x_sorted[:, -1] - x_sorted[:, -2]


def per_image_agreement(logits_dict, consensus_argmax):
    backbone_argmaxes = np.stack([v.argmax(1) for v in logits_dict.values()], axis=0)
    return (backbone_argmaxes == consensus_argmax[None, :]).mean(axis=0)


def ensemble_entropy(logits_dict):
    """Entropy of averaged logits."""
    avg = np.mean(list(logits_dict.values()), axis=0)
    return entropy(avg)


# ── selective prediction evaluation ──────────────────────────────────
def compute_auroc(signal, is_error):
    """AUROC — signal higher = more confident = lower error risk.
    If signal is confidence (higher = better), use as is.
    Entropy/confusion: higher entropy = less confident = more likely error.
    We need to check: is_error=1 means error. For confidence signals,
    we rank by signal (higher confidence → lower error)."""
    order = np.argsort(-signal)  # high signal first
    n_pos = int(is_error.sum())
    n_neg = len(is_error) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    ranks = rankdata(-signal, method="average")
    rank_sum = (ranks[is_error == 1]).sum()
    auroc = (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return auroc


def compute_selective_acc(signal, is_error, coverage):
    """Accuracy when keeping top coverage fraction by signal."""
    n_keep = max(1, int(round(len(signal) * coverage)))
    idx = np.argsort(-signal)[:n_keep]
    acc = 1.0 - is_error[idx].mean()
    return acc


def evaluate_signal(name, signal, labels, consensus_pred):
    gt = labels.astype(bool)
    is_error = (~gt[np.arange(len(labels)), consensus_pred]).astype(float)
    auroc = compute_auroc(signal, is_error)
    acc_at_50 = compute_selective_acc(signal, is_error, 0.5)
    acc_at_70 = compute_selective_acc(signal, is_error, 0.7)
    return {"auroc": auroc, "selacc@50": acc_at_50, "selacc@70": acc_at_70}


def bootstrap_ci(signal_a, signal_b, labels, consensus_pred, n_boot=10000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(labels)
    gt = labels.astype(bool)
    is_error = (~gt[np.arange(len(labels)), consensus_pred]).astype(float)
    deltas = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        a = compute_auroc(signal_a[idx], is_error[idx])
        b = compute_auroc(signal_b[idx], is_error[idx])
        deltas.append(b - a)
    deltas = np.sort(deltas)
    return {
        "mean": float(np.mean(deltas)),
        "ci_lo": float(deltas[int(0.025 * n_boot)]),
        "ci_hi": float(deltas[int(0.975 * n_boot)]),
    }


# ── variance decomposition (Phase 4) ─────────────────────────────────
def variance_decomposition(logits_dict, labels, consensus_argmax):
    """Per-image variance decomposition of multi-backbone logits.
    
    For each image: logits are (5, C) — 5 backbone views.
    Total variance = trace of covariance across all dimensions.
    Decompose into: within-class variance (models disagree about which class)
                    between-class variance (classes are well-separated).
    
    Key insight:
      entropy ≈ f(total_variance)
      agreement ≈ 1 - (within_class_var / total_var)
    """
    stacked = np.stack(list(logits_dict.values()), axis=1)  # (N, 5, C)
    N, M, C = stacked.shape
    
    # Mean across backbones
    mean_logits = stacked.mean(axis=1)  # (N, C)
    
    # Total variance per image: mean squared deviation from overall mean
    # = tr(cov) across all backbone-class dimensions
    centered = stacked - stacked.mean(axis=(1, 2), keepdims=True)
    total_var_per_img = (centered ** 2).sum(axis=(1, 2)) / (M * C - 1)
    
    # Between-backbone variance: how much do models disagree?
    # For each class, variance of the backbone estimates
    backbone_var_per_class = stacked.var(axis=1, ddof=1)  # (N, C)
    within_class_var = backbone_var_per_class.sum(axis=1)  # (N,) — sum across classes
    
    # Between class variance: how well-separated are class means?
    class_means = mean_logits  # (N, C)
    class_centered = class_means - class_means.mean(axis=1, keepdims=True)
    between_class_var = (class_centered ** 2).sum(axis=1) / (C - 1)
    
    # Entropy and agreement
    ent = ensemble_entropy(logits_dict)
    agr = per_image_agreement(logits_dict, consensus_argmax)
    
    # Agree fraction as normalized disagreement
    # agreement = 1 - normalized model disagreement
    max_var_per_img = backbone_var_per_class.max(axis=1)
    
    return {
        "total_var": total_var_per_img,
        "within_class_var": within_class_var,
        "between_class_var": between_class_var,
        "entropy": ent,
        "agreement": agr,
        "backbone_var_per_class": backbone_var_per_class,
    }


def corr(x, y):
    mask = ~(np.isnan(x) | np.isnan(y))
    return np.corrcoef(x[mask], y[mask])[0, 1]


# ── main ─────────────────────────────────────────────────────────────
def main():
    results = {}
    for subset in SUBSETS:
        print(f"\n{'='*60}")
        print(f"  {subset}")
        print(f"{'='*60}")
        
        logits_dict, labels = load_logits(subset)
        if len(logits_dict) == 0:
            continue
        
        n_classes = labels.shape[1]
        cn = CLASS_NAMES[subset]
        consensus_pred = np.mean(list(logits_dict.values()), axis=0).argmax(1)
        gt_multilabel = labels.astype(bool)
        is_correct = gt_multilabel[np.arange(len(labels)), consensus_pred]
        is_error = (~is_correct).astype(float)
        consensus_acc = is_correct.mean()
        
        err_rate = is_error.mean()
        
        # ── Phase 2: single-backbone baselines ──
        single_results = {}
        best_single = {"auroc": 0, "name": "", "signal": None}
        
        for mk, l in logits_dict.items():
            sigs = {
                "max_logit": max_logit(l),
                "softmax_response": softmax_response(l),
                "entropy": -entropy(l),
                "margin": margin(l),
            }
            evals = {}
            for sname, sig in sigs.items():
                evals[sname] = evaluate_signal(sname, sig, labels, consensus_pred)
            single_results[mk] = evals
            if evals["entropy"]["auroc"] > best_single["auroc"]:
                best_single = {"auroc": evals["entropy"]["auroc"], "name": mk}
        
        # ── Ensemble signals ──
        ensemble_ent = -ensemble_entropy(logits_dict)
        ensemble_agr = per_image_agreement(logits_dict, consensus_pred)
        
        ensemble_eval = {
            "agreement_top1": evaluate_signal("agreement", ensemble_agr, labels, consensus_pred),
            "ensemble_entropy": evaluate_signal("entropy", ensemble_ent, labels, consensus_pred),
        }
        
        # ── Bootstrap: agreement vs ensemble entropy ──
        ci = bootstrap_ci(ensemble_ent, ensemble_agr, labels, consensus_pred, n_boot=N_BOOT)
        
        # ── Phase 4: variance decomposition ──
        vd = variance_decomposition(logits_dict, labels, consensus_pred)
        
        # Correlations: entropy vs total_var, agreement vs var_ratio
        var_ratio = vd["within_class_var"] / (vd["total_var"] + 1e-12)
        
        corr_ent_total = corr(vd["entropy"], vd["total_var"])
        corr_agr_ratio = corr(vd["agreement"], 1 - var_ratio)
        corr_agr_var = corr(vd["agreement"], vd["within_class_var"])
        
        # Stratify by variance ratio
        low_var_mask = var_ratio < np.median(var_ratio)
        high_var_mask = var_ratio >= np.median(var_ratio)
        
        def auroc_on_mask(signal, mask):
            if mask.sum() < 10:
                return None
            return compute_auroc(signal[mask], is_error[mask])
        
        subset_result = {
            "n_images": len(labels),
            "consensus_accuracy": float(consensus_acc),
            "error_rate": float(err_rate),
            "single_backbone_baselines": single_results,
            "best_single_model": best_single["name"],
            "best_single_auroc": best_single["auroc"],
            "ensemble_signals": ensemble_eval,
            "bootstrap_agreement_minus_entropy": ci,
            "variance_decomposition": {
                "corr_entropy_vs_total_var": float(corr_ent_total),
                "corr_agreement_vs_var_ratio": float(corr_agr_ratio),
                "corr_agreement_vs_within_class_var": float(corr_agr_var),
                "auroc_low_var_agreement": auroc_on_mask(ensemble_agr, low_var_mask),
                "auroc_low_var_entropy": auroc_on_mask(ensemble_ent, low_var_mask),
                "auroc_high_var_agreement": auroc_on_mask(ensemble_agr, high_var_mask),
                "auroc_high_var_entropy": auroc_on_mask(ensemble_ent, high_var_mask),
            },
        }
        
        results[subset] = subset_result
        
        # ── Print summary ──
        print(f"  N={len(labels)}, consensus_acc={consensus_acc:.4f}")
        print(f"  Single backbone best entropy AUROC: {best_single['name']}={best_single['auroc']:.4f}")
        print(f"  Ensemble entropy AUROC:           {ensemble_eval['ensemble_entropy']['auroc']:.4f}")
        print(f"  Agreement AUROC:                  {ensemble_eval['agreement_top1']['auroc']:.4f}")
        print(f"  Agreement - Entropy:              {ci['mean']:.4f} [{ci['ci_lo']:.4f}, {ci['ci_hi']:.4f}]")
        print(f"  Variance decomposition:")
        print(f"    corr(entropy, total_var) = {corr_ent_total:.3f}")
        print(f"    corr(agreement, 1-var_ratio) = {corr_agr_ratio:.3f}")
        print(f"  Low var ratio: agr AUROC={auroc_on_mask(ensemble_agr, low_var_mask):.3f}, "
              f"ent AUROC={auroc_on_mask(ensemble_ent, low_var_mask):.3f}")
        print(f"  High var ratio: agr AUROC={auroc_on_mask(ensemble_agr, high_var_mask):.3f}, "
              f"ent AUROC={auroc_on_mask(ensemble_ent, high_var_mask):.3f}")
    
    # ── Save ──
    report = results
    report_path = OUT / "anchoring_theory_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nSaved {report_path}")
    
    txt = []
    for subset, r in results.items():
        txt.append(f"\n{'='*60}")
        txt.append(f"  {subset}  (N={r['n_images']})")
        txt.append(f"{'='*60}")
        txt.append(f"  Consensus accuracy: {r['consensus_accuracy']:.4f}")
        txt.append(f"  Best single model entropy AUROC: {r['best_single_auroc']:.4f}")
        txt.append(f"  Ensemble entropy AUROC: {r['ensemble_signals']['ensemble_entropy']['auroc']:.4f}")
        txt.append(f"  Agreement AUROC: {r['ensemble_signals']['agreement_top1']['auroc']:.4f}")
        ci = r['bootstrap_agreement_minus_entropy']
        txt.append(f"  Δ(agr−ent) CI: {ci['mean']:.4f} [{ci['ci_lo']:.4f}, {ci['ci_hi']:.4f}]")
        txt.append(f"  Selective acc @50: agr={r['ensemble_signals']['agreement_top1']['selacc@50']:.4f}, "
                   f"ent={r['ensemble_signals']['ensemble_entropy']['selacc@50']:.4f}")
        vd = r['variance_decomposition']
        txt.append(f"  Variance decomposition:")
        txt.append(f"    ρ(entropy, total_var) = {vd['corr_entropy_vs_total_var']:.3f}")
        txt.append(f"    ρ(agreement, 1−var_ratio) = {vd['corr_agreement_vs_var_ratio']:.3f}")
        txt.append(f"    Low var ratio: agr={vd['auroc_low_var_agreement']:.3f}, ent={vd['auroc_low_var_entropy']:.3f}")
        txt.append(f"    High var ratio: agr={vd['auroc_high_var_agreement']:.3f}, ent={vd['auroc_high_var_entropy']:.3f}")
        txt.append(f"  Single model baselines:")
        for mk, ev in r['single_backbone_baselines'].items():
            txt.append(f"    {mk}: max_logit={ev['max_logit']['auroc']:.3f} "
                       f"softmax={ev['softmax_response']['auroc']:.3f} "
                       f"entropy={ev['entropy']['auroc']:.3f} "
                       f"margin={ev['margin']['auroc']:.3f}")
    
    txt_path = OUT / "anchoring_theory_report.txt"
    with open(txt_path, "w") as f:
        f.write("\n".join(txt))
    print(f"Saved {txt_path}")


if __name__ == "__main__":
    main()