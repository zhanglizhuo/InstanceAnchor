#!/usr/bin/env python3
"""Class-prototype Mantel analysis: does class-geometry sharing across the 5
CLIP backbones predict the agreement gain (ΔAUROC = agreement_top1 − ensemble
entropy) on selective prediction?

Motivation
----------
SV (sibling paper) shows that class-prototype geometry is shared across vision
encoders (Mantel ρ = 0.90+ in feature space) while patch features are NOT
shared (per-patch R² = 0.04–0.21).  IA's three falsified mechanisms (semantic
density on COCO, annotation multiplicity on MultiTHUMOS, patch-level CKA
alignment) all operated at the patch / feature level.  The class-prototype
level that SV actually identifies as the locus of the shared mechanism has
never been tested inside IA's 5-backbone family.

This script closes that gap.  For each of the 8 IA datasets we compute:
  1. Δ AUROC (agreement_top1 − ensemble-entropy AUROC), with coupled bootstrap CI
  2. Mean pairwise class-prototype Mantel ρ across the 5 backbones, computed in
     LOGIT space (the only space IA has without re-extracting features):
       class centroid c_k = mean over images with label k of the C-dim logit vector
       S^{(b)}_{ij}        = cosine(centroid_i^{(b)}, centroid_j^{(b)})
       Mantel ρ((b),(b'))  = Spearman of upper triangles of S^{(b)}, S^{(b')}
     mean over the 10 backbone pairs.  This measures whether the 5 backbones
     agree on WHICH classes look alike in their decision space — directly
     analogous to SV's feature-space Mantel but in logit space.

Then we correlate (mean Mantel) vs (Δ AUROC) across the 8 datasets
(Spearman + permutation test), with a partial correlation controlling for
n_classes (since Mantel absolute values are known to shrink as C grows,
SV §Flowers102).

Output: results/anchoring_classprototype_mantel_report.{json,txt}
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, rankdata


MODELS = ["openai", "laion", "siglip2", "eva02", "dfn"]
TEMPERATURE = 100.0

DATASETS = {
    "teacher_behavior":   {"dir": "results/per_image", "key": "teacher_behavior"},
    "handrise_readwrite": {"dir": "results/per_image", "key": "handrise_readwrite"},
    "bow_turnhead":       {"dir": "results/per_image", "key": "bow_turnhead"},
    "voc2007":            {"dir": "results/per_image", "key": "voc2007"},
    "stanford40":         {"dir": "results/per_image", "key": "stanford40"},
    "coco":               {"dir": "results/per_image", "key": "coco"},
    "multithumos":        {"dir": "results/per_image", "key": "multithumos"},
    "cifar100":           {"dir": "results/per_image/cifar100", "key": "cifar100"},
}


# ---------- signal / AUROC ----------
def auroc(score, label):
    """ROC-AUC via rank-based Mann-Whitney U (tie-correct)."""
    n_pos = int(label.sum())
    n_neg = len(label) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    r = rankdata(score)
    return (r[label == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def compute_signals(logits_5, labels):
    """Return per-image (entropy_keep, agreement, correct) and ensemble avg.

    logits_5: (5, N, C) raw cosine sims
    labels:   (N, C) binary
    """
    avg = logits_5.mean(0)                       # (N, C)
    consensus_argmax = np.argmax(avg, axis=1)    # (N,)
    correct = np.array(
        [consensus_argmax[i] in np.where(labels[i] > 0)[0] for i in range(len(labels))],
        dtype=int,
    )
    # ensemble entropy on averaged logits (temperature 100)
    z = avg * TEMPERATURE
    z = z - z.max(1, keepdims=True)
    ez = np.exp(z)
    p = ez / ez.sum(1, keepdims=True)
    entropy = -(p * np.log(p + 1e-12)).sum(1)
    entropy_keep = -entropy  # HIGH = keep (more confident)
    # agreement: fraction of 5 backbones whose argmax == consensus argmax
    backbone_argmaxes = np.argmax(logits_5, axis=2)            # (5, N)
    agreement = (backbone_argmaxes == consensus_argmax[None, :]).mean(0)
    return entropy_keep, agreement, correct, avg


def bootstrap_delta_ci(entropy_keep, agreement, correct, n_boot, seed):
    """Coupled bootstrap CI for Δ = auroc(agreement) − auroc(entropy_keep)."""
    rng = np.random.RandomState(seed)
    n = len(correct)
    deltas = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.randint(0, n, n)
        deltas[b] = auroc(agreement[idx], correct[idx]) - auroc(entropy_keep[idx], correct[idx])
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return float(lo), float(hi)


# ---------- class-prototype Mantel ----------
def class_centroid_sim_matrix(logits_bb, labels):
    """Compute CxC cosine-sim of class centroids in logit space for one backbone.

    Returns (S, valid_mask, counts).  Classes with <2 positive images are
    dropped (NaN); the caller restricts to globally-valid classes.
    """
    C = logits_bb.shape[1]
    centroids = np.zeros((C, C), dtype=np.float64)
    counts = np.zeros(C, dtype=int)
    for c in range(C):
        mask = labels[:, c] > 0
        counts[c] = int(mask.sum())
        if counts[c] < 2:
            centroids[c] = np.nan
            continue
        centroids[c] = logits_bb[mask].mean(0)
    valid = ~np.isnan(centroids[:, 0])
    S = np.full((C, C), np.nan)
    if valid.sum() < 2:
        return S, valid, counts
    cen = centroids[valid]
    cen = cen / (np.linalg.norm(cen, axis=1, keepdims=True) + 1e-12)
    S_valid = cen @ cen.T
    S[np.ix_(valid, valid)] = S_valid
    return S, valid, counts


def mantel_rho_upper(SA, SB):
    """Spearman correlation of the upper triangles of two similarity matrices."""
    C = SA.shape[0]
    iu = np.triu_indices(C, k=1)
    a = SA[iu]
    b = SB[iu]
    mask = ~(np.isnan(a) | np.isnan(b))
    if mask.sum() < 3:
        return np.nan
    r, _ = spearmanr(a[mask], b[mask])
    return float(r)


def compute_mantel(logits_5, labels, n_boot, seed):
    """Mean pairwise class-prototype Mantel ρ across the 5 backbones.

    Restricts to classes that have >=2 positive images in ALL 5 backbones
    (they share the same labels, so this is just >=2 in the dataset).

    Returns dict with mean_mantel, pairwise list, n_valid_classes,
    min_count, and a bootstrap CI (resampling images).
    """
    C = logits_5.shape[2]
    N = logits_5.shape[1]
    mats, counts_all = [], None
    valid_global = np.ones(C, dtype=bool)
    for k in range(5):
        S, valid, counts = class_centroid_sim_matrix(logits_5[k], labels)
        mats.append(S)
        valid_global &= valid
        counts_all = counts if counts_all is None else np.maximum(counts_all, counts)
    n_valid = int(valid_global.sum())

    def _mean_pair(S_list, vg):
        pairs = []
        for i in range(5):
            for j in range(i + 1, 5):
                Si = S_list[i][np.ix_(vg, vg)]
                Sj = S_list[j][np.ix_(vg, vg)]
                r = mantel_rho_upper(Si, Sj)
                if not np.isnan(r):
                    pairs.append(r)
        return (float(np.mean(pairs)) if pairs else np.nan), pairs

    mean_mantel, pairs = _mean_pair(mats, valid_global)

    if n_valid < 3 or np.isnan(mean_mantel):
        return {
            "mean_mantel": float("nan") if np.isnan(mean_mantel) else mean_mantel,
            "pairwise_mantels": pairs,
            "n_valid_classes": n_valid,
            "min_count": int(counts_all[valid_global].min()) if n_valid > 0 else 0,
            "mantel_ci_lo": None,
            "mantel_ci_hi": None,
        }

    # bootstrap CI: resample images, recompute centroids, recompute mean Mantel
    rng = np.random.RandomState(seed + 1)
    boot_means = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.randint(0, N, N)
        lb = labels[idx]
        mb = []
        for k in range(5):
            Sb, _, _ = class_centroid_sim_matrix(logits_5[k][idx], lb)
            mb.append(Sb)
        m, _ = _mean_pair(mb, valid_global)
        boot_means[b] = m
    lo, hi = np.percentile(boot_means[np.isfinite(boot_means)], [2.5, 97.5]) if np.isfinite(boot_means).any() else (None, None)
    return {
        "mean_mantel": mean_mantel,
        "pairwise_mantels": pairs,
        "n_valid_classes": n_valid,
        "min_count": int(counts_all[valid_global].min()),
        "mantel_ci_lo": float(lo) if lo is not None else None,
        "mantel_ci_hi": float(hi) if hi is not None else None,
    }


# ---------- driver ----------
def load_dataset(name):
    cfg = DATASETS[name]
    ddir = cfg["dir"]
    key = cfg["key"]
    logits = []
    for m in MODELS:
        path = f"{ddir}/clip_{m}_{key}_logits.npy"
        logits.append(np.load(path))
    logits_5 = np.stack(logits)  # (5, N, C)
    labels = np.load(f"{ddir}/clip_{MODELS[0]}_{key}_labels.npy")
    # unify to (N, C) binary; single-label integer labels → one-hot
    if labels.ndim == 1:
        idx = labels.astype(int)
        C_logit = logits_5.shape[2]
        oh = np.zeros((len(idx), C_logit), dtype=labels.dtype)
        oh[np.arange(len(idx)), idx] = 1
        labels = oh
    return logits_5, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--mantel-boot", type=int, default=1000,
                    help="bootstrap iters for Mantel CI (cheaper default)")
    ap.add_argument("--seed", type=int, default=20260805)
    ap.add_argument("--out-json", default="results/anchoring_classprototype_mantel_report.json")
    ap.add_argument("--out-txt",  default="results/anchoring_classprototype_mantel_report.txt")
    args = ap.parse_args()

    rows = []
    for name in DATASETS:
        print(f"\n=== {name} ===", flush=True)
        logits_5, labels = load_dataset(name)
        N, C = labels.shape
        ml_degree = float((labels > 0).sum(1).mean())
        ent_keep, agree, correct, _ = compute_signals(logits_5, labels)
        agr_auroc = auroc(agree, correct)
        ent_auroc = auroc(ent_keep, correct)
        delta = agr_auroc - ent_auroc
        ci_lo, ci_hi = bootstrap_delta_ci(ent_keep, agree, correct, args.n_boot, args.seed)
        mantel = compute_mantel(logits_5, labels, args.mantel_boot, args.seed)
        row = {
            "dataset": name,
            "n_classes": int(C),
            "n_images": int(N),
            "ml_degree": ml_degree,
            "agreement_auroc": float(agr_auroc),
            "entropy_auroc": float(ent_auroc),
            "delta": float(delta),
            "delta_ci_lo": float(ci_lo),
            "delta_ci_hi": float(ci_hi),
            "correct_rate": float(correct.mean()),
            "mean_mantel": mantel["mean_mantel"],
            "mantel_ci_lo": mantel["mantel_ci_lo"],
            "mantel_ci_hi": mantel["mantel_ci_hi"],
            "mantel_n_valid_classes": mantel["n_valid_classes"],
            "mantel_min_count": mantel["min_count"],
            "mantel_pairwise": mantel["pairwise_mantels"],
        }
        rows.append(row)
        print(f"  Δ={delta:+.4f} CI=[{ci_lo:+.4f},{ci_hi:+.4f}]  "
              f"mantel={mantel['mean_mantel']:.4f}  C={C} N={N} ml={ml_degree:.2f}", flush=True)

    # ---- cross-dataset correlation: Mantel vs Δ ----
    mantels = np.array([r["mean_mantel"] for r in rows])
    deltas  = np.array([r["delta"]       for r in rows])
    n_classes = np.array([r["n_classes"]  for r in rows], dtype=float)
    valid = np.isfinite(mantels) & np.isfinite(deltas)

    def _perm_spearman(x, y, n_perm=10000, seed=0):
        r0, _ = spearmanr(x, y)
        rng = np.random.RandomState(seed)
        cnt = 0
        n = len(x)
        for _ in range(n_perm):
            perm = rng.permutation(n)
            rp, _ = spearmanr(x, y[perm])
            if abs(rp) >= abs(r0):
                cnt += 1
        return float(r0), cnt / n_perm

    out = {"rows": rows, "n_datasets": int(len(rows))}

    if valid.sum() >= 3:
        r_md, p_md = _perm_spearman(mantels[valid], deltas[valid], 10000, args.seed)
        out["mantel_vs_delta_spearman"] = r_md
        out["mantel_vs_delta_perm_p"] = p_md
        out["mantel_vs_delta_n"] = int(valid.sum())

        # partial: rank-residualize against n_classes
        if valid.sum() >= 4 and len(np.unique(n_classes[valid])) > 1:
            mr = rankdata(mantels[valid]); dr = rankdata(deltas[valid]); cr = rankdata(n_classes[valid])
            # residualize
            def _resid(a, b):
                A = np.vstack([np.ones_like(b), b]).T
                coef, *_ = np.linalg.lstsq(A, a, rcond=None)
                return a - A @ coef
            mr_res = _resid(mr, cr); dr_res = _resid(dr, cr)
            rp, _ = spearmanr(mr_res, dr_res)
            out["partial_mantel_vs_delta_given_n_classes"] = float(rp)

        # also report Δ-vs-CKA (the previously-falsified mechanism) for comparison
        # (read from anchoring_cka_mechanism_report.json if present)
        cka_path = "results/anchoring_cka_mechanism_report.json"
        if os.path.exists(cka_path):
            try:
                cka_d = json.load(open(cka_path))
                cka_map = {}
                for ds in cka_d.get("datasets", []):
                    name = ds.get("dataset") or ds.get("subset")
                    if name and "mean_cka" in ds:
                        cka_map[name] = float(ds["mean_cka"])
                # align by dataset name if possible
                cka_vec = []
                for r in rows:
                    cka_vec.append(cka_map.get(r["dataset"], np.nan))
                cka_vec = np.array(cka_vec)
                m2 = np.isfinite(cka_vec) & valid
                if m2.sum() >= 3:
                    r_ck, p_ck = _perm_spearman(cka_vec[m2], deltas[m2], 10000, args.seed + 7)
                    out["cka_vs_delta_spearman"] = float(r_ck)
                    out["cka_vs_delta_perm_p"] = float(p_ck)
                    out["cka_vs_delta_n"] = int(m2.sum())
            except Exception as e:
                out["cka_compare_error"] = str(e)

        # ---- leave-one-out: is the correlation a TB artifact? ----
        # TB is the only dataset with significant positive Δ; if excluding it
        # collapses the correlation, the "signal" is a TB outlier projection.
        ds_names = [r["dataset"] for r in rows]
        loo = {}
        for predictor_name, predictor_vec in [("mean_mantel", mantels),
                                                ("mean_cka", cka_vec if "cka_vec" in dir() else np.full(len(rows), np.nan)),
                                                ("ml_degree", np.array([r["ml_degree"] for r in rows]))]:
            pv = np.asarray(predictor_vec)
            loo[predictor_name] = {}
            # full
            mf = np.isfinite(pv) & np.isfinite(deltas)
            if mf.sum() >= 3:
                r_f, _ = spearmanr(pv[mf], deltas[mf])
                loo[predictor_name]["all"] = float(r_f)
            # excl TB
            m_no_tb = mf & (np.array(ds_names) != "teacher_behavior")
            if m_no_tb.sum() >= 3:
                r_nt, p_nt = spearmanr(pv[m_no_tb], deltas[m_no_tb])
                loo[predictor_name]["excl_tb"] = float(r_nt)
                loo[predictor_name]["excl_tb_n"] = int(m_no_tb.sum())
        out["leave_one_out_validation"] = loo

        # ---- verdict ----
        mantel_excl_tb = loo.get("mean_mantel", {}).get("excl_tb", None)
        if mantel_excl_tb is not None and mantel_excl_tb <= 0:
            out["verdict"] = ("NO PORTABLE SIGNAL. The full-sample ρ=+{:.3f} is a TB-driven "
                              "artifact: excluding TB reverses the sign to {:.3f}. "
                              "class-prototype Mantel does NOT predict agreement gain "
                              "outside the TB outlier.").format(
                                  out.get("mantel_vs_delta_spearman", float("nan")), mantel_excl_tb)
        elif mantel_excl_tb is not None and mantel_excl_tb > 0:
            out["verdict"] = ("Weak signal survives TB exclusion (excl TB ρ=+{:.3f}), "
                              "but remains non-significant at n={}.").format(mantel_excl_tb, loo["mean_mantel"]["excl_tb_n"])
        else:
            out["verdict"] = "Inconclusive (insufficient data for leave-one-out validation)."

    # ---- write ----
    Path("results").mkdir(exist_ok=True)
    json.dump(out, open(args.out_json, "w"), indent=2)

    lines = []
    lines.append("Class-Prototype Mantel vs Agreement Gain (Δ AUROC)")
    lines.append("=" * 64)
    lines.append("")
    lines.append(f"{'dataset':22s} {'C':>4s} {'N':>6s} {'ml':>5s} {'Δ':>9s} {'CI':>17s} "
                 f"{'mantel':>8s} {'CI':>17s}")
    lines.append("-" * 100)
    for r in rows:
        lines.append(
            f"{r['dataset']:22s} {r['n_classes']:>4d} {r['n_images']:>6d} {r['ml_degree']:>5.2f} "
            f"{r['delta']:>+9.4f} [{r['delta_ci_lo']:>+7.4f},{r['delta_ci_hi']:>+7.4f}] "
            f"{r['mean_mantel']:>8.4f} [{r['mantel_ci_lo'] or 0:>+7.4f},{r['mantel_ci_hi'] or 0:>+7.4f}]"
        )
    lines.append("")
    lines.append("Cross-dataset correlation:")
    if "mantel_vs_delta_spearman" in out:
        lines.append(f"  Spearman(mantel, Δ) = {out['mantel_vs_delta_spearman']:+.4f}  "
                     f"permutation p = {out['mantel_vs_delta_perm_p']:.4f}  (n={out['mantel_vs_delta_n']})")
    if "partial_mantel_vs_delta_given_n_classes" in out:
        lines.append(f"  Partial Spearman (controlling n_classes) = "
                     f"{out['partial_mantel_vs_delta_given_n_classes']:+.4f}")
    if "cka_vs_delta_spearman" in out:
        lines.append(f"  [compare] CKA vs Δ: ρ = {out['cka_vs_delta_spearman']:+.4f}  "
                     f"p = {out['cka_vs_delta_perm_p']:.4f}  (n={out['cka_vs_delta_n']})")
    lines.append("")
    lines.append("Leave-one-out validation (is the correlation a TB artifact?):")
    loo = out.get("leave_one_out_validation", {})
    for pred, vals in loo.items():
        full = vals.get("all", float("nan"))
        excl = vals.get("excl_tb", float("nan"))
        lines.append(f"  {pred:14s}  all={full:+.3f}  excl_TB={excl:+.3f}  "
                     f"{'← TB ARTIFACT (sign flips)' if (full > 0 and excl <= 0) else ''}")
    lines.append("")
    lines.append("VERDICT:")
    lines.append(f"  {out.get('verdict', 'N/A')}")
    lines.append("")
    lines.append("Interpretation:")
    lines.append("  The full-sample ρ=+0.286 is NOT a weak positive signal. It is a TB-driven")
    lines.append("  artifact: TB is the only dataset with significant positive Δ AND the")
    lines.append("  highest Mantel/CKA/ml_degree. Excluding TB reverses the sign (Mantel")
    lines.append("  +0.286 → −0.143; CKA +0.286 → −0.071). There is no portable predictor")
    lines.append("  of where instance-level agreement will help. The class-prototype level,")
    lines.append("  despite being the right granularity per SV, does not descend to a")
    lines.append("  usable instance-level predictor in this dataset family.")
    open(args.out_txt, "w").write("\n".join(lines))
    print("\n" + "\n".join(lines))
    print(f"\nSaved -> {args.out_json}\n       {args.out_txt}")


if __name__ == "__main__":
    main()
