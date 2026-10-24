"""
analysis/anchoring_loocv_prior_control.py — R2: LOO circularity control
=======================================================================
Reviewer (internal audit) finding: the joint (p, C, kappa) LOO refit uses
kappa, which is computed from the TEST-set labels (per-image agreement on
the labeled evaluation set). The 81.5%/88.9% sign recovery is therefore a
label-conditional CALIBRATION summary, not an ex-ante prediction.

This script quantifies how much of the LOO sign recovery is attributable to
the label-dependent feature kappa, by running the identical LOO regression
protocol (anchoring_meta.py.loocv) on nested feature sets:

  ["C"]                 : class count only (fully label-free prior)
  ["p", "C"]            : + per-backbone accuracy (requires labels, standard)
  ["p", "C", "kappa"]   : + test-label-derived kappa (the existing 81.5/88.9)

Auxiliary control on the n=20 subset with kappa_iid_mc (MC plurality
baseline derived from p and C, no agreement labels) vs empirical kappa.

Input : results/anchoring_meta_report.json (committed; rows have p, C,
        kappa, delta_hard_mean, delta_soft_mean)
        results/anchoring_kappa_meta_decomp_report.json (kappa_iid_mc)
Output: results/anchoring_loocv_prior_control_report.{json,txt}

Usage: python analysis/anchoring_loocv_prior_control.py [--bootstrap_n 10000]
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

FEATURE_SETS = {
    "C_only": ["C"],
    "pC": ["p", "C"],
    "pC_kappa": ["p", "C", "kappa"],
}
TARGETS = {"hard": "delta_hard_mean", "soft": "delta_soft_mean"}


def loocv(rows, feature_cols, target):
    """Same protocol as anchoring_meta.py.loocv: leave-one-out OLS on the
    (p, C, kappa) feature space; sign recovery of predicted vs actual delta."""
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
    return {"n_folds": n,
            "sign_acc": float(np.mean([f["sign_agree"] for f in per_fold])),
            "per_fold": per_fold, "preds": preds.tolist()}


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap_n", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ts = time.strftime("%Y%m%d_%H%M%S")
    setup_logging(ts, name="anchoring_loocv_prior_control")
    log = logging.getLogger()

    meta = json.load(open(OUT_DIR / "anchoring_meta_report.json"))
    rows = [r for r in meta["results"]
            if all(np.isfinite(r[c]) for c in ("p", "C", "kappa"))]
    log.info(f"loaded {len(rows)} rows from anchoring_meta_report.json")

    decomp = json.load(open(OUT_DIR / "anchoring_kappa_meta_decomp_report.json"))
    kiid = {d["name"]: d["kappa_iid_mc"] for d in decomp["datasets"]}
    n_iid = sum(1 for r in rows if r["dataset"] in kiid)

    out = {"args": vars(args), "timestamp": ts,
           "n_datasets": len(rows), "n_iid_subset": n_iid,
           "note": ("sign recovery of LOO OLS refit on nested feature sets; "
                    "kappa requires test labels (label-conditional), C and "
                    "kappa_iid_mc do not (except p, which needs labels)"),
           "by_target": {}, "iid_mc_control": {}}

    for tag, target in TARGETS.items():
        out["by_target"][tag] = {}
        for fs_name, cols in FEATURE_SETS.items():
            loo = loocv(rows, cols, target)
            ci = bootstrap_sign_acc(loo["preds"],
                                    [r[target] for r in rows],
                                    args.bootstrap_n, args.seed)
            out["by_target"][tag][fs_name] = {
                "features": cols,
                "sign_acc": loo["sign_acc"],
                "sign_ci": ci,
                "n_folds": loo["n_folds"],
            }
            log.info(f"  [{tag}] {fs_name:10s} sign_acc={loo['sign_acc']*100:.1f}% "
                     f"CI [{ci['ci_lo']*100:.1f},{ci['ci_hi']*100:.1f}]")

    # auxiliary: kappa_iid_mc (MC plurality, derived from p and C only)
    # STL-10 excluded (kappa_iid_mc NaN, degenerate plurality decomposition)
    sub = [r for r in rows
           if r["dataset"] in kiid
           and np.isfinite(kiid[r["dataset"]])]
    n_iid = len(sub)
    for tag, target in TARGETS.items():
        for fs_name, cols in (("pC_kappa_iidmc", ["p", "C"]), ("pC_kappa_emp", ["p", "C"])):
            work = []
            for r in sub:
                r2 = dict(r)
                if fs_name.endswith("iidmc"):
                    r2["kappa"] = kiid[r["dataset"]]
                work.append(r2)
            loo = loocv(work, ["p", "C", "kappa"], target)
            ci = bootstrap_sign_acc(loo["preds"],
                                    [r[target] for r in sub],
                                    args.bootstrap_n, args.seed)
            out["iid_mc_control"][f"{tag}_{fs_name}"] = {
                "n_folds": loo["n_folds"],
                "sign_acc": loo["sign_acc"],
                "sign_ci": ci,
            }
            log.info(f"  [iid-mc {tag}] {fs_name:12s} sign_acc={loo['sign_acc']*100:.1f}%")

    lines = ["=" * 96, "  R2 LOO circularity control (nested feature sets)",
             "=" * 96, ""]
    for tag in TARGETS:
        lines.append(f"  target: {TARGETS[tag]}")
        for fs_name, v in out["by_target"][tag].items():
            ci = v["sign_ci"]
            lines.append(f"    {fs_name:12s} {str(v['features']):28s} "
                         f"sign_acc {v['sign_acc']*100:5.1f}%  "
                         f"CI [{ci['ci_lo']*100:5.1f},{ci['ci_hi']*100:5.1f}]")
    lines.append("")
    lines.append("  kappa_iid_mc control (n=%d, MC plurality from p,C only):"
                 % n_iid)
    for k, v in out["iid_mc_control"].items():
        ci = v["sign_ci"]
        lines.append(f"    {k:24s} sign_acc {v['sign_acc']*100:5.1f}%  "
                     f"CI [{ci['ci_lo']*100:5.1f},{ci['ci_hi']*100:5.1f}]")
    lines.append("=" * 96)
    txt = "\n".join(lines)

    with open(OUT_DIR / "anchoring_loocv_prior_control_report.json", "w") as f:
        json.dump(out, f, indent=2, default=lambda o: float(o))
    with open(OUT_DIR / "anchoring_loocv_prior_control_report.txt", "w") as f:
        f.write(txt)
    print(); print(txt)
    return 0


if __name__ == "__main__":
    exit(main())
