"""
analysis/anchoring_regime_gap_correlation.py — Delta_alpha vs empirical delta
============================================================================
Computes the in-sample correlation between the theoretical conditional-agreement
gap Delta_alpha and the empirical AUROC delta (hard and soft agreement vs
entropy), at n=8 (original datasets) and n=27, with Pearson and Spearman
statistics, from the committed anchoring_meta_report.json rows.

This supersedes the ad-hoc n=7 values in anchoring_regime_correlation.json
(which excluded MultiTHUMOS without a stated protocol) and provides the
traceable numbers for the paper's "In-sample correlation" sentence.

Output: results/anchoring_regime_gap_correlation_report.{json,txt}

Usage: python analysis/anchoring_regime_gap_correlation.py
"""

import json
import time
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase0_oracle_analysis import OUT_DIR  # noqa: E402
from scipy import stats

CORE8 = {"teacher_behavior", "handrise_readwrite", "bow_turnhead", "voc2007",
         "stanford40", "coco", "cifar100", "multithumos"}


def main():
    meta = json.load(open(OUT_DIR / "anchoring_meta_report.json"))
    rows = [r for r in meta["results"]
            if np.isfinite(r.get("Delta_alpha", float("nan")))
            and np.isfinite(r.get("delta_hard_mean", float("nan")))
            and np.isfinite(r.get("delta_soft_mean", float("nan")))]
    out = {"timestamp": time.strftime("%Y%m%d_%H%M%S"), "correlations": {}}
    for scope, sub in (("n8_core", [r for r in rows if r["dataset"] in CORE8]),
                       ("n27_all", rows)):
        x = [r["Delta_alpha"] for r in sub]
        for target, key in (("delta_hard_mean", "hard"),
                            ("delta_soft_mean", "soft")):
            y = [r[target] for r in sub]
            pr = stats.pearsonr(x, y)
            sr = stats.spearmanr(x, y)
            out["correlations"][f"{scope}_{key}"] = {
                "n": len(sub),
                "pearson_rho": float(pr[0]), "pearson_p": float(pr[1]),
                "spearman_rho": float(sr[0]), "spearman_p": float(sr[1]),
            }
    with open(OUT_DIR / "anchoring_regime_gap_correlation_report.json", "w") as f:
        json.dump(out, f, indent=2)
    lines = ["=" * 70, "  Delta_alpha vs empirical delta (from meta report rows)",
             "=" * 70]
    for k, v in out["correlations"].items():
        lines.append(f"  {k}: n={v['n']} pearson {v['pearson_rho']:+.4f} "
                     f"(p={v['pearson_p']:.3f}) spearman {v['spearman_rho']:+.4f} "
                     f"(p={v['spearman_p']:.3f})")
    lines.append("=" * 70)
    txt = "\n".join(lines)
    with open(OUT_DIR / "anchoring_regime_gap_correlation_report.txt", "w") as f:
        f.write(txt)
    print(txt)


if __name__ == "__main__":
    main()
