#!/usr/bin/env python3
"""Regime-stratified soft agreement vs MSP comparison (Exp E extension).

Q: Claude cross-validation review suggested proving soft agreement
significantly leads MSP in a specific regime rather than merely tying.
This script stratifies the 20-dataset soft-vs-MSP bootstrap results
(existing: anchoring_sota_csf_report.json + anchoring_meta_report.json)
by (1) multi-labelness (ml_frac > 1.0), (2) OOD vs standard, and reports
per-stratum W/T/L counts and mean delta, plus a permutation sign test.

Reads committed evidence only; no recompute of per-dataset bootstraps.
Output: results/anchoring_sota_regime_report.{json,txt}
"""
import json
from pathlib import Path
from itertools import combinations
import numpy as np

OUT_DIR = Path("results")
SOTA = OUT_DIR / "anchoring_sota_csf_report.json"
META = OUT_DIR / "anchoring_meta_report.json"
OUT = OUT_DIR / "anchoring_sota_regime_report.json"
OUT_TXT = OUT_DIR / "anchoring_sota_regime_report.txt"

OOD_DATASETS = {"imagenet_a", "imagenet_r",
                # OOD expansion (reviewer round 3, pre-registered list)
                "imagenet_sketch", "imagenet_v2_matched",
                "imagenet_v2_threshold", "imagenet_v2_top", "imagenet_c",
                # non-ImageNet OOD (reviewer round 3 confound fix)
                "cifar10c", "cifar100c"}


def load():
    sota = json.load(open(SOTA))["results"]
    meta = {r["dataset"]: r for r in json.load(open(META))["results"]}
    rows = []
    for r in sota:
        m = meta[r["dataset"]]
        rows.append(
            {
                "dataset": r["dataset"],
                "C": r["C"],
                "acc": r["acc"],
                "ml_frac": m["ml_frac"],
                "multilabel": m["ml_frac"] > 1.0,
                "ood": r["dataset"] in OOD_DATASETS,
                "soft_vs_msp_mean": r["soft_vs_msp_mean"],
                "soft_vs_msp_lo": r["soft_vs_msp_lo"],
                "soft_vs_msp_hi": r["soft_vs_msp_hi"],
            }
        )
    return rows


def stratum_stats(rows):
    w = t = l = 0
    means = []
    for r in rows:
        means.append(r["soft_vs_msp_mean"])
        if r["soft_vs_msp_lo"] > 0:
            w += 1
        elif r["soft_vs_msp_hi"] < 0:
            l += 1
        else:
            t += 1
    mean_d = float(np.mean(means))
    # one-sample permutation test on mean delta (sign flip, 100k)
    rng = np.random.default_rng(42)
    arr = np.array(means)
    obs = arr.mean()
    perms = rng.choice([-1.0, 1.0], size=(100_000, len(arr)))
    null = (perms * arr).mean(1)
    p = float(np.mean(np.abs(null) >= abs(obs)))
    return {"n": len(rows), "wins": w, "ties": t, "losses": l,
            "mean_delta": mean_d, "perm_p": p,
            "datasets": [r["dataset"] for r in rows]}


def main():
    rows = load()
    strata = {
        "all_20": rows,
        "multilabel": [r for r in rows if r["multilabel"]],
        "singlelabel": [r for r in rows if not r["multilabel"]],
        "ood": [r for r in rows if r["ood"]],
        "standard_non_ood": [r for r in rows if not r["ood"]],
    }
    out = {"timestamp": "20260814", "strata": {}}
    lines = []
    for name, srows in strata.items():
        st = stratum_stats(srows)
        out["strata"][name] = st
        lines.append(
            f"{name:20s} n={st['n']:3d}  W/T/L={st['wins']}/{st['ties']}/{st['losses']}"
            f"  mean_delta={st['mean_delta']:+.4f}  perm_p={st['perm_p']:.4f}"
        )
    # per-dataset detail
    lines.append("")
    lines.append(f"{'dataset':22s} {'ml':6s} {'ood':4s} {'acc':7s} {'soft-msp':9s} verdict")
    for r in sorted(rows, key=lambda x: -x["soft_vs_msp_mean"]):
        if r["soft_vs_msp_lo"] > 0:
            v = "WIN"
        elif r["soft_vs_msp_hi"] < 0:
            v = "LOSS"
        else:
            v = "tie"
        lines.append(
            f"{r['dataset']:22s} {str(r['multilabel']):6s} {str(r['ood']):4s} {r['acc']:.3f} "
            f"{r['soft_vs_msp_mean']:+.4f}  {v}"
        )
    json.dump(out, open(OUT, "w"), indent=2)
    txt = "\n".join(lines)
    open(OUT_TXT, "w").write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()