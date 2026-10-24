"""
analysis/anchoring_leaveout_sensitivity.py — R6: leave-out sensitivity
======================================================================
Reviewer (internal audit) finding: the soft-agreement headline win rate
(23/27 vs entropy) is concentrated in the 7 ImageNet-family OOD benchmarks.
This script quantifies:

  1. leave-out-ImageNet-family sensitivity: W/T/L and mean delta of soft
     (and hard) agreement vs entropy on the remaining n=20 datasets,
     plus the same leave-out for soft vs MSP;
  2. win-magnitude bucketing: how many of soft's wins vs entropy have
     |delta| < 0.02 / [0.02, 0.03) / >= 0.03 (bootstrap CI must exclude 0
     for a "win", so small-delta wins are still significant, just small);
  3. per-stratum leave-out summaries (OOD, non-ImageNet OOD, multilabel,
     standard single-label).

Input : results/anchoring_meta_report.json (committed; per-dataset
        delta_hard/soft_{mean,lo,hi} and p/kappa/p_star)
        results/anchoring_sota_csf_report.json (soft vs MSP deltas)
Output: results/anchoring_leaveout_sensitivity_report.{json,txt}

Usage: python analysis/anchoring_leaveout_sensitivity.py
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

IMAGENET_FAMILY = {"imagenet_r", "imagenet_a", "imagenet_sketch",
                   "imagenet_v2_matched", "imagenet_v2_threshold",
                   "imagenet_v2_top", "imagenet_c"}
OOD_KEYS = {"imagenet_r", "imagenet_a", "imagenet_sketch",
            "imagenet_v2_matched", "imagenet_v2_threshold",
            "imagenet_v2_top", "imagenet_c", "cifar10c", "cifar100c"}
MULTILABEL_KEYS = {"teacher_behavior", "voc2007", "coco", "multithumos"}


def classify(lo, hi):
    if lo > 0:
        return "win"
    if hi < 0:
        return "loss"
    return "tie"


def summarize(rows, prefix):
    w = t = l = 0
    deltas = []
    for r in rows:
        cls = classify(r[f"delta_{prefix}_lo"], r[f"delta_{prefix}_hi"])
        w += cls == "win"; t += cls == "tie"; l += cls == "loss"
        deltas.append(r[f"delta_{prefix}_mean"])
    return {"n": len(rows), "win": w, "tie": t, "loss": l,
            "mean_delta": float(np.mean(deltas)),
            "mean_abs_delta": float(np.mean(np.abs(deltas))),
            "min_delta": float(np.min(deltas)), "max_delta": float(np.max(deltas))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap_n", type=int, default=10000)
    args = parser.parse_args()

    ts = time.strftime("%Y%m%d_%H%M%S")
    setup_logging(ts, name="anchoring_leaveout_sensitivity")
    log = logging.getLogger()

    meta = json.load(open(OUT_DIR / "anchoring_meta_report.json"))
    sota = json.load(open(OUT_DIR / "anchoring_sota_csf_report.json"))
    rows = meta["results"]
    by_name = {r["dataset"]: r for r in rows}
    sota_rows = sota["results"]
    sota_by_name = {r["dataset"]: r for r in sota_rows}
    log.info(f"loaded {len(rows)} rows (meta), {len(sota_rows)} rows (sota)")

    def subset(pred):
        return [r for r in rows if pred(r["dataset"])]

    out = {"args": vars(args), "timestamp": ts,
           "imagenet_family": sorted(IMAGENET_FAMILY)}

    # 1. leave-out-ImageNet sensitivity (hard + soft vs entropy)
    all_rows = subset(lambda k: True)
    no_inet = subset(lambda k: k not in IMAGENET_FAMILY)
    only_inet = subset(lambda k: k in IMAGENET_FAMILY)
    out["vs_entropy"] = {
        "all_27": {sig: summarize(all_rows, sig)
                   for sig in ("hard", "soft")},
        "leave_out_imagenet_n20": {sig: summarize(no_inet, sig)
                                   for sig in ("hard", "soft")},
        "imagenet_only_n7": {sig: summarize(only_inet, sig)
                             for sig in ("hard", "soft")},
    }

    # 2. soft vs MSP leave-out (from sota_csf report)
    def sota_summarize(pred):
        rs = [r for r in sota_rows if pred(r["dataset"])]
        w = t = l = 0
        deltas = []
        for r in rs:
            lo, hi = r["soft_vs_msp_lo"], r["soft_vs_msp_hi"]
            cls = classify(lo, hi)
            w += cls == "win"; t += cls == "tie"; l += cls == "loss"
            deltas.append(r["soft_vs_msp_mean"])
        return {"n": len(rs), "win": w, "tie": t, "loss": l,
                "mean_delta": float(np.mean(deltas))}

    out["vs_msp"] = {
        "all_27": sota_summarize(lambda k: True),
        "leave_out_imagenet_n20": sota_summarize(
            lambda k: k not in IMAGENET_FAMILY),
        "imagenet_only_n7": sota_summarize(lambda k: k in IMAGENET_FAMILY),
    }

    # 3. win-magnitude bucketing (soft vs entropy)
    buckets = {"small_lt_0p02": [], "mid_0p02_0p03": [], "large_ge_0p03": []}
    for r in all_rows:
        if classify(r["delta_soft_lo"], r["delta_soft_hi"]) != "win":
            continue
        d = r["delta_soft_mean"]
        if abs(d) < 0.02:
            buckets["small_lt_0p02"].append(r["dataset"])
        elif abs(d) < 0.03:
            buckets["mid_0p02_0p03"].append(r["dataset"])
        else:
            buckets["large_ge_0p03"].append(r["dataset"])
    out["win_magnitude_buckets"] = {k: {"n": len(v), "datasets": v}
                                    for k, v in buckets.items()}

    # 4. stratum summaries (soft vs entropy)
    strata = {
        "ood_all": lambda k: k in OOD_KEYS,
        "ood_non_imagenet": lambda k: k in {"cifar10c", "cifar100c"},
        "ood_imagenet": lambda k: k in IMAGENET_FAMILY,
        "multilabel": lambda k: k in MULTILABEL_KEYS,
        "standard_single_label": lambda k: k not in OOD_KEYS
        and k not in MULTILABEL_KEYS,
    }
    out["strata_soft_vs_entropy"] = {
        name: summarize(subset(pred), "soft") for name, pred in strata.items()
    }
    # saturated-regime classification for narrative
    out["saturated_datasets"] = sorted(
        r["dataset"] for r in all_rows
        if r["p"] < r["p_star"])

    lines = ["=" * 96, "  R6 leave-out sensitivity + win-magnitude buckets",
             "=" * 96, ""]
    for block in ("vs_entropy", "vs_msp"):
        lines.append(f"  soft/hard vs entropy (or MSP): {block}")
        if block == "vs_entropy":
            for scope, d in out[block].items():
                for sig, s in d.items():
                    lines.append(
                        f"    {scope:24s} {sig:4s} n={s['n']:>2}  "
                        f"W{s['win']:>2}/T{s['tie']:>2}/L{s['loss']:>2}  "
                        f"mean {s['mean_delta']:+.4f}")
        else:
            for scope, s in out[block].items():
                lines.append(
                    f"    {scope:24s} {s['n']:>2}  "
                    f"W{s['win']:>2}/T{s['tie']:>2}/L{s['loss']:>2}  "
                    f"mean {s['mean_delta']:+.4f}")
        lines.append("")
    lines.append("  soft-win magnitude buckets (vs entropy, 23 wins):")
    for k, v in out["win_magnitude_buckets"].items():
        lines.append(f"    {k:16s} n={v['n']:>2}  {sorted(v['datasets'])}")
    lines.append("")
    lines.append("  strata (soft vs entropy):")
    for name, s in out["strata_soft_vs_entropy"].items():
        lines.append(f"    {name:26s} n={s['n']:>2}  "
                     f"W{s['win']:>2}/T{s['tie']:>2}/L{s['loss']:>2}  "
                     f"mean {s['mean_delta']:+.4f}")
    lines.append("")
    lines.append("  saturated datasets (p < p*): %d %s"
                 % (len(out["saturated_datasets"]), out["saturated_datasets"]))
    lines.append("=" * 96)
    txt = "\n".join(lines)

    with open(OUT_DIR / "anchoring_leaveout_sensitivity_report.json", "w") as f:
        json.dump(out, f, indent=2, default=lambda o: float(o))
    with open(OUT_DIR / "anchoring_leaveout_sensitivity_report.txt", "w") as f:
        f.write(txt)
    print(); print(txt)
    return 0


if __name__ == "__main__":
    exit(main())
