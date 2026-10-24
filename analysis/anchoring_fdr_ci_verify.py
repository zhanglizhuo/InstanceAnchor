"""
analysis/anchoring_fdr_ci_verify.py — R4 companion: CI-win survival under FDR
=============================================================================
Cross-checks that every CI-based win in the committed B=10,000 reports
(B=CI excludes 0 criterion) survives Benjamini-Hochberg FDR at q=0.05 in the
216-test family of results/anchoring_fdr_report.json (27 datasets x 8 CSFs,
B=2,000 coupled-bootstrap p-values).

Reads (all committed):
  results/anchoring_fdr_report.json        (per-test q_all, q_within_csf)
  results/anchoring_meta_report.json       (soft vs entropy CI, B=10000)
  results/anchoring_sota_csf_report.json   (soft vs 7 CSFs CI, B=10000)

Output: results/anchoring_fdr_ci_verify_report.{json,txt}

Usage: python analysis/anchoring_fdr_ci_verify.py [--q 0.05]
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase0_oracle_analysis import setup_logging, OUT_DIR  # noqa: E402

CSFS = ["entropy", "msp", "margin", "max_logit", "energy",
        "ensemble_mi", "pred_var", "hard_agree"]


def cls(lo, hi):
    if lo > 0:
        return "win"
    if hi < 0:
        return "loss"
    return "tie"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--q", type=float, default=0.05)
    args = parser.parse_args()

    ts = time.strftime("%Y%m%d_%H%M%S")
    setup_logging(ts, name="anchoring_fdr_ci_verify")
    log = logging.getLogger()

    fdr = json.load(open(OUT_DIR / "anchoring_fdr_report.json"))
    meta = json.load(open(OUT_DIR / "anchoring_meta_report.json"))
    sota = json.load(open(OUT_DIR / "anchoring_sota_csf_report.json"))

    qmap = {(t["csf"], t["dataset"]): (t["q_all"], t["q_within_csf"])
            for t in fdr["tests"]}

    out = {"args": vars(args), "timestamp": ts, "q": args.q,
           "per_csf": {}, "all_survive": True}
    lines = ["=" * 96,
             "  R4 CI-win survival under BH-FDR q=0.05 (216-test family)",
             "=" * 96, ""]

    for csf in CSFS:
        if csf == "entropy":
            ws = [r["dataset"] for r in meta["results"]
                  if cls(r["delta_soft_lo"], r["delta_soft_hi"]) == "win"]
        else:
            ws = [r["dataset"] for r in sota["results"]
                  if cls(r[f"soft_vs_{csf}_lo"], r[f"soft_vs_{csf}_hi"]) == "win"]
        lost = [d for d in ws if qmap[(csf, d)][0] >= args.q]
        maxq = max(qmap[(csf, d)][0] for d in ws) if ws else float("nan")
        out["per_csf"][csf] = {"n_ci_wins": len(ws), "max_q_all": maxq,
                               "lost_to_fdr": lost}
        if lost:
            out["all_survive"] = False
        lines.append(f"  {csf:12s} CI-wins {len(ws):>2}  max q_all={maxq:.4f}  "
                     f"lost to FDR: {lost if lost else 'NONE'}")
        log.info(f"{csf}: {len(ws)} wins, max q {maxq:.4f}, lost {lost}")

    lines.append("")
    lines.append(f"  ALL CI-BASED WINS SURVIVE BH-FDR at q={args.q}: "
                 f"{out['all_survive']}")
    lines.append("=" * 96)
    txt = "\n".join(lines)

    with open(OUT_DIR / "anchoring_fdr_ci_verify_report.json", "w") as f:
        json.dump(out, f, indent=2, default=lambda o: float(o))
    with open(OUT_DIR / "anchoring_fdr_ci_verify_report.txt", "w") as f:
        f.write(txt)
    print(); print(txt)
    return 0


if __name__ == "__main__":
    exit(main())
