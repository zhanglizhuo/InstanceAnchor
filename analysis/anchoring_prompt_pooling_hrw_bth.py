"""
analysis/anchoring_prompt_pooling_hrw_bth.py — B5 HRW/BTH 15-vote evidence
=========================================================================
The B5 paragraph claims 15-vote (CAPE-A/B/C pooled) numbers for HRW and BTH,
but the only committed evidence (crossprompt_feasibility_report.json) covers
TeacherBehavior. This script computes the same 15-vote protocol for the two
remaining SCB5 subsets from the committed CAPE-B/C per-image logits:

  15-vote stack = concat(CAPE-A logits, CAPE-B logits, CAPE-C logits) per
  backbone (K=5 backbones x 3 prompt sets = 15 votes); consensus = argmax of
  the 15-vote mean; correctness vs 15-vote consensus; agreement = fraction
  of the 15 votes matching the consensus; entropy = softmax entropy of the
  15-vote mean at T=100 (negative). Changes are computed vs the 5-vote
  (CAPE-A only) values; all deltas are coupled-bootstrap (B=10000, seed 42).

Input : results/per_image/clip_{bb}_{subset}_logits[_capeb|_capec].npy
Output: results/anchoring_prompt_pooling_hrw_bth_report.{json,txt}

Usage: python analysis/anchoring_prompt_pooling_hrw_bth.py
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
from pilot_perimage_anchoring import BACKBONES, auroc, bootstrap_auroc_diff  # noqa: E402

T = 100.0


def load_votes(subset):
    """15 votes per image: CAPE-A, CAPE-B, CAPE-C stacks concatenated."""
    votes = []
    gt = None
    for bb in BACKBONES:
        for suffix in ("", "_capeb", "_capec"):
            lp = DATA_DIR / f"clip_{bb}_{subset}_logits{suffix}.npy"
            bp = DATA_DIR / f"clip_{bb}_{subset}_labels{suffix}.npy"
            if not lp.exists():
                raise FileNotFoundError(lp)
            votes.append(np.load(lp).astype(np.float32))
            labels = np.load(bp).astype(np.float32)
            if gt is None:
                gt = labels
            elif not np.array_equal(gt, labels):
                raise RuntimeError(f"label mismatch {bb}/{suffix} on {subset}")
    return np.stack(votes, axis=0), gt


def correctness(cons, gt):
    if gt.ndim == 1:
        return cons == gt.astype(int)
    n = gt.shape[0]
    return np.array([cons[i] in np.where(gt[i] > 0)[0] for i in range(n)],
                    dtype=bool)


def softmax_entropy(z):
    z = z * T
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    p = e / e.sum(axis=1, keepdims=True)
    return (p * np.log(p + 1e-12)).sum(axis=1)


def analyze(subset, bootstrap_n, seed, log):
    votes, gt = load_votes(subset)
    V, N, C = votes.shape
    avg = votes.mean(axis=0)
    cons = np.argmax(avg, axis=1)
    correct = correctness(cons, gt)
    hard = (np.argmax(votes, axis=2) == cons[None, :]).mean(axis=0)
    ent = softmax_entropy(avg)
    # 5-vote (CAPE-A only) references
    avg5 = votes[::3].mean(axis=0)
    cons5 = np.argmax(avg5, axis=1)
    correct5 = correctness(cons5, gt)
    hard5 = (np.argmax(votes[::3], axis=2) == cons5[None, :]).mean(axis=0)
    ent5 = softmax_entropy(avg5)
    row = {
        "subset": subset, "N": int(N), "C": int(C),
        "consensus5_acc": float(correct5.mean()),
        "consensus15_acc": float(correct.mean()),
        "agreement5_auroc": auroc(hard5, correct),
        "agreement15_auroc": auroc(hard, correct),
        "entropy5_auroc": auroc(ent5, correct),
        "entropy15_auroc": auroc(ent, correct),
    }
    row["agr_change"] = bootstrap_auroc_diff(hard, hard5, correct,
                                             bootstrap_n, seed)
    row["ent_change"] = bootstrap_auroc_diff(ent, ent5, correct,
                                             bootstrap_n, seed)
    row["delta15"] = bootstrap_auroc_diff(hard, ent, correct,
                                          bootstrap_n, seed)
    row["delta5_at15target"] = bootstrap_auroc_diff(hard5, ent5, correct,
                                                    bootstrap_n, seed)
    log.info(f"{subset}: cons {row['consensus5_acc']:.3f}->{row['consensus15_acc']:.3f} "
             f"ent+{row['ent_change']['mean']:+.4f} agr+{row['agr_change']['mean']:+.4f} "
             f"d15={row['delta15']['mean']:+.4f}")
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap_n", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    ts = time.strftime("%Y%m%d_%H%M%S")
    setup_logging(ts, name="anchoring_prompt_pooling_hrw_bth")
    log = logging.getLogger()

    rows = [analyze(s, args.bootstrap_n, args.seed, log)
            for s in ("handrise_readwrite", "bow_turnhead")]
    out = {"args": vars(args), "timestamp": ts, "results": rows,
           "note": ("15-vote = CAPE-A/B/C pooled (5 backbones x 3 sets); "
                    "ALL signals (5-vote and 15-vote) evaluated against 15-vote consensus correctness (common target, matching crossprompt_feasibility.py); "
                    "changes vs 5-vote (CAPE-A) evaluated on the shared 5-vote "
                    "correctness target (same protocol as the TB row in "
                    "crossprompt_feasibility_report.json)")}
    with open(OUT_DIR / "anchoring_prompt_pooling_hrw_bth_report.json", "w") as f:
        json.dump(out, f, indent=2, default=lambda o: float(o))
    lines = ["=" * 90, "  B5 HRW/BTH 15-vote prompt pooling", "=" * 90]
    for r in rows:
        ac = r["agr_change"]; ec = r["ent_change"]; d15 = r["delta15"]
        lines.append(f"  {r['subset']}: cons {r['consensus5_acc']*100:.1f}%->"
                     f"{r['consensus15_acc']*100:.1f}%")
        d5 = r["delta5_at15target"]
        lines.append(f"    ent change {ec['mean']:+.4f} CI[{ec['ci_lo']:+.4f},{ec['ci_hi']:+.4f}]"
                     f" | agr change {ac['mean']:+.4f} CI[{ac['ci_lo']:+.4f},{ac['ci_hi']:+.4f}]"
                     f" | delta5@15 {d5['mean']:+.4f} CI[{d5['ci_lo']:+.4f},{d5['ci_hi']:+.4f}]"
                     f" | delta15 {d15['mean']:+.4f} CI[{d15['ci_lo']:+.4f},{d15['ci_hi']:+.4f}]")
    lines.append("=" * 90)
    txt = "\n".join(lines)
    with open(OUT_DIR / "anchoring_prompt_pooling_hrw_bth_report.txt", "w") as f:
        f.write(txt)
    print(); print(txt)
    return 0


if __name__ == "__main__":
    exit(main())
