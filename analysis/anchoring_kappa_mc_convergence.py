"""
analysis/anchoring_kappa_mc_convergence.py — MC convergence check for the
kappa i.i.d.-MC decomposition (appendix A, 19 datasets).

Reviewer-style concern (2026-08-16): the appendix states N=10^5 simulated
images; for high-accuracy datasets the wrong-consensus conditioning branch
may contain few events, inflating the estimation noise of kappa_iid_mc.

This script verifies convergence for all 19 datasets of the committed
decomposition report:
  (1) counts the wrong-consensus conditioning events at N=10^5,
  (2) recomputes kappa_iid_mc at N=10^7 (chunked, memory-bounded) using the
      identical estimator/seed semantics as simulate_iid_kappa (ties ->
      lowest class id via argmax on counts; voter b correct w.p. p_b,
      else uniform over C-1 wrong classes),
  (3) reports the relative delta between the two scales.

Input : results/anchoring_kappa_meta_decomp_report.json (canonical p, C)
Output: results/anchoring_kappa_mc_convergence_report.{json,txt}

Usage: python analysis/anchoring_kappa_mc_convergence.py
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase0_oracle_analysis import OUT_DIR  # noqa: E402
from anchoring_kappa_iid_baseline import chunked_iid_stats, simulate_iid_kappa  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")





def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk", type=int, default=100_000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    log = logging.getLogger("anchoring_kappa_mc_convergence")
    log.info("starting MC convergence check (N=1e5 vs 1e7, chunk=%d)", args.chunk)
    t0 = time.time()

    decomp = json.load(open(OUT_DIR / "anchoring_kappa_meta_decomp_report.json"))
    from anchoring_kappa_meta_decomp import per_backbone_accs_meta  # noqa: E402
    stability = []
    for row in sorted(decomp["datasets"], key=lambda d: d["n_wrong_consensus"])[:3]:
        name = row["name"]
        p_arr = per_backbone_accs_meta(name)
        if p_arr is None:
            p_arr = [next(d["p"] for d in decomp["datasets"]
                          if d["name"] == name)] * 5
        C = next(d["C"] for d in decomp["datasets"] if d["name"] == name)
        a = chunked_iid_stats(p_arr, 5, C, N=10_000_000, seed=args.seed)
        b = chunked_iid_stats(p_arr, 5, C, N=50_000_000, seed=args.seed)
        stability.append({
            "dataset": name,
            "kappa_iid_n1e7": a["kappa_iid"], "kappa_iid_n5e7": b["kappa_iid"],
            "n_wrong_n1e7": a["n_wrong_consensus"], "n_wrong_n5e7": b["n_wrong_consensus"],
            "rel_delta_pct_1e7_vs_5e7":
                (b["kappa_iid"] - a["kappa_iid"]) / b["kappa_iid"] * 100.0,
        })
        log.info("stability %-22s n_wrong %d -> %d, kappa %.3f -> %.3f (%+.3f%%)",
                 name, a["n_wrong_consensus"], b["n_wrong_consensus"],
                 a["kappa_iid"], b["kappa_iid"],
                 stability[-1]["rel_delta_pct_1e7_vs_5e7"])
    rows = []
    for row in decomp["datasets"]:
        p, C = row["p"], row["C"]
        r5 = simulate_iid_kappa(p, 5, C, N=100_000, seed=args.seed)
        r7 = chunked_iid_stats(p, 5, C, N=10_000_000, seed=args.seed)
        rel = (r7["kappa_iid"] - r5["kappa_iid"]) / r7["kappa_iid"] * 100.0
        rows.append({
            "dataset": row["name"], "C": C, "p": p,
            "kappa_iid_mc_committed": row["kappa_iid_mc"],
            "kappa_iid_n1e5": r5["kappa_iid"],
            "kappa_iid_n1e7": r7["kappa_iid"],
            "n_wrong_consensus_at_1e5": r5["consensus_accuracy_iid"] is not None
            and int(round((1 - r5["consensus_accuracy_iid"]) * 100_000)),
            "n_wrong_consensus_at_1e7": r7["n_wrong_consensus"],
            "rel_delta_pct_1e5_vs_1e7": rel,
        })
        log.info("%-22s C=%3d p=%.3f n_wrong@1e5=%6d rel_delta=%+.3f%%",
                 row["name"], C, p,
                 rows[-1]["n_wrong_consensus_at_1e5"], rel)

    report = {
        "script": __file__,
        "args": {"chunk": args.chunk, "seed": args.seed},
        "stability_lowest3_1e7_vs_5e7": stability,
        "timestamp": time.strftime("%Y%m%d_%H%M%S"),
        "check": "MC convergence of kappa_iid_mc across the 19-dataset "
                 "decomposition (appendix A)",
        "rows": rows,
        "max_abs_rel_delta_pct": float(max(abs(r["rel_delta_pct_1e5_vs_1e7"])
                                          for r in rows)),
        "min_n_wrong_consensus_at_1e5": min(r["n_wrong_consensus_at_1e5"]
                                            for r in rows),
    }
    json_path = OUT_DIR / "anchoring_kappa_mc_convergence_report.json"
    txt_path = OUT_DIR / "anchoring_kappa_mc_convergence_report.txt"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=1)
    lines = [
        "MC convergence check: kappa_iid_mc at N=1e5 vs N=1e7 "
        f"(chunk={args.chunk}, seed={args.seed})",
        f"{'dataset':22s} {'C':>4s} {'p':>7s} {'n_wrong@1e5':>12s} "
        f"{'rel_delta%':>10s}",
    ]
    for r in rows:
        lines.append(f"{r['dataset']:22s} {r['C']:4d} {r['p']:7.3f} "
                     f"{r['n_wrong_consensus_at_1e5']:12d} "
                     f"{r['rel_delta_pct_1e5_vs_1e7']:+10.3f}")
    lines.append(f"max |rel_delta| = "
                 f"{report['max_abs_rel_delta_pct']:.3f}%  "
                 f"min n_wrong@1e5 = {report['min_n_wrong_consensus_at_1e5']}")
    with open(txt_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    log.info("done in %.1fs; max|rel_delta|=%.3f%% min_n_wrong@1e5=%d",
             time.time() - t0, report["max_abs_rel_delta_pct"],
             report["min_n_wrong_consensus_at_1e5"])
    print(json.dumps({
        "max_abs_rel_delta_pct": round(report["max_abs_rel_delta_pct"], 3),
        "min_n_wrong_consensus_at_1e5": report["min_n_wrong_consensus_at_1e5"],
    }))


if __name__ == "__main__":
    main()
