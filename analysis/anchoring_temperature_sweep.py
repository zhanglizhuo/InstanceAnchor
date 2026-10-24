"""
analysis/anchoring_temperature_sweep.py — Temperature robustness check
=====================================================================
Addresses Domain reviewer concern: agreement is T-invariant (argmax doesn't
change with temperature), but entropy depends on T. The +0.067 AUROC gap
could be partly a temperature artifact if T=100 disadvantages entropy.

Protocol:
  - Load existing per-image logits (no new GPU needed).
  - Sweep T in {1, 10, 25, 50, 100, 200, 500, 1000}.
  - For each T, recompute entropy = softmax(logits * T), then AUROC.
  - Agreement is T-invariant (computed once).
  - Plot Δ(agreement - entropy) as a function of T.

GATE: if Δ remains positive across all T values, the gap is not a T artifact.

Output:
  results/anchoring_temperature_sweep_report.{json,txt}

Reproducibility:
  python analysis/anchoring_temperature_sweep.py
  python analysis/anchoring_temperature_sweep.py --subset handrise_readwrite
"""
import argparse, json, sys, time, logging
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase0_oracle_analysis import load_clip, OUT_DIR
from pilot_perimage_anchoring import BACKBONES, auroc, bootstrap_auroc_diff

TEMPS = [1, 10, 25, 50, 100, 200, 500, 1000]


def softmax_entropy_at_T(logits, T):
    z = logits * T
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    p = e / e.sum(axis=1, keepdims=True)
    return -np.sum(p * np.log(p + 1e-12), axis=1)


def main():
    parser = argparse.ArgumentParser(description="Temperature sweep analysis")
    parser.add_argument("--subset", default="teacher_behavior",
                        choices=["teacher_behavior", "handrise_readwrite", "bow_turnhead"])
    parser.add_argument("--temperatures", nargs="+", type=int,
                        default=[1, 10, 25, 50, 100, 200, 500, 1000])
    parser.add_argument("--bootstrap-n", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])
    log = logging.getLogger(__name__)
    ts = time.strftime("%Y%m%d_%H%M%S")

    subset = args.subset
    temps = args.temperatures
    n_boot = args.bootstrap_n
    seed = args.seed
    all_logits = []; gt = None
    for bb in BACKBONES:
        clip = load_clip(bb, subset, log)
        logits, labels = clip
        if gt is None: gt = labels
        all_logits.append(logits)
    logits_stack = np.stack(all_logits, axis=0)
    avg = logits_stack.mean(0)
    cons = np.argmax(avg, 1)
    N = len(gt)
    correct = np.array([cons[i] in np.where(gt[i] > 0)[0] for i in range(N)], dtype=bool)

    backbone_args = np.argmax(logits_stack, axis=2)
    alpha = (backbone_args == cons[None, :]).mean(0)
    auroc_alpha = auroc(alpha, correct)

    results = {"subset": subset, "n_images": N, "agreement_auroc": float(auroc_alpha),
               "temperature_sweep": [], "timestamp": ts}

    log.info(f"Agreement AUROC (T-invariant): {auroc_alpha:.4f}")
    log.info(f"{'T':>6s}  {'ent_AUROC':>10s}  {'Δ(agr-ent)':>12s}  {'CI':>20s}")
    for T in temps:
        ent = softmax_entropy_at_T(avg, T)
        auroc_ent = auroc(-ent, correct)
        ci = bootstrap_auroc_diff(alpha, -ent, correct, n_boot, seed)
        delta = auroc_alpha - auroc_ent
        entry = {"T": T, "entropy_auroc": float(auroc_ent),
                 "delta": float(delta), "delta_ci": ci}
        results["temperature_sweep"].append(entry)
        log.info(f"{T:>6d}  {auroc_ent:>10.4f}  {delta:>+12.4f}  [{ci['ci_lo']:>+.4f}, {ci['ci_hi']:>+.4f}]")

    deltas = [e["delta"] for e in results["temperature_sweep"]]
    results["gate"] = {
        "min_delta": float(min(deltas)),
        "max_delta": float(max(deltas)),
        "all_positive": bool(min(deltas) > 0),
        "best_T": temps[int(np.argmax(deltas))],
        "worst_T": temps[int(np.argmin(deltas))],
    }
    g = results["gate"]
    log.info(f"\nGate: all Δ positive = {g['all_positive']}  "
             f"range [{g['min_delta']:+.4f}, {g['max_delta']:+.4f}]  "
             f"best T={g['best_T']}  worst T={g['worst_T']}")

    out_json = OUT_DIR / "anchoring_temperature_sweep_report.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"Report: {out_json}")


if __name__ == "__main__":
    main()
