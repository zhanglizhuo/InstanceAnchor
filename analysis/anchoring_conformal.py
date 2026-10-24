"""
analysis/anchoring_conformal.py — Conformal prediction comparison
==================================================================
Compares agreement vs entropy as nonconformity scores in a minimal
conformal prediction framework. For each target coverage (50%, 70%, 90%),
uses a 50/50 cal/test split to calibrate the threshold from the calibration
set and evaluate coverage + accuracy on the test set.

Output: results/anchoring_conformal_report.json
"""
import argparse, json, sys, logging
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase0_oracle_analysis import load_clip, SUBSET_CONFIG, OUT_DIR, setup_logging
from pilot_perimage_anchoring import BACKBONES, softmax_entropy, auroc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", default="teacher_behavior")
    parser.add_argument("--n-splits", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ts = __import__('time').strftime("%Y%m%d_%H%M%S")
    setup_logging(ts, name="anchoring_conformal")
    log = logging.getLogger()

    # Load data
    all_logits = []; gt = None
    for bb in BACKBONES:
        clip = load_clip(bb, args.subset, log)
        if clip is None: return 1
        logits, labels = clip
        if gt is None: gt = labels
        all_logits.append(logits)
    stack = np.stack(all_logits, 0); avg = stack.mean(0); cons = np.argmax(avg, 1)
    N = len(gt)
    correct = np.array([cons[i] in np.where(gt[i] > 0)[0] for i in range(N)], dtype=bool)

    ent, _ = softmax_entropy(avg)
    s_ent = -ent
    agr = (np.argmax(stack, 2) == cons[None, :]).mean(0)

    rng = np.random.RandomState(args.seed)
    targets = [0.50, 0.70, 0.90]
    half = N // 2

    results = {"subset": args.subset, "n_images": N, "n_splits": args.n_splits,
               "consensus_accuracy": float(correct.mean()),
               "agreement_auroc": float(auroc(agr, correct)),
               "entropy_auroc": float(auroc(s_ent, correct)),
               "conformal": {f"{int(t*100)}pct": {} for t in targets}}

    for t in targets:
        t_str = str(int(t * 100)) + "pct"
        r_agr_eff, r_agr_cov, r_agr_acc = [], [], []
        r_ent_eff, r_ent_cov, r_ent_acc = [], [], []

        for split in range(args.n_splits):
            idx = rng.permutation(N)
            cal_idx, test_idx = idx[:half], idx[half:]

            for signal, eff_list, cov_list, acc_list in [
                (agr, r_agr_eff, r_agr_cov, r_agr_acc),
                (s_ent, r_ent_eff, r_ent_cov, r_ent_acc),
            ]:
                cal_scores = signal[cal_idx]
                threshold = np.percentile(-cal_scores, 100 * (1 - t))
                kept = signal[test_idx] >= -threshold
                coverage = float(kept.mean())
                accuracy = float(correct[test_idx][kept].mean()) if kept.sum() > 0 else 0.0
                cov_list.append(coverage)
                acc_list.append(accuracy)

        results["conformal"][t_str]["agreement"] = {
            "coverage_mean": float(np.mean(r_agr_cov)),
            "coverage_std": float(np.std(r_agr_cov)),
            "accuracy_mean": float(np.mean(r_agr_acc)),
            "accuracy_std": float(np.std(r_agr_acc)),
        }
        results["conformal"][t_str]["entropy"] = {
            "coverage_mean": float(np.mean(r_ent_cov)),
            "coverage_std": float(np.std(r_ent_cov)),
            "accuracy_mean": float(np.mean(r_ent_acc)),
            "accuracy_std": float(np.std(r_ent_acc)),
        }

        log.info(f"  @{t*100:.0f}%: agr acc={np.mean(r_agr_acc)*100:.2f}%±{np.std(r_agr_acc)*100:.1f} "
                 f"ent acc={np.mean(r_ent_acc)*100:.2f}%±{np.std(r_ent_acc)*100:.1f}")

    out = OUT_DIR / "anchoring_conformal_report.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"Report: {out}")
    return 0

if __name__ == "__main__":
    exit(main())
