"""
analysis/anchoring_leave_class_out.py — Single-class driver verification
========================================================================
Addresses the Devil's Advocate's claim that removing `answer` class collapses
the low-anchor gain to +0.019. The DA derived this from per-class AUROC
differences, which is not how B2's Δ is computed (B2 uses a single AUROC on
the pooled low-anchor image set). This script recomputes correctly.

Protocol:
  1. Load all 5 CLIP backbones on TeacherBehavior.
  2. Compute consensus, correctness, entropy, agreement (same as Exp A/B2).
  3. Stratify by primary class (first GT label).
  4. Low-anchor set: {guide, answer, on-stage interaction, stand} per llm-annotation.
  5. Compute Δ(agreement - entropy) AUROC on:
     (a) All low-anchor images (should match B2's +0.127)
     (b) Low-anchor MINUS answer (the DA's claim)
  6. Bootstrap CI (B=10000, coupled).

Output:
  results/anchoring_leave_class_out_report.{json,txt}

Reproducibility:
  python analysis/anchoring_leave_class_out.py
  python analysis/anchoring_leave_class_out.py --subset bow_turnhead
"""
import argparse, json, sys, time, logging
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase0_oracle_analysis import load_clip, SUBSET_CONFIG, OUT_DIR
from pilot_perimage_anchoring import BACKBONES, softmax_entropy, auroc, bootstrap_auroc_diff

LOW_ANCHOR_CLASSES = ["guide", "answer", "on-stage interaction", "stand"]


def main():
    parser = argparse.ArgumentParser(description="Leave-class-out analysis")
    parser.add_argument("--subset", default="teacher_behavior",
                        choices=["teacher_behavior", "handrise_readwrite", "bow_turnhead"])
    parser.add_argument("--bootstrap-n", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])
    log = logging.getLogger(__name__)
    ts = time.strftime("%Y%m%d_%H%M%S")

    subset = args.subset
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
    primary = np.array([np.where(gt[i] > 0)[0][0] for i in range(N)])
    class_names = SUBSET_CONFIG[subset]["classes"]

    ent, _ = softmax_entropy(avg)
    backbone_args = np.argmax(logits_stack, axis=2)
    alpha = (backbone_args == cons[None, :]).mean(0)

    def eval_mask(mask, label):
        c = correct[mask]; e = ent[mask]; a = alpha[mask]
        if c.sum() == 0 or (~c).sum() == 0:
            return None
        ae = auroc(-e, c); aa = auroc(a, c)
        ci = bootstrap_auroc_diff(a, -e, c, n_boot, seed)
        return {"label": label, "n": int(mask.sum()), "n_correct": int(c.sum()),
                "entropy_auroc": float(ae), "agreement_auroc": float(aa),
                "delta": float(aa - ae), "delta_ci": ci}

    results = {"subset": subset, "n_images": N, "timestamp": ts,
               "low_anchor_classes": LOW_ANCHOR_CLASSES,
               "analyses": {}}

    results["analyses"]["full_tb"] = eval_mask(np.ones(N, dtype=bool), "Full TeacherBehavior")

    low_idx = np.array([i for i in range(N) if class_names[primary[i]] in LOW_ANCHOR_CLASSES])
    results["analyses"]["low_anchor_all"] = eval_mask(
        np.isin(np.arange(N), low_idx), "Low-anchor (all 4 classes)")

    answer_mask = np.array([class_names[primary[i]] == "answer" for i in range(N)])
    results["analyses"]["answer_only"] = eval_mask(answer_mask, "answer class only")

    full_no_answer = ~answer_mask
    results["analyses"]["full_tb_minus_answer"] = eval_mask(
        full_no_answer, "Full TeacherBehavior minus answer (reviewer single-class-driver check)")

    low_no_answer = np.isin(np.arange(N), low_idx) & ~answer_mask
    results["analyses"]["low_anchor_minus_answer"] = eval_mask(low_no_answer, "Low-anchor minus answer")

    # Per low-anchor class breakdown
    per_class = {}
    for cls in LOW_ANCHOR_CLASSES:
        mask = np.array([class_names[primary[i]] == cls for i in range(N)])
        per_class[cls] = eval_mask(mask, cls)
    results["analyses"]["per_low_anchor_class"] = per_class

    out_json = OUT_DIR / "anchoring_leave_class_out_report.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    log.info(f"\n{'='*70}")
    log.info("LEAVE-CLASS-OUT VERIFICATION")
    log.info(f"{'='*70}")
    for key, val in results["analyses"].items():
        if val is None or "delta" not in val:
            continue
        log.info(f"  {val['label']:35s} N={val['n']:5d}  "
                 f"ent={val['entropy_auroc']:.4f}  agr={val['agreement_auroc']:.4f}  "
                 f"Δ={val['delta']:+.4f} CI=[{val['delta_ci']['ci_lo']:+.4f},{val['delta_ci']['ci_hi']:+.4f}]")
    la = results["analyses"]["low_anchor_all"]
    lna = results["analyses"]["low_anchor_minus_answer"]
    log.info(f"\n  DA claimed low-anchor-minus-answer Δ = +0.019")
    log.info(f"  Actual: {lna['delta']:+.4f} CI=[{lna['delta_ci']['ci_lo']:+.4f},{lna['delta_ci']['ci_hi']:+.4f}]")
    log.info(f"  Verdict: {'DA WRONG (gain survives)' if lna['delta_ci']['ci_lo'] > 0 else 'DA correct'}")
    log.info(f"\nReport: {out_json}")


if __name__ == "__main__":
    main()
