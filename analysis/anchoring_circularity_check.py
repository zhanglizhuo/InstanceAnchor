"""
analysis/anchoring_circularity_check.py — Circularity control for Exp A
=======================================================================
Addresses the reviewers' circularity concern: in the original setup,
`consensus = argmax(mean(all 5))` appears in BOTH:
  - correct(x) = 1[consensus in GT]
  - alpha(x)   = (1/5) * sum_b 1[argmax(l_b) == consensus]
This mechanical coupling could inflate alpha's AUROC.

Three controls, each progressively breaking the coupling:

  C1. LEAVE-ONE-OUT (LOO)
      For each backbone b, consensus_loo_b = argmax(mean(other 4)).
      alpha_loo_b = 1[argmax(l_b) == consensus_loo_b].
      alpha_loo = (1/5) * sum_b alpha_loo_b.
      Breaks the per-backbone coupling (b never votes on its own consensus).

  C2. SPLIT-HALF (zero circularity)
      Split 5 backbones into group_A (3) and group_B (2).
      consensus_A = argmax(mean(group_A)); correct_A = 1[consensus_A in GT].
      alpha_B = (1/2) * sum_{b in group_B} 1[argmax(l_b) == consensus_A].
      entropy_A = entropy of mean(group_A).
      Group_B agreement predicts group_A correctness -- zero shared computation.
      We rotate through all C(5,3)=10 splits and average.

  C3. CROSS-FAMILY (independent model)
      consensus_clip = argmax(mean(5 CLIPs)); correct_clip = standard.
      alpha_vB = 1[vit_b_16 argmax == consensus_clip]  (single non-CLIP model).
      If alpha's AUROC were purely mechanical, alpha_vB should also predict
      correct_clip well. (Phase 3 already showed it does NOT: AUROC ~0.47.)

GATE: If alpha_loo retains AUROC >= 0.68 on TeacherBehavior (within ~0.03
of the original 0.706), the circularity does not materially inflate the
headline. If it drops below 0.65, the gain is substantially circular.

Output:
  results/anchoring_circularity_report.{json,txt}
  logs/anchoring_circularity_<timestamp>.log

Reproducibility:
  python analysis/anchoring_circularity_check.py
"""
import argparse
import json
import logging
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase0_oracle_analysis import SUBSET_CONFIG, load_clip, setup_logging, OUT_DIR  # noqa: E402
from pilot_perimage_anchoring import (  # noqa: E402
    BACKBONES, TEMPERATURE, softmax_entropy, auroc, bootstrap_auroc_diff,
)

REPO = Path(__file__).resolve().parent.parent


def load_crossfamily_logits(subset, log):
    """Load vit_b_16 cross-family logits if available."""
    p = OUT_DIR / "per_image" / f"crossfamily_vit_b_16_{subset}_logits.npy"
    if not p.exists():
        log.warning(f"Cross-family logits not found: {p}")
        return None
    return np.load(p)


def compute_alpha_original(logits_stack):
    """Original alpha: fraction agreeing with 5-backbone consensus."""
    consensus = np.argmax(logits_stack.mean(axis=0), axis=1)
    backbone_argmaxes = np.argmax(logits_stack, axis=2)
    agree = (backbone_argmaxes == consensus[None, :]).astype(np.float32)
    return agree.mean(axis=0), consensus


def compute_alpha_loo(logits_stack):
    """Leave-one-out alpha: backbone b agrees with consensus of OTHER 4."""
    K, N, C = logits_stack.shape
    alpha_loo = np.zeros(N, dtype=np.float32)
    for b in range(K):
        mask = np.ones(K, dtype=bool)
        mask[b] = False
        consensus_loo_b = np.argmax(logits_stack[mask].mean(axis=0), axis=1)
        backbone_argmax_b = np.argmax(logits_stack[b], axis=1)
        alpha_loo += (backbone_argmax_b == consensus_loo_b).astype(np.float32)
    return alpha_loo / K


def compute_split_half(logits_stack, gt):
    """Split-half: group_A consensus, group_B agreement. Average over splits."""
    K, N, C = logits_stack.shape
    idx = list(range(K))
    auroc_agree_list = []
    auroc_ent_list = []
    for group_a_size in (3,):
        for group_a in combinations(idx, group_a_size):
            group_b = [i for i in idx if i not in group_a]
            logits_a = logits_stack[list(group_a)].mean(axis=0)
            consensus_a = np.argmax(logits_a, axis=1)
            correct_a = np.array(
                [consensus_a[i] in np.where(gt[i] > 0)[0] for i in range(N)],
                dtype=bool,
            )
            if correct_a.sum() == 0 or (~correct_a).sum() == 0:
                continue
            alpha_b = np.zeros(N, dtype=np.float32)
            for b in group_b:
                backbone_argmax_b = np.argmax(logits_stack[b], axis=1)
                alpha_b += (backbone_argmax_b == consensus_a).astype(np.float32)
            alpha_b /= len(group_b)
            ent_a, _ = softmax_entropy(logits_a)
            auroc_agree_list.append(auroc(alpha_b, correct_a))
            auroc_ent_list.append(auroc(-ent_a, correct_a))
    return {
        "n_splits": len(auroc_agree_list),
        "agreement_auroc_mean": float(np.mean(auroc_agree_list)),
        "agreement_auroc_std": float(np.std(auroc_agree_list)),
        "entropy_auroc_mean": float(np.mean(auroc_ent_list)),
        "entropy_auroc_std": float(np.std(auroc_ent_list)),
        "delta_mean": float(np.mean(np.array(auroc_agree_list) - np.array(auroc_ent_list))),
    }


def evaluate(subset, bootstrap_n, seed, log):
    log.info(f"=== {subset} ===")
    all_logits = []
    gt = None
    for bb in BACKBONES:
        clip = load_clip(bb, subset, log)
        if clip is None:
            return None
        logits, labels = clip
        if gt is None:
            gt = labels
        elif not np.array_equal(gt, labels):
            raise RuntimeError(f"label mismatch on {subset}")
        all_logits.append(logits)
    logits_stack = np.stack(all_logits, axis=0)
    N = logits_stack.shape[1]
    avg_logits = logits_stack.mean(axis=0)
    consensus_5 = np.argmax(avg_logits, axis=1)
    correct = np.array(
        [consensus_5[i] in np.where(gt[i] > 0)[0] for i in range(N)], dtype=bool,
    )

    ent_avg, _ = softmax_entropy(avg_logits)

    alpha_orig, _ = compute_alpha_original(logits_stack)
    alpha_loo = compute_alpha_loo(logits_stack)

    auroc_ent = auroc(-ent_avg, correct)
    auroc_orig = auroc(alpha_orig, correct)
    auroc_loo = auroc(alpha_loo, correct)

    delta_orig_vs_ent = bootstrap_auroc_diff(alpha_orig, -ent_avg, correct, bootstrap_n, seed)
    delta_loo_vs_ent = bootstrap_auroc_diff(alpha_loo, -ent_avg, correct, bootstrap_n, seed)
    delta_loo_vs_orig = bootstrap_auroc_diff(alpha_loo, alpha_orig, correct, bootstrap_n, seed)

    split_half = compute_split_half(logits_stack, gt)

    vB_logits = load_crossfamily_logits(subset, log)
    c3 = None
    if vB_logits is not None:
        vB_argmax = np.argmax(vB_logits, axis=1)
        alpha_vB = (vB_argmax == consensus_5).astype(np.float32)
        c3 = {
            "alpha_vB_auroc": float(auroc(alpha_vB, correct)),
            "note": "single non-CLIP model agreeing with CLIP consensus",
        }

    result = {
        "subset": subset,
        "n_images": N,
        "n_correct": int(correct.sum()),
        "consensus_accuracy": float(correct.mean()),
        "C0_original": {
            "entropy_auroc": float(auroc_ent),
            "agreement_auroc": float(auroc_orig),
            "delta_agree_minus_ent": delta_orig_vs_ent,
        },
        "C1_leave_one_out": {
            "agreement_loo_auroc": float(auroc_loo),
            "delta_loo_minus_ent": delta_loo_vs_ent,
            "delta_loo_minus_orig": delta_loo_vs_orig,
        },
        "C2_split_half": split_half,
        "C3_cross_family": c3,
        "gate": {
            "original_agreement_auroc": float(auroc_orig),
            "loo_agreement_auroc": float(auroc_loo),
            "drop": float(auroc_orig - auroc_loo),
            "passes": bool(auroc_loo >= auroc_orig - 0.03),
        },
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subsets", nargs="+",
                        default=["teacher_behavior", "handrise_readwrite", "bow_turnhead"])
    parser.add_argument("--bootstrap_n", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ts = time.strftime("%Y%m%d_%H%M%S")
    from pathlib import Path as _P
    log_dir = _P("logs"); log_dir.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_dir / f"anchoring_circularity_{ts}.log", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )
    log = logging.getLogger(__name__)
    log.info(f"Args: {vars(args)}")

    results = []
    for subset in args.subsets:
        r = evaluate(subset, args.bootstrap_n, args.seed, log)
        if r is not None:
            results.append(r)
            log.info(f"  entropy={r['C0_original']['entropy_auroc']:.4f}  "
                     f"agree_orig={r['C0_original']['agreement_auroc']:.4f}  "
                     f"agree_loo={r['C1_leave_one_out']['agreement_loo_auroc']:.4f}  "
                     f"split_half_agree={r['C2_split_half']['agreement_auroc_mean']:.4f}")

    report = {
        "args": vars(args),
        "timestamp": ts,
        "description": "Circularity control: LOO, split-half, cross-family",
        "results": results,
    }
    out_json = OUT_DIR / "anchoring_circularity_report.json"
    out_txt = OUT_DIR / "anchoring_circularity_report.txt"
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)

    log.info(f"\n{'='*80}")
    log.info("CIRCULARITY CHECK SUMMARY")
    log.info(f"{'='*80}")
    for r in results:
        s = r["subset"]
        c0 = r["C0_original"]
        c1 = r["C1_leave_one_out"]
        c2 = r["C2_split_half"]
        log.info(f"\n  {s} (N={r['n_images']})")
        log.info(f"    Original:    ent={c0['entropy_auroc']:.4f}  agr={c0['agreement_auroc']:.4f}  "
                 f"Δ={c0['delta_agree_minus_ent']['mean']:+.4f}")
        log.info(f"    LOO:         agr_loo={c1['agreement_loo_auroc']:.4f}  "
                 f"Δ_vs_ent={c1['delta_loo_minus_ent']['mean']:+.4f}  "
                 f"Δ_vs_orig={c1['delta_loo_minus_orig']['mean']:+.4f}")
        log.info(f"    Split-half:  agr={c2['agreement_auroc_mean']:.4f}±{c2['agreement_auroc_std']:.4f}  "
                 f"ent={c2['entropy_auroc_mean']:.4f}  Δ={c2['delta_mean']:+.4f}  ({c2['n_splits']} splits)")
        if r.get("C3_cross_family"):
            log.info(f"    Cross-fam:   alpha_vB={r['C3_cross_family']['alpha_vB_auroc']:.4f}")
        g = r["gate"]
        log.info(f"    GATE: orig={g['original_agreement_auroc']:.4f} → loo={g['loo_agreement_auroc']:.4f}  "
                 f"drop={g['drop']:+.4f}  {'PASS' if g['passes'] else 'FAIL'}")

    log.info(f"\nReport: {out_json}")
    return report


if __name__ == "__main__":
    main()
