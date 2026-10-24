"""
analysis/anchoring_mechanism.py — Exp B: WHY agreement wins on TB, loses on easy
=================================================================================
Formalises the Exp-A mechanism lead (gain driven by the low-anchor `answer`
class). Tests four sub-hypotheses:

  B1. Multi-label degree:  agreement's edge over entropy grows with #GT classes
                            per image (more co-occurrence -> entropy blinder).
  B2. Anchor level:         agreement wins on low-anchor (intent) primary-class
                            images, ties/loses on high-anchor. (class-level
                            anchoring -> instance-level link to siblings #1/#3.)
  B3. Complementarity:      agreement carries info entropy does not:
                            (a) Spearman corr(agreement, entropy) low;
                            (b) among the high-entropy (uncertain) half,
                                agreement still discriminates correctness;
                            (c) combined rank-sum(agreement,entropy) beats both.
  B4. Saturation:           on easy subsets entropy is near-ceiling and agreement
                            is saturated (high mean, low std) -> no room to help.
  B5. Prompt dimension:     extending the ensemble along the prompt axis
                            (CAPE A/B/C -> 15 votes) deepens saturation (consensus
                            acc rises) and entropy gains ~2x the AUROC of
                            agreement — the saturation mechanism reproduces
                            along a SECOND ensemble dimension.

All CPU, existing .npy. Output: results/anchoring_mechanism_report.{json,txt}

Reproducibility:
  python analysis/anchoring_mechanism.py
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase0_oracle_analysis import SUBSET_CONFIG, load_clip, setup_logging, OUT_DIR, DATA_DIR  # noqa: E402
from pilot_perimage_anchoring import (  # noqa: E402
    BACKBONES, softmax_entropy, auroc, _rankdata_average, bootstrap_auroc_diff,
)

# Canonical low-anchor (intent) classes on TeacherBehavior, from
# llm-annotation/step4_selective_annotation.py:113-119 (AGENTS.md rule 16).
LOW_ANCHOR_TB = {"guide", "answer", "on-stage interaction", "stand"}
HIGH_ANCHOR_TB = {"blackboard-writing", "teacher", "screen", "blackboard"}

CAPE_SUFFIX = {"A": "", "B": "_capeb", "C": "_capec"}


def load_cape_votes(subset, log):
    """Load 5 backbones x 3 CAPE sets for a subset.
    Returns dict cape->(votes (B,N) int, logits (N,C) averaged over B) and labels (N,C),
    or None if any file is missing."""
    sets = {}
    labels = None
    for cape, suffix in CAPE_SUFFIX.items():
        logits = []
        for bb in BACKBONES:
            lp = DATA_DIR / f"clip_{bb}_{subset}_logits{suffix}.npy"
            bp = DATA_DIR / f"clip_{bb}_{subset}_labels{suffix}.npy"
            if not lp.exists() or not bp.exists():
                log.warning(f"B5: CAPE-{cape} data missing for {subset}")
                return None
            logits.append(np.load(lp))
            if labels is None:
                labels = np.load(bp)
            elif not np.array_equal(labels, np.load(bp)):
                raise RuntimeError(f"label mismatch {subset} CAPE-{cape}")
        sets[cape] = {"votes": np.argmax(np.stack(logits), 2),  # (B,N)
                      "avg": np.mean(np.stack(logits), 0)}      # (N,C)
    return sets, labels


def b5_prompt_dimension(subset, bootstrap_n, seed, log):
    """B5: does the agreement-vs-entropy comparison survive extending the
    ensemble from 5 votes (CAPE-A only) to 15 votes (5 backbones x 3 CAPE
    prompt sets)?  If consensus accuracy rises (saturation deepens) and
    entropy benefits ~2x more than agreement, the saturation mechanism
    reproduces along the prompt dimension."""
    loaded = load_cape_votes(subset, log)
    if loaded is None:
        return None
    sets, labels = loaded
    n = len(labels)

    avg5 = sets["A"]["avg"]
    avg15 = np.mean([sets[c]["avg"] for c in "ABC"], 0)
    cons5 = np.argmax(avg5, 1)
    cons15 = np.argmax(avg15, 1)
    correct = labels[np.arange(n), cons15] > 0  # unified target: 15-vote consensus

    ent5, _ = softmax_entropy(avg5)
    ent15, _ = softmax_entropy(avg15)
    # quantization to the vote grid is REQUIRED: mean across 3 CAPE sets of
    # k/5-valued votes produces 1e-16 rounding noise on ~1/3 of entries; the
    # tie-aware ranker in auroc() treats that noise as distinct values, which
    # shifts AUROC by ~1.4e-3 depending on summation order (verified
    # 2026-08-10). Rounding to k/15 makes AUROC order-invariant.
    agr5 = np.round((sets["A"]["votes"] == cons5[None, :]).mean(0) * 5) / 5
    agr15 = np.round(
        np.mean([(sets[c]["votes"] == cons15[None, :]).mean(0) for c in "ABC"], 0) * 15
    ) / 15

    def dc(s, o, y):
        d = bootstrap_auroc_diff(s, o, y, bootstrap_n, seed)
        return {"mean": float(d["mean"]), "ci_lo": float(d["ci_lo"]), "ci_hi": float(d["ci_hi"])}

    return {
        "consensus_acc_5vote": float((labels[np.arange(n), cons5] > 0).mean()),
        "consensus_acc_15vote": float(correct.mean()),
        "auroc_entropy_5": auroc(-ent5, correct),
        "auroc_entropy_15": auroc(-ent15, correct),
        "auroc_agreement_5": auroc(agr5, correct),
        "auroc_agreement_15": auroc(agr15, correct),
        "delta_agreement15_vs_entropy15": dc(agr15, -ent15, correct),
        "delta_entropy_gain_15v5": dc(-ent15, -ent5, correct),
        "delta_agreement_gain_15v5": dc(agr15, agr5, correct),
    }


def spearman(a, b):
    ra = _rankdata_average(a); rb = _rankdata_average(b)
    ra = ra - ra.mean(); rb = rb - rb.mean()
    return float((ra * rb).sum() / (np.sqrt((ra * ra).sum() * (rb * rb).sum()) + 1e-12))


def load_subset(subset, log):
    allL = []; gt = None
    for bb in BACKBONES:
        clip = load_clip(bb, subset, log)
        if clip is None:
            return None
        logits, labels = clip
        if gt is None:
            gt = labels
        elif not np.array_equal(gt, labels):
            raise RuntimeError(f"label mismatch on {subset}")
        allL.append(logits)
    return np.stack(allL, 0), gt


def signals_and_correct(logits_stack, gt):
    avg = logits_stack.mean(0)
    ent, _ = softmax_entropy(avg)
    s_ent = -ent  # higher = keep
    carg = np.argmax(avg, 1)
    correct = np.array([carg[i] in np.where(gt[i] > 0)[0]
                        for i in range(len(gt))], dtype=bool)
    barg = np.argmax(logits_stack, 2)
    s_agree = (barg == carg[None, :]).mean(0)  # higher = keep
    return s_ent, s_agree, correct


def stratify_auroc(s_ent, s_agree, correct, mask_fn, groups):
    out = {}
    for label, mask in groups:
        m = mask_fn(mask)
        n = int(m.sum())
        if n < 30 or correct[m].std() == 0:
            out[label] = {"n": n, "skipped": True}
            continue
        out[label] = {
            "n": n,
            "base_acc": float(correct[m].mean()),
            "entropy_auroc": auroc(s_ent[m], correct[m]),
            "agreement_auroc": auroc(s_agree[m], correct[m]),
            "agree_minus_ent": auroc(s_agree[m], correct[m]) - auroc(s_ent[m], correct[m]),
        }
    return out


def evaluate(subset, bootstrap_n, seed, log):
    log.info(f"=== {subset} ===")
    loaded = load_subset(subset, log)
    if loaded is None:
        return None
    logits_stack, gt = loaded
    n, num_classes = gt.shape
    s_ent, s_agree, correct = signals_and_correct(logits_stack, gt)
    primary = np.array([np.where(gt[i] > 0)[0][0] for i in range(n)])
    ngt = gt.sum(1).astype(int)  # multi-label degree

    result = {"subset": subset, "n_images": int(n), "num_classes": num_classes,
              "consensus_accuracy": float(correct.mean())}

    # B1: multi-label degree (TB only)
    if SUBSET_CONFIG[subset]["multilabel"]:
        deg_groups = [(f"{k}_classes", ngt == k) for k in [1, 2, 3]]
        deg_groups.append(("4+_classes", ngt >= 4))
        result["B1_multilabel_degree"] = stratify_auroc(
            s_ent, s_agree, correct, lambda m: m, deg_groups)

        # B2: anchor level of primary class (TB only)
        classes = SUBSET_CONFIG[subset]["classes"]
        low_idx = [i for i, c in enumerate(classes) if c in LOW_ANCHOR_TB]
        high_idx = [i for i, c in enumerate(classes) if c in HIGH_ANCHOR_TB]
        result["B2_anchor_level"] = stratify_auroc(
            s_ent, s_agree, correct, lambda m: m,
            [("low_anchor(primary)", np.isin(primary, low_idx)),
             ("high_anchor(primary)", np.isin(primary, high_idx))])

    # B3: complementarity
    result["B3_complementarity"] = {
        "spearman_agree_entropy": spearman(s_agree, s_ent),
        "conditional_high_entropy_half": {
            "n": int((s_ent >= np.median(s_ent)).sum()),
            "agreement_auroc_within": auroc(s_agree[s_ent >= np.median(s_ent)],
                                            correct[s_ent >= np.median(s_ent)]),
        },
        "conditional_low_agreement_half": {
            "n": int((s_agree <= np.median(s_agree)).sum()),
            "entropy_auroc_within": auroc(s_ent[s_agree <= np.median(s_agree)],
                                          correct[s_agree <= np.median(s_agree)]),
        },
        "combined_ranksum_auroc": auroc((_rankdata_average(s_agree)
                                         + _rankdata_average(s_ent)) / 2.0, correct),
    }
    # CI on combined vs agreement (is combined > agreement? i.e. does entropy add?)
    combined = (_rankdata_average(s_agree) + _rankdata_average(s_ent)) / 2.0
    result["B3_complementarity"]["combined_minus_agreement_ci"] = bootstrap_auroc_diff(
        combined, s_agree, correct, bootstrap_n, seed)
    result["B3_complementarity"]["combined_minus_entropy_ci"] = bootstrap_auroc_diff(
        combined, s_ent, correct, bootstrap_n, seed)

    # B4: saturation
    result["B4_saturation"] = {
        "agreement": {"mean": float(s_agree.mean()), "std": float(s_agree.std()),
                       "n_distinct": int(np.unique(s_agree).size)},
        "entropy(-)": {"mean": float(s_ent.mean()), "std": float(s_ent.std())},
    }

    # B5: prompt-dimension reproduction of the saturation mechanism (CAPE A/B/C)
    result["B5_prompt_dimension"] = b5_prompt_dimension(subset, bootstrap_n, seed, log)

    log.info(f"  corr(agree,ent)={result['B3_complementarity']['spearman_agree_entropy']:+.3f}  "
             f"combined_AUROC={result['B3_complementarity']['combined_ranksum_auroc']:.4f}  "
             f"(agree={auroc(s_agree,correct):.4f}, ent={auroc(s_ent,correct):.4f})")
    return result


def format_report(results, args):
    lines = []
    lines.append("=" * 86)
    lines.append("  Exp B — Mechanism: why agreement wins on TB, loses on easy subsets")
    lines.append("=" * 86)
    lines.append(f"  Bootstrap B={args.bootstrap_n} seed={args.seed}")
    lines.append("")
    for res in results:
        lines.append(f"── {res['subset']} | N={res['n_images']} | consensus acc="
                     f"{res['consensus_accuracy']*100:.2f}% ──")
        if res.get("B1_multilabel_degree"):
            lines.append("  B1 multi-label degree (#GT classes per image):")
            lines.append(f"    {'stratum':<14} {'N':>5} {'acc':>6} {'entAUROC':>9} "
                         f"{'agrAUROC':>9} {'agr-ent':>8}")
            for lbl, d in res["B1_multilabel_degree"].items():
                if d.get("skipped"):
                    lines.append(f"    {lbl:<14} {d['n']:>5}  (skipped: n<30 or no variance)")
                    continue
                lines.append(f"    {lbl:<14} {d['n']:>5} {d['base_acc']*100:>5.2f}% "
                             f"{d['entropy_auroc']:>9.4f} {d['agreement_auroc']:>9.4f} "
                             f"{d['agree_minus_ent']:>+8.4f}")
        if res.get("B2_anchor_level"):
            lines.append("  B2 anchor level of PRIMARY class:")
            lines.append(f"    {'stratum':<24} {'N':>5} {'acc':>6} {'entAUROC':>9} "
                         f"{'agrAUROC':>9} {'agr-ent':>8}")
            for lbl, d in res["B2_anchor_level"].items():
                if d.get("skipped"):
                    lines.append(f"    {lbl:<24} {d['n']:>5}  (skipped)")
                    continue
                lines.append(f"    {lbl:<24} {d['n']:>5} {d['base_acc']*100:>5.2f}% "
                             f"{d['entropy_auroc']:>9.4f} {d['agreement_auroc']:>9.4f} "
                             f"{d['agree_minus_ent']:>+8.4f}")
        b3 = res["B3_complementarity"]
        cma = b3["combined_minus_agreement_ci"]; cme = b3["combined_minus_entropy_ci"]
        lines.append("  B3 complementarity:")
        lines.append(f"    Spearman corr(agreement, entropy) = {b3['spearman_agree_entropy']:+.3f}  "
                     f"(low => complementary)")
        ch = b3["conditional_high_entropy_half"]
        cl = b3["conditional_low_agreement_half"]
        lines.append(f"    within high-entropy half (n={ch['n']}): agreement AUROC = "
                     f"{ch['agreement_auroc_within']:.4f}  (does agreement still split where entropy gives up?)")
        lines.append(f"    within low-agreement half (n={cl['n']}): entropy AUROC = "
                     f"{cl['entropy_auroc_within']:.4f}")
        lines.append(f"    combined rank-sum AUROC = {b3['combined_ranksum_auroc']:.4f}")
        lines.append(f"      combined - agreement = {cma['mean']:+.4f} CI[{cma['ci_lo']:+.4f},{cma['ci_hi']:+.4f}]")
        lines.append(f"      combined - entropy   = {cme['mean']:+.4f} CI[{cme['ci_lo']:+.4f},{cme['ci_hi']:+.4f}]")
        b4 = res["B4_saturation"]
        lines.append("  B4 saturation diagnostic:")
        lines.append(f"    agreement: mean={b4['agreement']['mean']:.3f} std={b4['agreement']['std']:.3f} "
                     f"distinct_vals={b4['agreement']['n_distinct']}")
        lines.append(f"    entropy(-): mean={b4['entropy(-)']['mean']:.3f} std={b4['entropy(-)']['std']:.3f}")
        b5 = res.get("B5_prompt_dimension")
        if b5:
            d1 = b5["delta_agreement15_vs_entropy15"]
            d2 = b5["delta_entropy_gain_15v5"]
            d3 = b5["delta_agreement_gain_15v5"]
            lines.append("  B5 prompt-dimension saturation (5bb x CAPE-A/B/C = 15 votes):")
            lines.append(f"    consensus acc: 5-vote {b5['consensus_acc_5vote']*100:.1f}%  ->  "
                         f"15-vote {b5['consensus_acc_15vote']*100:.1f}%")
            lines.append(f"    AUROC  entropy: 5v={b5['auroc_entropy_5']:.4f} 15v={b5['auroc_entropy_15']:.4f} | "
                         f"agreement: 5v={b5['auroc_agreement_5']:.4f} 15v={b5['auroc_agreement_15']:.4f}")
            lines.append(f"    entropy gain 15v-5v  = {d2['mean']:+.4f} CI[{d2['ci_lo']:+.4f},{d2['ci_hi']:+.4f}]")
            lines.append(f"    agreement gain 15v-5v = {d3['mean']:+.4f} CI[{d3['ci_lo']:+.4f},{d3['ci_hi']:+.4f}]")
            lines.append(f"    agreement15 - entropy15 = {d1['mean']:+.4f} CI[{d1['ci_lo']:+.4f},{d1['ci_hi']:+.4f}]")
        lines.append("")
    lines.append("=" * 86)
    lines.append("  Reading: B1/B2 +agr-ent => agreement beats entropy in that stratum.")
    lines.append("           B3 combined>both => signals complementary. B4 high agreement mean+low std => saturated.")
    lines.append("           B5: expanding the ensemble along the prompt dimension (CAPE A/B/C) raises consensus")
    lines.append("           accuracy (deeper saturation) and entropy then benefits ~2x more than agreement.")
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subsets", nargs="+",
                        default=["teacher_behavior", "handrise_readwrite", "bow_turnhead"])
    parser.add_argument("--bootstrap_n", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ts = time.strftime("%Y%m%d_%H%M%S")
    setup_logging(ts, name="anchoring_mechanism")
    log = logging.getLogger()
    log.info(f"Exp B anchoring_mechanism | args={vars(args)}")

    results = []
    for subset in args.subsets:
        r = evaluate(subset, args.bootstrap_n, args.seed, log)
        if r is not None:
            results.append(r)
    if not results:
        log.error("No results."); return 1

    txt = format_report(results, args)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "anchoring_mechanism_report.json", "w") as f:
        json.dump({"args": vars(args), "timestamp": ts, "results": results}, f, indent=2)
    with open(OUT_DIR / "anchoring_mechanism_report.txt", "w") as f:
        f.write(txt)
    print(); print(txt)
    return 0


if __name__ == "__main__":
    exit(main())
