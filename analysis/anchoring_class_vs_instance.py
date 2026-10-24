"""
analysis/anchoring_class_vs_instance.py — Exp C: class- vs instance-level anchoring
=================================================================================
Differentiates our INSTANCE-level anchoring (cross-backbone agreement) from
VisualAnchor's CLASS-level AnchorScore (sibling #3, per-class CLIP zero-shot
acc, LAION-2B). Three questions:

  C1. Which predicts per-image CLIP correctness best?
        - instance agreement (ours, per-image)
        - class-mean agreement (OURS but collapsed to class granularity)
        - VisualAnchor AnchorScore[primary class] (the published sibling signal)
  C2. Does instance agreement discriminate WITHIN classes? (per-class agreement
        AUROC, aggregated.) If yes -> it carries per-image info beyond any class
        prior; this is the crisp "instance != class" evidence.
C3. Complementarity: does combining the class-level prior (AnchorScore) with
    instance agreement beat instance agreement alone?
    (If yes, the two granularities carry complementary information.)

Output: results/anchoring_class_vs_instance_report.{json,txt}

Reproducibility:
  python analysis/anchoring_class_vs_instance.py
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase0_oracle_analysis import SUBSET_CONFIG, load_clip, setup_logging, OUT_DIR, REPO_ROOT  # noqa: E402
from pilot_perimage_anchoring import (  # noqa: E402
    BACKBONES, softmax_entropy, auroc, _rankdata_average, bootstrap_auroc_diff,
)

import os
# Exp C baseline lock (2026-08-10): the canonical VA AnchorScore snapshot
# lives in THIS repo (results/sibling_snapshots/...). The script prefers it
# so IA numbers never drift when the VA repo changes. Fall back to the live
# VA repo only if the snapshot is missing, and warn loudly.
VA_SNAPSHOT_PATH = (REPO_ROOT / "results" / "sibling_snapshots"
                    / "va_anchorscore_teacherbehavior_snapshot.json")
VA_ANCHORSCORE_PATH = Path(
    os.environ.get("VA_ANCHORSCORE_PATH",
                   str(REPO_ROOT.parent / "VisualAnchor" / "results" / "01_core"
                       / "anchor_score_scb5" / "anchor_scores.json"))
)


def load_va_anchorscore_tb(log):
    """Load per-class AnchorScore for TeacherBehavior; normalise class names
    to the lowercase canonical form (AGENTS.md rule 15). Reads the locked
    snapshot first (Exp C baseline lock 2026-08-10), falls back to the live
    VA repo copy otherwise."""
    src = None
    if VA_SNAPSHOT_PATH.exists():
        src = VA_SNAPSHOT_PATH
    elif VA_ANCHORSCORE_PATH.exists():
        log.warning(f"VA snapshot missing ({VA_SNAPSHOT_PATH}); "
                    f"falling back to live VA repo ({VA_ANCHORSCORE_PATH})")
        src = VA_ANCHORSCORE_PATH
    else:
        log.warning(f"va anchor_scores not found: {VA_SNAPSHOT_PATH} / {VA_ANCHORSCORE_PATH}")
        return None
    d = json.load(open(src))
    tb = d.get("TeacherBehavior", {})
    raw = tb.get("per_class_acc", {})
    out = {}
    for key, v in raw.items():
        norm = key.strip().lower()
        out[norm] = float(v["acc"]) * 0.01  # acc is in %, store as fraction
    return out  # {lowercase class: acc fraction}


def _tie_aware_selacc(signal, correct, coverage, n_seeds, rng):
    """Selective accuracy at `coverage`, breaking ties uniformly at random
    (averaged over n_seeds). The instance agreement has only ~6 distinct
    values -> huge ties -> naive argsort injects arbitrary index-order noise."""
    n = len(signal); m = max(1, int(round(coverage * n)))
    accs = np.empty(n_seeds)
    for s in range(n_seeds):
        jitter = rng.random(n)
        order = np.lexsort((jitter, -signal))  # primary -signal, secondary random
        accs[s] = correct[order[:m]].mean()
    return float(accs.mean())


def _heldout_class_mean(primary, s_inst, n_folds, seed):
    """Out-of-sample class-mean: for each fold, compute the per-class mean of
    s_inst on the OTHER folds and assign to the held-out fold. Removes the
    in-sample optimistic bias of computing the class mean on the eval set itself."""
    rng = np.random.RandomState(seed)
    n = len(primary)
    fold = rng.randint(0, n_folds, size=n)
    out = np.zeros(n, dtype=np.float32)
    num_classes = int(primary.max()) + 1
    for f in range(n_folds):
        train_mask = fold != f
        for c in range(num_classes):
            mtrain = train_mask & (primary == c)
            if mtrain.any():
                mean_c = float(s_inst[mtrain].mean())
                out[(fold == f) & (primary == c)] = mean_c
    return out


def _within_class_selacc_lift(primary, s_inst, correct, coverage, n_seeds, rng):
    """Within each class, keep top-cov by instance agreement (tie-aware); report
    the accuracy lift over the class base accuracy, sample-size weighted. This is
    the cleanest test of whether instance agreement carries per-image signal
    uncontaminated by between-class variance."""
    lifts = []; weights = []
    for c in np.unique(primary):
        m = primary == c
        if m.sum() < 30 or correct[m].std() == 0:
            continue
        base = float(correct[m].mean())
        sa = _tie_aware_selacc(s_inst[m], correct[m], coverage, n_seeds, rng)
        lifts.append(sa - base); weights.append(int(m.sum()))
    w = np.array(weights, dtype=float); l = np.array(lifts)
    return float((l * w).sum() / w.sum())


def evaluate(subset, bootstrap_n, seed, log):
    log.info(f"=== {subset} ===")
    if not SUBSET_CONFIG[subset]["multilabel"]:
        log.info("  skip (Exp C is TeacherBehavior-only: class-level prior needs >=4 classes)")
        return None
    allL = []; gt = None
    for bb in BACKBONES:
        clip = load_clip(bb, subset, log)
        if clip is None:
            return None
        logits, labels = clip
        if gt is None:
            gt = labels
        elif not np.array_equal(gt, labels):
            raise RuntimeError("label mismatch")
        allL.append(logits)
    logits_stack = np.stack(allL, 0)
    n, num_classes = gt.shape
    classes = SUBSET_CONFIG[subset]["classes"]
    avg = logits_stack.mean(0)
    carg = np.argmax(avg, 1)
    correct = np.array([carg[i] in np.where(gt[i] > 0)[0] for i in range(n)], dtype=bool)
    primary = np.array([np.where(gt[i] > 0)[0][0] for i in range(n)])
    barg = np.argmax(logits_stack, 2)
    s_inst = (barg == carg[None, :]).mean(0)                       # instance agreement
    # class-mean agreement (our signal collapsed to class granularity)
    cls_mean_agree = np.zeros(num_classes)
    for c in range(num_classes):
        m = primary == c
        cls_mean_agree[c] = float(s_inst[m].mean()) if m.any() else 0.0
    s_clsmean = cls_mean_agree[primary]
    # VisualAnchor AnchorScore prior
    va = load_va_anchorscore_tb(log)
    if va is not None:
        s_va = np.array([va.get(classes[c], float("nan")) for c in primary])
        va_ok = ~np.isnan(s_va)
    else:
        s_va = np.full(n, np.nan); va_ok = np.zeros(n, dtype=bool)

    result = {"subset": subset, "n_images": int(n), "num_classes": num_classes,
              "consensus_accuracy": float(correct.mean()),
              "va_anchorscore_per_class": {classes[c]: va.get(classes[c])
                                           for c in range(num_classes) if va}}

    # C1: AUROC of the three signals
    c1 = {
        "instance_agreement": auroc(s_inst, correct),
        "class_mean_agreement": auroc(s_clsmean, correct),
    }
    if va_ok.all():
        c1["va_anchorscore_primary"] = auroc(s_va, correct)
    elif va_ok.any():
        c1["va_anchorscore_primary"] = auroc(s_va[va_ok], correct[va_ok])
        c1["va_anchorscore_n"] = int(va_ok.sum())
    result["C1_auroc"] = c1
    c1["instance_minus_classmean_ci"] = bootstrap_auroc_diff(
        s_inst, s_clsmean, correct, bootstrap_n, seed)
    if va_ok.all():
        c1["instance_minus_va_ci"] = bootstrap_auroc_diff(
            s_inst, s_va, correct, bootstrap_n, seed)

    # C2: within-class agreement discrimination
    per_class = {}
    weighted, wsum = 0.0, 0
    for c in range(num_classes):
        m = primary == c
        if m.sum() < 30 or correct[m].std() == 0:
            continue
        a = auroc(s_inst[m], correct[m])
        per_class[classes[c]] = {"n": int(m.sum()), "acc": float(correct[m].mean()),
                                  "within_class_agreement_auroc": a}
        weighted += a * int(m.sum()); wsum += int(m.sum())
    result["C2_within_class"] = {
        "per_class": per_class,
        "weighted_mean_within_class_auroc": weighted / wsum if wsum else None,
    }

    # C3: complementarity — instance + class prior vs instance alone
    c3 = {}
    if va_ok.all():
        combined_va = (_rankdata_average(s_inst) + _rankdata_average(s_va)) / 2.0
        c3["instance_plus_va_anchorscore_auroc"] = auroc(combined_va, correct)
        c3["instance_plus_va_minus_instance_ci"] = bootstrap_auroc_diff(
            combined_va, s_inst, correct, bootstrap_n, seed)
    combined_cm = (_rankdata_average(s_inst) + _rankdata_average(s_clsmean)) / 2.0
    c3["instance_plus_classmean_auroc"] = auroc(combined_cm, correct)
    c3["instance_plus_classmean_minus_instance_ci"] = bootstrap_auroc_diff(
        combined_cm, s_inst, correct, bootstrap_n, seed)
    result["C3_combined"] = c3

    # C4: FAIR selective-accuracy comparison (tie-aware + held-out class-mean)
    rng = np.random.RandomState(seed)
    s_cm_heldout = _heldout_class_mean(primary, s_inst, n_folds=5, seed=seed)
    covs = [0.3, 0.5, 0.7, 0.9]
    c4 = {"coverages": covs, "n_tie_seeds": 200,
          "selective_accuracy": {
              cov: {
                  "instance": _tie_aware_selacc(s_inst, correct, cov, 200, rng),
                  "classmean_heldout": _tie_aware_selacc(s_cm_heldout, correct, cov, 200, rng),
                  **({"va_anchorscore": _tie_aware_selacc(s_va, correct, cov, 200, rng)}
                     if va_ok.all() else {}),
              } for cov in covs},
          "within_class_instance_lift": {
              cov: _within_class_selacc_lift(primary, s_inst, correct, cov, 200, rng)
              for cov in covs}}
    result["C4_fair_selective"] = c4
    log.info("  C4 fair selective acc (tie-aware, held-out classmean):")
    for cov in covs:
        r = c4["selective_accuracy"][cov]
        log.info(f"    cov={int(cov*100)}%: instance={r['instance']*100:.2f}% "
                 f"classmean_ho={r['classmean_heldout']*100:.2f}%"
                 + (f" VA={r.get('va_anchorscore',float('nan'))*100:.2f}%" if 'va_anchorscore' in r else ""))
    log.info(f"  C1 AUROC: inst={c1['instance_agreement']:.4f} "
             f"clsmean={c1['class_mean_agreement']:.4f} "
             + (f"VA={c1.get('va_anchorscore_primary',float('nan')):.4f}" if va is not None else ""))
    log.info(f"  C2 weighted within-class agreement AUROC = "
             f"{result['C2_within_class']['weighted_mean_within_class_auroc']:.4f}")
    return result


def format_report(results, args):
    lines = []
    lines.append("=" * 86)
    lines.append("  Exp C — class-level vs instance-level anchoring (differentiate from VisualAnchor)")
    lines.append("=" * 86)
    lines.append(f"  Bootstrap B={args.bootstrap_n} seed={args.seed}")
    lines.append("")
    for res in results:
        lines.append(f"── {res['subset']} | N={res['n_images']} | consensus acc="
                     f"{res['consensus_accuracy']*100:.2f}% ──")
        if res.get("va_anchorscore_per_class"):
            lines.append("  VisualAnchor AnchorScore (LAION-2B per-class acc):")
            lines.append("    " + "  ".join(f"{k}={v*100:.1f}%"
                                            for k, v in res["va_anchorscore_per_class"].items()))
        c1 = res["C1_auroc"]
        lines.append("  C1 AUROC for predicting per-image correctness:")
        lines.append(f"    instance agreement        : {c1['instance_agreement']:.4f}   (ours)")
        lines.append(f"    class-mean agreement      : {c1['class_mean_agreement']:.4f}   (ours, class granularity)")
        if "va_anchorscore_primary" in c1:
            lines.append(f"    VA AnchorScore[primary]   : {c1['va_anchorscore_primary']:.4f}   (sibling #3, class-level) "
                         + (f"(n={c1.get('va_anchorscore_n')})" if "va_anchorscore_n" in c1 else ""))
        c2 = res["C2_within_class"]
        lines.append(f"  C2 within-class agreement AUROC (weighted mean) = "
                     f"{c2['weighted_mean_within_class_auroc']:.4f}  (>0.5 => instance adds info beyond class prior)")
        lines.append(f"    {'class':<22} {'N':>5} {'acc':>6} {'withinCls_agrAUROC':>18}")
        for cls, d in c2["per_class"].items():
            lines.append(f"    {cls:<22} {d['n']:>5} {d['acc']*100:>5.2f}% {d['within_class_agreement_auroc']:>18.4f}")
        c3 = res["C3_combined"]
        lines.append("  C3 combined signal vs instance agreement alone:")
        if "instance_plus_va_anchorscore_auroc" in c3:
            ci = c3["instance_plus_va_minus_instance_ci"]
            lines.append(f"    instance + VA AnchorScore : {c3['instance_plus_va_anchorscore_auroc']:.4f}  "
                         f"(delta vs instance: {ci['mean']:+.4f} CI[{ci['ci_lo']:+.4f},{ci['ci_hi']:+.4f}])")
        ci = c3["instance_plus_classmean_minus_instance_ci"]
        lines.append(f"    instance + class-mean     : {c3['instance_plus_classmean_auroc']:.4f}  "
                     f"(delta vs instance: {ci['mean']:+.4f} CI[{ci['ci_lo']:+.4f},{ci['ci_hi']:+.4f}])")
        # C4 fair selective accuracy
        c4 = res["C4_fair_selective"]
        lines.append("  C4 FAIR selective accuracy (tie-aware, 200 seeds; classmean=5-fold held-out):")
        lines.append(f"    {'coverage':>8} | {'instance':>9} {'classmean_ho':>12} "
                     f"{'VA':>8} | {'withinCls_inst_lift':>20}")
        for cov in c4["coverages"]:
            r = c4["selective_accuracy"][cov]
            va_str = f"{r.get('va_anchorscore',float('nan'))*100:>7.2f}%" if 'va_anchorscore' in r else "   n/a "
            lift = c4["within_class_instance_lift"][cov] * 100
            lines.append(f"    {int(cov*100):>7}% | {r['instance']*100:>8.2f}% "
                         f"{r['classmean_heldout']*100:>11.2f}% {va_str} | {lift:>+19.2f}pp")
        lines.append("")
    lines.append("=" * 86)
    lines.append("  Reading: C1 instance>class-mean>VA => granularity helps. "
                 "C2>0.5 => instance signal not redundant with class. "
                 "C3 delta CI >0 => class prior adds complementary info on top of instance (both deltas CI>0).")
    lines.append("  C4 within-class lift >0 => instance agreement selects better-than-random WITHIN classes.")
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subsets", nargs="+", default=["teacher_behavior"])
    parser.add_argument("--bootstrap_n", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ts = time.strftime("%Y%m%d_%H%M%S")
    setup_logging(ts, name="anchoring_class_vs_instance")
    log = logging.getLogger()
    log.info(f"Exp C | args={vars(args)}")

    results = []
    for subset in args.subsets:
        r = evaluate(subset, args.bootstrap_n, args.seed, log)
        if r is not None:
            results.append(r)
    if not results:
        log.error("No results."); return 1

    txt = format_report(results, args)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "anchoring_class_vs_instance_report.json", "w") as f:
        json.dump({"args": vars(args), "timestamp": ts, "results": results}, f, indent=2)
    with open(OUT_DIR / "anchoring_class_vs_instance_report.txt", "w") as f:
        f.write(txt)
    print(); print(txt)
    return 0


if __name__ == "__main__":
    exit(main())
