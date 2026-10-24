"""
analysis/anchoring_synthetic_colabel.py — P0: synthetic co-occurrence control
=============================================================================
Elevation-plan P0. Attacks hard-flaw #4 (no mechanism): converts the
"co-occurrence ambiguity drives agreement's edge" story from an observed
pattern into a controlled manipulation. Key insight: logits are class-
conditional; the ground-truth label set is the one thing we can change
freely, so synthetic multi-label experiments cost zero GPU.

Two manipulation axes:
  (1) Label injection (pure GT manipulation, logits untouched) — RANDOM only:
      correctness = consensus-argmax in {true label} union {k-1 randomly
      injected classes}. Negative control: label-set expansion alone.
      (Confusable injection is circular by construction: the runner-up by
      avg-logit IS the consensus argmax whenever the true class is not #1,
      so the target becomes trivially correct.)
  (2) Image merging (logit mixing, co-occurrence proxy):
      synthetic sample = mean logits of k images; GT = union of their
      labels. k = 2, 3, 4. Plus k=2 with confusable pairing (partner j
      chosen so its class ranks high in the anchor's averaged logits —
      co-occurrence the model itself finds confusing).

Testbed: CIFAR-100 (single-label; agreement loses -0.043 natively) with
VOC2007 (native multi-label GT; tie -0.007) as control.

Question: does the agreement-minus-entropy AUROC advantage EMERGE at
k = 2-3 and recede at k = 4+ (replicating TeacherBehavior's inverted-U)?

Output: results/anchoring_synthetic_colabel_report.{json,txt}

Reproducibility:
  python analysis/anchoring_synthetic_colabel.py [--bootstrap_n 10000]
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
from pilot_perimage_anchoring import (  # noqa: E402
    BACKBONES, softmax_entropy, auroc, aupr, bootstrap_auroc_diff,
)

CIFAR_DIR = DATA_DIR / "cifar100"


def load_dataset(dataset, log):
    """Load 5-backbone logits + GT labels. Returns (stack (K,N,C), gt).

    gt: CIFAR -> (N,) int classes; VOC2007 -> (N,C) one-hot multi-label."""
    stack = []
    gt = None
    for bb in BACKBONES:
        if dataset == "cifar100":
            lp, bp = CIFAR_DIR / f"clip_{bb}_cifar100_logits.npy", CIFAR_DIR / f"clip_{bb}_cifar100_labels.npy"
        else:
            lp, bp = DATA_DIR / f"clip_{bb}_{dataset}_logits.npy", DATA_DIR / f"clip_{bb}_{dataset}_labels.npy"
        if not lp.exists():
            log.warning(f"missing {lp}")
            return None
        logits = np.load(lp).astype(np.float32)
        labels = np.load(bp).astype(np.float32)
        if gt is None:
            gt = labels
        elif not np.array_equal(gt, labels):
            raise RuntimeError(f"label mismatch on {dataset}")
        stack.append(logits)
    return np.stack(stack, axis=0), gt


def signals_from(logits_stack, avg_logits):
    ent_avg, _ = softmax_entropy(avg_logits)
    backbone_argmaxes = np.argmax(logits_stack, axis=2)
    consensus_argmax = np.argmax(avg_logits, axis=1)
    agreement = (backbone_argmaxes == consensus_argmax[None, :]).mean(axis=0)
    return agreement, -ent_avg, consensus_argmax


def correctness_from_sets(consensus, gt_sets):
    """gt_sets: list of python sets (class indices)."""
    return np.array([c in s for c, s in zip(consensus, gt_sets)], dtype=bool)


def run_config(agreement, entropy, correct, bootstrap_n, seed, log, tag):
    if correct.std() == 0:
        log.info(f"  {tag}: degenerate target (all {bool(correct[0])}), skipped")
        return None
    if correct.mean() > 0.99:
        log.info(f"  {tag}: target too easy (acc={correct.mean():.4f}); bootstrap would "
                 f"degenerate, skipped")
        return None
    delta = bootstrap_auroc_diff(agreement, entropy, correct, bootstrap_n, seed)
    out = {
        "tag": tag,
        "n": int(len(correct)),
        "correct_frac": float(correct.mean()),
        "agreement_mean": float(agreement.mean()),
        "entropy_mean": float(entropy.mean()),
        "entropy_auroc": auroc(entropy, correct),
        "agreement_auroc": auroc(agreement, correct),
        "entropy_aupr": aupr(entropy, correct),
        "agreement_aupr": aupr(agreement, correct),
        "delta_mean": delta["mean"],
        "delta_ci_lo": delta["ci_lo"],
        "delta_ci_hi": delta["ci_hi"],
    }
    log.info(f"  {tag}: acc={out['correct_frac']:.3f} ent={out['entropy_auroc']:.3f} "
             f"agr={out['agreement_auroc']:.3f} d={delta['mean']:+.4f} "
             f"CI[{delta['ci_lo']:+.4f},{delta['ci_hi']:+.4f}]")
    return out


def label_injection(logits_stack, gt, dataset, k, mode, rng):
    """Correctness with k-1 injected classes. Returns gt_sets (list of sets).

    mode='confusable': top runner-up classes by avg logit; 'random': uniform."""
    K, N, C = logits_stack.shape
    avg = logits_stack.mean(axis=0)
    if dataset == "cifar100":
        primary = gt.astype(int)
        real_sets = [{int(y)} for y in primary]
    else:  # one-hot
        real_sets = [set(np.where(gt[i] > 0)[0].tolist()) for i in range(N)]
    consensus = np.argmax(avg, axis=1)
    for i in range(N):
        s = set(real_sets[i])
        if len(s) >= k:
            continue  # already at least k labels; no injection needed
        if mode == "confusable":
            order = np.argsort(-avg[i])
            for c in order:
                if c not in s:
                    s.add(int(c))
                    if len(s) >= k:
                        break
        else:
            while len(s) < k:
                c = int(rng.randint(0, C))
                if c not in s:
                    s.add(c)
        real_sets[i] = s
    return real_sets, consensus


def image_merging(logits_stack, gt, dataset, k, rng, n_synth, pairing="random"):
    """Synthetic samples from k merged images. Returns (stack', gt_sets, consensus).

    pairing='random': partners sampled uniformly.
    pairing='confusable': for anchor i, partner j maximizes avg_logits[i, y_j]
    (classes the anchor model finds confusable). k=2 only."""
    K, N, C = logits_stack.shape
    if dataset == "cifar100":
        y = gt.astype(int)
    else:
        y = np.array([np.where(gt[i] > 0)[0] for i in range(N)], dtype=object)
    anchor = rng.randint(0, N, size=n_synth)
    idx = np.empty((n_synth, k), dtype=np.int64)
    idx[:, 0] = anchor
    if pairing == "confusable" and k == 2:
        avg = logits_stack.mean(axis=0)
        for i in range(n_synth):
            cands = rng.randint(0, N, size=20)
            if dataset == "cifar100":
                conf = np.array([avg[anchor[i], int(y[c])] for c in cands])
            else:
                conf = np.array([avg[anchor[i], int(y[c][0])] for c in cands])
            idx[i, 1] = cands[int(np.argmax(conf))]
    else:
        idx[:, 1:] = rng.randint(0, N, size=(n_synth, k - 1))
    merged = logits_stack[:, idx, :].mean(axis=2)  # (K, n_synth, k, C) -> (K, n_synth, C)
    avg = merged.mean(axis=0)
    consensus = np.argmax(avg, axis=1)
    gt_sets = []
    for i in range(n_synth):
        if dataset == "cifar100":
            s = set(int(v) for v in y[idx[i]])
        else:
            s = set()
            for j in range(k):
                s |= set(int(v) for v in y[idx[i, j]])
        gt_sets.append(s)
    return merged, gt_sets, consensus


def evaluate(dataset, bootstrap_n, seed, log):
    log.info(f"=== {dataset} ===")
    loaded = load_dataset(dataset, log)
    if loaded is None:
        return None
    logits_stack, gt = loaded
    K, N, C = logits_stack.shape
    avg_logits = logits_stack.mean(axis=0)
    rng = np.random.RandomState(seed)

    out = {"dataset": dataset, "n_images": int(N), "n_classes": int(C),
           "configs": []}

    # native baseline (k=1 for CIFAR; real multi-label GT for VOC)
    agreement, entropy, consensus = signals_from(logits_stack, avg_logits)
    if dataset == "cifar100":
        native_correct = consensus == gt.astype(int)
        native_tag = "native_k1_singlelabel"
    else:
        native_correct = np.array(
            [consensus[i] in np.where(gt[i] > 0)[0] for i in range(N)], dtype=bool)
        native_tag = "native_multilabel_GT"
    r = run_config(agreement, entropy, native_correct, bootstrap_n, seed, log, native_tag)
    if r:
        out["configs"].append(r)

    # axis 1: label injection — random only (confusable injection is circular:
    # the runner-up by avg-logit IS the consensus argmax whenever the true
    # class is not #1, making the target trivially correct by construction)
    for k in (2, 3, 4):
        gt_sets, cons = label_injection(logits_stack, gt, dataset, k, "random", rng)
        correct = correctness_from_sets(cons, gt_sets)
        r = run_config(agreement, entropy, correct, bootstrap_n, seed, log,
                       f"inject_k{k}_random")
        if r:
            out["configs"].append(r)

    # axis 2: image merging (logit mixing; non-circular)
    for k in (2, 3, 4):
        merged, gt_sets, cons = image_merging(logits_stack, gt, dataset, k, rng,
                                              n_synth=min(N, 8000), pairing="random")
        magr, ment, _ = signals_from(merged, merged.mean(axis=0))
        correct = correctness_from_sets(cons, gt_sets)
        r = run_config(magr, ment, correct, bootstrap_n, seed, log,
                       f"merge_k{k}")
        if r:
            out["configs"].append(r)

    # axis 2b: k=2 merging with confusable pairing — for each anchor image i,
    # the partner j is chosen so that y_j ranks high in i's averaged logits
    # (co-occurring classes that the model itself finds confusable)
    merged, gt_sets, cons = image_merging(logits_stack, gt, dataset, 2, rng,
                                          n_synth=min(N, 8000), pairing="confusable")
    magr, ment, _ = signals_from(merged, merged.mean(axis=0))
    correct = correctness_from_sets(cons, gt_sets)
    r = run_config(magr, ment, correct, bootstrap_n, seed, log,
                   f"merge_k2_confusable")
    if r:
        out["configs"].append(r)
    return out


def format_report(results):
    lines = []
    lines.append("=" * 88)
    lines.append("  P0 — Synthetic co-occurrence control (does multi-labelness cause the")
    lines.append("       agreement advantage? Zero-GPU: logits fixed, GT manipulated.)")
    lines.append("=" * 88)
    for ds in results:
        lines.append("")
        lines.append(f"── {ds['dataset']} | N={ds['n_images']} C={ds['n_classes']} ──")
        lines.append(f"    {'config':>28} | {'acc':>7} | {'ent':>7} | {'agr':>7} | "
                     f"{'delta':>8} | CI")
        lines.append("    " + "-" * 74)
        for c in ds["configs"]:
            lines.append(f"    {c['tag']:>28} | {c['correct_frac']*100:>6.1f}% | "
                         f"{c['entropy_auroc']:>7.3f} | {c['agreement_auroc']:>7.3f} | "
                         f"{c['delta_mean']:>+8.4f} | "
                         f"[{c['delta_ci_lo']:+.4f},{c['delta_ci_hi']:+.4f}]")
    lines.append("")
    lines.append("  Question: does delta EMERGE at k=2-3 and recede at k=4+ (TB inverted-U)?")
    lines.append("=" * 88)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["cifar100", "voc2007"])
    parser.add_argument("--bootstrap_n", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ts = time.strftime("%Y%m%d_%H%M%S")
    setup_logging(ts, name="anchoring_synthetic_colabel")
    log = logging.getLogger()

    results = []
    for ds in args.datasets:
        r = evaluate(ds, args.bootstrap_n, args.seed, log)
        if r is None:
            log.error(f"{ds} data missing; abort.")
            return 1
        results.append(r)

    txt = format_report(results)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "anchoring_synthetic_colabel_report.json", "w") as f:
        json.dump({"args": vars(args), "timestamp": ts, "results": results}, f, indent=2)
    with open(OUT_DIR / "anchoring_synthetic_colabel_report.txt", "w") as f:
        f.write(txt)
    print(); print(txt)
    return 0


if __name__ == "__main__":
    exit(main())
