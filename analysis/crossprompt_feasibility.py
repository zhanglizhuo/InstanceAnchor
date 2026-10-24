"""
analysis/crossprompt_feasibility.py — Cross-prompt agreement feasibility check
===============================================================================
Q: does per-image consistency ACROSS CAPE prompt sets (A/B/C) carry
selective-prediction signal beyond cross-backbone agreement?

This is a feasibility probe for "Turnaround A" (new signal dimension,
zero extraction cost):
  - prompt-consistency: for each backbone, do its argmax predictions agree
    across the 3 CAPE prompt sets?
  - cross-prompt consensus: average logits over (5 backbones x 3 sets) = 15
    votes; agreement = fraction of votes matching the 15-vote consensus.

Comparisons (all on TeacherBehavior, the only agreement-positive subset):
  - agreement_15 (15-vote consensus) vs entropy_15  [compute-matched: both
    use the same 15 logit vectors]
  - prompt-consistency (5%) vs entropy_5 (CAPE-A ensemble, 1x compute)
  - entropy_15 vs entropy_5  (does adding prompts help entropy at all?)
  - Spearman corr between prompt-consistency, agreement_5, agreement_15
  - simple z-scored additive combination

Direction convention: signal high = keep / low = abstain (AUROC vs correct).
entropy inputs are sign-flipped (-entropy).

Data: results/per_image/clip_{bb}_teacher_behavior_logits{_capeb,_capec}.npy

Output: results/crossprompt_feasibility_report.{json,txt}
Reproducibility: python analysis/crossprompt_feasibility.py
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "analysis"))

from pilot_perimage_anchoring import auroc, bootstrap_auroc_diff  # noqa: E402

BACKBONES = ["openai", "laion", "siglip2", "eva02", "dfn"]
SUBSET = "teacher_behavior"
CAPE_SUFFIX = {"A": "", "B": "_capeb", "C": "_capec"}
N_BOOT = 10_000
SEED = 42
OUT = REPO_ROOT / "results" / "crossprompt_feasibility_report"


def spearman(x, y):
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    return np.corrcoef(rx, ry)[0, 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bootstrap", type=int, default=N_BOOT)
    args = ap.parse_args()

    data_dir = REPO_ROOT / "results" / "per_image"
    by_set = {}
    for cape, suffix in CAPE_SUFFIX.items():
        logits = []
        for bb in BACKBONES:
            lp = data_dir / f"clip_{bb}_{SUBSET}_logits{suffix}.npy"
            logits.append(np.load(lp))
        by_set[cape] = np.stack(logits)          # (B, N, C)
    lit = data_dir / f"clip_{BACKBONES[0]}_{SUBSET}_labels.npy"
    labels = np.load(lit)
    N = labels.shape[0]

    capes = np.stack([by_set[c] for c in "ABC"])  # (3, B, N, C)

    def argmaxes_of(x):
        return np.argmax(x, axis=-1)

    # ---- signals ---------------------------------------------------------
    bb_arg = np.stack([argmaxes_of(by_set[c]) for c in "ABC"])  # (3, B, N)
    prompt_consistency = np.mean(bb_arg == bb_arg[0], axis=0).mean(axis=0)
    # (N,) in [1/3, 2/3, 1]: fraction of (backbone, prompt) votes matching
    # each backbone's CAPE-A argmax, averaged over backbones -> uses ALL 3
    # prompt sets, i.e. 3x compute.

    avg15 = np.mean(np.stack([by_set[c] for c in "ABC"]), axis=(0, 1))  # (N, C)
    cons15 = np.argmax(avg15, axis=1)
    votes15 = np.stack([argmaxes_of(by_set[c]) for c in "ABC"])    # (3,B,N)
    # quantize to k/15 grid: cross-set averaging of k/5 votes produces
    # 1e-16 noise that the tie-aware ranker misreads as distinct values
    # (AUROC shift ~1.4e-3, order-dependent). Rounding fixes it.
    agreement_15 = np.round(np.mean(votes15 == cons15[None, None, :], axis=(0, 1)) * 15) / 15

    avg5 = np.mean(by_set["A"], axis=0)
    cons5 = np.argmax(avg5, axis=1)
    votes5 = argmaxes_of(by_set["A"])                              # (B, N)
    agreement_5 = np.round(np.mean(votes5 == cons5[None, :], axis=0) * 5) / 5

    from pilot_perimage_anchoring import softmax_entropy
    entropy_5, _ = softmax_entropy(avg5)
    entropy_15, _ = softmax_entropy(avg15)

    correct15 = labels[np.arange(N), cons15] > 0
    correct5 = labels[np.arange(N), cons5] > 0

    # ---- AUROC table ----------------------------------------------------
    sigs = {
        "agreement_5": (agreement_5, correct15),
        "agreement_15": (agreement_15, correct15),
        "prompt_consistency": (prompt_consistency, correct15),
        "entropy_5": (-entropy_5, correct15),
        "entropy_15": (-entropy_15, correct15),
    }
    aurocs = {k: auroc(s, y) for k, (s, y) in sigs.items()}

    # ---- coupled bootstrap deltas --------------------------------------
    def delta_scores(sa, sb, y):
        d = bootstrap_auroc_diff(sa, sb, y, args.bootstrap, SEED)
        return {"mean": float(d["mean"]), "ci_lo": float(d["ci_lo"]), "ci_hi": float(d["ci_hi"])}

    def delta(a, b, y):
        return delta_scores(sigs[a][0], sigs[b][0], y)

    deltas = {
        "agreement15_vs_entropy15": delta("agreement_15", "entropy_15", correct15),
        "promptcons_vs_entropy15": delta("prompt_consistency", "entropy_15", correct15),
        "entropy15_vs_entropy5": delta("entropy_15", "entropy_5", correct15),
        "agreement15_vs_agreement5": delta("agreement_15", "agreement_5", correct15),
        "agreement5_vs_entropy5": delta("agreement_5", "entropy_5", correct15),
    }

    # ---- redundancy: are the agreement signals the same info? -----------
    corr = {
        "spearman_agr5_vs_promptcons": spearman(agreement_5, prompt_consistency),
        "spearman_agr15_vs_promptcons": spearman(agreement_15, prompt_consistency),
        "spearman_agr5_vs_agr15": spearman(agreement_5, agreement_15),
    }

    # ---- additive combination (z-scored) --------------------------------
    def z(x):
        return (x - x.mean()) / (x.std() + 1e-12)

    combo = z(agreement_15) + z(prompt_consistency)
    aurocs["combo_agr15+pc"] = auroc(combo, correct15)
    deltas["combo_vs_entropy15"] = delta_scores(combo, sigs["entropy_15"][0], correct15)

    report = {
        "subset": SUBSET,
        "n_images": int(N),
        "backbones": BACKBONES,
        "note": "CAPE A/B/C per-image logits; consensus needs NO new extraction",
        "consensus5_acc": float(correct5.mean()),
        "consensus15_acc": float(correct15.mean()),
        "auroc_table": {k: float(v) for k, v in aurocs.items()},
        "coupled_deltas": deltas,
        "redundancy_spearman": corr,
    }
    txt = [
        f"Cross-prompt agreement feasibility ({SUBSET}, N={N})",
        f"  consensus CAPE-A acc={correct5.mean():.4f}  consensus 15-vote acc={correct15.mean():.4f}",
        "AUROC (higher=keep):",
        *[f"  {k:24s} {v:.4f}" for k, v in aurocs.items()],
        "Coupled deltas (mean, 95% CI):",
        *[f"  {k:26s} {v['mean']:+.4f} [{v['ci_lo']:+.4f}, {v['ci_hi']:+.4f}]"
          for k, v in deltas.items()],
        "Spearman redundancy:",
        *[f"  {k:26s} {v:+.3f}" for k, v in corr.items()],
    ]
    with open(OUT.with_suffix(".json"), "w") as f:
        json.dump({"args": vars(args), "results": report}, f, indent=1)
    with open(OUT.with_suffix(".txt"), "w") as f:
        f.write("\n".join(txt) + "\n")
    print("\n".join(txt))


if __name__ == "__main__":
    main()