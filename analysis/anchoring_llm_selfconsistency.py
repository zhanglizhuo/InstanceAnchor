"""
analysis/anchoring_llm_selfconsistency.py — kappa decomposition for LLM
self-consistency
=========================================================================
Transfers the paper's core diagnostic to LLM self-consistency voting:
sampled rollouts are the "voters" (exchangeable draws of one model), the C
answer options are the classes, the plurality answer is the consensus, and
agreement alpha is the fraction of samples voting the consensus.

Same protocol as the vision ensembles (anchoring_kappa_iid_baseline /
anchoring_deep_ensemble):
  p           = pooled per-sample accuracy (mean over questions)
  E[a|c] / E[a|w] = mean agreement on consensus-correct / consensus-wrong
                    questions
  kappa_emp   = E[a|w] * (C-1) / (1-p)
  kappa_iid   = simulate_iid_kappa(p, K, C)   (identical function/seed)
  plurality_share = kappa_iid / kappa_emp

Interpretation note (stated in the paper): the i.i.d. baseline uses the
POOLED accuracy; question-level difficulty spread makes real samples
correlated within a question (easy questions: most samples right; hard
questions: most wrong), and this difficulty correlation is the "shared
bias" component that the residual (1 - share) quantifies.

Also reports agreement-binned accuracy (cross-check with Bahuguna 2026,
"When Self-Consistency Backfires": the highest-agreement bin is only
~50% correct on hard problems).

Input : results/per_llm/llm_sc_{model}_{dataset}_{n_samples}.jsonl
Output: results/anchoring_llm_selfconsistency_report.{json,txt}

Usage: python analysis/anchoring_llm_selfconsistency.py [--jsonl_dir results/per_llm]
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase0_oracle_analysis import setup_logging, OUT_DIR  # noqa: E402
from anchoring_kappa_iid_baseline import simulate_iid_kappa  # noqa: E402


def simulate_iid_kappa_perq(p_list, K, C, M=200000, seed=42):
    """i.i.d. null GIVEN question difficulty: each simulated question draws a
    per-question accuracy from the empirical distribution {p_i}, then K
    independent voters at that accuracy. This isolates the plurality
    mechanics under the difficulty structure; the residual vs empirical kappa
    is the within-question choice correlation (shared bias).

    Mirrors simulate_iid_kappa (ties -> lowest index via argmax on counts)."""
    rng = np.random.RandomState(seed)
    p_list = np.asarray(p_list, dtype=float)
    p = rng.choice(p_list, size=M)
    correct = rng.random((M, K)) < p[:, None]
    preds = np.where(correct, 0, rng.randint(1, C, size=(M, K)))
    counts = np.zeros((M, C))
    for k in range(K):
        counts[np.arange(M), preds[:, k]] += 1
    cons = np.argmax(counts, axis=1)
    cons_correct = (cons == 0)
    alpha = counts[np.arange(M), cons] / K
    e_a_w = float(alpha[~cons_correct].mean()) if (~cons_correct).sum() else float("nan")
    p_mean = float(p.mean())
    kappa = e_a_w * (C - 1) / (1 - p_mean) if (1 - p_mean) > 0 else float("inf")
    return {"E_alpha_wrong_iid": e_a_w, "kappa_iid": kappa,
            "consensus_accuracy_iid": float(cons_correct.mean()),
            "wrong_consensus_rate_iid": float((~cons_correct).mean())}


def analyze_file(path, log):
    records = [json.loads(l) for l in open(path) if l.strip()]
    rows = []
    for r in records:
        C = int(r["C"])
        votes = [v for v in r["votes"] if 0 <= v < C]
        if len(votes) >= 2 and r["gt"] >= 0:
            rows.append({"idx": r["idx"], "gt": int(r["gt"]), "votes": votes,
                         "C": C, "K": len(r["votes"])})
    if not rows:
        log.warning(f"{path}: no valid rows")
        return None
    C = rows[0]["C"]
    K = max(r["K"] for r in rows)
    p_i, correct, alpha = [], [], []
    for r in rows:
        v = np.array(r["votes"], dtype=int)
        p_i.append(float((v == r["gt"]).mean()))
        hist = np.bincount(v, minlength=C)
        cons = int(np.argmax(hist))          # ties -> lowest index (== MC)
        correct.append(cons == r["gt"])
        alpha.append(float((v == cons).mean()))
    p_i = np.array(p_i)
    correct = np.array(correct)
    alpha = np.array(alpha)
    p = float(p_i.mean())
    e_a_c = float(alpha[correct].mean()) if correct.sum() else float("nan")
    e_a_w = float(alpha[~correct].mean()) if (~correct).sum() else float("nan")
    kemp = e_a_w * (C - 1) / (1 - p) if (1 - p) > 0 else float("inf")
    sim_pooled = simulate_iid_kappa(p, K, C)
    sim_perq = simulate_iid_kappa_perq(p_i, K, C)
    share_perq = sim_perq["kappa_iid"] / kemp if kemp > 0 else float("nan")

    bins = []
    if len(rows) >= 8:
        qs = np.quantile(alpha, [0.25, 0.5, 0.75])
        edges = [-np.inf] + list(qs) + [np.inf]
        for b in range(4):
            m = (alpha >= edges[b]) & (alpha < edges[b + 1])
            bins.append({"lo": float(edges[b]), "hi": float(edges[b + 1]),
                         "n": int(m.sum()),
                         "acc": float(correct[m].mean()) if m.sum() else None})

    row = {
        "file": Path(path).name, "n_questions": len(rows),
        "C": C, "K": K, "p": p, "p_std": float(p_i.std()),
        "consensus_acc": float(correct.mean()),
        "wrong_consensus_rate_empirical": float((~correct).mean()),
        "E_alpha_correct": e_a_c, "E_alpha_wrong": e_a_w,
        "kappa_empirical": float(kemp),
        "kappa_iid_pooled": float(sim_pooled["kappa_iid"]),
        "wrong_consensus_rate_iid_pooled":
            float(1 - sim_pooled["consensus_accuracy_iid"]),
        "kappa_iid_perq": float(sim_perq["kappa_iid"]),
        "wrong_consensus_rate_iid_perq": sim_perq["wrong_consensus_rate_iid"],
        "plurality_share_perq": float(share_perq),
        "agreement_bins": bins,
    }
    log.info(f"{row['file']}: n={row['n_questions']} C={C} K={K} p={p:.3f} "
             f"cons={row['consensus_acc']:.3f} "
             f"wrong_emp={1-correct.mean():.3f} "
             f"wrong_iid_pooled={row['wrong_consensus_rate_iid_pooled']:.3f} "
             f"wrong_iid_perq={sim_perq['wrong_consensus_rate_iid']:.3f} "
             f"k_emp={kemp:.1f} k_iid_perq={sim_perq['kappa_iid']:.1f} "
             f"share_perq={share_perq*100:.0f}%")
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl_dir", default="results/per_llm")
    args = parser.parse_args()
    ts = time.strftime("%Y%m%d_%H%M%S")
    setup_logging(ts, name="anchoring_llm_selfconsistency")
    log = logging.getLogger()

    jdir = OUT_DIR / "per_llm" if args.jsonl_dir == "results/per_llm" \
        else Path(args.jsonl_dir)
    rows = []
    for p in sorted(jdir.glob("llm_sc_*.jsonl")):
        r = analyze_file(p, log)
        if r:
            rows.append(r)
    if not rows:
        log.error("no jsonl found")
        return 1

    out = {"args": vars(args), "timestamp": ts, "results": rows}
    with open(OUT_DIR / "anchoring_llm_selfconsistency_report.json", "w") as f:
        json.dump(out, f, indent=2, default=lambda o: float(o))
    lines = ["=" * 90, "  kappa decomposition: LLM self-consistency",
             "=" * 90]
    for r in rows:
        lines.append(f"  {r['file']}: n={r['n_questions']} C={r['C']} K={r['K']} "
                     f"p={r['p']:.3f} consensus={r['consensus_acc']:.3f}")
        lines.append(f"    wrong-consensus rate: empirical={r['wrong_consensus_rate_empirical']:.3f} "
                     f"iid-pooled={r['wrong_consensus_rate_iid_pooled']:.3f} "
                     f"iid-perq={r['wrong_consensus_rate_iid_perq']:.3f}")
        lines.append(f"    E[a|c]={r['E_alpha_correct']:.3f} "
                     f"E[a|w]={r['E_alpha_wrong']:.3f} "
                     f"kappa_emp={r['kappa_empirical']:.1f} "
                     f"kappa_iid_perq={r['kappa_iid_perq']:.1f} "
                     f"share_perq={r['plurality_share_perq']*100:.0f}%")
        bin_txt = " | ".join(
            f"[{b['lo']:.2f},{b['hi']:.2f}):acc={b['acc']:.2f}"
            if b['acc'] is not None else "" for b in r['agreement_bins'])
        lines.append(f"    agreement bins: {bin_txt}")
    lines.append("=" * 90)
    txt = "\n".join(lines)
    with open(OUT_DIR / "anchoring_llm_selfconsistency_report.txt", "w") as f:
        f.write(txt)
    print(txt)
    return 0


if __name__ == "__main__":
    exit(main())
