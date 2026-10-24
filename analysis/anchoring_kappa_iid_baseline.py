"""
analysis/anchoring_kappa_iid_baseline.py — Cross-validate κ attribution
========================================================================
Tests whether empirical κ is dominated by plurality mechanics (iid) or
correlated errors. Simulates iid backbone votes at matched (p, C, K) for
each dataset and compares the iid-MC κ to the empirical κ from the
saturation report.

Output:
  results/anchoring_kappa_iid_baseline_report.{json,txt}

Reproducibility:
  python analysis/anchoring_kappa_iid_baseline.py
"""
import json, sys, logging, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase0_oracle_analysis import OUT_DIR

log = logging.getLogger(__name__)


def simulate_iid_kappa(p, K, C, N=100000, seed=42):
    """Simulate iid backbone votes with per-backbone accuracies p (scalar or
    array of length K); return E[alpha|correct], E[alpha|wrong], implied kappa.
    Per-backbone p_b: backbone b is correct with probability p_b and, when
    incorrect, chooses uniformly among the C-1 wrong classes."""
    chunk = 200_000
    rng = np.random.RandomState(seed)
    p = np.atleast_1d(p)
    if len(p) == 1:
        p = np.full(K, p[0])
    assert len(p) == K
    correct = rng.random((N, K)) < p[None, :]
    preds = np.where(correct, 0, rng.randint(1, C, size=(N, K)))
    onehot = np.zeros((N, K, C))
    for k in range(K):
        onehot[np.arange(N), k, preds[:, k]] = 1
    avg = onehot.mean(1)
    cons = np.argmax(avg, 1)
    cons_correct = (cons == 0)
    alpha = (preds == cons[:, None]).mean(1)

    e_a_correct = float(alpha[cons_correct].mean()) if cons_correct.sum() > 0 else float('nan')
    e_a_wrong = float(alpha[~cons_correct].mean()) if (~cons_correct).sum() > 0 else float('nan')
    p_mean = float(p.mean())
    marginal = (1 - p_mean) / (C - 1)
    kappa = e_a_wrong * (C - 1) / (1 - p_mean) if (1 - p_mean) > 0 else float('inf')
    return {
        "E_alpha_correct_iid": e_a_correct,
        "E_alpha_wrong_iid": e_a_wrong,
        "marginal_rate": marginal,
        "kappa_iid": kappa,
        "consensus_accuracy_iid": float(cons_correct.mean()),
    }

def chunked_iid_stats(p, K, C, N, seed=42):
    # NOTE: chunk size is FIXED at 200_000. numpy 2.x RandomState uses
    # batch-size-dependent vectorized fill paths, so the RNG stream (and
    # therefore the estimate) changes with the chunk size. Canonical value
    # 200_000 matches the committed decomposition report. Do not add a
    # chunk parameter.
    """Memory-bounded version of simulate_iid_kappa; returns the full dict.
    Each chunk draws independent votes; consensus ties resolve to the lowest
    class id (argmax on counts), matching simulate_iid_kappa exactly."""
    chunk = 200_000
    rng = np.random.RandomState(seed)
    p = np.atleast_1d(p)
    if len(p) == 1:
        p = np.full(K, p[0])
    n_wrong = 0
    sum_a_c = 0.0
    sum_a_w = 0.0
    cons_correct_total = 0
    done = 0
    while done < N:
        m = min(chunk, N - done)
        correct = rng.random((m, K)) < p[None, :]
        preds = np.where(correct, 0, rng.randint(1, C, size=(m, K)))
        counts = np.zeros((m, C))
        for k in range(K):
            counts[np.arange(m), preds[:, k]] += 1
        cons = np.argmax(counts, axis=1)
        cons_correct = (cons == 0)
        alpha = (preds == cons[:, None]).mean(axis=1)
        n_wrong += int((~cons_correct).sum())
        sum_a_c += float(alpha[cons_correct].sum())
        sum_a_w += float(alpha[~cons_correct].sum())
        cons_correct_total += int(cons_correct.sum())
        done += m
    e_a_c = sum_a_c / cons_correct_total if cons_correct_total > 0 else float("nan")
    e_a_w = sum_a_w / n_wrong if n_wrong > 0 else float("nan")
    p_mean = float(p.mean())
    marginal = (1 - p_mean) / (C - 1)
    kappa = e_a_w * (C - 1) / (1 - p_mean) if (1 - p_mean) > 0 else float("inf")
    return {
        "E_alpha_correct_iid": e_a_c,
        "E_alpha_wrong_iid": e_a_w,
        "marginal_rate": marginal,
        "kappa_iid": kappa,
        "consensus_accuracy_iid": float(cons_correct_total / N),
        "n_wrong_consensus": n_wrong,
    }

def per_backbone_accs(name):
    """Per-backbone Hit@1 (argmax in any GT class) from the per-image npy,
    matching the main analysis protocol. Returns list of 5 accuracies or None."""
    import numpy as _np
    from pilot_perimage_anchoring import BACKBONES
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from phase0_oracle_analysis import load_clip, SUBSET_CONFIG
    out_dir = OUT_DIR / "per_image"

    def _acc(logits, labels):
        cons_argmax = logits.argmax(1)
        if labels.ndim == 1:
            gt_pos = labels.astype(int)[:, None] == _np.arange(logits.shape[1])[None, :]
        else:
            gt_pos = labels > 0
        return float(_np.array([cons_argmax[i] in _np.where(gt_pos[i])[0]
                                for i in range(len(cons_argmax))], dtype=bool).mean())

    try:
        if name in ("TeacherBehavior", "HandriseReadWrite", "BowTurnHead"):
            subset = {"TeacherBehavior": "teacher_behavior",
                      "HandriseReadWrite": "handrise_readwrite",
                      "BowTurnHead": "bow_turnhead"}[name]
            accs = []
            for bb in BACKBONES:
                clip = load_clip(bb, subset, logging.getLogger("x"))
                if clip is None:
                    return None
                accs.append(_acc(*clip))
            return accs
        if name in ("VOC2007", "Stanford40", "COCO"):
            ds = {"VOC2007": "voc2007", "Stanford40": "stanford40", "COCO": "coco"}[name]
            accs = []
            for bb in BACKBONES:
                lp = out_dir / f"clip_{bb}_{ds}_logits.npy"
                bp = out_dir / f"clip_{bb}_{ds}_labels.npy"
                if not lp.exists():
                    return None
                accs.append(_acc(_np.load(str(lp)), _np.load(str(bp))))
            return accs
        if name == "CIFAR-100":
            accs = []
            for bb in BACKBONES:
                lp = out_dir / "cifar100" / f"clip_{bb}_cifar100_logits.npy"
                bp = out_dir / "cifar100" / f"clip_{bb}_cifar100_labels.npy"
                if not lp.exists():
                    return None
                accs.append(_acc(_np.load(str(lp)), _np.load(str(bp))))
            return accs
        if name == "MultiTHUMOS":
            from anchoring_multithumos import load_multithumos
            stack, gt = load_multithumos()
            return [_acc(stack[k], gt) for k in range(stack.shape[0])]
    except Exception as e:  # noqa: BLE001
        logging.getLogger(__name__).warning("per-backbone accs unavailable for %s: %s", name, e)
    return None


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])
    ts = time.strftime("%Y%m%d_%H%M%S")

    sat = json.load(open(OUT_DIR / "anchoring_saturation_model_report.json"))
    results = []
    for r in sat["datasets"]:
        p_arr = per_backbone_accs(r["name"])
        if p_arr is None:
            p_arr = [r["per_backbone_acc_p"]] * 5
        sim = simulate_iid_kappa(p_arr, 5, r["C"])
        ratio = r["kappa"] / sim["kappa_iid"] if sim["kappa_iid"] > 0 else float('inf')
        plurality_share = sim["kappa_iid"] / r["kappa"] if r["kappa"] > 0 else 0
        out = {
            "name": r["name"], "C": r["C"], "p": r["per_backbone_acc_p"],
            "per_backbone_accs": p_arr,
            "kappa_empirical": r["kappa"],
            "kappa_iid_mc": sim["kappa_iid"],
            "kappa_ratio_real_over_iid": ratio,
            "plurality_share_of_kappa": plurality_share,
            "correlation_share_of_kappa": 1 - plurality_share,
            "E_alpha_correct_empirical": r["alpha_correct_E"],
            "E_alpha_correct_iid": sim["E_alpha_correct_iid"],
            "E_alpha_wrong_empirical": r["alpha_wrong_E"],
            "E_alpha_wrong_iid": sim["E_alpha_wrong_iid"],
        }
        results.append(out)
        log.info(f"{r['name']:22s} C={r['C']:>3d} p={r['per_backbone_acc_p']:.3f}  "
                 f"κ_emp={r['kappa']:>6.2f}  κ_iid={sim['kappa_iid']:>6.2f}  "
                 f"ratio={ratio:.2f}  plurality_share={plurality_share*100:.0f}%")

    log.info(f"\n{'='*80}")
    log.info("CROSS-VALIDATION: κ attribution (plurality mechanics vs correlated errors)")
    log.info(f"{'='*80}")
    log.info(f"\n{'Dataset':22s} {'C':>3s} {'κ_emp':>7s} {'κ_iid':>7s} {'ratio':>6s} {'plurality%':>11s} {'correlation%':>13s}")
    for r in results:
        log.info(f"{r['name']:22s} {r['C']:>3d} {r['kappa_empirical']:>7.2f} "
                 f"{r['kappa_iid_mc']:>7.2f} {r['kappa_ratio_real_over_iid']:>6.2f} "
                 f"{r['plurality_share_of_kappa']*100:>10.0f}% {r['correlation_share_of_kappa']*100:>12.0f}%")

    report = {"timestamp": ts, "datasets": results}
    out_json = OUT_DIR / "anchoring_kappa_iid_baseline_report.json"
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)
    log.info(f"\nReport: {out_json}")


if __name__ == "__main__":
    main()
