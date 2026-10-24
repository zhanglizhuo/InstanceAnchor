"""
analysis/anchoring_kappa_meta_decomp.py — κ i.i.d.-MC decomposition for 20 datasets
====================================================================================
Extends the κ decomposition (originally 8 datasets) to all 20 meta-study datasets.
Reads κ/C/p from anchoring_meta_report.json; loads per-backbone npy for the
12 new meta datasets; runs the same i.i.d. plurality-vote MC simulation as
anchoring_kappa_iid_baseline.py.

Output: results/anchoring_kappa_meta_decomp_report.{json,txt}
"""

import json, sys, time, logging
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase0_oracle_analysis import OUT_DIR, DATA_DIR, setup_logging  # noqa
from pilot_perimage_anchoring import BACKBONES  # noqa
from anchoring_kappa_iid_baseline import chunked_iid_stats  # noqa

CIFAR_DIR = DATA_DIR / "cifar100"
META_DIR = DATA_DIR / "meta"


def per_backbone_accs_meta(name):
    """Per-backbone Hit@1 for all 20 datasets from npy files."""
    def _acc(logits, labels):
        cons = logits.argmax(1)
        if labels.ndim == 1:
            gt_pos = labels.astype(int)[:, None] == np.arange(logits.shape[1])[None, :]
        else:
            gt_pos = labels > 0
        return float(np.array([cons[i] in np.where(gt_pos[i])[0]
                               for i in range(len(cons))], dtype=bool).mean())

    accs = []
    for bb in BACKBONES:
        if name == "cifar100":
            lp, bp = CIFAR_DIR / f"clip_{bb}_cifar100_logits.npy", CIFAR_DIR / f"clip_{bb}_cifar100_labels.npy"
        elif name in ("teacher_behavior","handrise_readwrite","bow_turnhead",
                       "voc2007","stanford40","coco","multithumos"):
            lp, bp = DATA_DIR / f"clip_{bb}_{name}_logits.npy", DATA_DIR / f"clip_{bb}_{name}_labels.npy"
        else:
            lp, bp = META_DIR / f"clip_{bb}_{name}_logits.npy", META_DIR / f"clip_{bb}_{name}_labels.npy"
        if not lp.exists():
            return None
        accs.append(_acc(np.load(str(lp)).astype(np.float32), np.load(str(bp)).astype(np.float32)))
    return accs


def main():
    mc_n = int(sys.argv[1]) if len(sys.argv) > 1 else 50_000_000
    ts = time.strftime("%Y%m%d_%H%M%S")
    setup_logging(ts, name="anchoring_kappa_meta_decomp")
    log = logging.getLogger()

    meta = json.load(open(OUT_DIR / "anchoring_meta_report.json"))
    results = []
    for r in meta["results"]:
        name = r["dataset"]
        C = r["C"]
        kappa_emp = r["kappa"]
        if not np.isfinite(kappa_emp) or kappa_emp <= 0:
            log.info(f"{name}: κ={kappa_emp} (skip)")
            continue
        p_arr = per_backbone_accs_meta(name)
        if p_arr is None:
            log.warning(f"{name}: npy missing, using pooled p={r['p']}")
            p_arr = [r["p"]] * 5
        sim = chunked_iid_stats(p_arr, 5, C, N=mc_n, seed=42)
        kappa_iid = sim["kappa_iid"]
        plurality_share = kappa_iid / kappa_emp if kappa_emp > 0 else 0
        out = {
            "name": name, "C": C, "p": r["p"],
            "kappa_empirical": float(kappa_emp),
            "kappa_iid_mc": float(kappa_iid),
            "plurality_share": float(plurality_share),
            "correlation_share": float(1 - plurality_share),
            "n_wrong_consensus": int(sim["n_wrong_consensus"]),
        }
        results.append(out)
        log.info(f"{name:20s} C={C:>4d} κ_emp={kappa_emp:>7.2f} κ_iid={kappa_iid:>7.2f} "
                 f"plurality={plurality_share*100:>5.0f}% correlation={(1-plurality_share)*100:>5.0f}%")

    report = {"timestamp": ts, "n_datasets": len(results), "datasets": results}
    with open(OUT_DIR / "anchoring_kappa_meta_decomp_report.json", "w") as f:
        json.dump(report, f, indent=2)

    lines = ["=" * 90, f"  κ decomposition (i.i.d.-MC plurality baseline) — {len(results)} datasets", "=" * 90]
    lines.append(f"  {'Dataset':>20} | {'C':>4} {'p':>6} | {'κ_emp':>8} {'κ_iid':>8} | "
                 f"{'plurality%':>11} {'correlation%':>13}")
    lines.append("  " + "-" * 78)
    for r in results:
        lines.append(f"  {r['name']:>20} | {r['C']:>4} {r['p']:>6.3f} | "
                     f"{r['kappa_empirical']:>8.2f} {r['kappa_iid_mc']:>8.2f} | "
                     f"{r['plurality_share']*100:>10.0f}% {r['correlation_share']*100:>12.0f}%")
    lines.append("=" * 90)
    txt = "\n".join(lines)
    with open(OUT_DIR / "anchoring_kappa_meta_decomp_report.txt", "w") as f:
        f.write(txt)
    print(); print(txt)


if __name__ == "__main__":
    main()
