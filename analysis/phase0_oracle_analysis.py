"""
analysis/phase0_oracle_analysis.py — Phase 0 Oracle Upper Bound (PROJECT GATE)
==============================================================================
Question: Does per-instance routing beat class-level routing by >= 5pp on
TeacherBehavior? If NOT, the InstanceAnchor project is abandoned (plan.md
Phase 0 decision criterion).

Metric (locked, per AGENTS.md design decision):
  sample-F1 = per-image F1 between predicted SET and GT set.
    F1(i) = 2*|pred_i & GT_i| / (|pred_i| + |GT_i|);  F1=0 if exactly one side
    empty; F1=1 if both empty (cannot happen: every image has >=1 GT).
  - CLIP predicted set: cosine logits thresholded (best threshold swept on the
    same validation set; this is an upper-bound analysis, in-sample threshold is
    acceptable and matches sibling find_best_threshold).
  - MLLM predicted set: pred_classes from the per-image jsonl (full-image
    inference protocol, NOT bbox-crop).

Systems compared on each subset:
  1. CLIP-only            : mean per-image F1 of the CLIP model
  2. MLLM-only            : mean per-image F1 of the MLLM
  3. Class-level oracle   : route each image by its PRIMARY GT class (first '1'
                            in gt_vector); for each primary class assign ALL its
                            images to the model with higher mean-F1 on that
                            class. This is the tightest class-level upper bound
                            (uses GT both to route and to pick the per-class
                            winner) -- the baseline InstanceAnchor must beat.
  4. Per-instance oracle  : per image, take max(F1_clip[i], F1_mllm[i]).
                            HEADLINE upper bound.

GATE: (per-instance oracle) - (class-level oracle) >= 5pp on TeacherBehavior,
      judged on the siglip2 main backbone. laion reported in appendix.

Data sources:
  results/per_image/clip_{model}_{subset}_logits.npy   (N, C) raw cosine sims
  results/per_image/clip_{model}_{subset}_labels.npy   (N, C) binary GT matrix
  results/per_image/mllm_{model}_{subset}_preds.jsonl  per-image preds

Output:
  results/phase0_oracle_report.{json,txt}
  logs/phase0_oracle_<timestamp>.log

Reproducibility:
  python analysis/phase0_oracle_analysis.py
  python analysis/phase0_oracle_analysis.py --clip_models siglip2 laion \
         --mllm_model qwen3.5_27b --bootstrap_n 10000 --seed 42
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "results" / "per_image"
OUT_DIR = REPO_ROOT / "results"
LOG_DIR = REPO_ROOT / "logs"

# Frozen subset config, copied from scb5-zeroshot/data/scb_dataset.py:57-79
# (SUBSET_CONFIG). Canonical class index order + names; .npy label matrices and
# MLLM gt_vectors use these integer indices. Do NOT change without re-extracting
# all per-image .npy (AGENTS.md rule 13).
SUBSET_CONFIG = {
    "teacher_behavior": {
        "classes": [
            "guide", "answer", "on-stage interaction",
            "blackboard-writing", "teacher", "stand", "screen", "blackboard",
        ],
        "multilabel": True,
    },
    "handrise_readwrite": {
        "classes": ["hand-raise", "read", "write"],
        "multilabel": False,
    },
    "bow_turnhead": {
        "classes": ["bow-head", "turn-head"],
        "multilabel": False,
    },
}

DEFAULT_CLIP_MODELS = ["siglip2", "laion"]   # siglip2 = main, laion = appendix
DEFAULT_MLLM = "qwen3.5_27b"                  # full coverage on TeacherBehavior
THRESHOLD_GRID = [0.10, 0.15, 0.20, 0.25, 0.30]  # matches metrics.find_best_threshold
GATE_PP = 5.0
GATE_SUBSET = "teacher_behavior"
GATE_BACKBONE = "siglip2"


def setup_logging(timestamp: str, name: str = "phase0_oracle") -> Path:
    """Configure logging to logs/<name>_<timestamp>.log + stdout. `name` must
    identify the calling script so log filenames are self-describing."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{name}_{timestamp}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )
    return log_path


def load_clip(model: str, subset: str, log: logging.Logger):
    lp = DATA_DIR / f"clip_{model}_{subset}_logits.npy"
    bp = DATA_DIR / f"clip_{model}_{subset}_labels.npy"
    if not lp.exists():
        log.warning(f"CLIP data missing: {lp}")
        return None
    logits = np.load(lp).astype(np.float32)
    labels = np.load(bp).astype(np.float32)
    return logits, labels


def load_mllm(model: str, subset: str, log: logging.Logger):
    """Return aligned arrays: indices (N,), gt (N,C), pred_sets (N,C binary)."""
    cfg = SUBSET_CONFIG[subset]
    classes = cfg["classes"]
    cidx = {c: i for i, c in enumerate(classes)}
    path = DATA_DIR / f"mllm_{model}_{subset}_preds.jsonl"
    if not path.exists():
        log.warning(f"MLLM data missing: {path}")
        return None
    recs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            recs.append(json.loads(line))
    recs.sort(key=lambda r: r["index"])
    n = len(recs)
    num_classes = len(classes)
    gt = np.zeros((n, num_classes), dtype=np.float32)
    pred = np.zeros((n, num_classes), dtype=np.float32)
    indices = np.zeros(n, dtype=np.int64)
    for row, r in enumerate(recs):
        indices[row] = r["index"]
        gv = r.get("gt_vector", [])
        for i, v in enumerate(gv):
            if 0 <= i < num_classes and v > 0:
                gt[row, i] = 1.0
        for c in r.get("pred_classes", []):
            if c in cidx:
                pred[row, cidx[c]] = 1.0
    return indices, gt, pred


def align_and_audit(clip_labels, clip_indices_full, mllm_indices, mllm_gt,
                    log: logging.Logger):
    """Align CLIP and MLLM by image index and audit that GT agrees on both sides.

    CLIP labels cover the full val set (indices 0..N-1). MLLM may cover a subset.
    We restrict to images present in BOTH. For each matched index we verify
    clip_labels[idx] == mllm_gt; a mismatch means the two pipelines labelled the
    same image differently -> abort (data integrity error).
    """
    idx_to_clip_row = {int(i): row for row, i in enumerate(clip_indices_full)}
    kept = []
    for mrow, midx in enumerate(mllm_indices):
        crow = idx_to_clip_row.get(int(midx))
        if crow is None:
            continue
        cl = clip_labels[crow]
        mg = mllm_gt[mrow]
        if not np.array_equal(cl, mg):
            log.error(
                f"GT mismatch at index {midx}: CLIP={cl.tolist()} "
                f"MLLM={mg.tolist()} -- aborting (data integrity error)."
            )
            raise RuntimeError(f"GT mismatch at image index {midx}")
        kept.append((crow, mrow))
    if not kept:
        log.error("No overlapping images between CLIP and MLLM.")
        raise RuntimeError("Empty alignment")
    clip_rows = np.array([k[0] for k in kept])
    mllm_rows = np.array([k[1] for k in kept])
    log.info(f"  aligned images: {len(kept)} (CLIP full={len(clip_labels)}, "
             f"MLLM={len(mllm_indices)})")
    return clip_rows, mllm_rows


def per_image_f1(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Per-image F1 between binary pred and gt sets. Both (N, C)."""
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = (pred & gt).sum(axis=1).astype(np.float32)
    denom = (pred.sum(axis=1) + gt.sum(axis=1)).astype(np.float32)
    f1 = np.zeros(pred.shape[0], dtype=np.float32)
    valid = denom > 0
    f1[valid] = 2.0 * inter[valid] / denom[valid]
    both_empty = (~pred.any(axis=1)) & (~gt.any(axis=1))
    f1[both_empty] = 1.0
    return f1


def best_clip_threshold(logits: np.ndarray, gt: np.ndarray, grid):
    """Sweep thresholds on min-max-normalised logits; pick best mean sample-F1."""
    lmin, lmax = logits.min(), logits.max()
    norm = (logits - lmin) / (lmax - lmin + 1e-8)
    best_t, best_f1 = grid[0], -1.0
    for t in grid:
        pred = (norm >= t).astype(np.float32)
        # guarantee at least one prediction per image (argmax fallback)
        nopred = pred.sum(axis=1) == 0
        if nopred.any():
            pred[np.where(nopred)[0], logits[np.where(nopred)[0]].argmax(axis=1)] = 1.0
        f1 = per_image_f1(pred, gt).mean()
        if f1 > best_f1:
            best_f1, best_t = float(f1), float(t)
    return best_t, best_f1


def clip_pred_set(logits: np.ndarray, threshold: float) -> np.ndarray:
    lmin, lmax = logits.min(), logits.max()
    norm = (logits - lmin) / (lmax - lmin + 1e-8)
    pred = (norm >= threshold).astype(np.float32)
    nopred = pred.sum(axis=1) == 0
    if nopred.any():
        pred[np.where(nopred)[0], logits[np.where(nopred)[0]].argmax(axis=1)] = 1.0
    return pred


def class_level_oracle(f1_clip, f1_mllm, gt):
    """Route each image by its PRIMARY GT class (first '1'); per primary class
    assign all images to the model with higher mean F1 on that class.
    Returns (system_mean_f1, per_primary_class_assignment)."""
    primary = np.array([np.where(gt[i] > 0)[0][0] for i in range(len(gt))])
    num_classes = gt.shape[1]
    assignment = {}
    system_f1 = np.zeros(len(gt), dtype=np.float32)
    for c in range(num_classes):
        mask = primary == c
        if not mask.any():
            continue
        clip_score = f1_clip[mask].mean()
        mllm_score = f1_mllm[mask].mean()
        chosen = "clip" if clip_score >= mllm_score else "mllm"
        assignment[int(c)] = {
            "chosen": chosen,
            "n_images": int(mask.sum()),
            "clip_f1": float(clip_score),
            "mllm_f1": float(mllm_score),
        }
        system_f1[mask] = f1_clip[mask] if chosen == "clip" else f1_mllm[mask]
    return float(system_f1.mean()), assignment


def bootstrap_system_f1(f1_values, n_boot, seed):
    rng = np.random.RandomState(seed)
    n = len(f1_values)
    means = np.empty(n_boot, dtype=np.float32)
    for b in range(n_boot):
        idx = rng.randint(0, n, size=n)
        means[b] = f1_values[idx].mean()
    return {
        "mean": float(f1_values.mean()),
        "ci_lo": float(np.percentile(means, 2.5)),
        "ci_hi": float(np.percentile(means, 97.5)),
        "n": int(n),
    }


def bootstrap_oracle_delta(f1_clip, f1_mllm, gt, n_boot, seed):
    """Bootstrap the per-instance-oracle minus class-level-oracle delta, since
    the two systems are coupled (same per-image F1 values) and must be
    resampled with matched indices."""
    rng = np.random.RandomState(seed)
    n = len(f1_clip)
    primary = np.array([np.where(gt[i] > 0)[0][0] for i in range(n)])
    deltas = np.empty(n_boot, dtype=np.float32)
    per_inst = np.maximum(f1_clip, f1_mllm)
    for b in range(n_boot):
        idx = rng.randint(0, n, size=n)
        fc, fm, gp = f1_clip[idx], f1_mllm[idx], primary[idx]
        pi = np.maximum(fc, fm).mean()
        # class-level on resample
        cls_f1 = np.zeros(n, dtype=np.float32)
        for c in np.unique(gp):
            m = gp == c
            csc, msc = fc[m].mean(), fm[m].mean()
            cls_f1[m] = fc[m] if csc >= msc else fm[m]
        deltas[b] = pi - cls_f1.mean()
    return {
        "delta_mean": float(deltas.mean()),
        "delta_ci_lo": float(np.percentile(deltas, 2.5)),
        "delta_ci_hi": float(np.percentile(deltas, 97.5)),
    }


def hit_at_1_clip(logits, gt):
    """CLIP Hit@1 (secondary metric): argmax in GT. Canonical definition,
    matches scb5-zeroshot/evaluation/metrics.py:compute_hit_at_k(k=1)."""
    top1 = np.argmax(logits, axis=1)
    return float(np.mean([top1[i] in np.where(gt[i] > 0)[0] for i in range(len(gt))]))


def analyse_subset(subset, clip_model, mllm_model, bootstrap_n, seed, log):
    log.info(f"=== Subset={subset} | CLIP={clip_model} | MLLM={mllm_model} ===")
    cfg = SUBSET_CONFIG[subset]

    clip = load_clip(clip_model, subset, log)
    if clip is None:
        return None
    clip_logits_full, clip_labels_full = clip
    clip_indices_full = np.arange(len(clip_logits_full))

    mllm = load_mllm(mllm_model, subset, log)
    if mllm is None:
        return None
    mllm_indices, mllm_gt_full, mllm_pred_full = mllm

    clip_rows, mllm_rows = align_and_audit(
        clip_labels_full, clip_indices_full, mllm_indices, mllm_gt_full, log
    )

    logits = clip_logits_full[clip_rows]
    gt = clip_labels_full[clip_rows]
    mllm_pred = mllm_pred_full[mllm_rows]
    n = len(logits)
    log.info(f"  evaluating on {n} aligned images (subset has {len(cfg['classes'])} classes)")

    # CLIP threshold + predicted set
    thr, _ = best_clip_threshold(logits, gt, THRESHOLD_GRID)
    clip_pred = clip_pred_set(logits, thr)
    log.info(f"  CLIP best threshold={thr}")

    f1_clip = per_image_f1(clip_pred, gt)
    f1_mllm = per_image_f1(mllm_pred, gt)

    # System F1 arrays
    per_inst_oracle_f1 = np.maximum(f1_clip, f1_mllm)
    cls_oracle_mean, cls_assign = class_level_oracle(f1_clip, f1_mllm, gt)

    # Bootstrap CIs (per_inst and cls use coupled resampling via bootstrap_oracle_delta;
    # individual systems bootstrapped independently for their own CIs)
    ci_clip = bootstrap_system_f1(f1_clip, bootstrap_n, seed)
    ci_mllm = bootstrap_system_f1(f1_mllm, bootstrap_n, seed)
    ci_perinst = bootstrap_system_f1(per_inst_oracle_f1, bootstrap_n, seed)
    # class-level system F1 array for independent bootstrap
    cls_f1_array = np.zeros(n, dtype=np.float32)
    primary = np.array([np.where(gt[i] > 0)[0][0] for i in range(n)])
    for c in range(gt.shape[1]):
        m = primary == c
        if not m.any():
            continue
        chosen = cls_assign[c]["chosen"]
        cls_f1_array[m] = f1_clip[m] if chosen == "clip" else f1_mllm[m]
    ci_cls = bootstrap_system_f1(cls_f1_array, bootstrap_n, seed)
    delta_ci = bootstrap_oracle_delta(f1_clip, f1_mllm, gt, bootstrap_n, seed)

    hit1 = hit_at_1_clip(logits, gt)

    result = {
        "subset": subset,
        "clip_model": clip_model,
        "mllm_model": mllm_model,
        "n_images": int(n),
        "clip_threshold": thr,
        "clip_hit_at_1": hit1,
        "systems": {
            "clip_only": ci_clip,
            "mllm_only": ci_mllm,
            "class_level_oracle": {**ci_cls, "per_class_assignment": cls_assign},
            "per_instance_oracle": ci_perinst,
        },
        "per_instance_minus_class_level": {
            "point": ci_perinst["mean"] - ci_cls["mean"],
            **delta_ci,
        },
    }
    log.info(f"  clip_only={ci_clip['mean']:.4f}  mllm_only={ci_mllm['mean']:.4f}  "
             f"class_oracle={ci_cls['mean']:.4f}  per_inst_oracle={ci_perinst['mean']:.4f}  "
             f"delta={ci_perinst['mean']-ci_cls['mean']:+.4f}")
    return result


def format_report(all_results, gate_result, args):
    lines = []
    lines.append("=" * 76)
    lines.append("  Phase 0 — Oracle Upper Bound Analysis (PROJECT GATE)")
    lines.append("  Metric: sample-F1 (per-image F1, predicted set vs GT set)")
    lines.append("=" * 76)
    lines.append("")
    lines.append(f"  MLLM: {args.mllm_model} | Bootstrap B={args.bootstrap_n} seed={args.seed}")
    lines.append(f"  Threshold grid: {THRESHOLD_GRID}")
    lines.append("")

    for res in all_results:
        lines.append(f"── {res['subset']} | CLIP={res['clip_model']} | "
                     f"MLLM={res['mllm_model']} | N={res['n_images']} ──")
        s = res["systems"]
        lines.append(f"  CLIP threshold (best): {res['clip_threshold']}")
        lines.append(f"  CLIP Hit@1 (secondary): {res['clip_hit_at_1']*100:.2f}%")
        for name, label in [
            ("clip_only", "CLIP-only          "),
            ("mllm_only", "MLLM-only          "),
            ("class_level_oracle", "Class-level oracle  "),
            ("per_instance_oracle", "Per-instance oracle "),
        ]:
            d = s[name]
            lines.append(f"  {label} sample-F1 = {d['mean']*100:6.2f}%   "
                         f"95% CI [{d['ci_lo']*100:.2f}, {d['ci_hi']*100:.2f}]")
        d = res["per_instance_minus_class_level"]
        lines.append(f"  Per-instance - class-level = {d['point']*100:+6.2f}pp   "
                     f"95% CI [{d['delta_ci_lo']*100:+.2f}, {d['delta_ci_hi']*100:+.2f}]")
        lines.append("")

    lines.append("=" * 76)
    lines.append("  GATE DECISION (>= %.0fpp on %s, %s backbone)" % (
        GATE_PP, GATE_SUBSET, GATE_BACKBONE))
    lines.append("=" * 76)
    if gate_result is None:
        lines.append(f"  [INCONCLUSIVE] No result for {GATE_SUBSET}/{GATE_BACKBONE}.")
    else:
        g = gate_result
        passed = g["delta_pp"] >= GATE_PP
        lines.append(f"  per-instance oracle  - class-level oracle = "
                     f"{g['delta_pp']:+.2f}pp  (CI [{g['ci_lo']:+.2f}, {g['ci_hi']:+.2f}])")
        lines.append(f"  criterion: delta >= {GATE_PP:.0f}pp   ->   "
                     f"{'PASS (project continues)' if passed else 'FAIL (ABANDON)'}")
    lines.append("")
    lines.append("  NOTE: MLLM data here is FULL-IMAGE zero-shot inference, NOT bbox-crop.")
    lines.append("  Existing pilot reports (rho=0.07 etc.) were computed on stale/different")
    lines.append("  MLLM data and must NOT be cited; this file is the new canonical source.")
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Phase 0 oracle upper bound (gate)")
    parser.add_argument("--clip_models", nargs="+", default=DEFAULT_CLIP_MODELS,
                        help="CLIP backbones (siglip2=main, laion=appendix)")
    parser.add_argument("--mllm_model", default=DEFAULT_MLLM,
                        help="MLLM model key matching jsonl filename")
    parser.add_argument("--subsets", nargs="+",
                        default=["teacher_behavior", "handrise_readwrite", "bow_turnhead"])
    parser.add_argument("--bootstrap_n", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_path = setup_logging(timestamp)
    log = logging.getLogger()
    log.info(f"Phase 0 oracle analysis started | log={log_path}")
    log.info(f"args: {vars(args)}")

    all_results = []
    for clip_model in args.clip_models:
        for subset in args.subsets:
            res = analyse_subset(subset, clip_model, args.mllm_model,
                                 args.bootstrap_n, args.seed, log)
            if res is not None:
                all_results.append(res)

    if not all_results:
        log.error("No results produced. Check data files.")
        return 1

    # Gate decision
    gate_res = next((r for r in all_results
                     if r["subset"] == GATE_SUBSET and r["clip_model"] == GATE_BACKBONE), None)
    gate_payload = None
    if gate_res is not None:
        d = gate_res["per_instance_minus_class_level"]
        gate_payload = {
            "delta_pp": d["point"] * 100,
            "ci_lo": d["delta_ci_lo"] * 100,
            "ci_hi": d["delta_ci_hi"] * 100,
            "passed": bool(d["point"] * 100 >= GATE_PP),
        }

    report_txt = format_report(all_results, gate_payload, args)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUT_DIR / "phase0_oracle_report.json"
    with open(json_path, "w") as f:
        json.dump({
            "args": vars(args),
            "timestamp": timestamp,
            "gate": gate_payload,
            "results": all_results,
            "config": {
                "threshold_grid": THRESHOLD_GRID,
                "gate_pp": GATE_PP,
                "gate_subset": GATE_SUBSET,
                "gate_backbone": GATE_BACKBONE,
                "metric": "sample-F1 (per-image F1, predicted set vs GT set)",
                "mllm_protocol": "full-image zero-shot (NOT bbox-crop)",
            },
        }, f, indent=2)
    txt_path = OUT_DIR / "phase0_oracle_report.txt"
    with open(txt_path, "w") as f:
        f.write(report_txt)

    log.info(f"JSON: {json_path}")
    log.info(f"TXT : {txt_path}")
    print()
    print(report_txt)
    return 0


if __name__ == "__main__":
    exit(main())
