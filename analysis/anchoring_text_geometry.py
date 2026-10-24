"""
analysis/anchoring_text_geometry.py — P1: answer-class text-geometry analysis
=============================================================================
Elevation-plan P1. Attacks hard-flaw #4 (no mechanism) and the unexplained
specificity of the `answer` class (Delta=+0.193 under CAPE-A, the largest
per-class agreement advantage). Question: are per-class agreement advantages
explained by the TEXT-SPACE geometry of the class prompts?

Encoder: Siglip2 text tower (google/siglip2-large-patch16-256, local HF
cache) — siglip2 is one of the paper's 5 backbones, so the text space is
in-family. (The openai text tower is unavailable locally: transformers 4.57
removed CLIP and open_clip needs network weights; extendable on A100.)

Metrics per class (over CAPE-A/B/C prompts, 9 per class):
  - prompt_drift:      mean pairwise cosine distance across all prompts
  - cross_set_drift:   mean cosine distance between prompts from different
                       CAPE sets (rephrasing sensitivity)
  - nn_confusion:      fraction of a class's prompts whose nearest
                       OTHER-class centroid is closer than the own centroid
  - cooc_proximity:    co-occurrence-weighted mean cosine similarity to the
                       centroids of classes that co-occur in GT labels
  - token_overlap:     mean Jaccard token overlap of a class's prompts with
                       other classes' prompts (non-embedding interpretable)
  - centroid_self_sim: mean cosine sim of prompts to own centroid (tightness)

Correlates (Spearman, permutation p, n=7-8 — exploratory):
  - per-class Delta (CAPE-A, from anchoring_main_report.json)
  - Delta drop across prompt sets (Delta_A - Delta_B, from P2 report)
  - per-class consensus accuracy (sanity: does text geometry predict acc?)

Output: results/anchoring_text_geometry_report.{json,txt}

Reproducibility:
  python analysis/anchoring_text_geometry.py
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, "/home/lizhuo/Work/Mayan/scb5-zeroshot")
from phase0_oracle_analysis import setup_logging, OUT_DIR, DATA_DIR  # noqa: E402
from prompts.cape_prompts import CAPE_SET_A, CAPE_SET_B, CAPE_SET_C  # noqa: E402

SUBSET = "teacher_behavior"
CLASSES = ["guide", "answer", "on-stage interaction",
           "blackboard-writing", "teacher", "stand", "screen", "blackboard"]


def load_encoder():
    """Siglip2 text tower + tokenizer from local HF cache."""
    import torch
    from transformers.models.siglip2.modeling_siglip2 import Siglip2TextModel
    from tokenizers import Tokenizer
    import glob

    cache = glob.glob(os.path.expanduser(
        "~/.cache/huggingface/hub/models--google--siglip2-large-patch16-256/"
        "snapshots/*"))[0]
    model = Siglip2TextModel.from_pretrained(
        "google/siglip2-large-patch16-256", local_files_only=True)
    model.eval()
    tok = Tokenizer.from_file(cache + "/tokenizer.json")
    return model, tok


def encode_prompts(model, tok, texts):
    import torch
    enc = tok.encode_batch(texts)
    ids = [e.ids for e in enc]
    attn = [e.attention_mask for e in enc]
    with torch.no_grad():
        out = model(input_ids=torch.tensor(ids), attention_mask=torch.tensor(attn))
    e = out.last_hidden_state[:, -1, :].numpy().astype(np.float32)
    e = e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-9)
    return e


def spearman(x, y, n_perm=20000, seed=42):
    from scipy.stats import spearmanr
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    r, p = spearmanr(x, y)
    rng = np.random.RandomState(seed)
    count = 0
    for _ in range(n_perm):
        rp = spearmanr(x, rng.permutation(y))[0]
        if abs(rp) >= abs(r):
            count += 1
    return {"rho": float(r), "p_approx": float(p),
            "p_perm": float((count + 1) / (n_perm + 1)), "n": int(len(x))}


def token_jaccard(prompts_a, prompts_b):
    a = set()
    for p in prompts_a:
        a |= set(p.lower().split())
    b = set()
    for p in prompts_b:
        b |= set(p.lower().split())
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def evaluate(bootstrap_n, seed, log):
    import torch  # noqa: F401
    log.info("loading siglip2 text tower")
    model, tok = load_encoder()

    # collect all prompts (9 per class) and embeddings
    sets = {"A": CAPE_SET_A, "B": CAPE_SET_B, "C": CAPE_SET_C}
    all_texts = []
    idx_by_class = {}
    for cls in CLASSES:
        idx_by_class[cls] = []
        for sname in ("A", "B", "C"):
            for p in sets[sname][cls]:
                idx_by_class[cls].append(len(all_texts))
                all_texts.append(p)
    emb = encode_prompts(model, tok, all_texts)
    emb_by_class = {c: emb[idx_by_class[c]] for c in CLASSES}
    centroid = {c: emb_by_class[c].mean(axis=0) for c in CLASSES}
    centroid = {c: v / (np.linalg.norm(v) + 1e-9) for c, v in centroid.items()}

    # co-occurrence counts from GT labels
    gt = np.load(DATA_DIR / f"clip_openai_{SUBSET}_labels.npy").astype(int)
    cooc = np.zeros((8, 8), dtype=float)
    for i in range(gt.shape[0]):
        pos = np.where(gt[i] > 0)[0]
        for a in pos:
            for b in pos:
                if a != b:
                    cooc[a, b] += 1
    if cooc.sum() > 0:
        cooc = cooc / cooc.sum()

    # per-class metrics
    metrics = {}
    for ci, cls in enumerate(CLASSES):
        E = emb_by_class[cls]
        n = len(E)
        sim = E @ E.T
        off = np.triu_indices(n, k=1)
        pair_sims = sim[off]
        metrics[cls] = {
            "n_prompts": n,
            "prompt_drift": float(1 - pair_sims.mean()),  # mean pairwise cos distance
            "centroid_self_sim": float((E @ centroid[cls]).mean()),
        }
        # cross-set drift
        ds = []
        set_ids = [i // 3 for i in range(n)]
        for a in range(n):
            for b in range(a + 1, n):
                if set_ids[a] != set_ids[b]:
                    ds.append(sim[a, b])
        metrics[cls]["cross_set_drift"] = float(1 - np.mean(ds))
        # nearest-other-class centroid confusion
        others = [centroid[c2] for j, c2 in enumerate(CLASSES) if j != ci]
        other_cent = np.stack(others)
        own_cent = centroid[cls]
        sims_to_own = E @ own_cent
        sims_to_others = (E @ other_cent.T).max(axis=1)
        metrics[cls]["nn_confusion"] = float((sims_to_others > sims_to_own).mean())
        # co-occurrence proximity (weighted similarity to co-occurring classes)
        w = cooc[ci]
        if w.sum() > 0:
            sims_cooc = np.array([centroid[CLASSES[j]] @ own_cent
                                  for j in range(8) if j != ci])
            ww = np.array([w[j] for j in range(8) if j != ci])
            metrics[cls]["cooc_proximity"] = float((ww * sims_cooc).sum() / ww.sum())
        else:
            metrics[cls]["cooc_proximity"] = None
        # token overlap with other classes (mean over other classes)
        ovs = [token_jaccard(sets["A"][cls] + sets["B"][cls] + sets["C"][cls],
                             sets["A"][CLASSES[j]] + sets["B"][CLASSES[j]] + sets["C"][CLASSES[j]])
               for j in range(8) if j != ci]
        metrics[cls]["token_overlap"] = float(np.mean(ovs))
        metrics[cls]["cooc_frac"] = float(w.sum())

    # target variables
    main = json.load(open(OUT_DIR / "anchoring_main_report.json"))
    tb = main["results"][0]
    delta_a = {}
    acc = {}
    for cls in CLASSES:
        cd = tb["per_class"].get(cls)
        if cd and cd.get("agreement_auroc") is not None:
            delta_a[cls] = cd["agreement_auroc"] - cd["entropy_auroc"]
            acc[cls] = cd["consensus_acc"]
    p2 = json.load(open(OUT_DIR / "anchoring_prompt_class_breakdown_report.json"))
    delta_b = {}
    for cls in CLASSES:
        cd = p2["results"]["B"]["per_class"].get(cls)
        if cd and cd.get("delta") is not None:
            delta_b[cls] = cd["delta"]

    # correlations (only over classes present in both)
    corrs = {}
    for metric in ("prompt_drift", "cross_set_drift", "nn_confusion",
                   "centroid_self_sim", "token_overlap", "cooc_proximity"):
        xs = []
        y_da = []
        y_drop = []
        y_acc = []
        for cls in CLASSES:
            if cls not in delta_a or metrics[cls][metric] is None:
                continue
            xs.append(metrics[cls][metric])
            y_da.append(delta_a[cls])
            y_acc.append(acc[cls])
            if cls in delta_b:
                y_drop.append(delta_a[cls] - delta_b[cls])
        out = {"vs_delta_A": spearman(np.array(xs), np.array(y_da))}
        out["vs_delta_drop_AB"] = spearman(np.array(xs[:len(y_drop)]),
                                           np.array(y_drop)) if len(y_drop) >= 4 else None
        out["vs_consensus_acc"] = spearman(np.array(xs), np.array(y_acc))
        corrs[metric] = out

    return {"metrics": metrics, "delta_A": delta_a, "delta_B": delta_b,
            "consensus_acc": acc, "correlations": corrs}


def format_report(res):
    lines = []
    lines.append("=" * 88)
    lines.append("  P1 — answer-class text geometry (siglip2 text tower, 9 prompts/class)")
    lines.append("=" * 88)
    lines.append("")
    lines.append(f"  {'class':>20} | {'dA':>7} | {'dB':>7} | {'drift':>6} | {'xset':>6} | "
                 f"{'nnconf':>6} | {'selfsim':>7} | {'tokov':>6} | {'coocp':>6}")
    lines.append("  " + "-" * 80)
    m = res["metrics"]
    for cls in CLASSES:
        dA = res["delta_A"].get(cls)
        dB = res["delta_B"].get(cls)
        mm = m[cls]
        lines.append(f"  {cls:>20} | "
                     f"{(f'{dA:+.3f}' if dA is not None else '  ---'):>7} | "
                     f"{(f'{dB:+.3f}' if dB is not None else '  ---'):>7} | "
                     f"{mm['prompt_drift']:>6.3f} | {mm['cross_set_drift']:>6.3f} | "
                     f"{mm['nn_confusion']:>6.3f} | {mm['centroid_self_sim']:>7.3f} | "
                     f"{mm['token_overlap']:>6.3f} | "
                     f"{(mm['cooc_proximity'] if mm['cooc_proximity'] is not None else 0):>6.3f}")
    lines.append("")
    lines.append("  Spearman correlations (permutation p, exploratory n=7-8):")
    lines.append(f"    {'metric':>18} | {'vs delta_A':>28} | {'vs delta_drop_AB':>28} | "
                 f"{'vs cons acc':>24}")
    lines.append("    " + "-" * 92)
    for metric, cs in res["correlations"].items():
        def fmt(v):
            return "---" if v is None else (f"rho={v['rho']:+.3f} p={v['p_perm']:.3f}")
        lines.append(f"    {metric:>18} | {fmt(cs['vs_delta_A']):>28} | "
                     f"{fmt(cs['vs_delta_drop_AB']):>28} | {fmt(cs['vs_consensus_acc']):>24}")
    lines.append("")
    lines.append("  Question: does text-space geometry explain answer's Delta=+0.193 and")
    lines.append("  its collapse across prompt sets?")
    lines.append("=" * 88)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap_n", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ts = time.strftime("%Y%m%d_%H%M%S")
    setup_logging(ts, name="anchoring_text_geometry")
    log = logging.getLogger()

    res = evaluate(args.bootstrap_n, args.seed, log)
    txt = format_report(res)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "anchoring_text_geometry_report.json", "w") as f:
        json.dump({"args": vars(args), "timestamp": ts, "results": res}, f, indent=2)
    with open(OUT_DIR / "anchoring_text_geometry_report.txt", "w") as f:
        f.write(txt)
    print(); print(txt)
    return 0


if __name__ == "__main__":
    exit(main())
