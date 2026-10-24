#!/usr/bin/env python3
"""Generate evidence files for claims that currently lack committed JSON traces.

Outputs:
  results/anchoring_soft_correlation.json   — soft-variant vs negative-entropy r=+0.94
  results/anchoring_borda_largec.json       — Borda on Stanford40/COCO (0.15, 0.33)
  results/anchoring_regime_correlation.json — regime-map gap-Δ ρ=0.57
  results/anchoring_error_overlap.json      — error-overlap factors (≥1.5, median 1.7)
"""

import json, os
import numpy as np
from scipy.stats import spearmanr

ROOT = os.path.join(os.path.dirname(__file__), '..', 'results')

# ── 1. Soft variant vs negative-entropy correlation (r=+0.94) ──────
def soft_correlation():
    from scipy.stats import pearsonr
    names = ['openai', 'laion', 'siglip2', 'eva02', 'dfn']
    logits = np.stack([np.load(os.path.join(ROOT, 'per_image', f'clip_{m}_teacher_behavior_logits.npy')) for m in names])
    N = logits.shape[1]; avg = logits.mean(0)
    T = 100; probs = np.exp(avg * T) / np.exp(avg * T).sum(1, keepdims=True)
    ent = -np.sum(probs * np.log(probs + 1e-12), 1)
    cons = avg.argmax(1)
    soft_alpha = np.zeros(N)
    for b in range(5):
        bb = np.exp(logits[b] * T) / np.exp(logits[b] * T).sum(1, keepdims=True)
        soft_alpha += bb[np.arange(N), cons]
    soft_alpha /= 5
    r_val, p_val = pearsonr(soft_alpha, -ent)
    report = {
        'script': __file__,
        'claim': 'soft-variant agreement correlates strongly with negative entropy',
        'paper_line': 'Exp A variant comparison (r=+0.94)',
        'pearson_r': round(float(r_val), 4),
        'p_value': float(p_val),
        'note': 'Computed per-image on TeacherBehavior (N=3240, T=100). r=+0.94 means the soft (probability-based) variant inherits entropy\'s confidence dependence — it is not an independent signal.',
    }
    with open(os.path.join(ROOT, 'anchoring_soft_correlation.json'), 'w') as f:
        json.dump(report, f, indent=1)
    print(f'  anchoring_soft_correlation.json: r={r_val:.4f}')

# ── 2. Borda count on Stanford40 and COCO ──────────────────────────
def borda_largec():
    d = json.load(open(os.path.join(ROOT, 'anchoring_continuous_variants_report.json')))
    results = {}
    for entry in d['datasets']:
        name = entry['name']
        if name in ('Stanford40', 'COCO'):
            results[name] = {
                'C': entry['C'],
                'N': entry['N'],
                'borda_auroc': entry['aurocs'].get('v5_borda'),
                'discrete_auroc': entry['aurocs'].get('discrete'),
                'entropy_auroc': entry['aurocs'].get('entropy'),
            }
    report = {
        'script': __file__,
        'claim': 'Borda count collapses on larger C datasets',
        'paper_line': 'Exp A variant comparison',
        'results': results,
    }
    with open(os.path.join(ROOT, 'anchoring_borda_largec.json'), 'w') as f:
        json.dump(report, f, indent=1)
    for name, r in results.items():
        print(f'  anchoring_borda_largec.json: {name} borda={r["borda_auroc"]:.3f} discrete={r["discrete_auroc"]:.3f}')

# ── 3. Regime map gap-Δ Spearman ρ=0.57 ───────────────────────────
def regime_correlation():
    ds = json.load(open(os.path.join(ROOT, 'anchoring_saturation_model_report.json')))
    meta = json.load(open(os.path.join(ROOT, 'anchoring_meta_report.json')))
    name_map = {'TeacherBehavior': 'teacher_behavior', 'HandriseReadWrite': 'handrise_readwrite',
                'BowTurnHead': 'bow_turnhead', 'VOC2007': 'voc2007', 'Stanford40': 'stanford40',
                'COCO': 'coco', 'CIFAR-100': 'cifar100', 'MultiTHUMOS': 'multithumos'}
    deltas = {nm: next(row['delta_hard_mean'] for row in meta['results']
                       if row['dataset'] == sn)
              for nm, sn in name_map.items()}
    gaps, dvs, nms = [], [], []
    for row in ds['datasets']:
        if row['name'] in deltas:
            gaps.append(row['gap_empirical']); dvs.append(deltas[row['name']]); nms.append(row['name'])
    rho, pv = spearmanr(gaps, dvs)
    report = {
        'script': __file__,
        'claim': 'agreement discriminative gap (Δ_alpha) correlates with empirical AUROC delta',
        'paper_line': 'Fig 9 caption',
        'spearman_rho': float(rho),
        'p_value': float(pv),
        'n': len(gaps),
        'datasets': nms,
        'gaps': gaps,
        'deltas': dvs,
    }
    with open(os.path.join(ROOT, 'anchoring_regime_correlation.json'), 'w') as f:
        json.dump(report, f, indent=1)
    print(f'  anchoring_regime_correlation.json: ρ={rho:.3f} p={pv:.3f} n={len(gaps)}')

# ── 4. Error-overlap factors (factor ≥1.5, median 1.7) ─────────────
def error_overlap():
    names = ['openai', 'laion', 'siglip2', 'eva02', 'dfn']
    labels = np.load(os.path.join(ROOT, 'per_image', 'clip_openai_teacher_behavior_labels.npy')).astype(int)
    preds = {}
    for m in names:
        L = np.load(os.path.join(ROOT, 'per_image', f'clip_{m}_teacher_behavior_logits.npy'))
        preds[m] = L.argmax(1)
    N_img = labels.shape[0]
    correct = {}
    for m in names:
        c = np.zeros(N_img, dtype=bool)
        for i in range(N_img):
            c[i] = labels[i][preds[m][i]] > 0
        correct[m] = c

    pairs = []
    for i, a in enumerate(names):
        for b in names[i+1:]:
            aw, bw = ~correct[a], ~correct[b]
            p_b_given_a = bw[aw].mean()                    # P(B wrong | A wrong)
            p_b_marginal = bw.mean()                        # P(B wrong)
            ratio = p_b_given_a / p_b_marginal               # conditional / marginal
            pairs.append({
                'pair': f'{a}+{b}',
                'P(B_wrong_given_A_wrong)': round(float(p_b_given_a), 4),
                'P(B_wrong_marginal)': round(float(p_b_marginal), 4),
                'conditional_marginal_ratio': round(float(ratio), 3),
            })
    ratios = [p['conditional_marginal_ratio'] for p in pairs]
    report = {
        'script': __file__,
        'claim': 'error-overlap analysis: CLIP-family pairs, conditional vs marginal error rates',
        'paper_line': 'Discussion deployment economics',
        'n_pairs': len(pairs),
        'min_ratio': min(ratios),
        'max_ratio': max(ratios),
        'median_ratio': np.median(ratios),
        'pairs': sorted(pairs, key=lambda x: x['conditional_marginal_ratio']),
    }
    with open(os.path.join(ROOT, 'anchoring_error_overlap.json'), 'w') as f:
        json.dump(report, f, indent=1)
    print(f'  anchoring_error_overlap.json: min={min(ratios):.2f} median={np.median(ratios):.2f} max={max(ratios):.2f}')

if __name__ == '__main__':
    soft_correlation()
    borda_largec()
    regime_correlation()
    error_overlap()
    print('\nDone → results/')
