#!/usr/bin/env python3
"""Generate ALL paper figures from evidence JSONs → paper/figures/*.pdf
Strict visual consistency: 4.5×3.0 (single), 7.5×3.0 (dual), 7.5×5.5 (quad);
schematics (matplotlib-drawn): 7.5×4.33 (method overview), 7.5×3.17 (mechanism contrast).
"""

import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

OUT = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(OUT, exist_ok=True)
ROOT = os.path.join(os.path.dirname(__file__), '..', 'results')

# ── strict style ───────────────────────────────────────────────────
SZ_S = (4.5, 3.0)   # single panel
SZ_D = (7.5, 3.0)   # dual (1×2)
SZ_Q = (7.5, 5.5)   # quad (2×2)
TEXTW = 6.93        # journal text width (in) used for font-size budgeting

# Okabe-Ito-derived muted palette (top-journal stance: low-saturation
# families; grayscale separability preserved — blue L=0.152 vs rust
# L=0.224, gap 0.072 ≈ the original 0.069; hatches remain as the second
# channel on bars).
C = {
    'blue':  '#4A7094', 'red':   '#B9704C', 'green': '#4E8D7C',
    'teal':  '#467B85', 'orange':'#B98A3E', 'violet':'#7E6B8F',
    'gray':  '#767676', 'light': '#B0B0B0', 'dark':  '#272727',
    'gold':  '#B89B45',
}
PAL_LINE = [C['blue'], C['red'], C['teal'], C['orange'], C['violet'], C['green']]

_SCALE_PHYS = 1.0   # printed-in / figure-in; set by _rc()

def FS(pt):
    """Font size in code points that prints at `pt` physical points,
    after scaling the figure to its final printed width."""
    return pt / _SCALE_PHYS

def _rc(fig_w, print_frac):
    """Reset style; budget font sizes so text prints at a fixed physical
    size regardless of final on-page width (top-journal requirement)."""
    global _SCALE_PHYS
    _SCALE_PHYS = (TEXTW * print_frac) / fig_w
    s = _SCALE_PHYS
    plt.rcdefaults()
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Liberation Serif', 'Nimbus Roman', 'DejaVu Serif'],
        'font.size': 9 / s,
        'axes.labelsize': 9 / s, 'axes.titlesize': 9 / s,
        'xtick.labelsize': 8 / s, 'ytick.labelsize': 8 / s,
        'legend.fontsize': 8 / s,
        'mathtext.fontset': 'stix',
        'axes.spines.right': False, 'axes.spines.top': False,
        'axes.linewidth': 0.8,
        'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
        'legend.frameon': False, 'legend.labelspacing': 0.2,
        'pdf.fonttype': 42, 'svg.fonttype': 'none',
        'figure.dpi': 300,
        # 0.05in pad: mathtext axis-label bboxes (rotated ylabels) are
        # underestimated by the tight-bbox pass; 0.03in left 2-3pt glyphs
        # (fig2 'Expected α', fig8 'p') clipped at the page edge.
        'savefig.bbox': 'tight', 'savefig.pad_inches': 0.07,
    })

def save(fig, name):
    fig.savefig(os.path.join(OUT, f'{name}.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(OUT, f'{name}.svg'), bbox_inches='tight')
    plt.close(fig)
    print(f'  {name}.pdf (+svg)')

def load(key):
    with open(os.path.join(ROOT, key)) as f:
        return json.load(f)

# ── shared line-plot helper ─────────────────────────────────────────
def line_plot(ax, covs, vals, color, label, linestyle='-', marker='', ms=0):
    kw = dict(color=color, linestyle=linestyle, linewidth=1.2, label=label)
    if marker:
        kw['marker'] = marker; kw['markersize'] = ms
        kw['markerfacecolor'] = 'white'; kw['markeredgewidth'] = 0.6
    ax.plot(covs, vals, **kw)

def finish(ax, xlabel='Coverage', ylabel='', legend=True, xlim=None, ylim=None):
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    if legend: ax.legend(fontsize=FS(8), handlelength=1.8)
    if xlim: ax.set_xlim(*xlim)
    if ylim: ax.set_ylim(*ylim)
    ax.tick_params(width=0.6, length=3)
    for spine in ['left','bottom']:
        ax.spines[spine].set_linewidth(0.6)

# ══════════════════════════════════════════════════════════════════════
# FIG 1: Risk-coverage — TeacherBehavior  [single]  SZ_S
# ══════════════════════════════════════════════════════════════════════
def fig1_risk_coverage():
    _rc(4.5, 0.48)
    tb = [r for r in load('anchoring_main_report.json')['results'] if r['subset']=='teacher_behavior'][0]
    fig, ax = plt.subplots(figsize=SZ_S)
    cis = load('anchoring_figure_cis_report.json')
    agr = cis['risk_tieaware']['teacher_behavior']['agreement']
    covs_a = [x['coverage'] for x in agr]
    mean_a = [x['mean_acc'] for x in agr]
    ax.fill_between(covs_a, [x['lo'] for x in agr], [x['hi'] for x in agr],
                    color=C['blue'], alpha=0.15, lw=0)
    line_plot(ax, covs_a, mean_a, C['blue'], 'Agreement (tie-aware)', '-', '', 0)
    rc = tb['signals']['entropy']['risk_coverage']
    line_plot(ax, [x['coverage'] for x in rc], [x['accuracy'] for x in rc], C['red'], 'Entropy', '--', '', 0)
    rc = tb['signals']['margin']['risk_coverage']
    line_plot(ax, [x['coverage'] for x in rc], [x['accuracy'] for x in rc], C['teal'], 'Margin', ':', '', 0)
    ax.axhline(tb['consensus_accuracy'], color=C['gray'], linestyle=':', linewidth=0.8, label=f'Consensus ({tb["consensus_accuracy"]:.2f})')
    dv = tb['deltas_vs_entropy']['agreement_top1_minus_entropy']
    ax.text(0.02, 0.045,
            r'$\Delta$AUROC $=$ %+.3f, CI [%+.3f, %+.3f]' % (dv['mean'], dv['ci_lo'], dv['ci_hi']),
            transform=ax.transAxes, fontsize=FS(7.5))
    finish(ax, 'Coverage', 'Selective Accuracy', ylim=(0.60,1.0), xlim=(0,1))
    fig.tight_layout(); save(fig, 'fig3_riskcoverage')

# ══════════════════════════════════════════════════════════════════════
# FIG 2: Per-class AUROC — TeacherBehavior  [single]  SZ_S
# ══════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════
# FIG 2: Per-class breakdown (merged as panel of FIG 4 mechanism)
# ══════════════════════════════════════════════════════════════════════
def fig2_per_class_panel(ax):
    tb = [r for r in load('anchoring_main_report.json')['results'] if r['subset']=='teacher_behavior'][0]
    pc_cis = {r['class']: r for r in load('anchoring_figure_cis_report.json')['per_class']}
    pc = tb['per_class']
    classes = sorted(pc.keys(), key=lambda c: pc[c]['consensus_acc'])
    names = [c.replace('-','\n').replace('on\nstage','on-stage') for c in classes]
    x = np.arange(len(classes)); w = 0.32
    ent_vals = [pc[c]['entropy_auroc'] for c in classes]
    agr_vals = [pc[c]['agreement_auroc'] for c in classes]
    acc_vals = [pc[c]['consensus_acc'] for c in classes]
    valid = [i for i in range(len(classes)) if ent_vals[i] is not None]
    x, names = x[valid], [names[i] for i in valid]
    ent_vals = [ent_vals[i] for i in valid]; agr_vals = [agr_vals[i] for i in valid]; acc_vals = [acc_vals[i] for i in valid]
    classes = [classes[i] for i in valid]
    ent_lo = [pc_cis[c]['entropy_ci'][0] if c in pc_cis and 'entropy_ci' in pc_cis[c] else np.nan for c in classes]
    ent_hi = [pc_cis[c]['entropy_ci'][1] if c in pc_cis and 'entropy_ci' in pc_cis[c] else np.nan for c in classes]
    agr_lo = [pc_cis[c]['agreement_ci'][0] if c in pc_cis and 'agreement_ci' in pc_cis[c] else np.nan for c in classes]
    agr_hi = [pc_cis[c]['agreement_ci'][1] if c in pc_cis and 'agreement_ci' in pc_cis[c] else np.nan for c in classes]
    ent_vals = np.array(ent_vals); agr_vals = np.array(agr_vals)

    # dot+interval, not bars: AUROC is an effect size; the previous truncated
    # bar axis (ylim 0.35) exaggerated per-class differences. Whiskers are
    # drawn only where the evidence file carries a CI (screen: none reported).
    off = 0.17
    for i in range(len(x)):
        for v, lo, hi, c, mk, lb in [(ent_vals[i], ent_lo[i], ent_hi[i], C['red'], 's', 'Entropy'),
                                     (agr_vals[i], agr_lo[i], agr_hi[i], C['blue'], 'o', 'Agreement')]:
            xi = x[i] + (-off if mk == 's' else off)
            if np.isfinite(lo) and np.isfinite(hi):
                ax.errorbar(xi, v, yerr=[[v - lo], [hi - v]], fmt='none', ecolor=c,
                            elinewidth=0.6, capsize=1.8, capthick=0.5, zorder=4)
            ax.plot(xi, v, mk, color=c, markerfacecolor='white', markeredgewidth=0.6,
                    markersize=3.5, zorder=5, label=lb if i == 0 else None)

    ax.scatter(x, acc_vals, marker='D', s=18, color=C['dark'], zorder=5, label='Base Acc.', edgecolors='white', linewidth=0.2)
    # single-line names + 90° rotation (2-line labels at 55° still left
    # two neighbour collisions per span-bbox audit). NOTE: the replace MUST
    # run before set_xticklabels — it was previously dead code placed after,
    # so 'blackboard-writing' rendered as two stacked lines (fixed 2026-10-22).
    names = [n.replace('\n','-') for n in names]
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=FS(6.5))
    ax.tick_params(axis='x', rotation=90, labelrotation=90)
    for lbl in ax.get_xticklabels(): lbl.set_ha('left')
    ax.axhline(0.5, color=C['light'], linestyle=':', linewidth=0.5)
    from matplotlib.lines import Line2D
    handles = [Line2D([0],[0], marker='s', color='w', markerfacecolor=C['red'], markersize=4.5, label='Entropy'),
               Line2D([0],[0], marker='o', color='w', markerfacecolor=C['blue'], markersize=4.5, label='Agreement'),
               Line2D([0],[0], marker='D', color='w', markerfacecolor=C['dark'], markersize=4,
                      markeredgecolor='white', label='Base Acc.')]
    ax.legend(handles=handles, fontsize=FS(8), loc='upper left', frameon=False)
    finish(ax, '', 'AUROC / Accuracy', ylim=(0.35, 1.0), legend=False)

# ══════════════════════════════════════════════════════════════════════
# FIG 6: Cost scaling  [single]  SZ_S
# ══════════════════════════════════════════════════════════════════════
def fig3_cost():
    _rc(4.5, 0.85)
    tb = [r for r in load('anchoring_cost_report.json')['results'] if r['subset']=='teacher_behavior'][0]
    ks, ent, agr = [], [], []
    for k in sorted(tb['by_k'].keys(), key=lambda x: int(x.split('=')[1])):
        v = tb['by_k'][k]; ks.append(v['k']); ent.append(v['ensemble_entropy_auroc']); agr.append(v['agreement_auroc'] or np.nan)
    fig, ax = plt.subplots(figsize=SZ_S)
    ck = load('anchoring_figure_cis_report.json')['cost_by_k']
    ent = np.array(ent); agr = np.array(agr)
    e_lo = np.array([ck[f'k={k}']['entropy_ci'][0] for k in ks])
    e_hi = np.array([ck[f'k={k}']['entropy_ci'][1] for k in ks])
    mask_e = np.array(e_hi) > np.array(e_lo)
    ax.errorbar(np.array(ks)[mask_e], ent[mask_e],
                yerr=[(ent - np.array(e_lo))[mask_e], (np.array(e_hi) - ent)[mask_e]],
                fmt='none', ecolor=C['red'], elinewidth=0.8, capsize=2.5, zorder=3)
    a_lo = np.array([ck[f'k={k}']['agreement_ci'][0] if 'agreement_ci' in ck[f'k={k}'] else np.nan for k in ks])
    a_hi = np.array([ck[f'k={k}']['agreement_ci'][1] if 'agreement_ci' in ck[f'k={k}'] else np.nan for k in ks])
    mask_a = np.array(a_hi) > np.array(a_lo)
    ax.errorbar(np.array(ks)[mask_a], agr[mask_a],
                yerr=[(agr - np.array(a_lo))[mask_a], (np.array(a_hi) - agr)[mask_a]],
                fmt='none', ecolor=C['blue'], elinewidth=0.8, capsize=2.5, zorder=3)
    ax.plot(ks, ent, 's-', color=C['red'], label='Entropy', linewidth=1.2, markersize=4, markerfacecolor='white', markeredgewidth=0.6)
    ax.plot(ks, agr, 'o-', color=C['blue'], label='Agreement', linewidth=1.2, markersize=4, markerfacecolor='white', markeredgewidth=0.6)
    ax.set_xticks(ks)
    # ylim floor 0.45: the k=1 entropy 95% CI (lo=0.488) was clipped by a
    # 0.50 floor — error bars must not be cut by the axis.
    finish(ax, 'Number of Backbones (k)', 'AUROC', ylim=(0.45, 0.80))
    fig.tight_layout(); save(fig, 'fig6_cost')

# ══════════════════════════════════════════════════════════════════════
# FIG 4: Mechanism (per-class + low/high anchor)  [triple 1×3]  (7.5×2.55)
# ══════════════════════════════════════════════════════════════════════
def fig4_mechanism():
    _rc(7.5, 0.99)
    tb = [r for r in load('anchoring_mechanism_report.json')['results'] if r['subset']=='teacher_behavior'][0]
    al = tb['B2_anchor_level']
    labels = ['Low-Anchor', 'High-Anchor']
    ent = [al['low_anchor(primary)']['entropy_auroc'], al['high_anchor(primary)']['entropy_auroc']]
    agr = [al['low_anchor(primary)']['agreement_auroc'], al['high_anchor(primary)']['agreement_auroc']]
    delta = [al['low_anchor(primary)']['agree_minus_ent'], al['high_anchor(primary)']['agree_minus_ent']]
    x = np.arange(len(labels)); w = 0.32

    anch = load('anchoring_figure_cis_report.json')['anchor_level']
    def ye(kind, keys):
        if kind == 'delta':
            vals = np.array(delta, dtype=float)
            cis_ = np.array([[anch[k]['delta_ci'][0], anch[k]['delta_ci'][1]]
                             for k in keys])
        else:
            vals = np.array([anch[k][f'{kind}_auroc'] for k in keys], dtype=float)
            cis_ = np.array([[anch[k][f'{kind}_ci'][0], anch[k][f'{kind}_ci'][1]]
                             for k in keys])
        return [vals - cis_[:, 0], cis_[:, 1] - vals]
    yerr_ent = ye('entropy', ['low_anchor', 'high_anchor'])
    yerr_agr = ye('agreement', ['low_anchor', 'high_anchor'])
    yerr_del = ye('delta', ['low_anchor', 'high_anchor'])
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(7.5, 2.55),
                                        gridspec_kw={'width_ratios':[0.9, 0.7, 1.4]})
    # dot+interval, not bars: AUROC is an effect size; a truncated bar axis
    # (ylim 0.40) exaggerates level differences and a zero-based axis
    # compresses them. Point estimates with 95% CI whiskers are the
    # journal-standard display for effect sizes with CIs.
    off = 0.16
    for i, xi in enumerate(x):
        for v, ye, c, mk, lb in [(ent[i], (yerr_ent[0][i], yerr_ent[1][i]), C['red'], 's', 'Entropy'),
                                 (agr[i], (yerr_agr[0][i], yerr_agr[1][i]), C['blue'], 'o', 'Agreement')]:
            ax1.errorbar(xi + (-off if mk == 's' else off), v, yerr=[[ye[0]], [ye[1]]],
                         fmt=mk, color=c, markerfacecolor='white', markeredgewidth=0.8,
                         markersize=4.5, elinewidth=0.9, capsize=2.5, capthick=0.8,
                         label=lb if i == 0 else None, zorder=5)
    ax1.set_xticks(x); ax1.set_xticklabels(labels, fontsize=FS(7))
    finish(ax1, '', 'AUROC', ylim=(0.40, 0.85))
    for spine in ['left','bottom']: ax2.spines[spine].set_linewidth(0.6)
    ax2.spines['right'].set_visible(False); ax2.spines['top'].set_visible(False)
    colors_d = [C['blue'] if d>0 else C['red'] for d in delta]
    # delta bars keep a meaningful zero baseline (difference measure), so
    # bars are legitimate here; only the level panels (a)/(c) moved to dots.
    ax2.bar(x, delta, w, color=colors_d, hatch='\\\\\\', edgecolor='white',
            yerr=yerr_del, error_kw={'elinewidth':0.6, 'capsize':2},
            linewidth=0.3, alpha=0.92)
    ax2.axhline(0, color=C['dark'], linewidth=0.5)
    ax2.set_xticks(x); ax2.set_xticklabels(labels, fontsize=FS(7))
    ax2.set_ylabel(r'$\Delta$ AUROC (Agr. $-$ Ent.)')
    ax2.tick_params(width=0.6, length=3)
    fig2_per_class_panel(ax3)
    for axx, letter in [(ax1, 'a'), (ax2, 'b'), (ax3, 'c')]:
        axx.text(0.02, 0.95, f'({letter})', transform=axx.transAxes, fontsize=FS(10),
                 fontweight='bold', va='top')
    fig.tight_layout(); save(fig, 'fig4_mechanism')

# ══════════════════════════════════════════════════════════════════════
# FIG 5: Instance vs class  [single]  SZ_S
# ══════════════════════════════════════════════════════════════════════
def fig5_instance_vs_class():
    _rc(4.5, 0.85)
    tb = [r for r in load('anchoring_class_vs_instance_report.json')['results'] if r['subset']=='teacher_behavior'][0]
    c4 = tb['C4_fair_selective']; covs = c4['coverages']
    methods = [('Instance Agree.','instance',C['blue'],'-','o',3.5),
               ('Class Mean (held-out)','classmean_heldout',C['red'],'--','s',3.5),
               ('VA AnchorScore','va_anchorscore',C['orange'],':','^',3.5)]
    fig, ax = plt.subplots(figsize=SZ_S)
    for lb, key, cl, ls, mk, ms in methods:
        vals = [c4['selective_accuracy'][str(c)][key] for c in covs]
        line_plot(ax, covs, vals, cl, lb, ls, mk, ms)
    finish(ax, 'Coverage', 'Selective Accuracy', xlim=(0.25,0.95))
    fig.tight_layout(); save(fig, 'fig5_classvsinstance')

# ══════════════════════════════════════════════════════════════════════
# FIG 6: Generalization  [quad 2×2]  SZ_Q
# ══════════════════════════════════════════════════════════════════════
def fig7_generalization():
    _rc(7.5, 1.0)
    cis = load('anchoring_figure_cis_report.json')
    am = load('anchoring_main_report.json')
    v2 = load('anchoring_v2_report.json')
    cf = load('anchoring_cifar_report.json')
    fig, axes = plt.subplots(2, 2, figsize=SZ_Q)
    datasets = [
        ('TeacherBehavior', am, 'teacher_behavior', 'subset', 'TB'),
        ('VOC2007', v2, 'voc2007', 'dataset', 'VOC2007'),
        ('Stanford40', v2, 'stanford40', 'dataset', 'Stanford40'),
        ('CIFAR-100', cf, None, 'results', 'CIFAR-100'),
    ]
    letters = 'abcd'
    for i, (ax, (label, report, key, kf, abbr)) in enumerate(zip(axes.flat, datasets)):
        letter = letters[i]
        res = report['results'] if kf=='results' else [r for r in report['results'] if r[kf]==key][0]
        sigs = res['signals']; d_info = res.get('deltas_vs_entropy',{}).get('agreement_top1_minus_entropy',{})
        dv = d_info.get('mean',float('nan')) if isinstance(d_info,dict) else float('nan')
        tk = cis['risk_tieaware'].get(key if key else 'cifar100')
        if 'risk_coverage' not in sigs['entropy'] and not tk:
            ax.text(0.5,0.5, f'{abbr}\nAgr={sigs["agreement_top1"]["auroc"]:.3f}\nEnt={sigs["entropy"]["auroc"]:.3f}\nΔ={dv:+.3f}',
                    ha='center',va='center',fontsize=FS(8),transform=ax.transAxes)
            ax.text(0.02, 0.95, f'({letter})', transform=ax.transAxes, fontsize=FS(10),
                    fontweight='bold', va='top')
            continue
        if 'risk_coverage' in sigs['entropy']:
            covs = [r['coverage'] for r in sigs['entropy']['risk_coverage']]
            er = [r['risk'] for r in sigs['entropy']['risk_coverage']]
        else:
            ep = tk['entropy']
            covs = [x['coverage'] for x in ep]
            er = [1 - x['accuracy'] for x in ep]
        ax.plot(covs, er, '--', color=C['red'], label='Entropy', linewidth=1.1)
        ag = cis['risk_tieaware'].get(key if key else 'cifar100', {}).get('agreement')
        if ag:
            acov = [x['coverage'] for x in ag]
            ax.fill_between(acov, [1 - x['hi'] for x in ag],
                            [1 - x['lo'] for x in ag],
                            color=C['blue'], alpha=0.15, lw=0)
            ax.plot(acov, [1 - x['mean_acc'] for x in ag], '-',
                    color=C['blue'], label='Agreement (tie-aware)', linewidth=1.1)
        else:
            ar = [r['risk'] for r in sigs['agreement_top1']['risk_coverage']]
            ax.plot(covs, ar, '-', color=C['blue'], label='Agreement', linewidth=1.1)
        ax.text(0.02, 0.95, f'({letter})', transform=ax.transAxes, fontsize=FS(10),
                fontweight='bold', va='top')
        ax.set_xlabel('Coverage', fontsize=FS(8)); ax.set_ylabel('Risk', fontsize=FS(8))
        ax.legend(fontsize=FS(7.5), loc='upper right', handlelength=1.5)
        ax.set_xlim(0,1.05)
        ax.spines['right'].set_visible(False); ax.spines['top'].set_visible(False)
        ax.spines['left'].set_linewidth(0.6); ax.spines['bottom'].set_linewidth(0.6)
        ax.tick_params(width=0.6, length=3)
    fig.tight_layout(); save(fig, 'fig7_generalization')

# ══════════════════════════════════════════════════════════════════════
# FIG 8: Saturation model  [dual 1×2]  SZ_D
# ══════════════════════════════════════════════════════════════════════
def fig8_saturation_model():
    _rc(7.5, 0.85)
    Cv, p = 8, np.linspace(0.05,0.95,500)
    kv = [1, 10.3, 50]
    ck = [C['light'], C['blue'], C['red']]
    ls = ['-', '--', ':']
    lbl = ['i.i.d. baseline', 'TB (κ=10.3)', 'saturated (κ=50)']
    fig, (ax1,ax2) = plt.subplots(1,2,figsize=SZ_D)
    # Panel (a)
    ax1.plot(p, p, 'k-', linewidth=1.2, label=r'$\mathbb{E}[\alpha\!\mid\!\mathrm{correct}]\approx p$')
    for k, c, sty, lb in zip(kv, ck, ls, lbl):
        ew = np.clip(k*(1-p)/(Cv-1), 0, 1)
        ax1.plot(p, ew, color=c, linestyle=sty, linewidth=1.2,
                 label=rf'$\mathbb{{E}}[\alpha\mid\mathrm{{wrong}}]$, {lb}')
    # TB operating point on panel (a)
    ax1.plot(0.67, 0.67, 'ko', ms=6, zorder=10, label='TB operating point (p=0.67, κ=10.3)')
    ax1.plot(0.67, 0.49, 'ko', ms=6, zorder=10)
    # Arrow connecting correct/wrong at TB point
    ax1.annotate('', xy=(0.67, 0.49), xytext=(0.67, 0.67),
                 arrowprops=dict(arrowstyle='->', color='gray', lw=1.2, ls='-'))
    ax1.text(0.68, 0.58, r'$\Delta_\alpha$', fontsize=FS(8), color='gray', rotation=90, va='center')
    finish(ax1, r'$p$', r'Expected $\alpha$ (C=8)', ylim=(0,1.02))
    ax1.set_xlim(0,1); ax1.legend(fontsize=FS(6.5), loc='upper left', labelspacing=0.5, ncol=1, framealpha=0.9)
    ax1.text(0.02, 0.95, '(a)', transform=ax1.transAxes, fontsize=FS(10),
             fontweight='bold', va='top')
    # Panel (b)
    for k, c, sty, lb in zip(kv, ck, ls, lbl):
        gap = p - k*(1-p)/(Cv-1)
        ax2.plot(p, gap, color=c, linestyle=sty, linewidth=1.2, label=lb)
    ax2.axhline(0, color=C['dark'], linestyle=':', linewidth=0.6)
    # TB operating point on panel (b)
    ax2.plot(0.67, 0.1835, 'ko', ms=6, zorder=10)
    ax2.annotate('TB: Δα=+0.18', (0.67, 0.1835), fontsize=FS(7), color='black',
                 xytext=(0.67, 0.35), textcoords='data',
                 ha='center', va='bottom',
                 arrowprops=dict(arrowstyle='->', color='black', lw=0.8))
    # Saturation regions with labels
    for k, c in zip(kv, ck):
        ps = k/(Cv-1+k)
        ax2.axvspan(0, ps, color=c, alpha=0.10)
    ax2.text(0.10, -0.20, 'Saturated\n(p < p*)', fontsize=FS(7), color='gray',
             ha='center', va='top', fontstyle='italic')
    # p* markers (plain-text star — see glyph-level comment below)
    for k, c in zip(kv, ck):
        ps = k/(Cv-1+k)
        ax2.axvline(ps, color=c, linestyle=':', linewidth=0.6, alpha=0.6)
        # plain-text 'p*': mathtext's star operator (U+22C6, STIXNonUnicode)
        # has a descender spike that genuinely overlaps the subscript digits
        # in p^\star_{10.3} (verified at glyph level via bbox audit)
        ax2.annotate('p* (κ=%g)'%k, (ps, 0.06), fontsize=FS(7), color=c,
                     ha='center', va='bottom', fontstyle='italic')
    ax2.set_ylim(-0.3, 1.0)
    ax2.legend(fontsize=FS(7), loc='upper left', labelspacing=0.4, ncol=1, framealpha=0.9)
    ax2.text(0.02, 0.95, '(b)', transform=ax2.transAxes, fontsize=FS(10),
             fontweight='bold', va='top')
    finish(ax2, r'$p$', r'$\Delta_\alpha$', legend=False)
    ax2.set_xlim(0,1)
    fig.tight_layout(); save(fig, 'fig2_saturation')

# ══════════════════════════════════════════════════════════════════════
# FIG 9: Phase diagram  [dual 1×2]  SZ_D
# ══════════════════════════════════════════════════════════════════════
def fig9_phase_diagram():
    _rc(7.5, 0.95)
    # P4: read from 20-dataset meta report (was 8-dataset loocv boundary)
    d = json.load(open(os.path.join(ROOT, 'anchoring_meta_report.json')))
    datasets = [dict(name=r['dataset'], C=r['C'], p=r['p'], kappa=r['kappa'],
                     gap=r['Delta_alpha'], delta=r['delta_hard_mean'],
                     delta_lo=r['delta_hard_lo'], delta_hi=r['delta_hard_hi'])
            for r in d['results']
            if np.isfinite(r['kappa']) and np.isfinite(r['Delta_alpha'])]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=SZ_D)
    mk_c = {'win':('o',C['blue']),'tie':('s',C['gray']),'loss':('^',C['red'])}

    # Deterministic label offsets (points, +x right / +y up) for the known
    # collision clusters — replaces the random walk for these datasets:
    #   (a) IN-C/Sketch neighbours; the ImageNet-V2 trio (κ=785/916/1012)
    #   (b) IN-R (rightmost, hit the panel letter); the V2 trio cluster
    # offsets solved geometrically from the actual axis transform (see audit
    # log 2026-10-22); random offsets repeatedly collided in the dense
    # κ∈[580,1010] cluster.
    # 2026-10-22 independent re-solve: all 16 labelled datasets placed at 6pt
    # (bumped from 5.5pt to meet the 6pt journal floor) via a greedy
    # declutter over the exact matplotlib transforms; 0 label-label,
    # label-point, label-legend and label-axes collisions.
    OFF_A = {'cifar10':(14,0), 'cifar100c':(8,0), 'cifar10c':(0,-14), 'eurosat':(8,0),
             'food101':(8,0), 'handrise_readwrite':(8,0), 'imagenet_a':(8,0),
             'imagenet_c':(0,-14), 'imagenet_r':(-6,14), 'imagenet_sketch':(8,-4),
             'imagenet_v2_matched':(8,-2), 'imagenet_v2_threshold':(4,16),
             'imagenet_v2_top':(0,18), 'multithumos':(-18,0),
             'stanford_cars':(8,2), 'teacher_behavior':(8,0)}
    OFF_B = {'cifar10':(8,0), 'cifar100c':(8,0), 'cifar10c':(-6,-14), 'eurosat':(8,0),
             'food101':(0,-22), 'handrise_readwrite':(8,0), 'imagenet_a':(0,10),
             'imagenet_c':(0,26), 'imagenet_r':(-8,-16), 'imagenet_sketch':(0,-18),
             'imagenet_v2_matched':(0,-22), 'imagenet_v2_threshold':(8,8),
             'imagenet_v2_top':(2,20), 'multithumos':(0,10),
             'stanford_cars':(0,-16), 'teacher_behavior':(8,0)}

    # (a) regime scatter — 27 points; label only hard-win + representative others
    label_set = {'teacher_behavior', 'imagenet_a', 'imagenet_r', 'imagenet_c',
                 'imagenet_sketch', 'imagenet_v2_matched', 'imagenet_v2_threshold',
                 'imagenet_v2_top', 'handrise_readwrite',
                 'cifar10', 'cifar10c', 'cifar100c',
                 'food101', 'eurosat', 'multithumos', 'stanford_cars'}
    short = {'teacher_behavior':'TB', 'handrise_readwrite':'HRW', 'bow_turnhead':'BTH',
             'multithumos':'MT', 'tiny_imagenet':'Tiny-IN', 'imagenet_a':'IN-A',
             'imagenet_r':'IN-R', 'imagenet_c':'IN-C', 'imagenet_sketch':'Sketch',
             'imagenet_v2_matched':'V2-m', 'imagenet_v2_threshold':'V2-t',
             'imagenet_v2_top':'V2-top', 'stanford_cars':'Cars', 'stanford40':'Stan40',
             'voc2007':'VOC', 'coco':'COCO', 'cifar100':'C100', 'cifar10':'C10',
             'cifar10c':'C10-C', 'cifar100c':'C100-C',
             'food101':'Food', 'eurosat':'Euro', 'fgvc_aircraft':'FGVC',
             'gtsrb':'GTSRB', 'dtd':'DTD', 'sun397':'SUN', 'stl10':'STL10'}
    import random; random.seed(42)
    for ds in datasets:
        oc = 'win' if ds['delta_lo']>0 else ('loss' if ds['delta_hi']<0 else 'tie')
        mk,co = mk_c[oc]
        ax1.scatter(ds['kappa'], ds['p'], c=co, marker=mk, s=30, edgecolors='white', linewidth=0.4, zorder=5)
        nm = short.get(ds['name'], ds['name'][:6])
        if ds['name'] in label_set or oc == 'win':
            ox, oy = OFF_A.get(ds['name'], (random.randint(-6,8), random.randint(-7,7)))
            ax1.annotate(nm, (ds['kappa'],ds['p']), textcoords="offset points", xytext=(ox,oy),
                        fontsize=FS(6), color=co, fontweight='bold' if oc=='win' else 'normal')
    # curve spans the full axis (was 700 — left a gap after the xlim fix)
    kr = np.logspace(np.log10(2), np.log10(2000), 500)
    for Cv, ls in [(3,'--'),(8,'-'),(20,':'),(100,'-.'),(200,'dotted')]:
        ax1.plot(kr, kr/(Cv-1+kr), linestyle=ls, color=C['gray'], linewidth=0.6)
    ax1.set_xscale('log')
    # xlim upper 2000: the ImageNet-V2 trio has κ=785/916/1012 — the old
    # 700 cap silently clipped 3 of the 27 points out of panel (a).
    finish(ax1, r'$\kappa$ (log)', r'$p$', legend=False, xlim=(2,2000), ylim=(0.1,0.85))
    ax1.text(0.02, 0.95, '(a)', transform=ax1.transAxes, fontsize=FS(10),
             fontweight='bold', va='top')
    from matplotlib.lines import Line2D
    l1 = [Line2D([0],[0], marker='o',color='w',markerfacecolor=C['blue'],markersize=5,label='Agr. wins'),
          Line2D([0],[0], marker='s',color='w',markerfacecolor=C['gray'],markersize=5,label='Tie'),
          Line2D([0],[0], marker='^',color='w',markerfacecolor=C['red'],markersize=5,label='Ent. wins'),
          Line2D([0],[0], color=C['gray'], linestyle='--', lw=0.7, label='p* (C=3)'),
          Line2D([0],[0], color=C['gray'], linestyle='-', lw=0.7, label='p* (C=8)'),
          Line2D([0],[0], color=C['gray'], linestyle=':', lw=0.7, label='p* (C=20)'),
          Line2D([0],[0], color=C['gray'], linestyle='-.', lw=0.7, label='p* (C=100)'),
          Line2D([0],[0], color=C['gray'], linestyle='dotted', lw=0.7, label='p* (C=200)')]
    ax1.legend(handles=l1, fontsize=FS(6), loc='lower right', framealpha=0.9, borderpad=0.4, labelspacing=0.3, ncol=2)

    # (b) theoretical gap-delta alignment (27 datasets)
    for ds in datasets:
        oc = 'win' if ds['delta_lo']>0 else ('loss' if ds['delta_hi']<0 else 'tie')
        mk,co = mk_c[oc]
        ax2.scatter(ds['gap'], ds['delta'], c=co, marker=mk, s=30,
                    edgecolors='white', linewidth=0.4, zorder=5)
        nm = short.get(ds['name'], ds['name'][:6])
        if ds['name'] in label_set or oc == 'win':
            ox, oy = OFF_B.get(ds['name'], (random.randint(-6,8), random.randint(-7,7)))
            ax2.annotate(nm, (ds['gap'], ds['delta']), textcoords="offset points",
                         xytext=(ox,oy), fontsize=FS(6), color=co)
    ax2.axhline(0, color=C['light'], linestyle='--', linewidth=0.6)
    finish(ax2, r'$\Delta_\alpha$ (gap)', r'AUROC $\Delta$ (Agr.$-$Ent.)', legend=False)
    ax2.text(0.97, 0.95, '(b)', transform=ax2.transAxes, fontsize=FS(10),
             fontweight='bold', va='top', ha='right')
    gaps = [ds['gap'] for ds in datasets]
    dvs = [ds['delta'] for ds in datasets]
    rho, pv = spearmanr(gaps, dvs)
    ax2.text(0.03, 0.93, r'$\rho=%.2f, p=%.2f$ (n=%d)'%(rho,pv,len(gaps)),
             transform=ax2.transAxes, fontsize=FS(7), fontstyle='italic')
    fig.tight_layout(); save(fig, 'fig8_phase')

def fig10_win_comparison():
    """Hard vs soft agreement win-rate bar chart (27 datasets)."""
    _rc(4.5, 0.5)
    d = json.load(open(os.path.join(ROOT, 'anchoring_meta_report.json')))
    wt = d['win_table']
    n = len(d['results'])
    fig, ax = plt.subplots(figsize=(4.5, 2.8))
    signals = ['Hard voting', 'Soft agreement']
    wins = [wt['hard_agreement']['win'], wt['soft_agreement']['win']]
    ties = [wt['hard_agreement']['tie'], wt['soft_agreement']['tie']]
    losses = [wt['hard_agreement']['loss'], wt['soft_agreement']['loss']]
    x = range(len(signals))
    w = 0.5
    ax.bar(x, wins, w, label='Win (CI>0)', color=C['blue'])
    ax.bar(x, ties, w, bottom=wins, label='Tie', color=C['gray'])
    ax.bar(x, losses, w, bottom=[a+b for a,b in zip(wins,ties)], label='Loss (CI<0)', color=C['red'])
    for i, (wi,ti,lo) in enumerate(zip(wins,ties,losses)):
        if wi: ax.text(i, wi/2, str(wi), ha='center', va='center', fontsize=FS(8), fontweight='bold', color='white')
        if ti: ax.text(i, wi+ti/2, str(ti), ha='center', va='center', fontsize=FS(8), color='white')
        if lo: ax.text(i, wi+ti+lo/2, str(lo), ha='center', va='center', fontsize=FS(8), color='white')
    ax.set_xticks(list(x)); ax.set_xticklabels(signals, fontsize=FS(9))
    ax.set_ylabel(f'Datasets (n={n})', fontsize=FS(9))
    ax.set_ylim(0, n)
    ax.legend(fontsize=FS(7), loc='upper left', framealpha=0.9)
    ax.spines['right'].set_visible(False); ax.spines['top'].set_visible(False)
    ax.spines['left'].set_linewidth(0.6); ax.spines['bottom'].set_linewidth(0.6)
    ax.tick_params(width=0.6, length=3)
    fig.tight_layout(); save(fig, 'fig9_win_comparison')


def fig_method_overview():
    """Method-overview schematic (orthogonal edges, wide panels/gutters).
    Canvas 1000x700 @ 7.5x5.25in (144.3 units per printed inch @ 6.93in).
    Physical budget: panels 300+300+290 = 890u = 6.17in printed (wider than
    the original 880u/6.10in); gutters 2x46u = 0.32in each (orig 40u);
    outer margins 9u. Inner-box padding 20u. All edges axis-aligned; the
    two gutter lanes cannot cross (alpha climbs x=676 below y=440; the
    entropy feed is a single flat lane at y=562)."""
    from matplotlib.patches import Polygon
    SZ_M = (7.5, 5.25)
    _rc(7.5, 1.0); fig = plt.figure(figsize=SZ_M)
    ax = fig.add_subplot(111)
    W, H = 1000.0, 700.0
    ax.set_xlim(0, W); ax.set_ylim(0, H)
    ax.axis('off')

    def rbox(x, y, w, h, fc, ec, rx=8, lw=1.2, ls='-', z=2, alpha=1.0):
        from matplotlib.patches import FancyBboxPatch
        ax.add_patch(FancyBboxPatch((x, y), w, h,
            boxstyle=f'round,pad=0,rounding_size={rx*0.55}',
            fc=fc, ec=ec, lw=lw, linestyle=ls, zorder=z, alpha=alpha))

    def txt(x, y, s, pt, color, bold=False, ha='center', z=4):
        ax.text(x, y, s, fontsize=FS(pt), color=color, ha=ha, va='center',
                fontweight='bold' if bold else 'normal', zorder=z)

    def arrow(x1, y1, x2, y2, color, lw=1.6, ms=10, z=3):
        from matplotlib.patches import FancyArrowPatch
        assert x1 == x2 or y1 == y2, 'edges must be axis-aligned'
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>',
            mutation_scale=ms, color=color, lw=lw, zorder=z))

    def line(x1, y1, x2, y2, color, lw=1.2, z=3):
        assert x1 == x2 or y1 == y2, 'edges must be axis-aligned'
        ax.plot([x1, x2], [y1, y2], color=color, lw=lw, zorder=z,
                solid_capstyle='butt')

    # ── geometry constants ───────────────────────────────────────────
    LP, MP, RP = (9, 309), (355, 655), (701, 991)      # panel x-spans
    LW, MW, RW = 300, 300, 290
    LCX, MCX, RCX = 159, 505, 846                        # panel centers

    # ── left panel: backbones ────────────────────────────────────────
    rbox(LP[0], 60, LW, 580, '#E8ECF2', '#B4C0E4', rx=12)
    txt(LCX, 615, 'Backbones (CLIP family)', 10, '#0F4D92', bold=True)
    bells = [
        ('OpenAI CLIP (ViT-L/14)', 'A. Radford et al., ICML 2021'),
        ('LAION-2B (ViT-H/14)', 'G. Ilharco et al., Zenodo 2021'),
        ('SigLIP2 (ViT-L/16-256)', 'M. Tschannen et al., arXiv 2025'),
        ('EVA-02-CLIP', 'Y. Fang et al., IVC 2024'),
        ('DFN-CLIP', 'A. Fang et al., ICLR 2024'),
    ]
    for i, (n, c) in enumerate(bells):
        y = 555 - i * 102
        rbox(29, y - 40, 260, 76, '#DBEAFE', '#93C5FD', rx=10)
        txt(LCX, y + 12, n, 9, '#1E3A5F', bold=True)
        txt(LCX, y - 16, c, 7, '#64748B')

    # ── middle panel: consensus ──────────────────────────────────────
    rbox(MP[0], 360, MW, 280, '#F5ECD0', '#B89B45', rx=12)
    txt(MCX, 615, 'Consensus Prediction', 10.5, '#4D4D4D', bold=True)
    rbox(375, 540, 260, 44, '#FEF08A', '#B89B45', rx=8)
    txt(MCX, 562, r'Avg Logits  $\bar{\ell}(x)$', 8.5, '#713F12', bold=True)
    rbox(375, 470, 260, 44, '#B98A3E', '#A8763B', rx=8, alpha=0.35)
    txt(MCX, 492, r'Consensus  $c(x){=}\arg\max_{y}\,\bar{\ell}(x)_{y}$', 8.5, '#78350F', bold=True)
    rbox(375, 400, 260, 44, '#B89B45', '#A8763B', rx=8)
    txt(MCX, 422, r'$c(x)\in \mathrm{GT}(x)$? (eval-time)', 8, '#78350F')

    # ── middle panel: anchoring signal ───────────────────────────────
    rbox(MP[0], 60, MW, 250, '#F0FDF4', '#BBF7D0', rx=12)
    txt(MCX, 285, 'Instance Anchoring Signal', 9, '#166534', bold=True)
    # box enlarged to y∈[78,234] so all six signal lines sit inside it
    # (previously 3 lines floated below the box)
    rbox(375, 78, 260, 156, '#4E8D7C', '#3E7A64', rx=8, alpha=0.2)
    txt(MCX, 222, r'Agreement  $\alpha(x)$', 8.5, '#14532D', bold=True)
    txt(MCX, 196, r'Fraction of backbones', 7, '#64748B')
    txt(MCX, 176, r'with $\arg\max = c(x)$', 7, '#64748B')
    txt(MCX, 136, r'5 backbones $\Rightarrow$ 6 values (hard)', 7, '#64748B')
    txt(MCX, 112, r'Soft agreement: Sec. 5.6', 7, '#166534')
    txt(MCX, 88, 'Higher agreement = more anchored', 7, '#64748B')

    # ── right panel: selective prediction ────────────────────────────
    rbox(RP[0], 60, RW, 580, '#FFF7ED', '#FED7AA', rx=12)
    txt(RCX, 615, 'Selective Prediction', 10.5, '#9A3412', bold=True)
    # entropy baseline (dashed), aligned with the avg-logits lane
    rbox(721, 516, 250, 84, '#FEF2F2', '#FECACA', rx=10, ls=(0, (5, 4)))
    txt(RCX, 574, 'Baseline: Entropy $H(x)$', 8.5, '#991B1B', bold=True)
    txt(RCX, 540, r'$H(\mathrm{softmax}(\bar{\ell}))$, same 5 logits', 7, '#64748B')
    # decision diamond (center 846,440; half 50)
    ax.add_patch(Polygon([(811, 390), (861, 440), (811, 490), (761, 440)],
        closed=True, fc='#FED7AA', ec='#B9704C', lw=1.6, zorder=2))
    txt(811, 440, r'$\alpha(x) > \tau$ ?', 8, '#7C2D12', bold=True)
    # Yes -> Pred (horizontal)
    arrow(861, 440, 896, 440, '#8BCF8B', ms=9)
    txt(879, 458, 'Yes', 8, '#166534', bold=True)
    rbox(902, 417, 68, 46, '#DCFCE7', '#86EFAC', rx=8)
    txt(936, 440, 'Pred.', 8, '#166534', bold=True)
    # No -> Abst (elbow)
    line(811, 390, 811, 362, C['red'], lw=1.2)
    arrow(811, 362, 896, 362, C['red'], ms=9)
    txt(825, 376, 'No', 8, C['red'], bold=True)
    rbox(902, 339, 68, 46, '#FEE2E2', '#FECACA', rx=8)
    txt(936, 362, 'Abst.', 8, '#991B1B', bold=True)
    # evaluation
    rbox(721, 80, 250, 200, '#E0E7FF', '#A5B4FC', rx=10)
    txt(RCX, 252, 'Evaluation', 9, '#3730A3', bold=True)
    txt(RCX, 222, 'AUROC, Risk-Coverage, AURC', 7, '#64748B')
    txt(RCX, 196, 'Coupled bootstrap ($B{=}10{,}000$)', 7, '#64748B')
    txt(RCX, 170, 'CAPE Set-A (main), $T{=}100$', 7, '#64748B')
    txt(RCX, 144, 'CAPE B/C for robustness', 7, '#64748B')

    # ── edges (axis-aligned; zero crossings by lane design) ─────────
    # E1: five backbones -> drop-bus x=330 -> avg logits (flat at y=574)
    for i in range(5):
        y = 555 - i * 102
        line(289, y, 330, y, '#93C5FD', lw=1.4)
    line(330, 147, 330, 574, '#93C5FD', lw=1.4)
    arrow(330, 574, 371, 574, '#93C5FD', lw=1.4, ms=9)
    # E2/E3: avg logits -> consensus -> GT check
    arrow(MCX, 540, MCX, 514, '#B89B45', ms=9)
    arrow(MCX, 470, MCX, 444, '#A8763B', ms=9)
    # E4: consensus c(x) feeds the anchoring signal
    arrow(MCX, 400, MCX, 314, '#3E7A64', lw=1.2, ms=8)
    txt(517, 357, r'$c(x)$', 7.5, '#3E7A64', ha='left')
    # E5: alpha -> decision diamond (gutter lane x=676, climbs 210->440)
    line(635, 210, 676, 210, C['green'], lw=1.4)
    line(676, 210, 676, 440, C['green'], lw=1.4)
    arrow(676, 440, 757, 440, C['green'], lw=1.4, ms=9)
    txt(735, 458, r'$\alpha(x)$', 7.5, C['green'])
    # E6: avg logits -> entropy baseline (single flat lane y=562)
    arrow(635, 562, 717, 562, '#999999', lw=1.0, ms=7)

    # ── bottom: datasets strip (two lines to stay within canvas) ────
    txt(500, 42, '27 datasets: TB (8cls), HRW, BTH, VOC2007, COCO, Stanford40, '
                 'CIFAR-100, MultiTHUMOS,', 7.5, '#475569')
    txt(500, 18, '+ 19 more (fine-grained, scene, OOD, corruptions)', 7.5, '#475569')

    save(fig, 'fig1_method')

# ══════════════════════════════════════════════════════════════════════
# main
# ══════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print('Generating ALL figures (strict visual consistency)...')
    fig1_risk_coverage()
    fig3_cost()
    fig4_mechanism()
    fig5_instance_vs_class()
    fig7_generalization()
    fig8_saturation_model()
    fig9_phase_diagram()
    fig10_win_comparison()
    fig_method_overview()
    print(f'\nDone → {OUT}/')
