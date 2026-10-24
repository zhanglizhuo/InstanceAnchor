# InstanceAnchor

**A mechanical account of cross-model agreement saturation: the plurality-vote ceiling and the Jensen-gap signal that survives it.**

Paper: *The Plurality Ceiling: A Mechanical Account of Cross-Model Agreement Saturation and the Jensen-Gap Signal That Survives It* — 31 pages (Elsevier cas-sc template), 9 figures, 8+ tables, 33 references (refs.bib, elsarticle-num-names).

## What this paper shows

For zero-shot CLIP, one natural abstention signal is **cross-backbone agreement**: run the image through several CLIP-family models and ask how much their predictions concur. This paper shows that the signal has a **dual structure**: hard voting (top-1 vote fraction) saturates and loses on most datasets, but its confidence-weighted cousin—**soft agreement** (mean softmax mass at the consensus class)—generalizes broadly.

**The signals.** Given 5 CLIP-family backbones (openai, laion, siglip2, eva02, dfn):
- **Hard agreement**: fraction of backbones whose argmax matches the consensus.
- **Soft agreement**: mean softmax probability mass at the consensus class across backbones.
Both are compared compute-matched against ensemble entropy, MSP, margin, max-logit, energy, ensemble MI, and prediction variance.

**Main findings.**

1. **Hard voting saturates**: wins on only 8/27 datasets (TeacherBehavior plus seven ImageNet-family OOD benchmarks), all in the unsaturated regime ($p < p^\star$). The plurality-induced ceiling ($\kappa$ decomposition, 54–108% across 27 datasets, i.i.d.-MC at $N{=}5\times10^7$) and the regime paradox (inverted-U) explain why.

2. **Soft agreement generalizes**: significantly outperforms entropy on **23/27** datasets, with only 3 small losses ($|\Delta| < 0.015$). It wins across both saturated and unsaturated regimes—where hard voting fails, soft agreement retains discriminative power.

3. **The Jensen gap mechanism**: soft agreement normalizes per-backbone softmaxes *before* averaging (preserving cross-backbone distribution-shape divergence), while MSP normalizes *after*. The Jensen gap $|\alpha_\text{soft} - \text{MSP}|$ independently predicts per-image error (AUROC 0.622). MI hierarchy: $I(\alpha_\text{soft}; \text{correct}) = 0.162 > I(\text{MSP}) = 0.132 > I(\text{entropy}) = 0.121$.

4. **SOTA comparison**: against 7 modern CSFs, soft agreement beats energy and prediction variance on all 27 datasets, ensemble MI on 26/27, max-logit and entropy on 23/27, and hard voting on 25/27; competitive with MSP (the strongest baseline: 14/6/7 W/T/L, positive mean advantage).

5. **Robust and causal**: advantage is robust to backbone count ($k{=}8$: 8/12 wins, 0 losses). A controlled co-occurrence manipulation (logit-merging on CIFAR-100/VOC2007) confirms that input-level visual co-occurrence causally drives agreement signal (label injection alone is inert).

6. **LOO at $n{=}27$**: the $\kappa$ boundary achieves 81.5% sign recovery (88.9% for soft agreement with joint refit), driven by the label-free class-count prior rather than the label-derived $\kappa$.

7. **Honest magnitude check (selective accuracy)**: soft agreement's AUROC edge is a ranking-level property. At matched coverage, it does not uniformly beat entropy on selective accuracy (mean $-2.76$pp at 50% coverage, 4/27 wins; TB $+1.36$pp@50%, $+6.61$pp@70%). The paper's deployment recommendation is therefore two-tiered: soft agreement for ranking the full pool, entropy for hard abstention thresholds.

## Key results (27 datasets, B=10000 coupled bootstrap, 5 backbones)

| Signal | Win | Tie | Loss |
|---|---|---|---|
| Hard voting ($\alpha_\text{top1}$) | 8 | 4 | 15 |
| **Soft agreement** ($\alpha_\text{soft}$) | **23** | 1 | 3 |

See `results/anchoring_meta_report.{json,txt}` for the full 27-dataset per-row table (N/C/p/acc/$\kappa$/$\Delta_\text{hard}$/$\Delta_\text{soft}$).

## Paper ↔ code ↔ evidence map

| Paper content | Script (`analysis/`) | Evidence (`results/`) |
|---|---|---|
| Exp A–D: 8-dataset main + mechanism + class-level + cost | `anchoring_main.py`, `anchoring_mechanism.py`, `anchoring_class_vs_instance.py`, `anchoring_cost*.py` | corresponding `*_report.{json,txt}` |
| Exp E: 27-dataset meta-study + soft agreement | `anchoring_meta.py` | `anchoring_meta_report.{json,txt}` |
| SOTA CSF comparison (8 baselines) | `anchoring_sota_csf.py` | `anchoring_sota_csf_report.{json,txt}` |
| Jensen gap mechanism | `anchoring_soft_mechanism.py` | `anchoring_soft_mechanism_report.{json,txt}` |
| κ decomposition (27 datasets, i.i.d.-MC, $N{=}5\times10^7$) | `anchoring_kappa_meta_decomp.py` | `anchoring_kappa_meta_decomp_report.{json,txt}` |
| P0: co-occurrence causal manipulation | `anchoring_synthetic_colabel.py` | `anchoring_synthetic_colabel_report.{json,txt}` |
| P1: text geometry (elimination) | `anchoring_text_geometry.py` | `anchoring_text_geometry_report.{json,txt}` |
| P2: CAPE per-class breakdown | `anchoring_prompt_class_breakdown.py` | `anchoring_prompt_class_breakdown_report.{json,txt}` |
| P3: signal-space expansion | `anchoring_signal_space.py` | `anchoring_signal_space_report.{json,txt}` |
| P5: backbone panel (k=5 vs k=8) | `anchoring_backbone_panel.py` | `anchoring_backbone_panel_report.{json,txt}` |
| Residual direct measurement (wrong-class coincidence, ρ=0.628) | `anchoring_error_overlap_27.py` | `anchoring_error_overlap_27_report.{json,txt}` |
| Selective accuracy at matched coverage (27 datasets) | `anchoring_selective_accuracy_27.py` | `anchoring_selective_accuracy_27_report.{json,txt}` |
| Generalization, robustness, cascade | `anchoring_v2.py`, `anchoring_robustness.py`, `anchoring_cascade.py`, ... | corresponding reports |

Figures are generated by `paper/generate_figures.py` (matplotlib only) into `paper/figures/fig1_method … fig9_win_comparison.pdf`.

## Reproduce

```bash
pip install -r requirements.txt
# Core analyses (CPU-only, ~15 min total):
python analysis/anchoring_main.py           # Exp A
python analysis/anchoring_meta.py           # Exp E: 27-dataset meta-study
python analysis/anchoring_sota_csf.py       # SOTA CSF comparison
python analysis/anchoring_soft_mechanism.py # Jensen gap mechanism
python analysis/anchoring_kappa_meta_decomp.py  # κ decomposition (27 datasets, N=5e7)
python analysis/anchoring_synthetic_colabel.py  # P0: causal co-occurrence
python analysis/anchoring_backbone_panel.py # P5: k=5 vs k=8
python analysis/anchoring_signal_space.py   # P3: signal expansion
python analysis/anchoring_error_overlap_27.py  # residual direct measurement
python analysis/anchoring_selective_accuracy_27.py  # matched-coverage check
```

Paper compile (Elsevier cas-sc, needs BibTeX):
```bash
cd paper && pdflatex InstantAnchor.tex && bibtex InstantAnchor && pdflatex InstantAnchor.tex && pdflatex InstantAnchor.tex
```

All scripts are deterministic-seeded, write JSON + a human-readable report to `results/`, and log to `logs/`. Pre-extracted per-image logits/labels (`.npy` pairs, 5 CLIP backbones × 27 datasets plus the cross-family panels and seed-diverse ensembles) are in `results/per_image/`—no GPU or model download required for the analyses.

## Data availability

Pre-extracted per-image logits + ground-truth labels (`results/per_image/`, 5 CLIP backbones × 27 datasets plus cross-family panels and seed-diverse ensembles) are released with the code; the SCB5 classroom-behavior benchmark and MLLM cascade outputs are available from the corresponding author upon reasonable request. Public benchmarks were sourced from HuggingFace and ModelScope mirrors; one dataset (Oxford-Flowers-102) was excluded due to a defective label-name mapping in the mirror (verified by feature inspection).

## Citation

```bibtex
@article{ma2026plurality,
  title   = {The Plurality Ceiling: A Mechanical Account of Cross-Model Agreement Saturation and the Shape-Divergence Signal That Survives It},
  author  = {Ma, Yan and Zhang, Lizhuo},
  year    = {2026},
  note    = {Preprint; manuscript under preparation for submission}
}
```

## Contact

Yan Ma and Lizhuo Zhang. This work is part of a series on visual anchoring in the SCB5 classroom-behavior setting (sibling papers: llm-annotation, VisualAnchor, SharedVisual, scb5-zeroshot).
