#!/usr/bin/env python3
"""Cross-validate anchoring_theory.py against canonical anchoring_main_report.json.

Checks every overlapping metric and reports discrepancies > 1e-4.
"""

import json, sys
from pathlib import Path

ROOT = Path("results")

with open(ROOT / "anchoring_main_report.json") as f:
    main = json.load(f)

with open(ROOT / "anchoring_theory_report.json") as f:
    theory = json.load(f)

SUBSET_MAP = {
    "teacher_behavior": "teacher_behavior",
    "handrise_readwrite": "handrise_readwrite",
    "bow_turnhead": "bow_turnhead",
}

errors = []

for theory_key, main_key in SUBSET_MAP.items():
    # find main result for this subset
    main_r = [r for r in main["results"] if r["subset"] == main_key]
    if not main_r:
        errors.append(f"[MISSING] {main_key} not in main report")
        continue
    main_r = main_r[0]

    theory_r = theory.get(theory_key)
    if not theory_r:
        errors.append(f"[MISSING] {theory_key} not in theory report")
        continue

    n_main = main_r["n_images"]
    n_theory = theory_r["n_images"]
    if n_main != n_theory:
        errors.append(f"[N MISMATCH] {theory_key}: main={n_main}, theory={n_theory}")

    # consensus accuracy
    acc_main = main_r["consensus_accuracy"]
    acc_theory = theory_r["consensus_accuracy"]
    if abs(acc_main - acc_theory) > 1e-4:
        errors.append(f"[CONSENSUS ACC] {theory_key}: main={acc_main:.6f}, theory={acc_theory:.6f}")

    # per-signal AUROC
    for sig_name, theory_name in [("entropy", "ensemble_entropy"), ("agreement_top1", "agreement_top1")]:
        main_sig = main_r["signals"].get(sig_name)
        theory_sig = theory_r["ensemble_signals"].get(theory_name)
        if main_sig is None:
            errors.append(f"[MISSING SIG] {sig_name} in main report")
            continue
        if theory_sig is None:
            errors.append(f"[MISSING SIG] {theory_name} in theory report")
            continue
        auroc_main = main_sig["auroc"]
        auroc_theory = theory_sig["auroc"]
        if abs(auroc_main - auroc_theory) > 1e-4:
            errors.append(f"[AUROC] {theory_key}/{sig_name}: main={auroc_main:.6f}, theory={auroc_theory:.6f}")

        # selective acc @50 and @70
        for cov in ["50pct", "70pct"]:
            acc_main = main_sig["acc_at"][cov]
            theory_key_cov = f"selacc@{cov.replace('pct', '')}"
            acc_theory = theory_sig.get(theory_key_cov)
            if acc_theory is None:
                # my report stores as selacc@50, selacc@70
                theory_key_cov = f"selacc@{int(cov.replace('pct', ''))}"
                acc_theory = theory_sig.get(theory_key_cov)
            if acc_theory is None:
                errors.append(f"[MISSING FIELD] {theory_name}.{theory_key_cov} in theory report")
                continue
            if abs(acc_main - acc_theory) > 1e-4:
                errors.append(f"[SELACC] {theory_key}/{sig_name}/{cov}: main={acc_main:.6f}, theory={acc_theory:.6f}")

    # bootstrap CI for agreement - entropy
    main_ci = main_r.get("deltas_vs_entropy", {}).get("agreement_top1_minus_entropy")
    theory_ci = theory_r.get("bootstrap_agreement_minus_entropy")
    if main_ci and theory_ci:
        for field in ["mean", "ci_lo", "ci_hi"]:
            v_main = main_ci[field]
            v_theory = theory_ci[field]
            if abs(v_main - v_theory) > 1e-3:
                errors.append(f"[BOOT CI {field}] {theory_key}: main={v_main:.6f}, theory={v_theory:.6f}")
    else:
        if main_ci is None:
            errors.append(f"[MISSING CI] core_bet in main report for {theory_key}")
        if theory_ci is None:
            errors.append(f"[MISSING CI] bootstrap in theory report for {theory_key}")

print("=" * 60)
print("  Cross-validation: anchoring_theory vs anchoring_main")
print("=" * 60)
print()
if not errors:
    print("  ✅ ALL CHECKS PASSED — every overlapping metric matches within tolerance.")
    print()
    # Summary table
    for theory_key, main_key in SUBSET_MAP.items():
        main_r = [r for r in main["results"] if r["subset"] == main_key][0]
        theory_r = theory.get(theory_key)
        print(f"  {theory_key}:")
        print(f"    consensus_acc:         {main_r['consensus_accuracy']:.4f}")
        print(f"    ensemble entropy AUROC:{main_r['signals']['entropy']['auroc']:.4f}  "
              f"(theory: {theory_r['ensemble_signals']['ensemble_entropy']['auroc']:.4f})")
        print(f"    agreement AUROC:       {main_r['signals']['agreement_top1']['auroc']:.4f}  "
              f"(theory: {theory_r['ensemble_signals']['agreement_top1']['auroc']:.4f})")
        mc = main_r.get("deltas_vs_entropy", {}).get("agreement_top1_minus_entropy", {})
        tc = theory_r.get("bootstrap_agreement_minus_entropy", {})
        print(f"    Δ(agr−ent):            {mc.get('mean', '?'):+.4f} [{mc.get('ci_lo', '?'):.4f},{mc.get('ci_hi', '?'):.4f}]  "
              f"(theory: {tc.get('mean', '?'):+.4f} [{tc.get('ci_lo', '?'):.4f},{tc.get('ci_hi', '?'):.4f}])")
else:
    print(f"  ❌ {len(errors)} DISCREPANC(IES) FOUND:")
    print()
    for e in errors:
        print(f"    {e}")

print()
print(f"Script:  analysis/anchoring_theory.py")
print(f"Input:   results/per_image/clip_*_*_logits.npy (15 files)")
print(f"Output:  results/anchoring_theory_report.json")
print(f"         results/anchoring_theory_report.txt")
print(f"Seed:    42  |  Bootstrap: 10000  |  Temperature: 100.0")
sys.exit(1 if errors else 0)