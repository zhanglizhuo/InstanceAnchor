#!/usr/bin/env python3
"""SUPERSEDED: tables are inline in paper/InstantAnchor.tex (cas-sc conversion 2026-08-17); paper/tables/ no longer exists. Do not use for paper numbers."""

import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RES = REPO / "results"
OUT = REPO / "paper" / "tables"
OUT.mkdir(parents=True, exist_ok=True)

SIGNAL_ORDER = ["entropy", "msp", "margin", "agreement_top1", "agreement_weighted", "logitvar"]
SIGNAL_LATEX = {"entropy":"Entropy","msp":"MSP","margin":"Margin","agreement_top1":"Agreement\\_top1",
                "agreement_weighted":"Agreement\\_weighted","logitvar":"LogitVar"}
CLASS_ORDER = ["guide","answer","on-stage interaction","blackboard-writing","teacher","stand","screen"]
CLASS_BOLD = {"answer"}

def fm(x, dp=3): return "---" if x is None else f"{x:.{dp}f}"
def fp(x, dp=2): return "---" if x is None else f"{x*100:.{dp}f}\\%"
def fci(c, dp=3): return f"[{c['ci_lo']:+.{dp}f},{c['ci_hi']:+.{dp}f}]"

# ── Table 1 (main): AUROC/AUPR/AURC/Δ across 3 subsets ─────────────
def table_main():
    d = json.load(open(RES/"anchoring_main_report.json"))
    lines = [r"\begin{tabular}{lcccc}", r"\toprule",
             r"Signal & AUROC & AUPR & AURC & $\Delta$ vs~entropy \\", r"\midrule"]
    for r in d["results"]:
        sub = r["subset"]; N = r["n_images"]
        labels = {"teacher_behavior":("TeacherBehavior","8cls"),"handrise_readwrite":("HandriseReadWrite","3cls"),
                  "bow_turnhead":("BowTurnHead","2cls")}
        disp, cl = labels[sub]
        lines.append(r"\multicolumn{5}{c}{\textbf{"+disp+"} ("+cl+", N="+str(N)+")} \\\\"); lines.append(r"\midrule")
        sigs = r["signals"]; dels = r.get("deltas_vs_entropy",{})
        if sub == "bow_turnhead":
            # Merge entropy/MSP/margin into one row (identical in 2-class case)
            s = sigs["entropy"]
            lines.append(f"Entropy / MSP / Margin\\textsuperscript{{*}} & {fm(s['auroc'])} & {fm(s['aupr'])} & {fm(s['aurc'])} & --- \\\\")
            for name in ["agreement_top1","agreement_weighted","logitvar"]:
                sd = sigs[name]; dk = dels.get(name+"_minus_entropy")
                ds = "---" if dk is None else f"${dk['mean']:+.3f}$~${fci(dk)}$"
                lines.append(f"{SIGNAL_LATEX[name]} & {fm(sd['auroc'])} & {fm(sd['aupr'])} & {fm(sd['aurc'])} & {ds} \\\\")
        else:
            winner = "agreement_top1" if sub=="teacher_behavior" else "entropy"
            for nm in SIGNAL_ORDER:
                if nm not in sigs: continue
                sd = sigs[nm]
                auroc = fm(sd["auroc"]); aupr = fm(sd["aupr"]); aurc = fm(sd["aurc"])
                if nm=="entropy": ds="---"
                else:
                    dk = dels.get(nm+"_minus_entropy")
                    ds = "---" if dk is None else f"${dk['mean']:+.3f}$~${fci(dk)}$"
                if nm==winner:
                    auroc=r"\textbf{"+auroc+"}"
                    if winner!="entropy":
                        aupr=r"\textbf{"+aupr+"}"; aurc=r"\textbf{"+aurc+"}"
                lines.append(f"{SIGNAL_LATEX[nm]} & {auroc} & {aupr} & {aurc} & {ds} \\\\")
    lines.append(r"\bottomrule"); lines.append(r"\end{tabular}")
    lines.append(r"\vspace{4pt}")
    lines.append(r"{\footnotesize\textsuperscript{*}For 2-class problems, entropy, MSP, and margin are monotonic transforms of $|p-0.5|$ and thus yield identical rankings (hence identical AUROC/AUPR/AURC).}")
    (OUT/"table_main.tex").write_text("\n".join(lines))

# ── Table 2 (perclass): per-class AUROC + Δ ─────────────────────────
def table_perclass():
    d = json.load(open(RES/"anchoring_main_report.json"))
    tb = d["results"][0]; pc = tb["per_class"]
    lines = [r"\begin{tabular}{lrrrrr}", r"\toprule",
             r"Class & $N$ & Acc & Entropy AUROC & Agreement AUROC & $\Delta$ \\", r"\midrule"]
    for cls in CLASS_ORDER:
        if cls not in pc: continue
        cd = pc[cls]; n = cd["n"]; acc = fm(cd["consensus_acc"])
        ent = cd.get("entropy_auroc"); agr = cd.get("agreement_auroc")
        es = fm(ent); ags = fm(agr)
        dvs = "---" if ent is None else f"${agr-ent:+.3f}$"
        cdn = cls.replace("_",r"\_")
        if cls in CLASS_BOLD:
            lines.append(rf"\textbf{{{cdn}}} & \textbf{{{n}}} & \textbf{{{acc}}} & \textbf{{{es}}} & \textbf{{{ags}}} & \textbf{{{dvs}}} \\\\")
        else:
            lines.append(f"{cdn} & {n} & {acc} & {es} & {ags} & {dvs} \\\\")
    lines.append(r"\bottomrule"); lines.append(r"\end{tabular}")
    lines.append(r"\vspace{4pt}")
    lines.append(r"{\footnotesize\textsuperscript{*}blackboard (N=40) excluded: 0\% accuracy, AUROC undefined. Screen (N=24) has wide confidence intervals.}")
    (OUT/"table_perclass.tex").write_text("\n".join(lines))

# ── Table 3 (classvsinst): instance vs class ────────────────────────
def table_classvsinst():
    d = json.load(open(RES/"anchoring_class_vs_instance_report.json"))
    r = d["results"][0]; c1=r["C1_auroc"]; c4=r["C4_fair_selective"]; c3=r["C3_combined"]; sel=c4["selective_accuracy"]
    lines = [r"\begin{tabular}{lccc}", r"\toprule",
             r"Signal & AUROC & Sel. Acc@30\% & Sel. Acc@50\% \\", r"\midrule"]
    lines.append(f"Instance agreement & \\textbf{{{fm(c1['instance_agreement'])}}} & \\textbf{{{fp(sel['0.3']['instance'])}}} & {fp(sel['0.5']['instance'])} \\\\")
    lines.append(f"Class mean (held-out) & {fm(c1['class_mean_agreement'])} & {fp(sel['0.3']['classmean_heldout'])} & {fp(sel['0.5']['classmean_heldout'])} \\\\")
    lines.append(f"VA AnchorScore & {fm(c1['va_anchorscore_primary'])} & {fp(sel['0.3']['va_anchorscore'])} & {fp(sel['0.5']['va_anchorscore'])} \\\\")
    lines.append(f"Instance + class mean & \\textbf{{{fm(c3['instance_plus_classmean_auroc'])}}} & --- & --- \\\\")
    lines.append(r"\bottomrule"); lines.append(r"\end{tabular}")
    lines.append(r"\vspace{4pt}")
    lines.append(r"{\footnotesize\textsuperscript{*}Combined signal selective accuracy not computable from held-out class-mean protocol; AUROC confirms complementarity.}")
    (OUT/"table_classvsinst.tex").write_text("\n".join(lines))

# ── Table 4 (cost): cost scaling ────────────────────────────────────
def table_cost():
    d = json.load(open(RES/"anchoring_cost_report.json")); tb = d["results"][0]; by_k = tb["by_k"]
    lines = [r"\begin{tabular}{lcccc}", r"\toprule",
             r"$k$ & $n$ subsets & Ensemble Ent AUROC & Agreement AUROC & Agr.\ Sel.\ Acc@30\% \\",
             r"\midrule"]
    for k in ["k=1","k=2","k=3","k=4","k=5"]:
        kd = by_k[k]; n_s = kd["n_subsets"]; ent = fm(kd.get("ensemble_entropy_auroc"))
        agr_raw = kd.get("agreement_auroc")
        if agr_raw is None:
            agr = "---"; sel = "---"
        else:
            agr = r"\textbf{"+fm(agr_raw)+"}" if k=="k=5" else fm(agr_raw)
            sel = r"\textbf{"+fp(kd["agreement_selacc"]["0.3"])+"}" if k=="k=5" else fp(kd["agreement_selacc"]["0.3"])
        lines.append(f"{kd['k']} & {n_s} & {ent} & {agr} & {sel} \\\\")
    lines.append(r"\bottomrule"); lines.append(r"\end{tabular}")
    lines.append(r"\vspace{4pt}")
    lines.append(r"{\footnotesize At $k{=}1$, agreement is undefined (single-backbone voting is trivial); ensemble entropy at $k{=}1$ achieves selective accuracy 72.90\%.}")
    (OUT/"table_cost.tex").write_text("\n".join(lines))

if __name__ == "__main__":
    table_main(); table_perclass(); table_classvsinst(); table_cost()
    print(f"Generated 4 table files → {OUT}/")
    for p in sorted(OUT.glob("*.tex")): print(f"  {p.name}")
