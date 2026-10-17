"""Supplementary Table S1: per-study results (writes paper/table_s1.tex and results/table_s1.csv)."""
import re

import numpy as np
import pandas as pd

exec(open("07_make_figures.py", encoding="utf8").read().split("def save(")[0].split("os.makedirs")[0])


def safe(u):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", u)


def label(u):
    name, sub = u.split("|")
    name = name.replace("_", " ")
    multi = u.startswith(("Cao_2021_Exp2", "Keidel_2024", "Thome_2023", "Yang_2022"))
    return f"{name} ({sub[1:]})" if multi else name


units = pd.read_csv("unit_summary.csv").query("magnitude_varying").set_index("unit").drop(index="Thrailkill_2022b|s1")
med = pd.read_csv("results/median_trials_per_subject.csv", index_col=0).iloc[:, 0]
h2 = pd.read_csv("results/h2main_per_study.csv").set_index("unit")
r2 = pd.read_csv("results/r2_per_study.csv").set_index("unit")
r2f = pd.read_csv("results/r2fmain_per_study.csv").set_index("unit")
mc = pd.read_csv("results/model_comparison.csv")
mc["unit"] = mc.unit.map({safe(u): u for u in units.index})
mc = mc.set_index("unit")


def ci(t, u, p):
    if u not in t.index or pd.isna(t.loc[u, p]):
        return ""
    return f"{t.loc[u, p]:.2f} [{t.loc[u, p + '_lo']:.2f}, {t.loc[u, p + '_hi']:.2f}]"


def flag(t, u):
    return "" if u not in t.index or not t.loc[u, "flag"] else "*"


rows = []
for u in units.sort_values("trials").index:
    def d(col):
        if u not in mc.index or pd.isna(mc.loc[u, col]):
            return ""
        return f"{mc.loc[u, col]:+.0f} ({mc.loc[u, col + '_se']:.0f})"
    rows.append({
        "Study": label(u) + (" †" if u == "Weidacker_2020|s1" else ""),
        "N": int(units.loc[u, "subjects"]), "Choices": int(units.loc[u, "trials"]),
        "Choices/person": int(med[u]), "H2 model": "pooled-slope" if h2.loc[u, "model_used"] == "h2p" else "full",
        "m_k": ci(h2, u, "mu[m_k]") + flag(h2, u), "m_alpha": ci(h2, u, "mu[m_a]") + flag(h2, u),
        "g1": ci(r2f, u, "mu[g1]") + flag(r2f, u), "R2 flag": flag(r2, u),
        "dLOO R2F-R2 (SE)": d("r2f_minus_r2"), "dLOO H2-R2F (SE)": d("h2_minus_r2f"),
    })
t = pd.DataFrame(rows)
t.to_csv("results/table_s1.csv", index=False)

esc = lambda s: str(s).replace("&", r"\&").replace("_", r"\_").replace("†", r"$^\dagger$").replace("*", r"$^*$")
with open("table_s1.tex", "w", encoding="utf8") as f:
    f.write("\\begin{table}[p]\\centering\\scriptsize\\setlength{\\tabcolsep}{3pt}\n")
    f.write("\\caption{\\textbf{Per-study results.} Slopes are posterior means [94\\% HDI]; $^*$ flagged for convergence "
            "($\\widehat{R}>1.01$, more than one chain discarded, or $\\widehat{R}$ not computable). "
            "$\\Delta\\mathrm{elpd}_{\\mathrm{LOO}}$ with SE in parentheses. $^\\dagger$ fitted but not pooled.}\n")
    f.write("\\begin{tabular}{lrrrlllllrr}\\toprule\n")
    f.write("Study & $N$ & Choices & /person & H2 & $m_k$ & $m_\\alpha$ & $g_1$ & R2 & R2F$-$R2 & H2$-$R2F \\\\\\midrule\n")
    for r in rows:
        f.write(" & ".join(esc(v) for v in r.values()) + " \\\\\n")
    f.write("\\bottomrule\\end{tabular}\\label{tab:s1}\\end{table}\n")
print(t.head(3).to_string()); print(len(t), "rows")
