"""Moderator analysis: do study-level design features explain between-study differences, and does the
failure of rational inattention hold in every kind of study?

Outcomes (per study, posterior mean and posterior SD as standard error):
  m_k, m_alpha      magnitude slopes of the discount rate and of choice sensitivity (main H2)
  g1                stochasticity slope added to R2 (main R2F; g1 = 0 is pure R2)
  dR2F              elpd_loo(R2F) - elpd_loo(R2), per 1,000 choices
  dH2               elpd_loo(H2) - elpd_loo(R2F), per 1,000 choices
Moderators (ITC Database study metadata, plus design features of the unit):
  incentive (potentially real vs hypothetical), online vs laboratory, time pressure, Kirby questionnaire vs other
  procedures, log magnitude range (residual 5-95% span), log median choices per person.
Method: random-effects meta-regression, one moderator at a time, REML tau^2 with the Knapp-Hartung
adjustment (t distribution, k - 2 df). Holm correction across all moderator x outcome tests. Subgroup pooled
estimates use the same REML random-effects model with an intercept only.

Outputs: results/moderators_metareg.csv, results/moderators_subgroups.csv, figures/figS1_moderators.pdf/png
"""
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import optimize, stats

exec(open("07_make_figures.py", encoding="utf8").read().split("def safe(")[0])  # palette + style


def safe(u):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", u)


def save(fig, name):
    fig.savefig(f"figures/{name}.pdf")
    fig.savefig(f"figures/{name}.png")
    plt.close(fig)
    print("wrote", name)


# ---------------- data ----------------
h2m = pd.read_csv("results/h2main_per_study.csv")
r2f = pd.read_csv("results/r2fmain_per_study.csv")
mc = pd.read_csv("results/model_comparison.csv").set_index("unit")
us = pd.read_csv("unit_summary.csv").set_index("unit")
mod = pd.read_csv("results/study_moderators_raw.csv").set_index("unit")
med = pd.read_csv("results/median_trials_per_subject.csv").set_index("unit").iloc[:, 0]

pool_units = h2m[h2m.in_pool].unit.tolist()
D = pd.DataFrame(index=pool_units)
D["m_k"], D["m_k_se"] = h2m.set_index("unit").loc[pool_units, ["mu[m_k]", "mu[m_k]_sd"]].T.values
D["m_a"], D["m_a_se"] = h2m.set_index("unit").loc[pool_units, ["mu[m_a]", "mu[m_a]_sd"]].T.values
r = r2f.set_index("unit").reindex(pool_units)
D["g1"], D["g1_se"] = r["mu[g1]"], r["mu[g1]_sd"]
c = mc.reindex([safe(u) for u in pool_units])
tr = us.loc[pool_units, "trials"].values
D["dR2F"], D["dR2F_se"] = (1000 * c.r2f_minus_r2 / tr).values, (1000 * c.r2f_minus_r2_se / tr).values
D["dH2"], D["dH2_se"] = (1000 * c.h2_minus_r2f / tr).values, (1000 * c.h2_minus_r2f_se / tr).values

M = mod.loc[pool_units]
D["Real incentive"] = (M.incentivization == "Potentially Real").astype(float).values
D["Online"] = (M.online_study.astype(str) == "1").astype(float).values
D["Time pressure"] = (M.time_pressure.astype(str) == "1").astype(float).values
D["Kirby questionnaire"] = (M.procedure == "Kirby Questionnaire").astype(float).values
D["log magnitude range"] = np.log(us.loc[pool_units, "resid_mag_span"].clip(upper=200)).values
D["log choices per person"] = np.log(med.reindex(pool_units)).values
D.to_csv("results/moderators_data.csv")

OUT = {"m_k": "Discount slope $m_k$", "m_a": "Sensitivity slope $m_" + chr(92) + "alpha$",
       "g1": "Noise slope $g_1$", "dR2F": "R2F $-$ R2 (elpd / 1,000)", "dH2": "H2 $-$ R2F (elpd / 1,000)"}
BIN = ["Real incentive", "Online", "Time pressure", "Kirby questionnaire"]
CONT = ["log magnitude range", "log choices per person"]
LEVELS = {"Real incentive": ("hypothetical", "potentially real"), "Online": ("laboratory", "online"),
          "Time pressure": ("no time pressure", "time pressure"), "Kirby questionnaire": ("other procedures", "Kirby questionnaire")}


def reml_metareg(y, v, X):
    """Random-effects meta-regression, REML tau^2, Knapp-Hartung SEs. Returns beta, se, df, tau2."""
    k, p = X.shape

    def nll(log_tau2):
        w = 1 / (v + np.exp(log_tau2))
        XtW = X.T * w
        A = XtW @ X
        b = np.linalg.solve(A, XtW @ y)
        res = y - X @ b
        return 0.5 * (np.sum(np.log(v + np.exp(log_tau2))) + np.linalg.slogdet(A)[1] + np.sum(w * res ** 2))

    opt = optimize.minimize_scalar(nll, bounds=(-20, np.log(10 * np.var(y) + 1e-9)), method="bounded")
    tau2 = np.exp(opt.x) if nll(opt.x) < nll(-20) else 0.0
    w = 1 / (v + tau2)
    A = (X.T * w) @ X
    b = np.linalg.solve(A, (X.T * w) @ y)
    res = y - X @ b
    q = max(1.0, np.sum(w * res ** 2) / (k - p))    # Knapp-Hartung scale (truncated at 1)
    se = np.sqrt(np.diag(np.linalg.inv(A)) * q)
    return b, se, k - p, tau2


rows, subs = [], []
for o in OUT:
    for m in BIN + CONT:
        d = D[[o, o + "_se", m]].dropna()
        X = np.column_stack([np.ones(len(d)), d[m].values])
        b, se, df, tau2 = reml_metareg(d[o].values, d[o + "_se"].values ** 2, X)
        _, _, _, tau2_0 = reml_metareg(d[o].values, d[o + "_se"].values ** 2, X[:, :1])
        t = b[1] / se[1]
        rows.append(dict(outcome=o, moderator=m, k=len(d), n_level1=int(d[m].sum()) if m in BIN else np.nan,
                         beta=b[1], ci_lo=b[1] - stats.t.ppf(0.975, df) * se[1],
                         ci_hi=b[1] + stats.t.ppf(0.975, df) * se[1], p=2 * stats.t.sf(abs(t), df),
                         R2_het=max(0.0, 1 - tau2 / tau2_0) if tau2_0 > 0 else 0.0))
        if m in BIN:
            for lev in (0, 1):
                dd = d[d[m] == lev]
                if len(dd) < 2:
                    continue
                b0, se0, df0, _ = reml_metareg(dd[o].values, dd[o + "_se"].values ** 2, np.ones((len(dd), 1)))
                h = stats.t.ppf(0.975, df0) * se0[0]
                subs.append(dict(outcome=o, moderator=m, level=LEVELS[m][lev], k=len(dd), est=b0[0],
                                 ci_lo=b0[0] - h, ci_hi=b0[0] + h))
    d = D[[o, o + "_se"]].dropna()
    b0, se0, df0, _ = reml_metareg(d[o].values, d[o + "_se"].values ** 2, np.ones((len(d), 1)))
    h = stats.t.ppf(0.975, df0) * se0[0]
    subs.append(dict(outcome=o, moderator="All studies", level="all", k=len(d), est=b0[0], ci_lo=b0[0] - h, ci_hi=b0[0] + h))

R = pd.DataFrame(rows)
order = np.argsort(R.p.values)
holm = np.empty(len(R))
running = 0.0
for rank, i in enumerate(order):
    running = max(running, min(1.0, (len(R) - rank) * R.p.values[i]))
    holm[i] = running
R["p_holm"] = holm
R.to_csv("results/moderators_metareg.csv", index=False)
S = pd.DataFrame(subs)
S.to_csv("results/moderators_subgroups.csv", index=False)
pd.set_option("display.width", 200)
print(R.round(3).to_string(index=False))
print(S.round(3).to_string(index=False))

# ---------------- figure S1: subgroup estimates ----------------
fig, axes = plt.subplots(1, 3, figsize=(FULL, 92 * MM), sharey=True, gridspec_kw=dict(wspace=0.2))
labels = ["All studies"] + [f"{LEVELS[m][lev]}" for m in BIN for lev in (1, 0)]
ypos = {}
y = 0
for lab in labels[::-1]:
    ypos[lab] = y
    y += 1
groups = [(m, LEVELS[m]) for m in BIN]
for ax, (o, col, ref) in zip(axes, [("m_k", BLUE, 1.0), ("m_a", ORANGE, 0.5), ("g1", AQUA, 0.0)]):
    s = S[S.outcome == o]
    for _, rr in s.iterrows():
        yy = ypos["All studies" if rr.moderator == "All studies" else rr.level]
        big = rr.moderator == "All studies"
        ax.hlines(yy, rr.ci_lo, rr.ci_hi, color=col, lw=1.6 if big else 1.1, alpha=0.9 if big else 0.6)
        ax.scatter(rr.est, yy, s=26 if big else 14, color=col if big else "white", edgecolor=col, lw=1, zorder=3,
                   marker="D" if big else "o")
        pass

    ax.axvline(ref, color=MUTED, lw=0.7, ls=(0, (3, 2)))
    ax.axvline(0, color=INK2, lw=0.6)
    for i in range(len(BIN)):
        ax.axhspan(ypos[LEVELS[BIN[i]][0]] - 0.5, ypos[LEVELS[BIN[i]][1]] + 0.5, color=GRID if i % 2 == 0 else "white",
                   alpha=0.45, lw=0, zorder=0)
    ax.set_title(f"{'abc'[['m_k', 'm_a', 'g1'].index(o)]}  {OUT[o]}")
    ax.set_xlabel("pooled estimate (95% CI)")
    ax.grid(True, axis="x", color=GRID, lw=0.5, zorder=0)
axes[0].set_yticks([ypos[l] for l in labels])
kk = S[(S.outcome == "m_k")].set_index("level").k; kk["All studies"] = kk.pop("all")
axes[0].set_yticklabels([f"{l}  (k = {kk[l]})" for l in labels], fontsize=6)
axes[0].get_yticklabels()[0].set_fontweight("bold")
fig.text(0.5, 0.005, "Dashed line: rational-inattention prediction.   Diamond: all studies.   Circles: subgroups.", ha="center", va="bottom", fontsize=5.6, color=INK2)
fig.subplots_adjust(left=0.2, bottom=0.17)
save(fig, "figS1_moderators")
