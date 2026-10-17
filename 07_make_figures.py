"""Publication figures for the ITC rational-inattention paper.

Reads the per-study and pooled tables written by 03_summarise.py and writes figures/fig*.pdf (vector)
and figures/fig*.png (300 dpi). Palette: reference data-viz palette, slots 1-3 (blue, orange, aqua),
neutral greys for text, axes and reference lines. Flagged (poorly converged) fits are drawn hollow.
"""
import os
import re

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

os.makedirs("figures", exist_ok=True)
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
mpl.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7.5, "axes.titlesize": 8.5, "axes.labelsize": 7.5,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "pdf.fonttype": 42, "savefig.dpi": 300, "figure.dpi": 150, "axes.titleweight": "bold",
    "axes.titlelocation": "left",
})
MM = 1 / 25.4
FULL, HALF = 180 * MM, 88 * MM


def safe(u):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", u)


def label(u):
    name, sub = u.split("|")
    name = name.replace("_", " ").replace("Kr�plin", "Kräplin")
    return name if sub == "s1" and not u.startswith(("Cao_2021_Exp2", "Keidel_2024", "Thome_2023", "Yang_2022")) \
        else f"{name} ({sub[1:]})"


def save(fig, name):
    fig.savefig(f"figures/{name}.pdf", bbox_inches="tight")
    fig.savefig(f"figures/{name}.png", bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


units = pd.read_csv("unit_summary.csv").query("magnitude_varying").set_index("unit")
units = units.drop(index="Thrailkill_2022b|s1", errors="ignore")
med = pd.read_csv("results/median_trials_per_subject.csv", index_col=0).iloc[:, 0]
h2 = pd.read_csv("results/h2main_per_study.csv").set_index("unit")
r2f = pd.read_csv("results/r2fmain_per_study.csv").set_index("unit")
pool_h2 = pd.read_csv("results/h2main_pooled.csv")
pool_r2f = pd.read_csv("results/r2fmain_pooled.csv")
mc = pd.read_csv("results/model_comparison.csv")
safe2unit = {safe(u): u for u in units.index}
mc["unit"] = mc.unit.map(safe2unit)
mc = mc.dropna(subset=["unit"]).set_index("unit")


def pooled(tab, param, subset):
    r = tab[(tab.param == param) & (tab.studies == subset)].iloc[0]
    return r.pooled, r.ci_lo, r.ci_hi, int(r.k)


# ---------------- Figure 1: the dataset ----------------
fig, (a, b) = plt.subplots(1, 2, figsize=(FULL, 62 * MM), gridspec_kw=dict(width_ratios=[1.15, 1], wspace=0.32))
d = units.assign(med=med.reindex(units.index))
short = d.med < 60
for m, col, lab in [(~short, BLUE, "≥ 60 choices per person (full H2)"),
                    (short, ORANGE, "< 60 choices per person (pooled-slope H2)")]:
    a.scatter(d.subjects[m], d.med[m], s=np.clip(d.trials[m] / 350, 9, 90), color=col, alpha=0.85,
              edgecolor="white", linewidth=0.6, label=lab, zorder=3)
a.set_xscale("log"); a.set_yscale("log")
a.axhline(60, color=MUTED, lw=0.7, ls=(0, (3, 2)), zorder=1)
a.text(a.get_xlim()[0] * 1.15 if a.get_xlim()[0] > 0 else 12, 64, "60-choice rule", color=INK2, fontsize=6)
a.set_xlabel("People per study (log scale)"); a.set_ylabel("Median choices per person (log scale)")
a.set_title("a  Studies differ in size and design")
a.grid(True, which="major", color=GRID, lw=0.5, zorder=0)
leg = a.legend(frameon=False, loc="upper right", handletextpad=0.3, borderaxespad=0.2, markerscale=1)
for h in leg.legend_handles:
    h.set_sizes([22])
a.text(0.02, 0.02, f"{len(d)} study × subset units, {int(d.subjects.sum()):,} people, {int(d.trials.sum()):,} choices\n"
       "marker area ∝ number of choices", transform=a.transAxes, fontsize=6, color=INK2, va="bottom")
span = d.resid_mag_span.clip(upper=200).sort_values()
b.barh(np.arange(len(span)), span.values, color=BLUE, height=0.72, zorder=3)
b.set_xscale("log"); b.set_yticks([])
b.axvline(2, color=MUTED, lw=0.7, ls=(0, (3, 2)))
b.set_ylim(-1, len(span) + 2.2)  # headroom above the bars for the threshold label
b.text(2.12, len(span) + 0.9, "← 2× inclusion threshold (dashed line)", fontsize=6, color=INK2, ha="left", va="center")
b.set_xlabel("Reward-magnitude range independent of ratio and delay\n(5–95% span of residual log amount; capped at 200×)")
b.set_ylabel("Studies (sorted)")
b.set_title("b  Magnitude varies independently of ratio and delay")
b.grid(True, axis="x", color=GRID, lw=0.5, zorder=0)
save(fig, "fig1_dataset")


# ---------------- Figure 2: model comparison ----------------
cmp = mc.join(units.trials)
pairs = [("r2f_minus_r2", "R2F vs R2", ORANGE), ("h2_minus_r2f", "H2 vs R2F", BLUE)]
fig, axes = plt.subplots(1, 2, figsize=(FULL, 118 * MM), sharey=False, gridspec_kw=dict(wspace=0.55))
for ax, (col, ttl, c) in zip(axes, pairs):
    t = cmp[[col, col + "_se", "trials"]].dropna()
    t = t.assign(per=1000 * t[col] / t.trials, se=1000 * t[col + "_se"] / t.trials).sort_values("per")
    y = np.arange(len(t))
    ax.hlines(y, t.per - 2 * t.se, t.per + 2 * t.se, color=c, lw=1.2, alpha=0.55, zorder=2)
    ax.scatter(t.per, y, s=14, color=c, edgecolor="white", linewidth=0.5, zorder=3)
    ax.axvline(0, color=INK2, lw=0.7)
    ax.set_yticks(y); ax.set_yticklabels([label(u) for u in t.index], fontsize=5.6)
    ax.set_xlabel("Δ elpd(LOO) per 1,000 choices (± 2 SE)")
    tot, se = cmp[col].sum(), np.sqrt((cmp[col + "_se"] ** 2).sum())
    n_first = int((t[col] > 2 * t[col + "_se"]).sum()); n_second = int((t[col] < -2 * t[col + "_se"]).sum())
    ax.set_title(f"{'ab'[pairs.index((col, ttl, c))]}  {ttl}\n{ttl.split(' vs ')[0]} better in {n_first} of {len(t)} studies",
                 linespacing=1.35)
    ax.text(0.98, 0.02, f"Total Δ elpd = {tot:+,.0f} (SE {se:,.0f})\nfavours second model: {n_second}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=6.3, color=INK2)
    ax.grid(True, axis="x", color=GRID, lw=0.5, zorder=0)
    ax.set_xlim(left=min(-5, (t.per - 2 * t.se).min() * 1.1))
save(fig, "fig2_model_comparison")


# ---------------- Figure 3: magnitude slopes (main H2) ----------------
t = h2[h2.in_pool].copy()
t = t.sort_values("mu[m_k]")
y = np.arange(len(t))
fig, axes = plt.subplots(1, 2, figsize=(FULL, 128 * MM), sharey=True, gridspec_kw=dict(wspace=0.08))
for ax, p, c, ttl, ref in [(axes[0], "mu[m_k]", BLUE, r"a  Discount-rate slope $m_k$", None),
                           (axes[1], "mu[m_a]", ORANGE, "b  Sensitivity slope $m_{" + chr(92) + "alpha}$", [0, -1])]:
    ax.hlines(y, t[p + "_lo"], t[p + "_hi"], color=c, lw=1.1, alpha=0.6, zorder=2)
    ok = ~t.flag
    ax.scatter(t[p][ok], y[ok], s=13, color=c, edgecolor="white", linewidth=0.5, zorder=3, label="converged")
    ax.scatter(t[p][~ok], y[~ok], s=13, facecolor="white", edgecolor=c, linewidth=0.9, zorder=3,
               label="flagged (r̂ > 1.01)")
    ax.axvline(0, color=INK2, lw=0.7)
    if ref:
        ax.axvline(-1, color=MUTED, lw=0.7, ls=(0, (3, 2)))
        ax.text(-1.02, len(t) + 0.3, "scale-free\nnoise", ha="right", fontsize=5.8, color=INK2, va="bottom")
        ax.text(0.03, len(t) + 0.3, "constant α", ha="left", fontsize=5.8, color=INK2, va="bottom")
    lines = []
    for sub, yy in [("all", -2.0), ("unflagged", -3.6)]:
        est, lo, hi, kk = pooled(pool_h2, p, sub)
        dia = np.array([[lo, yy], [est, yy + 0.6], [hi, yy], [est, yy - 0.6]])
        ax.add_patch(plt.Polygon(dia, closed=True, facecolor=c if sub == "all" else "white", edgecolor=c, lw=0.9))
        lines.append(f"{'all' if sub == 'all' else 'converged'}, k={kk}: {est:.2f} [{lo:.2f}, {hi:.2f}]")
    ax.set_title(ttl); ax.set_xlabel("Study-level mean slope (94% HDI)")
    ax.grid(True, axis="x", color=GRID, lw=0.5, zorder=0)
axes[0].set_yticks(list(y) + [-2.0, -3.6])
axes[0].set_yticklabels([label(u) + ("  ◦" if mu == "h2p" else "") for u, mu in zip(t.index, t.model_used)]
                        + ["Pooled: all studies", "Pooled: converged only"], fontsize=5.6)
for tl in axes[0].get_yticklabels()[-2:]:
    tl.set_fontweight("bold"); tl.set_color(INK)
axes[0].set_ylim(-4.6, len(t) + 2.6)
axes[0].legend(frameon=False, loc="lower right", fontsize=6, handletextpad=0.2, borderaxespad=0.6)
axes[0].text(0.0, -0.13, "◦ short questionnaire: slopes estimated once per study (pooled-slope H2)",
             transform=axes[0].transAxes, fontsize=6, color=INK2)
save(fig, "fig3_magnitude_slopes")


# ---------------- Figure 4: R2F stochasticity slope ----------------
t = r2f[r2f.in_pool].sort_values("mu[g1]")
y = np.arange(len(t))
fig, ax = plt.subplots(figsize=(HALF * 1.15, 128 * MM))
ax.hlines(y, t["mu[g1]_lo"], t["mu[g1]_hi"], color=AQUA, lw=1.1, alpha=0.65, zorder=2)
ok = ~t.flag
ax.scatter(t["mu[g1]"][ok], y[ok], s=13, color=AQUA, edgecolor="white", linewidth=0.5, zorder=3, label="converged")
ax.scatter(t["mu[g1]"][~ok], y[~ok], s=13, facecolor="white", edgecolor=AQUA, linewidth=0.9, zorder=3, label="flagged")
ax.axvline(0, color=INK2, lw=0.8)
ax.text(0.04, len(t) + 0.3, "R2 prediction\n(g1 = 0)", fontsize=5.8, color=INK2, va="bottom")
for sub, yy in [("all", -2.0), ("unflagged", -3.6)]:
    est, lo, hi, kk = pooled(pool_r2f, "mu[g1]", sub)
    ax.add_patch(plt.Polygon([[lo, yy], [est, yy + 0.6], [hi, yy], [est, yy - 0.6]], closed=True,
                             facecolor=AQUA if sub == "all" else "white", edgecolor=AQUA, lw=0.9))
ax.set_yticks(list(y) + [-2.0, -3.6])
ax.set_yticklabels([label(u) for u in t.index] + ["Pooled: all studies", "Pooled: converged only"], fontsize=5.6)
for tl in ax.get_yticklabels()[-2:]:
    tl.set_fontweight("bold"); tl.set_color(INK)
ax.set_ylim(-4.6, len(t) + 1.8)
ax.set_xlabel("Free stochasticity slope $g_1$ (94% HDI)")
ax.set_title("Choice noise deviates from the R2 prediction")
ax.grid(True, axis="x", color=GRID, lw=0.5, zorder=0)
ax.legend(frameon=False, loc="lower right", fontsize=6)
save(fig, "fig4_stochasticity_slope")


# ---------------- Figure 5: are the two magnitude effects linked? ----------------
fig, (a, b) = plt.subplots(1, 2, figsize=(FULL, 70 * MM), gridspec_kw=dict(wspace=0.85, width_ratios=[1, 1.1]))
t = h2[h2.in_pool]
a.errorbar(t["mu[m_k]"], t["mu[m_a]"], xerr=t["mu[m_k]_sd"], yerr=t["mu[m_a]_sd"], fmt="none",
           ecolor=GRID, elinewidth=0.8, zorder=1)
for m, fc in [(~t.flag, BLUE), (t.flag, "white")]:
    a.scatter(t["mu[m_k]"][m], t["mu[m_a]"][m], s=np.clip(t.n_subj[m] / 6, 8, 70), facecolor=fc, edgecolor=BLUE,
              linewidth=0.8, zorder=3)
r = np.corrcoef(t["mu[m_k]"], t["mu[m_a]"])[0, 1]
a.set_xlabel("$m_k$ (study mean)"); a.set_ylabel("$m_{" + chr(92) + "alpha}$ (study mean)")
a.set_title(f"a  Across studies (r = {r:.2f}, k = {len(t)})")
a.axhline(0, color=INK2, lw=0.6); a.axvline(0, color=INK2, lw=0.6)
a.grid(True, color=GRID, lw=0.5, zorder=0)
a.text(0.98, 0.03, "marker area ∝ people; hollow = flagged", transform=a.transAxes, ha="right", fontsize=6, color=INK2)
t2 = h2[h2.in_pool & (h2.model_used == "h2")].sort_values("corr[m_k, m_a]")
y = np.arange(len(t2))
b.hlines(y, t2["corr[m_k, m_a]_lo"], t2["corr[m_k, m_a]_hi"], color=BLUE, lw=1.1, alpha=0.6)
b.scatter(t2["corr[m_k, m_a]"][~t2.flag], y[~t2.flag], s=13, color=BLUE, edgecolor="white", linewidth=0.5, zorder=3)
b.scatter(t2["corr[m_k, m_a]"][t2.flag], y[t2.flag], s=13, facecolor="white", edgecolor=BLUE, linewidth=0.9, zorder=3)
b.axvline(0, color=INK2, lw=0.7)
est, lo, hi, kk = pooled(pool_h2, "corr[m_k, m_a]", "all")
b.add_patch(plt.Polygon([[lo, -2], [est, -1.4], [hi, -2], [est, -2.6]], closed=True, facecolor=BLUE, edgecolor=BLUE))
b.set_yticks(list(y) + [-2])
b.set_yticklabels([label(u) for u in t2.index] + [f"Pooled (k={kk})"], fontsize=5.4)
b.get_yticklabels()[-1].set_fontweight("bold")
b.set_ylim(-3.2, len(t2))
b.set_xlim(-1, 1); b.set_xticks([-1, -0.5, 0, 0.5, 1])
b.set_xlabel("Correlation between people's $m_k$ and $m_{" + chr(92) + "alpha}$ (94% HDI)")
b.set_title("b  Within studies (full-model studies)")
b.grid(True, axis="x", color=GRID, lw=0.5, zorder=0)
save(fig, "fig5_slope_correlation")
