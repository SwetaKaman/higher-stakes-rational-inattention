"""Overview figure (figures/fig0_overview.pdf/png): analysis pipeline, the three models, and the headline result.

a  Pipeline from the ITC Database to evaluation, with icons and the actual counts.
b  The three models. Each card shows choice curves P(LL) against the relative value difference
   (V_LL - V_SS) / r for low, medium and high rewards. The steepness of these curves is the effective precision
   alpha * Delta V, which scales as r^e: R2 predicts e = 1.5; the fitted H2 (m_alpha = -0.74) gives e ~ 0.31 and the
   fitted R2F (g1 = -1.10) gives e ~ 0.34 (exact elasticities on the real designs, 14_r2_elasticity.py).
c  Total out-of-sample advantage over R2 on the 37 studies with all three fits.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle
from scipy.stats import norm

exec(open("07_make_figures.py", encoding="utf8").read().split("def safe(")[0])  # palette + style

DARK = {BLUE: "#1d4f8f", ORANGE: "#b5431b", AQUA: "#127a56", "grey": "#3a3a38"}
TINT = {BLUE: "#edf4fc", ORANGE: "#fdf0ea", AQUA: "#e9f7f2", "grey": "#f3f3f1"}
EDGE = {BLUE: "#2a78d6", ORANGE: "#eb6834", AQUA: "#1baf7a", "grey": "#6f6e6a", "white": "#cfcfcb"}
TINT["white"] = "white"
PILL = {BLUE: "#d9e7f8", ORANGE: "#fbdccd", AQUA: "#cfeee2"}
BS = chr(92)


def save(fig, name):
    fig.savefig(f"figures/{name}.pdf")
    fig.savefig(f"figures/{name}.png", dpi=600)
    plt.close(fig)
    print("wrote", name)


# headline numbers, computed from the results (37 studies with H2, R2 and R2F fits)
mc = pd.read_csv("results/model_comparison.csv").dropna(subset=["h2_minus_r2", "r2f_minus_r2"])
TOT = {"H2": mc.h2_minus_r2.sum(), "R2F": mc.r2f_minus_r2.sum(), "R2": 0.0}
K = len(mc)
N_H2_R2 = int((pd.read_csv("results/model_comparison.csv").h2_minus_r2 > 0).sum())
N_H2_ALL = int(pd.read_csv("results/model_comparison.csv").h2_minus_r2.notna().sum())
N_R2F_R2 = int((mc.r2f_minus_r2 > 0).sum())

fig = plt.figure(figsize=(FULL, 104 * MM))


def box(x, y, w, h, key, lw=0.9, r=0.012):
    fig.add_artist(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", transform=fig.transFigure,
                                  facecolor=TINT[key], edgecolor=EDGE[key], linewidth=lw, zorder=-5))


def arrow(x0, y0, x1, y1):
    fig.add_artist(FancyArrowPatch((x0, y0), (x1, y1), transform=fig.transFigure, arrowstyle="-|>",
                                   mutation_scale=7, color=INK, lw=0.8, zorder=2))


def icon_axes(cx, cy, w, h):
    ax = fig.add_axes([cx - w / 2, cy - h / 2, w, h])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    return ax


# ---------------- icons (simple flat line drawings) ----------------
def icon_database(ax, col):
    for i, off in enumerate([0.14, 0.07, 0.0]):
        x0, y0 = 0.12 + off, 0.12 + off
        ax.add_patch(Rectangle((x0, y0), 0.62, 0.5, facecolor="white", edgecolor=col, lw=0.7, zorder=i))
        if i == 2:
            ax.add_patch(Rectangle((x0, y0 + 0.4), 0.62, 0.1, facecolor=col, edgecolor=col, lw=0.7, zorder=3, alpha=0.85))
            for k in range(1, 4):
                ax.plot([x0, x0 + 0.62], [y0 + 0.1 * k, y0 + 0.1 * k], color=col, lw=0.4, zorder=3)
            for k in range(1, 4):
                ax.plot([x0 + 0.155 * k] * 2, [y0, y0 + 0.4], color=col, lw=0.4, zorder=3)


def icon_cleaning(ax, col):
    ax.add_patch(Rectangle((0.1, 0.18), 0.55, 0.55, facecolor="white", edgecolor=col, lw=0.7))
    ax.add_patch(Rectangle((0.1, 0.63), 0.55, 0.1, facecolor="#d9d8d4", edgecolor=col, lw=0.7))
    for k in range(1, 4):
        ax.plot([0.1, 0.65], [0.18 + 0.1125 * k, 0.18 + 0.1125 * k], color=col, lw=0.35)
        ax.plot([0.1 + 0.1375 * k] * 2, [0.18, 0.63], color=col, lw=0.35)
    ax.add_patch(Polygon([[0.55, 0.52], [0.92, 0.52], [0.77, 0.32], [0.77, 0.1], [0.7, 0.14], [0.7, 0.32]],
                         closed=True, facecolor=col, edgecolor=col, lw=0.6, zorder=4))


def icon_selection(ax, col, tick):
    ax.add_patch(Rectangle((0.2, 0.08), 0.6, 0.78, facecolor="white", edgecolor="#6f6e6a", lw=0.7))
    for k, yy in enumerate([0.68, 0.47, 0.26]):
        ax.plot([0.27, 0.32, 0.4], [yy, yy - 0.05, yy + 0.07], color=tick, lw=1.1, solid_capstyle="round")
        ax.plot([0.47, 0.72], [yy, yy], color="#9e9d98", lw=0.8)


def icon_posterior(ax, col):
    x = np.linspace(0, 1, 200)
    for mu, a in [(0.44, 0.35), (0.5, 0.55), (0.56, 0.9)]:
        y = 0.08 + 0.78 * np.exp(-0.5 * ((x - mu) / 0.11) ** 2)
        ax.fill_between(x, 0.08, y, color=col, alpha=0.18 * a + 0.05, lw=0)
        ax.plot(x, y, color=DARK[AQUA] if a > 0.8 else col, lw=0.8)
    ax.plot([0.05, 0.95], [0.08, 0.08], color="#6f6e6a", lw=0.6)


def icon_bars(ax, col):
    for k, hh in enumerate([0.28, 0.45, 0.6, 0.8]):
        ax.add_patch(Rectangle((0.22 + 0.16 * k, 0.08), 0.11, hh, facecolor=["#b9b8b3", "#9e9d98", "#7f7e7a", "#5d5c59"][k],
                               lw=0))
    ax.plot([0.15, 0.88], [0.08, 0.08], color="#3a3a38", lw=0.7)
    ax.plot([0.15, 0.15], [0.08, 0.92], color="#3a3a38", lw=0.7)


# ---------------- panel a: pipeline ----------------
fig.text(0.008, 0.985, "a", fontsize=10, fontweight="bold", va="top")
fig.text(0.04, 0.982, "Analysis pipeline", fontsize=8, fontweight="bold", va="top")
steps = [
    ("ITC Database", BLUE, icon_database, "100 studies\n11,852 people\n1.17 M choices"),
    ("Cleaning", "grey", icon_cleaning, "database exclusions\nSS > 0, LL > SS, LL later\n≥ 10 choices per person"),
    ("Study selection", ORANGE, icon_selection, "41 qualified\n40 fitted\n39 pooled"),
    ("Hierarchical\nBayesian fitting", AQUA, icon_posterior, "PyMC + NumPyro\nNUTS sampling\nthree models per study"),
    ("Evaluation", "grey", icon_bars, "PSIS-LOO model comparison\nmeta-analytic pooling\nmodel recovery"),
]
n, x0, gap, y, h = len(steps), 0.025, 0.026, 0.655, 0.29
w = (1 - 2 * x0 - gap * (n - 1)) / n
for i, (title, key, icon, txt) in enumerate(steps):
    x = x0 + i * (w + gap)
    box(x, y, w, h, key)
    ty = y + h - 0.02 if "\n" in title else y + h - 0.025
    fig.text(x + w / 2 + 0.012, ty, title, ha="center", va="top", fontsize=6.6, fontweight="bold", linespacing=1.1,
             color=DARK[key] if key != "grey" else INK)
    fig.text(x + 0.016, ty - 0.012, str(i + 1), ha="center", va="center", fontsize=5.8, fontweight="bold",
             color="white", bbox=dict(boxstyle="circle,pad=0.28", fc=DARK[key], ec="none"))
    ia = icon_axes(x + w / 2, y + h - 0.118, 0.075, 0.1)
    if icon is icon_selection:
        icon(ia, EDGE[key], ORANGE)
    else:
        icon(ia, EDGE[key])
    fig.text(x + w / 2, y + 0.022, txt, ha="center", va="bottom", fontsize=5.7, color=INK, linespacing=1.4)
    if i < n - 1:
        arrow(x + w + 0.003, y + h / 2, x + w + gap - 0.003, y + h / 2)

# ---------------- panel b: models ----------------
fig.text(0.008, 0.6, "b", fontsize=10, fontweight="bold", va="top")
fig.text(0.04, 0.597, "Three models of how reward magnitude $r$ shapes choice", fontsize=8, fontweight="bold", va="top")
bx0, bx1 = 0.025, 0.705
fig.add_artist(plt.Line2D([bx0 + 0.02, bx0 + 0.02, bx1 - 0.02, bx1 - 0.02], [0.522, 0.535, 0.535, 0.522],
                          transform=fig.transFigure, color=INK2, lw=0.6))
fig.text((bx0 + bx1) / 2, 0.535, "  Same data and choice rule, compared by out-of-sample fit  ", ha="center", va="center",
         fontsize=6, color=INK, backgroundcolor="white")
cards = [
    ("H2", "Descriptive", BLUE, "Discount rate and choice noise both\nvary freely with reward magnitude.",
     "$" + BS + "log k = c_k - m_k " + BS + "log r$\n$" + BS + "log" + BS + "alpha = c_" + BS + "alpha + m_" + BS + "alpha " + BS + "log r_" + BS + "Sigma$",
     0.31, "fitted: precision $" + BS + "propto r^{0.31}$"),
    ("R2", "Rational inattention", ORANGE, "Noise is set by an attention cost, so\nthe model fixes how noise scales.",
     "$k = 1/(" + BS + "beta r)$,   $" + BS + "alpha$ fixed by $" + BS + "beta$",
     1.5, "predicted: precision $" + BS + "propto r^{1.5}$"),
    ("R2F", "R2 + free noise slope", AQUA, "R2 with the noise slope $g_1$ left free\n($g_1 = 0$ recovers R2).",
     "$" + BS + "alpha = " + BS + "alpha_{R2}" + BS + "," + BS + "exp(g_0 + g_1 " + BS + "log r_" + BS + "Sigma)$",
     0.34, "fitted: precision $" + BS + "propto r^{0.34}$"),
]
cw, cg, cy, ch = (bx1 - bx0 - 2 * 0.012) / 3, 0.012, 0.03, 0.47
xv = np.linspace(-1, 1, 200)
mags = [("High", 3.0), ("Medium", 1.0), ("Low", 1 / 3)]
shades = {BLUE: ["#1d4f8f", "#2a78d6", "#8fbbe9"], ORANGE: ["#b5431b", "#eb6834", "#f5b195"],
          AQUA: ["#127a56", "#1baf7a", "#8fd8bd"]}
for i, (short, name, col, desc, eq, e, note) in enumerate(cards):
    cx = bx0 + i * (cw + cg)
    box(cx, cy, cw, ch, col)
    fig.text(cx + 0.012, cy + ch - 0.022, short, ha="left", va="top", fontsize=8, fontweight="bold", color=DARK[col])
    fig.text(cx + 0.012 + 0.032 + 0.006 * (len(short) - 2), cy + ch - 0.026, name, ha="left", va="top", fontsize=6.4,
             fontweight="bold", color=DARK[col])
    fig.text(cx + 0.012, cy + ch - 0.07, desc, ha="left", va="top", fontsize=5.6, color=INK, linespacing=1.35)
    fig.text(cx + 0.012, cy + ch - 0.13, eq, ha="left", va="top", fontsize=5.6, color=INK2, linespacing=1.5)
    ax = fig.add_axes([cx + 0.045, cy + 0.1, cw * 0.46, 0.155])
    for (lab, rr), c in zip(mags, shades[col]):
        s = 2.2 * rr ** e
        ax.plot(xv, 0.015 + 0.97 * norm.cdf(s * xv), color=c, lw=1.15, label=lab)
    ax.axhline(0.5, color=MUTED, lw=0.5, ls=(0, (2, 2)))
    ax.axvline(0, color=MUTED, lw=0.5, ls=(0, (2, 2)))
    ax.set_xlim(-1, 1); ax.set_ylim(0, 1.02)
    ax.set_xticks([-1, 0, 1]); ax.set_yticks([0, 0.5, 1]); ax.set_yticklabels(["0", "0.5", "1"])
    ax.tick_params(labelsize=5, length=2, pad=1.5)
    ax.set_xlabel("relative value difference", fontsize=5.3, labelpad=1.5)
    if i == 0:
        ax.set_ylabel("P(choose later)", fontsize=5.3, labelpad=1.5)
    for sp in ["left", "bottom"]:
        ax.spines[sp].set_color(MUTED); ax.spines[sp].set_linewidth(0.6)
    leg = ax.legend(title="reward", loc="center left", bbox_to_anchor=(1.03, 0.5), frameon=False, fontsize=5,
                    title_fontsize=5.2, handlelength=1.2, handletextpad=0.4, labelspacing=0.3)
    leg._legend_box.align = "left"
    fig.text(cx + cw / 2, cy + 0.024, note, ha="center", va="center", fontsize=5.8, color=DARK[col], style="italic",
             bbox=dict(boxstyle="round,pad=0.45,rounding_size=0.5", fc=PILL[col], ec="none"))

# ---------------- panel c: headline result ----------------
fig.text(0.722, 0.6, "c", fontsize=10, fontweight="bold", va="top")
fig.text(0.752, 0.597, "Out-of-sample fit", fontsize=8, fontweight="bold", va="top")
fig.text(0.752, 0.55, f"ΔELPD vs R2, {K} studies", fontsize=6, color=INK2, va="top")
box(0.73, 0.03, 0.245, 0.47, "white")
ac = fig.add_axes([0.785, 0.25, 0.135, 0.22])
order = [("H2", BLUE), ("R2F", AQUA), ("R2", ORANGE)]
for j, (mname, col) in enumerate(order):
    yy = 2 - j
    v = TOT[mname]
    ac.plot([0, v], [yy, yy], color=col, lw=4.2, solid_capstyle="butt", alpha=0.85)
    ac.scatter([v], [yy], s=30, color=col, edgecolor="white", lw=0.6, zorder=3)
    ac.text(-1500, yy, mname, ha="right", va="center", fontsize=7, fontweight="bold", color=DARK[col])
    ac.text(v + 2600, yy, f"+{v:,.0f}" if v > 0 else "0 (reference)", ha="left", va="center", fontsize=6, color=INK)
ac.axvline(0, color=INK, lw=0.7)
ac.set_xlim(-200, 31000); ac.set_ylim(-0.6, 2.6)
for sp in ["left", "top", "right"]:
    ac.spines[sp].set_visible(False)
ac.spines["bottom"].set_color(MUTED); ac.spines["bottom"].set_linewidth(0.6)
ac.set_yticks([])
ac.set_xticks([0, 10000, 20000, 30000]); ac.set_xticklabels(["0", "10k", "20k", "30k"])
ac.tick_params(axis="x", labelsize=5, length=2, pad=1.5)
ac.set_xlabel("ΔELPD vs R2", fontsize=5.6, labelpad=2)
fig.text(0.8525, 0.085, f"H2 beats R2 in {N_H2_R2} of {N_H2_ALL} studies;\nR2F beats R2 in {N_R2F_R2} of {K}.",
         ha="center", va="center", fontsize=6, color=INK, linespacing=1.4,
         bbox=dict(boxstyle="round,pad=0.7,rounding_size=0.6", fc="#efefed", ec="none"))
save(fig, "fig0_overview")
