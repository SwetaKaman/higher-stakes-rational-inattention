"""Posterior predictive checks: observed vs model-predicted choice probabilities (main H2, R2, main R2F).

For every pooled study, the posterior-mean P(choose larger-later) per trial is computed from 300 draws
of the saved posterior (fits/<model>/<unit>_theta.nc) using the same likelihood code as the fits.
Writes results/ppc_trials.csv.gz (per-trial predictions) and figures/fig6_ppc.(pdf|png):
  a  calibration: observed proportion LL vs predicted, pooled over all trials (20 quantile bins);
  b  misfit across reward magnitude: observed − predicted P(LL), by within-study magnitude quintile and
     by how strongly the choice was predicted (sharp vs uncertain trials).
"""
import importlib.util
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.special as sps
import xarray as xr

spec = importlib.util.spec_from_file_location("fm", "02_fit_models.py")
fm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fm)
exec(open("07_make_figures.py", encoding="utf8").read().split("# ---------------- Figure 1")[0])  # style


def safe(u):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", u)


REUSE = os.path.exists("results/ppc_trials.csv.gz")
d = None if REUSE else pd.read_csv("clean_data.csv.gz", low_memory=False)
h2 = pd.read_csv("results/h2main_per_study.csv").set_index("unit")
r2f = pd.read_csv("results/r2fmain_per_study.csv").set_index("unit")
units = h2.index[h2.in_pool]
rng = np.random.default_rng(2026)
out = []
for u in ([] if REUSE else units):
    g = fm.prepare(d, u)
    row = dict()
    for label, model in [("H2", h2.loc[u, "model_used"]), ("R2", "r2"), ("R2F", r2f.loc[u, "model_used"])]:
        f = f"fits/{model}/{safe(u)}_theta.nc"
        if not os.path.exists(f):
            continue
        ds = xr.open_dataset(f)
        th = ds["theta_full" if model in fm.BASE else "theta"].values.reshape(-1, ds.sizes["subject"], ds["theta_full" if model in fm.BASE else "theta"].shape[-1])
        eps = ds["eps"].values.reshape(-1)
        pick = rng.choice(th.shape[0], size=min(300, th.shape[0]), replace=False)
        subj = list(ds["subject"].values)
        sidx = pd.Index(subj).get_indexer(g.subj_ident.values)
        acc = np.zeros(len(g))
        for k in pick:
            p = fm.prob(np, lambda x: 0.5 * sps.erfc(-x / np.sqrt(2)), th[k][sidx], eps[k], g, fm.BASE.get(model, model))
            acc += np.clip(p, 1e-9, 1 - 1e-9)
        row[label] = acc / len(pick)
        ds.close()
    lm = g.lsum - g.lsum.mean()
    q = pd.qcut(lm.rank(method="first"), 5, labels=False)
    out.append(pd.DataFrame({"unit": u, "choice": g.choice.values, "mag_quintile": q.values, **row}))
    print(u, "done", flush=True)
if REUSE:
    ppc = pd.read_csv("results/ppc_trials.csv.gz")
else:
    ppc = pd.concat(out, ignore_index=True)
    ppc.to_csv("results/ppc_trials.csv.gz", index=False)


def calib_slope(y, p):
    """Logistic regression y ~ a + b*logit(p) by IRLS; returns b and its SE."""
    x = np.column_stack([np.ones(len(p)), np.log(p / (1 - p))])
    w = np.zeros(2)
    for _ in range(50):
        eta = x @ w; mu = 1 / (1 + np.exp(-eta)); W = mu * (1 - mu)
        H = x.T @ (x * W[:, None]); g = x.T @ (y - mu)
        step = np.linalg.solve(H, g); w += step
        if np.abs(step).max() < 1e-8:
            break
    return w[1], np.sqrt(np.linalg.inv(H)[1, 1])

models = [("R2", ORANGE), ("R2F", AQUA), ("H2", BLUE)]
fig, (a, b) = plt.subplots(1, 2, figsize=(FULL, 72 * MM), gridspec_kw=dict(wspace=0.32))
a.plot([0, 1], [0, 1], color=MUTED, lw=0.7, ls=(0, (3, 2)))
for m, c in models:
    t = ppc.dropna(subset=[m])
    bins = pd.qcut(t[m].rank(method="first"), 20, labels=False)
    agg = t.groupby(bins).agg(pred=(m, "mean"), obs=("choice", "mean"), n=("choice", "size"))
    se = np.sqrt(agg.obs * (1 - agg.obs) / agg.n)
    a.errorbar(agg.pred, agg.obs, yerr=2 * se, fmt="o-", color=c, ms=3.2, lw=1.0, elinewidth=0.6, label=m,
               markeredgecolor="white", markeredgewidth=0.4)
a.set_xlabel("Predicted P(larger-later)"); a.set_ylabel("Observed proportion larger-later")
a.set_title("a  Calibration (all trials)")
a.set_xlim(0, 1); a.set_ylim(0, 1); a.set_aspect("equal")
a.grid(True, color=GRID, lw=0.5); a.legend(frameon=False, loc="upper left")
# b: calibration slope (sharpness) by within-study magnitude quintile; 1 = correctly sharp, <1 = overconfident
x = np.arange(5)
rows = []
for m, c in models:
    t = ppc.dropna(subset=[m])
    res = [calib_slope(t.choice.values[t.mag_quintile == q], np.clip(t[m].values[t.mag_quintile == q], 1e-6, 1 - 1e-6))
           for q in range(5)]
    est = np.array([r[0] for r in res]); se = np.array([r[1] for r in res])
    rows += [dict(model=m, quintile=q + 1, slope=e, se=s_) for q, (e, s_) in enumerate(zip(est, se))]
    b.errorbar(x + {"R2": -0.1, "R2F": 0, "H2": 0.1}[m], est, yerr=2 * se, color=c, marker="o", ms=3.5, lw=1.1,
               elinewidth=0.7, markeredgecolor="white", markeredgewidth=0.4, label=m)
pd.DataFrame(rows).to_csv("results/ppc_calibration_slopes.csv", index=False)
b.axhline(1, color=INK2, lw=0.7)
b.text(4.35, 1.0, "correct" + chr(10) + "sharpness", fontsize=5.8, color=INK2, va="center")
b.set_xticks(x); b.set_xticklabels(["smallest", "2", "3", "4", "largest"])
b.set_xlim(-0.4, 4.9)
b.set_xlabel("Reward magnitude (within-study quintile)")
b.set_ylabel("Calibration slope (± 2 SE)")
b.set_title("b  Is choice sharpness right at every magnitude?")
b.grid(True, axis="y", color=GRID, lw=0.5)
b.legend(frameon=False, loc="lower left")
save(fig, "fig6_ppc")
