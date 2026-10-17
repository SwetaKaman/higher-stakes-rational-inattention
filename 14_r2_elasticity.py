"""Numerical check of the formal predictions of R2 on the real designs (Proposition 1 and 2 in the paper).

For every trial of the 39 pooled studies, using each participant's posterior-mean log beta from the R2 fit and the
participant's running reward variance s2_u (held fixed), we compute the local elasticities of
  alpha            (choice sensitivity),
  dV = V_LL - V_SS (value difference), and
  z = alpha * dV   (effective precision, the argument of the probit)
with respect to a common scaling of both amounts, r -> lambda * r, at lambda = 1 (central finite difference in log
lambda). Also the elasticity of the discount rate k (exactly 1 by construction).
Output: results/r2_elasticities.csv (per study) and printed summary.
"""
import importlib.util
import os
import re

import numpy as np
import pandas as pd
import xarray as xr

spec = importlib.util.spec_from_file_location("fm", "02_fit_models.py")
fm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fm)


def safe(u):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", u)


def r2_terms(rS, rL, tS, tL, beta, s2):
    kS, kL = 1 / (beta * rS), 1 / (beta * rL)
    DS, DL = 1 / (1 + kS * tS), 1 / (1 + kL * tL)
    alpha = 1 / np.sqrt(DL ** 2 * s2 / (beta * rL) + DS ** 2 * s2 / (beta * rS))
    dv = rL * DL - rS * DS
    return alpha, dv


d = pd.read_csv("clean_data.csv.gz", low_memory=False)
units = pd.read_csv("results/h2main_per_study.csv").query("in_pool").unit.tolist()
h = 1e-3
rows, allz = [], []
for u in units:
    f = f"fits/r2/{safe(u)}_theta.nc"
    if not os.path.exists(f):
        continue
    g = fm.prepare(d, u)
    post = xr.open_dataset(f)
    lb = post["theta"].mean(("chain", "draw")).values[:, 0]
    subj_order = post["theta"]["subject"].values
    lb_map = dict(zip(subj_order, lb))
    beta = np.exp(g.subj_ident.map(lb_map).values)
    rS, rL, tS, tL, s2 = g.ss_value.values, g.ll_value.values, g.ss_days.values, g.ll_days.values, g.s2_u.values
    a_p, v_p = r2_terms(rS * np.exp(h), rL * np.exp(h), tS, tL, beta, s2)
    a_m, v_m = r2_terms(rS * np.exp(-h), rL * np.exp(-h), tS, tL, beta, s2)
    e_a = (np.log(a_p) - np.log(a_m)) / (2 * h)
    e_v = (np.log(np.abs(v_p)) - np.log(np.abs(v_m))) / (2 * h)
    ok = np.isfinite(e_v) & (np.abs(v_p) > 1e-9) & (np.abs(v_m) > 1e-9)
    e_z = (e_a + e_v)[ok]
    allz.append(e_z)
    rows.append(dict(unit=u, trials=int(ok.sum()), e_alpha_median=np.median(e_a[ok]), e_dv_median=np.median(e_v[ok]),
                     e_z_median=np.median(e_z), e_z_q10=np.quantile(e_z, 0.1), e_z_q90=np.quantile(e_z, 0.9)))
    print(u, rows[-1]["trials"], round(rows[-1]["e_z_median"], 3), flush=True)

R = pd.DataFrame(rows)

# ---- fitted H2 and R2F: the same elasticity of z = alpha * dV, using participant posterior means ----
h2m = pd.read_csv("results/h2main_per_study.csv").set_index("unit")
r2fm = pd.read_csv("results/r2fmain_per_study.csv").set_index("unit")


def theta_means(model, u):
    ds = xr.open_dataset(f"fits/{model}/{safe(u)}_theta.nc")
    v = "theta_full" if "theta_full" in ds else "theta"
    th = ds[v].mean(("chain", "draw")).values
    subj = ds[v]["subject"].values
    return dict(zip(subj, th))


def z_h2(g, th, lam):
    rS, rL = g.ss_value.values * lam, g.ll_value.values * lam
    cS = np.log(rS) - g.lr_ss.mean(); cL = np.log(rL) - g.lr_ss.mean(); cM = np.log(rS + rL) - g.lsum.mean()
    kS, kL = np.exp(th[:, 0] - th[:, 1] * cS), np.exp(th[:, 0] - th[:, 1] * cL)
    alpha = np.exp(th[:, 2] + th[:, 3] * cM)
    return alpha * (rL / (1 + kL * g.ll_days.values) - rS / (1 + kS * g.ss_days.values))


def z_r2f(g, th, lam):
    rS, rL = g.ss_value.values * lam, g.ll_value.values * lam
    a, dv = r2_terms(rS, rL, g.ss_days.values, g.ll_days.values, np.exp(th[:, 0]), g.s2_u.values)
    cM = np.log(rS + rL) - g.lsum.mean()
    return a * np.exp(th[:, 1] + th[:, 2] * cM) * dv


ez = {"H2": [], "R2F": []}
for u in R.unit:
    g = fm.prepare(d, u)
    for lab, model, fn in [("H2", h2m.loc[u, "model_used"], z_h2), ("R2F", r2fm.loc[u, "model_used"], z_r2f)]:
        f = f"fits/{model}/{safe(u)}_theta.nc"
        if not os.path.exists(f):
            continue
        m = theta_means(model, u)
        th = np.vstack(g.subj_ident.map(m).values)
        zp, zm = fn(g, th, np.exp(h)), fn(g, th, np.exp(-h))
        ok = (np.abs(zp) > 1e-9) & (np.abs(zm) > 1e-9)
        e = (np.log(np.abs(zp[ok])) - np.log(np.abs(zm[ok]))) / (2 * h)
        ez[lab].append(e)
        R.loc[R.unit == u, f"e_z_{lab}_median"] = np.median(e)
for lab in ez:
    z2 = np.concatenate(ez[lab])
    print(f"{lab}: effective-precision elasticity median {np.median(z2):.3f}, 10-90% [{np.quantile(z2, .1):.3f}, "
          f"{np.quantile(z2, .9):.3f}], per-study medians {R[f'e_z_{lab}_median'].min():.3f} to "
          f"{R[f'e_z_{lab}_median'].max():.3f}, median of study medians {R[f'e_z_{lab}_median'].median():.3f}")
print("R2: median of study medians %.3f" % R.e_z_median.median())
R.to_csv("results/r2_elasticities.csv", index=False)
z = np.concatenate(allz)
print("studies:", len(R), "trials:", len(z))
print("effective-precision elasticity: median %.3f, 10-90%% [%.3f, %.3f], mean %.3f" % (
    np.median(z), np.quantile(z, 0.1), np.quantile(z, 0.9), z.mean()))
print("per-study medians range: %.3f to %.3f" % (R.e_z_median.min(), R.e_z_median.max()))
print("alpha elasticity: per-study medians %.3f to %.3f" % (R.e_alpha_median.min(), R.e_alpha_median.max()))
