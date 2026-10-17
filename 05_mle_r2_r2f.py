"""R2 vs R2F by per-subject maximum likelihood (time-limited substitute for hierarchical fits).

Same likelihoods as 02_fit_models.py (G&B R2.m), with a per-subject lapse as in G&B fit_models.m:
  R2 : params log_beta, lapse                        (k = 2)
  R2F: params log_beta, g0, g1, lapse                (k = 4)
       alpha_R2F = alpha_R2 * exp(g0 + g1 * (log(r_LL + r_SS) - study mean))
Multiple starts (L-BFGS-B); R2F is also started from the R2 optimum with g0 = g1 = 0, so its
maximum log-likelihood is never below R2's. Runs on all subjects of the 41 magnitude-varying units.

Outputs: results/mle_r2_r2f_subjects.csv, results/mle_r2_r2f_per_study.csv, results/mle_r2_r2f_pooled.csv
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import ndtr, expit

LAPSE_MAX = 0.99


def running_var(ss, ll):
    x = np.column_stack([ss, ll]).ravel()
    n = np.arange(2, len(x) + 1, 2)
    c1, c2 = np.cumsum(x)[n - 1], np.cumsum(x ** 2)[n - 1]
    return np.maximum(1e-5, (c2 - c1 ** 2 / n) / (n - 1))


def nll(p, s, free):
    beta = np.exp(p[0])
    lapse = LAPSE_MAX * expit(p[-1])
    DS = 1 / (1 + s["tS"] / (beta * s["rS"]))
    DL = 1 / (1 + s["tL"] / (beta * s["rL"]))
    alpha = 1 / np.sqrt(s["s2"] * (DL ** 2 / (beta * s["rL"]) + DS ** 2 / (beta * s["rS"])))
    if free:
        alpha = alpha * np.exp(p[1] + p[2] * s["cM"])
    P = (1 - lapse) * ndtr(alpha * (s["rL"] * DL - s["rS"] * DS)) + lapse / 2
    P = np.clip(P, 1e-12, 1 - 1e-12)
    return -np.sum(np.where(s["y"] == 1, np.log(P), np.log1p(-P)))


def best(fun, starts, bounds):
    out = None
    for x0 in starts:
        r = minimize(fun, x0, method="L-BFGS-B", bounds=bounds)
        if out is None or r.fun < out.fun:
            out = r
    return out


d = pd.read_csv("clean_data.csv.gz", low_memory=False)
units = pd.read_csv("unit_summary.csv").query("magnitude_varying").unit.tolist()
import os, sys
if len(sys.argv) > 1:
    units = sys.argv[1:]
rows = []
for u in units:
    g = d[d.unit == u]
    lsum_mean = np.log(g.ss_value + g.ll_value).mean()
    lb0 = 4.0 - np.log(g.ss_value).mean()                  # k ~ 0.02/day at typical SS amount
    for sid, s in g.groupby("subj_ident", sort=False):
        S = dict(rS=s.ss_value.values, rL=s.ll_value.values, tS=s.ss_days.values, tL=s.ll_days.values,
                 y=s.choice.values, s2=running_var(s.ss_value.values, s.ll_value.values),
                 cM=np.log(s.ss_value.values + s.ll_value.values) - lsum_mean)
        b2 = [(-20, 20), (-10, 10)]
        r2 = best(lambda p: nll(p, S, False),
                  [[lb0 + o, l] for o in (-4, 0, 4) for l in (-4, -1)], b2)
        b4 = [(-20, 20), (-10, 10), (-5, 5), (-10, 10)]
        r2f = best(lambda p: nll(p, S, True),
                   [[r2.x[0], 0, 0, r2.x[1]]] + [[lb0 + o, g0, g1, -3] for o in (-3, 3) for g0 in (0,)
                                                 for g1 in (-1, 1)], b4)
        n = len(s)
        rows.append(dict(unit=u, subj_ident=sid, n_trials=n,
                         nll_r2=r2.fun, nll_r2f=min(r2f.fun, r2.fun),
                         log_beta_r2=r2.x[0], lapse_r2=LAPSE_MAX * expit(r2.x[1]),
                         log_beta_r2f=r2f.x[0], g0=r2f.x[1], g1=r2f.x[2], lapse_r2f=LAPSE_MAX * expit(r2f.x[3]),
                         aic_r2=2 * r2.fun + 4, aic_r2f=2 * r2f.fun + 8,
                         bic_r2=2 * r2.fun + 2 * np.log(n), bic_r2f=2 * r2f.fun + 4 * np.log(n)))
    print(u, g.subj_ident.nunique(), "subjects done", flush=True)

sub = pd.DataFrame(rows)
sub.to_csv("results/mle_r2_r2f_subjects.csv", index=False)

# per study: summed AIC/BIC differences (positive = R2F better), subjects favouring each, g1 summary
sub["dAIC"] = sub.aic_r2 - sub.aic_r2f
sub["dBIC"] = sub.bic_r2 - sub.bic_r2f
per = sub.groupby("unit").agg(n_subj=("subj_ident", "size"), n_trials=("n_trials", "sum"),
                              sum_dAIC=("dAIC", "sum"), sum_dBIC=("dBIC", "sum"),
                              subj_favour_r2f_bic=("dBIC", lambda x: int((x > 0).sum())),
                              g1_median=("g1", "median"), g1_mean=("g1", "mean"), g1_sd=("g1", "std")).reset_index()
per["g1_se"] = per.g1_sd / np.sqrt(per.n_subj)
per["study_favours_bic"] = np.where(per.sum_dBIC > 0, "R2F", "R2")
per["study_favours_aic"] = np.where(per.sum_dAIC > 0, "R2F", "R2")
per.to_csv("results/mle_r2_r2f_per_study.csv", index=False)


def dl(y, se):
    w = 1 / se ** 2
    fe = np.sum(w * y) / np.sum(w)
    q = np.sum(w * (y - fe) ** 2)
    tau2 = max(0.0, (q - (len(y) - 1)) / (np.sum(w) - np.sum(w ** 2) / np.sum(w)))
    w2 = 1 / (se ** 2 + tau2)
    est, s = np.sum(w2 * y) / np.sum(w2), np.sqrt(1 / np.sum(w2))
    return est, est - 1.96 * s, est + 1.96 * s, np.sqrt(tau2)


pooled = []
for name, p in [("all", per), ("excl. Weidacker_2020", per[per.unit != "Weidacker_2020|s1"])]:
    ok = p[p.g1_se > 0]
    est, lo, hi, tau = dl(ok.g1_mean.values, ok.g1_se.values)
    pooled.append(dict(studies=name, k=len(p), total_dAIC=p.sum_dAIC.sum(), total_dBIC=p.sum_dBIC.sum(),
                       studies_favour_r2f_bic=int((p.sum_dBIC > 0).sum()),
                       studies_favour_r2_bic=int((p.sum_dBIC <= 0).sum()),
                       studies_favour_r2f_aic=int((p.sum_dAIC > 0).sum()),
                       pooled_g1=est, g1_ci_lo=lo, g1_ci_hi=hi, g1_tau=tau,
                       median_subject_g1=sub[sub.unit.isin(p.unit)].g1.median()))
pooled = pd.DataFrame(pooled)
pooled.to_csv("results/mle_r2_r2f_pooled.csv", index=False)
print(pooled.round(3).to_string(index=False))
