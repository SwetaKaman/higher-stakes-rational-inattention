"""Step 2: hierarchical Bayesian fits per study x subset unit, following the likelihoods in
Gershman & Bhui (2020) code (github.com/sjgershm/rational-discounting: H2.m, R2.m).

Common choice rule (G&B):  P(LL) = (1 - eps) * Phi(alpha * (V_LL - V_SS)) + eps / 2
  V_j = r_j / (1 + k_j t_j), raw currency units; eps = lapse, one per study ~ Beta(1, 20).

Models
  h2   (G&B H2.m)  log k_j   = c_k - m_k * log r_j                 (k per option, own reward)
                   log alpha = c_a + m_a * log(r_LL + r_SS)
                   subject (c_k, m_k, c_a, m_a) ~ MVN(mu, Sigma), LKJ(2). log magnitudes are
                   centred within unit (does not change slopes).
  r2   (G&B R2.m)  k_j = 1 / (beta r_j);  s2e_j = s2_u / (beta r_j)
                   alpha = 1 / sqrt(sum_j D_j^2 s2e_j);  s2_u = running variance of all rewards the
                   subject has seen up to the current trial (both options, n-1, floor 1e-5).
                   subject log beta ~ Normal(mu, sd).
  r2f  R2 + free stochasticity slope: as r2 but alpha multiplied by
                   exp(g0 + g1 * log(r_LL + r_SS) [centred]); subject (log beta, g0, g1) ~ MVN.
                   r2f with g0 = g1 = 0 is exactly r2.

Chains: 4 (numpyro/JAX). Chains whose mean total log-likelihood is > 50 below the best chain are
dropped and counted. LOO/WAIC are trial-level, computed from the kept chains.

Usage: .venv/Scripts/python 02_fit_models.py --model h2|r2|r2f [--no-lapse] [--skip-done] [unit ...]
Outputs fits/<model>[_nolapse]/<unit>_summary.csv, <unit>_theta.nc (posterior without
log-likelihood), <unit>_pointwise.csv.gz (per-trial elpd_loo, for model comparison).
"""
import os
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=" + os.environ.get("NDEV", "4"))
import argparse
import re
import time

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import scipy.special as sps
import xarray as xr

PARAMS = {"h2": ["c_k", "m_k", "c_a", "m_a"], "h2p": ["c_k", "c_a"], "r2": ["log_beta"],
          "r2f": ["log_beta", "g0", "g1"], "r2fp": ["log_beta", "g0"]}
BASE = {"h2p": "h2", "r2fp": "r2f"}   # pooled-slope variants and the full model whose likelihood they use
# r2fp = R2F with the stochasticity slope g1 as one study-level parameter (short-questionnaire rule).
# h2p = "H2-pooled-slopes": same likelihood as h2, but m_k and m_a are single study-level parameters
# (no per-subject slopes). Used as the main model for studies with median < 60 choices per subject.
DROP_LL = 50.0
BIG_TRIALS, BIG_SUBJ = 15000, 400   # "big" studies run their chains sequentially (memory)
# Robustness variants (defaults reproduce the main analysis):
#   LINK=logit         logistic instead of probit choice rule
#   LAPSE_MODE=subject one lapse rate per participant, logit(eps_s) ~ Normal(eps_mu, eps_sd), instead of one per study
LINK = os.environ.get("LINK", "probit")
LAPSE_MODE = os.environ.get("LAPSE_MODE", "study")
PHI_PT = pt.sigmoid if LINK == "logit" else (lambda x: 0.5 * pt.erfc(-x / np.sqrt(2)))
PHI_NP = sps.expit if LINK == "logit" else (lambda x: 0.5 * sps.erfc(-x / np.sqrt(2)))


def safe(u):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", u)


def running_var(g):
    """G&B load_data.m: s2_u(n) = max(1e-5, var([X1(1:n); X2(1:n)])), per subject, file order."""
    out = np.empty(len(g))
    pos = 0
    for _, s in g.groupby("subj_ident", sort=False):
        x = np.column_stack([s.ss_value.values, s.ll_value.values]).ravel()
        n = np.arange(2, len(x) + 1, 2)
        c1, c2 = np.cumsum(x)[n - 1], np.cumsum(x ** 2)[n - 1]
        v = (c2 - c1 ** 2 / n) / (n - 1)
        out[pos:pos + len(s)] = np.maximum(1e-5, v)
        pos += len(s)
    return out


def prepare(d, unit, max_subj=None):
    g = d[d.unit == unit].copy()
    subj = g.subj_ident.unique()
    if max_subj and len(subj) > max_subj:     # fixed-seed random subsample of subjects
        keep = np.random.default_rng(2026).choice(subj, max_subj, replace=False)
        g = g[g.subj_ident.isin(keep)]
    g = g.sort_values("subj_ident", kind="stable")       # keep file order within subject
    g["s2_u"] = running_var(g)
    g["subj"] = pd.factorize(g.subj_ident)[0]
    g["lr_ss"], g["lr_ll"] = np.log(g.ss_value), np.log(g.ll_value)
    g["lsum"] = np.log(g.ss_value + g.ll_value)
    return g


def prob(xp, Phi, th, eps, g, model):
    """P(LL) for numpy (xp=np) or pytensor (xp=pt). th[..., obs, param]."""
    rS, rL, tS, tL = g.ss_value.values, g.ll_value.values, g.ss_days.values, g.ll_days.values
    cS, cL = g.lr_ss.values - g.lr_ss.mean(), g.lr_ll.values - g.lr_ss.mean()
    cM = g.lsum.values - g.lsum.mean()
    if model in ("h2", "h2p"):
        kS = xp.exp(th[..., 0] - th[..., 1] * cS)
        kL = xp.exp(th[..., 0] - th[..., 1] * cL)
        alpha = xp.exp(th[..., 2] + th[..., 3] * cM)
        dv = rL / (1 + kL * tL) - rS / (1 + kS * tS)
    else:
        beta = xp.exp(th[..., 0])
        kS, kL = 1 / (beta * rS), 1 / (beta * rL)
        DS, DL = 1 / (1 + kS * tS), 1 / (1 + kL * tL)
        s2 = g.s2_u.values
        alpha = 1 / xp.sqrt(DL ** 2 * s2 / (beta * rL) + DS ** 2 * s2 / (beta * rS))
        if model == "r2f":
            alpha = alpha * xp.exp(th[..., 1] + th[..., 2] * cM)
        dv = rL * DL - rS * DS
    return (1 - eps) * Phi(alpha * dv) + eps / 2


def build(g, model, lapse):
    names = PARAMS[model]
    P = len(names)
    coords = {"subject": np.unique(g.subj_ident), "param": names, "param_": names,
              "slope_name": {"h2p": ["m_k", "m_a"], "r2fp": ["g1"]}.get(model, []),
              "param_full": PARAMS[BASE.get(model, model)]}
    # rough location for c_a: alpha ~ 3 / typical SS amount at the unit's mean magnitude
    loc_a = np.log(3.0) - g.lr_ss.mean()
    prior_mu = {"h2": ([-4.0, 0.5, loc_a, 0.0], [3, 1, 3, 1]),
                "h2p": ([-4.0, loc_a], [3, 3]),
                "r2fp": ([-g.lr_ss.mean() + 4.0, 0.0], [4, 2]),
                "r2": ([g.lr_ss.mean() * -1 + 4.0], [4]),
                "r2f": ([-g.lr_ss.mean() + 4.0, 0.0, 0.0], [4, 2, 1])}[model]
    with pm.Model(coords=coords) as m:
        mu = pm.Normal("mu", mu=np.array(prior_mu[0]), sigma=np.array(prior_mu[1]), dims="param")
        sd_scale = {"h2": [1.5, 0.5, 1.5, 0.5], "h2p": [1.5, 1.5], "r2fp": [1.5, 1.0], "r2": [1.5], "r2f": [1.5, 1.0, 0.5]}[model]
        if P > 1:
            chol, corr, sd = pm.LKJCholeskyCov("chol", n=P, eta=2.0, compute_corr=True,
                                               sd_dist=pm.HalfNormal.dist(np.array(sd_scale), shape=P))
            pm.Deterministic("corr", corr, dims=("param", "param_"))
            pm.Deterministic("sd", sd, dims="param")
            z = pm.Normal("z", 0, 1, dims=("subject", "param"))
            theta = pm.Deterministic("theta", mu + pt.dot(z, chol.T), dims=("subject", "param"))
        else:
            sd = pm.HalfNormal("sd", np.array(sd_scale), dims="param")
            z = pm.Normal("z", 0, 1, dims=("subject", "param"))
            theta = pm.Deterministic("theta", mu + z * sd, dims=("subject", "param"))
        ones = pt.ones((len(coords["subject"]),))
        if model == "h2p":   # study-level slopes, same priors as mu[m_k], mu[m_a] in h2
            slope = pm.Normal("slope", mu=np.array([0.5, 0.0]), sigma=np.array([1.0, 1.0]), dims="slope_name")
            theta = pm.Deterministic("theta_full", pt.stack([theta[:, 0], slope[0] * ones, theta[:, 1],
                                                              slope[1] * ones], axis=1), dims=("subject", "param_full"))
        elif model == "r2fp":   # study-level g1, same prior as mu[g1] in r2f
            slope = pm.Normal("slope", mu=np.array([0.0]), sigma=np.array([1.0]), dims="slope_name")
            theta = pm.Deterministic("theta_full", pt.stack([theta[:, 0], theta[:, 1], slope[0] * ones], axis=1),
                                     dims=("subject", "param_full"))
        if lapse and LAPSE_MODE == "subject":   # invlogit(-3) = 0.047, close to the Beta(1, 20) mean
            e_mu, e_sd = pm.Normal("eps_mu", -3.0, 1.0), pm.HalfNormal("eps_sd", 1.0)
            e_z = pm.Normal("eps_z", 0, 1, dims="subject")
            eps = pm.Deterministic("eps", pm.math.invlogit(e_mu + e_sd * e_z), dims="subject")[g.subj.values]
        else:
            eps = pm.Beta("eps", 1, 20) if lapse else 0.0
        p = prob(pt, PHI_PT, theta[g.subj.values], eps, g, BASE.get(model, model))
        p = pt.clip(p, 1e-12, 1 - 1e-12)
        pm.Bernoulli("y", p=p, observed=g.choice.values)
    return m


def pointwise_loglik(idata, g, model, lapse, draw_batch=250):
    """Per-trial log-likelihood (chain, draw, obs) as float32, computed in batches of draws."""
    th = idata.posterior["theta_full" if model in BASE else "theta"].values   # chain, draw, subj, param
    per_subj = lapse and LAPSE_MODE == "subject"
    eps = (idata.posterior["eps"].values if per_subj else idata.posterior["eps"].values[..., None]) if lapse else 0.0
    y = g.choice.values
    ll = np.empty(th.shape[:2] + (len(g),), dtype=np.float32)
    for c in range(th.shape[0]):
        for d0 in range(0, th.shape[1], draw_batch):
            d1 = min(d0 + draw_batch, th.shape[1])
            e = (eps[c, d0:d1][:, g.subj.values] if per_subj else eps[c, d0:d1]) if lapse else 0.0
            p = prob(np, PHI_NP, th[c, d0:d1][:, g.subj.values], e, g, BASE.get(model, model))
            p = np.clip(p, 1e-12, 1 - 1e-12)
            ll[c, d0:d1] = np.where(y == 1, np.log(p), np.log1p(-p))
    return ll


def posterior_reff(posterior):
    """Relative efficiency exactly as az.loo computes it: mean 'mean'-ESS of all posterior values / n draws."""
    if posterior.sizes["chain"] == 1:
        return 1.0
    ess_p = az.ess(posterior, method="mean")
    n = posterior.sizes["chain"] * posterior.sizes["draw"]
    return float(np.hstack([ess_p[v].values.flatten() for v in ess_p.data_vars]).mean() / n)


def batched_loo_waic(ll, reff, obs_batch=2000):
    """PSIS-LOO and WAIC computed over batches of observations; exact, because both are sums over
    trials and use one relative efficiency (reff) for all trials, as az.loo does."""
    C, D, N = ll.shape
    loo_i, k_i, lppd_i, pw_i = (np.empty(N) for _ in range(4))
    for o0 in range(0, N, obs_batch):
        o1 = min(o0 + obs_batch, N)
        blk = ll[:, :, o0:o1].astype(np.float64)
        ds = az.from_dict(log_likelihood={"y": blk})
        lo = az.loo(ds, pointwise=True, reff=reff)
        loo_i[o0:o1], k_i[o0:o1] = lo.loo_i.values, lo.pareto_k.values
        flat = blk.reshape(C * D, -1)
        lppd_i[o0:o1] = np.logaddexp.reduce(flat, axis=0) - np.log(C * D)
        pw_i[o0:o1] = flat.var(axis=0)   # ddof=0, as az.waic
    waic_i = lppd_i - pw_i
    return dict(elpd_loo=loo_i.sum(), se_loo=np.sqrt(N * loo_i.var()), p_loo=lppd_i.sum() - loo_i.sum(),
                loo_i=loo_i, pareto_k=k_i, elpd_waic=waic_i.sum(), se_waic=np.sqrt(N * waic_i.var()))


def fit_unit(d, unit, model, lapse, draws, tune, chains, outdir, target_accept=0.95, max_subj=None):
    g = prepare(d, unit, max_subj)
    m = build(g, model, lapse)
    t0 = time.time()
    method = os.environ.get("CHAIN_METHOD", "auto")
    if method == "auto":   # big studies: one chain at a time to bound peak memory
        method = "sequential" if (len(g) >= BIG_TRIALS or g.subj_ident.nunique() >= BIG_SUBJ) else "parallel"
    keep_vars = (["mu", "sd", "eps", "eps_mu", "eps_sd"] + (["corr"] if len(PARAMS[model]) > 1 else [])
                 + (["slope", "theta_full"] if model in BASE else ["theta"]))
    keep_vars = [v for v in keep_vars if v in m.named_vars]
    idata = pm.sample(model=m, draws=draws, tune=tune, chains=chains, nuts_sampler="numpyro",
                      target_accept=target_accept, random_seed=2026,
                      progressbar=os.environ.get("PROGRESS", "0") == "1", var_names=keep_vars,
                      nuts_sampler_kwargs=dict(chain_method=method, postprocessing_vectorize="scan"))
    secs = time.time() - t0
    ll = pointwise_loglik(idata, g, model, lapse)
    tot = ll.sum(-1).mean(-1)
    keep = np.where(tot > tot.max() - DROP_LL)[0]
    post = idata.posterior.isel(chain=keep)
    ss = idata.sample_stats.isel(chain=keep)
    kept = az.InferenceData(posterior=post, sample_stats=ss)
    var_names = (["mu", "sd"] + (["slope"] if model in BASE else []) + (["corr"] if len(PARAMS[model]) > 1 else [])
                 + ((["eps_mu", "eps_sd"] if LAPSE_MODE == "subject" else ["eps"]) if lapse else []))
    s = az.summary(kept, var_names=var_names, hdi_prob=0.94)
    s.index = [i.replace("slope[", "mu[") for i in s.index]   # study-level slopes reported as mu[m_k], mu[g1], ...
    s = s[~s.index.str.contains(r"corr\[(\w+), \1\]", regex=True)]
    lw = batched_loo_waic(ll[keep], posterior_reff(post))
    del ll
    meta = dict(unit=unit, link=LINK, lapse_mode=LAPSE_MODE, subsampled=bool(max_subj and d[d.unit == unit].subj_ident.nunique() > max_subj), model=model, lapse=lapse, seconds=round(secs), n_subj=g.subj_ident.nunique(),
                n_trials=len(g), chains_dropped=chains - len(keep),
                chain_loglik=";".join(f"{v:.1f}" for v in tot),
                divergences=int(ss.diverging.sum()), max_rhat=float(s.r_hat.max()),
                min_ess_bulk=float(s.ess_bulk.min()), elpd_loo=lw["elpd_loo"], se_loo=lw["se_loo"],
                p_loo=lw["p_loo"], n_pareto_k_gt_0_7=int((lw["pareto_k"] > 0.7).sum()),
                elpd_waic=lw["elpd_waic"], se_waic=lw["se_waic"], chain_method=method)
    for k, v in meta.items():
        s[k] = v
    base = os.path.join(outdir, safe(unit))
    s.to_csv(base + "_summary.csv")
    pd.DataFrame({"subj_ident": g.subj_ident.values, "elpd_loo_i": lw["loo_i"],
                  "pareto_k": lw["pareto_k"]}).to_csv(base + "_pointwise.csv.gz", index=False)
    post.to_netcdf(base + "_theta.nc")
    pd.Series(np.unique(g.subj_ident), name="subj_ident").to_csv(base + "_subjects.csv", index=False)
    flag = " FLAG" if (meta["chains_dropped"] > 1 or meta["max_rhat"] > 1.01) else ""
    print(f"{unit}: {model}{'' if lapse else '-nolapse'} {meta['n_subj']} subj {len(g)} trials "
          f"{secs:.0f}s dropped={meta['chains_dropped']} div={meta['divergences']} "
          f"rhat={meta['max_rhat']:.3f} elpd_loo={lw['elpd_loo']:.1f} chains={method}{flag}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("units", nargs="*")
    ap.add_argument("--model", default="h2", choices=list(PARAMS))
    ap.add_argument("--no-lapse", action="store_true")
    ap.add_argument("--draws", type=int, default=1500)
    ap.add_argument("--tune", type=int, default=1500)
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--skip-done", action="store_true")
    ap.add_argument("--target-accept", type=float, default=0.95)
    ap.add_argument("--max-subj", type=int, default=None)
    ap.add_argument("--out-suffix", default="", help="e.g. _refit -> fits/h2_refit")
    a = ap.parse_args()
    outdir = os.path.join("fits", a.model + ("_nolapse" if a.no_lapse else "") + a.out_suffix)
    os.makedirs(outdir, exist_ok=True)
    d = pd.read_csv(os.environ.get("DATA_FILE", "clean_data.csv.gz"), low_memory=False)
    units = a.units or (pd.read_csv("unit_summary.csv").query("magnitude_varying")
                        .sort_values("trials").unit.tolist())
    for u in units:
        if a.skip_done and os.path.exists(os.path.join(outdir, safe(u) + "_summary.csv")):
            continue
        try:
            fit_unit(d, u, a.model, not a.no_lapse, a.draws, a.tune, a.chains, outdir, a.target_accept, a.max_subj)
        except Exception as e:
            print(f"{u}: {a.model} FAILED {e!r}", flush=True)
