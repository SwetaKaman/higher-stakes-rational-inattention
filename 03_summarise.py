"""Step 3: collect per-study fits, pool them, and compare models by LOO.

Outputs:
  results/h2main_per_study.csv main H2 result per study (model_used = h2 or h2p by the 60-choice rule)
  results/h2main_pooled.csv    random-effects (DerSimonian-Laird) pooled estimates across studies
  results/h2full_*.csv         sensitivity: full per-subject-slope H2 wherever fitted
  results/model_comparison.csv per-study elpd_loo differences between models (pointwise SE)
"""
import glob
import os
import re

import numpy as np
import pandas as pd

os.makedirs("results", exist_ok=True)
DIAG = ["n_subj", "n_trials", "chains_dropped", "divergences", "max_rhat", "min_ess_bulk",
        "elpd_loo", "se_loo", "n_pareto_k_gt_0_7", "seconds"]


EXCLUDE_FROM_POOL = {"Weidacker_2020|s1"}   # 14 subjects; listed but not pooled
EXCLUDED = {"Thrailkill_2022b|s1"}          # removed from the analysis entirely (see EXCLUSIONS.md)
# Rule (2026-10-13): studies with median < 60 choices per subject use H2-pooled-slopes (h2p) as the
# main H2 result; their full-model h2 fits are kept as sensitivity results.
_med = pd.read_csv("results/median_trials_per_subject.csv", index_col=0).iloc[:, 0]
SHORT = set(_med[_med < 60].index) - {"Thrailkill_2022b|s1"}


def load(model):
    """Per-study rows; a refit in fits/<model>_refit replaces the original fit."""
    files = {os.path.basename(f): f for f in sorted(glob.glob(f"fits/{model}/*_summary.csv"))}
    files.update({os.path.basename(f): f for f in sorted(glob.glob(f"fits/{model}_refit/*_summary.csv"))})
    rows = []
    for f in sorted(files.values()):
        s = pd.read_csv(f, index_col=0)
        if s.unit.iat[0] in EXCLUDED:
            continue
        r = {"unit": s.unit.iat[0]}
        for p in s.index:
            if p.startswith(("mu[", "sd[", "corr[", "eps")):
                for c, n in [("mean", ""), ("sd", "_sd"), ("hdi_3%", "_lo"), ("hdi_97%", "_hi")]:
                    r[p + n] = s.loc[p, c]
        r.update({k: s[k].iat[0] for k in DIAG})
        r["refit"] = "_refit" in f
        rows.append(r)
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["unit"])


def dersimonian_laird(y, se):
    w = 1 / se ** 2
    fe = np.sum(w * y) / np.sum(w)
    q = np.sum(w * (y - fe) ** 2)
    tau2 = max(0.0, (q - (len(y) - 1)) / (np.sum(w) - np.sum(w ** 2) / np.sum(w)))
    w2 = 1 / (se ** 2 + tau2)
    est = np.sum(w2 * y) / np.sum(w2)
    s = np.sqrt(1 / np.sum(w2))
    i2 = max(0.0, (q - (len(y) - 1)) / q) if q > 0 else 0.0
    return est, est - 1.96 * s, est + 1.96 * s, np.sqrt(tau2), i2


def pool(model, params, t=None):
    t = load(model) if t is None else t
    if not len(t):
        return
    t["flag"] = (t.chains_dropped > 1) | (t.max_rhat > 1.01) | t.max_rhat.isna()
    t["in_pool"] = ~t.unit.isin(EXCLUDE_FROM_POOL)
    t.to_csv(f"results/{model}_per_study.csv", index=False)
    h2 = t[t.in_pool]
    rows = []
    for p in params:
        for subset, d in [("all", h2), ("unflagged", h2[~h2.flag])]:
            if p.startswith("corr") and "model_used" in d:   # per-subject corr only exists in full-model fits
                d = d[d.model_used == "h2"]
            if len(d) < 2:
                continue
            y, se = d[p].values, d[p + "_sd"].values
            if p.startswith("corr"):                       # pool correlations on Fisher-z scale
                z = np.arctanh(np.clip(y, -0.999, 0.999))
                zse = se / (1 - np.clip(y, -0.999, 0.999) ** 2)
                est, lo, hi, tau, i2 = dersimonian_laird(z, zse)
                est, lo, hi = np.tanh([est, lo, hi])
            else:
                est, lo, hi, tau, i2 = dersimonian_laird(y, se)
            rows.append(dict(param=p, studies=subset, k=len(d), pooled=est, ci_lo=lo, ci_hi=hi,
                             tau=tau, I2=i2, n_hdi_excl_0=int(((d[p + "_lo"] > 0) | (d[p + "_hi"] < 0)).sum())))
    if model.startswith("h2"):   # across-study correlation of study-level mean slopes
        r = np.corrcoef(h2["mu[m_k]"], h2["mu[m_a]"])[0, 1]
        rows.append(dict(param="across-study corr(mu[m_k], mu[m_a])", studies="all", k=len(h2), pooled=r))
    pooled = pd.DataFrame(rows)
    pooled.to_csv(f"results/{model}_pooled.csv", index=False)
    print(model, f"({t.flag.sum()} of {len(t)} studies flagged)")
    print(pooled.round(3).to_string(index=False))


# sensitivity: full per-subject-slope model wherever it was fitted
pool("h2", ["mu[m_k]", "mu[m_a]", "corr[m_k, m_a]"])
os.replace("results/h2_per_study.csv", "results/h2full_per_study.csv")
os.replace("results/h2_pooled.csv", "results/h2full_pooled.csv")
# main H2: h2p for short-questionnaire studies, h2 otherwise
full, short = load("h2"), load("h2p")
main = pd.concat([full[~full.unit.isin(SHORT)].assign(model_used="h2"),
                  short[short.unit.isin(SHORT)].assign(model_used="h2p")], ignore_index=True)
pending = sorted(SHORT - set(short.unit))
print("short-questionnaire studies still awaiting h2p fit:", len(pending), pending)
pool("h2main", ["mu[m_k]", "mu[m_a]", "corr[m_k, m_a]"], main)
pool("r2", ["mu[log_beta]"])
# main R2F: r2fp (study-level g1) for short-questionnaire studies, r2f otherwise
pool("r2f", ["mu[g1]", "mu[g0]", "mu[log_beta]"])
f_full, f_short = load("r2f"), load("r2fp")
r2f_main = pd.concat([f_full[~f_full.unit.isin(SHORT)].assign(model_used="r2f"),
                      f_short[f_short.unit.isin(SHORT)].assign(model_used="r2fp")], ignore_index=True)
pool("r2fmain", ["mu[g1]", "mu[g0]", "mu[log_beta]"], r2f_main)

# model comparison from pointwise LOO (same trials, same order in every model)
models = [m for m in ["h2", "h2p", "r2", "r2f", "r2fp"] if os.path.isdir(f"fits/{m}")]
units = sorted({os.path.basename(f)[:-len("_pointwise.csv.gz")]
                for m in models for f in glob.glob(f"fits/{m}/*_pointwise.csv.gz")}
               - {re.sub(r"[^A-Za-z0-9_.-]+", "_", u) for u in EXCLUDE_FROM_POOL})
rows = []
for u in units:
    pw = {m: pd.read_csv(f"fits/{m}/{u}_pointwise.csv.gz").elpd_loo_i.values
          for m in models if os.path.exists(f"fits/{m}/{u}_pointwise.csv.gz")}
    if u in {re.sub(r"[^A-Za-z0-9_.-]+", "_", x) for x in EXCLUDED}:
        continue
    # main versions by the 60-choice rule: pooled-slope variants replace the full models for short studies
    is_short = u in {re.sub(r"[^A-Za-z0-9_.-]+", "_", x) for x in SHORT}
    for full_m, pooled_m in [("h2", "h2p"), ("r2f", "r2fp")]:
        if is_short:
            # pooled-slope model replaces the full one; if it could not be fitted, the full model is kept
            if pooled_m in pw:
                pw[full_m] = pw.pop(pooled_m)
        else:
            pw.pop(pooled_m, None)
    r = {"unit": u}
    for m, v in pw.items():
        r[f"elpd_{m}"] = v.sum()
    for a, b in [("r2f", "r2"), ("h2", "r2"), ("h2", "r2f")]:
        if a in pw and b in pw:
            diff = pw[a] - pw[b]
            r[f"{a}_minus_{b}"] = diff.sum()
            r[f"{a}_minus_{b}_se"] = np.sqrt(len(diff)) * diff.std()
    rows.append(r)
if rows:
    mc = pd.DataFrame(rows)
    mc.to_csv("results/model_comparison.csv", index=False)
    for c in [c for c in mc.columns if c.endswith(("_r2", "_r2f")) and "minus" in c]:
        d, se = mc[c].dropna(), mc[c + "_se"].dropna()
        print(f"{c}: studies={len(d)} total={d.sum():.1f} (SE {np.sqrt((se ** 2).sum()):.1f}); "
              f"favours first model (diff > 2 SE) in {(d > 2 * se).sum()}, second in {(d < -2 * se).sum()}")
