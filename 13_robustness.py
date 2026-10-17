"""Robustness of the main conclusions to the choice rule, for the six model-recovery studies.

Variants (run_robust_logit.bat, run_robust_subjlapse.bat):
  main       probit choice rule, one lapse rate per study (fits/<model>/)
  logit      logistic choice rule (fits/<model>_logit/)
  subjlapse  probit, one lapse rate per participant (fits/<model>_subjlapse/)
  noiset     R2/R2F choice noise grows with delay, NOISE_T=1 (fits/<model>_noiset/; H2 from the main fit)
For each study and variant: elpd_loo differences R2F - R2 and H2 - R2 (SE from pointwise differences),
and the magnitude slopes m_k, m_alpha (H2) and g1 (R2F). Output: results/robustness.csv
"""
import os
import re

import numpy as np
import pandas as pd

STUDIES = ["Keidel_2024|s1", "Berenson_2016|s1", "Castrellon_2019|s1", "Bixter_2019|s1", "Yang_2022|s1",
           "Scherbaum_2018a|s1"]
SHORT = {"Keidel_2024|s1", "Berenson_2016|s1", "Bixter_2019|s1"}
VARIANTS = {"main": "", "logit": "_logit", "subjlapse": "_subjlapse", "noiset": "_noiset"}


def safe(u):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", u)


def get(t, p):
    return (t.loc[p, "mean"], t.loc[p, "hdi_3%"], t.loc[p, "hdi_97%"]) if p in t.index else (np.nan,) * 3


rows = []
for u in STUDIES:
    fmod, hmod = ("r2fp", "h2p") if u in SHORT else ("r2f", "h2")
    for var, suf in VARIANTS.items():
        pw, summ = {}, {}
        for lab, m in [("R2", "r2"), ("R2F", fmod), ("H2", hmod)]:
            base = f"fits/{m}{suf}/{safe(u)}"
            if var == "noiset" and lab == "H2":   # H2 has no delay-noise variant: use the main fit
                base = f"fits/{m}/{safe(u)}"
            if os.path.exists(base + "_pointwise.csv.gz"):
                pw[lab] = pd.read_csv(base + "_pointwise.csv.gz").elpd_loo_i.values
                summ[lab] = pd.read_csv(base + "_summary.csv", index_col=0)
        if len(pw) < 3:
            continue
        r = dict(study=u, variant=var, n_trials=len(pw["R2"]))
        for a, b in [("R2F", "R2"), ("H2", "R2"), ("H2", "R2F")]:
            d = pw[a] - pw[b]
            r[f"{a}_minus_{b}"], r[f"{a}_minus_{b}_se"] = d.sum(), np.sqrt(len(d)) * d.std()
        for name, lab, p in [("m_k", "H2", "mu[m_k]"), ("m_a", "H2", "mu[m_a]"), ("g1", "R2F", "mu[g1]")]:
            r[name], r[name + "_lo"], r[name + "_hi"] = get(summ[lab], p)
        r["max_rhat"] = max(float(s.max_rhat.iat[0]) for s in summ.values())
        rows.append(r)

res = pd.DataFrame(rows)
res.to_csv("results/robustness.csv", index=False)
pd.set_option("display.width", 220)
print(res.drop(columns=[c for c in res.columns if c.endswith(("_lo", "_hi"))]).round(2).to_string(index=False))
