"""Recovery check for the participant-level correlation between the two magnitude slopes (m_k, m_alpha).

Question: if the two magnitude effects were strongly linked across participants (as rational inattention implies,
Proposition 3), would the hierarchical H2 fit detect it, given the real designs and sample sizes?

--simulate : for four well-converged studies with participant-specific slopes, draw participant parameters from the
             study's fitted H2 population (posterior-mean mu, sd, correlation matrix, lapse rate), but with
             corr(m_k, m_alpha) set to RHO; simulate choices on the real trials -> sim/sim_corr.csv.gz and
             run_corr_recovery.bat (fits H2 with DATA_FILE=sim/sim_corr.csv.gz, out-suffix _corrrec)
(no flag)  : summarise the recovered corr[m_k, m_a] -> results/corr_recovery.csv
"""
import importlib.util
import os
import re
import sys

import numpy as np
import pandas as pd
import scipy.special as sps
import xarray as xr

spec = importlib.util.spec_from_file_location("fm", "02_fit_models.py")
fm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fm)

STUDIES = ["Cao_2021_Exp2|s2", "Kräplin_2020b|s1", "Zgonnikov_2019|s1", "Stevens_2016|s1"]
RHO = 0.9
NAMES = ["c_k", "m_k", "c_a", "m_a"]


def safe(u):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", u)


def nearest_pd(c):
    w, v = np.linalg.eigh(c)
    c2 = v @ np.diag(np.clip(w, 1e-3, None)) @ v.T
    d = np.sqrt(np.diag(c2))
    return c2 / np.outer(d, d)


def simulate():
    d = pd.read_csv("clean_data.csv.gz", low_memory=False)
    rng = np.random.default_rng(2027)
    out, used = [], []
    for u in STUDIES:
        g = fm.prepare(d, u)
        ds = xr.open_dataset(f"fits/h2/{safe(u)}_theta.nc")
        mu = ds["mu"].mean(("chain", "draw")).values
        sd = ds["sd"].mean(("chain", "draw")).values
        corr = ds["corr"].mean(("chain", "draw")).values
        eps = float(ds["eps"].mean())
        corr[1, 3] = corr[3, 1] = RHO
        corr = nearest_pd(corr)
        cov = np.outer(sd, sd) * corr
        n = g.subj.max() + 1
        th = rng.multivariate_normal(mu, cov, size=n)
        p = fm.prob(np, lambda x: 0.5 * sps.erfc(-x / np.sqrt(2)), th[g.subj.values], eps, g, "h2")
        sim = d[d.unit == u].copy()
        sim = sim.set_index(sim.index).loc[g.index]
        sim["choice"] = (rng.random(len(g)) < p).astype(int)
        sim["unit"] = "SIMCORR_" + u
        out.append(sim)
        used.append(dict(study=u, true_corr=corr[1, 3]))
        print(u, "simulated", len(sim), "trials; corr(m_k, m_a) used =", round(corr[1, 3], 3),
              "; sample corr of drawn slopes =", round(np.corrcoef(th[:, 1], th[:, 3])[0, 1], 3), flush=True)
    pd.concat(out).to_csv("sim/sim_corr.csv.gz", index=False)
    pd.DataFrame(used).to_csv("sim/sim_corr_true.csv", index=False)
    with open("run_corr_recovery.bat", "w", newline="\r\n") as f:
        f.write("@echo off\ncd /d \"%~dp0\"\nset PYTHONIOENCODING=utf-8\nset DATA_FILE=sim/sim_corr.csv.gz\n")
        for u in STUDIES:
            f.write(f".venv\\Scripts\\python.exe -u 02_fit_models.py --model h2 --out-suffix _corrrec --skip-done "
                    f"\"SIMCORR_{u}\" >> logs\\corr_recovery.log 2>&1\n")
        f.write("echo CORR_DONE >> logs\\corr_recovery.log\n")
    print("wrote sim/sim_corr.csv.gz and run_corr_recovery.bat")


def summarise():
    rows = []
    true = pd.read_csv("sim/sim_corr_true.csv").set_index("study").true_corr
    for u in STUDIES:
        f = f"fits/h2_corrrec/{safe('SIMCORR_' + u)}_summary.csv"
        real = f"fits/h2/{safe(u)}_summary.csv"
        if not os.path.exists(f):
            continue
        t, r = pd.read_csv(f, index_col=0), pd.read_csv(real, index_col=0)
        key = "corr[m_k, m_a]"
        rows.append(dict(study=u, n_subj=int(t.n_subj.iat[0]), true_corr=true[u], recovered=t.loc[key, "mean"],
                         lo=t.loc[key, "hdi_3%"], hi=t.loc[key, "hdi_97%"], rhat=t.max_rhat.iat[0],
                         real_data=r.loc[key, "mean"], real_lo=r.loc[key, "hdi_3%"], real_hi=r.loc[key, "hdi_97%"]))
    res = pd.DataFrame(rows)
    res.to_csv("results/corr_recovery.csv", index=False)
    print(res.round(3).to_string(index=False))


if __name__ == "__main__":
    simulate() if "--simulate" in sys.argv else summarise()
