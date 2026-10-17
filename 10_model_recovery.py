"""Model recovery: if R2 were the true model, would our pipeline favour it?

For six study designs (real trials, real participants' running reward variances), choices are simulated
from R2 using each participant's posterior-mean log beta and the study's posterior-mean lapse rate from the
R2 fit to the real data. The simulated data are then fitted with R2, R2F and H2 (pooled-slope variants for
short questionnaires) by the same pipeline, and compared by PSIS-LOO.

Step 1 (this script, --simulate): writes sim/sim_data.csv.gz with units "SIMR2_<study>".
Step 2: run_model_recovery.bat fits r2 / r2f(p) / h2(p) to those units with DATA_FILE=sim/sim_data.csv.gz.
Step 3 (--summarise): results/model_recovery.csv and printed comparison.
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

STUDIES = ["Keidel_2024|s1", "Berenson_2016|s1", "Castrellon_2019|s1", "Bixter_2019|s1", "Yang_2022|s1",
           "Scherbaum_2018a|s1"]
SHORT = {"Keidel_2024|s1", "Berenson_2016|s1", "Bixter_2019|s1"}


def safe(u):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", u)


def simulate():
    d = pd.read_csv("clean_data.csv.gz", low_memory=False)
    rng = np.random.default_rng(2026)
    out = []
    for u in STUDIES:
        g = fm.prepare(d, u)
        ds = xr.open_dataset(f"fits/r2/{safe(u)}_theta.nc")
        th = ds["theta"].mean(("chain", "draw"))                      # subject x param (log beta)
        eps = float(ds["eps"].mean())
        sidx = pd.Index(ds["subject"].values).get_indexer(g.subj_ident.values)
        p = fm.prob(np, lambda x: 0.5 * sps.erfc(-x / np.sqrt(2)), th.values[sidx], eps, g, "r2")
        sim = d[d.unit == u].copy()
        sim = sim.loc[g.index]                                          # same order as prepare()
        sim["choice"] = (rng.uniform(size=len(sim)) < p).astype(int)
        sim["unit"] = "SIMR2_" + u
        out.append(sim)
        print(u, "simulated", len(sim), "trials; mean P(LL) =", round(sim.choice.mean(), 3), "eps =", round(eps, 3))
    os.makedirs("sim", exist_ok=True)
    pd.concat(out).to_csv("sim/sim_data.csv.gz", index=False)
    with open("run_model_recovery.bat", "w") as f:
        f.write('@echo off\r\ncd /d "%~dp0"\r\nset PYTHONIOENCODING=utf-8\r\nset DATA_FILE=sim/sim_data.csv.gz\r\n')
        for u in STUDIES:
            su = "SIMR2_" + u
            fmod, hmod = ("r2fp", "h2p") if u in SHORT else ("r2f", "h2")
            for m in ["r2", fmod, hmod]:
                f.write(f'.venv\\Scripts\\python.exe -u 02_fit_models.py --model {m} --out-suffix _recovery '
                        f'--skip-done "{su}" >> logs\\model_recovery.log 2>&1\r\n')
        f.write("echo RECOVERY_DONE >> logs\\model_recovery.log\r\n")
    print("wrote sim/sim_data.csv.gz and run_model_recovery.bat")


def summarise():
    rows = []
    for u in STUDIES:
        su = safe("SIMR2_" + u)
        fmod, hmod = ("r2fp", "h2p") if u in SHORT else ("r2f", "h2")
        pw = {}
        for lab, m in [("R2", "r2"), ("R2F", fmod), ("H2", hmod)]:
            f = f"fits/{m}_recovery/{su}_pointwise.csv.gz"
            if os.path.exists(f):
                pw[lab] = pd.read_csv(f).elpd_loo_i.values
        r = {"study": u, "n_trials": len(next(iter(pw.values()))) if pw else 0}
        for a, b in [("R2", "R2F"), ("R2", "H2")]:
            if a in pw and b in pw:
                diff = pw[a] - pw[b]
                r[f"{a}_minus_{b}"] = diff.sum()
                r[f"{a}_minus_{b}_se"] = np.sqrt(len(diff)) * diff.std()
        s = f"fits/{fmod}_recovery/{su}_summary.csv"
        if os.path.exists(s):
            t = pd.read_csv(s, index_col=0)
            if "mu[g1]" in t.index:
                r["g1"] = t.loc["mu[g1]", "mean"]; r["g1_lo"] = t.loc["mu[g1]", "hdi_3%"]; r["g1_hi"] = t.loc["mu[g1]", "hdi_97%"]
        rows.append(r)
    res = pd.DataFrame(rows)
    res.to_csv("results/model_recovery.csv", index=False)
    print(res.round(2).to_string(index=False))


if __name__ == "__main__":
    simulate() if "--simulate" in sys.argv else summarise()
