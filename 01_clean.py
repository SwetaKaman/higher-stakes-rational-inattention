"""Step 1: clean the ITC database and identify magnitude-varying units.

Outputs:
  clean_data.parquet / clean_data.csv.gz  - cleaned trials (delays in days)
  unit_summary.csv                         - per study x subset unit summary
"""
import numpy as np
import pandas as pd

TO_DAYS = {"Days": 1.0, "Weeks": 7.0, "Months": 365.25 / 12, "Hours": 1 / 24, "Seconds": 1 / 86400}

cols = ["paper", "subj_ident", "subset", "subset_label", "session", "trial_idx", "ss_value", "ss_time",
        "ll_value", "ll_time", "choice", "rt", "subj_excl", "trial_excl", "time_unit", "currency"]
d = pd.read_csv("all_data.csv", usecols=cols, low_memory=False)
steps = [("raw", len(d), d.subj_ident.nunique())]

def log(name):
    steps.append((name, len(d), d.subj_ident.nunique()))

d = d[~d.subj_excl & ~d.trial_excl]; log("database subj/trial exclusions")
d = d[d.ss_value > 0]; log("SS > 0")
d = d[d.ll_value > d.ss_value]; log("LL > SS")
d = d[d.ll_time > d.ss_time]; log("LL later than SS")
n = d.groupby("subj_ident").subj_ident.transform("size")
d = d[n >= 10].copy(); log(">= 10 trials per subject")

f = d.time_unit.map(TO_DAYS)
d["ss_days"] = d.ss_time * f
d["ll_days"] = d.ll_time * f
d["unit"] = d.paper + "|s" + d.subset.astype(str)
d["ratio"] = d.ll_value / d.ss_value
d["delay"] = d.ll_days - d.ss_days

for name, nt, ns in steps:
    print(f"{name:30s} trials={nt:>9,d} subjects={ns:>6,d}")
print("units (paper x subset):", d.unit.nunique(), " papers:", d.paper.nunique())

# Choice coding check: P(choice=1) should rise with LL/SS ratio if 1 = larger-later.
print(d.groupby(pd.qcut(d.ratio, 5, duplicates="drop"), observed=True).choice.mean())


def magnitude_varies(g):
    """Magnitude varies >= 2x independently of ratio and delay:
    within some (rounded ratio, delay) cell, SS amounts span >= 2x."""
    key = [g.ratio.round(3), g.delay.round(3)]
    span = g.groupby(key).ss_value.agg(lambda s: s.max() / s.min())
    return bool((span >= 2).any()), float(span.max())


def residual_magnitude_span(g):
    """Primary criterion: 5-95% span of log SS magnitude after regressing out
    log ratio and log delay (OLS). >= 2x counts as magnitude-varying."""
    y = np.log(g.ss_value.values)
    if np.ptp(y) < 1e-9:
        return 1.0
    X = np.column_stack([np.ones(len(g)), np.log(g.ratio), np.log(g.delay)])
    r = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
    return float(np.exp(np.quantile(r, 0.95) - np.quantile(r, 0.05)))


rows = []
for u, g in d.groupby("unit"):
    strict, span = magnitude_varies(g)
    rspan = residual_magnitude_span(g)
    mv = rspan >= 2
    rows.append(dict(unit=u, paper=g.paper.iat[0], subset=g.subset.iat[0], trials=len(g),
                     subjects=g.subj_ident.nunique(), ss_min=g.ss_value.min(), ss_max=g.ss_value.max(),
                     overall_mag_span=g.ss_value.max() / g.ss_value.min(),
                     within_cell_mag_span=span, strict_cell_varying=strict,
                     resid_mag_span=rspan, magnitude_varying=mv,
                     n_ratio=g.ratio.round(3).nunique(), n_delay=g.delay.round(3).nunique(),
                     p_ll=g.choice.mean()))
units = pd.DataFrame(rows)
units.to_csv("unit_summary.csv", index=False)
mv = units[units.magnitude_varying]
print(f"magnitude-varying units: {len(mv)} (papers {mv.paper.nunique()}), "
      f"subjects {mv.subjects.sum():,d}, trials {mv.trials.sum():,d}")

d.drop(columns=["subj_excl", "trial_excl"]).to_csv("clean_data.csv.gz", index=False)
