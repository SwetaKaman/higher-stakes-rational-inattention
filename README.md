# Rational inattention and the magnitude dependence of choice noise: code and results

Code and derived results for the manuscript *"Higher stakes sharpen intertemporal choices far less than rational inattention
predicts"*.

## Data

The trial-level data come from the ITC Database (Pongratz & Schoemann, 2026, *Scientific Data*, 13, 323,
https://doi.org/10.1038/s41597-026-06947-4). Download the data file from the OSF project
(https://doi.org/10.17605/OSF.IO/3WSAE; licence CC BY-NC-SA) and save it as `all_data.csv` in this folder.
The raw data are not redistributed here.

## Setup

Python 3.11 is required. Create a virtual environment called `.venv` in this folder and install the packages:

```
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

(On macOS/Linux use `.venv/bin/python` and run the commands in the `.bat` files directly.)

## Running the analysis (in order)

| Step | Command | What it does | Main outputs |
|---|---|---|---|
| 1 | `python 01_clean.py` | Cleans the database, converts delays to days, finds magnitude-varying studies | `clean_data.csv.gz`, `unit_summary.csv` |
| 2 | `run_main_fits.bat` | Fits H2, H2 (pooled slopes), R2, R2F and R2F (pooled slope) to every study, plus the R2 sensitivity refits | `fits/<model>/` |
| 3 | `python 03_summarise.py` | Per-study summaries, model comparison (PSIS-LOO), random-effects pooling | `results/*.csv` |
| 4 | `python 08_ppc.py` | Posterior predictive checks and calibration slopes | `results/ppc_*.csv`, Figure 7 |
| 5 | `python 10_model_recovery.py --simulate`, then `run_model_recovery.bat`, then `python 10_model_recovery.py` | Model recovery: simulate from R2 and refit all models | `sim/`, `results/model_recovery.csv` |
| 6 | `run_robust_logit.bat`, `run_robust_subjlapse.bat`, then `python 13_robustness.py` | Refits with a logistic link and with per-participant lapse rates | `results/robustness.csv` |
| 7 | `python 12_moderators.py` | Moderator meta-regressions and subgroup estimates | `results/moderators_*.csv`, Figure 8 |
| 8 | `python 14_r2_elasticity.py` | Exact elasticities of choice precision (Propositions 2 and 4) on the real designs | `results/r2_elasticities.csv` |
| 9 | `python 07_make_figures.py`, `python 11_method_figure.py`, `python 09_supp_table.py` | Figures 1 to 6 and Table A.1 | `figures/`, `table_s1.tex` |

Model fitting is the slow step (days of CPU time for all studies). Every fit can be run for a single study, e.g.
`python 02_fit_models.py --model r2f "Keidel_2024|s1"`. Options: `--draws`, `--tune`, `--target-accept`,
`--out-suffix`; environment variables `LINK=logit` and `LAPSE_MODE=subject` select the robustness variants.

## What is included

In the GitHub repository the `fits/` folder is stored as `fits.zip` (GitHub limits uploads to 100 files); unzip it in this folder to restore `fits/`.

* `fits/<model>/<study>_summary.csv`: posterior summaries (means, SDs, 94% HDIs, R-hat, ESS) and fit diagnostics for
  every fit, including model recovery (`*_recovery`) and robustness (`*_logit`, `*_subjlapse`) fits.
* `fits/<model>/<study>_pointwise.csv.gz`: trial-wise PSIS-LOO values and Pareto-k diagnostics, which are enough to
  reproduce every model comparison without refitting.
* `results/`: all derived tables used in the paper.
* `figures/`: all figures (vector PDF).
* `sim/sim_data.csv.gz`: the data simulated from R2 for the model-recovery analysis.

The full posterior draws (`*_theta.nc`, about 4 GB) are not included because of their size. They are available
from the author on request, or can be regenerated with step 2.

## Exclusions

See `EXCLUSIONS.md`.
