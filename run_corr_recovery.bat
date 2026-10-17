@echo off
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
set DATA_FILE=sim/sim_corr.csv.gz
.venv\Scripts\python.exe -u 02_fit_models.py --model h2 --out-suffix _corrrec --skip-done "SIMCORR_Cao_2021_Exp2|s2" >> logs\corr_recovery.log 2>&1
.venv\Scripts\python.exe -u 02_fit_models.py --model h2 --out-suffix _corrrec --skip-done "SIMCORR_Kräplin_2020b|s1" >> logs\corr_recovery.log 2>&1
.venv\Scripts\python.exe -u 02_fit_models.py --model h2 --out-suffix _corrrec --skip-done "SIMCORR_Zgonnikov_2019|s1" >> logs\corr_recovery.log 2>&1
.venv\Scripts\python.exe -u 02_fit_models.py --model h2 --out-suffix _corrrec --skip-done "SIMCORR_Stevens_2016|s1" >> logs\corr_recovery.log 2>&1
echo CORR_DONE >> logs\corr_recovery.log
