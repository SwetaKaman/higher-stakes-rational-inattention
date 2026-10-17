@echo off
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
set DATA_FILE=sim/sim_data.csv.gz
.venv\Scripts\python.exe -u 02_fit_models.py --model r2 --out-suffix _recovery --skip-done "SIMR2_Keidel_2024|s1" >> logs\model_recovery.log 2>&1
.venv\Scripts\python.exe -u 02_fit_models.py --model r2fp --out-suffix _recovery --skip-done "SIMR2_Keidel_2024|s1" >> logs\model_recovery.log 2>&1
.venv\Scripts\python.exe -u 02_fit_models.py --model h2p --out-suffix _recovery --skip-done "SIMR2_Keidel_2024|s1" >> logs\model_recovery.log 2>&1
.venv\Scripts\python.exe -u 02_fit_models.py --model r2 --out-suffix _recovery --skip-done "SIMR2_Berenson_2016|s1" >> logs\model_recovery.log 2>&1
.venv\Scripts\python.exe -u 02_fit_models.py --model r2fp --out-suffix _recovery --skip-done "SIMR2_Berenson_2016|s1" >> logs\model_recovery.log 2>&1
.venv\Scripts\python.exe -u 02_fit_models.py --model h2p --out-suffix _recovery --skip-done "SIMR2_Berenson_2016|s1" >> logs\model_recovery.log 2>&1
.venv\Scripts\python.exe -u 02_fit_models.py --model r2 --out-suffix _recovery --skip-done "SIMR2_Castrellon_2019|s1" >> logs\model_recovery.log 2>&1
.venv\Scripts\python.exe -u 02_fit_models.py --model r2f --out-suffix _recovery --skip-done "SIMR2_Castrellon_2019|s1" >> logs\model_recovery.log 2>&1
.venv\Scripts\python.exe -u 02_fit_models.py --model h2 --out-suffix _recovery --skip-done "SIMR2_Castrellon_2019|s1" >> logs\model_recovery.log 2>&1
.venv\Scripts\python.exe -u 02_fit_models.py --model r2 --out-suffix _recovery --skip-done "SIMR2_Bixter_2019|s1" >> logs\model_recovery.log 2>&1
.venv\Scripts\python.exe -u 02_fit_models.py --model r2fp --out-suffix _recovery --skip-done "SIMR2_Bixter_2019|s1" >> logs\model_recovery.log 2>&1
.venv\Scripts\python.exe -u 02_fit_models.py --model h2p --out-suffix _recovery --skip-done "SIMR2_Bixter_2019|s1" >> logs\model_recovery.log 2>&1
.venv\Scripts\python.exe -u 02_fit_models.py --model r2 --out-suffix _recovery --skip-done "SIMR2_Yang_2022|s1" >> logs\model_recovery.log 2>&1
.venv\Scripts\python.exe -u 02_fit_models.py --model r2f --out-suffix _recovery --skip-done "SIMR2_Yang_2022|s1" >> logs\model_recovery.log 2>&1
.venv\Scripts\python.exe -u 02_fit_models.py --model h2 --out-suffix _recovery --skip-done "SIMR2_Yang_2022|s1" >> logs\model_recovery.log 2>&1
.venv\Scripts\python.exe -u 02_fit_models.py --model r2 --out-suffix _recovery --skip-done "SIMR2_Scherbaum_2018a|s1" >> logs\model_recovery.log 2>&1
.venv\Scripts\python.exe -u 02_fit_models.py --model r2f --out-suffix _recovery --skip-done "SIMR2_Scherbaum_2018a|s1" >> logs\model_recovery.log 2>&1
.venv\Scripts\python.exe -u 02_fit_models.py --model h2 --out-suffix _recovery --skip-done "SIMR2_Scherbaum_2018a|s1" >> logs\model_recovery.log 2>&1
echo RECOVERY_DONE >> logs\model_recovery.log
