@echo off
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
if not exist logs mkdir logs
.venv\Scripts\python.exe -u 02_fit_models.py --model h2 --skip-done >> logs\h2.log 2>&1
.venv\Scripts\python.exe -u 02_fit_models.py --model h2p --skip-done >> logs\h2p.log 2>&1
.venv\Scripts\python.exe -u 02_fit_models.py --model r2 --skip-done >> logs\r2.log 2>&1
.venv\Scripts\python.exe -u 02_fit_models.py --model r2f --skip-done >> logs\r2f.log 2>&1
.venv\Scripts\python.exe -u 02_fit_models.py --model r2fp --skip-done >> logs\r2fp.log 2>&1
set CHAIN_METHOD=parallel
.venv\Scripts\python.exe -u 02_fit_models.py --model r2 --out-suffix _sens --tune 3000 --draws 1500 --target-accept 0.99 --skip-done "Zhou_2022|s1" "Gluth_2017_Exp2|s1" "Chiong_2016|s1" "Zhang_2022|s1" "Lempert_2017b|s1" "Civai_2016|s1" >> logs\r2_sens.log 2>&1
