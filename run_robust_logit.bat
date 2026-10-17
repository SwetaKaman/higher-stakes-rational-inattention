@echo off
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
set LINK=logit
.venv\Scripts\python.exe -u 02_fit_models.py --model r2 --out-suffix _logit --skip-done "Keidel_2024|s1" >> logs\robust_logit.log 2>&1
.venv\Scripts\python.exe -u 02_fit_models.py --model r2fp --out-suffix _logit --skip-done "Keidel_2024|s1" >> logs\robust_logit.log 2>&1
.venv\Scripts\python.exe -u 02_fit_models.py --model h2p --out-suffix _logit --skip-done "Keidel_2024|s1" >> logs\robust_logit.log 2>&1
.venv\Scripts\python.exe -u 02_fit_models.py --model r2 --out-suffix _logit --skip-done "Berenson_2016|s1" >> logs\robust_logit.log 2>&1
.venv\Scripts\python.exe -u 02_fit_models.py --model r2fp --out-suffix _logit --skip-done "Berenson_2016|s1" >> logs\robust_logit.log 2>&1
.venv\Scripts\python.exe -u 02_fit_models.py --model h2p --out-suffix _logit --skip-done "Berenson_2016|s1" >> logs\robust_logit.log 2>&1
.venv\Scripts\python.exe -u 02_fit_models.py --model r2 --out-suffix _logit --skip-done "Castrellon_2019|s1" >> logs\robust_logit.log 2>&1
.venv\Scripts\python.exe -u 02_fit_models.py --model r2f --out-suffix _logit --skip-done "Castrellon_2019|s1" >> logs\robust_logit.log 2>&1
.venv\Scripts\python.exe -u 02_fit_models.py --model h2 --out-suffix _logit --skip-done "Castrellon_2019|s1" >> logs\robust_logit.log 2>&1
.venv\Scripts\python.exe -u 02_fit_models.py --model r2 --out-suffix _logit --skip-done "Bixter_2019|s1" >> logs\robust_logit.log 2>&1
.venv\Scripts\python.exe -u 02_fit_models.py --model r2fp --out-suffix _logit --skip-done "Bixter_2019|s1" >> logs\robust_logit.log 2>&1
.venv\Scripts\python.exe -u 02_fit_models.py --model h2p --out-suffix _logit --skip-done "Bixter_2019|s1" >> logs\robust_logit.log 2>&1
.venv\Scripts\python.exe -u 02_fit_models.py --model r2 --out-suffix _logit --skip-done "Yang_2022|s1" >> logs\robust_logit.log 2>&1
.venv\Scripts\python.exe -u 02_fit_models.py --model r2f --out-suffix _logit --skip-done "Yang_2022|s1" >> logs\robust_logit.log 2>&1
.venv\Scripts\python.exe -u 02_fit_models.py --model h2 --out-suffix _logit --skip-done "Yang_2022|s1" >> logs\robust_logit.log 2>&1
.venv\Scripts\python.exe -u 02_fit_models.py --model r2 --out-suffix _logit --skip-done "Scherbaum_2018a|s1" >> logs\robust_logit.log 2>&1
.venv\Scripts\python.exe -u 02_fit_models.py --model r2f --out-suffix _logit --skip-done "Scherbaum_2018a|s1" >> logs\robust_logit.log 2>&1
.venv\Scripts\python.exe -u 02_fit_models.py --model h2 --out-suffix _logit --skip-done "Scherbaum_2018a|s1" >> logs\robust_logit.log 2>&1
echo ROBUST_DONE >> logs\robust_logit.log
