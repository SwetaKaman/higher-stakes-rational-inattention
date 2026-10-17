@echo off
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
set NOISE_T=1
rem robustness: R2 and R2F with choice noise that grows with delay (NOISE_T=1)
.venv\Scripts\python.exe -u 02_fit_models.py --model r2f --out-suffix _noiset --skip-done "Scherbaum_2018a|s1" "Yang_2022|s1" "Castrellon_2019|s1" >> logs\fixes_B.log 2>&1
.venv\Scripts\python.exe -u 02_fit_models.py --model r2fp --out-suffix _noiset --skip-done "Bixter_2019|s1" "Berenson_2016|s1" "Keidel_2024|s1" >> logs\fixes_B.log 2>&1
.venv\Scripts\python.exe -u 02_fit_models.py --model r2 --out-suffix _noiset --skip-done "Scherbaum_2018a|s1" "Yang_2022|s1" "Bixter_2019|s1" "Castrellon_2019|s1" "Berenson_2016|s1" "Keidel_2024|s1" >> logs\fixes_B.log 2>&1
echo ROBUST_DONE >> logs\fixes_B.log
