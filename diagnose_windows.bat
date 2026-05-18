@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo ==================================================
echo Env AI Validator - diagnostics
echo ==================================================
echo.
echo Current folder:
cd
echo.
echo Python launcher:
py -3 --version
echo.
echo Python from PATH:
python --version
echo.
echo Virtual environment:
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" --version
  ".venv\Scripts\python.exe" -c "import sys; print(sys.executable)"
) else (
  echo .venv not found
)
echo.
echo Required files:
dir README.md requirements-minimal.txt integrated_test_app.py sample_data\sample_env_report.pdf
echo.
echo Last install log:
if exist install_log.txt (
  type install_log.txt
) else (
  echo install_log.txt not found
)
echo.
pause
