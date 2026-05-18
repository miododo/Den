@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set FLAGS_use_mkldnn=0
if "%OCR_MAX_PAGES%"=="" set OCR_MAX_PAGES=all
if "%OCR_FORCE_LOCAL%"=="" set OCR_FORCE_LOCAL=0
if "%OCR_FALLBACK_ON_ZERO_RECORDS%"=="" set OCR_FALLBACK_ON_ZERO_RECORDS=1
if "%APP_HOST%"=="" set APP_HOST=127.0.0.1
if "%APP_PORT%"=="" set APP_PORT=8010

set "PYTHON_CMD="
py -3 --version >nul 2>nul
if not errorlevel 1 set "PYTHON_CMD=py -3"
if defined PYTHON_CMD goto RUN

python --version >nul 2>nul
if not errorlevel 1 set "PYTHON_CMD=python"
if defined PYTHON_CMD goto RUN

echo [ERROR] Python was not found.
pause
exit /b 1

:RUN
echo Using %PYTHON_CMD%
%PYTHON_CMD% -m uvicorn integrated_test_app:app --host %APP_HOST% --port %APP_PORT%
pause
