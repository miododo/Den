@echo off
chcp 65001 >nul
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo [ERROR] 未找到 .venv。请先运行 install_windows.bat 或按 README 安装依赖。
  pause
  exit /b 1
)

set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set FLAGS_use_mkldnn=0
if "%OCR_MAX_PAGES%"=="" set OCR_MAX_PAGES=all
if "%OCR_FORCE_LOCAL%"=="" set OCR_FORCE_LOCAL=1
if "%OCR_FALLBACK_ON_ZERO_RECORDS%"=="" set OCR_FALLBACK_ON_ZERO_RECORDS=1
if "%APP_HOST%"=="" set APP_HOST=127.0.0.1
if "%APP_PORT%"=="" set APP_PORT=8010

".venv\Scripts\python.exe" -m uvicorn integrated_test_app:app --host %APP_HOST% --port %APP_PORT%
pause
