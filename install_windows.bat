@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

set "LOG=%CD%\install_log.txt"
set "PIP_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple"
set "PIP_TRUST=--trusted-host pypi.tuna.tsinghua.edu.cn"

echo ==================================================
echo Env AI Validator - Windows installer
echo ==================================================
echo.
echo This window will show progress. Do not close it.
echo Log file: %LOG%
echo [%date% %time%] install started > "%LOG%"

set "PYTHON_CMD="
call :TRY_PYTHON "py -3.12"
call :TRY_PYTHON "py -3.11"
call :TRY_PYTHON "py -3.10"
call :TRY_PYTHON "py -3.13"
call :TRY_PYTHON "python"
if defined PYTHON_CMD goto PYTHON_FOUND

echo.
echo [ERROR] Compatible Python was not found.
echo Please install standard Python 3.10 - 3.13. Do not use Python 3.13 free-threading / 3.13t.
echo.
echo Detected Python runtimes:
py -0p
echo [ERROR] Compatible Python not found. >> "%LOG%"
pause
exit /b 1

:PYTHON_FOUND
echo.
echo Using Python command: %PYTHON_CMD%
echo Using Python command: %PYTHON_CMD% >> "%LOG%"

echo.
echo [1/3] Creating virtual environment .venv ...
%PYTHON_CMD% -m venv .venv
if errorlevel 1 goto FAIL

echo.
echo [2/3] Upgrading pip ...
".venv\Scripts\python.exe" -m pip install --upgrade pip -i %PIP_INDEX% %PIP_TRUST%
if errorlevel 1 goto FAIL

echo.
echo [3/3] Installing minimal Web dependencies ...
".venv\Scripts\pip.exe" install -r requirements-minimal.txt -i %PIP_INDEX% %PIP_TRUST%
if errorlevel 1 goto FAIL

echo.
".venv\Scripts\python.exe" -c "import fastapi, fitz, PIL, numpy, requests, jinja2, openpyxl; print('Dependency check: OK')"
if errorlevel 1 goto FAIL

echo [%date% %time%] install completed >> "%LOG%"
echo.
echo ==================================================
echo Install completed.
echo Next: run start_release_web.bat
echo Optional OCR: run install_ocr_windows.bat after the Web app works.
echo ==================================================
pause
exit /b 0

:FAIL
echo [%date% %time%] install failed >> "%LOG%"
echo.
echo [ERROR] Install failed.
echo.
echo Common causes:
echo 1. Python is not installed or is too old.
echo 2. pip download failed.
echo 3. Network/proxy blocked Python package downloads.
echo.
echo Manual retry:
echo .venv\Scripts\pip.exe install -r requirements-minimal.txt -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn
echo.
pause
exit /b 1

:TRY_PYTHON
if defined PYTHON_CMD goto :eof
%~1 -c "import sys, sysconfig; ok = sys.version_info[0] == 3 and sys.version_info[1] in (10, 11, 12, 13) and sysconfig.get_config_var('Py_GIL_DISABLED') != 1; print(sys.version); raise SystemExit(0 if ok else 1)" >nul 2>&1
if not errorlevel 1 set "PYTHON_CMD=%~1"
goto :eof
