@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion
cd /d "%~dp0"

set "LOG=%~dp0install_ocr_log.txt"
set "PIP_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple"
set "PIP_TRUST=--trusted-host pypi.tuna.tsinghua.edu.cn"

echo ==================================================
echo 可选 OCR 依赖安装器
echo ==================================================
echo.

if not exist ".venv\Scripts\python.exe" (
  echo [ERROR] 未找到 .venv。请先运行 install_windows.bat。
  pause
  exit /b 1
)

echo [%date% %time%] optional OCR install started > "%LOG%"
echo 正在安装 PaddleOCR / RapidOCR 等可选 OCR 依赖，可能需要较长时间...
".venv\Scripts\pip.exe" install -r requirements-ocr.txt -i %PIP_INDEX% %PIP_TRUST% >> "%LOG%" 2>&1
if errorlevel 1 (
  echo.
  echo [ERROR] OCR 依赖安装失败。Web 基础功能仍可使用。
  echo 详细原因见 install_ocr_log.txt。
  type "%LOG%"
  pause
  exit /b 1
)

echo.
echo OCR 依赖安装完成。现在运行 start_release_web.bat。
pause
