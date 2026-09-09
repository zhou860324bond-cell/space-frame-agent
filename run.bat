@echo off
chcp 65001 >nul
REM 控制台默认 GBK：中文会是乱码，个别字符还会抛 UnicodeEncodeError 把脚本打死
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
cd /d "%~dp0"
if "%1"=="_inner" goto inner
call "%~f0" _inner > "%~dp0run_log.txt" 2>&1
set "RUN_CODE=%ERRORLEVEL%"
type "%~dp0run_log.txt"
echo.
pause
exit /b %RUN_CODE%

:inner
echo ==== space frame kernel ====
set "PYEXE=python"
where py >nul 2>nul && set "PYEXE=py -3"
%PYEXE% -V
if errorlevel 1 (
  echo ERROR: Python not found. Install Python 3.10+ and tick "Add python.exe to PATH".
  exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
  echo [1/3] creating .venv ...
  %PYEXE% -m venv .venv
  if errorlevel 1 (
    echo ERROR: virtual environment creation failed.
    exit /b 1
  )
) else (
  echo [1/3] .venv exists, reusing
)
set "VPY=%~dp0.venv\Scripts\python.exe"

echo [2/3] installing dependencies ...
"%VPY%" -m pip install --quiet --upgrade pip
"%VPY%" -m pip install --quiet -r requirements.txt
if errorlevel 1 "%VPY%" -m pip install --quiet -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
if errorlevel 1 (
  echo ERROR: dependency install failed.
  exit /b 1
)

echo [3/3] running tests and the example
set "PYTHONPATH=%~dp0src"
"%VPY%" -m pytest --basetemp="%~dp0.pytest-tmp"
if errorlevel 1 (
  echo ERROR: tests failed. Examples were not run.
  exit /b 1
)
echo.
"%VPY%" examples\run_json.py examples\portal_frame_cases.json
if errorlevel 1 (
  echo ERROR: JSON end-to-end example failed.
  exit /b 1
)
echo.
"%VPY%" examples\agent_demo.py
if errorlevel 1 (
  echo ERROR: Agent offline demo failed.
  exit /b 1
)
echo.
echo DONE.
exit /b 0
