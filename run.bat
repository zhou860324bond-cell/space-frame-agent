@echo off
chcp 65001 >nul
REM 控制台默认 GBK：中文会是乱码，个别字符还会抛 UnicodeEncodeError 把脚本打死
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
cd /d "%~dp0"
if "%1"=="_inner" goto inner
REM 把用户参数一起带过去：这里是脚本用 _inner 重新调用自己（为了把整段
REM 输出收进 run_log.txt），不透传的话 :inner 里看到的 %1 永远是 _inner，
REM 用户敲的 --quick 在这一步就没了。
call "%~f0" _inner %* > "%~dp0run_log.txt" 2>&1
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
  echo [1/4] creating .venv ...
  %PYEXE% -m venv .venv
  if errorlevel 1 (
    echo ERROR: virtual environment creation failed.
    exit /b 1
  )
) else (
  echo [1/4] .venv exists, reusing
)
set "VPY=%~dp0.venv\Scripts\python.exe"

echo [2/4] installing dependencies ...
"%VPY%" -m pip install --quiet --upgrade pip
"%VPY%" -m pip install --quiet -r requirements.txt
if errorlevel 1 "%VPY%" -m pip install --quiet -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
if errorlevel 1 (
  echo ERROR: dependency install failed.
  exit /b 1
)

REM Examples run BEFORE the regression, on purpose.
REM
REM The old order was: full regression (1600+ tests, several minutes) and
REM only then the examples -- and a single failing test aborted with
REM "Examples were not run". So on any machine with a quirk (no GPU, a
REM non-Chinese locale, a missing system library) the user waited minutes
REM and never got to see what this thing actually does.
REM
REM Showing the product first costs nothing: the regression still runs
REM right after, and its result is still what decides the exit code.
echo [3/4] running the examples
set "PYTHONPATH=%~dp0src"
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

if /i "%~2"=="--quick" (
  echo [4/4] skipping the full regression ^(--quick^)
  echo.
  echo DONE ^(examples only^). Drop --quick for the full regression.
  exit /b 0
)

echo [4/4] running the full regression ^(a few minutes^)
"%VPY%" -m pytest --basetemp="%~dp0.pytest-tmp"
if errorlevel 1 (
  echo.
  echo ERROR: the regression failed. The examples above still ran, so the
  echo        core works here -- the failure is likely environmental.
  echo        Run run_desktop.bat and pick Doctor for a per-item check.
  exit /b 1
)
echo.
echo DONE.
exit /b 0
