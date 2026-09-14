@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo [ERROR] Missing .venv. Run run.bat once to create the environment.
  exit /b 1
)
".venv\Scripts\python.exe" tools\run_incremental_tests.py %*
exit /b %errorlevel%
