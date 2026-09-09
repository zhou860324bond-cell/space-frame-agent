@echo off
chcp 65001 >nul
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo ERROR: .venv not found. Run run.bat first.
  exit /b 1
)
".venv\Scripts\python.exe" tools\render_viewport.py
exit /b %errorlevel%
