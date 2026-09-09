@echo off
REM Submit every benchmark job, then tell you what to do next.
REM The heavy lifting is in run_jobs.py -- a cmd FOR loop calling abaqus.bat
REM silently aborts the whole batch partway through.
cd /d "%~dp0"

where abaqus >nul 2>nul
if errorlevel 1 (
  echo.
  echo   Abaqus not found on PATH.
  echo   Open the "Abaqus Command" shortcut from the Start menu and run this there.
  echo.
  pause
  exit /b 1
)

set "PY=..\.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=python"

"%PY%" run_jobs.py %*
echo.
pause
