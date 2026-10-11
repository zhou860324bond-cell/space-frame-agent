@echo off
setlocal
cd /d "%~dp0"
set "ABAQUS_CMD=abaqus"
where abaqus >nul 2>nul
if errorlevel 1 (
  if exist "E:\SIMULIA\Abaqus\Commands\abaqus.BAT" (
    set "ABAQUS_CMD=E:\SIMULIA\Abaqus\Commands\abaqus.BAT"
  ) else (
    echo Abaqus command not found. Open Abaqus Command and run this batch again.
    pause
    exit /b 1
  )
)
set "PYTHONHOME="
set "PYTHONPATH="
call "%ABAQUS_CMD%" python run_case.py "%ABAQUS_CMD%"
if errorlevel 1 (
  echo Analysis failed. Inspect the console files and the job logs.
  pause
  exit /b 1
)
echo Analysis completed. Open IV5_global_and_node3.cae and local_h10.odb.
pause
