@echo off
chcp 65001 >nul
REM 控制台默认 GBK：中文会是乱码，个别字符还会抛 UnicodeEncodeError 把脚本打死
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
REM Run the evaluation set several times and summarise pass stability.
REM Key goes in deepseek.key. Default 3 rounds; pass --times N to change.
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" ( echo Run run.bat first. & pause & exit /b 1 )
if not exist "deepseek.key" (
  echo.
  echo   deepseek.key not found. Create it in this folder with your API key on one line.
  echo.
  pause
  exit /b 1
)
set "PYTHONPATH=%~dp0src;%~dp0evalset"
".venv\Scripts\python.exe" -m pip install --quiet -r requirements.txt openai
".venv\Scripts\python.exe" evalset\run_repeat.py %*
echo.
pause
