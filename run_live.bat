@echo off
chcp 65001 >nul
REM 控制台默认 GBK：中文会是乱码，个别字符还会抛 UnicodeEncodeError 把脚本打死
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
REM Run the live LLM probe cases. Put the API key in deepseek.key next to this file.
REM This script never reads or prints the key value itself.
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo No .venv found. Run run.bat first to create the environment.
  pause
  exit /b 1
)

if not exist "deepseek.key" (
  echo.
  echo   deepseek.key not found.
  echo.
  echo   Create a file named  deepseek.key  in this folder containing
  echo   ONLY your API key on a single line, then run this again.
  echo   It is already listed in .gitignore.
  echo.
  pause
  exit /b 1
)

set "PYTHONPATH=%~dp0src"
".venv\Scripts\python.exe" -m pip install --quiet -r requirements.txt openai
".venv\Scripts\python.exe" examples\agent_live.py %*
echo.
pause
