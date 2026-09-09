@echo off
chcp 65001 >nul
REM 控制台默认 GBK：中文会是乱码，个别字符还会抛 UnicodeEncodeError 把脚本打死
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
REM Launch the graphical interface in your browser.
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" ( echo Run run.bat first. & pause & exit /b 1 )
set "PYTHONPATH=%~dp0src"

REM Always sync requirements first. Installing only streamlit used to leave
REM matplotlib missing, and the app died on import.
echo Checking dependencies ...
".venv\Scripts\python.exe" -m pip install --quiet -r requirements.txt
if errorlevel 1 (
  ".venv\Scripts\python.exe" -m pip install --quiet -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
)
".venv\Scripts\python.exe" -c "import streamlit, plotly, matplotlib, numpy, scipy, jsonschema" 2>nul
if errorlevel 1 (
  echo.
  echo Dependency install failed. Run this manually and read the error:
  echo     .venv\Scripts\python.exe -m pip install -r requirements.txt
  echo.
  pause
  exit /b 1
)

echo.
echo Starting the interface. Your browser should open at http://localhost:8501
echo Close this window to stop it.
echo.
".venv\Scripts\python.exe" -m streamlit run gui_app.py
pause
