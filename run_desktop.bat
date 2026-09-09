@echo off
setlocal
REM Keep this launcher ASCII-only. cmd.exe can misparse UTF-8 batch files.
cd /d "%~dp0"

set "PYTHONUTF8=1"
set "QT_API=pyside6"
set "PYTHONPATH="
set "APP_PYTHON="
set "APP_PYTHON_ARGS="

REM Prefer the project virtual environment when its base interpreter still exists.
if not exist ".venv\Scripts\python.exe" goto try_py_launcher
".venv\Scripts\python.exe" --version >nul 2>&1
if errorlevel 1 goto try_py_launcher
set "APP_PYTHON=.venv\Scripts\python.exe"
goto python_ready

:try_py_launcher
py -3 --version >nul 2>&1
if errorlevel 1 goto try_path_python
set "APP_PYTHON=py"
set "APP_PYTHON_ARGS=-3"
goto use_external_python

:try_path_python
python --version >nul 2>&1
if errorlevel 1 goto try_bundled_python
set "APP_PYTHON=python"
goto use_external_python

:try_bundled_python
set "APP_PYTHON=%USERPROFILE%\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
if not exist "%APP_PYTHON%" goto no_python

:use_external_python
REM Reuse the installed project packages when the venv launcher itself is stale.
if exist ".venv\Lib\site-packages" set "PYTHONPATH=%CD%\.venv\Lib\site-packages"

:python_ready
echo Python: %APP_PYTHON% %APP_PYTHON_ARGS%
"%APP_PYTHON%" %APP_PYTHON_ARGS% --version
if errorlevel 1 goto no_python

"%APP_PYTHON%" %APP_PYTHON_ARGS% -c "import PySide6, pyvista, pyvistaqt" 2>nul
if errorlevel 1 goto missing_dependencies

if /i "%~1"=="--check" goto doctor

echo Starting desktop application...
"%APP_PYTHON%" %APP_PYTHON_ARGS% -m desktop.app
if errorlevel 1 goto failed
exit /b 0

:doctor
"%APP_PYTHON%" %APP_PYTHON_ARGS% -m desktop.app --doctor
exit /b %errorlevel%

:missing_dependencies
echo.
echo Desktop dependencies are missing: PySide6, pyvista, or pyvistaqt.
echo Run run.bat to repair the project environment.
echo.
if /i not "%~1"=="--check" pause
exit /b 1

:no_python
echo.
echo Python 3.10 or newer was not found.
echo Rebuild .venv or install Python, then run this launcher again.
echo.
if /i not "%~1"=="--check" pause
exit /b 1

:failed
echo.
echo ================= Desktop startup failed =================
"%APP_PYTHON%" %APP_PYTHON_ARGS% -m desktop.app --doctor
echo ==========================================================
echo Copy the complete diagnostic output when reporting the problem.
echo.
pause
exit /b 1
