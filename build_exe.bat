@echo off
REM Keep this launcher ASCII-only. cmd.exe can misparse UTF-8 batch files.
REM
REM Build a standalone Windows folder-app with PyInstaller.
REM   build_exe.bat            normal build (no console window)
REM   build_exe.bat --debug    keeps a console window so startup errors are visible
REM
REM Output: dist\FrameLab\FrameLab.exe   (about 600 MB - 1 GB, this is normal)
setlocal
cd /d "%~dp0"

set "PYTHONUTF8=1"
set "FRAMELAB_CONSOLE="
if /i "%~1"=="--debug" set "FRAMELAB_CONSOLE=1"

if not exist ".venv\Scripts\python.exe" goto no_venv
set "PY=.venv\Scripts\python.exe"

echo [1/4] Checking desktop dependencies...
"%PY%" -c "import PySide6, pyvista, pyvistaqt" 2>nul
if errorlevel 1 goto missing_desktop

echo [2/4] Installing PyInstaller if needed...
"%PY%" -m PyInstaller --version >nul 2>&1
if errorlevel 1 "%PY%" -m pip install "pyinstaller>=6.6"
if errorlevel 1 goto pip_failed

echo [3/4] Building (this takes 5-15 minutes, do not close this window)...
rmdir /s /q build 2>nul
rmdir /s /q dist\FrameLab 2>nul
"%PY%" -m PyInstaller --noconfirm --clean build_tools\frame_lab.spec
if errorlevel 1 goto build_failed

echo [4/4] Verifying that no API key was bundled...
if exist "dist\FrameLab\deepseek.key" goto key_leaked
dir /s /b "dist\FrameLab\*.key" >nul 2>&1
if not errorlevel 1 goto key_leaked

echo.
echo ==================== Build finished ====================
echo   dist\FrameLab\FrameLab.exe
echo.
echo   Share the WHOLE dist\FrameLab folder, zipped.
echo   The exe alone will not run.
echo ========================================================
echo.
pause
exit /b 0

:no_venv
echo.
echo .venv not found. Run run.bat first to build the project environment.
echo.
pause
exit /b 1

:missing_desktop
echo.
echo PySide6 / pyvista / pyvistaqt are missing from .venv.
echo Run run.bat to repair the environment, then try again.
echo.
pause
exit /b 1

:pip_failed
echo.
echo Could not install PyInstaller. Check the network connection.
echo.
pause
exit /b 1

:build_failed
echo.
echo ===================== Build failed =====================
echo Copy the LAST 40 lines above and send them over.
echo Most failures here are a missing hidden import.
echo ========================================================
echo.
pause
exit /b 1

:key_leaked
echo.
echo ABORTED: an API key file was found inside dist\FrameLab.
echo Do NOT share this build. Remove the key file and rebuild.
echo.
pause
exit /b 1
