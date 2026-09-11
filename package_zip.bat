@echo off
REM Keep this launcher ASCII-only. cmd.exe can misparse UTF-8 batch files.
REM
REM Zip dist\FrameLab for sharing, and REFUSE if an API key is inside.
REM
REM   package_zip.bat
REM
REM Output: FrameLab.zip in the project root.
setlocal
cd /d "%~dp0"

if not exist "dist\FrameLab\FrameLab.exe" goto no_build

echo Checking for API keys inside the build...
dir /s /b "dist\FrameLab\*.key" 2>nul | findstr /r "." >nul
if not errorlevel 1 goto key_found
if exist "dist\FrameLab\deepseek.key" goto key_found

echo Copying the distribution notes...
copy /y "build_tools\分发说明.md" "dist\FrameLab\" >nul 2>&1

echo Zipping (this takes a few minutes)...
del /q FrameLab.zip 2>nul
powershell -NoProfile -Command "Compress-Archive -Path 'dist\FrameLab' -DestinationPath 'FrameLab.zip' -CompressionLevel Optimal"
if errorlevel 1 goto zip_failed

echo.
echo ==================== Ready to share ====================
for %%F in (FrameLab.zip) do echo   FrameLab.zip   %%~zF bytes
echo ========================================================
echo.
pause
exit /b 0

:no_build
echo.
echo dist\FrameLab\FrameLab.exe not found. Run build_exe.bat first.
echo.
pause
exit /b 1

:key_found
echo.
echo ======================= REFUSED ========================
echo An API key file was found inside dist\FrameLab.
echo Sharing it would hand your API quota to everyone who
echo gets this zip.
echo.
echo Delete it first:
echo   del "dist\FrameLab\deepseek.key"
echo.
echo For your own testing, set the DEEPSEEK_API_KEY
echo environment variable instead - it never lands in the zip.
echo ========================================================
echo.
pause
exit /b 1

:zip_failed
echo.
echo Compress-Archive failed. Zip dist\FrameLab by hand instead.
echo.
pause
exit /b 1
