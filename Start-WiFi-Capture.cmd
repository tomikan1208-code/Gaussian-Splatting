@echo off
setlocal
if "%~1"=="" (
  powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0wifi_capture\start.ps1"
) else (
  powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0wifi_capture\start.ps1" -OutputFolder "%~1"
)
set "CAPTURE_STATUS=%ERRORLEVEL%"
if not "%CAPTURE_STATUS%"=="0" pause
exit /b %CAPTURE_STATUS%
