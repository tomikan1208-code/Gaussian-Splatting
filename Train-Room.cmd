@echo off
setlocal DisableDelayedExpansion
if "%~1"=="" (
    powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0Train-Room.ps1" -SelectQuality
) else (
    powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0Train-Room.ps1" -SelectQuality -InputFolder "%~f1"
)
set "TRAIN_RESULT=%ERRORLEVEL%"
echo.
pause
exit /b %TRAIN_RESULT%
