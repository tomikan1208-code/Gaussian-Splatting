@echo off
setlocal DisableDelayedExpansion
"%~dp0apps\blender-5.3.0\blender-5.3.0-windows-x64\5.3\python\bin\python.exe" -B "%~dp0view_feature_matches1.py" %*
if errorlevel 1 pause
