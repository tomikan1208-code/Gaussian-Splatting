@echo off
setlocal
set "BLENDER_USER_CONFIG="
set "BLENDER_USER_EXTENSIONS="
set "BLENDER_USER_SCRIPTS="
set "BLENDER_USER_RESOURCES="
set "SPLAT_BLENDER=%~dp0apps\blender-5.3.0\blender-5.3.0-windows-x64\blender.exe"
if "%~1"=="" (
    start "" "%SPLAT_BLENDER%" --factory-startup --disable-autoexec --python "%~dp0open_latest_room.py"
) else (
    start "" "%SPLAT_BLENDER%" --disable-autoexec %*
)
