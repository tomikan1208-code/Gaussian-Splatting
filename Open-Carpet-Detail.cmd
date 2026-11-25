@echo off
setlocal
set "BLENDER_USER_CONFIG="
set "BLENDER_USER_EXTENSIONS="
set "BLENDER_USER_SCRIPTS="
set "BLENDER_USER_RESOURCES="
set "SPLAT_BLENDER=%~dp0apps\blender-5.3.0\blender-5.3.0-windows-x64\blender.exe"
set "SPLAT_PREVIEW=%~dp0regions\carpet_9954_v001\Carpet-Detail.blend"
if not exist "%SPLAT_BLENDER%" (
    echo Blender 5.3 was not found.
    pause
    exit /b 1
)
if not exist "%SPLAT_PREVIEW%" (
    echo Carpet-Detail.blend was not found.
    pause
    exit /b 1
)
start "" "%SPLAT_BLENDER%" --factory-startup --disable-autoexec "%SPLAT_PREVIEW%"

