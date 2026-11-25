[CmdletBinding()]
param([switch]$WiFi)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$cache = Join-Path $PSScriptRoot 'downloads\windows-setup'
$pythonPath = 'apps\blender-5.3.0\blender-5.3.0-windows-x64\5.3\python\bin\python.exe'
$python = Join-Path $PSScriptRoot $pythonPath

function Install-Archive($Name, $Destination, $Files, $Url, $Sha256) {
    $folder = Join-Path $PSScriptRoot $Destination
    $incomplete = Join-Path $folder '.setup-incomplete'
    $missing = @($Files | Where-Object { !(Test-Path -LiteralPath (Join-Path $folder $_) -PathType Leaf) })
    if (!$missing.Count -and !(Test-Path -LiteralPath $incomplete)) { return }

    $archive = Join-Path $cache ([uri]$Url).Segments[-1]
    if (!(Test-Path -LiteralPath $archive) -or (Get-FileHash -LiteralPath $archive).Hash -ne $Sha256) {
        Write-Host "Downloading $Name..."
        & curl.exe --fail --location --retry 2 --connect-timeout 15 --progress-bar --output "$archive.part" -- $Url
        if ($LASTEXITCODE) { throw "Could not download $Name. Check your internet connection and try again." }
        if ((Get-FileHash -LiteralPath "$archive.part").Hash -ne $Sha256) {
            throw "The $Name download failed its SHA-256 check. Try again."
        }
        Move-Item -LiteralPath "$archive.part" -Destination $archive -Force
    }

    Write-Host "Extracting $Name..."
    New-Item -ItemType Directory -Path $folder -Force | Out-Null
    New-Item -ItemType File -Path $incomplete -Force | Out-Null
    & tar.exe -xf $archive -C $folder
    if ($LASTEXITCODE -or @($Files | Where-Object {
        !(Test-Path -LiteralPath (Join-Path $folder $_) -PathType Leaf)
    }).Count) { throw "Could not extract $Name. Check free disk space and try again." }
    Remove-Item -LiteralPath $incomplete, $archive -Force
}

function Install-PythonPackages($Destination, $Files, $Packages) {
    $folder = Join-Path $PSScriptRoot $Destination
    if (!@($Files | Where-Object { !(Test-Path -LiteralPath (Join-Path $folder $_) -PathType Leaf) }).Count) { return }

    & $python -m pip install --disable-pip-version-check --index-url https://pypi.org/simple `
        --only-binary=:all: --upgrade --target $folder @Packages
    if ($LASTEXITCODE) { throw "Could not install the Python packages in $Destination. Try again." }
}

New-Item -ItemType Directory -Path $cache -Force | Out-Null
try {
    $setupLock = [IO.File]::Open((Join-Path $cache 'setup.lock'), 'OpenOrCreate', 'ReadWrite', 'None')
} catch [IO.IOException] {
    throw 'Windows setup is already running. Wait for the other window to finish.'
}
try {
    Install-Archive 'Blender 5.3.0' 'apps\blender-5.3.0' `
        @('blender-5.3.0-windows-x64\blender.exe', 'blender-5.3.0-windows-x64\5.3\python\bin\python.exe') `
        'https://download.blender.org/release/Blender5.3/blender-5.3.0-windows-x64.zip' `
        '3ac567a63d59d9934f7b9213bfb27ba05760e1fb366f8f55c381573ec3eabfa8'
    Install-Archive 'COLMAP 4.2.1' 'apps\colmap-4.2.1-cuda' @('bin\colmap.exe', 'bin\onnxruntime.dll') `
        'https://github.com/colmap/colmap/releases/download/4.2.1/colmap-x64-windows-cuda.zip' `
        'e9c5cbd84c2ea986d2e970a2473fc2d2e6b34a2cdcf5d3df2765c319a63af881'
    Install-Archive 'Brush 0.3.0' 'apps\brush-0.3.0' @('brush_app.exe') `
        'https://github.com/ArthurBrussee/brush/releases/download/v0.3.0/brush-app-x86_64-pc-windows-msvc.zip' `
        'b68e3e9cf052d51bf3ee30776fa5a364de7f2ba13b58443128ff797bb7bcfcd6'
    Install-Archive 'FFmpeg 9.0.2' 'apps\ffmpeg' `
        @('ffmpeg-9.0.2-essentials_build\bin\ffmpeg.exe', 'ffmpeg-9.0.2-essentials_build\bin\ffprobe.exe') `
        'https://github.com/GyanD/codexffmpeg/releases/download/9.0.2/ffmpeg-9.0.2-essentials_build.zip' `
        '60f467265b1e312373dbcd92200c2618a74850f98d3d078e94296bb3fa2047ba'

    Install-PythonPackages 'apps\heif-python-packages' `
        @('PIL\_imaging.cp313-win_amd64.pyd', '_pillow_heif.cp313-win_amd64.pyd') `
        @('Pillow==12.3.0', 'pillow-heif==1.7.0')
    Install-PythonPackages 'apps\pycolmap-4.2.1-python313' @('pycolmap\_core.cp313-win_amd64.pyd') `
        @('--no-deps', 'pycolmap==4.2.1')
    if ($WiFi) {
        Install-PythonPackages 'apps\wifi-python-packages' @('cryptography\hazmat\bindings\_rust.pyd') `
            @('cryptography==50.0.2')
    }
} finally {
    $setupLock.Dispose()
}
