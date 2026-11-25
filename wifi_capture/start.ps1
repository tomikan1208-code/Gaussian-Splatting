[CmdletBinding()]
param(
    [string]$OutputFolder,
    [switch]$NoBrowser
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$pcUrl = 'http://127.0.0.1:8765/'
$project = Split-Path -Parent $PSScriptRoot

$running = $null
try {
    $running = Invoke-RestMethod -Uri ($pcUrl + 'api/state') -TimeoutSec 2
} catch {
    # A fresh launch has no local listener.
}
if ($running -and $running.PSObject.Properties.Name -contains 'service' -and $running.service -eq 'house-wifi-capture') {
    Write-Host ('The Wi-Fi Capture server is running. Save to: ' + $running.output)
    if (!$NoBrowser) { Start-Process -FilePath $pcUrl }
    exit 0
}

if (!$OutputFolder) { $OutputFolder = Join-Path $project 'wifi-captures' }
$uvRuntimes = Join-Path $env:APPDATA 'uv\python'
$uvPython = Get-ChildItem -LiteralPath $uvRuntimes -Directory -Filter 'cpython-3.13.*-windows-x86_64-none' -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime -Descending |
    ForEach-Object { Join-Path $_.FullName 'python.exe' }
$candidates = @($uvPython) + @(
    (Join-Path $env:LOCALAPPDATA 'Programs\Python\Python311\python.exe'),
    (Join-Path $project 'apps\blender-5.3.0\blender-5.3.0-windows-x64\5.3\python\bin\python.exe')
)
$python = $candidates | Where-Object { $_ -and (Test-Path -LiteralPath $_ -PathType Leaf) } | Select-Object -First 1
if (!$python) { throw 'A local Python runtime could not be found.' }

$openssl = Get-Command openssl.exe -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source -First 1
if (!$openssl) {
    $git = Get-Command git.exe -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source -First 1
    if ($git) { $openssl = Join-Path (Split-Path (Split-Path $git)) 'usr\bin\openssl.exe' }
}

$addresses = Get-NetIPConfiguration |
    Where-Object IPv4DefaultGateway |
    ForEach-Object {
        foreach ($address in $_.IPv4Address) {
            if ($address.IPAddress -match '^(192\.168\.|10\.|172\.(1[6-9]|2[0-9]|3[01])\.)') {
                '{0}/{1}' -f $address.IPAddress, $address.PrefixLength
            }
        }
    } |
    Sort-Object { if ($_ -like '192.168.*') { 0 } else { 1 } }, { $_ } -Unique

$options = @('--output', $OutputFolder)
foreach ($address in $addresses) { $options += @('--lan', $address) }
if ($openssl -and (Test-Path -LiteralPath $openssl -PathType Leaf)) { $options += @('--openssl', $openssl) }
if (!$NoBrowser) { $options += '--open' }
$server = Join-Path $PSScriptRoot 'server.py'
& $python -B -u $server @options
exit $LASTEXITCODE
