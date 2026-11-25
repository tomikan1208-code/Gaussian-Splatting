[CmdletBinding()]
param(
    [string]$OutputFolder,
    [switch]$NoBrowser
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$pcUrl = 'http://127.0.0.1:8765/'
$project = Split-Path -Parent $PSScriptRoot
if (!$OutputFolder) { $OutputFolder = Join-Path $project 'input' }
$OutputFolder = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($OutputFolder)

$running = $null
try {
    $running = Invoke-RestMethod -Uri ($pcUrl + 'api/state') -TimeoutSec 2
} catch {
    # A fresh launch has no local listener.
}
if ($running -and $running.PSObject.Properties.Name -contains 'service' -and $running.service -eq 'house-wifi-capture') {
    Write-Host ('The Wi-Fi Capture server is running. Save to: ' + $running.output)
    if (!$NoBrowser) { Start-Process -FilePath $pcUrl }
    if ($running.output -ne $OutputFolder -or $running.PSObject.Properties.Name -notcontains 'training') {
        Write-Host 'Stop the running server from its PC page, then open this launcher again.'
        exit 1
    }
    exit 0
}

& (Join-Path $project 'Setup-Windows.ps1') -WiFi
$python = Join-Path $project 'apps\blender-5.3.0\blender-5.3.0-windows-x64\5.3\python\bin\python.exe'

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
if (!$NoBrowser) { $options += '--open' }
$server = Join-Path $PSScriptRoot 'server.py'
& $python -B -u $server @options
exit $LASTEXITCODE
