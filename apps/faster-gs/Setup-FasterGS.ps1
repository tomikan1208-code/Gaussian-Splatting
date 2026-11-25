[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Require-Success([string]$Action) {
    if ($LASTEXITCODE) { throw "$Action exited with code $LASTEXITCODE." }
}

function Restore-Source([string]$Url, [string]$Path, [string]$Revision) {
    if (!(Test-Path -LiteralPath (Join-Path $Path '.git'))) {
        git clone --no-checkout --depth 1 $Url $Path
        Require-Success 'git clone'
        git -C $Path fetch --depth 1 origin $Revision
        Require-Success 'git fetch'
        git -C $Path checkout --detach $Revision
        Require-Success 'git checkout'
    }
    $current = git -C $Path rev-parse HEAD
    Require-Success 'git rev-parse'
    if ($current -ne $Revision) { throw "Expected $Revision in $Path; found $current." }
}

function Apply-SourcePatch([string]$Path, [string]$PatchName) {
    $patch = Join-Path $PSScriptRoot $PatchName
    git -C $Path apply --reverse --check $patch 2>$null
    if ($LASTEXITCODE) {
        git -C $Path apply --check $patch
        Require-Success 'git apply --check'
        git -C $Path apply $patch
        Require-Success 'git apply'
    }
}

$nerficg = Join-Path $PSScriptRoot 'nerficg'
$fasterGs = Join-Path $nerficg 'src/Methods/FasterGS'
$ssim = Join-Path $PSScriptRoot 'fused-ssim'
Restore-Source 'https://github.com/nerficg-project/nerficg.git' $nerficg '623ad16ec6e05a5c5cedacc046273b63eac2686e'
Restore-Source 'https://github.com/nerficg-project/faster-gaussian-splatting.git' $fasterGs '4f8e18690464250c2443bfed79d17a36f5c99faf'
Restore-Source 'https://github.com/rahul-goel/fused-ssim.git' $ssim 'a7c48d6dd7ac6dc39a7958c7c4452e0b10418f38'
Apply-SourcePatch $nerficg 'nerficg.patch'
Apply-SourcePatch $fasterGs 'faster-gs.patch'

$base = 'https://developer.download.nvidia.com/compute/cuda/redist'
$manifest = Invoke-RestMethod "$base/redistrib_12.8.1.json"
$cuda = Join-Path $PSScriptRoot 'cuda'
$downloads = Join-Path $PSScriptRoot '../../downloads/faster-gs-cuda'
New-Item -ItemType Directory -Force -Path $cuda,$downloads | Out-Null
$components = 'cuda_nvcc','cuda_cudart','cuda_cccl','libcublas','libcusparse','libcusolver','libcurand'
foreach ($component in $components) {
    $record = $manifest.$component.'windows-x86_64'
    $zip = Join-Path $downloads (Split-Path $record.relative_path -Leaf)
    if (!(Test-Path -LiteralPath $zip) -or (Get-FileHash -LiteralPath $zip).Hash -ne $record.sha256) {
        curl.exe --fail --location --silent --show-error --retry 2 -o $zip "$base/$($record.relative_path)"
        Require-Success "Download $component"
        if ((Get-FileHash -LiteralPath $zip).Hash -ne $record.sha256) { throw "Invalid CUDA archive hash: $zip" }
    }
    $extract = @('-xf', $zip, '--strip-components=1', '-C', $cuda)
    if ($component.StartsWith('lib')) {
        $extract += [System.IO.Path]::GetFileNameWithoutExtension($zip) + '/include'
    }
    tar.exe @extract
    Require-Success "Extract $component"
}

$venv = Join-Path $PSScriptRoot '.venv'
$python = Join-Path $venv 'Scripts/python.exe'
$env:UV_CACHE_DIR = Join-Path $PSScriptRoot 'uv-cache'
if (!(Test-Path -LiteralPath $python)) {
    uv venv --python 3.11 $venv
    Require-Success 'Create Faster-GS Python environment'
}
uv pip install --python $python --index 'https://download.pytorch.org/whl/cu128' 'torch==2.8.0+cu128' 'torchvision==0.23.0+cu128'
Require-Success 'Install PyTorch CUDA'
uv pip install --python $python -r (Join-Path $PSScriptRoot 'requirements.txt')
Require-Success 'Install Faster-GS dependencies'
& (Join-Path $PSScriptRoot 'Build-Extensions.ps1')
