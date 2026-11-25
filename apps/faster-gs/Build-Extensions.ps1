[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$python = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
$cuda = Join-Path $PSScriptRoot 'cuda'
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio/Installer/vswhere.exe'
$vs = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (!$vs) { throw 'Visual Studio with the C++ x64 build tools is required.' }
$vcvars = Join-Path $vs 'VC/Auxiliary/Build/vcvars64.bat'
$startup = [Diagnostics.ProcessStartInfo]::new('cmd.exe', ('/d /s /c ""{0}" >nul && set"' -f $vcvars))
$startup.UseShellExecute = $false
$startup.RedirectStandardOutput = $true
$compiler = [Diagnostics.Process]::Start($startup)
$compilerEnvironment = $compiler.StandardOutput.ReadToEnd() -split '\r?\n'
$compiler.WaitForExit()
if ($compiler.ExitCode) { throw 'Could not initialize the Visual Studio C++ compiler.' }
$compiler.Dispose()
foreach ($line in $compilerEnvironment) {
    if ($line -match '^([^=]+)=(.*)$') {
        [Environment]::SetEnvironmentVariable($Matches[1], $Matches[2], 'Process')
    }
}

$env:CUDA_PATH = $cuda
$env:CUDA_HOME = $cuda
$env:PATH = "$cuda\bin;$(Split-Path $python);$env:PATH"
$env:DISTUTILS_USE_SDK = '1'
$env:MAX_JOBS = '2'
$env:VSLANG = '1033'
$env:TORCH_CUDA_ARCH_LIST = & $python -c "import torch; print('%s.%s' % torch.cuda.get_device_capability())"
if ($LASTEXITCODE) { throw 'Could not initialize PyTorch CUDA.' }
$env:CUDA_ARCHITECTURES = $env:TORCH_CUDA_ARCH_LIST.Replace('.', '')
$env:UV_CACHE_DIR = Join-Path $PSScriptRoot 'uv-cache'

$backend = Join-Path $PSScriptRoot 'nerficg/src/Methods/FasterGS/FasterGSCudaBackend'
$morton = Join-Path $PSScriptRoot 'nerficg/src/CudaUtils/MortonEncoding'
$ssim = Join-Path $PSScriptRoot 'fused-ssim'
foreach ($source in $backend, $morton, $ssim) {
    & uv pip install --python $python --no-build-isolation $source
    if ($LASTEXITCODE) { throw "CUDA extension build failed: $source" }
}
& $python -c 'import torch; from FasterGSCudaBackend import _C; from MortonEncoding import morton_encode; from fused_ssim import fused_ssim; print(torch.__version__, torch.cuda.get_device_name())'
if ($LASTEXITCODE) { throw 'Could not import the compiled CUDA extensions.' }
