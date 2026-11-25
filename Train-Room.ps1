[CmdletBinding()]
param(
    [string]$InputFolder,
    [string]$FramesPath,
    [ValidateRange(200, 90000)][int]$Steps = 12000,
    [ValidateRange(10000, 6000000)][int]$MaxSplats = 4000000,
    [ValidateRange(0, 16384)][int]$MaxEdge = 0,
    [ValidateRange(0.25, 60)][double]$VideoFps = 5,
    [double]$AverageSteps,
    [string]$ReuseRun,
    [ValidateSet('loma','sift')][string]$Matcher,
    [ValidateSet('auto','exhaustive','selective')][string]$Pairing = 'auto',
    [ValidateSet('auto','incremental','global')][string]$Mapper = 'auto',
    [ValidateSet('brush','faster-gs','faster-gs-optimized')][string]$Trainer = 'brush',
    [switch]$SelectQuality,
    [string]$InitPly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$culture = [Globalization.CultureInfo]::InvariantCulture
if (!$InputFolder) { $InputFolder = Join-Path $PSScriptRoot 'input' }

if ($SelectQuality) {
    Write-Host 'ガウススプラットの作成設定'
    Write-Host "入力フォルダ  $InputFolder"
    Write-Host '  4K（UHD）  3840 × 2160 px → 3840'
    Write-Host '  フルHD     1920 × 1080 px → 1920'
    Write-Host '  原寸       元の解像度    → 0'

    $edgeDefault = if ($MaxEdge) { "$MaxEdge px" } else { '原寸' }
    while ($true) {
        Write-Host "学習画像の解像度（長辺 px） [Enterで $edgeDefault] " -NoNewline
        $answer = [Console]::ReadLine()
        if ($null -eq $answer) { exit 0 }
        $answer = $answer.Normalize([Text.NormalizationForm]::FormKC).Trim()
        if (!$answer) { break }

        $edge = 0
        if ([int]::TryParse($answer, [Globalization.NumberStyles]::AllowThousands, $culture, [ref]$edge) -and
            $edge -ge 0 -and $edge -le 16384) {
            $MaxEdge = $edge
            break
        }
        Write-Host '長辺を 0〜16384 の整数で入力してください。0は原寸です。'
    }

    $fpsDefault = $VideoFps.ToString($culture)
    while ($true) {
        Write-Host "動画の抽出フレームレート（fps） [Enterで $fpsDefault] " -NoNewline
        $answer = [Console]::ReadLine()
        if ($null -eq $answer) { exit 0 }
        $answer = $answer.Normalize([Text.NormalizationForm]::FormKC).Trim()
        if (!$answer) { break }

        $fps = 0.0
        if ([double]::TryParse($answer, [Globalization.NumberStyles]::Float, $culture, [ref]$fps) -and
            $fps -ge 0.25 -and $fps -le 60) {
            $VideoFps = $fps
            break
        }
        Write-Host 'フレームレートを 0.25〜60 fps の範囲で入力してください。小数も使えます。'
    }

    $defaultAverage = 20
    do {
        Write-Host "画像1枚あたりの平均ステップ数 [Enterで $defaultAverage] " -NoNewline
        $answer = [Console]::ReadLine()
        if ($null -eq $answer) { exit 0 }
        $answer = $answer.Trim()
        if (!$answer) { $answer = "$defaultAverage" }
        $valid = [double]::TryParse($answer, [Globalization.NumberStyles]::Float, $culture, [ref]$AverageSteps)
    } until ($valid -and $AverageSteps -gt 0 -and -not [double]::IsInfinity($AverageSteps))

    $splatPrompt = 'ガウシアンの上限点数 [Enterで {0:N0}、例 400万 または 4000000] ' -f $MaxSplats
    while ($true) {
        Write-Host $splatPrompt -NoNewline
        $answer = [Console]::ReadLine()
        if ($null -eq $answer) { exit 0 }
        $answer = $answer.Normalize([Text.NormalizationForm]::FormKC).Trim()
        if (!$answer) { break }

        $amount = 0
        if ($answer -match '^(?<amount>[0-9]+|[0-9]{1,3}(?:,[0-9]{3})+)(?<unit>万)?点?$' -and
            [int]::TryParse($Matches['amount'].Replace(',', ''), [ref]$amount)) {
            $pointLimit = if ($Matches['unit']) { [long]$amount * 10000 } else { $amount }
            if ($pointLimit -ge 10000 -and $pointLimit -le 6000000) {
                $MaxSplats = [int]$pointLimit
                break
            }
        }
        Write-Host '上限は1万〜600万点の範囲で入力してください。'
    }

    $trainers = @{ '1' = 'brush'; '2' = 'faster-gs'; '3' = 'faster-gs-optimized' }
    while ($true) {
        Write-Host "学習方式  1: Brush、2: Faster-GS、3: Faster-GS 高速版 [Enterで $Trainer] " -NoNewline
        $answer = [Console]::ReadLine()
        if ($null -eq $answer) { exit 0 }
        $answer = $answer.Normalize([Text.NormalizationForm]::FormKC).Trim()
        if (!$answer) { break }
        if ($trainers.ContainsKey($answer)) {
            $Trainer = $trainers[$answer]
            break
        }
    }

    $edgeDescription = if ($MaxEdge) { "長辺 $MaxEdge px" } else { '原寸' }
    Write-Host ('実行設定  {0}、平均{1}ステップ/枚、最大{2:N0}点、{3}fps、{4}' -f $edgeDescription, $AverageSteps.ToString($culture), $MaxSplats, $VideoFps.ToString($culture), $Trainer)
}

& (Join-Path $PSScriptRoot 'Setup-Windows.ps1')
$python = Join-Path $PSScriptRoot 'apps/blender-5.3.0/blender-5.3.0-windows-x64/5.3/python/bin/python.exe'
$trainScript = Join-Path $PSScriptRoot 'train_room.py'
$rate = $VideoFps.ToString($culture)
$options = @('--input-folder',$InputFolder,'--steps',"$Steps",'--max-splats',"$MaxSplats",
    '--max-edge',"$MaxEdge",'--video-fps',$rate,'--mapper',$Mapper,'--trainer',$Trainer,'--pairing',$Pairing)
if (!$Matcher -and !$ReuseRun) {
    $models = Join-Path $PSScriptRoot 'apps/colmap-4.2.1-cuda/models/loma-b128'
    $weights = @('loma_detector.onnx', 'loma_descriptor_dedode_b.onnx', 'loma_matcher_B128.onnx')
    if (@($weights | Where-Object { !(Test-Path -LiteralPath (Join-Path $models $_) -PathType Leaf) }).Count) {
        $Matcher = 'sift'
    }
}
if ($Matcher) { $options += @('--matcher',$Matcher) }
if ($FramesPath) { $options += @('--frames-path',$FramesPath) }
if ($ReuseRun) { $options += @('--reuse-run',$ReuseRun) }
if ($InitPly) { $options += @('--init-ply',$InitPly) }
if ($SelectQuality -or $PSBoundParameters.ContainsKey('AverageSteps')) {
    $options += @('--average-steps',$AverageSteps.ToString($culture))
}

& $python $trainScript @options
exit $LASTEXITCODE
