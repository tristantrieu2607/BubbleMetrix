<#
.SYNOPSIS
    Zero-shot Cellpose on the sampled froth video frames + the labelled crop.
.DESCRIPTION
    Frames: roi\frames\frame_*.png (t = 0, 30, 60, 120, 180 s of 180WW.MP4).
    ROI: BubSAM3's in-focus auto-ROI (--auto-roi), the same ROI the BubSAM3 runs use.
    Outputs: cellpose\results\video\<frame>\cellpose_*  and  eval\results\f299_partial\.
    Progress is written to cellpose\results\video\log.txt as it goes.
#>
$ErrorActionPreference = 'Continue'      # Python logs to stderr; judge success by exit code only
$Root   = $PSScriptRoot
$Froth  = Split-Path -Parent $Root
$Python = Join-Path $Froth '.venv\Scripts\python.exe'
$Out    = Join-Path $Root 'results\video'
$Log    = Join-Path $Out 'log.txt'
$env:PYTHONUNBUFFERED = '1'
New-Item -ItemType Directory -Force -Path $Out | Out-Null
function Say($m) { $l = "[{0}] {1}" -f (Get-Date -Format 'HH:mm:ss'), $m; Write-Host $l -ForegroundColor Cyan; Add-Content $Log $l }

Say 'START cellpose video run'
$failed = 0
foreach ($f in Get-ChildItem (Join-Path $Froth 'roi\frames\frame_*.png') | Sort-Object Name) {
    $tag = $f.BaseName -replace '^frame_', ''
    $dir = Join-Path $Out $tag
    if (Test-Path (Join-Path $dir 'cellpose_labels.npy')) { Say "skip $tag (done)"; continue }
    Say "frame $tag"
    & $Python (Join-Path $Root 'run_cellpose.py') --image $f.FullName --auto-roi --out-dir $dir 2>&1 |
        ForEach-Object { "$_" } | Tee-Object -FilePath (Join-Path $dir 'run.log') -Append | Write-Host
    if ($LASTEXITCODE -ne 0) { Say "FAILED $tag (exit $LASTEXITCODE)"; $failed++ } else { Say "done $tag" }
}

Say 'labelled crop f299_partial'
# not redirected: run_eval.ps1 stops on errors, and redirecting a native stderr stream
# in Windows PowerShell 5.1 would turn Cellpose's log lines into terminating errors
try { & (Join-Path $Froth 'eval\run_eval.ps1') -Only cellpose } catch { Say "eval error: $_" }
if (-not (Test-Path (Join-Path $Froth 'eval\results\f299_partial\cellpose_zero_shot_score.json'))) { Say 'FAILED labelled crop'; $failed++ }

if ($failed -eq 0) { Say 'ALL DONE' } else { Say "FINISHED WITH $failed FAILURE(S)" }
exit $failed
