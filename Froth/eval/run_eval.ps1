<#
.SYNOPSIS
    Run every froth method on an evaluation case and score them with one harness.
.DESCRIPTION
    For the case's image: BubSAM3 (~10 min on CPU for the 384 px f299 crop),
    Cellpose (zero-shot, minutes) and bubseg (seconds), then eval\froth_eval.py
    scores each and writes results\<case>\comparison.md.

    Method outputs go to results\<case>\runs\, never to roi\results or
    cellpose\results, so an evaluation cannot overwrite a production run.
    A method that already has outputs there is skipped; -Force reruns it.
.EXAMPLE
    .\run_eval.ps1                         # case f299_partial, all three methods
.EXAMPLE
    .\run_eval.ps1 -Only cellpose          # just Cellpose, e.g. after a version change
#>
param(
    [string]   $Case = 'f299_partial',
    [ValidateSet('bubsam3', 'cellpose', 'bubseg')]
    [string[]] $Only = @('cellpose'),   # bubsam3 / bubseg now live in progress\ (superseded)
    [switch]   $Force
)
$ErrorActionPreference = 'Stop'
$Eval   = $PSScriptRoot
$Froth  = Split-Path -Parent $Eval
$Python = Join-Path $Froth '.venv\Scripts\python.exe'
if (-not (Test-Path $Python)) { throw "Interpreter not found: $Python`nRun Froth\setup.ps1 once." }

$cases = Get-Content (Join-Path $Eval 'cases.json') -Raw | ConvertFrom-Json
if (-not $cases.$Case) { throw "Unknown case '$Case' - see eval\cases.json" }
$Image = Join-Path $Eval $cases.$Case.image
$Runs  = Join-Path $Eval "results\$Case\runs"
New-Item -ItemType Directory -Force -Path $Runs | Out-Null

function Step($m) { Write-Host "`n=== $m" -ForegroundColor Cyan }
function Check($what) { if ($LASTEXITCODE -ne 0) { throw "$what failed ($LASTEXITCODE)" } }
$Score = Join-Path $Eval 'froth_eval.py'

if ($Only -contains 'bubsam3') {
    $out = Join-Path $Runs 'bubsam3'
    if ($Force -or -not (Test-Path (Join-Path $out 'bubsam3_instances.npz'))) {
        Step 'BubSAM3 (default settings) - about 10 minutes on CPU'
        Push-Location (Join-Path $Froth 'roi')
        try { & $Python run_bubsam3.py --image $Image --no-roi --out-dir $out } finally { Pop-Location }
        Check 'BubSAM3'
    } else { Step 'BubSAM3: outputs exist, skipping (use -Force to rerun)' }
    & $Python $Score score --case $Case --pred (Join-Path $out 'bubsam3_instances.npz') --name 'BubSAM3'; Check 'scoring BubSAM3'
}
if ($Only -contains 'cellpose') {
    $out = Join-Path $Runs 'cellpose'
    if ($Force -or -not (Test-Path (Join-Path $out 'cellpose_labels.npy'))) {
        Step 'Cellpose (zero-shot) - first run downloads the model'
        & $Python (Join-Path $Froth 'cellpose\run_cellpose.py') --image $Image --no-roi --out-dir $out
        Check 'Cellpose'
    } else { Step 'Cellpose: outputs exist, skipping (use -Force to rerun)' }
    # the label map is Cellpose's exact answer; score that rather than the outlines
    & $Python $Score score --case $Case --pred (Join-Path $out 'cellpose_labels.npy') --name 'Cellpose (zero-shot)'; Check 'scoring Cellpose'
}
if ($Only -contains 'bubseg') {
    $out = Join-Path $Runs 'bubseg_labels.npy'
    Step 'bubseg froth path (ROI = whole image)'
    & $Python (Join-Path $Eval 'baselines\run_bubseg.py') --image $Image --out $out; Check 'bubseg'
    & $Python $Score score --case $Case --pred $out --name 'bubseg (ROI = whole image)'; Check 'scoring bubseg'
}

Step "Comparison  ->  eval\results\$Case\comparison.md"
& $Python $Score table --case $Case
