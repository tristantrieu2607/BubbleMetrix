<#
.SYNOPSIS
    Cellpose on a froth frame (zero-shot), then measure it in mm with BubSAM3's measure_bubbles.py.
.EXAMPLE
    .\run.ps1                                   # frame_0000, same static ROI as BubSAM3
.EXAMPLE
    .\run.ps1 -Image ..\eval\data\180WW_f00299_eval_sub_x168y464.png -NoROI -MmPerPx 0.10956
.EXAMPLE
    .\run.ps1 -Gates -Prefix cellpose_gated     # apply BubSAM3's shape gates afterwards
.PARAMETER Diameter
    Expected bubble diameter in px. Leave unset for zero-shot; setting it is a tuned run
    and is recorded as such in <prefix>_run.json.
.PARAMETER MmPerPx
    Scale for measurement. Needed for crops, where the cell walls are not in view.
    On full frames leave it unset and measure_bubbles.py finds the 55 mm tube walls.
#>
param(
    [string] $Image  = '',
    [string] $Prefix = 'cellpose',
    [switch] $NoROI,
    [switch] $AutoROI,
    [switch] $Gates,
    [double] $Diameter = 0,
    [double] $MmPerPx  = 0,
    [switch] $Show
)
$ErrorActionPreference = 'Stop'
$Root    = $PSScriptRoot
$Froth   = Split-Path -Parent $Root
$Python  = Join-Path $Froth '.venv\Scripts\python.exe'
$Results = Join-Path $Root 'results'
if (-not (Test-Path $Python)) { throw "Interpreter not found: $Python`nRun Froth\setup.ps1 once." }
if ([string]::IsNullOrWhiteSpace($Image)) { $Image = Join-Path $Froth 'roi\frames\frame_0000.png' }
if (-not (Test-Path $Image)) { throw "Image not found: $Image" }
$Image = (Resolve-Path $Image).Path
$inv = [Globalization.CultureInfo]::InvariantCulture

$a = @((Join-Path $Root 'run_cellpose.py'), '--image', $Image, '--out-prefix', $Prefix, '--out-dir', $Results)
if ($NoROI)   { $a += '--no-roi' }
if ($AutoROI) { $a += '--auto-roi' }
if ($Gates)   { $a += '--gates' }
if ($Diameter -gt 0) { $a += @('--diameter', $Diameter.ToString($inv)) }
Write-Host "`n=== Cellpose" -ForegroundColor Cyan
& $Python @a
if ($LASTEXITCODE -ne 0) { throw "run_cellpose.py failed ($LASTEXITCODE)" }

# Measurement is BubSAM3's, unchanged: one calibration, one D32 formula for every method.
$m = @('measure_bubbles.py', (Join-Path $Results "${Prefix}_masks.csv"), '--image', $Image)
if ($AutoROI) { $m += '--auto-roi' }
if ($MmPerPx -gt 0) { $m += @('--mm-per-px', $MmPerPx.ToString($inv)) } else { $m += '--calibration-overlay' }
Write-Host "`n=== Measuring" -ForegroundColor Cyan
Push-Location (Join-Path $Froth 'roi')
try { & $Python @m } finally { Pop-Location }
if ($LASTEXITCODE -ne 0) { throw "measure_bubbles.py failed ($LASTEXITCODE)" }

if ($Show) {
    foreach ($f in @("${Prefix}_overlay.png", "${Prefix}_distribution.png")) {
        $p = Join-Path $Results $f; if (Test-Path $p) { Invoke-Item $p }
    }
}
