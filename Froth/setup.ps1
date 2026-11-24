<#
.SYNOPSIS
    Create Froth\.venv - one Python environment for BubSAM3, Cellpose and the evaluation.
.DESCRIPTION
    Run once:   .\setup.ps1
    Re-run with -Force to rebuild it from scratch.
    Downloads torch (CPU), segment-anything and cellpose from PyPI, ~1 GB, a few minutes.
    Cellpose's model weights download separately the first time Cellpose runs.
#>
param([switch] $Force)
$ErrorActionPreference = 'Stop'
$Root = $PSScriptRoot
$Venv = Join-Path $Root '.venv'
$Py   = Join-Path $Venv 'Scripts\python.exe'

if ($Force -and (Test-Path $Venv)) { Remove-Item -Recurse -Force $Venv }

if (-not (Test-Path $Py)) {
    # Python 3.12, the same interpreter every other environment on this drive uses.
    $base = $null
    $known = 'C:\Users\Laptop K1\AppData\Local\Programs\Python\Python312\python.exe'
    if (Get-Command py -ErrorAction SilentlyContinue) {
        try { & py -3.12 -c "import sys" *> $null; if ($LASTEXITCODE -eq 0) { $base = @('py', '-3.12') } } catch { }
    }
    if (-not $base -and (Test-Path $known)) { $base = @($known) }
    if (-not $base) { $base = @('python') }
    Write-Host "=== creating $Venv with $($base -join ' ')" -ForegroundColor Cyan
    if ($base.Count -gt 1) { & $base[0] $base[1] -m venv $Venv } else { & $base[0] -m venv $Venv }
    if ($LASTEXITCODE -ne 0) { throw 'venv creation failed' }
}

Write-Host '=== installing requirements (a few minutes the first time)' -ForegroundColor Cyan
& $Py -m pip install --upgrade pip
& $Py -m pip install -r (Join-Path $Root 'requirements.txt')
if ($LASTEXITCODE -ne 0) { throw 'pip install failed' }

Write-Host '=== checking imports' -ForegroundColor Cyan
& $Py -c "import torch, segment_anything, cellpose, cv2, skimage, scipy, matplotlib; print('torch', torch.__version__, '| cuda', torch.cuda.is_available()); print('cellpose', getattr(cellpose, 'version', '?')); print('opencv', cv2.__version__)"
if ($LASTEXITCODE -ne 0) { throw 'import check failed' }

Write-Host "`nReady. Interpreter: $Py" -ForegroundColor Green
