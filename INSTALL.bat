@echo off
rem One-time setup for BubbleMetrix (Windows, Python 3.12). Takes 10-20 min, ~3 GB download.
title BubbleMetrix setup
cd /d "%~dp0"

set PY=py -3.12
%PY% -c "import sys" >nul 2>&1 || set PY=python

echo === 1/3  Pulp + GUI environment (BubMask\.venv)
if not exist "BubMask\.venv\Scripts\python.exe" %PY% -m venv "BubMask\.venv" || goto :fail
"BubMask\.venv\Scripts\python.exe" -m pip install --upgrade pip
"BubMask\.venv\Scripts\python.exe" -m pip install -r "BubMask\requirements.txt" || goto :fail

echo === 2/3  Froth environment (Froth\.venv, Cellpose)
powershell -NoProfile -ExecutionPolicy Bypass -File "Froth\setup.ps1" || goto :fail

echo === 3/3  Checking the pulp model
if exist "BubMask\runs\rr_full_e20\best.pt" (
  echo Model found.
) else (
  echo.
  echo  ! Pulp model missing. Download best.pt from the GitHub Releases page and put it in
  echo    BubMask\runs\rr_full_e20\best.pt
)
echo.
echo Done. Start the app with launch_gui.bat
pause
exit /b 0

:fail
echo.
echo Setup failed - see the message above.
pause
exit /b 1
