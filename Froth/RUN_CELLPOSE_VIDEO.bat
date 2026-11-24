@echo off
rem Double-click: set up Froth\.venv (first time only), then zero-shot Cellpose on the
rem sampled froth frames and the labelled crop. Progress: cellpose\results\video\log.txt
title Froth - Cellpose zero-shot
cd /d "%~dp0"
echo === Setting up Froth\.venv (first time takes several minutes) ===
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup.ps1"
if errorlevel 1 (
  echo SETUP FAILED > "%~dp0cellpose\results\video_setup_failed.txt"
  echo.& echo Setup failed - see the messages above. & pause & exit /b 1
)
echo === Running Cellpose ===
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0cellpose\run_video.ps1"
echo.
echo Finished. You can close this window.
pause
